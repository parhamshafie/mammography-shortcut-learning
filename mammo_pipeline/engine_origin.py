# -*- coding: utf-8 -*-
"""Training/eval orchestration for the 3-class dataset-origin diagnostic (Sec. 2.6 / Fig. 3):
NLBSD vs. CBIS-DDSM vs. CMMD, predicted from the image itself instead of the clinical label.
Same encoder/preprocessing/augmentation as the binary task, just a 3-way head and cross-entropy.
Single patient-level 70/10/20 train/val/test split, same as the binary task -- not cross-validation."""
import time, copy, logging
from pathlib import Path
import numpy as np

from . import data as D
from . import model as M
from . import stats as S
from . import plotting as P
from .utils import setup_logging, seed_everything, write_repro_snapshot, write_csv, atomic_torch_save

log = logging.getLogger("mammo_pipeline")


def build_origin_dataset(cfg):
    import pandas as pd
    classes = cfg["classes"]
    assert classes == ["nlbsd", "ddsm", "cmmd"], "fixed order, must match Fig. 3's row/column order"

    nlbsd = D.load_manifest(cfg)[["image_path", "case_id"]].copy()
    nlbsd["case_id"] = "nlbsd_" + nlbsd["case_id"].astype(str)
    nlbsd["origin"] = 0
    nlbsd["source"] = "nlbsd"

    ddsm = D.load_external_flat_dataset(cfg["external_datasets"]["ddsm"], "origin", 1)
    cmmd = D.load_external_flat_dataset(cfg["external_datasets"]["cmmd"], "origin", 2)

    if cfg["smoke_test"]["enabled"]:
        s = cfg["smoke_test"]
        need = s["per_class_train"] + s["per_class_val"] + s["per_class_test"]
        nlbsd = nlbsd.sample(min(need, len(nlbsd)), random_state=0).reset_index(drop=True)
        ddsm = ddsm.sample(min(need, len(ddsm)), random_state=0).reset_index(drop=True)
        cmmd = cmmd.sample(min(need, len(cmmd)), random_state=0).reset_index(drop=True)

    df = pd.concat([nlbsd, ddsm, cmmd], ignore_index=True)
    log.info("[ORIGIN] %d images | nlbsd=%d ddsm=%d cmmd=%d", len(df),
             (df["origin"] == 0).sum(), (df["origin"] == 1).sum(), (df["origin"] == 2).sum())
    return df


def run_inference(model, loader, device, amp):
    import torch, torch.nn.functional as F
    model.eval()
    P_, Y_, PA_, CI_ = [], [], [], []
    with torch.no_grad():
        for x, yb, paths, cids in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                probs = F.softmax(model(x).float(), dim=1)
            P_.append(probs.cpu().numpy()); Y_.append(yb.numpy()); PA_.extend(paths); CI_.extend(cids)
    return np.concatenate(P_, axis=0), np.concatenate(Y_), PA_, CI_


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, ep, hb):
    import torch, torch.nn.functional as F
    model.train()
    tot, n = 0.0, 0
    amp = cfg["train"]["amp"] and device.type == "cuda"
    t0 = time.time()
    for bi, (x, yb, _, _) in enumerate(loader):
        x = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = F.cross_entropy(model(x), yb)
        if amp:
            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"]:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if cfg["train"]["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()
        tot += float(loss.item()) * len(yb); n += len(yb)
        if hb and (bi + 1) % hb == 0:
            log.info("    ep%d batch %d/%d loss=%.4f", ep, bi + 1, len(loader), tot / max(n, 1))
    return tot / max(n, 1)


def run_split(df, cfg, out_dir, device):
    import torch
    from torch.utils.data import DataLoader
    class_names = cfg["classes"]; n_classes = len(class_names)
    smoke = cfg["smoke_test"]["enabled"]
    img_size = cfg["smoke_test"]["image_size"] if smoke else cfg["data"]["image_size"]
    epochs = cfg["smoke_test"]["epochs"] if smoke else cfg["train"]["num_epochs"]

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    calib_df = df[df["split"] == "val"].reset_index(drop=True)
    fit_df = df[df["split"] == "train"].reset_index(drop=True)
    log.info("=== SPLIT === fit=%d calib=%d test=%d", len(fit_df), len(calib_df), len(test_df))

    mean, std = cfg["data"]["norm_mean"], cfg["data"]["norm_std"]
    ds_kw = dict(image_size=img_size, mean=mean, std=std, label_col="origin", label_dtype="long")
    fit_ds = D.MammoDataset(fit_df, augment=True, aug_cfg=cfg["augment"], **ds_kw)
    calib_ds = D.MammoDataset(calib_df, augment=False, aug_cfg=cfg["augment"], **ds_kw)
    test_ds = D.MammoDataset(test_df, augment=False, aug_cfg=cfg["augment"], **ds_kw)
    bs = cfg["train"]["batch_size"]
    nw = 0 if smoke else cfg["data"]["num_workers"]
    quotas = dict(enumerate(D.even_quotas(bs, n_classes)))
    sampler = D.BalancedBatchSampler(fit_df["origin"].to_numpy(), bs, quotas)
    fit_loader = DataLoader(fit_ds, batch_sampler=sampler, num_workers=nw)
    calib_loader = DataLoader(calib_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    net = M.build_model(cfg, n_outputs=n_classes).to(device)
    net.amp_enabled = bool(cfg["train"]["amp"] and device.type == "cuda")
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)

    def unwrap(m):
        return m.module if isinstance(m, torch.nn.DataParallel) else m

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=cfg["train"]["head_lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=cfg["train"]["amp"] and device.type == "cuda")

    ckpt_dir = out_dir / "checkpoints"; ckpt_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt, best_ckpt = ckpt_dir / "last.pth", ckpt_dir / "best.pth"
    history, best_metric, start_ep, no_improve = [], -1.0, 1, 0
    select_by = cfg["control"].get("select_by", "macro_auc")
    patience = cfg["train"].get("early_stopping_patience", 10)

    if cfg["control"]["resume"] and last_ckpt.exists():
        c = torch.load(last_ckpt, map_location=device, weights_only=False)
        unwrap(net).load_state_dict(c["model"]); optimizer.load_state_dict(c["opt"])
        start_ep = c["epoch"] + 1; history = c.get("history", [])
        best_metric = c.get("best_metric", -1.0); no_improve = c.get("epochs_no_improve", 0)
        log.info("[RESUMED] from epoch %d", start_ep)

    hb = cfg["logging"]["heartbeat_batches"]
    for ep in range(start_ep, epochs + 1):
        tr_loss = train_one_epoch(net, fit_loader, optimizer, scaler, device, cfg, ep, hb)
        vp, vy, _, _ = run_inference(net, calib_loader, device, cfg["train"]["amp"])
        vm = S.compute_multiclass_metrics(vy, vp, class_names)
        history.append({"epoch": ep, "train_loss": tr_loss,
                        "val_balanced_acc": vm["balanced_acc"], "val_macro_auc": vm["macro_auc"]})
        log.info("  [ep %d/%d] loss=%.4f val_bal_acc=%.4f val_macro_auc=%.4f",
                 ep, epochs, tr_loss, vm["balanced_acc"], vm["macro_auc"])
        metric = vm["macro_auc"] if select_by == "macro_auc" else vm["balanced_acc"]
        atomic_torch_save({"epoch": ep, "model": unwrap(net).state_dict(), "opt": optimizer.state_dict(),
                           "history": history, "best_metric": best_metric, "epochs_no_improve": no_improve}, last_ckpt)
        if metric > best_metric:
            best_metric = metric; no_improve = 0
            atomic_torch_save({"epoch": ep, "model": copy.deepcopy(unwrap(net).state_dict()),
                               "best_metric": best_metric, "val_balanced_acc": vm["balanced_acc"],
                               "val_macro_auc": vm["macro_auc"]}, best_ckpt)
            log.info("  [BEST] ep%d metric=%.4f", ep, best_metric)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  [EARLY-STOP] after %d epochs", patience)
                break

    if cfg["control"]["use_best_epoch"] and best_ckpt.exists():
        bc = torch.load(best_ckpt, map_location=device, weights_only=False)
        unwrap(net).load_state_dict(bc["model"])

    tp, ty, tpaths, tcids = run_inference(net, test_loader, device, cfg["train"]["amp"])
    img_m = S.compute_multiclass_metrics(ty, tp, class_names); img_m.update(level="image")
    cy, cp, _ = S.aggregate_case_probs(tpaths, tcids, ty, tp, n_classes)
    case_m = S.compute_multiclass_metrics(cy, cp, class_names); case_m.update(level="case")
    per_sample = []
    for pa, ci, yy, pp in zip(tpaths, tcids, ty, tp):
        row = {"path": pa, "case_id": ci, "label": int(yy), "split": "test"}
        row.update({f"prob_{c}": float(pp[i]) for i, c in enumerate(class_names)})
        per_sample.append(row)
    return {"history": history, "img_m": img_m, "case_m": case_m, "per_sample": per_sample}


def run_training(cfg, out_dir, device):
    df = build_origin_dataset(cfg)
    df = D.make_single_split(df, cfg, out_dir, label_col="origin")
    class_names = cfg["classes"]; n_classes = len(class_names)

    r = run_split(df, cfg, out_dir, device)

    write_csv(r["history"], out_dir / "training_history.csv")
    write_csv([r["img_m"]], out_dir / "metrics_image.csv")
    write_csv([r["case_m"]], out_dir / "metrics_case.csv")
    if cfg["outputs"]["per_sample_csv"]:
        write_csv(r["per_sample"], out_dir / "per_sample_predictions.csv")

    import csv as csv_mod
    for name, m in (("image", r["img_m"]), ("case", r["case_m"])):
        cm = np.array(m["_confusion_matrix"])
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum != 0) * 100
        with open(out_dir / f"confusion_matrix_{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv_mod.writer(f)
            w.writerow(["true\\pred"] + class_names)
            for i, cname in enumerate(class_names):
                w.writerow([cname] + [f"{cm_pct[i, j]:.2f}" for j in range(n_classes)])
        if cfg["outputs"]["generate_plots"]:
            P.plot_confusion_matrix(cm_pct, ["NLBSD", "CBIS-DDSM", "CMMD"],
                                    out_dir / "plots" / f"confusion_matrix_{name}.pdf",
                                    title=f"Dataset-origin confusion matrix ({name}-level)")
    log.info("[DONE] CSVs written -> %s", out_dir)
    log.info("STATUS: SUCCESS")


def run(cfg):
    seed_everything(cfg["train"]["seed"])
    run_root = Path(cfg["paths"]["output_root"]) / ("smoke_test" if cfg["smoke_test"]["enabled"] else cfg["run_name"])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "plots").mkdir(parents=True, exist_ok=True)
    setup_logging(run_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    if cfg["outputs"]["repro_snapshot"]:
        write_repro_snapshot(run_root, cfg)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("MODE=%s | smoke=%s | device=%s", cfg["mode"], cfg["smoke_test"]["enabled"], device)

    if cfg["mode"] == "plot":
        csv_in = cfg.get("plot_from_csv") or (run_root / "per_sample_predictions.csv")
        P.make_plots_origin(csv_in, run_root / "plots", cfg["classes"])
        log.info("STATUS: SUCCESS"); return
    if cfg["mode"] in ("train", "eval"):
        run_training(cfg, run_root, device)
        return
    raise ValueError(f"Unknown mode: {cfg['mode']}")
