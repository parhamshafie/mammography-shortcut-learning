# -*- coding: utf-8 -*-
"""Training/eval orchestration for the binary (normal vs. suspicious) task. One `run(cfg)` drives
all four configurations -- NLBSD-only baseline, +CMMD, +DDSM, +DDSM+CMMD -- the only thing that
changes between them is which entries under cfg["external_datasets"] are enabled.

Single patient-level 70/10/20 train/val/test split (Sec. 2.4) -- not cross-validation.

Phase A (Table 2, top block): external positives go into training; the NLBSD test split stays fixed.
Phase B (Table 2, bottom block): a patient-level slice of each external dataset is held out of
training and added to the NLBSD test set instead, giving a "NLBSD u external" mixed test set. Both
come out of the same run -- set cfg["phase_b"]["enabled"] = False to skip Phase B and use 100% of the
external positives for training.

Operating-point metrics (sensitivity/specificity/accuracy) use one decision threshold, picked on the
baseline model's own validation split and reused unchanged by every other configuration (Sec. 2.7).
The baseline run saves it to baseline_threshold.json; other configs point at that file via
cfg["shared_threshold"]["baseline_threshold_json"].
"""
import time, copy, logging
from pathlib import Path
import numpy as np

from . import data as D
from . import model as M
from . import losses as L
from . import stats as S
from . import plotting as P
from .utils import setup_logging, seed_everything, write_repro_snapshot, write_csv, atomic_torch_save

log = logging.getLogger("mammo_pipeline")


def build_external_pools(cfg):
    """Loads every enabled external dataset, patient-splits each into a training pool and a
    Phase-B holdout, concatenates across datasets."""
    import pandas as pd
    ext_cfg = cfg.get("external_datasets", {})
    pb = cfg.get("phase_b", {"enabled": False})
    frac = pb.get("external_holdout_frac", 0.0) if pb.get("enabled", False) else 0.0
    smoke = cfg["smoke_test"]["enabled"]
    train_parts, holdout_parts = [], []
    for name, dcfg in ext_cfg.items():
        if not dcfg.get("enabled", False):
            continue
        full = D.load_external_flat_dataset(dcfg, "label", 1)
        tr, ho = D.split_external_holdout(full, frac, pb.get("seed", 123))
        if smoke:
            s = cfg["smoke_test"]
            tr = tr.sample(min(len(tr), s.get("ext_train_pos", 15)), random_state=0).reset_index(drop=True)
            ho = ho.sample(min(len(ho), s.get("ext_holdout_pos", 5)), random_state=0).reset_index(drop=True)
        train_parts.append(tr); holdout_parts.append(ho)
    if not train_parts:
        empty = pd.DataFrame(columns=["image_path", "label", "case_id", "source"])
        return empty, empty.copy()
    train_pool = pd.concat(train_parts, ignore_index=True)
    holdout = pd.concat(holdout_parts, ignore_index=True)
    log.info("[EXTERNAL] train_pool=%d images | phase_b_holdout=%d images", len(train_pool), len(holdout))
    return train_pool, holdout


def run_inference(model, loader, device, amp):
    import torch
    model.eval()
    P_, Y_, PA_, CI_ = [], [], [], []
    with torch.no_grad():
        for x, yb, paths, cids in loader:
            x = x.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logit = model(x)
            P_.append(torch.sigmoid(logit).float().cpu().numpy())
            Y_.append(yb.numpy()); PA_.extend(paths); CI_.extend(cids)
    return np.concatenate(P_), np.concatenate(Y_), PA_, CI_


def train_one_epoch(model, loader, optimizer, scaler, device, cfg, ep, hb):
    import torch
    model.train()
    tot, n = 0.0, 0
    amp = cfg["train"]["amp"] and device.type == "cuda"
    t0 = time.time()
    for bi, (x, yb, _, _) in enumerate(loader):
        x = x.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            loss = L.compute_loss(model(x), yb)
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
            rate = (bi + 1) / max(time.time() - t0, 1e-6)
            log.info("    ep%d batch %d/%d loss=%.4f %.1f b/s", ep, bi + 1, len(loader), tot / max(n, 1), rate)
    return tot / max(n, 1)


def compute_and_save_thresholds(net, calib_loader, device, amp, out_dir, agg_rule):
    """Youden's J on the baseline model's own validation split -- image-level and case-level. This
    is only called for the baseline run; every other config loads these numbers back instead of
    picking its own (Sec. 2.7)."""
    import json
    vp, vy, vpaths, vcids = run_inference(net, calib_loader, device, amp)
    thr_img = S.youden_threshold(vy, vp) if len(np.unique(vy)) > 1 else 0.5
    cy, cp, _ = S.aggregate_case(vpaths, vcids, vy, vp, agg_rule)
    thr_case = S.youden_threshold(cy, cp) if len(np.unique(cy)) > 1 else 0.5
    thresholds = {"image": float(thr_img), "case": float(thr_case)}
    with open(out_dir / "baseline_threshold.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    log.info("[THRESHOLD] baseline thresholds from validation split: image=%.4f case=%.4f -> %s",
             thr_img, thr_case, out_dir / "baseline_threshold.json")
    return thresholds


def load_shared_thresholds(cfg):
    """Returns {'image':..., 'case':...} loaded from another run's baseline_threshold.json, or None
    if this run has no cfg['shared_threshold'] set (i.e. this run IS the baseline)."""
    import json
    path = cfg.get("shared_threshold", {}).get("baseline_threshold_json")
    if not path:
        return None
    with open(path) as f:
        return json.load(f)


def run_split(df, cfg, out_dir, device, external_train_pool, external_holdout):
    import torch
    import pandas as pd
    from torch.utils.data import DataLoader

    smoke = cfg["smoke_test"]["enabled"]
    img_size = cfg["smoke_test"]["image_size"] if smoke else cfg["data"]["image_size"]
    boot_n = cfg["smoke_test"]["bootstrap_n"] if smoke else 1000
    epochs = cfg["smoke_test"]["epochs"] if smoke else cfg["train"]["num_epochs"]

    test_df = df[df["split"] == "test"].reset_index(drop=True)
    calib_df = df[df["split"] == "val"].reset_index(drop=True)
    fit_df = df[df["split"] == "train"].reset_index(drop=True)
    n_fit_nlbsd = len(fit_df)
    if len(external_train_pool):
        fit_df = pd.concat([fit_df, external_train_pool], ignore_index=True)
    log.info("=== SPLIT === fit=%d (nlbsd=%d + ext=%d) calib=%d test=%d holdout=%d",
             len(fit_df), n_fit_nlbsd, len(fit_df) - n_fit_nlbsd, len(calib_df), len(test_df),
             len(external_holdout))

    mean, std = cfg["data"]["norm_mean"], cfg["data"]["norm_std"]
    ds_kw = dict(image_size=img_size, mean=mean, std=std)
    fit_ds = D.MammoDataset(fit_df, augment=True, aug_cfg=cfg["augment"], **ds_kw)
    calib_ds = D.MammoDataset(calib_df, augment=False, aug_cfg=cfg["augment"], **ds_kw)
    test_ds = D.MammoDataset(test_df, augment=False, aug_cfg=cfg["augment"], **ds_kw)
    bs = cfg["train"]["batch_size"]
    nw = 0 if smoke else cfg["data"]["num_workers"]
    pos_q = cfg["sampler"]["positives_per_batch"]
    sampler = D.BalancedBatchSampler(fit_df["label"].to_numpy(), bs, {1: pos_q, 0: bs - pos_q})
    fit_loader = DataLoader(fit_ds, batch_sampler=sampler, num_workers=nw)
    calib_loader = DataLoader(calib_ds, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)

    net = M.build_model(cfg, n_outputs=1).to(device)
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
    history, best_auc, start_ep, no_improve = [], -1.0, 1, 0
    patience = cfg["train"].get("early_stopping_patience", 10)

    if cfg["control"]["resume"] and last_ckpt.exists():
        c = torch.load(last_ckpt, map_location=device, weights_only=False)
        unwrap(net).load_state_dict(c["model"]); optimizer.load_state_dict(c["opt"])
        start_ep = c["epoch"] + 1; history = c.get("history", [])
        best_auc = c.get("best_auc", -1.0); no_improve = c.get("epochs_no_improve", 0)
        log.info("[RESUMED] from epoch %d", start_ep)

    hb = cfg["logging"]["heartbeat_batches"]
    for ep in range(start_ep, epochs + 1):
        tr_loss = train_one_epoch(net, fit_loader, optimizer, scaler, device, cfg, ep, hb)
        vp, vy, _, _ = run_inference(net, calib_loader, device, cfg["train"]["amp"])
        from sklearn.metrics import roc_auc_score
        v_auc = float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1 else 0.0
        history.append({"epoch": ep, "train_loss": tr_loss, "val_auc": v_auc})
        log.info("  [ep %d/%d] loss=%.4f val_auc=%.4f", ep, epochs, tr_loss, v_auc)
        atomic_torch_save({"epoch": ep, "model": unwrap(net).state_dict(), "opt": optimizer.state_dict(),
                           "history": history, "best_auc": best_auc, "epochs_no_improve": no_improve}, last_ckpt)
        if v_auc > best_auc:
            best_auc = v_auc; no_improve = 0
            atomic_torch_save({"epoch": ep, "model": copy.deepcopy(unwrap(net).state_dict()),
                               "best_auc": best_auc, "val_auc": v_auc}, best_ckpt)
            log.info("  [BEST] ep%d val_auc=%.4f", ep, best_auc)
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info("  [EARLY-STOP] after %d epochs", patience)
                break

    if cfg["control"]["use_best_epoch"] and best_ckpt.exists():
        bc = torch.load(best_ckpt, map_location=device, weights_only=False)
        unwrap(net).load_state_dict(bc["model"])

    thresholds = load_shared_thresholds(cfg)
    if thresholds is None:
        thresholds = compute_and_save_thresholds(net, calib_loader, device, cfg["train"]["amp"],
                                                  out_dir, cfg["aggregation"]["rule"])
    else:
        log.info("[THRESHOLD] using shared baseline thresholds: image=%.4f case=%.4f",
                 thresholds["image"], thresholds["case"])

    tp, ty, tpaths, tcids = run_inference(net, test_loader, device, cfg["train"]["amp"])
    img_m = S.compute_metrics(ty, tp, boot_n, "image", threshold=thresholds["image"]); img_m.update(domain="nlbsd")
    cy, cp, _ = S.aggregate_case(tpaths, tcids, ty, tp, cfg["aggregation"]["rule"])
    case_m = S.compute_metrics(cy, cp, boot_n, "case", threshold=thresholds["case"]); case_m.update(domain="nlbsd")
    per_sample = [{"path": pa, "case_id": ci, "label": int(yy), "prob": float(pp),
                   "split": "test", "domain": "nlbsd"} for pa, ci, yy, pp in zip(tpaths, tcids, ty, tp)]
    out = {"history": history, "img_m": [img_m], "case_m": [case_m], "per_sample": per_sample}

    if len(external_holdout):
        ho_ds = D.MammoDataset(external_holdout, augment=False, aug_cfg=cfg["augment"], **ds_kw)
        ho_loader = DataLoader(ho_ds, batch_size=bs, shuffle=False, num_workers=nw)
        hp, hy, hpaths, hcids = run_inference(net, ho_loader, device, cfg["train"]["amp"])
        mp_img, my_img = np.concatenate([tp, hp]), np.concatenate([ty, hy])
        m_paths, m_cids = list(tpaths) + list(hpaths), list(tcids) + list(hcids)
        mixed_img_m = S.compute_metrics(my_img, mp_img, boot_n, "image", threshold=thresholds["image"])
        mixed_img_m.update(domain="mixed")
        mcy, mcp, _ = S.aggregate_case(m_paths, m_cids, my_img, mp_img, cfg["aggregation"]["rule"])
        mixed_case_m = S.compute_metrics(mcy, mcp, boot_n, "case", threshold=thresholds["case"])
        mixed_case_m.update(domain="mixed")
        ho_samples = [{"path": pa, "case_id": ci, "label": int(yy), "prob": float(pp),
                       "split": "test_external_holdout", "domain": "mixed"}
                      for pa, ci, yy, pp in zip(hpaths, hcids, hy, hp)]
        out["img_m"].append(mixed_img_m); out["case_m"].append(mixed_case_m)
        out["per_sample"] += ho_samples
        log.info("  [PHASE-B] mixed-domain AUC=%.4f", mixed_img_m.get("roc_auc", float("nan")))
    return out


def run_training(cfg, out_dir, device):
    df = D.load_manifest(cfg)
    if cfg["smoke_test"]["enabled"]:
        df = D.smoke_subset(df, cfg)
    df = D.make_single_split(df, cfg, out_dir)
    ext_train_pool, ext_holdout = build_external_pools(cfg)

    r = run_split(df, cfg, out_dir, device, ext_train_pool, ext_holdout)

    write_csv(r["history"], out_dir / "training_history.csv")
    write_csv(r["img_m"], out_dir / "metrics_image.csv")
    write_csv(r["case_m"], out_dir / "metrics_case.csv")
    if cfg["outputs"]["per_sample_csv"]:
        write_csv(r["per_sample"], out_dir / "per_sample_predictions.csv")
    log.info("[DONE] CSVs written -> %s", out_dir)
    if cfg["outputs"]["generate_plots"]:
        P.make_plots(out_dir / "per_sample_predictions.csv", out_dir / "plots")
    log.info("STATUS: SUCCESS")


def run(cfg):
    """Top-level entry point -- what each run_*.py script calls."""
    seed_everything(cfg["train"]["seed"])
    run_root = Path(cfg["paths"]["output_root"]) / ("smoke_test" if cfg["smoke_test"]["enabled"] else cfg["run_name"])
    run_root.mkdir(parents=True, exist_ok=True)
    setup_logging(run_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    if cfg["outputs"]["repro_snapshot"]:
        write_repro_snapshot(run_root, cfg)
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("MODE=%s | smoke=%s | device=%s", cfg["mode"], cfg["smoke_test"]["enabled"], device)

    if cfg["mode"] == "plot":
        csv_in = cfg.get("plot_from_csv") or (run_root / "per_sample_predictions.csv")
        P.make_plots(csv_in, run_root / "plots")
        log.info("STATUS: SUCCESS"); return
    if cfg["mode"] in ("train", "eval"):
        run_training(cfg, run_root, device)   # eval reuses the checkpoint resume logic, skips training
        return
    raise ValueError(f"Unknown mode: {cfg['mode']}")
