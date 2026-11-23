# -*- coding: utf-8 -*-
"""
t-SNE / PCA plot of the frozen Mammo-CLIP encoder features (JMI revision, Reviewer 2 comment 12).

Fully standalone: one file, press Run in PyCharm. Only BASE (data folder) needs to be correct.
The data loading, preprocessing, histogram matching and Mammo-CLIP loading are copied unchanged from the
Experiment 3 script (origin_harmonized_mammoclip.py), so the features match the dataset-origin experiments.

No training is needed: the encoder is frozen, so its features do not depend on the linear head.

What it does
  1. Loads NLBSD, CBIS-DDSM and CMMD (all cases, benign and malignant -- same as Experiments 2 and 3).
  2. Makes its own patient-level 70/10/20 split (seed 42) and uses only TEST images. Nothing from earlier runs is used.
  3. Samples up to N_PER_DATASET test images per dataset (same number from each, so no dataset dominates).
  4. Extracts the 2048-d pooled encoder features (no augmentation), with and without histogram matching.
  5. Standardizes the features, reduces to 50 dimensions with PCA, then runs t-SNE (fixed seed).
  6. Saves: Figure_tsne.png (600 dpi) + .pdf, a PCA version, the features (.npz) and the 2-D coordinates (.csv),
     and a small text file with the silhouette score of each embedding.

Output folder: <BASE>/imagenet_control/tsne_features_independent
"""
import os, re, sys, csv, json, time, copy, random, logging, argparse
from pathlib import Path
import numpy as np

# =====================================================================================================
# CONFIG -- edit the paths marked  <-- CHECK
# =====================================================================================================
BASE = r"D:\path\to\data"   # <-- your data folder; everything below is relative to it

CFG = {
    # ---------------- NLBSD ----------------
    "nlbsd_manifest": os.path.join(BASE, "nlbs_manifest.csv"),          # <-- CHECK file name
    "nlbsd_image_root": "",              # prefix for RELATIVE paths in the manifest ("" if paths are absolute)
    # The manifest was written before the folder moved: rewrite this prefix in every path (old, new).
    "nlbsd_path_replace": None,          # paths in the manifest are already correct (D:\path\to\data\png\...)
    # Column names; None = auto-detect. case_id "" = don't use a column, derive it from the path
    # ('Patient ID' is empty in this manifest).
    "nlbsd_cols": {"image_path": "PNG Path", "label": "cancer", "case_id": ""},
    # How to derive case_id from the path: "uuid" = the UUID folder (.../png/<group>/<uuid>/<side>/<view>/x.png);
    # "parent_dir" = folder containing the image; an int = that path component.
    "nlbsd_case_id_from_path": "uuid",
    # Keep only rows whose 'Status' column equals this (None = keep all).
    "nlbsd_status_keep": "OK",
    # STRONGLY RECOMMENDED: the split.csv written by the ORIGINAL Mammo-CLIP baseline run
    # (<old_output_root>/nlbsd_baseline/split.csv). Using it guarantees the exact same NLBSD test set as
    # the paper, so the ImageNet and Mammo-CLIP results are directly comparable.  <-- CHECK
    "original_split_csv": os.path.join(BASE, "imagenet_control", "nlbsd_baseline", "split.csv"),   # server split

    # ---------------- CMMD ----------------
    "cmmd_clinical_csv": os.path.join(BASE, "CMMD_clinicaldata_revision.csv"),   # <-- CHECK file location
    "cmmd_image_root": os.path.join(BASE, "CMMD"),   # preprocessed PNG folders D1-0001, D2-0001 ...
    # A patient folder is included only if it has a Malignant row and NO Benign row (CMMD lists some
    # patients twice, one row per breast; if images in a folder can't be told apart by side, a mixed
    # patient would leak benign images in, so they are dropped and reported).
    "cmmd_drop_mixed_patients": True,

    # ---------------- CBIS-DDSM ----------------
    "ddsm_label_csv": os.path.join(BASE, "ddsm_invasive.csv"),           # <-- CHECK file name   # columns png_path,cancer
    "ddsm_malignant_values": ["invasive"],
    # CBIS-DDSM folders ending in _1, _2 ... hold ROI masks and lesion crops, not full mammograms.
    # True = keep only full-mammogram folders (..._LEFT_CC, ..._RIGHT_MLO, ...).
    "ddsm_full_mammograms_only": True,

    # "malignant" = only malignant external images (what you asked for). "all" = every external image,
    # which is what the paper repo's loader did (it globbed every PNG under the folder).  <-- CHECK vs. original runs
    "external_filter": "malignant",

    # ---------------- output ----------------
    "output_root": os.path.join(BASE, "imagenet_control"),   # put the server folders nlbsd_baseline + nlbsd_plus_cmmd here

    # ---------------- identical to the paper ----------------
    "image_size": [1520, 912],                                  # H, W
    # Kept identical to the Mammo-CLIP runs so only the weights change. Set to "imagenet" to use
    # ImageNet mean/std instead (then say so in the paper).
    "normalization": "paper",
    # ---------- this experiment ----------
    "encoder": "mammoclip",                       # "mammoclip" (paper) or "imagenet"
    "mammo_clip_checkpoint": os.path.join(BASE, "b5-model-best-epoch-7.tar"),   # <-- CHECK location
    "harmonize": True,                            # histogram matching to the NLBSD-training reference
    "harmonize_ref_images": 300,                  # NLBSD training images used to build the reference
    "origin_run_name": "origin_harmonized_mammoclip",
    "norm_paper": {"mean": [0.3089279] * 3, "std": [0.25053555408335154] * 3},
    "norm_imagenet": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    "split": {"test_frac": 0.2, "val_frac": 0.1, "seed": 42},
    "head_dropout": 0.3,
    # Images per encoder pass (batch of 80 is split into 80/encoder_chunk passes). Must be <= ~40 at
    # 1520x912. 20 = what 4-GPU DataParallel gave per GPU in the original runs, if they used 4 GPUs.
    "encoder_chunk": 8,     # images per encoder pass; lower to 4 if you get "CUDA out of memory"
    "train": {"num_epochs": 30, "batch_size": 80, "origin_batch_size": 81, "positives_per_batch": 20,
              "lr": 3e-5, "weight_decay": 1e-4, "amp": True, "grad_clip": 5.0, "seed": 42,
              "patience": 10, "num_workers": 4},   # num_workers 0 if loading crashes on Windows
    "augment": {"rotation_deg": 10, "translate_frac": 0.05, "scale_min": 0.9, "scale_max": 1.1,
                "brightness": 0.05, "contrast_low": 0.90, "contrast_high": 1.10},
    "bootstrap_in_run": 1000, "bootstrap_compare": 2000, "alpha": 0.05,
}

# What the PyCharm "Run" button does (no command-line arguments needed).
# Finished runs (with metrics.csv) are skipped automatically.
RUN_PLAN = []   # not used by the t-SNE script

RUNS = {
    "baseline":  {"run_name": "nlbsd_baseline",       "external": []},
    "cmmd":      {"run_name": "nlbsd_plus_cmmd",      "external": ["cmmd"]},
    "ddsm":      {"run_name": "nlbsd_plus_ddsm",      "external": ["ddsm"]},
    "ddsm_cmmd": {"run_name": "nlbsd_plus_ddsm_cmmd", "external": ["ddsm", "cmmd"]},
}
DISPLAY = {"baseline": "NLBSD", "cmmd": "NLBSD + CMMD+", "ddsm": "NLBSD + DDSM+",
           "ddsm_cmmd": "NLBSD + DDSM+ + CMMD+"}
ORIGIN_CLASSES = ["nlbsd", "ddsm", "cmmd"]

log = logging.getLogger("imagenet_control")


# =====================================================================================================
# utils
# =====================================================================================================
def setup_logging(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO); log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        h.setFormatter(fmt); log.addHandler(h)


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def write_csv(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if not k.startswith("_") and k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def atomic_save(obj, path):
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp"); torch.save(obj, tmp); os.replace(tmp, path)


def read_table(path):
    """Reads comma- or tab-separated files (also Excel exports with a BOM)."""
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def path_key(p):
    """Machine-independent image key: last 5 path parts, e.g. abnormal/<uuid>/left-c/CC/IM-...png
    (server paths and D:\\ paths of the same image give the same key)."""
    return "/".join(str(p).replace("\\", "/").split("/")[-5:]).lower()


def norm_stats():
    n = CFG["norm_paper"] if CFG["normalization"] == "paper" else CFG["norm_imagenet"]
    return n["mean"], n["std"]


# =====================================================================================================
# data lists
# =====================================================================================================
def _pick(df, given, candidates, what):
    if given:
        if given not in df.columns:
            raise KeyError(f"NLBSD manifest has no column '{given}' for {what}. Columns: {list(df.columns)}")
        return given
    low = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def detect_nlbsd_columns(df):
    c = CFG["nlbsd_cols"]
    ip = _pick(df, c["image_path"], ["PNG Path", "File Path", "png_path", "image_path", "filepath", "file_path", "path", "png"], "image path")
    lb = _pick(df, c["label"], ["Cancer", "label", "suspicious", "abnormal", "target", "y"], "label")
    if c["case_id"] == "":
        return ip, lb, None
    ci = _pick(df, c["case_id"], ["case_id", "Patient ID", "patient_id", "PatientID", "exam_id", "study_id"], "case id")
    if ip is None or lb is None:
        raise KeyError(f"Could not auto-detect image-path/label columns in the NLBSD manifest. "
                       f"Columns: {list(df.columns)} -> set CFG['nlbsd_cols'].")
    return ip, lb, ci


def load_nlbsd():
    import pandas as pd
    df = read_table(CFG["nlbsd_manifest"])
    ip, lb, ci = detect_nlbsd_columns(df)
    keep = CFG.get("nlbsd_status_keep")
    if keep and "Status" in df.columns:
        log.info("[NLBSD] Status values: %s -> keeping '%s'", df["Status"].value_counts().to_dict(), keep)
        df = df[df["Status"].astype(str) == keep].reset_index(drop=True)
    if ci is not None and df[ci].isna().mean() > 0.5:
        log.warning("[NLBSD] case_id column '%s' is mostly empty -> deriving case_id from the path", ci)
        ci = None
    out = pd.DataFrame({"image_path": df[ip].astype(str).str.replace("\\", "/", regex=False),
                        "label": df[lb].astype(int)})
    rep = CFG.get("nlbsd_path_replace")
    if rep:
        out["image_path"] = out["image_path"].str.replace(rep[0], rep[1], regex=False)
    root = CFG["nlbsd_image_root"]
    if root:
        out["image_path"] = out["image_path"].apply(lambda p: p if os.path.isabs(p) else os.path.join(root, p))
    if ci is not None:
        out["case_id"] = df[ci].astype(str)
        how = f"column '{ci}'"
    else:
        rule = CFG["nlbsd_case_id_from_path"]
        if rule == "uuid":
            uu = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
            def _uid(p):   # the UUID-named folder closest to the image
                hits = [x for x in Path(p).parts if uu.fullmatch(x)]
                return hits[-1] if hits else Path(p).parent.parent.parent.name
            out["case_id"] = out["image_path"].apply(_uid)
        elif rule == "parent_dir":
            out["case_id"] = out["image_path"].apply(lambda p: Path(p).parent.name)
        else:
            out["case_id"] = out["image_path"].apply(lambda p: p.split("/")[int(rule)])
        how = f"path ({rule})"
    bad = sorted(set(out["label"].unique()) - {0, 1})
    if bad:
        raise ValueError(f"NLBSD label column '{lb}' has values other than 0/1: {bad}")
    exists = out["image_path"].apply(os.path.exists)
    miss = int((~exists).sum())
    if miss:
        for p in out.loc[~exists, "image_path"].head(10):
            log.warning("[NLBSD missing] %s", p)
        if miss / len(out) > 0.10:
            raise RuntimeError(f"{miss}/{len(out)} NLBSD files missing -- check nlbsd_image_root / manifest paths.")
        log.warning("Dropping %d missing NLBSD files.", miss)
    out = out[exists].reset_index(drop=True)
    out["source"] = "nlbsd"
    log.info("[NLBSD] columns: path='%s' label='%s' case_id=%s", ip, lb, how)
    log.info("[NLBSD] %d images | %d cases | pos=%d neg=%d | pos cases=%d", len(out), out["case_id"].nunique(),
             int((out.label == 1).sum()), int((out.label == 0).sum()),
             int(out.groupby("case_id")["label"].max().sum()))
    return out


CMMD_ID = re.compile(r"^D[12]-\d{4}$")


def find_cmmd_folders(root):
    """Walks `root` and returns {patient_id: folder}. Stops descending once a D1-xxxx/D2-xxxx folder is found."""
    found = {}
    for dirpath, dirnames, _ in os.walk(root):
        keep = []
        for d in dirnames:
            if CMMD_ID.match(d):
                if d in found:
                    log.warning("[CMMD] duplicate folder for %s: %s and %s (keeping first)", d, found[d],
                                os.path.join(dirpath, d))
                else:
                    found[d] = os.path.join(dirpath, d)
            elif not d.startswith("."):
                keep.append(d)
        dirnames[:] = keep
    return found


def load_cmmd(label_col="label", label_value=1):
    import pandas as pd
    clin = read_table(CFG["cmmd_clinical_csv"])
    idc = "ID1" if "ID1" in clin.columns else clin.columns[0]
    clin[idc] = clin[idc].astype(str).str.strip()
    cls = clin["classification"].astype(str).str.strip().str.lower()
    per_pat = clin.assign(_c=cls).groupby(idc)["_c"].agg(lambda s: set(s))
    mal = {p for p, s in per_pat.items() if "malignant" in s}
    mixed = {p for p, s in per_pat.items() if "malignant" in s and "benign" in s}
    if CFG["external_filter"] == "malignant":
        keep = mal - mixed if CFG["cmmd_drop_mixed_patients"] else mal
    else:
        keep = set(per_pat.index)
    folders = find_cmmd_folders(CFG["cmmd_image_root"])
    rows, no_png, dcm_only = [], [], []
    for pid in sorted(keep):
        f = folders.get(pid)
        if f is None:
            continue
        pngs = sorted(p for p in Path(f).rglob("*") if p.suffix.lower() == ".png")
        if not pngs:
            (dcm_only if any(p.suffix.lower() == ".dcm" for p in Path(f).rglob("*")) else no_png).append(pid)
            continue
        rows += [{"image_path": str(p), label_col: label_value, "case_id": f"cmmd_{pid}", "source": "cmmd"}
                 for p in pngs]
    log.info("[CMMD] clinical: %d patients | malignant=%d | mixed benign+malignant=%d (%s)",
             len(per_pat), len(mal), len(mixed), "dropped" if CFG["cmmd_drop_mixed_patients"] else "kept")
    log.info("[CMMD] folders found under %s: %d | selected patients with folder: %d | missing folder: %d",
             CFG["cmmd_image_root"], len(folders), len({r['case_id'] for r in rows}),
             len([p for p in keep if p not in folders]))
    if dcm_only:
        raise RuntimeError(f"[CMMD] {len(dcm_only)} selected folders contain only DICOM, no PNG (e.g. "
                           f"{folders[dcm_only[0]]}). Point cmmd_image_root at the PREPROCESSED PNG folders.")
    if no_png:
        log.warning("[CMMD] %d selected folders have no images (e.g. %s)", len(no_png), no_png[:5])
    if not rows:
        raise RuntimeError("[CMMD] no images selected -- check cmmd_image_root.")
    df = pd.DataFrame(rows)
    log.info("[CMMD] %d images | %d patients", len(df), df.case_id.nunique())
    return df


DDSM_FULL = re.compile(r"^(Calc|Mass)-(Training|Test)_P_\d+_(LEFT|RIGHT)_(CC|MLO)$", re.I)
DDSM_PID = re.compile(r"(P_\d+)", re.I)


def load_ddsm(label_col="label", label_value=1):
    import pandas as pd
    if not CFG["ddsm_label_csv"]:
        raise RuntimeError("Set CFG['ddsm_label_csv'] to the CBIS-DDSM CSV (columns png_path,cancer).")
    lab = read_table(CFG["ddsm_label_csv"])
    lab["png_path"] = lab["png_path"].astype(str).str.strip()
    lab["cancer"] = lab["cancer"].astype(str).str.strip().str.lower()
    log.info("[DDSM] label values: %s", lab["cancer"].value_counts().to_dict())
    n0 = len(lab)
    if CFG["ddsm_full_mammograms_only"]:
        is_full = lab["png_path"].apply(lambda p: bool(DDSM_FULL.match(Path(p).parent.name)))
        log.info("[DDSM] full-mammogram rows: %d | ROI/crop rows (_1, _2 ... folders) excluded: %d",
                 int(is_full.sum()), int((~is_full).sum()))
        lab = lab[is_full]
    if CFG["external_filter"] == "malignant":
        mal_vals = [v.lower() for v in CFG["ddsm_malignant_values"]]
        lab = lab[lab["cancer"].isin(mal_vals)]
    # The same image can appear on several rows (one per lesion) -- keep it once, malignant if any row is.
    lab = lab.drop_duplicates("png_path")
    exists = lab["png_path"].apply(os.path.exists)
    if (~exists).sum():
        log.warning("[DDSM] %d listed files missing (e.g. %s)", int((~exists).sum()),
                    lab.loc[~exists, "png_path"].head(3).tolist())
    lab = lab[exists]
    if lab.empty:
        raise RuntimeError("[DDSM] no images selected.")
    df = pd.DataFrame({"image_path": lab["png_path"].values, label_col: label_value, "source": "ddsm",
                       "case_id": ["ddsm_" + (DDSM_PID.search(Path(p).parent.name).group(1).upper()
                                             if DDSM_PID.search(Path(p).parent.name) else Path(p).parent.name)
                                   for p in lab["png_path"]]})
    log.info("[DDSM] %d of %d rows kept -> %d images | %d patients", len(df), n0, len(df), df.case_id.nunique())
    return df


def load_external(name, label_col="label", label_value=1):
    return {"cmmd": load_cmmd, "ddsm": load_ddsm}[name](label_col, label_value)


# =====================================================================================================
# split (identical algorithm to the paper repo)
# =====================================================================================================
def stratified_case_split(df, frac, seed, label_col):
    rng = np.random.default_rng(seed)
    case_lab = df.groupby("case_id")[label_col].max()
    a = []
    for lab in sorted(case_lab.unique()):
        cs = np.array(case_lab[case_lab == lab].index); rng.shuffle(cs)
        a.extend(cs[:int(round(len(cs) * frac))])
    a = set(a)
    return df[df.case_id.isin(a)].reset_index(drop=True), df[~df.case_id.isin(a)].reset_index(drop=True)


def make_split(df, out_dir, label_col, reuse_csv=None):
    import pandas as pd
    pinned = out_dir / "split.csv"
    src = reuse_csv if (reuse_csv and Path(reuse_csv).exists()) else (pinned if pinned.exists() else None)
    if src:
        sp = pd.read_csv(src)[["case_id", "split"]].drop_duplicates()
        sp["case_id"] = sp["case_id"].astype(str)
        df = df.drop(columns=["split"], errors="ignore").merge(sp, on="case_id", how="left")
        n_un = int(df["split"].isna().sum())
        if n_un:
            raise RuntimeError(f"{n_un} images have case_ids not in {src}. The case_id definition doesn't match "
                               f"the original run -- fix nlbsd_case_id_from_path / nlbsd_cols['case_id'].")
        log.info("Split loaded from %s", src)
        if not pinned.exists():
            df[["case_id", "split"]].drop_duplicates().to_csv(pinned, index=False)
        return df
    s = CFG["split"]
    test, rem = stratified_case_split(df, s["test_frac"], s["seed"], label_col)
    val, train = stratified_case_split(rem, s["val_frac"] / (1 - s["test_frac"]), s["seed"] + 1, label_col)
    m = {**{c: "train" for c in train.case_id.unique()}, **{c: "val" for c in val.case_id.unique()},
         **{c: "test" for c in test.case_id.unique()}}
    df = df.copy(); df["split"] = df["case_id"].map(m)
    df[["case_id", "split"]].drop_duplicates().to_csv(pinned, index=False)
    log.info("Split created (seed %d) -> %s", s["seed"], pinned)
    return df


# =====================================================================================================
# dataset / sampler (identical to the paper repo)
# =====================================================================================================
HIST_LEVELS = np.linspace(0.0, 1.0, 1001, dtype=np.float32)


def match_histogram(a, ref_q):
    """Histogram matching of breast pixels (> 0; background is 0 after preprocessing) to reference quantiles."""
    fg = a > 0
    if fg.sum() < 100:
        return a
    v = a[fg]
    src_q = np.quantile(v, HIST_LEVELS).astype(np.float32)
    src_q = np.maximum.accumulate(src_q + np.arange(len(src_q), dtype=np.float32) * 1e-7)   # strictly increasing
    out = a.copy()
    out[fg] = np.interp(v, src_q, ref_q).astype(np.float32)
    return out


def build_hist_reference(df_train_nlbsd, image_size, n_images, out_dir):
    """Reference intensity quantiles from NLBSD training images only (no test data involved)."""
    ds = MammoDataset(df_train_nlbsd.sample(min(n_images, len(df_train_nlbsd)), random_state=0),
                      image_size, augment=False, label_col="origin", label_dtype="long")
    import cv2
    vals = []
    for i in range(len(ds)):
        a = ds._read(ds.df.iloc[i]["image_path"])
        if a.shape != (ds.H, ds.W):
            a = cv2.resize(a, (ds.W, ds.H), interpolation=cv2.INTER_LINEAR)
        a = a - a.min(); mx = a.max()
        if mx > 0:
            a = a / mx
        fg = a[a > 0]
        if fg.size:
            vals.append(np.quantile(fg, HIST_LEVELS))
    ref = np.mean(np.stack(vals), axis=0).astype(np.float32)
    ref = np.maximum.accumulate(ref)
    json.dump({"levels": HIST_LEVELS.tolist(), "reference_quantiles": ref.tolist(), "n_images": len(vals)},
              open(out_dir / "histogram_reference.json", "w"))
    log.info("[HARMONIZE] reference built from %d NLBSD training images", len(vals))
    return ref


class MammoDataset:
    def __init__(self, df, image_size, augment, label_col="label", label_dtype="float", hist_ref=None):
        self.df = df.reset_index(drop=True)
        self.hist_ref = None if hist_ref is None else np.asarray(hist_ref, np.float32)
        self.H, self.W = image_size
        mean, std = norm_stats()
        self.mean = np.array(mean, np.float32).reshape(3, 1, 1)
        self.std = np.array(std, np.float32).reshape(3, 1, 1)
        self.augment, self.label_col, self.label_dtype = augment, label_col, label_dtype

    def __len__(self):
        return len(self.df)

    def _read(self, path):
        from PIL import Image
        try:
            a = np.array(Image.open(path), dtype=np.float32)
        except Exception as e:
            log.warning("[read-fail] %s (%s) -> zero image", path, e)
            return np.zeros((self.H, self.W), np.float32)
        return a.mean(axis=2) if a.ndim == 3 else a

    def _augment(self, a):
        import cv2
        c = CFG["augment"]; h, w = a.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), random.uniform(-c["rotation_deg"], c["rotation_deg"]),
                                    random.uniform(c["scale_min"], c["scale_max"]))
        M[0, 2] += random.uniform(-c["translate_frac"], c["translate_frac"]) * w
        M[1, 2] += random.uniform(-c["translate_frac"], c["translate_frac"]) * h
        a = cv2.warpAffine(a, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        a = np.clip(a + random.uniform(-c["brightness"], c["brightness"]), 0, 1)
        return np.clip(a * random.uniform(c["contrast_low"], c["contrast_high"]), 0, 1)

    def __getitem__(self, i):
        import torch, cv2
        r = self.df.iloc[i]
        a = self._read(r["image_path"])
        if a.shape != (self.H, self.W):
            a = cv2.resize(a, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        a = a - a.min(); mx = a.max()
        if mx > 0:
            a = a / mx
        if self.hist_ref is not None:
            a = match_histogram(a, self.hist_ref)
        if self.augment:
            a = self._augment(a)
        t = (np.repeat(a[None], 3, axis=0).astype(np.float32) - self.mean) / self.std
        y = (torch.tensor(float(r[self.label_col])) if self.label_dtype == "float"
             else torch.tensor(int(r[self.label_col]), dtype=torch.long))
        return torch.from_numpy(t), y, str(r["image_path"]), str(r["case_id"])


class BalancedBatchSampler:
    def __init__(self, labels, batch_size, quotas, seed=0):
        self.labels = np.asarray(labels); self.bs = batch_size; self.quotas = quotas
        self.idx = {c: np.where(self.labels == c)[0] for c in quotas}
        self.rng = np.random.default_rng(seed)
        self.n = max(1, len(self.labels) // batch_size)

    def __iter__(self):
        for _ in range(self.n):
            b = []
            for c, q in self.quotas.items():
                if len(self.idx[c]):
                    b.extend(self.rng.choice(self.idx[c], q, replace=len(self.idx[c]) < q).tolist())
            self.rng.shuffle(b)
            yield b

    def __len__(self):
        return self.n


# =====================================================================================================
# model: EfficientNet-B5, ImageNet weights, frozen, LayerNorm -> Dropout(0.3) -> Linear
# =====================================================================================================
def load_mammoclip(encoder):
    """Same loading as the paper's pipeline: keys 'image_encoder.*' from the Mammo-CLIP checkpoint."""
    import torch
    ck = CFG["mammo_clip_checkpoint"]
    if not os.path.exists(ck):
        raise FileNotFoundError(f"Mammo-CLIP checkpoint not found: {ck}  -> set CFG['mammo_clip_checkpoint']")
    c = torch.load(ck, map_location="cpu", weights_only=False)
    state = c["model"] if (isinstance(c, dict) and "model" in c) else c
    enc = {k[len("image_encoder."):]: v for k, v in state.items() if k.startswith("image_encoder.")}
    miss, unexp = encoder.load_state_dict(enc, strict=False)
    tot = len(encoder.state_dict()); ok = tot - len(miss)
    log.info("[MAMMO-CLIP] encoder weights matched %d/%d (unexpected keys: %d)", ok, tot, len(unexp))
    if ok < 0.9 * tot:
        raise RuntimeError("Less than 90% of Mammo-CLIP encoder weights matched -- wrong checkpoint file?")


def build_model(n_out, pretrained=True):
    import torch, torch.nn as nn
    from efficientnet_pytorch import EfficientNet

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.amp = False
            if CFG.get("encoder") == "mammoclip" and pretrained:
                self.encoder = EfficientNet.from_name("efficientnet-b5")
                load_mammoclip(self.encoder)
            else:
                self.encoder = (EfficientNet.from_pretrained("efficientnet-b5") if pretrained
                                else EfficientNet.from_name("efficientnet-b5"))
            d = self.encoder._fc.in_features
            self.encoder._fc = nn.Identity(); self.encoder._dropout = nn.Identity()
            self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(CFG["head_dropout"]), nn.Linear(d, n_out))
            for p in self.encoder.parameters():
                p.requires_grad = False

        def forward(self, x):
            # Encoder is frozen: run it without building an autograd graph (saves GPU memory; identical
            # outputs). BatchNorm still behaves exactly as before in train/eval mode.
            # A full batch of 80 at 1520x912 exceeds PyTorch's 32-bit indexing limit inside the encoder, so
            # the frozen encoder sees the batch in chunks of `encoder_chunk` images; features are concatenated.
            ch = CFG.get("encoder_chunk", 20)
            with torch.cuda.amp.autocast(enabled=self.amp):
                with torch.no_grad():
                    f = torch.cat([self.encoder._avg_pooling(self.encoder.extract_features(xc)).flatten(1)
                                   for xc in torch.split(x, ch)])
                o = self.head(f.float() if not self.amp else f)
            return o.squeeze(-1) if n_out == 1 else o

    return Net()


# =====================================================================================================
# metrics / statistics (identical definitions to the paper repo)
# =====================================================================================================
def youden(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def boot_auc_ci(y, p, n, seed=42, conf=0.95):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y), np.asarray(p)
    point = float(roc_auc_score(y, p))
    pi_, ni_ = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed); aucs = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pi_, len(pi_)), rng.choice(ni_, len(ni_))])
        aucs.append(roc_auc_score(y[idx], p[idx]))
    lo = (1 - conf) / 2 * 100
    return point, float(np.percentile(aucs, lo)), float(np.percentile(aucs, 100 - lo))


def boot_rate_ci(y, pred, n, seed=42):
    """Stratified bootstrap CIs for sensitivity, specificity, accuracy at a fixed threshold."""
    y, pred = np.asarray(y), np.asarray(pred)
    pi_, ni_ = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed); se, sp, ac = [], [], []
    for _ in range(n):
        a, b = rng.choice(pi_, len(pi_)), rng.choice(ni_, len(ni_))
        se.append(pred[a].mean()); sp.append(1 - pred[b].mean())
        ac.append((pred[a].sum() + (1 - pred[b]).sum()) / (len(a) + len(b)))
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(se), q(sp), q(ac)


def binary_metrics(y, p, thr, level, n_boot):
    from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
    y, p = np.asarray(y).astype(int), np.asarray(p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {"level": level, "n": len(y), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
         "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
         "threshold": thr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
         "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
         "accuracy": (tp + tn) / max(len(y), 1)}
    if n_boot:
        _, m["roc_auc_ci_lo"], m["roc_auc_ci_hi"] = boot_auc_ci(y, p, n_boot)
        (m["sens_ci_lo"], m["sens_ci_hi"]), (m["spec_ci_lo"], m["spec_ci_hi"]), \
            (m["acc_ci_lo"], m["acc_ci_hi"]) = boot_rate_ci(y, pred, n_boot)
    return m


def aggregate_max(case_ids, y, p):
    d = {}
    for c, yy, pp in zip(case_ids, y, p):
        a = d.setdefault(c, [0, 0.0]); a[0] = max(a[0], int(yy)); a[1] = max(a[1], float(pp))
    k = list(d)
    return np.array([d[c][0] for c in k]), np.array([d[c][1] for c in k])


def multiclass_metrics(y, probs, names):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    y, probs = np.asarray(y), np.asarray(probs); pred = probs.argmax(1); n = len(names)
    m = {"n": len(y), "balanced_acc": float(balanced_accuracy_score(y, pred))}
    for i, f in enumerate(f1_score(y, pred, labels=list(range(n)), average=None, zero_division=0)):
        m[f"f1_{names[i]}"] = float(f)
    try:
        m["macro_auc"] = float(roc_auc_score(y, probs, multi_class="ovr", average="macro", labels=list(range(n))))
    except ValueError:
        m["macro_auc"] = float("nan")
    for i, c in enumerate(names):
        yb = (y == i).astype(int)
        m[f"auc_{c}"] = float(roc_auc_score(yb, probs[:, i])) if 0 < yb.sum() < len(yb) else float("nan")
    m["_cm"] = confusion_matrix(y, pred, labels=list(range(n)))
    return m


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1; i = j
    out = np.empty(N); out[J] = T
    return out


def delong_test(y, pa, pb):
    """Two-sided DeLong test for two correlated ROC curves (Sun & Xu 2014 fast form)."""
    from scipy.stats import norm
    y = np.asarray(y).astype(int); order = (-y).argsort(); m = int(y.sum()); n = len(y) - m
    X = np.vstack([pa, pb])[:, order]
    tx = np.array([_midrank(r[:m]) for r in X]); ty = np.array([_midrank(r[m:]) for r in X])
    tz = np.array([_midrank(r) for r in X])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var <= 0:
        return float("nan"), aucs
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(2 * (1 - norm.cdf(abs(z)))), aucs


def holm(p):
    p = np.asarray(p, float)
    if np.isnan(p).any():          # undefined DeLong p (identical predictions) -> don't report significance
        return np.full(len(p), np.nan)
    o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for i, k in enumerate(o):
        run = max(run, min((m - i) * p[k], 1.0)); adj[k] = run
    return adj


# =====================================================================================================
# training loops
# =====================================================================================================
def make_loaders(fit, val, test, label_col, label_dtype, quotas, bs, img_size, nw, hist_ref=None):
    from torch.utils.data import DataLoader
    kw = dict(image_size=img_size, label_col=label_col, label_dtype=label_dtype, hist_ref=hist_ref)
    sampler = BalancedBatchSampler(fit[label_col].to_numpy(), bs, quotas)
    import torch
    ex = dict(num_workers=nw, pin_memory=False, persistent_workers=nw > 0)
    return (DataLoader(MammoDataset(fit, augment=True, **kw), batch_sampler=sampler, **ex),
            DataLoader(MammoDataset(val, augment=False, **kw), batch_size=bs, **ex),
            DataLoader(MammoDataset(test, augment=False, **kw), batch_size=bs, **ex))


def infer(net, loader, device, multiclass):
    import torch, torch.nn.functional as F
    net.eval(); P, Y, PA, CI = [], [], [], []
    with torch.no_grad():
        for x, y, pa, ci in loader:
            out = net(x.to(device, non_blocking=True)).float()
            P.append((F.softmax(out, 1) if multiclass else torch.sigmoid(out)).cpu().numpy())
            Y.append(y.numpy()); PA += list(pa); CI += list(ci)
    return np.concatenate(P), np.concatenate(Y), PA, CI


def fit(net, loaders, device, out_dir, multiclass, names, epochs, smoke):
    """Same loop as the paper: AdamW, AMP, grad-clip 5, early stopping on val AUC (macro AUC for origin)."""
    import torch, torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    t = CFG["train"]; fit_l, val_l, _ = loaders
    amp = t["amp"] and device.type == "cuda"; net.amp = amp
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=t["lr"], weight_decay=t["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ck = out_dir / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    last, best = ck / "last.pth", ck / "best.pth"
    hist, best_m, start, bad = [], -1.0, 1, 0
    if last.exists() and not smoke:
        c = torch.load(last, map_location=device, weights_only=False)
        net.load_state_dict(c["model"]); opt.load_state_dict(c["opt"])
        start, hist, best_m, bad = c["epoch"] + 1, c["history"], c["best"], c["bad"]
        log.info("[RESUMED] from epoch %d", start)
    for ep in range(start, epochs + 1):
        net.train(); tot, n, t0 = 0.0, 0, time.time()
        for bi, (x, y, _, _) in enumerate(fit_l):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                out = net(x)
                loss = F.cross_entropy(out, y) if multiclass else F.binary_cross_entropy_with_logits(out, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), t["grad_clip"])
            scaler.step(opt); scaler.update()
            tot += float(loss.item()) * len(y); n += len(y)
            if (bi + 1) % 50 == 0:
                log.info("    ep%d batch %d/%d loss=%.4f (%.1f s)", ep, bi + 1, len(fit_l), tot / n, time.time() - t0)
        vp, vy, _, _ = infer(net, val_l, device, multiclass)
        if multiclass:
            vm = multiclass_metrics(vy, vp, names); metric = vm["macro_auc"]
            hist.append({"epoch": ep, "train_loss": tot / n, "val_macro_auc": metric, "val_bal_acc": vm["balanced_acc"]})
        else:
            metric = float(roc_auc_score(vy, vp)) if len(np.unique(vy)) > 1 else 0.0
            hist.append({"epoch": ep, "train_loss": tot / n, "val_auc": metric})
        log.info("  [ep %d/%d] loss=%.4f val=%.4f (%.0f s)", ep, epochs, tot / n, metric, time.time() - t0)
        if metric > best_m:
            best_m, bad = metric, 0
            atomic_save({"epoch": ep, "model": copy.deepcopy(net.state_dict()), "val": metric}, best)
            log.info("  [BEST] ep%d val=%.4f", ep, metric)
        else:
            bad += 1
        atomic_save({"epoch": ep, "model": net.state_dict(), "opt": opt.state_dict(), "history": hist,
                     "best": best_m, "bad": bad}, last)
        if bad >= t["patience"]:
            log.info("  [EARLY-STOP] no improvement for %d epochs", t["patience"]); break
    if best.exists():
        net.load_state_dict(torch.load(best, map_location=device, weights_only=False)["model"])
    write_csv(hist, out_dir / "training_history.csv")
    return net


def check_gpu(device, need_gb=3):
    import torch
    if device.type != "cuda":
        return
    free, total = torch.cuda.mem_get_info()
    log.info("[GPU] %s | free %.1f / %.1f GB", torch.cuda.get_device_name(0), free / 1e9, total / 1e9)
    if free / 1e9 < need_gb:
        raise RuntimeError(f"Only {free / 1e9:.1f} GB free on this GPU -- another job is using it. "
                           f"Resubmit with an exclusive GPU: -gpu \"num=1:mode=exclusive_process\"")


def run_dir(name, smoke):
    d = Path(CFG["output_root"]) / (("smoke/" + name) if smoke else name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def train_binary(key, smoke=False):
    import torch, pandas as pd
    spec = RUNS[key]; out = run_dir(spec["run_name"], smoke)
    if not smoke and (out / "metrics.csv").exists():
        print(f"[SKIP] {DISPLAY[key]} already finished ({out / 'metrics.csv'}) -- delete that folder to rerun it.")
        return
    setup_logging(out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    seed_everything(CFG["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_gpu(device)
    log.info("=== %s | ImageNet EfficientNet-B5 frozen linear probe | device=%s | external_filter=%s | norm=%s ===",
             DISPLAY[key], device, CFG["external_filter"], CFG["normalization"])
    json.dump({"run": key, "cfg": CFG, "time": time.ctime()}, open(out / "config_snapshot.json", "w"), indent=2, default=str)

    df = load_nlbsd()
    if smoke:
        df = pd.concat([df[df.label == 1].sample(min(25, (df.label == 1).sum()), random_state=0),
                        df[df.label == 0].sample(25, random_state=0)])
    df = make_split(df, out, "label", None if smoke else CFG["original_split_csv"])
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    ext = [load_external(e) for e in spec["external"]]
    if smoke:
        ext = [e.sample(min(10, len(e)), random_state=0) for e in ext]
    fit_df = pd.concat([tr] + ext, ignore_index=True)

    counts = [{"set": "train", "source": s, "images": int((fit_df.source == s).sum()),
               "pos_images": int(((fit_df.source == s) & (fit_df.label == 1)).sum()),
               "cases": int(fit_df[fit_df.source == s].case_id.nunique())} for s in fit_df.source.unique()]
    counts += [{"set": n_, "source": "nlbsd", "images": len(d), "pos_images": int(d.label.sum()),
                "cases": int(d.case_id.nunique())} for n_, d in (("val", va), ("test", te))]
    write_csv(counts, out / "data_counts.csv")
    for c in counts:
        log.info("  counts %s", c)

    t = CFG["train"]; bs = t["batch_size"]
    img = [256, 160] if smoke else CFG["image_size"]
    loaders = make_loaders(fit_df, va, te, "label", "float", {1: t["positives_per_batch"], 0: bs - t["positives_per_batch"]},
                           bs, img, 0 if smoke else t["num_workers"])
    net = build_model(1).to(device)
    net = fit(net, loaders, device, out, False, None, 2 if smoke else t["num_epochs"], smoke)

    thr_file = run_dir(RUNS["baseline"]["run_name"], smoke) / "baseline_threshold.json"
    if key == "baseline":
        vp, vy, _, vc = infer(net, loaders[1], device, False)
        cy, cp = aggregate_max(vc, vy, vp)
        thr = {"image": youden(vy, vp), "case": youden(cy, cp)}
        json.dump(thr, open(thr_file, "w"), indent=2)
        log.info("[THRESHOLD] Youden on baseline val split: %s", thr)
    else:
        if not thr_file.exists():
            raise RuntimeError(f"Run the baseline first: {thr_file} not found.")
        thr = json.load(open(thr_file)); log.info("[THRESHOLD] shared baseline thresholds: %s", thr)

    tp, ty, tpa, tci = infer(net, loaders[2], device, False)
    nb = 50 if smoke else CFG["bootstrap_in_run"]
    mi = binary_metrics(ty, tp, thr["image"], "image", nb)
    cy, cp = aggregate_max(tci, ty, tp)
    mc = binary_metrics(cy, cp, thr["case"], "case", nb)
    write_csv([mi, mc], out / "metrics.csv")
    write_csv([{"path": a, "case_id": c, "label": int(y), "prob": float(p), "split": "test", "domain": "nlbsd"}
               for a, c, y, p in zip(tpa, tci, ty, tp)], out / "per_sample_predictions.csv")
    log.info("[RESULT] %s image AUC=%.3f [%.3f, %.3f] Sen=%.2f Spc=%.2f Acc=%.2f | case AUC=%.3f",
             DISPLAY[key], mi["roc_auc"], mi.get("roc_auc_ci_lo", np.nan), mi.get("roc_auc_ci_hi", np.nan),
             mi["sensitivity"], mi["specificity"], mi["accuracy"], mc["roc_auc"])
    log.info("STATUS: SUCCESS")


def save_harmonization_check(df, img_size, href, out_dir, k=2):
    """Before/after picture for a few training images per dataset (visual sanity check)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = [df[df.source == s].sample(min(k, int((df.source == s).sum())), random_state=1) for s in ORIGIN_CLASSES]
    rows = pd_concat(rows)
    ds = MammoDataset(rows, img_size, augment=False, label_col="origin", label_dtype="long")
    import cv2
    fig, axes = plt.subplots(len(rows), 2, figsize=(4, 2.2 * len(rows)))
    for i in range(len(rows)):
        a = ds._read(rows.iloc[i]["image_path"])
        if a.shape != (ds.H, ds.W):
            a = cv2.resize(a, (ds.W, ds.H), interpolation=cv2.INTER_LINEAR)
        a = a - a.min(); a = a / a.max() if a.max() > 0 else a
        for j, im in enumerate((a, match_histogram(a, href))):
            axes[i, j].imshow(im, cmap="gray", vmin=0, vmax=1); axes[i, j].axis("off")
            axes[i, j].set_title(f"{rows.iloc[i]['source']} {'before' if j == 0 else 'after'}", fontsize=6)
    fig.tight_layout(); fig.savefig(out_dir / "harmonization_check.png", dpi=150); plt.close(fig)


def pd_concat(parts):
    import pandas as pd
    return pd.concat(parts, ignore_index=True)


def train_origin(smoke=False):
    import torch, pandas as pd
    out = run_dir(CFG["origin_run_name"], smoke)
    if not smoke and (out / "metrics.csv").exists():
        print(f"[SKIP] dataset-origin run already finished ({out / 'metrics.csv'}) -- delete that folder to rerun it.")
        return
    setup_logging(out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    seed_everything(CFG["train"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_gpu(device)
    log.info("=== Dataset-origin 3-class | encoder=%s | harmonize=%s | device=%s ===", CFG["encoder"], CFG["harmonize"], device)
    json.dump({"run": "origin", "cfg": CFG, "time": time.ctime()}, open(out / "config_snapshot.json", "w"), indent=2, default=str)

    n = load_nlbsd()[["image_path", "case_id", "source"]].copy()
    n["case_id"] = "nlbsd_" + n["case_id"].astype(str); n["origin"] = 0
    d = load_ddsm("origin", 1); c = load_cmmd("origin", 2)
    parts = [n, d, c]
    if smoke:
        parts = [p.sample(min(30, len(p)), random_state=0) for p in parts]
    df = pd.concat(parts, ignore_index=True)
    df = make_split(df, out, "origin")
    tr, va, te = (df[df.split == s].reset_index(drop=True) for s in ("train", "val", "test"))
    write_csv([{"set": s, "source": src, "images": int((x.source == src).sum()), "cases": int(x[x.source == src].case_id.nunique())}
               for s, x in (("train", tr), ("val", va), ("test", te)) for src in ORIGIN_CLASSES], out / "data_counts.csv")

    t = CFG["train"]; bs = t["origin_batch_size"]
    q = {i: bs // 3 + (1 if i < bs % 3 else 0) for i in range(3)}
    img = [256, 160] if smoke else CFG["image_size"]
    href = (build_hist_reference(tr[tr.source == "nlbsd"], img, 30 if smoke else CFG["harmonize_ref_images"], out)
            if CFG["harmonize"] else None)
    if href is not None:
        save_harmonization_check(tr, img, href, out)
    loaders = make_loaders(tr, va, te, "origin", "long", q, bs, img, 0 if smoke else t["num_workers"], hist_ref=href)
    net = build_model(3).to(device)
    net = fit(net, loaders, device, out, True, ORIGIN_CLASSES, 2 if smoke else t["num_epochs"], smoke)

    tp, ty, tpa, tci = infer(net, loaders[2], device, True)
    mi = multiclass_metrics(ty, tp, ORIGIN_CLASSES); mi["level"] = "image"
    agg = {}
    for ci, yy, pp in zip(tci, ty, tp):
        a = agg.setdefault(ci, [int(yy), np.zeros(3), 0]); a[1] += pp; a[2] += 1
    mc = multiclass_metrics(np.array([v[0] for v in agg.values()]),
                            np.stack([v[1] / v[2] for v in agg.values()]), ORIGIN_CLASSES); mc["level"] = "case"
    write_csv([mi, mc], out / "metrics.csv")
    rows = []
    for a, ci, yy, pp in zip(tpa, tci, ty, tp):
        r = {"path": a, "case_id": ci, "label": int(yy)}; r.update({f"prob_{k}": float(pp[i]) for i, k in enumerate(ORIGIN_CLASSES)})
        rows.append(r)
    write_csv(rows, out / "per_sample_predictions.csv")
    for name, m in (("image", mi), ("case", mc)):
        cm = m["_cm"]; pct = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
        with open(out / f"confusion_matrix_{name}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["true\\pred"] + ORIGIN_CLASSES)
            for i, k in enumerate(ORIGIN_CLASSES):
                w.writerow([k] + [f"{v:.2f}" for v in pct[i]])
        plot_confusion(pct, out / f"confusion_matrix_{name}.png", f"Dataset origin ({name}-level), ImageNet encoder")
    log.info("[RESULT] origin image: balanced acc=%.4f macro AUC=%.4f F1=%s", mi["balanced_acc"], mi["macro_auc"],
             {k: round(mi[f"f1_{k}"], 3) for k in ORIGIN_CLASSES})
    log.info("STATUS: SUCCESS")


# =====================================================================================================
# comparison (DeLong + Holm), table, forest plot
# =====================================================================================================
def compare(smoke=False):
    import pandas as pd
    out = run_dir("comparison", smoke)
    setup_logging(out / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    nb = 200 if smoke else CFG["bootstrap_compare"]
    thr = json.load(open(run_dir(RUNS["baseline"]["run_name"], smoke) / "baseline_threshold.json"))
    preds = {}
    for k, s in RUNS.items():
        f = run_dir(s["run_name"], smoke) / "per_sample_predictions.csv"
        if not f.exists():
            raise FileNotFoundError(f"{f} missing -- train '{k}' first.")
        preds[k] = pd.read_csv(f)
    allrows = []
    for level in ("image", "case"):
        def prep(d):
            if level == "image":
                return d.assign(key=d["path"].map(path_key))[["key", "label", "prob"]]
            return d.groupby("case_id").agg(label=("label", "max"), prob=("prob", "max")).reset_index() \
                    .rename(columns={"case_id": "key"})
        base = prep(preds["baseline"])
        b_auc, b_lo, b_hi = boot_auc_ci(base.label, base.prob, nb)
        bm = binary_metrics(base.label, base.prob, thr[level], level, nb)
        rows = [{"config": DISPLAY["baseline"], "level": level, "n": len(base), "n_pos": int(base.label.sum()),
                 "auc": b_auc, "ci_lo": b_lo, "ci_hi": b_hi, "delta_auc": 0.0, "p_raw": None, "p_holm": None,
                 **{k: bm[k] for k in ("sensitivity", "specificity", "accuracy", "sens_ci_lo", "sens_ci_hi",
                                       "spec_ci_lo", "spec_ci_hi", "acc_ci_lo", "acc_ci_hi")}}]
        raw = []
        for k in ("cmmd", "ddsm", "ddsm_cmmd"):
            m = base.merge(prep(preds[k]), on="key", suffixes=("_b", "_c"))
            if len(m) != len(base):
                raise RuntimeError(f"{k}: only {len(m)}/{len(base)} test items match the baseline -- different test sets.")
            y = m.label_b.to_numpy()
            p_raw, _ = delong_test(y, m.prob_b.to_numpy(), m.prob_c.to_numpy())
            a, lo, hi = boot_auc_ci(y, m.prob_c, nb)
            cm = binary_metrics(y, m.prob_c.to_numpy(), thr[level], level, nb)
            raw.append(p_raw)
            rows.append({"config": DISPLAY[k], "level": level, "n": len(m), "n_pos": int(y.sum()), "auc": a,
                         "ci_lo": lo, "ci_hi": hi, "delta_auc": a - b_auc, "p_raw": p_raw,
                         **{kk: cm[kk] for kk in ("sensitivity", "specificity", "accuracy", "sens_ci_lo", "sens_ci_hi",
                                                  "spec_ci_lo", "spec_ci_hi", "acc_ci_lo", "acc_ci_hi")}})
        for r, ph in zip(rows[1:], holm(raw)):
            r["p_holm"] = float(ph); r["significant"] = bool(ph < CFG["alpha"]) if not np.isnan(ph) else None
        write_csv(rows, out / f"comparison_{level}.csv")
        forest(rows, out / f"forest_{level}.png", level)
        allrows += rows
    write_csv(allrows, out / "comparison_all.csv")
    log.info("\n%s", summary_text(allrows))
    (out / "SUMMARY.txt").write_text(summary_text(allrows))
    log.info("STATUS: SUCCESS")


def summary_text(rows):
    L = ["ImageNet EfficientNet-B5 (frozen linear probe) -- fixed NLBSD test set",
         f"external_filter={CFG['external_filter']}  normalization={CFG['normalization']}", ""]
    for lv in ("image", "case"):
        L.append(f"[{lv}-level]  {'config':<24}{'AUC':>7}  {'95% CI':<17}{'dAUC':>8}{'p_Holm':>9}{'Sen':>6}{'Spc':>6}{'Acc':>6}")
        for r in rows:
            if r["level"] != lv:
                continue
            ph = ("" if r["p_holm"] is None else "n/a" if np.isnan(r["p_holm"])
                  else f"{r['p_holm']:.3g}" if r["p_holm"] >= 0.001 else "<0.001")
            L.append(f"            {r['config']:<24}{r['auc']:>7.3f}  [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
                     f"{r['delta_auc']:>+8.3f}{ph:>9}{r['sensitivity']:>6.2f}{r['specificity']:>6.2f}{r['accuracy']:>6.2f}")
        L.append("")
    return "\n".join(L)


def forest(rows, path, level):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 3))
    ys = list(range(len(rows)))[::-1]
    for y, r in zip(ys, rows):
        ax.errorbar(r["auc"], y, xerr=[[r["auc"] - r["ci_lo"]], [r["ci_hi"] - r["auc"]]], fmt="o", color="k", capsize=3)
    ax.axvline(rows[0]["auc"], ls="--", color="gray")
    # "CMMD+" -> "CMMD$^{+}$" (Reviewer 1 minor comment: superscript on the configuration labels)
    ax.set_yticks(ys); ax.set_yticklabels([re.sub(r"(\w)\+", r"\1$^{+}$", r["config"]) for r in rows])
    ax.set_xlabel(f"{level}-level AUC (95% bootstrap CI)"); ax.set_title("ImageNet-pretrained encoder")
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def plot_confusion(pct, path, title):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    labs = ["NLBSD", "CBIS-DDSM", "CMMD"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct[i, j]:.1f}", ha="center", va="center", color="white" if pct[i, j] > 50 else "black")
    ax.set_xticks(range(3)); ax.set_xticklabels(labs); ax.set_yticks(range(3)); ax.set_yticklabels(labs)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


# =====================================================================================================
# prepare: build lists, print counts, save contact sheets (no GPU)
# =====================================================================================================
def prepare():
    out = run_dir("prepare", False)
    setup_logging(out / f"prepare_{time.strftime('%Y%m%d_%H%M%S')}.log")
    log.info("external_filter=%s | ddsm_full_mammograms_only=%s | normalization=%s",
             CFG["external_filter"], CFG["ddsm_full_mammograms_only"], CFG["normalization"])
    n = load_nlbsd(); n.to_csv(out / "nlbsd_list.csv", index=False)
    log.info("[NLBSD] example rows:\n%s", n.head(5).to_string())
    if CFG["original_split_csv"]:
        import pandas as pd
        sp = pd.read_csv(CFG["original_split_csv"]); sp["case_id"] = sp["case_id"].astype(str)
        hit = n.case_id.isin(set(sp.case_id)).mean()
        log.info("[SPLIT] %.1f%% of NLBSD images have a case_id in the original split.csv (must be 100%%)", hit * 100)
        if hit < 1:
            log.warning("[SPLIT] example original case_ids: %s | ours: %s", sp.case_id.head(5).tolist(), n.case_id.head(5).tolist())
    else:
        log.warning("[SPLIT] original_split_csv not set: the split will be regenerated with seed 42. It only matches "
                    "the paper's test set if the manifest rows and case_ids are identical to the original run.")
    lists = {"nlbsd": n}
    for name in ("cmmd", "ddsm"):
        try:
            lists[name] = load_external(name); lists[name].to_csv(out / f"{name}_list.csv", index=False)
            log.info("[%s] example rows:\n%s", name.upper(), lists[name].head(3).to_string())
        except Exception as e:
            log.error("[%s] FAILED: %s", name.upper(), e)
    contact_sheet(lists, out / "contact_sheet.png")
    log.info("Check contact_sheet.png: every row should look like the same preprocessing "
             "(breast cropped, background black, breast on the same side, full mammograms not ROI crops).")
    try:
        import efficientnet_pytorch, torch
        log.info("efficientnet_pytorch %s, torch %s OK", efficientnet_pytorch.__version__, torch.__version__)
        build_model(1, pretrained=True)
        log.info("ImageNet EfficientNet-B5 weights available (cached in ~/.cache/torch).")
    except Exception as e:
        log.error("Could not load ImageNet weights here (%s). Run `prepare` on a node with internet once so the "
                  "weights are cached, or copy efficientnet-b5-*.pth into ~/.cache/torch/hub/checkpoints/.", e)


def contact_sheet(lists, path, k=6):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from PIL import Image
    names = [x for x in ("nlbsd", "ddsm", "cmmd") if x in lists]
    fig, axes = plt.subplots(len(names), k, figsize=(k * 1.6, len(names) * 2.4))
    axes = np.atleast_2d(axes)
    for r, nm in enumerate(names):
        d = lists[nm]
        if "label" in d.columns and nm == "nlbsd":
            d = d[d.label == 1] if (d.label == 1).sum() >= k else d
        s = d.sample(min(k, len(d)), random_state=1)
        for c in range(k):
            ax = axes[r, c]; ax.axis("off")
            if c < len(s):
                try:
                    a = np.array(Image.open(s.iloc[c].image_path), dtype=np.float32)
                    a = a.mean(2) if a.ndim == 3 else a
                    ax.imshow(a, cmap="gray"); ax.set_title(Path(s.iloc[c].image_path).name[:18], fontsize=5)
                except Exception as e:
                    ax.set_title(f"read error", fontsize=5)
        axes[r, 0].text(-0.1, 0.5, nm.upper(), transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)



# =====================================================================================================
# t-SNE part
# =====================================================================================================
# settings
# =====================================================================================================
N_PER_DATASET = 300          # test images per dataset (300 x 3 = 900 points); lower if it is slow
VARIANTS = ["original", "harmonized"]   # "original" = Experiment 2 input, "harmonized" = Experiment 3 input
PERPLEXITY = 30
SEED = 42
BATCH = 8                    # images per encoder pass; 4 if "CUDA out of memory"
NUM_WORKERS = 4              # 0 if loading crashes on Windows

CFG["encoder"] = "mammoclip"
CFG["external_filter"] = "all"          # Experiments 2 and 3 used all CMMD and CBIS-DDSM cases
OUT = Path(CFG["output_root"]) / "tsne_features_independent"   # new folder: nothing from earlier runs is reused

NAMES = {"nlbsd": "NLBSD", "cmmd": "CMMD", "ddsm": "CBIS-DDSM"}
ORDER = ["nlbsd", "cmmd", "ddsm"]
STYLE = {"nlbsd": dict(c="#1f77b4", marker="o"),     # colour-blind-safe colours + different marker shapes
         "cmmd":  dict(c="#d95f02", marker="s"),
         "ddsm":  dict(c="#7570b3", marker="^")}
log = logging.getLogger("imagenet_control")


# =====================================================================================================
def load_all():
    """All three datasets, all cases (benign and malignant), with a NEW patient-level 70/10/20 split made here.
    Nothing is read from earlier runs."""
    import pandas as pd
    n = load_nlbsd()[["image_path", "case_id", "source"]].copy()
    n["case_id"] = "nlbsd_" + n["case_id"].astype(str); n["origin"] = 0
    d = load_ddsm("origin", 1); c = load_cmmd("origin", 2)
    df = pd.concat([n, d, c], ignore_index=True)
    df = make_split(df, OUT, "origin")          # split.csv is written to OUT (and reused only from OUT)
    cnt = df.groupby(["source", "split"]).case_id.nunique().unstack()
    log.info("Patients per dataset and split:\n%s", cnt)
    cnt.to_csv(OUT / "patients_per_split.csv")
    return df


def load_test_images(df):
    import pandas as pd
    te = df[df.split == "test"]
    parts = [g.sample(min(N_PER_DATASET, len(g)), random_state=SEED) for _, g in te.groupby("source")]
    out = pd.concat(parts, ignore_index=True)
    log.info("Test images used per dataset: %s", out.source.value_counts().to_dict())
    return out


def build_reference(df):
    """Histogram reference from NLBSD TRAINING images of this script's own split (same method as Experiment 3)."""
    f = OUT / "histogram_reference.json"
    if f.exists():
        log.info("Histogram reference loaded from %s", f)
        return np.asarray(json.load(open(f))["reference_quantiles"], np.float32)
    tr = df[(df.split == "train") & (df.source == "nlbsd")]
    log.info("Building histogram reference from %d NLBSD training images", min(CFG["harmonize_ref_images"], len(tr)))
    return build_hist_reference(tr, CFG["image_size"], CFG["harmonize_ref_images"], OUT)


def extract(df, hist_ref, device, net):
    import torch
    from torch.utils.data import DataLoader
    ds = MammoDataset(df, CFG["image_size"], augment=False, label_col="origin", label_dtype="long", hist_ref=hist_ref)
    dl = DataLoader(ds, batch_size=BATCH, num_workers=NUM_WORKERS, pin_memory=False)
    enc = net.encoder.eval(); feats = []
    with torch.no_grad():
        for i, (x, _, _, _) in enumerate(dl):
            f = enc._avg_pooling(enc.extract_features(x.to(device))).flatten(1)
            feats.append(f.float().cpu().numpy())
            if i % 20 == 0:
                log.info("  %d / %d images", min((i + 1) * BATCH, len(ds)), len(ds))
    return np.concatenate(feats)


def embed(X):
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    Z = StandardScaler().fit_transform(X)
    P50 = PCA(n_components=50, random_state=SEED).fit_transform(Z)
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    P2 = pca.transform(Z)
    try:
        ts = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", learning_rate="auto", max_iter=1000, random_state=SEED)
    except TypeError:            # older scikit-learn uses n_iter
        ts = TSNE(n_components=2, perplexity=PERPLEXITY, init="pca", learning_rate="auto", n_iter=1000, random_state=SEED)
    T2 = ts.fit_transform(P50)
    return T2, P2, pca.explained_variance_ratio_


def plot(coords, src, path, xlabel, ylabel, titles):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    k = len(coords)
    fig, axes = plt.subplots(1, k, figsize=(3.4 * k, 3.3), squeeze=False)
    for ax, C, t in zip(axes[0], coords, titles):
        for s in ORDER:
            m = src == s
            ax.scatter(C[m, 0], C[m, 1], s=7, alpha=0.7, linewidths=0, label=NAMES[s], **STYLE[s])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        if t:
            ax.set_title(t, fontsize=9)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, markerscale=2.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(path.with_suffix(".png"), dpi=600); fig.savefig(path.with_suffix(".pdf")); plt.close(fig)
    print("saved", path.with_suffix(".png"))


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    check_gpu(device)
    all_df = load_all()
    df = load_test_images(all_df)
    src = df.source.to_numpy()
    net = build_model(3).to(device)          # head is not used; only the frozen Mammo-CLIP encoder
    href = build_reference(all_df) if "harmonized" in VARIANTS else None

    from sklearn.metrics import silhouette_score
    T, P, titles, report = [], [], [], []
    for v in VARIANTS:
        f = OUT / f"features_{v}.npz"
        if f.exists():
            X = np.load(f)["X"]; log.info("[%s] features loaded from %s", v, f)
        else:
            log.info("[%s] extracting features ...", v)
            X = extract(df, href if v == "harmonized" else None, device, net)
            np.savez_compressed(f, X=X, source=src, path=df.image_path.to_numpy())
        T2, P2, evr = embed(X)
        T.append(T2); P.append(P2)
        titles.append("Original images" if v == "original" else "After histogram matching")
        lab = np.array([ORDER.index(s) for s in src])
        report.append(f"{v}: silhouette (2048-d standardized features) = "
                      f"{silhouette_score(StandardScaler_(X), lab):.3f} | silhouette (t-SNE 2-D) = "
                      f"{silhouette_score(T2, lab):.3f} | PCA variance PC1+PC2 = {evr.sum():.3f}")
        import pandas as pd
        pd.DataFrame({"path": df.image_path, "source": src, "tsne_1": T2[:, 0], "tsne_2": T2[:, 1],
                      "pc_1": P2[:, 0], "pc_2": P2[:, 1]}).to_csv(OUT / f"coords_{v}.csv", index=False)

    one = len(VARIANTS) == 1
    plot(T, src, OUT / "Figure_tsne", "t-SNE 1", "t-SNE 2", [None] if one else titles)
    plot(P, src, OUT / "Figure_pca", "PC 1", "PC 2", [None] if one else titles)
    (OUT / "SUMMARY.txt").write_text(
        f"Mammo-CLIP frozen encoder features, test images only, {N_PER_DATASET} per dataset "
        f"(actual: {dict(zip(*np.unique(src, return_counts=True)))})\n"
        f"t-SNE: standardize -> PCA 50 -> t-SNE (perplexity {PERPLEXITY}, seed {SEED})\n" + "\n".join(report) + "\n")
    print((OUT / "SUMMARY.txt").read_text())


def StandardScaler_(X):
    from sklearn.preprocessing import StandardScaler
    return StandardScaler().fit_transform(X)


if __name__ == "__main__":
    main()
