# -*- coding: utf-8 -*-
"""Manifest loading, external-dataset loading, patient-level splitting, the image Dataset, and the
balanced batch sampler. Shared by both the binary (normal/suspicious) and 3-class (dataset-origin)
pipelines.

NLBSD is the only dataset with a real manifest (a CSV with a clinical label column). CBIS-DDSM and
CMMD are abnormal-enriched with no normal exams, so there's no label to look up for them -- every
image found under their image_root is just assigned whatever label the caller asks for.
"""
import os, random, logging
from pathlib import Path
import numpy as np

log = logging.getLogger("mammo_pipeline")


def load_manifest(cfg):
    import pandas as pd
    cols = cfg["columns"]
    mc = cfg.get("manifest", {})
    df = pd.read_csv(cfg["paths"]["manifest_csv"])
    ip, lb, ci = cols["image_path"], cols["label"], cols["case_id"]
    if ip not in df.columns:
        raise KeyError(f"Manifest missing image-path column '{ip}'. Has: {list(df.columns)}")
    if lb not in df.columns:
        raise KeyError(f"Manifest missing label column '{lb}'. Has: {list(df.columns)}")
    df = df.rename(columns={ip: "image_path", lb: "label"})
    df["label"] = df["label"].astype(int)
    df["image_path"] = df["image_path"].astype(str)
    if mc.get("normalize_backslashes", True):
        df["image_path"] = df["image_path"].str.replace("\\", "/", regex=False)
    if ci in df.columns:
        df = df.rename(columns={ci: "case_id"})
    elif mc.get("derive_case_id_from_path", False):
        idx = mc.get("case_id_path_index", 1)
        df["case_id"] = df["image_path"].apply(lambda p: p.split("/")[idx])
    else:
        raise KeyError(f"No case_id column '{ci}' and derive_case_id_from_path is off.")
    df["case_id"] = df["case_id"].astype(str)
    if mc.get("swap_dcm_to_png", True):
        df["image_path"] = df["image_path"].str.replace(r"(?i)\.dcm$", ".png", regex=True)
    if mc.get("lefted_suffix_for_right", False):
        lat_col = mc.get("laterality_col", "Image Laterality")
        suffix = mc.get("lefted_suffix", "_lefted")
        rv = str(mc.get("right_value", "R")).upper()
        if lat_col in df.columns:
            is_right = df[lat_col].astype(str).str.upper() == rv
            df.loc[is_right, "image_path"] = df.loc[is_right, "image_path"].str.replace(
                r"(?i)\.png$", f"{suffix}.png", regex=True)
            log.info("Applied '%s' suffix to %d right-side images.", suffix, int(is_right.sum()))
    root = cfg["paths"].get("image_root", "") or ""
    if root:
        df["image_path"] = df["image_path"].apply(lambda p: p if os.path.isabs(p) else os.path.join(root, p))
    exists = df["image_path"].apply(lambda p: Path(str(p)).exists())
    n_missing = int((~exists).sum())
    if n_missing:
        for p in df.loc[~exists, "image_path"].head(20):
            log.warning("[SKIP] missing file: %s", p)
        log.warning("[SKIP] %d of %d files missing.", n_missing, len(df))
        if n_missing / max(len(df), 1) > 0.10:
            raise RuntimeError(f"{n_missing / len(df):.0%} of manifest files are missing -- "
                               f"check paths.image_root and the manifest settings before continuing.")
    df = df.loc[exists].reset_index(drop=True)
    log.info("Manifest: %d images | %d cases | pos=%d neg=%d", len(df), df["case_id"].nunique(),
             int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
    return df


def load_external_flat_dataset(ext_cfg, label_col="label", label_value=1):
    """Globs every image under ext_cfg['image_root'] and assigns label_col=label_value to all of
    them -- CBIS-DDSM/CMMD have no manifest, so this is the entire loading step for them."""
    import pandas as pd
    name = ext_cfg["name"]
    root = Path(ext_cfg["image_root"])
    pattern = ext_cfg.get("glob_pattern", "**/*.png")
    if not root.exists():
        raise FileNotFoundError(f"[{name}] image_root does not exist: {root}")
    paths = sorted(root.glob(pattern))
    if not paths:
        raise RuntimeError(f"[{name}] no images found under {root} (pattern='{pattern}').")
    idx = ext_cfg.get("case_id_path_index", 0)
    rows = []
    for p in paths:
        parts = p.relative_to(root).parts
        raw_cid = parts[idx] if idx < len(parts) else p.stem
        rows.append({"image_path": str(p), label_col: label_value,
                     "case_id": f"{name}_{raw_cid}", "source": name})
    df = pd.DataFrame(rows)
    log.info("[%s] %d images | %d cases | %s=%s", name, len(df), df["case_id"].nunique(),
             label_col, label_value)
    return df


def split_external_holdout(df, frac, seed):
    """Patient-level split into (train_pool, holdout); holdout is reserved for a mixed-domain
    evaluation and never touches training."""
    if df.empty or frac <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    cases = np.array(df["case_id"].unique())
    rng.shuffle(cases)
    holdout_cases = set(cases[:int(round(len(cases) * frac))])
    holdout = df[df["case_id"].isin(holdout_cases)].reset_index(drop=True)
    train_pool = df[~df["case_id"].isin(holdout_cases)].reset_index(drop=True)
    return train_pool, holdout


def make_single_split(df, cfg, out_dir, label_col="label"):
    """Patient-level train/val/test split (70/10/20 by default, Sec. 2.4) -- a single split, not
    cross-validation. Pinned to a CSV so re-runs reuse the same assignment."""
    import pandas as pd
    split_file = out_dir / "split.csv"
    if split_file.exists():
        df = df.merge(pd.read_csv(split_file), on="case_id", how="left")
        log.info("Loaded pinned split from %s", split_file)
        return df
    sc = cfg.get("split", {})
    test_frac = sc.get("test_frac", 0.2)
    val_frac = sc.get("val_frac", 0.1)
    seed = sc.get("seed", 42)
    test_df, remain_df = stratified_case_split(df, test_frac, seed, label_col)
    val_frac_of_remain = val_frac / max(1.0 - test_frac, 1e-9)
    val_df, train_df = stratified_case_split(remain_df, val_frac_of_remain, seed + 1, label_col)

    split_map = {}
    split_map.update({cid: "train" for cid in train_df["case_id"].unique()})
    split_map.update({cid: "val" for cid in val_df["case_id"].unique()})
    split_map.update({cid: "test" for cid in test_df["case_id"].unique()})
    df = df.copy()
    df["split"] = df["case_id"].map(split_map)
    df[["case_id", "split"]].drop_duplicates().to_csv(split_file, index=False)
    log.info("Wrote pinned split -> %s (train=%d val=%d test=%d cases)", split_file,
             train_df["case_id"].nunique(), val_df["case_id"].nunique(), test_df["case_id"].nunique())
    return df


def stratified_case_split(df, frac, seed, label_col="label"):
    """Splits cases into (A, B) by frac, stratified on case-level label. Works for any number of
    classes: iterates whatever unique values are in label_col."""
    rng = np.random.default_rng(seed)
    case_lab = df.groupby("case_id")[label_col].max()
    a_cases = []
    for lab in sorted(case_lab.unique()):
        cs = np.array(case_lab[case_lab == lab].index)
        rng.shuffle(cs)
        a_cases.extend(cs[:int(round(len(cs) * frac))])
    a_set = set(a_cases)
    a = df[df["case_id"].isin(a_set)].reset_index(drop=True)
    b = df[~df["case_id"].isin(a_set)].reset_index(drop=True)
    return a, b


def smoke_subset(df, cfg, seed=0):
    s = cfg["smoke_test"]
    pos = df[df["label"] == 1]
    neg = df[df["label"] == 0]
    pos = pos.sample(min(s["train_pos"] + s["val_pos"] + s["test_pos"], len(pos)), random_state=seed)
    neg = neg.sample(min(s["train_neg"] + s["val_neg"] + s["test_neg"], len(neg)), random_state=seed)
    import pandas as pd
    out = pd.concat([pos, neg]).reset_index(drop=True)
    log.info("[SMOKE] subset: %d images (pos=%d neg=%d)", len(out), len(pos), len(neg))
    return out


class MammoDataset:
    """label_dtype: 'float' for the binary sigmoid head, 'long' for the multiclass softmax head."""

    def __init__(self, df, image_size, mean, std, augment, aug_cfg, label_col="label", label_dtype="float"):
        self.df = df.reset_index(drop=True)
        self.H, self.W = image_size
        self.mean = np.array(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.array(std, dtype=np.float32).reshape(3, 1, 1)
        self.augment = augment
        self.aug_cfg = aug_cfg
        self.label_col = label_col
        self.label_dtype = label_dtype

    def __len__(self):
        return len(self.df)

    def _read(self, path):
        from PIL import Image
        try:
            arr = np.array(Image.open(path), dtype=np.float32)
        except Exception as e:
            log.warning("[SKIP-read] %s (%s) -> zero image", path, e)
            return np.zeros((self.H, self.W), dtype=np.float32)
        return arr.mean(axis=2) if arr.ndim == 3 else arr

    def _augment(self, arr):
        import cv2
        a = self.aug_cfg
        h, w = arr.shape
        angle = random.uniform(-a["rotation_deg"], a["rotation_deg"])
        scale = random.uniform(a["scale_min"], a["scale_max"])
        tx = random.uniform(-a["translate_frac"], a["translate_frac"]) * w
        ty = random.uniform(-a["translate_frac"], a["translate_frac"]) * h
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[0, 2] += tx; M[1, 2] += ty
        arr = cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        arr = np.clip(arr + random.uniform(-a["brightness"], a["brightness"]), 0, 1)
        return np.clip(arr * random.uniform(a["contrast_low"], a["contrast_high"]), 0, 1)

    def __getitem__(self, idx):
        import torch, cv2
        row = self.df.iloc[idx]
        arr = self._read(row["image_path"])
        if arr.shape != (self.H, self.W):
            arr = cv2.resize(arr, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        arr = arr - arr.min()
        mx = arr.max()
        if mx > 0:
            arr = arr / mx
        if self.augment:
            arr = self._augment(arr)
        t = np.repeat(arr[None, :, :], 3, axis=0).astype(np.float32)
        t = (t - self.mean) / self.std
        label = (torch.tensor(float(row[self.label_col])) if self.label_dtype == "float"
                 else torch.tensor(int(row[self.label_col]), dtype=torch.long))
        return torch.from_numpy(t), label, str(row["image_path"]), str(row["case_id"])


class BalancedBatchSampler:
    """class_quotas: {class_value: count_per_batch}. Falls back to whatever's available if a class
    pool runs out (keeps smoke tests / tiny folds from crashing)."""

    def __init__(self, labels, batch_size, class_quotas, seed=0):
        self.labels = np.asarray(labels)
        self.bs = batch_size
        self.quotas = class_quotas
        self.idx_by_class = {c: np.where(self.labels == c)[0] for c in class_quotas}
        self.rng = np.random.default_rng(seed)
        self.n_batches = max(1, len(self.labels) // batch_size)

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for c, q in self.quotas.items():
                pool = self.idx_by_class[c]
                if len(pool) == 0:
                    continue
                batch.extend(self.rng.choice(pool, q, replace=len(pool) < q).tolist())
            if not batch:
                continue
            self.rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches


def even_quotas(batch_size, n_classes):
    """[base, base, ...] with the remainder spread over the first few classes, summing to batch_size."""
    base = batch_size // n_classes
    rem = batch_size - base * n_classes
    return [base + (1 if i < rem else 0) for i in range(n_classes)]
