# -*- coding: utf-8 -*-
"""Logging, seeding, and small IO helpers shared by every entry script."""
import os, sys, csv, json, time, random, logging, subprocess
from pathlib import Path
import numpy as np

log = logging.getLogger("mammo_pipeline")


def setup_logging(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger("mammo_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    class _Flush(logging.Handler):
        def emit(self, record):
            fh.flush(); sh.flush()

    logger.addHandler(_Flush())
    return logger


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def atomic_torch_save(obj, path: Path):
    import torch
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_repro_snapshot(out_dir: Path, cfg: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    snap = {"config": cfg, "git_hash": git_hash(), "time": time.ctime()}
    with open(out_dir / "repro_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2, default=str)


def write_csv(rows, path):
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k.startswith("_") or k in keys:
                continue
            keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
