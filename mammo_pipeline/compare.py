# -*- coding: utf-8 -*-
"""Pairs each mixed-config run's predictions against the baseline's on the shared NLBSD test images
and runs the DeLong + Holm-Bonferroni comparison (Sec. 2.7, Table 2, Fig. 2). Pure CSV + stats, no
torch or dataset access needed."""
import time, logging
from pathlib import Path

from . import stats as S
from . import plotting as P
from .utils import setup_logging, write_csv

log = logging.getLogger("mammo_pipeline")


def load_test_predictions(csv_path, level, agg_rule):
    import pandas as pd
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing predictions file: {csv_path}. Run that config's script first.")
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"]
    if "domain" in df.columns:
        df = df[df["domain"] == "nlbsd"]
    df = df[["path", "case_id", "label", "prob"]].drop_duplicates(subset=["path"]).reset_index(drop=True)
    if level == "image":
        return df.rename(columns={"path": "key"})
    if agg_rule != "max":
        raise ValueError(f"Unsupported aggregation_rule '{agg_rule}' (only 'max' is implemented).")
    g = df.groupby("case_id").agg(label=("label", "max"), prob=("prob", "max")).reset_index()
    return g.rename(columns={"case_id": "key"})


def pair_on_key(base_df, cfg_df, min_match_frac):
    merged = base_df.merge(cfg_df, on="key", suffixes=("_base", "_cfg"), how="inner")
    if len(merged) == 0:
        raise RuntimeError("No overlapping images between baseline and config -- check both runs used the same NLBSD manifest.")
    frac = len(merged) / max(len(base_df), 1)
    if frac < min_match_frac:
        raise RuntimeError(f"Only {frac:.1%} of baseline test images matched (threshold={min_match_frac:.0%}). "
                           f"The two runs likely used different NLBSD manifests/folds -- aborting.")
    merged["label"] = merged["label_base"]
    return merged


def run_level(level, cfg, out_dir):
    log.info("=== LEVEL: %s ===", level)
    boot = cfg["bootstrap"]
    base_df = load_test_predictions(cfg["baseline"]["per_sample_csv"], level, cfg["aggregation_rule"])
    base_auc, base_lo, base_hi = S.stratified_bootstrap_auc_ci(base_df["label"], base_df["prob"],
                                                                boot["n"], boot["seed"], boot["confidence"])
    rows = [{"name": cfg["baseline"]["name"], "level": level, "n": len(base_df),
            "n_pos": int((base_df["label"] == 1).sum()), "auc": base_auc, "ci_lo": base_lo, "ci_hi": base_hi,
            "delta_auc_vs_baseline": 0.0, "delong_p_raw": None, "p_holm": None, "significant": None}]
    raw_pvals, pending = [], []
    for c in cfg["configs"]:
        cfg_df = load_test_predictions(c["per_sample_csv"], level, cfg["aggregation_rule"])
        merged = pair_on_key(base_df, cfg_df, cfg["min_match_frac"])
        y = merged["label"].to_numpy()
        p_raw, (auc_a, auc_b) = S.delong_roc_test(y, merged["prob_base"].to_numpy(), merged["prob_cfg"].to_numpy())
        cfg_auc, cfg_lo, cfg_hi = S.stratified_bootstrap_auc_ci(y, merged["prob_cfg"].to_numpy(),
                                                                 boot["n"], boot["seed"], boot["confidence"])
        log.info("[%s:%s] n=%d auc=%.4f [%.4f, %.4f] raw_p=%.4g", level, c["name"], len(merged), cfg_auc, cfg_lo, cfg_hi, p_raw)
        raw_pvals.append(p_raw)
        pending.append({"name": c["name"], "level": level, "n": len(merged), "n_pos": int((y == 1).sum()),
                        "auc": cfg_auc, "ci_lo": cfg_lo, "ci_hi": cfg_hi,
                        "delta_auc_vs_baseline": cfg_auc - base_auc, "delong_p_raw": p_raw})
    for r, ph in zip(pending, S.holm_bonferroni(raw_pvals) if raw_pvals else []):
        r["p_holm"] = float(ph); r["significant"] = bool(ph < cfg["alpha"])
        rows.append(r)
    write_csv(rows, out_dir / f"comparison_table_{level}.csv")
    P.forest_plot(rows, out_dir / f"forest_plot_{level}.pdf", level, cfg["alpha"])
    return rows


def run(cfg):
    out_dir = Path(cfg["output_root"]) / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(out_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log")
    all_rows = []
    for level in cfg["levels"]:
        all_rows += run_level(level, cfg, out_dir)
    write_csv(all_rows, out_dir / "comparison_table_all_levels.csv")
    log.info("STATUS: SUCCESS")
