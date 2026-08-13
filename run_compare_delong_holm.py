# -*- coding: utf-8 -*-
"""Pairs the baseline's and each mixed config's predictions on the shared NLBSD test set, runs
DeLong's test + Holm-Bonferroni correction, and draws the Fig. 2 forest plot (Sec. 2.7). No GPU or
dataset needed -- just reads the per_sample_predictions.csv each run_*.py script already wrote.
Run this after run_baseline.py and the three run_nlbsd_*.py scripts have finished.

    python run_compare_delong_holm.py
"""
from mammo_pipeline.compare import run

# ---- edit this ----
OUTPUT_ROOT = "/path/to/outputs"

CONFIG = {
    "run_name": "delong_comparison",
    "output_root": OUTPUT_ROOT,
    "baseline": {"name": "NLBSD", "per_sample_csv": f"{OUTPUT_ROOT}/nlbsd_baseline/per_sample_predictions.csv"},
    "configs": [
        {"name": "NLBSD + CMMD+", "per_sample_csv": f"{OUTPUT_ROOT}/nlbsd_plus_cmmd/per_sample_predictions.csv"},
        {"name": "NLBSD + DDSM+", "per_sample_csv": f"{OUTPUT_ROOT}/nlbsd_plus_ddsm/per_sample_predictions.csv"},
        {"name": "NLBSD + DDSM+ + CMMD+", "per_sample_csv": f"{OUTPUT_ROOT}/nlbsd_plus_ddsm_cmmd/per_sample_predictions.csv"},
    ],
    "levels": ["image", "case"],
    "aggregation_rule": "max",
    "bootstrap": {"n": 2000, "seed": 42, "confidence": 0.95},
    "alpha": 0.05,
    "min_match_frac": 0.90,
}

if __name__ == "__main__":
    run(CONFIG)
