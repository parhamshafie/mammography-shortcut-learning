# -*- coding: utf-8 -*-
"""
Rank-based error analysis (JMI revision, Reviewer 2 comment 10).

Compares the predictions of two models on the same NLBSD test images:
  base  = model trained on NLBSD alone
  mixed = model trained on NLBSD + CBIS-DDSM+ + CMMD+  (ImageNet-initialized encoder)

Each image is ranked among all test images by predicted cancer probability and the rank is expressed
as a percentile (0 = least suspicious, 100 = most suspicious). This avoids depending on a single
decision threshold. Changes in the percentile ranks of the cancer images are tested with the
Wilcoxon signed-rank test, and the most-changed images are listed for the example figure.

Input: one CSV with one row per NLBSD test image and the columns
  key, label (1 = cancer, 0 = non-cancer), prob_base, prob_mix, group (abnormal / normal / false_positive), path
Usage:
  python error_analysis.py predictions.csv --out error_analysis_out
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon


def percentile_rank(p):
    """Percentile rank of each score among all scores (average rank for ties), in [0, 100]."""
    r = rankdata(p, method="average")
    return 100.0 * (r - 1) / (len(p) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="error_analysis_out")
    ap.add_argument("--n_examples", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_csv(a.csv)
    df["pct_base"] = percentile_rank(df["prob_base"].values)
    df["pct_mix"] = percentile_rank(df["prob_mix"].values)
    df["change"] = df["pct_mix"] - df["pct_base"]

    can = df[df["label"] == 1]
    non = df[df["label"] == 0]
    n_down = int((can["change"] < 0).sum())
    stat, p = wilcoxon(can["pct_base"], can["pct_mix"])
    top_cut_base = np.percentile(df["prob_base"], 90)
    top_cut_mix = np.percentile(df["prob_mix"], 90)
    top_base = int((can["prob_base"] >= top_cut_base).sum())
    top_mix = int((can["prob_mix"] >= top_cut_mix).sum())

    lines = [
        f"Test images: {len(df)} ({len(can)} cancer, {len(non)} non-cancer)",
        f"Cancer images ranked less suspicious after adding external data: {n_down}/{len(can)} "
        f"({100 * n_down / len(can):.0f}%)",
        f"Median percentile rank of cancer images: {can['pct_base'].median():.1f} -> {can['pct_mix'].median():.1f} "
        f"(Wilcoxon signed-rank p = {p:.4f})",
        f"Cancer images among the 10% most suspicious test images: {top_base} -> {top_mix}",
        f"Mean change in percentile rank of non-cancer images: {non['change'].mean():+.2f}",
    ]
    for g, sub in non.groupby("group") if "group" in non else []:
        lines.append(f"  non-cancer group '{g}': mean change {sub['change'].mean():+.2f} (n = {len(sub)})")
    print("\n".join(lines))
    with open(os.path.join(a.out, "summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    drop = can.nsmallest(a.n_examples, "change").assign(kind="cancer_dropped")
    rise = non.nlargest(a.n_examples, "change").assign(kind="noncancer_increased")
    pd.concat([drop, rise]).to_csv(os.path.join(a.out, "error_examples.csv"), index=False)
    df.to_csv(os.path.join(a.out, "error_analysis_all_images.csv"), index=False)


if __name__ == "__main__":
    main()
