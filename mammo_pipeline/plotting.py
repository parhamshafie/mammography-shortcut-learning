# -*- coding: utf-8 -*-
"""ROC/PR curves for the binary task, the confusion matrix for the origin task, and the forest
plot for the cross-model comparison."""
import logging
import numpy as np
from .stats import compute_multiclass_metrics

log = logging.getLogger("mammo_pipeline")


def make_plots(per_sample_csv, out_dir):
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(per_sample_csv)
    domains = sorted(df["domain"].unique()) if "domain" in df.columns else [None]
    for dom in domains:
        sub = df if dom is None else df[df["domain"] == dom]
        if sub["label"].nunique() < 2:
            continue
        suffix = "" if dom is None else f"_{dom}"
        y = sub["label"].to_numpy(); p = sub["prob"].to_numpy()

        fpr, tpr, _ = roc_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, lw=2); ax.plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
        ax.set(xlabel="1 - Specificity", ylabel="Sensitivity", title=f"ROC (image){suffix}")
        fig.savefig(out_dir / f"roc{suffix}.pdf", bbox_inches="tight"); plt.close(fig)

        pre, rec, _ = precision_recall_curve(y, p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.step(rec, pre, lw=2, where="post")
        ax.set(xlabel="Recall", ylabel="Precision", title=f"PR (image){suffix}")
        fig.savefig(out_dir / f"pr{suffix}.pdf", bbox_inches="tight"); plt.close(fig)
    log.info("[PLOT] wrote roc/pr -> %s", out_dir)


def plot_confusion_matrix(cm_pct, class_labels, out_path, title="Normalized confusion matrix"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm_pct = np.asarray(cm_pct)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(class_labels))); ax.set_xticklabels(class_labels)
    ax.set_yticks(range(len(class_labels))); ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted dataset"); ax.set_ylabel("True dataset"); ax.set_title(title)
    for i in range(cm_pct.shape[0]):
        for j in range(cm_pct.shape[1]):
            val = cm_pct[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    color="white" if val > 50 else "black", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Percentage (%)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def make_plots_origin(per_sample_csv, out_dir, class_names):
    import pandas as pd
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(per_sample_csv)
    y = df["label"].to_numpy()
    probs = df[[f"prob_{c}" for c in class_names]].to_numpy()
    m = compute_multiclass_metrics(y, probs, class_names)
    plot_confusion_matrix(m["_confusion_matrix_pct"], ["NLBSD", "CBIS-DDSM", "CMMD"],
                          out_dir / "confusion_matrix.pdf",
                          title=f"Dataset-origin confusion matrix (macro AUC={m['macro_auc']:.4f})")
    log.info("[PLOT] confusion_matrix.pdf -> %s | balanced_acc=%.4f macro_auc=%.4f",
             out_dir, m["balanced_acc"], m["macro_auc"])


def forest_plot(rows, out_path, level, alpha):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 1.1 * len(rows) + 1))
    ys = list(range(len(rows), 0, -1))
    baseline_auc = rows[0]["auc"]
    for y, r in zip(ys, rows):
        ax.errorbar(r["auc"], y, xerr=[[r["auc"] - r["ci_lo"]], [r["ci_hi"] - r["auc"]]],
                   fmt="o", capsize=4, lw=2, markersize=7,
                   color="tab:blue" if r is rows[0] else "tab:orange")
        label = f"{r['auc']:.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        if r.get("p_holm") is not None:
            sig = "significant" if r["p_holm"] < alpha else "n.s."
            label += f"; Holm p={r['p_holm']:.3g} ({sig})"
        ax.text(1.02, y, label, transform=ax.get_yaxis_transform(), va="center", fontsize=9)
    ax.axvline(baseline_auc, color="grey", linestyle="--", lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([r["name"] for r in rows])
    ax.set_xlabel("AUC-ROC with 95% bootstrap CI")
    ax.set_xlim(0.45, 1.0)
    ax.set_title(f"AUC comparison vs. baseline ({level}-level, DeLong test, Holm-corrected)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
