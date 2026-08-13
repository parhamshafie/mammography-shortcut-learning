# -*- coding: utf-8 -*-
"""Binary/multiclass metrics, DeLong's test, Holm-Bonferroni correction, bootstrap CIs. No torch
dependency -- this module works from CSVs alone, which is what run_compare_delong_holm.py relies on."""
import numpy as np


# ---------------------------------------------------------------------------
# Binary classification metrics
# ---------------------------------------------------------------------------
def binary_ece(y, p, n_bins=15):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf >= bins[i]) & (conf < bins[i + 1] if i < n_bins - 1 else conf <= bins[i + 1])
        if m.sum():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def youden_threshold(y, p):
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr - fpr))])


def compute_metrics(y, p, bootstrap_n, level, threshold=None):
    """threshold=None picks Youden's J on (y, p) itself. Pass a fixed value to reuse a threshold
    selected elsewhere (Sec. 2.7: one threshold, picked on the baseline model, applied unchanged to
    every other configuration)."""
    from sklearn.metrics import (roc_auc_score, average_precision_score, confusion_matrix,
                                 f1_score, brier_score_loss)
    y = np.asarray(y); p = np.asarray(p)
    m = {"level": level, "n": len(y), "n_pos": int((y == 1).sum()), "n_neg": int((y == 0).sum())}
    try:
        m["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        m["roc_auc"] = float("nan")
    try:
        m["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        m["pr_auc"] = float("nan")
    if threshold is not None:
        thr = threshold
    else:
        thr = youden_threshold(y, p) if len(np.unique(y)) > 1 else 0.5
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m.update({"threshold": thr, "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
              "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
              "npv": tn / max(tn + fn, 1), "f1": float(f1_score(y, pred, zero_division=0)),
              "ece": binary_ece(y, p)})
    try:
        m["brier"] = float(brier_score_loss(y, p))
    except Exception:
        m["brier"] = float("nan")
    if bootstrap_n and len(np.unique(y)) > 1:
        _, ci_lo, ci_hi = stratified_bootstrap_auc_ci(y, p, bootstrap_n, seed=42, confidence=0.95)
        if not (isinstance(ci_lo, float) and np.isnan(ci_lo)):
            m["roc_auc_ci_lo"] = ci_lo
            m["roc_auc_ci_hi"] = ci_hi
    return m


def aggregate_case(paths, case_ids, y, p, rule="max"):
    d = {}
    for cid, yy, pp in zip(case_ids, y, p):
        if cid not in d:
            d[cid] = {"y": 0, "p": 0.0}
        d[cid]["y"] = max(d[cid]["y"], int(yy))
        d[cid]["p"] = max(d[cid]["p"], float(pp))
    cids = list(d.keys())
    return np.array([d[c]["y"] for c in cids]), np.array([d[c]["p"] for c in cids]), cids


def stratified_bootstrap_auc_ci(y, p, n_boot, seed, confidence):
    """Resamples positives and negatives separately -- the CI convention used throughout the paper."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y); p = np.asarray(p)
    point = float(roc_auc_score(y, p))
    pos_idx = np.where(y == 1)[0]; neg_idx = np.where(y == 0)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return point, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        try:
            aucs.append(roc_auc_score(y[idx], p[idx]))
        except Exception:
            continue
    if not aucs:
        return point, float("nan"), float("nan")
    lo, hi = (1 - confidence) / 2 * 100, (1 - (1 - confidence) / 2) * 100
    return point, float(np.percentile(aucs, lo)), float(np.percentile(aucs, hi))


# ---------------------------------------------------------------------------
# Multiclass (dataset-origin) metrics
# ---------------------------------------------------------------------------
def compute_multiclass_metrics(y_true, probs, class_names):
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score, confusion_matrix
    from sklearn.preprocessing import label_binarize
    y_true = np.asarray(y_true); probs = np.asarray(probs)
    y_pred = probs.argmax(axis=1)
    n = len(class_names)
    m = {"n": len(y_true), "balanced_acc": float(balanced_accuracy_score(y_true, y_pred))}
    f1s = f1_score(y_true, y_pred, labels=list(range(n)), average=None, zero_division=0)
    for i, cname in enumerate(class_names):
        m[f"f1_{cname}"] = float(f1s[i])
    try:
        m["macro_auc"] = float(roc_auc_score(y_true, probs, multi_class="ovr", average="macro",
                                             labels=list(range(n))))
    except Exception:
        m["macro_auc"] = float("nan")
    yb = label_binarize(y_true, classes=list(range(n)))
    for i, cname in enumerate(class_names):
        try:
            m[f"auc_{cname}"] = (float(roc_auc_score(yb[:, i], probs[:, i]))
                                 if 0 < yb[:, i].sum() < len(yb) else float("nan"))
        except Exception:
            m[f"auc_{cname}"] = float("nan")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    m["_confusion_matrix"] = cm.tolist()
    row_sum = cm.sum(axis=1, keepdims=True)
    m["_confusion_matrix_pct"] = (np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float),
                                            where=row_sum != 0) * 100).tolist()
    return m


def aggregate_case_probs(paths, case_ids, y, probs, n_classes):
    """Case-level aggregation: mean softmax prob across a case's views."""
    d = {}
    for cid, yy, pp in zip(case_ids, y, probs):
        if cid not in d:
            d[cid] = {"y": int(yy), "p_sum": np.zeros(n_classes, dtype=np.float64), "n": 0}
        d[cid]["p_sum"] += pp
        d[cid]["n"] += 1
    cids = list(d.keys())
    y_out = np.array([d[c]["y"] for c in cids])
    p_out = np.stack([d[c]["p_sum"] / max(d[c]["n"], 1) for c in cids])
    return y_out, p_out, cids


# ---------------------------------------------------------------------------
# DeLong's test for two correlated ROC curves
# (Sun & Xu 2014, the standard O(n log n) reformulation of DeLong et al. 1988)
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted_T, m):
    n = preds_sorted_T.shape[1] - m
    pos = preds_sorted_T[:, :m]
    neg = preds_sorted_T[:, m:]
    k = preds_sorted_T.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted_T[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    if k == 1:
        sx = np.array([[sx]]); sy = np.array([[sy]])
    return aucs, sx / m + sy / n


def delong_roc_test(y_true, probs_a, probs_b):
    """Two-sided p-value + (auc_a, auc_b) for two correlated ROC curves on the same samples."""
    from scipy.stats import norm
    y_true = np.asarray(y_true).astype(int)
    order = (-y_true).argsort()
    m = int(y_true.sum())
    n = len(y_true) - m
    if m == 0 or n == 0:
        return float("nan"), (float("nan"), float("nan"))
    preds_sorted_T = np.vstack([np.asarray(probs_a), np.asarray(probs_b)])[:, order]
    aucs, cov = _fast_delong(preds_sorted_T, m)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return float("nan"), (float(aucs[0]), float(aucs[1]))
    z = diff / np.sqrt(var)
    return float(2 * (1 - norm.cdf(abs(z)))), (float(aucs[0]), float(aucs[1]))


def holm_bonferroni(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = pvalues[order]
    adjusted_sorted = np.empty(m)
    running_max = 0.0
    for i, p in enumerate(sorted_p):
        running_max = max(running_max, min((m - i) * p, 1.0))
        adjusted_sorted[i] = running_max
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted
