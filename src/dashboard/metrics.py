"""Evaluation metrics for the dashboard: confusion matrix, ROC/AUC,
classification report and permutation feature importance.

Everything is computed from scratch (no scikit-learn) so the dashboard
stays dependency-light: NumPy only.
"""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return {'tn','fp','fn','tp'} counts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return {
        "tn": int(np.sum((y_true == 0) & (y_pred == 0))),
        "fp": int(np.sum((y_true == 0) & (y_pred == 1))),
        "fn": int(np.sum((y_true == 1) & (y_pred == 0))),
        "tp": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall and F1, rounded to 4 decimals."""
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": round((tp + tn) / total, 4) if total else 0.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "confusion": cm,
    }


def roc_curve(y_true: np.ndarray, probs: np.ndarray, n_points: int = 25) -> dict:
    """ROC points at evenly spaced thresholds plus AUC (trapezoid rule)."""
    y_true = np.asarray(y_true)
    probs = np.asarray(probs)
    thresholds = np.linspace(0.0, 1.0, n_points)
    fpr: list[float] = []
    tpr: list[float] = []
    for t in thresholds:
        pred = (probs >= t).astype(int)
        cm = confusion_matrix(y_true, pred)
        fp, tn = cm["fp"], cm["tn"]
        tp, fn = cm["tp"], cm["fn"]
        fpr.append(round(fp / (fp + tn), 4) if fp + tn else 0.0)
        tpr.append(round(tp / (tp + fn), 4) if tp + fn else 0.0)
    # AUC via the rank (Mann-Whitney) method: P(a random churner's score
    # exceeds a random stayer's), with ties counted a half. This is the
    # standard AUC and does not depend on the threshold grid used for the
    # chart points above (a trapezoid over that grid underestimates it
    # when several thresholds share one FPR).
    order = np.argsort(probs, kind="mergesort")
    sorted_probs = probs[order]
    ranks = np.empty(len(probs), dtype=np.float64)
    i = 0
    while i < len(probs):
        j = i
        while j + 1 < len(probs) and sorted_probs[j + 1] == sorted_probs[i]:
            j += 1
        # Average rank (1-based) shared by this tie group.
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        auc = 0.0
    else:
        auc = float(
            (np.sum(ranks[y_true == 1]) - n_pos * (n_pos + 1) / 2.0)
            / (n_pos * n_neg)
        )
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": [round(float(t), 4) for t in thresholds],
        "auc": round(auc, 4),
    }


def permutation_importance(
    model, X: np.ndarray, y: np.ndarray, seed: int = 5
) -> list[dict]:
    """Drop in accuracy when one feature column is shuffled (one pass).

    Returns features ranked most -> least important, with the raw drop.
    """
    from .data import FEATURES

    baseline = float(np.mean(model.predict(X) == y))
    rng = np.random.default_rng(seed)
    importances: list[dict] = []
    for col, name in enumerate(FEATURES):
        shuffled = X.copy()
        shuffled[:, col] = shuffled[rng.permutation(len(X)), col]
        score = float(np.mean(model.predict(shuffled) == y))
        importances.append(
            {"feature": name, "importance": round(baseline - score, 4)}
        )
    importances.sort(key=lambda d: d["importance"], reverse=True)
    return importances
