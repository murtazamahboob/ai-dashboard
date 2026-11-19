"""Tests for evaluation metrics."""

import numpy as np

from dashboard.metrics import (
    classification_report,
    confusion_matrix,
    permutation_importance,
    roc_curve,
)
from dashboard.model import MLP


def test_confusion_matrix_counts():
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 1, 0])
    cm = confusion_matrix(y_true, y_pred)
    assert cm == {"tn": 2, "fp": 1, "fn": 1, "tp": 2}
    assert sum(cm.values()) == 6


def test_classification_report_perfect_and_worst():
    y = np.array([0, 0, 1, 1])
    perfect = classification_report(y, y)
    assert perfect["accuracy"] == 1.0
    assert perfect["precision"] == 1.0
    assert perfect["recall"] == 1.0
    assert perfect["f1"] == 1.0
    worst = classification_report(y, 1 - y)
    assert worst["accuracy"] == 0.0
    assert worst["precision"] == 0.0
    assert worst["recall"] == 0.0


def test_roc_curve_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = roc_curve(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]))
    assert perfect["auc"] > 0.95
    assert len(perfect["fpr"]) == len(perfect["tpr"]) == 25
    # Endpoints: threshold 0 catches everything, threshold 1 catches nothing.
    assert perfect["tpr"][0] == 1.0 and perfect["fpr"][0] == 1.0
    assert perfect["tpr"][-1] == 0.0 and perfect["fpr"][-1] == 0.0
    # AUC is bounded and monotone-ish in quality.
    mediocre = roc_curve(y, np.array([0.4, 0.5, 0.6, 0.4, 0.5, 0.6]))
    assert 0.0 <= mediocre["auc"] <= 1.0
    assert perfect["auc"] >= mediocre["auc"]


def test_permutation_importance_ranks_and_sums():
    from dashboard.data import generate, split, standardize

    X, y = generate(n=800, seed=42)
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
    model = MLP(seed=123)
    model.train(Xtr_s, ytr, epochs=20)
    importance = permutation_importance(model, Xte_s, yte)
    assert len(importance) == 8
    names = [d["feature"] for d in importance]
    assert len(set(names)) == 8
    values = [d["importance"] for d in importance]
    assert values == sorted(values, reverse=True)
    # Shuffling a real signal feature must hurt at least once.
    assert max(values) > 0.0
    # Tenure / support calls / contract should be among the top drivers.
    top3 = set(names[:3])
    assert top3 & {"tenure_months", "support_calls", "contract_months", "monthly_charges"}
