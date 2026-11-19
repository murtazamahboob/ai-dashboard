"""Train the churn model, evaluate it, and export everything the web
dashboard needs: weights, training history and evaluation metrics.

Run: ``python examples/demo.py`` (also writes outputs/report.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from dashboard.data import FEATURES, generate, split, standardize
from dashboard.metrics import (
    classification_report,
    permutation_importance,
    roc_curve,
)
from dashboard.model import MLP


def build(seed_data: int = 42, epochs: int = 60) -> dict:
    """Train + evaluate; return a dict with every artifact's content."""
    X, y = generate(n=2000, seed=seed_data)
    X_train, X_test, y_train, y_test = split(X, y)
    Xtr, Xte, mean, std = standardize(X_train, X_test)
    model = MLP(seed=123)
    history = model.train(Xtr, y_train, epochs=epochs)
    probs_test = model.predict_proba(Xte)
    pred_test = model.predict(Xte)
    report = classification_report(y_test, pred_test)
    roc = roc_curve(y_test, probs_test)
    importance = permutation_importance(model, Xte, y_test)
    # Live-stream seed rows: 12 representative customers with predictions.
    rng = np.random.default_rng(2026)
    sample_idx = rng.choice(len(X_test), size=12, replace=False)
    live_rows = [
        {
            "customer": f"C-{1000 + int(i)}",
            "features": {
                name: round(float(X_test[i, j]), 1)
                for j, name in enumerate(FEATURES)
            },
            "churn_probability": round(float(probs_test[i]), 3),
            "predicted_churn": bool(pred_test[i]),
            "actual_churn": bool(y_test[i]),
        }
        for i in sample_idx
    ]
    return {
        "model": model,
        "mean": mean,
        "std": std,
        "history": history,
        "report": report,
        "roc": roc,
        "importance": importance,
        "live_rows": live_rows,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "churn_rate": round(float(y.mean()), 4),
        "final_train_loss": history[-1]["loss"],
        "final_train_accuracy": history[-1]["accuracy"],
    }


def export(result: dict, root: Path) -> dict:
    """Write web/data/*.json; return the paths written."""
    data_dir = root / "web" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    weights_path = data_dir / "weights.json"
    result["model"].save_weights(weights_path, result["mean"], result["std"])
    history_path = data_dir / "training_history.json"
    history_path.write_text(json.dumps({"history": result["history"]}, indent=2))
    metrics_path = data_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "report": result["report"],
                "roc": result["roc"],
                "importance": result["importance"],
                "live_rows": result["live_rows"],
                "n_train": result["n_train"],
                "n_test": result["n_test"],
                "churn_rate": result["churn_rate"],
            },
            indent=2,
        )
    )
    return {
        "weights": str(weights_path),
        "history": str(history_path),
        "metrics": str(metrics_path),
    }


def summary(result: dict) -> dict:
    return {
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "churn_rate": result["churn_rate"],
        "test_accuracy": result["report"]["accuracy"],
        "test_precision": result["report"]["precision"],
        "test_recall": result["report"]["recall"],
        "test_f1": result["report"]["f1"],
        "roc_auc": result["roc"]["auc"],
        "confusion": result["report"]["confusion"],
        "final_train_loss": result["final_train_loss"],
        "final_train_accuracy": result["final_train_accuracy"],
        "epochs": len(result["history"]),
        "top_feature": result["importance"][0]["feature"],
    }
