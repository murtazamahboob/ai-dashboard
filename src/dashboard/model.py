"""A small multilayer perceptron (MLP) trained from scratch in NumPy.

Architecture: 8 -> 16 -> 8 -> 1, ReLU hidden layers, sigmoid output.
Trained with minibatch gradient descent and Adam. The trained weights
are exported to ``web/data/weights.json`` so the dashboard runs the
*exact same model* live in the browser (see ``web/js/model.js``) — the
Python and JavaScript forwards are tested to agree to 1e-6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


class MLP:
    """8 -> 16 -> 8 -> 1 binary classifier."""

    LAYER_SIZES = (8, 16, 8, 1)

    def __init__(self, seed: int = 123):
        rng = np.random.default_rng(seed)
        self.params: dict[str, np.ndarray] = {}
        sizes = self.LAYER_SIZES
        for i in range(len(sizes) - 1):
            scale = np.sqrt(2.0 / sizes[i])  # He initialisation
            self.params[f"W{i + 1}"] = rng.normal(0, scale, (sizes[i], sizes[i + 1]))
            self.params[f"b{i + 1}"] = np.zeros(sizes[i + 1])

    # -- forward ---------------------------------------------------------
    def forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return (probabilities, cache) for a batch X (n, 8)."""
        cache: dict[str, np.ndarray] = {"A0": X}
        A = X
        for layer in (1, 2):
            Z = A @ self.params[f"W{layer}"] + self.params[f"b{layer}"]
            A = _relu(Z)
            cache[f"Z{layer}"] = Z
            cache[f"A{layer}"] = A
        Z3 = A @ self.params["W3"] + self.params["b3"]
        probs = _sigmoid(Z3).reshape(-1)
        cache["Z3"] = Z3
        cache["A3"] = probs
        return probs, cache

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs, _ = self.forward(np.atleast_2d(X))
        return probs

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -- training --------------------------------------------------------
    @staticmethod
    def _loss(probs: np.ndarray, y: np.ndarray) -> float:
        eps = 1e-12
        p = np.clip(probs, eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 60,
        batch_size: int = 128,
        lr: float = 0.01,
        seed: int = 99,
    ) -> list[dict]:
        """Train with Adam; return per-epoch history of loss/accuracy."""
        rng = np.random.default_rng(seed)
        n = len(X)
        m = {k: np.zeros_like(v) for k, v in self.params.items()}
        v = {k: np.zeros_like(v) for k, v in self.params.items()}
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        step = 0
        history: list[dict] = []
        yf = y.astype(np.float64)
        for epoch in range(1, epochs + 1):
            perm = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                xb, yb = X[idx], yf[idx]
                probs, cache = self.forward(xb)
                batch = len(idx)
                # Backprop for binary cross-entropy + sigmoid output.
                dZ3 = ((probs - yb) / batch).reshape(-1, 1)
                grads: dict[str, np.ndarray] = {}
                grads["W3"] = cache["A2"].T @ dZ3
                grads["b3"] = dZ3.sum(axis=0)
                dA2 = dZ3 @ self.params["W3"].T
                dZ2 = dA2 * _relu_grad(cache["Z2"])
                grads["W2"] = cache["A1"].T @ dZ2
                grads["b2"] = dZ2.sum(axis=0)
                dA1 = dZ2 @ self.params["W2"].T
                dZ1 = dA1 * _relu_grad(cache["Z1"])
                grads["W1"] = cache["A0"].T @ dZ1
                grads["b1"] = dZ1.sum(axis=0)
                # Adam update.
                step += 1
                for key in self.params:
                    g = grads[key]
                    m[key] = beta1 * m[key] + (1 - beta1) * g
                    v[key] = beta2 * v[key] + (1 - beta2) * (g * g)
                    m_hat = m[key] / (1 - beta1**step)
                    v_hat = v[key] / (1 - beta2**step)
                    self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            probs_all, _ = self.forward(X)
            history.append(
                {
                    "epoch": epoch,
                    "loss": round(self._loss(probs_all, yf), 4),
                    "accuracy": round(float(np.mean(self.predict(X) == y)), 4),
                }
            )
        return history

    # -- persistence -----------------------------------------------------
    def to_dict(self, feature_mean: np.ndarray, feature_std: np.ndarray) -> dict:
        return {
            "architecture": list(self.LAYER_SIZES),
            "features": [
                "tenure_months",
                "monthly_charges",
                "total_charges",
                "contract_months",
                "support_calls",
                "data_usage_gb",
                "age",
                "num_services",
            ],
            "feature_mean": [round(float(v), 6) for v in feature_mean],
            "feature_std": [round(float(v), 6) for v in feature_std],
            "weights": {
                key: [[round(float(v), 6) for v in row] for row in val]
                if val.ndim == 2
                else [round(float(v), 6) for v in val]
                for key, val in self.params.items()
            },
        }

    def save_weights(self, path: str | Path, feature_mean, feature_std) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(feature_mean, feature_std), indent=2))
