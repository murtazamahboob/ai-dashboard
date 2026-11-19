"""Tests for the MLP: forward pass, training, and weight export."""

import json

import numpy as np

from dashboard.data import generate, split, standardize
from dashboard.model import MLP


def _small_data():
    X, y = generate(n=600, seed=42)
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr_s, Xte_s, mean, std = standardize(Xtr, Xte)
    return Xtr_s, Xte_s, ytr, yte, mean, std


def test_forward_shapes_and_range():
    model = MLP(seed=1)
    X = np.random.default_rng(0).normal(size=(5, 8))
    probs, cache = model.forward(X)
    assert probs.shape == (5,)
    assert np.all((probs >= 0) & (probs <= 1))
    assert cache["A1"].shape == (5, 16)
    assert cache["A2"].shape == (5, 8)


def test_predict_binary():
    model = MLP(seed=1)
    X = np.random.default_rng(0).normal(size=(10, 8))
    assert set(np.unique(model.predict(X))) <= {0, 1}
    assert model.predict_proba(X).shape == (10,)


def test_training_reduces_loss_and_learns():
    Xtr, Xte, ytr, yte, _, _ = _small_data()
    model = MLP(seed=123)
    before = model._loss(model.predict_proba(Xtr), ytr)
    history = model.train(Xtr, ytr, epochs=25)
    after = history[-1]["loss"]
    assert after < before - 0.05
    assert len(history) == 25
    assert all(history[i]["loss"] >= history[-1]["loss"] - 0.15
               for i in range(len(history)))
    acc = float(np.mean(model.predict(Xte) == yte))
    assert acc > 0.75, f"test accuracy {acc:.3f} too low"


def test_full_training_accuracy_threshold():
    """The demo configuration must clear the bar the README claims."""
    X, y = generate(n=2000, seed=42)
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr_s, Xte_s, _, _ = standardize(Xtr, Xte)
    model = MLP(seed=123)
    model.train(Xtr_s, ytr, epochs=60)
    acc = float(np.mean(model.predict(Xte_s) == yte))
    assert acc >= 0.80, f"test accuracy {acc:.3f} below 0.80"


def test_training_deterministic():
    Xtr, _, ytr, _, _, _ = _small_data()
    m1, m2 = MLP(seed=123), MLP(seed=123)
    h1 = m1.train(Xtr, ytr, epochs=5)
    h2 = m2.train(Xtr, ytr, epochs=5)
    assert h1 == h2


def test_weight_export_roundtrip(tmp_path):
    _, _, _, _, mean, std = _small_data()
    model = MLP(seed=1)
    path = tmp_path / "weights.json"
    model.save_weights(path, mean, std)
    data = json.loads(path.read_text())
    assert data["architecture"] == [8, 16, 8, 1]
    assert len(data["features"]) == 8
    assert len(data["feature_mean"]) == 8
    assert len(data["feature_std"]) == 8
    assert np.array(data["weights"]["W1"]).shape == (8, 16)
    assert np.array(data["weights"]["W2"]).shape == (16, 8)
    assert np.array(data["weights"]["W3"]).shape == (8, 1)
    assert np.array(data["weights"]["b1"]).shape == (16,)
    assert np.array(data["weights"]["b3"]).shape == (1,)
    assert all(s > 0 for s in data["feature_std"])
