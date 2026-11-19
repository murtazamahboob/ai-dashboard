"""Tests for the synthetic churn dataset."""

import numpy as np
import pytest

from dashboard.data import FEATURES, generate, split, standardize


def test_generate_shape_and_labels():
    X, y = generate(n=500, seed=42)
    assert X.shape == (500, 8)
    assert y.shape == (500,)
    assert set(np.unique(y)) == {0, 1}
    # Churn rate is meaningful but not degenerate.
    assert 0.15 < y.mean() < 0.65


def test_generate_deterministic():
    X1, y1 = generate(n=200, seed=1)
    X2, y2 = generate(n=200, seed=1)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
    X3, _ = generate(n=200, seed=2)
    assert not np.array_equal(X1, X3)


def test_feature_ranges():
    X, _ = generate(n=2000, seed=42)
    assert X[:, 0].min() >= 0 and X[:, 0].max() <= 72        # tenure
    assert X[:, 1].min() >= 20 and X[:, 1].max() <= 120      # charges
    assert X[:, 2].min() >= 0                                 # total charges
    assert set(np.unique(X[:, 3])) <= {1.0, 12.0, 24.0}      # contract
    assert X[:, 4].min() >= 0 and X[:, 4].max() <= 10         # support calls
    assert X[:, 5].min() >= 0 and X[:, 5].max() <= 500        # usage
    assert X[:, 6].min() >= 18 and X[:, 6].max() <= 80        # age
    assert set(np.unique(X[:, 7])) <= {1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    assert not np.isnan(X).any()
    assert len(FEATURES) == 8


def test_total_charges_tracks_tenure_times_charges():
    X, _ = generate(n=1000, seed=3)
    ratio = X[:, 2] / np.maximum(X[:, 0] * X[:, 1], 1e-9)
    # Derived with 0.85-1.15 noise; only check rows with tenure > 1.
    mask = X[:, 0] > 1
    assert np.all(ratio[mask] > 0.8) and np.all(ratio[mask] < 1.2)


def test_churn_signal_is_real():
    """High support calls + short tenure should churn more often."""
    X, y = generate(n=4000, seed=42)
    risky = (X[:, 4] >= 6) & (X[:, 0] < 12)
    safe = (X[:, 4] <= 1) & (X[:, 0] > 48) & (X[:, 3] == 24)
    assert y[risky].mean() > y[safe].mean() + 0.3


def test_split_sizes_and_disjoint():
    X, y = generate(n=1000, seed=42)
    Xtr, Xte, ytr, yte = split(X, y, test_frac=0.2, seed=7)
    assert len(Xtr) == 800 and len(Xte) == 200
    assert len(ytr) == 800 and len(yte) == 200
    # No row appears in both splits (compare on tenure+charges+age).
    keys_tr = {tuple(r) for r in Xtr[:, [0, 1, 6]]}
    keys_te = {tuple(r) for r in Xte[:, [0, 1, 6]]}
    assert keys_tr.isdisjoint(keys_te)
    with pytest.raises(ValueError):
        split(X, y, test_frac=1.5)


def test_standardize_zero_mean_unit_variance():
    X, y = generate(n=1000, seed=42)
    Xtr, Xte, _, _ = split(X, y)
    Xtr_s, Xte_s, mean, std = standardize(Xtr, Xte)
    assert np.allclose(Xtr_s.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(Xtr_s.std(axis=0), 1, atol=1e-10)
    assert np.all(std > 0)
    # Test split uses the TRAIN mean/std, not its own.
    assert np.allclose(Xte_s * std + mean, Xte)
