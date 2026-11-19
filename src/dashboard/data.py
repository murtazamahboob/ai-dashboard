"""Synthetic customer-churn dataset for the AI dashboard.

The data is generated deterministically (seeded NumPy RNG) from a known
latent rule, so the bundled demo is reproducible and the trained model has
a real signal to learn. No real customer data is used anywhere.

Features (8, all numeric):
    tenure_months     months as a customer (0-72)
    monthly_charges   monthly bill in USD (20-120)
    total_charges     tenure * monthly_charges (derived, noisy)
    contract_months   1, 12 or 24 month contract
    support_calls     support calls in the last 6 months (0-10)
    data_usage_gb     monthly data usage in GB (0-500)
    age               customer age (18-80)
    num_services      subscribed services (1-6)

Label: churn (1) / stay (0). The latent log-odds reward long tenure,
long contracts and high usage, and penalise high charges, many support
calls and few services, with one interaction (high charges hurt most
when tenure is short). A small amount of label noise keeps the task
honest: the Bayes accuracy is below 100%.
"""

from __future__ import annotations

import numpy as np

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "contract_months",
    "support_calls",
    "data_usage_gb",
    "age",
    "num_services",
]

# (low, high) for uniform sampling of each feature, in FEATURES order.
FEATURE_RANGES = [
    (0.0, 72.0),
    (20.0, 120.0),
    (0.0, 1.0),  # total_charges is derived, not sampled
    (1.0, 24.0),
    (0.0, 10.0),
    (0.0, 500.0),
    (18.0, 80.0),
    (1.0, 6.0),
]

CONTRACT_CHOICES = np.array([1.0, 12.0, 24.0])


def _logit(X: np.ndarray) -> np.ndarray:
    """Latent log-odds of churning. X is raw (unstandardised) features."""
    tenure = X[:, 0]
    charges = X[:, 1]
    contract = X[:, 3]
    calls = X[:, 4]
    usage = X[:, 5]
    services = X[:, 7]
    # Interaction: expensive + new customer churns fastest.
    interaction = (charges - 70.0) / 50.0 * np.maximum(0.0, (12.0 - tenure)) / 12.0
    return (
        -2.20
        - 0.09 * tenure
        + 0.056 * charges
        - 0.11 * contract
        + 0.76 * calls
        - 0.008 * usage
        - 0.44 * services
        + 1.8 * interaction
    )


def generate(n: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate ``n`` labelled rows. Returns (X, y) with X float64 (n, 8)."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    X = np.empty((n, len(FEATURES)), dtype=np.float64)
    X[:, 0] = rng.uniform(0, 72, n)                       # tenure
    X[:, 1] = rng.uniform(20, 120, n)                     # monthly charges
    X[:, 3] = rng.choice(CONTRACT_CHOICES, size=n)        # contract
    X[:, 4] = rng.integers(0, 11, n).astype(float)         # support calls
    X[:, 5] = rng.uniform(0, 500, n)                      # data usage
    X[:, 6] = rng.uniform(18, 80, n)                      # age
    X[:, 7] = rng.integers(1, 7, n).astype(float)          # services
    # total_charges is derived from tenure and monthly charges, with noise.
    X[:, 2] = X[:, 0] * X[:, 1] * rng.uniform(0.85, 1.15, n)
    logits = _logit(X)
    probs = 1.0 / (1.0 + np.exp(-logits))
    # 4% label noise: a few customers behave against the rule.
    flip = rng.random(n) < 0.04
    y = (rng.random(n) < probs).astype(int)
    y[flip] = 1 - y[flip]
    return X, y


def split(
    X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic stratified-ish split (shuffled, seed fixed)."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError("test_frac must be between 0 and 1")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_frac)
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-mean / unit-variance scaling fitted on the training split."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_train - mean) / std, (X_test - mean) / std, mean, std
