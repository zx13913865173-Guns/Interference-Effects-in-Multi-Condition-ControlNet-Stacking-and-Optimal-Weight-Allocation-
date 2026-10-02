"""
Fixed-A single-B variation experiment and decomposition.

Decomposes Delta C_A_total into:
  Delta C_A_reallocation + Delta C_A_interference
"""

import numpy as np


def compute_reallocation(c_a_norm_at_w_a1, c_a_norm_at_w_a2):
    """Compliance change caused by reducing A's own weight, in absence of B."""
    return c_a_norm_at_w_a2 - c_a_norm_at_w_a1


def compute_interference(c_a_norm_at_b1, c_a_norm_at_b2):
    """Compliance change caused by increasing B's weight while A is fixed."""
    return c_a_norm_at_b2 - c_a_norm_at_b1


def decompose(c_a_norm_at_w_a1, c_a_norm_at_w_a2,
              c_a_norm_at_b1_fixed_a, c_a_norm_at_b2_fixed_a):
    """
    Returns dict with total, reallocation, interference, and residual.
    """
    total = c_a_norm_at_w_a2 - c_a_norm_at_w_a1
    reallocation = compute_reallocation(c_a_norm_at_w_a1, c_a_norm_at_w_a2)
    interference = compute_interference(c_a_norm_at_b1_fixed_a, c_a_norm_at_b2_fixed_a)
    residual = total - (reallocation + interference)
    return {
        "total": float(total),
        "reallocation": float(reallocation),
        "interference": float(interference),
        "residual": float(residual)
    }


def bootstrap_ci(values, n_resamples=1000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=float)
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    means = []
    for _ in range(n_resamples):
        sample = rng.choice(values, size=n, replace=True)
        means.append(sample.mean())
    means = np.array(means)
    low = float(np.percentile(means, 100 * alpha / 2))
    high = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return low, high
