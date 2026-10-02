"""
Standardized Interference Index (SII) computation.

For each prompt, fit a linear regression of C_A_norm on w_B across five B levels.
The slope is the per-prompt SII. Report the mean across prompts with bootstrap CI.
"""

import numpy as np


def compute_sii_per_prompt(compliance_norm_list, b_weight_list):
    """
    compliance_norm_list: list of normalized compliance values for condition A
    b_weight_list: corresponding B weights
    Returns the absolute slope.
    """
    x = np.array(b_weight_list, dtype=float)
    y = np.array(compliance_norm_list, dtype=float)
    if len(x) < 2:
        return 0.0
    x_mean = x.mean()
    y_mean = y.mean()
    denom = ((x - x_mean) ** 2).sum()
    if denom == 0:
        return 0.0
    slope = ((x - x_mean) * (y - y_mean)).sum() / denom
    return abs(float(slope))


def bootstrap_ci(values, n_resamples=1000, alpha=0.05, seed=42):
    """Bootstrap confidence interval over prompts."""
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


def compute_sii(all_prompt_compliance_norm, all_prompt_b_weights):
    """
    all_prompt_compliance_norm: list of lists, one per prompt
    all_prompt_b_weights: list of lists, one per prompt
    Returns mean SII and 95% CI.
    """
    per_prompt_sii = []
    for c_list, w_list in zip(all_prompt_compliance_norm, all_prompt_b_weights):
        per_prompt_sii.append(compute_sii_per_prompt(c_list, w_list))
    mean_sii = float(np.mean(per_prompt_sii))
    low, high = bootstrap_ci(per_prompt_sii)
    return mean_sii, low, high
