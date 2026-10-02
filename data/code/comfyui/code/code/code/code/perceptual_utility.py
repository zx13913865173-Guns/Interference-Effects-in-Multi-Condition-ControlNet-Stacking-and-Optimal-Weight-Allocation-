"""
Perceptual utility function and optimal ratio search.

U(w) = lambda1 * C_A_norm + lambda2 * C_B_norm + lambda3 * S_subj
       + lambda4 * (1 - abs(C_A_norm - C_B_norm)) - lambda5 * Conflict
"""

import numpy as np


DEFAULT_LAMBDAS = {
    "lambda1": 0.2,
    "lambda2": 0.2,
    "lambda3": 0.2,
    "lambda4": 0.2,
    "lambda5": 0.2
}


def perceptual_utility(c_a_norm, c_b_norm, s_subj, conflict, lambdas=None):
    lam = lambdas or DEFAULT_LAMBDAS
    imbalance_term = 1.0 - abs(c_a_norm - c_b_norm)
    return (
        lam["lambda1"] * c_a_norm
        + lam["lambda2"] * c_b_norm
        + lam["lambda3"] * s_subj
        + lam["lambda4"] * imbalance_term
        - lam["lambda5"] * conflict
    )


def search_optimal_ratio(ratios, c_a_norm_list, c_b_norm_list,
                         s_subj_list, conflict_list, lambdas=None):
    """
    ratios: list of (w_a, w_b) tuples
    Returns best ratio and utility.
    """
    best_ratio = None
    best_utility = -float("inf")
    for ratio, c_a, c_b, s, conf in zip(ratios, c_a_norm_list, c_b_norm_list,
                                        s_subj_list, conflict_list):
        u = perceptual_utility(c_a, c_b, s, conf, lambdas)
        if u > best_utility:
            best_utility = u
            best_ratio = ratio
    return best_ratio, best_utility


def sensitivity_analysis(ratios, c_a_norm_list, c_b_norm_list,
                         s_subj_list, conflict_list):
    """
    Compare equal-lambda vs. artist-ranking-derived lambdas.
    """
    equal_lambdas = DEFAULT_LAMBDAS
    artist_lambdas = {
        "lambda1": 0.25,
        "lambda2": 0.25,
        "lambda3": 0.20,
        "lambda4": 0.15,
        "lambda5": 0.15
    }
    ratio_equal, u_equal = search_optimal_ratio(
        ratios, c_a_norm_list, c_b_norm_list, s_subj_list, conflict_list, equal_lambdas
    )
    ratio_artist, u_artist = search_optimal_ratio(
        ratios, c_a_norm_list, c_b_norm_list, s_subj_list, conflict_list, artist_lambdas
    )
    return {
        "equal_lambdas": {"ratio": ratio_equal, "utility": u_equal},
        "artist_lambdas": {"ratio": ratio_artist, "utility": u_artist}
    }
