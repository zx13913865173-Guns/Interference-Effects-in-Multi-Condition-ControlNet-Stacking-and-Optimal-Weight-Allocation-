"""
Auto-balancing tool for multi-condition ControlNet stacking.

Input: condition A, condition B, CFG scale, model family.
Output: recommended weights for A and B.
"""

import argparse
import json
import os

DEFAULT_COEFF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "interference_coefficients.json"
)


def load_coefficients(path=None):
    path = path or DEFAULT_COEFF_PATH
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def auto_balance(condition_a, condition_b, cfg, model_family, coeff_table):
    if model_family not in coeff_table:
        raise ValueError(
            "Model family '{}' not found. Available: {}".format(
                model_family, list(coeff_table.keys())
            )
        )

    family = coeff_table[model_family]
    if condition_a not in family or condition_b not in family[condition_a]:
        raise ValueError(
            "Coefficient not found for {} -> {} on {}".format(
                condition_a, condition_b, model_family
            )
        )

    entry = family[condition_a][condition_b]
    raw = entry["raw_interference"]

    w_a = 1.0 / (1.0 + raw)
    w_b = 1.0 - w_a

    cfg_cfg = coeff_table.get("cfg_compensation", {})
    threshold = cfg_cfg.get("threshold", 10.0)
    bonus = cfg_cfg.get("weaker_condition_bonus", 0.05)

    if cfg > threshold:
        w_b += bonus
        w_a -= bonus

    w_a = max(0.1, min(0.9, w_a))
    w_b = 1.0 - w_a

    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "model_family": model_family,
        "cfg": cfg,
        "raw_interference": raw,
        "recommended_ratio": "{}:{}".format(int(round(w_a * 100)), int(round(w_b * 100))),
        "weight_a": round(w_a, 2),
        "weight_b": round(w_b, 2)
    }


def main():
    parser = argparse.ArgumentParser(description="Auto-balance ControlNet weights.")
    parser.add_argument("--condition_a", type=str, required=True,
                        help="First condition, e.g., canny")
    parser.add_argument("--condition_b", type=str, required=True,
                        help="Second condition, e.g., depth")
    parser.add_argument("--cfg", type=float, default=7.5,
                        help="CFG scale, default 7.5")
    parser.add_argument("--model", type=str, default="sdxl",
                        choices=["sdxl", "sd15", "flux"],
                        help="Model family")
    parser.add_argument("--coeff_path", type=str, default=None,
                        help="Path to interference_coefficients.json")
    args = parser.parse_args()

    coeff_table = load_coefficients(args.coeff_path)
    result = auto_balance(
        args.condition_a,
        args.condition_b,
        args.cfg,
        args.model,
        coeff_table
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
