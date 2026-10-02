import argparse
import json

# Load interference coefficient table from JSON
COEFF_TABLE = {
    "sdxl": {
        "canny": {"depth": {"raw": 0.42, "sii": 0.38}},
        "openpose": {"canny": {"raw": 0.61, "sii": 0.56}},
        "scribble": {"depth": {"raw": 0.33, "sii": 0.31}},
    },
    "sd15": {
        "canny": {"depth": {"raw": 0.40, "sii": 0.36}},
        "openpose": {"canny": {"raw": 0.55, "sii": 0.53}},
    },
    "flux": {
        "canny": {"depth": {"raw": 0.36, "sii": 0.33}},
        "openpose": {"canny": {"raw": 0.52, "sii": 0.50}},
    }
}

def auto_balance(condition_a, condition_b, cfg, model_family):
    if model_family not in COEFF_TABLE:
        raise ValueError(f"Model family {model_family} not in coefficient table.")
    coeff = COEFF_TABLE[model_family]
    if condition_a not in coeff or condition_b not in coeff[condition_a]:
        raise ValueError(f"Coefficient not found for {condition_a} -> {condition_b} on {model_family}.")
    raw = coeff[condition_a][condition_b]["raw"]
    # Initial weights inversely proportional to dominance
    w_a = 1.0 / (1.0 + raw)
    w_b = 1.0 - w_a

    if cfg > 10.0:
        w_b += 0.05
        w_a -= 0.05

    w_a = max(0.1, min(0.9, w_a))
    w_b = 1.0 - w_a
    return {condition_a: round(w_a, 2), condition_b: round(w_b, 2)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition_a", type=str, required=True)
    parser.add_argument("--condition_b", type=str, required=True)
    parser.add_argument("--cfg", type=float, default=7.5)
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "sd15", "flux"])
    args = parser.parse_args()
    weights = auto_balance(args.condition_a, args.condition_b, args.cfg, args.model)
    print(json.dumps(weights, indent=2))
