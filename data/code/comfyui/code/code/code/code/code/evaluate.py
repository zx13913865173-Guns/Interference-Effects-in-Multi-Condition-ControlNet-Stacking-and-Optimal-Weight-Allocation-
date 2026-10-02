"""
Main evaluation script.

Computes condition compliance, raw interference index, SII,
fixed-A decomposition, and perceptual utility for generated images.

Usage:
    python code/evaluate.py --input_dir outputs/ --output_dir results/
"""

import argparse
import os
import numpy as np
import pandas as pd

from utils import (
    compute_canny_iou,
    compute_depth_ssim,
    compute_pose_matching,
    compute_scribble_similarity,
    compute_normalized_compliance
)


def evaluate_directory(input_dir):
    """
    Placeholder for loading generated images, reference images,
    and condition maps, then computing per-image compliance.

    Replace the body with your actual data loading logic.
    """
    results = []
    # Example structure:
    # for each generated image:
    #     canny_iou = compute_canny_iou(gen, ref)
    #     depth_ssim = compute_depth_ssim(gen_depth, ref_depth)
    #     results.append({...})
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = evaluate_directory(args.input_dir)
    df.to_csv(os.path.join(args.output_dir, "compliance_metrics.csv"), index=False)
    print("Saved compliance_metrics.csv to", args.output_dir)


if __name__ == "__main__":
    main()
