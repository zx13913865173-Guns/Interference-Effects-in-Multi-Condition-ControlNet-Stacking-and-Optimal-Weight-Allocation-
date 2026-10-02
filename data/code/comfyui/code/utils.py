"""
Utility functions for condition compliance metrics.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def compute_canny_iou(gen_img, ref_img, low=100, high=200):
    """Edge IoU between generated and reference Canny edge maps."""
    gen_edges = cv2.Canny(gen_img, low, high)
    ref_edges = cv2.Canny(ref_img, low, high)
    intersection = np.logical_and(gen_edges, ref_edges).sum()
    union = np.logical_or(gen_edges, ref_edges).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_depth_ssim(gen_depth, ref_depth):
    """SSIM between generated and reference depth maps."""
    data_range = ref_depth.max() - ref_depth.min()
    if data_range == 0:
        return 0.0
    return float(ssim(gen_depth, ref_depth, data_range=data_range))


def compute_pose_matching(gen_kpts, ref_kpts, threshold=0.05):
    """Keypoint matching rate between generated and reference poses."""
    if len(ref_kpts) == 0:
        return 0.0
    matched = 0
    for ref in ref_kpts:
        dists = np.linalg.norm(gen_kpts - ref, axis=1)
        if dists.min() < threshold:
            matched += 1
    return float(matched) / float(len(ref_kpts))


def compute_scribble_similarity(gen_simplified, ref_scribble):
    """Shape similarity between simplified generated image and Scribble reference."""
    gen_bin = gen_simplified > 0.5
    ref_bin = ref_scribble > 0.5
    intersection = np.logical_and(gen_bin, ref_bin).sum()
    union = np.logical_or(gen_bin, ref_bin).sum()
    return float(intersection) / float(union) if union > 0 else 0.0


def compute_normalized_compliance(c_a, c_a_min, c_a_single):
    """Normalized compliance for SII computation."""
    denom = c_a_single - c_a_min
    if denom == 0:
        return 0.0
    return (c_a - c_a_min) / denom
