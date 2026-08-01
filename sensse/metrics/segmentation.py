"""
Segmentation metrics for SENSSE.
"""

import numpy as np

from scipy.spatial.distance import cdist
from scipy.ndimage import binary_erosion


# ============================================================
# Dice
# ============================================================

def dice_score(
    y_true,
    y_pred,
    smooth=1e-6
):

    intersection = np.sum(
        y_true * y_pred
    )

    return (
        2.0 * intersection + smooth
    ) / (
        np.sum(y_true)
        +
        np.sum(y_pred)
        +
        smooth
    )


# ============================================================
# Multi-class Dice
# ============================================================

def multiclass_dice(
    y_true,
    y_pred,
    num_classes
):

    scores = {}

    for c in range(num_classes):

        scores[c] = dice_score(
            y_true == c,
            y_pred == c
        )

    return scores


# ============================================================
# Surface voxels
# ============================================================

def surface_voxels(mask):

    eroded = binary_erosion(mask)

    return np.argwhere(
        mask ^ eroded
    )


# ============================================================
# Hausdorff 95
# ============================================================

def hd95(
    y_true,
    y_pred
):

    A = surface_voxels(y_true)
    B = surface_voxels(y_pred)

    if len(A) == 0 or len(B) == 0:
        return np.nan

    D = cdist(A, B)

    d_ab = np.min(D, axis=1)
    d_ba = np.min(D, axis=0)

    return np.percentile(
        np.concatenate([d_ab, d_ba]),
        95
    )


# ============================================================
# Mean Surface Distance
# ============================================================

def mean_surface_distance(
    y_true,
    y_pred
):

    A = surface_voxels(y_true)
    B = surface_voxels(y_pred)

    if len(A) == 0 or len(B) == 0:
        return np.nan

    D = cdist(A, B)

    return np.mean(
        np.concatenate([
            np.min(D, axis=1),
            np.min(D, axis=0)
        ])
    )


# ============================================================
# Evaluate One Case
# ============================================================

def evaluate_segmentation(
    y_true,
    y_pred,
    num_classes
):

    results = {}

    dices = []

    hds = []

    msds = []

    for c in range(1, num_classes):

        gt = y_true == c

        pred = y_pred == c

        d = dice_score(gt, pred)

        # h = hd95(gt, pred)

        # m = mean_surface_distance(
        #     gt,
        #     pred
        # )

        dices.append(d)
        # hds.append(h)
        # msds.append(m)

    results["DSC"] = np.nanmean(dices)

    # results["95HD"] = np.nanmean(hds)

    # results["MSD"] = np.nanmean(msds)

    return results
