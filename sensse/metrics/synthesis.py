"""
Synthesis metrics.
"""

import numpy as np

from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio
)

def mae(
    y_true,
    y_pred
):

    return np.mean(
        np.abs(
            y_true - y_pred
        )
    )

def mse(
    y_true,
    y_pred
):

    return np.mean(
        (
            y_true - y_pred
        ) ** 2
    )

def rmse(
    y_true,
    y_pred
):

    return np.sqrt(
        mse(
            y_true,
            y_pred
        )
    )

def psnr(
    y_true,
    y_pred
):

    return peak_signal_noise_ratio(
        y_true,
        y_pred,
        data_range=1.0
    )

def ssim(
    y_true,
    y_pred
):

    return structural_similarity(
        y_true,
        y_pred,
        data_range=1.0
    )

def evaluate_synthesis(
    y_true,
    y_pred
):

    return {

        "MAE": mae(
            y_true,
            y_pred
        ),

        "RMSE": rmse(
            y_true,
            y_pred
        ),

        "PSNR": psnr(
            y_true,
            y_pred
        ),

        "SSIM": ssim(
            y_true,
            y_pred
        )
    }
