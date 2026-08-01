"""
Uncertainty metrics for SENSSE.
"""

import numpy as np

def expected_calibration_error(
    confidence,
    correctness,
    n_bins=15
):

    bins = np.linspace(
        0,
        1,
        n_bins + 1
    )

    ece = 0

    for i in range(n_bins):

        mask = (
            (confidence >= bins[i])
            &
            (confidence < bins[i+1])
        )

        if np.sum(mask) == 0:
            continue

        acc = np.mean(
            correctness[mask]
        )

        conf = np.mean(
            confidence[mask]
        )

        ece += (
            np.sum(mask)
            / len(confidence)
        ) * abs(acc - conf)

    return ece

def brier_score(
    y_true,
    probs
):

    return np.mean(
        np.sum(
            (probs - y_true) ** 2,
            axis=-1
        )
    )

def entropy(
    probs,
    eps=1e-8
):

    return -np.sum(
        probs *
        np.log(
            probs + eps
        ),
        axis=-1
    )


def uncertainty_error_overlap(
    uncertainty,
    error,
    percentile=90
):

    u_thresh = np.percentile(
        uncertainty,
        percentile
    )

    e_thresh = np.percentile(
        error,
        percentile
    )

    high_u = uncertainty >= u_thresh

    high_e = error >= e_thresh

    intersection = np.sum(
        high_u & high_e
    )

    union = np.sum(
        high_u | high_e
    )

    if union == 0:
        return 0

    return intersection / union
def ause(
    uncertainty,
    error
):
    """
    Area Under Sparsification Error.
    """

    idx = np.argsort(
        uncertainty
    )

    sorted_error = error[idx]

    curve = []

    for k in range(
        1,
        len(sorted_error)
    ):

        curve.append(
            np.mean(
                sorted_error[:-k]
            )
        )

    return np.mean(curve)

def negative_log_likelihood(
    y_true,
    probs,
    eps=1e-8
):

    return -np.mean(
        np.sum(
            y_true *
            np.log(
                probs + eps
            ),
            axis=-1
        )
    )
