"""
Dataset loading utilities.

Supports both:
    - 2D      (num_slices = 1)
    - 2.5D    (num_slices = 3, 5, 7, ...)
"""

import os
import numpy as np

from tqdm import tqdm

from .preprocessing import (
    load_nifti,
    one_hot_encode,
    check_shapes
)


# ==========================================================
# 2.5D CONTEXT EXTRACTION
# ==========================================================

def get_25d_slice(
    volume,
    idx,
    num_slices
):
    """
    Extract a 2.5D stack centered on idx.

    Example:
        num_slices = 5

        [i-2, i-1, i, i+1, i+2]

    Border slices are replicated.
    """

    if num_slices % 2 == 0:
        raise ValueError(
            "num_slices must be odd."
        )

    half = num_slices // 2

    slices = []

    for offset in range(
        -half,
        half + 1
    ):

        current_idx = min(
            max(
                idx + offset,
                0
            ),
            volume.shape[2] - 1
        )

        slices.append(
            volume[:, :, current_idx]
        )

    return np.stack(
        slices,
        axis=-1
    )


# ==========================================================
# LOAD ALL SLICES
# ==========================================================

def load_all_slices(
    data_dir,
    num_classes,
    num_slices=5
):

    input_slices = []
    seg_slices = []
    ct_slices = []
    patient_ids = []

    for patient in tqdm(
        sorted(
            os.listdir(data_dir)
        )
    ):

        patient_path = os.path.join(
            data_dir,
            patient
        )

        if not os.path.isdir(
            patient_path
        ):
            continue

        session = os.listdir(
            patient_path
        )[0]

        session_path = os.path.join(
            patient_path,
            session
        )

        cbct = load_nifti(
            os.path.join(
                session_path,
                "CBCT.nii.gz"
            )
        )

        seg = load_nifti(
            os.path.join(
                session_path,
                "mask.nii.gz"
            )
        ).astype(np.int32)

        ct = load_nifti(
            os.path.join(
                session_path,
                "CT.nii.gz"
            )
        ).astype(np.float32)

        if not check_shapes(
            cbct,
            seg,
            ct
        ):
            continue

        for i in range(
            cbct.shape[2]
        ):

            # ==========================================
            # 2.5D INPUT
            # ==========================================

            cbct_slice = get_25d_slice(
                cbct,
                i,
                num_slices=num_slices
            ).astype(np.float32)

            # ==========================================
            # CENTRAL SLICE TARGETS
            # ==========================================

            seg_slice = seg[:, :, i]

            seg_onehot = one_hot_encode(
                seg_slice,
                num_classes
            )

            ct_slice = (
                ct[:, :, i]
                [..., np.newaxis]
                .astype(np.float32)
            )

            input_slices.append(
                cbct_slice
            )

            seg_slices.append(
                seg_onehot
            )

            ct_slices.append(
                ct_slice
            )

            patient_ids.append(
                patient
            )

    return (
        np.stack(input_slices),
        np.stack(seg_slices),
        np.stack(ct_slices),
        np.array(patient_ids)
    )