"""
TensorFlow dataset generators.

AXIAL 2.5D VERSION

Input:
    (512, 512, num_slices)

Outputs:
    Segmentation: (512, 512, num_classes)
    Synthesis:    (512, 512, 1)
"""

import os
import numpy as np
import tensorflow as tf
import nibabel as nib

from .augmentations import apply_augmentations


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
# SLICE GENERATOR
# ==========================================================

def slice_generator(
    data_dir,
    patient_list,
    num_classes,
    num_aug=2,
    include_original=True,
    num_slices=5
):

    patient_list = set(patient_list)

    for patient in sorted(
        os.listdir(data_dir)
    ):

        if patient not in patient_list:
            continue

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

        cbct_path = os.path.join(
            session_path,
            "CBCT.nii.gz"
        )

        seg_path = os.path.join(
            session_path,
            "mask.nii.gz"
        )

        ct_path = os.path.join(
            session_path,
            "CT.nii.gz"
        )

        if not all(
            os.path.exists(p)
            for p in [
                cbct_path,
                seg_path,
                ct_path
            ]
        ):
            continue

        cbct = nib.load(
            cbct_path
        ).get_fdata()

        seg = nib.load(
            seg_path
        ).get_fdata().astype(
            np.int32
        )

        ct = nib.load(
            ct_path
        ).get_fdata().astype(
            np.float32
        )

        if (
            cbct.shape != seg.shape
            or cbct.shape != ct.shape
        ):
            continue

        # ==================================================
        # AXIAL SLICES
        # ==================================================

        for i in range(
            cbct.shape[2]
        ):

            cbct_slice = get_25d_slice(
                cbct,
                i,
                num_slices=num_slices
            ).astype(
                np.float32
            )

            seg_slice = (
                seg[:, :, i]
                .astype(np.int32)
            )

            ct_slice = (
                ct[:, :, i]
                [..., np.newaxis]
                .astype(np.float32)
            )

            augmented = apply_augmentations(
                cbct_slice,
                seg_slice,
                ct_slice,
                num_aug=num_aug,
                include_original=include_original
            )

            for (
                aug_cbct,
                aug_seg,
                aug_ct
            ) in augmented:

                seg_onehot = tf.one_hot(
                    aug_seg,
                    depth=num_classes
                )

                yield (
                    aug_cbct,
                    (
                        seg_onehot,
                        aug_ct
                    )
                )


# ==========================================================
# TF DATASET WRAPPER
# ==========================================================

def get_dataset(
    data_dir,
    patient_list,
    num_classes,
    batch_size=2,
    shuffle=True,
    num_aug=2,
    include_original=True,
    num_slices=5
):

    output_signature = (
        tf.TensorSpec(
            shape=(512, 512, num_slices),
            dtype=tf.float32
        ),
        (
            tf.TensorSpec(
                shape=(
                    512,
                    512,
                    num_classes
                ),
                dtype=tf.float32
            ),
            tf.TensorSpec(
                shape=(512, 512, 1),
                dtype=tf.float32
            )
        )
    )

    dataset = tf.data.Dataset.from_generator(
        lambda: slice_generator(
            data_dir=data_dir,
            patient_list=patient_list,
            num_classes=num_classes,
            num_aug=num_aug,
            include_original=include_original,
            num_slices=num_slices
        ),
        output_signature=output_signature
    )

    if shuffle:

        dataset = dataset.shuffle(
            buffer_size=100
        )

    dataset = dataset.repeat()

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )

    return dataset