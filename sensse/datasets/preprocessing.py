"""
Preprocessing utilities.
"""

import nibabel as nib
import numpy as np
import tensorflow as tf


def load_nifti(path):
    return nib.load(path).get_fdata()


def normalize_ct(
    image,
    hu_min=-1024,
    hu_max=3071
):
    image = np.clip(
        image,
        hu_min,
        hu_max
    )

    image = (
        image - hu_min
    ) / (
        hu_max - hu_min
    )

    return image.astype(np.float32)


def one_hot_encode(
    mask,
    num_classes
):

    return tf.one_hot(
        mask.astype(np.int32),
        depth=num_classes
    ).numpy()


def check_shapes(
    cbct,
    mask,
    ct
):

    return (
        cbct.shape == mask.shape
        and
        cbct.shape == ct.shape
    )
