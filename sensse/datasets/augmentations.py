"""
Data augmentation functions.
"""

import random
import numpy as np
import scipy.ndimage


def random_flip(
    image,
    mask,
    ct
):

    if random.random() > 0.5:

        image = np.flip(
            image,
            axis=1
        )

        mask = np.flip(
            mask,
            axis=1
        )

        ct = np.flip(
            ct,
            axis=1
        )

    return image, mask, ct


def random_rotate(
    image,
    mask,
    ct,
    max_angle=15
):

    angle = random.uniform(
        -max_angle,
        max_angle
    )

    image = scipy.ndimage.rotate(
        image,
        angle,
        reshape=False,
        order=1
    )

    mask = scipy.ndimage.rotate(
        mask,
        angle,
        reshape=False,
        order=0
    )

    ct = scipy.ndimage.rotate(
        ct,
        angle,
        reshape=False,
        order=1
    )

    return image, mask, ct


def random_intensity(
    image,
    factor_range=(0.9, 1.1)
):

    factor = random.uniform(
        *factor_range
    )

    return image * factor


def apply_augmentations(
    cbct,
    seg,
    ct,
    num_aug=2,
    include_original=True
):

    augmented = []

    if include_original:
        augmented.append(
            (cbct, seg, ct)
        )

    for _ in range(num_aug):

        aug_cbct = cbct.copy()
        aug_seg = seg.copy()
        aug_ct = ct.copy()

        if random.random() > 0.5:

            aug_cbct, aug_seg, aug_ct = random_flip(
                aug_cbct,
                aug_seg,
                aug_ct
            )

        if random.random() > 0.5:

            aug_cbct, aug_seg, aug_ct = random_rotate(
                aug_cbct,
                aug_seg,
                aug_ct
            )

        if random.random() > 0.5:

            aug_cbct = random_intensity(
                aug_cbct
            )

        augmented.append(
            (
                aug_cbct,
                aug_seg,
                aug_ct
            )
        )

    return augmented
