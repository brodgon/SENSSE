"""
inference.py

Prediction script.
"""

import os
import numpy as np
import nibabel as nib

from sensse.models.sensse import (
    build_sensse
)

from sensse.uncertainty.segmentation import (
    alpha_from_logits,
    total_uncertainty as seg_unc
)

from sensse.uncertainty.synthesis import (
    predictive_mean,
    aleatoric_uncertainty,
    epistemic_uncertainty,
    total_uncertainty
)


def get_25d_slice(
    volume,
    idx,
    num_slices
):

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
            volume.shape[2]-1
        )

        slices.append(
            volume[:, :, current_idx]
        )

    return np.stack(
        slices,
        axis=-1
    )


def predict_volume(
    cbct_path,
    weights_path,
    output_dir,
    num_slices=5,
    num_classes=10
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    volume = nib.load(
        cbct_path
    )

    cbct = volume.get_fdata()

    affine = volume.affine
    input_shape = (cbct.shape[0],cbct.shape[1],num_slices)

    model = build_sensse(
        input_shape=input_shape,
        num_classes=num_classes
    )

    model.load_weights(
        weights_path
    )

    sct = []

    aleatoric = []

    epistemic = []

    total = []

    segmentation = []

    seg_u = []

    for i in range(
        cbct.shape[-1]
    ):

        slice_img = get_25d_slice(
            cbct,
            i,
            num_slices=input_shape[-1]
        ).astype(np.float32)

        slice_img = np.expand_dims(
            slice_img,
            axis=0
        )

        seg_logits, syn_pred = (
            model.predict(
                slice_img,
                verbose=0
            )
        )

        alpha = alpha_from_logits(
            seg_logits
        )

        pred_mask = np.argmax(
            alpha.numpy(),
            axis=-1
        )[0]

        mu = predictive_mean(
            syn_pred
        )[0,...,0].numpy()

        sct.append(mu)

        segmentation.append(
            pred_mask
        )

        aleatoric.append(
            aleatoric_uncertainty(
                syn_pred
            )[0,...,0].numpy()
        )

        epistemic.append(
            epistemic_uncertainty(
                syn_pred
            )[0,...,0].numpy()
        )

        total.append(
            total_uncertainty(
                syn_pred
            )[0,...,0].numpy()
        )

        seg_u.append(
            seg_unc(alpha)[0,...,0].numpy()
        )

    nib.save(
        nib.Nifti1Image(
            np.stack(sct,-1),
            affine
        ),
        os.path.join(
            output_dir,
            "sCT.nii.gz"
        )
    )

    nib.save(
        nib.Nifti1Image(
            np.stack(segmentation,-1),
            affine
        ),
        os.path.join(
            output_dir,
            "segmentation.nii.gz"
        )
    )

    nib.save(
        nib.Nifti1Image(
            np.stack(aleatoric,-1),
            affine
        ),
        os.path.join(
            output_dir,
            "aleatoric.nii.gz"
        )
    )

    nib.save(
        nib.Nifti1Image(
            np.stack(epistemic,-1),
            affine
        ),
        os.path.join(
            output_dir,
            "epistemic.nii.gz"
        )
    )

    nib.save(
        nib.Nifti1Image(
            np.stack(total,-1),
            affine
        ),
        os.path.join(
            output_dir,
            "total_uncertainty.nii.gz"
        )
    )
