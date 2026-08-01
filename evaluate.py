"""
evaluate.py

Generic evaluation script.

Evaluates:
    - CT synthesis
    - Segmentation

Outputs:
    per_patient.csv
    summary.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib

from sensse.metrics.synthesis import (
    evaluate_synthesis
)

from sensse.metrics.segmentation import (
    evaluate_segmentation,
    dice_score
)

# =====================================================
# CONFIG
# =====================================================

HU_MIN = -1024
HU_MAX = 3071

NUM_CLASSES = 10


# =====================================================
# HELPERS
# =====================================================

def denormalize(x):

    return (
        x * (HU_MAX - HU_MIN)
        + HU_MIN
    )


# =====================================================
# MAIN
# =====================================================

def evaluate_folder(
    data_dir,
    predictions_dir,
    output_csv,
    output_summary,
    num_classes=10
):

    patients = sorted([
        p
        for p in os.listdir(data_dir)
        if os.path.isdir(
            os.path.join(
                data_dir,
                p
            )
        )
    ])

    results = []

    for patient in patients:

        print(patient)

        patient_dir = os.path.join(
            data_dir,
            patient
        )

        session = os.listdir(
            patient_dir
        )[0]

        session_dir = os.path.join(
            patient_dir,
            session
        )

        # ======================================
        # GT
        # ======================================

        gt_ct = nib.load(
            os.path.join(
                session_dir,
                "CT.nii.gz"
            )
        ).get_fdata().astype(
            np.float32
        )

        gt_seg = nib.load(
            os.path.join(
                session_dir,
                "mask.nii.gz"
            )
        ).get_fdata().astype(
            np.int32
        )

        # ======================================
        # PREDICTIONS
        # ======================================

        pred_dir = os.path.join(
            predictions_dir,
            patient
        )

        pred_ct = nib.load(
            os.path.join(
                pred_dir,
                "sCT.nii.gz"
            )
        ).get_fdata().astype(
            np.float32
        )

        pred_seg = nib.load(
            os.path.join(
                pred_dir,
                "segmentation.nii.gz"
            )
        ).get_fdata().astype(
            np.int32
        )

        # ======================================
        # HU MAE
        # ======================================

        gt_ct_hu = denormalize(
            gt_ct
        )

        pred_ct_hu = denormalize(
            pred_ct
        )

        mae_hu = np.mean(
            np.abs(
                gt_ct_hu -
                pred_ct_hu
            )
        )

        # ======================================
        # SYNTHESIS
        # ======================================

        syn_metrics = evaluate_synthesis(
            gt_ct,
            pred_ct
        )

        # ======================================
        # BONE DICE
        # ======================================

        bone_gt = (
            gt_ct_hu > 300
        )

        bone_pred = (
            pred_ct_hu > 300
        )

        dice_bone = dice_score(
            bone_gt,
            bone_pred
        )

        # ======================================
        # SEGMENTATION
        # ======================================

        seg_metrics = evaluate_segmentation(
            gt_seg,
            pred_seg,
            num_classes=num_classes
        )

        # ======================================
        # STORE
        # ======================================

        results.append({

            "Patient":
                patient,

            "MAE_HU":
                mae_hu,

            "MAE":
                syn_metrics["MAE"],

            "RMSE":
                syn_metrics["RMSE"],

            "PSNR":
                syn_metrics["PSNR"],

            "SSIM":
                syn_metrics["SSIM"],

            "Dice_HU300":
                dice_bone,

            "DSC":
                seg_metrics["DSC"]
        })

    # ======================================
    # SAVE
    # ======================================

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        output_csv,
        index=False
    )

    summary = df.mean(
        numeric_only=True
    )

    summary.to_frame(
        name="Mean"
    ).to_csv(
        output_summary
    )

    print("\nSummary")
    print(summary)


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True
    )

    parser.add_argument(
        "--predictions_dir",
        required=True
    )

    parser.add_argument(
        "--output_csv",
        default="results.csv"
    )

    parser.add_argument(
        "--output_summary",
        default="summary.csv"
    )

    parser.add_argument(
        "--num_classes",
        type=int,
        default=10
    )

    args = parser.parse_args()

    evaluate_folder(
        data_dir=args.data_dir,
        predictions_dir=args.predictions_dir,
        output_csv=args.output_csv,
        output_summary=args.output_summary,
        num_classes=args.num_classes
    )