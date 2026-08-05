<h1 align="center">Dataset Preparation</h1>

This document describes the recommended dataset structure and preprocessing steps required to train and evaluate SENSSE.

---

# Overview

SENSSE is modality-agnostic and can be used with different medical imaging modalities. However, all input images, target images, and segmentation masks must be spatially aligned and share a consistent image geometry.

---

# Data Requirements

Each training sample should contain:

- An input image (e.g., CBCT, MR, PET).
- A target image for synthesis (e.g., CT).
- A segmentation mask.
- Consistent image geometry across all modalities.

Recommended format:

```text
NIfTI (.nii.gz)
```

Although additional formats can be supported, NIfTI is recommended because it preserves voxel spacing, orientation, and image metadata.

---

# Recommended Dataset Structure

The repository assumes the following structure:

```text
DATASET/

├── Patient001/
│   └── Session1/
│       ├── CBCT.nii.gz
│       ├── CT.nii.gz
│       └── mask.nii.gz
│
├── Patient002/
│   └── Session1/
│       ├── CBCT.nii.gz
│       ├── CT.nii.gz
│       └── mask.nii.gz
│
└── ...
```

Where:

- `CBCT.nii.gz` corresponds to the network input.
- `CT.nii.gz` corresponds to the synthesis target.
- `mask.nii.gz` contains segmentation labels.

---

# Custom Dataset Organization

Users can freely adopt a different directory structure.

If a different structure is used, the data loading functions located in:

```text
sensse/datasets/
```

must be adapted accordingly, particularly:

```text
loaders.py
generators.py
```

The network itself is independent of the dataset organization and only requires access to:

- Input image volume.
- Target image volume.
- Segmentation mask volume.

---

# Spatial Preprocessing

Before training, all images should satisfy the following conditions.

## Registration

Input images and target images should be spatially aligned.

For example:

```text
CBCT ↔ CT
MR ↔ CT
PET ↔ CT
```

Misregistration can negatively impact both synthesis and segmentation performance.

Rigid or deformable registration may be used depending on the application.

---

## Consistent Geometry

All modalities should share:

- Identical voxel dimensions.
- Identical image orientation.
- Identical field of view (FOV).
- Identical matrix size.

For example:

```text
CBCT: 512 × 512 × 275
CT:   512 × 512 × 275
Mask: 512 × 512 × 275
```

The loader assumes corresponding voxels represent the same anatomical location.

---

## Cropping and Resampling

If necessary:

1. Resample all modalities to a common voxel spacing.
2. Crop or pad images to a common field of view.
3. Verify that image dimensions and orientations match.

---

# Normalization

Normalization is strongly recommended to ensure stable training.

The recommended strategy depends on the image modality.

---

## CT and CBCT

For Hounsfield Unit (HU)-based modalities:

### Step 1

Clip the intensity values to a clinically meaningful range.

Example:

```text
[-1024, 3071] HU
```

### Step 2

Apply min-max normalization:

```text
normalized = (x - HUmin) / (HUmax - HUmin)
```

Resulting range:

```text
[0,1]
```

This is the normalization strategy used in the SENSSE manuscript.

---

## MR Images

For MRI data, scanner-specific intensity scales make global HU-like normalization inappropriate.

A common strategy is:

1. Normalize each volume independently.
2. Use min-max normalization or z-score normalization.

Example:

```text
normalized = (x - min(x)) / (max(x) - min(x))
```

performed separately for every volume.

---

# Segmentation Masks

Segmentation masks should contain integer labels.

Example:

```text
0 -> Background
1 -> Brainstem
2 -> Spinal Cord
3 -> Parotid Left
...
```

Labels must be consistent across all patients.

---

# 2.5D Inputs

SENSSE supports both conventional 2D training and configurable 2.5D inputs.

The number of neighboring slices is controlled by:

```yaml
num_slices: 5
```

Examples:

```yaml
num_slices: 1
```

Standard 2D input.

```yaml
num_slices: 5
```

Two neighboring slices on each side plus the central slice.

This configuration was used in the manuscript.

Input size:

```text
(H, W, 5)
```

---

# Quality Control Checklist

Before training, verify:

- [ ] NIfTI format.
- [ ] Input and target images are registered.
- [ ] Images share the same voxel space.
- [ ] Images share the same field of view.
- [ ] Images have identical dimensions.
- [ ] CT intensities are clipped and normalized.
- [ ] Masks contain consistent label definitions.
- [ ] No missing files exist.
