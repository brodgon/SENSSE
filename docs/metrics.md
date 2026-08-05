# Metrics

This document describes all evaluation metrics used to assess image synthesis, segmentation performance, uncertainty quality, calibration, and reliability within the SENSSE framework.

---

# 1. Notation

Let ```math y``` denote the ground-truth value.

```math
\hat y
```

denote the predicted value.

```math
N
```

represent the number of evaluated voxels.

For segmentation:

```math
A
```

denotes the set of voxels belonging to the ground-truth structure.

```math
B
```

denotes the set of voxels predicted by the model.

---

# 2. Image Synthesis Metrics

## 2.1 Mean Absolute Error (MAE)

MAE measures the average absolute error between synthesized and reference images.

```math
MAE
=
\frac{1}{N}
\sum_{i=1}^{N}
|y_i-\hat y_i|
```

Lower values indicate better image fidelity.

---

## 2.2 Root Mean Squared Error (RMSE)

RMSE penalizes larger reconstruction errors more heavily than MAE.

```math
RMSE
=
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
}
```

Lower values indicate better performance.

---

## 2.3 Peak Signal-to-Noise Ratio (PSNR)

PSNR quantifies image reconstruction quality relative to the maximum intensity value.

```math
PSNR
=
20
\log_{10}
(MAX_I)
-
10
\log_{10}
(MSE)
```

where

```math
MSE
=
\frac{1}{N}
\sum_{i=1}^{N}
(y_i-\hat y_i)^2
```

Higher values indicate better reconstruction quality.

---

## 2.4 Structural Similarity Index (SSIM)

SSIM evaluates image similarity considering luminance, contrast, and structural information.

```math
SSIM(x,y)
=
\frac{
(2\mu_x\mu_y+C_1)
(2\sigma_{xy}+C_2)
}{
(\mu_x^2+\mu_y^2+C_1)
(\sigma_x^2+\sigma_y^2+C_2)
}
```

where:

- μ denotes the mean intensity
- σ denotes standard deviation
- σxy denotes covariance

SSIM ranges from:

```math
[-1,1]
```

with

```math
1
```

representing perfect correspondence.

---

## 2.5 Multi-Scale SSIM (MS-SSIM)

MS-SSIM extends SSIM by computing similarity at multiple image resolutions.

```math
MS\text{-}SSIM
=
\prod_{m=1}^{M}
SSIM_m^{\alpha_m}
```

where

```math
M
```

denotes the number of scales.

Higher values indicate higher structural similarity.

---

## 2.6 Dice HU > 300

To assess anatomical consistency of high-density structures, binary masks are created by thresholding CT volumes at:

```math
300 HU
```

The overlap between reference and synthesized masks is computed via Dice Similarity Coefficient.

---

# 3. Segmentation Metrics

## 3.1 Dice Similarity Coefficient (DSC)

DSC measures overlap between predicted and ground-truth segmentations.

```math
DSC
=
\frac{
2|A\cap B|
}{
|A|+|B|
}
```

Values range from:

```math
0
```

(no overlap)

to

```math
1
```

(perfect overlap).

Higher values are better.

---

## 3.2 Hausdorff Distance (HD)

Hausdorff Distance measures the largest surface discrepancy between segmentations.

```math
HD(A,B)
=
\max
\Big(
\sup_{a\in A}
d(a,B),
\sup_{b\in B}
d(b,A)
\Big)
```

where

```math
d(a,B)
```

denotes the minimum Euclidean distance from point

```math
a
```

to set

```math
B
```

---

## 3.3 95th Percentile Hausdorff Distance (HD95)

Because HD is sensitive to outliers, HD95 is often preferred.

```math
HD95
=
95^{th}
\ percentile
\ of \ surface \ distances
```

Lower values indicate better boundary agreement.

---

## 3.4 Mean Surface Distance (MSD)

MSD measures the average distance between segmentation surfaces.

```math
MSD
=
\frac{
1
}{
|S_A|
}
\sum_{a\in S_A}
d(a,S_B)
```

where:

```math
S_A
```

and

```math
S_B
```

represent segmentation surfaces.

Lower values indicate better performance.

---

# 4. Calibration Metrics

Calibration evaluates whether confidence values correspond to actual accuracy.

---

## 4.1 Expected Calibration Error (ECE)

Predictions are grouped into

```math
M
```

confidence bins.

ECE is computed as:

```math
ECE
=
\sum_{m=1}^{M}
\frac{|B_m|}{N}
\left|
acc(B_m)-conf(B_m)
\right|
```

where:

- \(acc(B_m)\) is bin accuracy
- \(conf(B_m)\) is average confidence

Lower values indicate better calibration.

---

## 4.2 Maximum Calibration Error (MCE)

MCE measures the largest calibration mismatch.

```math
MCE
=
\max_m
\left|
acc(B_m)-conf(B_m)
\right|
```

Lower values are preferred.

---

# 5. Probabilistic Metrics

## 5.1 Negative Log-Likelihood (NLL)

NLL evaluates the probability assigned to the correct outcome.

For classification:

```math
NLL
=
-
\frac1N
\sum_{i=1}^{N}
\sum_{c=1}^{C}
y_{ic}
\log(p_{ic})
```

Lower values indicate better calibrated probabilistic predictions.

---

## 5.2 Brier Score

The Brier Score measures the squared error between probabilities and labels.

```math
BS
=
\frac1N
\sum_{i=1}^{N}
\sum_{c=1}^{C}
(p_{ic}-y_{ic})^2
```

Lower values are better.

---

# 6. Reliability Metrics

Reliability metrics evaluate whether uncertainty correctly identifies prediction errors.

---

## 6.1 Uncertainty-Error Overlap (UEO)

UEO quantifies overlap between regions of high uncertainty and high prediction error.

Let:

```math
U
```

be the set of voxels above a selected uncertainty percentile.

Let:

```math
E
```

be the set of voxels above an error percentile.

Then:

```math
UEO
=
\frac{
|U\cap E|
}{
|U\cup E|
}
```

Higher values indicate uncertainty is concentrated in error-prone regions.

---

## 6.2 Area Under Sparsification Error (AUSE)

AUSE evaluates how prediction error decreases as uncertain predictions are progressively removed.

Let:

```math
R(k)
```

be the residual error after removing the

```math
k
```

most uncertain predictions.

Then:

```math
AUSE
=
\int
R(k)
dk
```

Lower values indicate better uncertainty ranking.

---

## 6.3 Area Under Risk-Coverage Curve (AURC)

Risk-Coverage analysis examines the trade-off between retained predictions and error.

```math
AURC
=
\int
Risk(Coverage)
\,dCoverage
```

Lower values indicate better uncertainty quality.

---

# 7. Correlation Metrics

## 7.1 Pearson Correlation Coefficient

Pearson correlation evaluates linear association between uncertainty and prediction error.

```math
\rho
=
\frac{
Cov(U,E)
}{
\sigma_U \sigma_E
}
```

Values range from:

```math
-1
```

to

```math
1
```

Higher positive values indicate uncertainty increases with prediction error.

---

## 7.2 Spearman Rank Correlation

Spearman correlation evaluates monotonic relationships.

```math
\rho_s
=
1-
\frac{
6
\sum d_i^2
}{
n(n^2-1)
}
```

where:

```math
d_i
```

represents rank differences.

Higher values indicate stronger correspondence between uncertainty and error.

---

# 8. Interpretation Guide

| Metric | Direction |
|----------|-----------|
| MAE | ↓ Lower better |
| RMSE | ↓ Lower better |
| PSNR | ↑ Higher better |
| SSIM | ↑ Higher better |
| MS-SSIM | ↑ Higher better |
| Dice | ↑ Higher better |
| HD95 | ↓ Lower better |
| MSD | ↓ Lower better |
| ECE | ↓ Lower better |
| MCE | ↓ Lower better |
| NLL | ↓ Lower better |
| Brier | ↓ Lower better |
| UEO | ↑ Higher better |
| AUSE | ↓ Lower better |
| AURC | ↓ Lower better |
| Pearson | ↑ Higher better |
| Spearman | ↑ Higher better |

---

# References

The implementation of these metrics follows standard formulations commonly adopted in medical image analysis, uncertainty quantification, and adaptive radiotherapy literature.
