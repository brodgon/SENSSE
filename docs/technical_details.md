# Technical Details

This document provides a detailed description of the SENSSE framework, including architectural design choices, evidential learning formulations, uncertainty quantification methodology, training objectives, and implementation details.

---

# 1. Introduction

SENSSE (Simultaneous Evidential Network for Synthesis and Segmentation) is a multitask deep learning framework designed for adaptive radiotherapy.

The network simultaneously performs:

- Image synthesis
- Anatomical segmentation
- Uncertainty quantification

within a single forward pass.

Unlike conventional pipelines where synthesis and segmentation are modeled independently, SENSSE jointly optimizes both tasks through shared representations and cross-task interactions.

---

# 2. Overview

Let

```math
x \in \mathbb{R}^{H \times W \times K}
```

represent an input image stack composed of:

```math
K
```

neighboring slices.

The network predicts:

```math
\hat y_{syn}
```

Synthetic image

and

```math
\hat y_{seg}
```

Segmentation map

corresponding to the central slice.

---

# 3. Architecture

The architecture is composed of:

1. Shared encoder
2. Bottleneck
3. Synthesis decoder
4. Segmentation decoder
5. Evidential output heads

---

# 4. Shared Encoder

The encoder consists of residual convolutional blocks.

Each block contains:

- Two 3×3 convolutions
- Batch normalization
- ReLU activations
- Identity skip connections

The encoder progressively extracts:

```math
F^1,F^2,\ldots,F^L
```

multi-scale feature maps.

These representations are shared across both tasks.

---

# 5. Bottleneck

The bottleneck represents the lowest spatial resolution and highest semantic abstraction level.

No downsampling is applied.

The bottleneck output

```math
F_B
```

is simultaneously provided to both decoders.

---

# 6. Attention-Gated Skip Connections

To improve spatial localization, skip connections are filtered through attention gates.

Given encoder features:

```math
x
```

and decoder context:

```math
g
```

attention coefficients are computed as

```math
\psi
=
\sigma
(
W_x x
+
W_g g
+
b
)
```

where:

- σ denotes the sigmoid activation function
- \(W_x\) and \(W_g\) represent learnable linear projections

The filtered skip connection is

```math
x'
=
\psi \odot x
```

where

```math
\odot
```

denotes element-wise multiplication.

---

# 7. Decoder Interaction Mechanism

A key contribution of SENSSE is the explicit interaction between synthesis and segmentation tasks.

At each decoder level, synthesis feature maps are transferred to the segmentation decoder.

Let:

```math
F_{syn}^{(i)}
```

and

```math
F_{seg}^{(i)}
```

be the synthesis and segmentation feature maps at scale

```math
i
```

respectively.

Features are fused as

```math
F_{fusion}^{(i)}
=
Conv_{1\times1}
\Big(
F_{seg}^{(i)}
\oplus
F_{syn}^{(i)}
\Big)
```

where

```math
\oplus
```

denotes channel concatenation.

This interaction promotes anatomical consistency between both tasks.

---

# 8. Evidential Image Synthesis

## 8.1 Normal-Inverse-Gamma Distribution

The synthesis branch predicts four parameters for every voxel:

```math
(\mu,\lambda,\alpha,\beta)
```

The predictive distribution follows a Normal-Inverse-Gamma (NIG) model:

```math
p(y)
=
NIG(y|\mu,\lambda,\alpha,\beta)
```

---

## 8.2 Predictive Mean

The predicted intensity corresponds to:

```math
\mathbb E[y]
=
\mu
```

---

## 8.3 Predictive Variance

The predictive variance is

```math
Var(y)
=
\frac{\beta}
     {\lambda(\alpha-1)}
\qquad \alpha > 1
```

This quantity combines both uncertainty sources:

- Aleatoric uncertainty
- Epistemic uncertainty

---

## 8.4 Negative Log-Likelihood

The NIG negative log-likelihood is

```math
L_{NLL}
=
\frac12
\log(\pi v)
+
\frac{(\mu-y)^2}{2v}
+
\frac{\alpha}{\beta}
\left(
\frac{(\mu-y)^2}{2}
+
\frac1{2v}
\right)
+
\log\Gamma(\alpha)
-
\alpha\log\beta
```

---

## 8.5 Evidential Regularization

To prevent overconfident predictions:

```math
R_{evid}
=
|\mu-y|
(2\alpha+\lambda)
```

---

## 8.6 Synthesis Loss

The final synthesis objective becomes

```math
L_{syn}
=
L_{NLL}
+
\eta_{evid}
R_{evid}
```

where

```math
\eta_{evid}
```

controls regularization strength.

---

# 9. Evidential Segmentation

## 9.1 Evidence Representation

Instead of predicting class probabilities directly, the network predicts non-negative evidence:

```math
e=
[e_1,e_2,\ldots,e_C]
```

---

## 9.2 Dirichlet Parameters

The evidence is converted into Dirichlet parameters:

```math
\alpha=e+1
```

---

## 9.3 Dirichlet Strength

The total evidence strength is

```math
S
=
\sum_{c=1}^{C}
\alpha_c
```

---

## 9.4 Expected Probabilities

Expected class probabilities are

```math
\hat p_c
=
\frac{\alpha_c}{S}
```

---

## 9.5 Predictive Uncertainty

Global segmentation uncertainty is

```math
u_{seg}
=
\frac{C}{S}
```

Higher evidence implies lower uncertainty.

---

## 9.6 Classwise Variance

Class-specific uncertainty is

```math
Var[p_c]
=
\frac{
\alpha_c(S-\alpha_c)
}{
S^2(S+1)
}
```

---

# 10. Expected Squared Error Loss

Given a one-hot target:

```math
y
```

the Expected Squared Error loss becomes

```math
L_{ESE}
=
\sum_{c=1}^{C}
(y_c-\hat p_c)^2
+
\sum_{c=1}^{C}
Var[p_c]
```

The first term penalizes classification error.

The second term incorporates predictive uncertainty.

---

# 11. Dirichlet Regularization

The Dirichlet prediction is regularized using the KL divergence to a uniform prior.

```math
KL[\alpha||1]
=
\log
\left(
\frac{
\Gamma(S)
}{
\prod_{c=1}^{C}
\Gamma(\alpha_c)
}
\right)
+
\sum_{c=1}^{C}
(\alpha_c-1)
\left[
\psi(\alpha_c)-\psi(S)
\right]
```

where:

- Γ denotes the Gamma function
- ψ denotes the Digamma function

---

# 12. Evidential Segmentation Loss

The evidential objective is

```math
L_{EDL}
=
L_{ESE}
+
\lambda
KL[\alpha||1]
```

---

# 13. KL Warmup

To avoid excessive regularization during early training:

```math
\lambda(t)
=
\lambda_{max}
\cdot
\min
\left(
1,
\frac{t}{T}
\right)
```

where:

- t denotes current epoch
- T denotes warmup length

---

# 14. Auxiliary Cross-Entropy

An additional cross-entropy term is included:

```math
L_{CE}
=
-
\frac1{HW}
\sum_{i=1}^{H}
\sum_{j=1}^{W}
\sum_{c=1}^{C}
y_{ijc}
\log
(
\hat p_{ijc}
)
```

---

# 15. Segmentation Loss

The final segmentation objective is

```math
L_{seg}
=
L_{CE}
+
L_{EDL}
```

---

# 16. Multitask Objective

The network is trained end-to-end through a combined loss:

```math
L_{total}
=
L_{syn}
+
L_{seg}
```

This formulation encourages the encoder to learn representations simultaneously informative for synthesis and segmentation.

---

# 17. 2.5D Input Strategy

SENSSE supports both 2D and 2.5D training.

Input tensors are:

```math
x
\in
\mathbb R^{H\times W\times K}
```

where:

```math
K
```

is an odd number.

The manuscript configuration uses:

```yaml
num_slices: 5
```

corresponding to:

```text
i-2
i-1
i
i+1
i+2
```

The central slice is used as prediction target.

---

# 18. Uncertainty Quantification

SENSSE estimates uncertainty in a single forward pass.

## Synthesis

Predicted from:

```math
(\mu,\lambda,\alpha,\beta)
```

via NIG statistics.

Provides:

- Predictive mean
- Predictive variance
- Aleatoric uncertainty
- Epistemic uncertainty

---

## Segmentation

Predicted from:

```math
\alpha
```

via Dirichlet statistics.

Provides:

- Expected probabilities
- Total uncertainty
- Classwise uncertainty

---

# 19. Implementation Details

Framework:

```text
TensorFlow 2.12
```

Optimizer:

```text
AdamW
```

Learning rate:

```text
1e-4
```

Batch size:

```text
4
```

Training epochs:

```text
150
```

Paper configuration:

```yaml
num_slices: 5
interaction: syn2seg
```

---

# 20. References

For complete theoretical background please refer to:

- Sensoy et al. (2018), Evidential Deep Learning
- Amini et al. (2020), Deep Evidential Regression
- Oktay et al. (2018), Attention U-Net
- Rodriguez-Gonzalez et al. (2026), SENSSE
