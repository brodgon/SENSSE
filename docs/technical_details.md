<h1 align="center">SENSSE formulation</h1>

SENSSE (Simultaneous Evidential Network for Synthesis and Segmentation) is a multitask deep learning framework designed for adaptive radiotherapy. The network simultaneously performs image synthesis, anatomical segmentation and uncertainty quantification of each task within a single forward pass.

Unlike conventional pipelines where synthesis and segmentation are modeled independently, SENSSE jointly optimizes both tasks through shared representations and cross-task interactions.

Let $x \in \mathbb{R}^{H \times W \times K}$ represent an input image stack composed of $K$ neighboring slices. The network simultaneously predicts $\hat y_{syn}$ (synthetic image) and $\hat y_{seg}$ (segmentation map) corresponding to the central slice.

---
# Architectural Components
The model is composed of a shared encoder which extracts anatomical features from the input slices, followed by a bottlencek which bifurcates in two task independent decoders which finalize in evidential output heads.

## Shared Encoder: Feature extraction

The encoder constitutes the backbone of the SENSSE architecture and is responsible for extracting hierarchical feature representations from the input image. It is shared by both the synthesis and segmentation branches, allowing the network to learn a common feature space that captures anatomical structures, intensity distributions, and contextual information relevant to both tasks.

Structurally, the encoder is composed of a sequence of residual convolutional blocks. Each block contains two 3×3 convolutional layers followed by batch normalization and ReLU activations, together with identity skip connections that facilitate gradient propagation and improve training stability. After each residual block, a max-pooling operation reduces the spatial resolution while increasing the receptive field, enabling the network to progressively capture increasingly abstract representations.

Formally, given an input image $x \in \mathbb{R}^{H \times W \times K}$, where \(K\) corresponds to the number of input slices, the encoder generates a hierarchy of multi-scale feature maps $F^1, F^2, \ldots, F^L$, with progressively reduced spatial resolution and increased semantic complexity. Early feature maps primarily encode local image characteristics such as edges, textures, and fine anatomical details, whereas deeper representations capture larger anatomical structures and contextual relationships. 

These feature maps are subsequently used in two ways. First, the deepest representation is forwarded to the bottleneck and decoder branches. Second, intermediate feature maps are preserved through skip connections and later integrated into the decoders using attention-gated fusion mechanisms. By sharing encoder representations across both tasks, SENSSE encourages the learning of complementary anatomical and intensity-based features that support simultaneous image synthesis and segmentation.

---
## Bottleneck: Shared representation

The bottleneck constitutes the deepest layer of the SENSSE architecture. At this point, the network has progressively aggregated contextual information through successive encoder stages, enabling the extraction of high-level anatomical and structural features while reducing sensitivity to local noise and image artifacts.

Formally, the bottleneck receives the deepest encoder representation and produces a latent feature map, $F_B$, which acts as a shared representation for both synthesis and segmentation tasks. Unlike task-specific feature extraction schemes, SENSSE employs a common bottleneck that allows both decoders to access the same global anatomical context, encouraging the learning of complementary representations that are simultaneously informative for image synthesis and organ delineation. The bottleneck output is subsequently provided to both decoder branches.

---

## Decoders

SENSSE employs two parallel decoder branches. The decoder architecture follows the general philosophy of U-Net-like models, combining semantic information extracted at deep layers with fine anatomical details recovered through skip connections. To improve feature selection and spatial localization, skip connections are filtered using attention gates before being fused into the reconstruction pathway.

The two decoder branches are structurally similar but serve different objectives. The synthesis decoder focuses on recovering anatomically consistent intensity information for synthetic image generation, whereas the segmentation decoder emphasizes structural delineation. To exploit the complementary nature of these tasks, SENSSE introduces explicit decoder interaction mechanisms that enable information exchange between synthesis and segmentation pathways during feature reconstruction.

### Decoder Reconstruction

Starting from the bottleneck representation $F_B$, each decoder progressively increases the spatial resolution through a sequence of upsampling operations. At each reconstruction stage, decoder features are combined with the corresponding encoder representations through attention-gated skip connections, allowing the network to recover fine anatomical details that may have been lost during downsampling.

Let $F_{dec}^{(i+1)}$ represent decoder features at level \(i+1\). After upsampling, these features are merged with filtered encoder representations originating from the corresponding encoder stage, yielding the reconstructed decoder feature map $F_{dec}^{(i)}$. Through this hierarchical reconstruction process, the decoder gradually transforms highly abstract latent features into anatomically meaningful high-resolution representations suitable for image synthesis and segmentation.


### Attention-Gated Skip Connections

To improve spatial localization and reduce the propagation of irrelevant information, SENSSE employs attention-gated skip connections instead of directly concatenating encoder and decoder features.

Given encoder feature maps $x$ and decoder context features $g$, attention coefficients are computed as

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
),
```

where:

- \(W_x\) and \(W_g\) denote learnable linear projections,
- \(b\) is a bias term,
- \(\sigma(\cdot)\) represents the sigmoid activation function.

The resulting attention coefficients act as a soft spatial mask identifying encoder regions that are most relevant to the current decoding context. The gated feature representation is then obtained through element-wise multiplication,

```math
x'
=
\psi \odot x,
```

where &\odot& denotes element-wise multiplication.

The filtered representation is subsequently merged with the upsampled decoder features, allowing the reconstruction process to focus on anatomically relevant structures while suppressing noisy or redundant activations. This mechanism improves localization accuracy and contributes to more precise boundaries in both synthetic images and segmentation outputs.


### Decoder Interaction Mechanism

A key component of SENSSE is the explicit interaction between the synthesis and segmentation decoders. Unlike conventional multitask architectures, where information sharing is restricted to a common encoder and the reconstruction pathways remain independent, SENSSE enables feature exchange during decoding. This design is motivated by the observation that image synthesis and segmentation are intrinsically related tasks in adaptive radiotherapy and can therefore benefit from complementary information learned by one another.

The synthesis branch is responsible for reconstructing anatomically consistent image intensities, whereas the segmentation branch focuses on identifying and delineating anatomical structures. As synthesized image features contain rich tissue-specific intensity information, these representations can provide valuable contextual cues to improve segmentation accuracy. Consequently, SENSSE adopts a Synthesis-to-Segmentation (Syn2Seg) interaction strategy, in which features learned by the synthesis decoder are transferred to the segmentation branch at multiple reconstruction stages.

Let $F_{syn}^{(i)}$ and $F_{seg}^{(i)}$ denote the synthesis and segmentation feature maps at decoder level \(i\), respectively. At each reconstruction stage, synthesis features are concatenated with the corresponding segmentation representations and projected into a common feature space through a learnable \(1\times1\) convolution,

```math
F_{fusion}^{(i)}
=
Conv_{1\times1}
\Big(
F_{seg}^{(i)}
\oplus
F_{syn}^{(i)}
\Big),
```

where $\oplus$ denotes channel-wise concatenation and $Conv_{1\times1}$ corresponds to a learnable projection layer.

The resulting fused representation combines anatomical information extracted by the segmentation pathway with complementary intensity information generated by the synthesis branch. This interaction is applied independently at every decoder scale, enabling the segmentation decoder to exploit synthesis-derived information across multiple spatial resolutions. At shallow decoder levels, transferred features provide fine-grained local intensity patterns, whereas deeper layers contribute broader anatomical and contextual information.

By continuously enriching segmentation features with synthesis representations, the model promotes more accurate anatomical delineation and improved structural consistency between synthesized images and segmentation outputs. This strategy explicitly exploits the complementary nature of both tasks while maintaining a unified multitask learning framework.

Although the configuration adopted in this work corresponds to the Syn2Seg interaction mechanism, the SENSSE framework also supports alternative interaction strategies, including independent decoders, Segmentation-to-Synthesis (Seg2Syn), and bidirectional feature exchange. These variants are described and evaluated separately in the ablation studies.

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
