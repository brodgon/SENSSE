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

Formally, given an input image $x \in \mathbb{R}^{H \times W \times K}$, where $K$ corresponds to the number of input slices, the encoder generates a hierarchy of multi-scale feature maps $F^1, F^2, \ldots, F^L$, with progressively reduced spatial resolution and increased semantic complexity. Early feature maps primarily encode local image characteristics such as edges, textures, and fine anatomical details, whereas deeper representations capture larger anatomical structures and contextual relationships. 

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

Let $F_{dec}^{(i+1)}$ represent decoder features at level $i+1$. After upsampling, these features are merged with filtered encoder representations originating from the corresponding encoder stage, yielding the reconstructed decoder feature map $F_{dec}^{(i)}$. Through this hierarchical reconstruction process, the decoder gradually transforms highly abstract latent features into anatomically meaningful high-resolution representations suitable for image synthesis and segmentation.


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

- $W_x$ and $W_g$ denote learnable linear projections,
- $b$ is a bias term,
- $\sigma(\cdot)$ represents the sigmoid activation function.

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

Let $F_{syn}^{(i)}$ and $F_{seg}^{(i)}$ denote the synthesis and segmentation feature maps at decoder level $i$, respectively. At each reconstruction stage, synthesis features are concatenated with the corresponding segmentation representations and projected into a common feature space through a learnable $1\times1$ convolution,

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

## Evidential Output Heads

A fundamental component of SENSSE is its ability to estimate uncertainty jointly with image synthesis and segmentation. Rather than producing only deterministic predictions, both task-specific output heads are designed to infer a predictive distribution from which uncertainty information can be directly derived. This allows SENSSE to provide voxel-wise confidence estimates alongside its primary outputs while maintaining a single forward-pass inference procedure.

The synthesis and segmentation branches rely on different evidential formulations due to the distinct nature of their prediction targets. Image synthesis is formulated as a continuous regression problem and therefore employs Deep Evidential Regression through a Normal-Inverse-Gamma (NIG) distribution. In contrast, segmentation is treated as a multi-class classification problem and utilizes Evidential Deep Learning (EDL), where class evidence is modeled through a Dirichlet distribution.

## Synthesis Head

The synthesis branch aims to predict a synthetic image together with an estimate of the uncertainty associated with each voxel intensity. Instead of directly regressing a single intensity value, the network predicts the parameters of a NIG distribution,

```math
(\mu,\lambda,\alpha,\beta),
```

which define a probability distribution over possible voxel intensities.

The predictive distribution can therefore be written as

```math
p(y)
=
NIG(y \mid \mu,\lambda,\alpha,\beta),
```

where $y$ denotes the target voxel intensity. Within this formulation, the parameter $\mu$ corresponds to the predictive mean and represents the synthesized intensity value, 

```math
\mathbb E[y]
=
\mu.
```

By predicting an entire evidential distribution instead of a single value, the network can explicitly model uncertainty together with the reconstruction itself.

The predictive variance associated with each voxel is given by

```math
Var(y)
=
\frac{\beta}
     {\lambda(\alpha-1)},
\qquad
\alpha > 1.
```

In addition to the predictive variance, Deep Evidential Regression allows uncertainty to be decomposed into aleatoric and epistemic components.

Aleatoric uncertainty captures intrinsic ambiguity present in the imaging data, such as image noise, motion artifacts, or poorly defined anatomical boundaries, and is computed as

```math
u_{ale}= \frac{\beta}{\alpha-1}.
```

Epistemic uncertainty reflects uncertainty in the model parameters and is typically associated with underrepresented anatomical configurations or out-of-distribution samples. It is defined as

```math
u_{epi} = \frac{\beta}{\lambda(\alpha-1)}.
```

The total predictive uncertainty is therefore obtained as

```math
u_{tot} = u_{ale} + u_{epi}.
```


Training is performed using the Deep Evidential Regression objective. The primary component corresponds to the negative log-likelihood of the NIG distribution,

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
\alpha\log\beta.
```

Minimizing this objective encourages accurate image synthesis while simultaneously learning meaningful predictive uncertainty.

A challenge in evidential learning is preventing the model from producing highly confident predictions when large errors are present. To address this issue, an evidential regularization term is incorporated,

```math
R_{evid}
=
|\mu-y|
(2\alpha+\lambda),
```

which penalizes unsupported confidence and encourages uncertainty to increase whenever reconstruction errors become large.

The final synthesis objective is therefore defined as

```math
L_{syn}
=
L_{NLL}
+
\eta_{evid}
R_{evid},
```

where $\eta_{evid}$ controls the strength of the evidential regularization term.

Through this formulation, the synthesis branch simultaneously produces synthetic CT predictions, predictive variances and uncertainty estimates within a single forward pass.

---

### Segmentation Head

While the synthesis task operates in a continuous regression setting, segmentation requires classification of every voxel into one of $C$ anatomical classes. Rather than directly predicting class probabilities, the segmentation head adopts the Evidential Deep Learning framework, which learns the amount of evidence supporting each class.

For each voxel, the model outputs a non-negative evidence vector,

```math
e
=
[e_1,e_2,\ldots,e_C].
```

Evidence values are transformed into Dirichlet concentration parameters through

```math
\alpha
=
e + 1.
```

The resulting Dirichlet distribution provides a probabilistic representation of class assignments and enables uncertainty estimation directly from the network outputs. The overall evidence strength is defined as

```math
S
=
\sum_{c=1}^{C}
\alpha_c,
```

which represents the total amount of support available for the classification decision. Expected class probabilities are obtained as

```math
\hat{p}_c
=
\frac{\alpha_c}{S}.
```

Unlike conventional softmax probabilities, these quantities are accompanied by an explicit measure of uncertainty derived from the evidence itself.

Segmentation uncertainty is computed as

```math
u_{seg}
=
\frac{C}{S}.
```

A high evidence strength corresponds to low uncertainty, whereas insufficient evidence yields larger uncertainty values. Consequently, uncertainty naturally emerges from the learned evidence representation rather than requiring additional post-processing procedures.

In addition to a global uncertainty measure, class-specific uncertainty can also be computed through the Dirichlet variance,

```math
Var[p_c]
=
\frac{
\alpha_c(S-\alpha_c)
}{
S^2(S+1)
}.
```

Training follows the Evidential Deep Learning formulation based on the Expected Squared Error (ESE) objective,

```math
L_{ESE}
=
\sum_{c=1}^{C}
(y_c-\hat p_c)^2
+
\sum_{c=1}^{C}
Var[p_c].
```

The first term penalizes classification errors, while the second explicitly incorporates predictive uncertainty and discourages overconfident predictions.

To further regularize the evidence distribution, a Kullback-Leibler divergence term is introduced between the predicted Dirichlet distribution and a non-informative uniform prior,

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
\right].
```

This regularization encourages uncertain predictions whenever the evidence supporting a decision is insufficient.

The resulting evidential segmentation loss becomes

```math
L_{EDL}
=
L_{ESE}
+
\lambda
KL[\alpha||1].
```

To avoid excessive regularization during the initial stages of training, the contribution of the KL divergence is progressively increased through a warmup schedule,

```math
\lambda(t)
=
\lambda_{max}
\cdot
\min
\left(
1,
\frac{t}{T}
\right),
```

where $t$ denotes the current epoch and $T$ corresponds to the warmup duration.

Finally, an auxiliary cross-entropy objective is combined with the evidential formulation,

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
),
```

yielding the final segmentation loss,

```math
L_{seg}
=
L_{CE}
+
L_{EDL}.
```

By combining evidence learning, probabilistic reasoning, and uncertainty-aware regularization, the segmentation head produces anatomically meaningful predictions together with voxel-wise confidence estimates that can be directly incorporated into downstream clinical decision-making workflows.

---

# References

For complete theoretical background please refer to:

- Sensoy et al. (2018), Evidential Deep Learning
- Amini et al. (2020), Deep Evidential Regression
- Oktay et al. (2018), Attention U-Net
- Rodriguez-Gonzalez et al. (2026), ENSE
- Rodriguez-Gonzalez et al. (2026), SENSSE
