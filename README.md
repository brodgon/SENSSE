# SENSSE

**Simultaneous Evidential Network for Synthesis and Segmentation with Uncertainty Quantification**

SENSSE is a deep learning framework for the simultaneous generation of synthetic CT (sCT) images and organ-at-risk (OAR) segmentation masks from cone-beam CT (CBCT) images, while providing uncertainty estimates for both tasks.

The framework combines multitask learning, Evidential Deep Learning (EDL), and Normal-Inverse-Gamma (NIG) regression to jointly perform image synthesis, segmentation, and uncertainty quantification within a unified architecture.

<p align="center">
  docs/images/Fig2_Inference.png
</p>

---

# Overview

Adaptive radiotherapy relies on accurate image synthesis and segmentation for treatment monitoring and dose recalculation. Traditionally, these tasks are modeled independently and often lack reliable uncertainty information.

SENSSE addresses these limitations by:

- Performing CBCT-to-CT synthesis and OAR segmentation simultaneously.
- Modeling uncertainties for both tasks.
- Leveraging evidential learning instead of computationally expensive sampling-based approaches.
- Supporting configurable 2D and 2.5D inputs.
- Providing a modular framework for training, inference, evaluation, and uncertainty analysis.

The architecture consists of a shared encoder and two task-specific decoders connected through configurable interaction mechanisms, enabling information exchange between synthesis and segmentation pathways.

---

# Scope and Objectives

The repository aims to provide a reproducible framework for:

- Simultaneous CBCT-to-CT synthesis and OAR segmentation.
- Uncertainty-aware medical image analysis.
- Investigation of evidential learning methods for image synthesis and segmentation.
- Development of adaptive radiotherapy workflows.


---

# Repository Structure

```text
SENSSE/

├── configs/
│   └── example_config.yaml
│
├── evidential_deep_learning/
│
├── experiments/
│   ├── ablation_connections.py
│   ├── ablation_regularization.py
│
├── sensse/
│   ├── datasets/
│   ├── losses/
│   ├── metrics/
│   ├── models/
│   ├── uncertainty/
│   └── utils/
│
├── train.py
├── inference.py
├── evaluate.py
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

# Quick Start

## Training

```bash
python train.py \
    --config configs/example_config.yaml
```

## Inference

```bash
python inference.py \
    --config configs/example_config.yaml
```

---

# Documentation

Detailed documentation is provided in the `docs/` folder.

| Document | Description |
|-----------|-------------|
| `dataset_preparation.md` | Dataset structure, preprocessing and normalization |
| `technical_details.md` | Architecture, losses and implementation details |
| `metrics.md` | Mathematical definitions of all evaluation metrics |
| `results.md` | Summary of experimental results and ablation studies |
| `faq.md` | Frequently asked questions |

---

# Model Highlights

- Simultaneous synthesis and segmentation.
- Evidential Deep Learning for uncertainty-aware segmentation.
- Normal-Inverse-Gamma regression for uncertainty-aware synthesis.
- Attention-gated dual-decoder architecture.
- Configurable decoder interaction mechanisms:
  - None
  - Syn2Seg
  - Seg2Syn
  - Bidirectional
- Configurable 2D and 2.5D inputs through the `num_slices` parameter.

---

# Citation

If you use SENSSE in your research, please cite:

The citation will be updated once the manuscript is formally published.

---

# Acknowledgements

This work was developed within the PROMISE research group at Universidad Rey Juan Carlos (URJC).

---

# License

This project is distributed under the license included in the `LICENSE` file.
