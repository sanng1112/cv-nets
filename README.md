# CV-Nets: Advanced Computer Vision Neural Network Framework

<div align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Status-Research%20Ready-success?style=for-the-badge"/>
</div>

## Overview
**CV-Nets** is a highly modular, advanced computer vision framework built on top of PyTorch. It is engineered to provide an extensible research environment for deploying state-of-the-art neural network architectures and mathematical methodologies, specifically targeting known bottlenecks in modern deep learning such as Neural Collapse, Rank Collapse, and Oversmoothing.

## Key Features & Architecture

### 1. Neural Network Components (`layers/` & `blocks/`)
The framework implements numerous specialized layers designed to combat standard architectural deficiencies:
- **Inverted Nonlinear Linear Attention (INLA):** A highly optimized attention mechanism integrating Information Bottleneck theories and Layer Normalization.
- **Simplex ETF Classifier:** A frozen classifier utilizing Equiangular Tight Frames (ETF) to natively enforce Neural Collapse and prevent overfitting in the final linear layers.
- **Spectral Scaled GCN:** A Graph Convolutional Network layer incorporating singular value capping via SVD to mitigate oversmoothing in deep feature propagation.
- **Variational Information Bottleneck:** A stochastic layer enforcing Gaussian noise sampling to reduce mutual information redundancy.

### 2. Comprehensive Loss Registry (`loss_fn/`)
A unified registry system supporting a wide array of objective functions across multiple domains:
- **Classification:** Cross Entropy, Focal Loss, PolyLoss.
- **Object Detection:** Smooth L1, GIoU, DIoU, CIoU.
- **Segmentation:** Dice Loss, Jaccard Index, Tversky Loss.
- **Metric Learning:** Triplet Margin, Contrastive Loss.

### 3. Optimization & Scheduling (`optim/`)
- **Weight Decay Decoupling:** Automated isolation of bias terms and normalization parameters from weight decay to preserve statistical distributions.
- **Native Sequential Schedulers:** Robust integrations of Linear Warmup paired with Cosine Annealing decay.

### 4. Robust Training Engine (`engine/`)
The `BaseTrainer` class provides an enterprise-grade training loop, fully supporting:
- **Automatic Mixed Precision (AMP):** For accelerated FP16 computation on supported hardware.
- **Gradient Accumulation:** Facilitating large effective batch sizes under constrained VRAM environments.
- **Model EMA (Exponential Moving Average):** Shadow weight smoothing for highly stable validation metrics.
- **Cloud Logging (WandB):** Real-time dashboard tracking via Weights & Biases.
- **Pre-Training Sanity Checks:** Automated diagnostics ensuring optimal graph connectivity and VRAM safety prior to execution.

### 5. Infinite Backbone Support (`models/builder.py`)
Seamless integration with `timm` (PyTorch Image Models):
- Instantly instantiate over 1000+ state-of-the-art backbones (ConvNeXt, Swin, MaxViT, ResNet) using `CVNetModel` builder.
- The builder automatically detaches legacy classifiers and dynamically attaches any of the 9 specialized `cv-nets` heads.

### 6. Edge Deployment & Quantization (`engine/`)
From research to production in a single line of code:
- **`engine/quantization.py`**: Convert networks to `FP16` / `BF16` (Half/Brain Precision) or `INT8` (Dynamic Quantization) to reduce VRAM footprint by up to 75% while boosting inference speeds.
- **`engine/exporter.py`**: Export trained weights to industry-standard formats including `ONNX` and `TorchScript` for seamless integration into TensorRT, OpenVINO, C++, or Mobile platforms.

## Installation

### Prerequisites
- Python 3.10 or higher.
- A CUDA-capable GPU (highly recommended for training).

### Setup via `uv`
The project leverages `uv` and `hatchling` for rapid dependency resolution and isolated environment management.

```bash
# 1. Clone the repository
git clone https://github.com/sanng1112/cv-nets.git
cd cv-nets

# 2. Synchronize dependencies
uv sync
```

## Quick Start

The framework is strictly driven by configuration files.

1. Review `CONFIG_GUIDE.md` for proper configuration formats.
2. Execute the sanity checker and integration tests:

```bash
uv run python scripts/test_trainer.py
```

## Documentation & Research Notes
Extensive translations and mathematical analyses of foundational papers implemented in this repository can be found under the `papers/phan_tich/` directory.

---
**Maintained by anng**
