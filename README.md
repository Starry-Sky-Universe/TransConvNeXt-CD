<div align="center">

# 🛰️ TransConvNeXt-CD

**Transformer-CNN Hybrid Network for Remote Sensing Change Detection**  

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny-success)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-LEVIR--CD-ff6b6b)](https://justchenhao.github.io/LEVIR/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![GitHub Stars](https://img.shields.io/badge/dynamic/json?logo=github&label=Stars&color=yellow&query=stargazers_count&suffix=%20&url=https%3A%2F%2Fapi.github.com%2Frepos%2FStarry-Sky-Universe%2FTransConvNeXt-CD)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)
[![Kaggle](https://img.shields.io/badge/Trained%20on-Kaggle%20P100-20BEFF?logo=kaggle)](https://kaggle.com)
[![F1](https://img.shields.io/badge/F1%20Score-90.85%25-brightgreen)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)
[![IoU](https://img.shields.io/badge/IoU-83.24%25-success)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

---

**English | [简体中文](README_CN.md)**

</div>

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Dataset: LEVIR-CD](#-dataset-levir-cd)
- [Results](#-results)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Training Details](#-training-details)
- [Citation](#-citation)
- [License](#-license)

---

## 🎯 Overview

**TransConvNeXt-CD** is a state-of-the-art change detection model for **remote sensing imagery**. It combines **ConvNeXt-Tiny** (modern CNN backbone) with **Cross-Attention Transformer** modules to capture both local spatial details and global semantic dependencies between bi-temporal satellite images.

### ✨ Key Features

- **ConvNeXt-Tiny Backbone** — Modernized CNN with excellent feature extraction capabilities
- **Cross-Attention Block** — Captures long-range dependencies between T1 and T2 temporal images
- **Spatial Difference Module** — Multi-level feature interaction for change localization
- **Deep Supervision** — Auxiliary loss at decoder stages for more stable training
- **8x Test-Time Augmentation** — D4 dihedral group (4 rotations × 2 flips) for robust inference
- **Mixed Precision Training** — FP16 training for faster execution with less GPU memory
- **Hybrid Loss Function** — BCE + Dice loss for precise pixel-level segmentation
- **Fully Automated Kaggle Pipeline** — Auto-detects dataset and weight paths

---

## 🏗️ Architecture

The model follows an **encoder-interaction-decoder** paradigm:

```
T1 Image ──┐                    ┌── SpatialDiff(96) ── Decoder 4 ──┐
           ├── ConvNeXt ─── Stage1 ─── SpatialDiff(192) ── Decoder 3 ─┤
T2 Image ──┘    Tiny     ─── Stage2 ─── SpatialDiff(384) ── Decoder 2 ─┤
                          ─── Stage3 ──── CrossAttention ───── Decoder 1 ─┤
                          ─── Stage4 ──── Fusion ────────────────────────┴── Change Map
```

### Component Details

| Module | Description |
|--------|------------|
| **Encoder** | ConvNeXt-Tiny (4 stages, channels: 96 → 192 → 384 → 768) |
| **Spatial Difference** | `Conv2d(B N ReLU Conv2d)` — learns change features via channel-wise concatenation |
| **Cross-Attention** | `MultiheadAttention(12 heads, dim=768)` with LayerNorm + FFN residual |
| **Decoder** | 4-level bilinear upsampling with SE attention and skip connections |
| **Deep Supervision** | Auxiliary heads at decoder stages 3 & 4 for gradient flow |
| **Output** | Single-channel change probability map (1024 × 1024) |

### Attention Mechanism

The **Cross-Attention Block** enables bidirectional information flow between temporal features:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

Where T1 features serve as Q/K/V attending to T2 features, and vice versa, enabling the model to learn **what changed** and **where it changed**.

---

## 📊 Dataset: LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) is a large-scale benchmark dataset for remote sensing building change detection.

### Dataset Statistics

| Property | Value |
|----------|-------|
| Image Size | 1024 × 1024 pixels |
| Spatial Resolution | 0.5 m/pixel |
| Total Image Pairs | 637 |
| Training Set | 445 pairs |
| Validation Set | 64 pairs |
| Testing Set | 128 pairs |
| Change Types | Building appearance, demolition, alteration |
| Time Range | 2002 – 2018 |
| Source | Google Earth |

### Sample Images

```
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   Image T1  │   Image T2  │    Label    │  Prediction  │
├─────────────┼─────────────┼─────────────┼─────────────┤
│  (pre-change)│ (post-change)│ (ground truth)│  (model output)│
└─────────────┴─────────────┴─────────────┴─────────────┘
```

> Dataset Paper: *H. Chen, et al. "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection."*

---

## 📈 Results

### 🔬 Test Set Performance (8x TTA on 128 test pairs)

| Metric | Score |
|--------|:-----:|
| **IoU** (Intersection over Union) | **83.24%** |
| **F1 Score** | **90.85%** |
| **Precision** | **91.89%** |
| **Recall** | **89.84%** |

### 📉 Training Progress

| Epoch | Val F1 | Val IoU | Val Precision | Val Recall |
|:-----:|:------:|:-------:|:-------------:|:----------:|
| 1 | 0.4856 | 0.3206 | 0.3349 | 0.8828 |
| 5 | 0.8044 | 0.6727 | 0.7838 | 0.8260 |
| 10 | 0.8547 | 0.7462 | 0.7958 | 0.9229 |
| 20 | 0.9035 | 0.8239 | 0.9036 | 0.9033 |
| 50 | 0.9050 | — | — | — |
| **150 (Best)** | **0.9078** | — | — | — |
| **Test (8x TTA)** | **0.9085** | **0.8324** | **0.9189** | **0.8984** |

> Training details: Tesla P100 GPU, 150 epochs, CosineAnnealingWarmRestarts scheduler, BCE+Dice loss.

### 🖼️ Qualitative Results

![Test Results Visualization](assets/results.png)

*Sample change detection results on the LEVIR-CD test set. From left to right: pre-change image (T1), post-change image (T2), ground truth change map, predicted change map with per-sample metrics.*

---

## 🚀 Quick Start

### Prerequisites

```bash
# Clone repository
git clone https://github.com/Starry-Sky-Universe/TransConvNeXt-CD.git
cd TransConvNeXt-CD

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Download **LEVIR-CD** from one of these sources:
   - [Official Website](https://justchenhao.github.io/LEVIR/)
   - [Kaggle Dataset](https://www.kaggle.com/datasets/balraj98/levir-cd)

2. Organize the dataset as follows:
```
/path/to/LEVIR/
├── train/
│   ├── A/          # Pre-change images (1024×1024)
│   ├── B/          # Post-change images (1024×1024)
│   └── label/      # Change labels (binary)
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### Training

```bash
# Basic training (from scratch)
python src/train.py --data /path/to/LEVIR

# Custom training configuration
python src/train.py \
    --data /path/to/LEVIR \
    --save checkpoints/best_model.pth \
    --epochs 150 \
    --batch_size 8 \
    --lr 2e-4
```

### Evaluation

```bash
# Evaluate with 8x TTA
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth

# Evaluate and visualize
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth \
    --visualize \
    --output results.png
```

### Jupyter Notebooks

For interactive experimentation:

```bash
jupyter notebook notebooks/train_model.ipynb   # Training pipeline
jupyter notebook notebooks/test_inference.ipynb # Testing & visualization
```

> Note: The notebooks include automated Kaggle path detection — they can run directly in Kaggle Notebooks with P100 GPU.

---

## 📁 Project Structure

```
TransConvNeXt-CD/
├── src/                           # 📦 Python source code
│   ├── config.py                  # Configuration & hyperparameters
│   ├── dataset.py                 # LEVIR-CD dataset loader with augmentations
│   ├── model.py                   # TransConvNeXt-CD architecture
│   ├── train.py                   # Training script (CLI)
│   ├── test.py                    # Evaluation & visualization (CLI)
│   └── utils.py                   # Loss, metrics, TTA utilities
├── notebooks/                     # 📓 Jupyter notebooks
│   ├── train_model.ipynb          # Training notebook (Kaggle-ready)
│   └── test_inference.ipynb       # Testing notebook with 8x TTA
├── assets/                        # 🖼️ Assets
│   └── results.png                # Test results visualization
├── requirements.txt               # 📋 Dependencies
├── .gitignore
├── LICENSE                        # ⚖️ MIT License
└── README.md                      # 📖 This file
```

---

## ⚙️ Training Details

### Hyperparameters

| Setting | Value |
|---------|-------|
| Image Size (Train) | 512 × 512 (random crop) |
| Image Size (Val/Test) | 1024 × 1024 (full image) |
| Batch Size (Train) | 8 |
| Batch Size (Validation) | 4 |
| Optimizer | AdamW (lr = 2×10⁻⁴, weight decay = 0.05) |
| Learning Rate Scheduler | CosineAnnealingWarmRestarts (T₀ = 15, Tₘᵤₗₜ = 2) |
| Loss Function | BCE + Dice |
| Precision | FP16 (mixed precision) |
| Epochs | 150 |
| GPU | NVIDIA Tesla P100 (16GB) |

### Data Augmentation (Training)

| Augmentation | Probability |
|-------------|:-----------:|
| Random Crop (512×512) | 100% |
| Horizontal Flip | 50% |
| Vertical Flip | 50% |
| Random Rotation 90° | 50% |
| Transpose | 50% |
| Elastic / Grid / Optical Distortion | 30% |
| Gaussian Noise / Brightness-Contrast / Hue-Saturation | 30% |

### Inference (8x TTA)

| Transform | Included |
|-----------|:--------:|
| Identity | ✅ |
| Rotate 90° | ✅ |
| Rotate 180° | ✅ |
| Rotate 270° | ✅ |
| Horizontal Flip | ✅ (+ above rotations) |

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. 🐛 **Report bugs** — Open an [issue](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD/issues)
2. 💡 **Suggest features** — Share your ideas via issues
3. 🔧 **Submit PRs** — Improve code, add features, fix bugs
4. 📖 **Improve docs** — Better documentation is always needed

---

## 📚 Citation

If you find this work useful in your research, please consider citing:

```bibtex
@misc{transconvnext-cd,
  author = {Starry-Sky-Universe},
  title = {TransConvNeXt-CD: Transformer-CNN Hybrid Network for Remote Sensing Change Detection},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Starry-Sky-Universe/TransConvNeXt-CD}},
  doi = {10.5281/zenodo.XXXXXXX}
}
```

### Related Work

- [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR/) — H. Chen, et al.
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — A ConvNet for the 2020s
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models by Ross Wightman

---

## ⚖️ License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If you find this project useful, please consider giving it a ⭐ star!**

[![GitHub Stars](https://img.shields.io/github/stars/Starry-Sky-Universe/TransConvNeXt-CD?style=social)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

</div>

---

## 🙋 FAQ

**Q: What GPU do I need?**
A: Training works on any GPU with ≥8GB VRAM. The model was trained on a Tesla P100 (16GB).

**Q: Can I use this with other datasets?**
A: Yes! The data loader can be easily adapted for other change detection datasets by modifying the dataset class.

**Q: How long does training take?**
A: ~6-8 hours on a Tesla P100 for 150 epochs with the default settings.

**Q: Does it work on CPU?**
A: Inference works on CPU but is slow. Training requires a GPU.