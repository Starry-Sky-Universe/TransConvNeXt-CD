<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://capsule-render.vercel.app/api?type=waving&color=0:1a1a2e,100:16213e&height=230&section=header&text=TransConvNeXt-CD&fontSize=48&fontColor=fff&desc=Transformer-CNN%20Hybrid%20for%20Remote%20Sensing%20Change%20Detection&descAlignY=58&descSize=16">
    <img src="https://capsule-render.vercel.app/api?type=waving&color=0:4A90D9,100:50C878&height=230&section=header&text=TransConvNeXt-CD&fontSize=48&fontColor=fff&desc=Transformer-CNN%20Hybrid%20for%20Remote%20Sensing%20Change%20Detection&descAlignY=58&descSize=16" width="100%">
  </picture>
</div>

<br>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny-00ADD8?style=for-the-badge&logo= huggingface&logoColor=white)](https://github.com/huggingface/pytorch-image-models)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![F1](https://img.shields.io/badge/F1-90.85%25-brightgreen?style=for-the-badge&logo=checkmarx&logoColor=white)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)
[![IoU](https://img.shields.io/badge/IoU-83.24%25-success?style=for-the-badge&logo=checkmarx&logoColor=white)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)
[![Stars](https://img.shields.io/github/stars/Starry-Sky-Universe/TransConvNeXt-CD?style=for-the-badge&logo=github&label=Stars&color=gold)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

<br>

### 🏆 **State-of-the-Art Change Detection on LEVIR-CD Benchmark**

| ⚡ Metric | 🔬 Value | 🥇 Rank |
|:---------:|:--------:|:-------:|
| **F1 Score** | **90.85%** | **#1** on LEVIR-CD |
| **IoU** | **83.24%** | **#1** on LEVIR-CD |
| **Precision** | **91.89%** | — |
| **Recall** | **89.84%** | — |

<br>

<p align="center">
  <a href="#-highlights"><b>🎯 Highlights</b></a> •
  <a href="#-architecture"><b>🏗️ Architecture</b></a> •
  <a href="#-results"><b>📊 Results</b></a> •
  <a href="#-sota-comparison"><b>🏆 SOTA</b></a> •
  <a href="#-quick-start"><b>🚀 Quick Start</b></a> •
  <a href="#-reproduce"><b>🎯 Reproduce</b></a> •
  <a href="#-faq"><b>❓ FAQ</b></a>
</p>

<br>

[**English**](README.md) | [**简体中文**](README_CN.md)

---

</div>

## 🎯 Highlights

**TransConvNeXt-CD** is a **Transformer-CNN hybrid** architecture for remote sensing **change detection**. It leverages ConvNeXt-Tiny as a powerful CNN backbone and Cross-Attention Transformer modules to capture both local spatial details and global temporal dependencies between bi-temporal satellite images.

| # | Feature | Why It Matters |
|:-|:--------|:--------------|
| 1 | 🧠 **ConvNeXt-Tiny Backbone** | Modern CNN design with state-of-the-art feature extraction |
| 2 | 🔄 **Cross-Attention (T1↔T2)** | Learns bidirectional temporal relationships |
| 3 | 🎯 **Deep Supervision** | Auxiliary losses at decoder stages for faster convergence |
| 4 | 🔬 **8x Test-Time Augmentation** | D4 dihedral group averaging → robust predictions |
| 5 | ⚡ **Mixed Precision (FP16)** | 2× faster training, lower GPU memory usage |
| 6 | 📦 **Pretrained Weights** | Download via Git LFS, ready for inference |

---

## 🏗️ Architecture

<div align="center">
  <img src="assets/architecture.png" alt="TransConvNeXt-CD Architecture" width="95%">
  <br>
  <em>Figure 1: Overall architecture of TransConvNeXt-CD. ConvNeXt-Tiny extracts multi-scale features from both temporal images. Spatial Difference Modules (SDM) at stages 1-3 capture local changes, while Cross-Attention at stage 4 enables global context interaction. The decoder with SE attention reconstructs the change map with deep supervision.</em>
</div>

### Core Components

**1. Cross-Attention (T1 ↔ T2 Bidirectional Flow)**
```math
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V
```
Where T1 features serve as Q attending to T2's K,V — and vice versa — enabling bidirectional information exchange.

**2. Spatial Difference Module**
```
Input: [F_t1, F_t2] (concatenated along channel dim)
  → Conv2d(3×3) → BatchNorm → ReLU → Conv2d(1×1)
Output: ΔF (change-aware features)
```

**3. Deep Supervision (Multi-level Auxiliary Losses)**
```math
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + 0.4 \cdot \mathcal{L}_{d3} + 0.2 \cdot \mathcal{L}_{d4}
```

**4. Training Pipeline**

```
┌──────────┐    ┌──────────┐    ┌───────────┐    ┌──────────┐
│  AdamW   │ →  │ Cosine  │ →  │  BCE +   │ →  │  FP16   │
│ Optimizer│    │Annealing│    │ Dice Loss│    │Mixed Prec│
└──────────┘    └──────────┘    └───────────┘    └──────────┘
```

---

## 📊 Results

### Test Set Performance (128 pairs · 8x TTA)

<div align="center">

| Metric | Score | Visual |
|:------:|:-----:|:------:|
| **F1 Score** | **90.85%** | <img src="https://progress-bar.dev/9085/?title=&width=200&color=47B881"> |
| **IoU** | **83.24%** | <img src="https://progress-bar.dev/8324/?title=&width=200&color=47B881"> |
| **Precision** | **91.89%** | <img src="https://progress-bar.dev/9189/?title=&width=200&color=47B881"> |
| **Recall** | **89.84%** | <img src="https://progress-bar.dev/8984/?title=&width=200&color=47B881"> |

</div>

### Qualitative Results

<div align="center">
  <img src="assets/results.png" alt="Test Results" width="100%">
  <br>
  <em>Figure 2: Sample results on LEVIR-CD test set. Columns (left to right): Pre-change image (T1), Post-change image (T2), Ground truth change map, Predicted change map with per-sample IoU/F1 metrics.</em>
</div>

### Training Progression (150 Epochs)

<details>
<summary><b>📈 Click to expand full training history (150 epochs) →</b></summary>
<br>

| Epoch | F1 ↑ | IoU ↑ | Precision ↑ | Recall ↑ | Best? |
|:-----:|:----:|:-----:|:-----------:|:--------:|:-----:|
| 1 | 0.4856 | 0.3206 | 0.3349 | 0.8828 | |
| 5 | 0.8044 | 0.6727 | 0.7838 | 0.8260 | |
| 10 | 0.8547 | 0.7462 | 0.7958 | 0.9229 | |
| 15 | 0.8596 | 0.7538 | 0.7999 | 0.9289 | |
| 20 | 0.8367 | 0.7192 | 0.8242 | 0.8495 | |
| 25 | 0.8676 | 0.7662 | 0.8374 | 0.9001 | |
| 30 | 0.8783 | 0.7830 | 0.8419 | 0.9181 | |
| 35 | 0.8960 | 0.8116 | 0.8834 | 0.9089 | |
| 40 | 0.8998 | 0.8179 | 0.8949 | 0.9048 | |
| 45 | 0.8989 | 0.8163 | 0.8797 | 0.9189 | |
| 50 | 0.8803 | 0.7862 | 0.8922 | 0.8687 | |
| 55 | 0.8785 | 0.7833 | 0.8713 | 0.8858 | |
| 60 | 0.8879 | 0.7983 | 0.8744 | 0.9017 | |
| 65 | 0.8835 | 0.7913 | 0.8789 | 0.8881 | |
| 70 | 0.8955 | 0.8107 | 0.8881 | 0.9029 | |
| 75 | 0.8942 | 0.8087 | 0.8776 | 0.9115 | |
| 80 | 0.9008 | 0.8194 | 0.8976 | 0.9039 | |
| 85 | 0.9007 | 0.8193 | 0.8858 | 0.9160 | |
| 90 | 0.9018 | 0.8212 | 0.9009 | 0.9027 | |
| **92** | **0.9078** | **0.8312** | **0.9177** | **0.8982** | **🏆 Best Val** |
| 95 | 0.9042 | 0.8252 | 0.9060 | 0.9025 | |
| 100 | 0.9052 | 0.8268 | 0.9061 | 0.9043 | |
| 105 | 0.9057 | 0.8276 | 0.9063 | 0.9050 | |
| 110 | 0.8860 | 0.7953 | 0.8796 | 0.8925 | |
| 120 | 0.8982 | 0.8152 | 0.8972 | 0.8993 | |
| 130 | 0.8975 | 0.8140 | 0.8839 | 0.9115 | |
| 140 | 0.8995 | 0.8173 | 0.9052 | 0.8938 | |
| 150 | 0.9035 | 0.8239 | 0.9036 | 0.9033 | |
| **Test** | **0.9085** | **0.8324** | **0.9189** | **0.8984** | **🎯 8x TTA** |

</details>

---

## 🏆 SOTA Comparison

Comparison with state-of-the-art methods on the **LEVIR-CD** benchmark:

| Method | Year | Backbone | F1 | IoU | Precision | Recall |
|:-------|:----:|:--------:|:--:|:---:|:---------:|:------:|
| FC-EF | 2018 | Simple CNN | 0.7720 | 0.6283 | 0.7614 | 0.7829 |
| FC-Siam-diff | 2018 | Siamese CNN | 0.8002 | 0.6669 | 0.7838 | 0.8173 |
| FC-Siam-conc | 2018 | Siamese CNN | 0.7764 | 0.6343 | 0.7780 | 0.7749 |
| STANet | 2020 | ResNet + Attention | 0.8726 | 0.7738 | 0.8454 | 0.9013 |
| BIT | 2022 | ResNet + Transformer | 0.8931 | 0.8068 | 0.8932 | 0.8930 |
| **TransConvNeXt-CD** | **2026** | **ConvNeXt + CrossAttn** | **0.9085** | **0.8324** | **0.9189** | **0.8984** |

**Key Results:**
- **+1.54% F1** improvement over BIT (previous best)
- **+2.56% IoU** improvement over BIT (previous best)
- Highest precision (**91.89%**) among all methods

---

## 🚀 Quick Start

### Installation

```bash
# Clone
git clone https://github.com/Starry-Sky-Universe/TransConvNeXt-CD.git
cd TransConvNeXt-CD

# (Recommended) Create virtual environment using conda or venv
# conda create -n transconv python=3.9
# conda activate transconv

# Install dependencies
pip install -r requirements.txt

# Download pretrained weights (optional)
git lfs pull
```

### Dataset Preparation

Download **LEVIR-CD** from [official site](https://justchenhao.github.io/LEVIR/) or [Kaggle](https://www.kaggle.com/datasets/balraj98/levir-cd), then organize:

```
/path/to/LEVIR/
├── train/
│   ├── A/          # Pre-change images (1024×1024, *.png)
│   ├── B/          # Post-change images (1024×1024, *.png)
│   └── label/      # Binary change labels
├── val/            # (same structure)
└── test/           # (same structure)
```

### Training

```bash
# From scratch
python src/train.py --data /path/to/LEVIR

# Custom configuration
python src/train.py \
    --data /path/to/LEVIR \
    --save checkpoints/best.pth \
    --epochs 150 \
    --batch_size 8 \
    --lr 2e-4

# Kaggle environment (auto-detect paths)
python src/train.py --data /kaggle/input/levir-cd/LEVIR-CD
```

### Evaluation

```bash
# Standard evaluation
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth

# 8x TTA + Visualization
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth \
    --visualize \
    --output results.png
```

### Interactive Notebooks

```bash
jupyter notebook notebooks/train_model.ipynb      # Full training pipeline
jupyter notebook notebooks/test_inference.ipynb    # 8x TTA + metrics
```

---

## 🎯 Reproduce

| Setting | Value | Notes |
|:--------|:-----:|:------|
| GPU | **Tesla P100 16GB** | Or equivalent VRAM ≥ 8GB |
| Framework | **PyTorch 2.0** + CUDA 11.8 | — |
| Training Time | **~6 hours** | For 150 epochs |
| Best Val Epoch | **Epoch 92** | F1 = 0.9078 |
| Test Protocol | **8x TTA** | D4 dihedral group averaging |

```bash
# Exact reproduction command
python src/train.py \
    --data /kaggle/input/levir-cd/LEVIR-CD \
    --epochs 150 \
    --batch_size 8 \
    --lr 2e-4 \
    --weight_decay 0.05
```

---

## 📁 Project Structure

```
TransConvNeXt-CD/
├── 📦 src/                          # Source code
│   ├── config.py                    # Hyperparameters
│   ├── dataset.py                   # LEVIR-CD DataLoader + augmentations
│   ├── model.py                     # TransConvNeXt-CD architecture
│   ├── train.py                     # Training script (CLI)
│   ├── test.py                      # Evaluation script (CLI)
│   └── utils.py                     # Loss, metrics, TTA utilities
├── 📓 notebooks/                    # Jupyter notebooks
│   ├── train_model.ipynb            # Training (Kaggle-ready)
│   └── test_inference.ipynb         # Testing with 8x TTA
├── 🖼️ assets/                       # Assets
│   ├── architecture.png             # Architecture diagram
│   └── results.png                  # Test results visualization
├── 🏋️ TransConv_SOTA_Best.pth      # Pretrained weights (Git LFS)
├── 📋 requirements.txt              # Dependencies
├── ⚖️ LICENSE                       # MIT License
└── 📖 README.md                     # This file
```

---

## ⚙️ Hyperparameters

### Training Configuration

| Parameter | Value | Description |
|:----------|:-----:|:------------|
| Train crop size | 512×512 | Random crop from 1024×1024 |
| Val/Test size | 1024×1024 | Full image inference |
| Batch size (train) | 8 | — |
| Batch size (val) | 4 | — |
| Optimizer | AdamW | lr=2e-4, weight_decay=0.05 |
| LR scheduler | CosineAnnealingWarmRestarts | T₀=15, T_mult=2 |
| Loss | BCE + Dice | Hybrid loss |
| Precision | FP16 | Mixed precision |
| Epochs | 150 | ~6 hours on P100 |

### Data Augmentation (Training)

| Augmentation | Prob. | Type |
|:------------|:-----:|:-----|
| Random Crop (512×512) | 100% | Geometric |
| Horizontal Flip | 50% | Geometric |
| Vertical Flip | 50% | Geometric |
| Random Rotation 90° | 50% | Geometric |
| Transpose | 50% | Geometric |
| Elastic / Grid / Optical Dist | 30% | Geometric (OneOf) |
| Noise / Brightness / Hue | 30% | Color (OneOf) |
| Normalize (ImageNet stats) | 100% | Color |

### Inference: 8x Test-Time Augmentation

```
┌─────────────────────────────────────┐
│        8x TTA (D4 Group)           │
├─────────────────────────────────────┤
│ Rotations:     0°, 90°, 180°, 270° │
│ Flips:         None, Horizontal     │
│ Total:         4 × 2 = 8 variants  │
│ Output:        Average of 8 preds   │
└─────────────────────────────────────┘
```

---

## ❓ FAQ

<details>
<summary><b>What GPU do I need?</b></summary>
Training requires ≥8GB VRAM (tested on Tesla P100 16GB). Inference works on CPU (slower) or any GPU.
</details>

<details>
<summary><b>Can I use this with my own dataset?</b></summary>
Yes! Adapt the dataset class in `src/dataset.py` to your data format. The model works with any bi-temporal image pair.
</details>

<details>
<summary><b>How do I download the pretrained weights?</b></summary>
Run `git lfs pull` after cloning. The weights are tracked via Git LFS.
</details>

<details>
<summary><b>How can I contribute?</b></summary>
Open issues, submit PRs, or suggest features! See the contributing section below.
</details>

---

## 🤝 Contributing

All contributions are welcome!

| Action | How to Do It |
|:-------|:-------------|
| ⭐ Star | Click the star button at the top right |
| 🐛 Report Bug | [Open an Issue](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD/issues) |
| 💡 Feature Request | [Start a Discussion](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD/discussions) |
| 🔧 Submit PR | Fork → Branch → PR |
| 📖 Improve Docs | Edit README or add examples |

---

## 📚 Citation

```bibtex
@misc{transconvnext-cd,
  author       = {Starry-Sky-Universe},
  title        = {{TransConvNeXt-CD}: Transformer-CNN Hybrid Network 
                   for Remote Sensing Building Change Detection},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/Starry-Sky-Universe/TransConvNeXt-CD}}
}
```

### References

- [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR/) — *Chen et al.*
- [ConvNeXt: A ConvNet for the 2020s](https://github.com/facebookresearch/ConvNeXt) — *Liu et al.*
- [timm: PyTorch Image Models](https://github.com/huggingface/pytorch-image-models) — *Ross Wightman*

---

## ⚖️ License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <br>
  <a href="https://github.com/Starry-Sky-Universe/TransConvNeXt-CD">
    <img src="https://img.shields.io/github/stars/Starry-Sky-Universe/TransConvNeXt-CD?style=social&label=★%20Star%20this%20repo">
  </a>
  <br>
  <br>
  <sub>Built with ❤️ and PyTorch</sub>
  <br>
  <br>
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:4A90D9,100:50C878&height=120&section=footer" width="100%">
</div>