<div align="center">

# 🛰️ TransConvNeXt-CD

**Transformer-CNN Hybrid Network for Remote Sensing Change Detection**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/timm-ConvNeXt-brightgreen)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-LEVIR--CD-important)](https://justchenhao.github.io/LEVIR/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/Paper-Pending-red)](https://github.com/Starry-Sky-Universe)
[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-blue?logo=kaggle)](https://kaggle.com)

---

</div>

## 📋 Overview

**TransConvNeXt-CD** is a state-of-the-art change detection model for remote sensing imagery. It combines **ConvNeXt-Tiny** as a powerful CNN backbone with **Cross-Attention Transformer** modules to capture both local spatial details and global semantic dependencies between bi-temporal images.

### Key Features
- **ConvNeXt-Tiny Backbone** - Modern CNN with excellent feature extraction
- **Cross-Attention Block** - Captures long-range dependencies between T1 and T2
- **Spatial Difference Module** - Multi-level feature interaction
- **Deep Supervision** - Auxiliary loss at decoder stages for better training
- **8x Test-Time Augmentation** - D4 dihedral group for robust inference
- **Mixed Precision Training** - Faster training with less memory
- **Hybrid Loss** - BCE + Dice for pixel-level accuracy

## 📊 Dataset: LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) is a large-scale remote sensing change detection dataset with:

| Property | Value |
|----------|-------|
| Image Size | 1024 × 1024 pixels |
| Resolution | 0.5 m/pixel |
| Total Pairs | 637 A/B image pairs |
| Training | 445 pairs |
| Validation | 64 pairs |
| Testing | 128 pairs |
| Changes | Buildings (new, gone, altered) |

Dataset download: [LEVIR-CD on Kaggle](https://www.kaggle.com/datasets/balraj98/levir-cd)

## 🏗️ Architecture

```
T1 Image ──┐                    ┌── SpatialDiff ── Decoder ──┐
           ├── ConvNeXt ── Cross ┤                            ├── Change Map
T2 Image ──┘           Attention └── Transformer ── Decoder ──┘
```

- **Encoder**: ConvNeXt-Tiny (4 stages, channels: 96→192→384→768)
- **Interaction**: Cross-Attention (8 heads, dim=768) + Spatial Difference (channels 96/192/384)
- **Decoder**: 4-level upsampling with SE attention and skip connections
- **Output**: Single-channel change probability map (1024×1024)

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
# Train on LEVIR-CD dataset
python src/train.py --data /path/to/LEVIR --epochs 150 --batch_size 8
```

### Evaluation

```bash
# Evaluate with 8x TTA
python src/test.py --data /path/to/LEVIR --weight TransConv_SOTA_Best.pth --visualize
```

### Jupyter Notebooks

The project includes two Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `notebooks/train_model.ipynb` | Full training pipeline with mixed precision |
| `notebooks/test_inference.ipynb` | Testing with 8x TTA + visualization |

## 📈 Results

| Metric | Score |
|--------|-------|
| IoU | TBD |
| F1 Score | TBD |
| Precision | TBD |
| Recall | TBD |

> Results will be updated after training completes on the LEVIR-CD test set.

## 📁 Project Structure

```
├── src/                         # Python modules
│   ├── config.py                # Configuration & hyperparameters
│   ├── dataset.py               # LEVIR-CD dataset loader
│   ├── model.py                 # TransConvNeXt-CD model definition
│   ├── train.py                 # Training script
│   ├── test.py                  # Evaluation & visualization script
│   └── utils.py                 # Loss, metrics, TTA utilities
├── notebooks/                   # Jupyter notebooks
│   ├── train_model.ipynb        # Training notebook
│   └── test_inference.ipynb     # Testing & inference notebook
├── requirements.txt             # Dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## 🧪 Training Details

| Setting | Value |
|---------|-------|
| Image Size (Train) | 512×512 (random crop) |
| Image Size (Val/Test) | 1024×1024 (full) |
| Batch Size (Train) | 8 |
| Batch Size (Val) | 4 |
| Optimizer | AdamW (lr=2e-4, wd=0.05) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=15, Tₘᵤₗₜ=2) |
| Loss Function | BCE + Dice |
| Precision | FP16 (mixed precision) |
| Epochs | 150 |

## 📚 Citation

```bibtex
@misc{transconvnext-cd,
  title = {TransConvNeXt-CD: Transformer-CNN Hybrid Network for Remote Sensing Change Detection},
  author = {Starry-Sky-Universe},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Starry-Sky-Universe/TransConvNeXt-CD}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [LEVIR-CD Dataset](https://justchenhao.github.io/LEVIR/) - Remote sensing change detection benchmark
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [timm](https://github.com/huggingface/pytorch-image-models) - PyTorch image models
- [Kaggle](https://kaggle.com) - GPU compute platform