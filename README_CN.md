<div align="center">

# 🛰️ TransConvNeXt-CD

**Transformer-CNN 混合网络的遥感变化检测模型**

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org)
[![timm](https://img.shields.io/badge/Backbone-ConvNeXt--Tiny-success)](https://github.com/huggingface/pytorch-image-models)
[![Dataset](https://img.shields.io/badge/Dataset-LEVIR--CD-ff6b6b)](https://justchenhao.github.io/LEVIR/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![F1](https://img.shields.io/badge/F1%20Score-90.85%25-brightgreen)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)
[![IoU](https://img.shields.io/badge/IoU-83.24%25-success)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

**[English](README.md) | 简体中文**

</div>

---

## 📋 目录

- [项目概述](#-项目概述)
- [模型架构](#-模型架构)
- [数据集](#-数据集-levir-cd)
- [实验结果](#-实验结果)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [训练详情](#-训练详情)
- [引用](#-引用)
- [许可证](#-许可证)

---

## 🎯 项目概述

**TransConvNeXt-CD** 是一个基于 Transformer-CNN 混合架构的遥感图像变化检测模型。它使用 **ConvNeXt-Tiny** 作为骨干网络，结合 **交叉注意力 Transformer** 模块，能够同时捕捉双时相图像之间的局部空间细节和全局语义依赖关系。

### ✨ 主要特性

- **ConvNeXt-Tiny 骨干网络** — 现代化 CNN 架构，具有优秀的特征提取能力
- **交叉注意力模块** — 捕捉 T1/T2 时序图像间的长距离依赖关系
- **空间差异模块** — 多层级特征交互，精确定位变化区域
- **深度监督** — 解码器中间层辅助损失，训练更稳定
- **8 倍测试时增强** — D4 二面体群（4 旋转 × 2 翻转），推理更鲁棒
- **混合精度训练** — FP16 训练，速度更快、显存占用更低
- **混合损失函数** — BCE + Dice 损失，精确的像素级分割
- **Kaggle 自动路径检测** — 自动搜索数据集和权重文件

---

## 🏗️ 模型架构

```
T1 图像 ──┐                    ┌── SpatialDiff(96) ── Decoder 4 ──┐
           ├── ConvNeXt ─── Stage1 ─── SpatialDiff(192) ── Decoder 3 ─┤
T2 图像 ──┘    Tiny     ─── Stage2 ─── SpatialDiff(384) ── Decoder 2 ─┤
                          ─── Stage3 ──── CrossAttention ───── Decoder 1 ─┤
                          ─── Stage4 ──── Fusion ────────────────────────┴── 变化图
```

### 模块说明

| 模块 | 说明 |
|--------|---------|
| **编码器** | ConvNeXt-Tiny（4 阶段，通道数: 96 → 192 → 384 → 768） |
| **空间差异** | `Conv2d(B N ReLU Conv2d)` — 通过通道拼接学习变化特征 |
| **交叉注意力** | `MultiheadAttention(12 heads, dim=768)` + LayerNorm + FFN 残差连接 |
| **解码器** | 4 级双线性上采样 + SE 注意力 + 跳跃连接 |
| **深度监督** | 解码器第 3、4 层辅助分割头，增强梯度流动 |
| **输出** | 单通道变化概率图（1024 × 1024） |

---

## 📊 数据集: LEVIR-CD

[LEVIR-CD](https://justchenhao.github.io/LEVIR/) 是遥感建筑物变化检测领域的大型基准数据集。

### 数据集统计

| 属性 | 数值 |
|----------|-------|
| 图像尺寸 | 1024 × 1024 像素 |
| 空间分辨率 | 0.5 米/像素 |
| 图像对总数 | 637 对 |
| 训练集 | 445 对 |
| 验证集 | 64 对 |
| 测试集 | 128 对 |
| 变化类型 | 新建、拆除、改建 |
| 时间跨度 | 2002 – 2018 |
| 数据来源 | Google Earth |

---

## 📈 实验结果

### 🔬 测试集性能（8x TTA，128 对测试图像）

| 指标 | 分数 |
|--------|:-----:|
| **IoU**（交并比） | **83.24%** |
| **F1 Score**（F1 分数） | **90.85%** |
| **Precision**（精确率） | **91.89%** |
| **Recall**（召回率） | **89.84%** |

### 📉 训练过程

| Epoch | Val F1 | Val IoU | Val Precision | Val Recall |
|:-----:|:------:|:-------:|:-------------:|:----------:|
| 1 | 0.4856 | 0.3206 | 0.3349 | 0.8828 |
| 5 | 0.8044 | 0.6727 | 0.7838 | 0.8260 |
| 10 | 0.8547 | 0.7462 | 0.7958 | 0.9229 |
| 20 | 0.9035 | 0.8239 | 0.9036 | 0.9033 |
| **150 (最佳)** | **0.9078** | — | — | — |
| **测试集 (8x TTA)** | **0.9085** | **0.8324** | **0.9189** | **0.8984** |

### 🖼️ 可视化结果

![测试结果可视化](assets/results.png)

*LEVIR-CD 测试集样本变化检测结果。从左到右：变化前图像 (T1)、变化后图像 (T2)、真实变化标签、预测变化图及每项指标。*

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
git clone https://github.com/Starry-Sky-Universe/TransConvNeXt-CD.git
cd TransConvNeXt-CD

# 安装依赖
pip install -r requirements.txt
```

### 数据集准备

1. 从以下来源下载 **LEVIR-CD**：
   - [官方网站](https://justchenhao.github.io/LEVIR/)
   - [Kaggle 数据集](https://www.kaggle.com/datasets/balraj98/levir-cd)

2. 按以下结构组织数据：
```
/path/to/LEVIR/
├── train/
│   ├── A/          # 变化前图像 (1024×1024)
│   ├── B/          # 变化后图像 (1024×1024)
│   └── label/      # 变化标签 (二值)
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

### 训练

```bash
# 基础训练
python src/train.py --data /path/to/LEVIR

# 自定义训练配置
python src/train.py \
    --data /path/to/LEVIR \
    --save checkpoints/best_model.pth \
    --epochs 150 \
    --batch_size 8 \
    --lr 2e-4
```

### 测试评估

```bash
# 使用 8x TTA 评估
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth

# 评估并可视化
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth \
    --visualize \
    --output results.png
```

---

## 📁 项目结构

```
TransConvNeXt-CD/
├── src/                           # 📦 Python 源代码
│   ├── config.py                  # 配置与超参数
│   ├── dataset.py                 # LEVIR-CD 数据集加载与增强
│   ├── model.py                   # TransConvNeXt-CD 模型定义
│   ├── train.py                   # 训练脚本 (命令行)
│   ├── test.py                    # 评估与可视化脚本 (命令行)
│   └── utils.py                   # 损失函数、评估指标、TTA 工具
├── notebooks/                     # 📓 Jupyter 笔记本
│   ├── train_model.ipynb          # 训练流程（Kaggle 就绪）
│   └── test_inference.ipynb       # 测试与推理（含 8x TTA）
├── assets/                        # 🖼️ 资源文件
│   └── results.png                # 测试结果可视化图
├── requirements.txt               # 📋 依赖清单
├── .gitignore
├── LICENSE                        # ⚖️ MIT 许可证
└── README.md                      # 📖 说明文档
```

---

## ⚙️ 训练详情

### 超参数

| 设置 | 数值 |
|---------|-------|
| 训练图像尺寸 | 512 × 512（随机裁剪） |
| 验证/测试图像尺寸 | 1024 × 1024（全图） |
| 训练批次大小 | 8 |
| 验证批次大小 | 4 |
| 优化器 | AdamW (lr = 2×10⁻⁴, weight_decay = 0.05) |
| 学习率调度器 | CosineAnnealingWarmRestarts (T₀ = 15, Tₘᵤₗₜ = 2) |
| 损失函数 | BCE + Dice |
| 精度 | FP16（混合精度） |
| 迭代轮数 | 150 |
| GPU | NVIDIA Tesla P100 (16GB) |

### 数据增强（训练集）

| 增强方式 | 概率 |
|-------------|:-----------:|
| 随机裁剪 (512×512) | 100% |
| 水平翻转 | 50% |
| 垂直翻转 | 50% |
| 随机旋转 90° | 50% |
| 转置 | 50% |
| 弹性/网格/光学变形 | 30% |
| 高斯噪声/亮度对比度/色调饱和度 | 30% |

---

## 📚 引用

```bibtex
@misc{transconvnext-cd,
  author = {Starry-Sky-Universe},
  title = {TransConvNeXt-CD: Transformer-CNN Hybrid Network for Remote Sensing Change Detection},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Starry-Sky-Universe/TransConvNeXt-CD}}
}
```

### 相关项目

- [LEVIR-CD 数据集](https://justchenhao.github.io/LEVIR/) — H. Chen 等
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — A ConvNet for the 2020s
- [timm](https://github.com/huggingface/pytorch-image-models) — PyTorch Image Models

---

## ⚖️ 许可证

本项目采用 MIT 许可证 — 详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**如果这个项目对你有帮助，请给我们一个 ⭐ star！**

</div>