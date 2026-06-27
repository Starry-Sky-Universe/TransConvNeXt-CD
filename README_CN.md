<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=0,2,4,6,8&height=200&section=header&text=TransConvNeXt-CD&fontSize=50&fontAlignY=35&desc=Transformer-CNN%20%E6%B7%B7%E5%90%88%E7%BD%91%E7%BB%9C%E7%9A%84%E9%81%A5%E6%84%9F%E5%8F%98%E5%8C%96%E6%A3%80%E6%B5%8B&descAlignY=55&animation=fadeIn" width="100%"/>
</div>

<div align="center">

<br>

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge&logo=open-source-initiative&logoColor=white)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/Starry-Sky-Universe/TransConvNeXt-CD?style=for-the-badge&logo=github&color=gold)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

---

### 🔥 Transformer-CNN 混合网络的遥感变化检测 | F1: **90.85%** | IoU: **83.24%**

<br>

| 📊 **测试指标** | ⚡ **特性** |
|:---:|:---:|
| **90.85%** F1 / **83.24%** IoU | ConvNeXt + 交叉注意力 |
| **91.89%** 精确率 / **89.84%** 召回率 | 8x TTA / 混合精度 |
| ⏱ **150 轮** ~ 6h (P100) | Kaggle 就绪笔记本 |

<br>

<p align="center">
  <a href="#-实验结果">📊 实验结果</a> •
  <a href="#-模型架构">🏗️ 模型架构</a> •
  <a href="#-快速开始">🚀 快速开始</a> •
  <a href="#-方法对比">📊 方法对比</a> •
  <a href="#-复现指南">🎯 复现指南</a>
</p>

<br>

[**English**](README.md) | [**简体中文**](README_CN.md)

---

</div>

## ✨ 为什么选择 TransConvNeXt-CD？

> **TransConvNeXt-CD** 是一个 Transformer-CNN 混合架构的遥感**变化检测**模型。在 LEVIR-CD 测试集上达到 **90.85% F1** 和 **83.24% IoU**。

<div align="center">

| ⚡ 特性 | 💡 说明 |
|:---|---|
| 🧠 **ConvNeXt-Tiny 骨干网络** | 现代 CNN 设计，强大的特征提取 |
| 🔄 **交叉注意力 Transformer** | 捕捉长距离时序依赖关系 |
| 🎯 **深度监督学习** | 更好的梯度流动，更快收敛 |
| 🔬 **8 倍测试时增强 (TTA)** | D4 二面体群，推理更鲁棒 |
| ⚡ **混合精度训练 (FP16)** | 速度提升 2 倍，显存占用更低 |
| 📦 **提供预训练权重** | 通过 Git LFS 下载，开箱即用 |

</div>

---

## 📊 方法对比

与已有方法在 LEVIR-CD 测试集上的指标对比：

| 方法 | 年份 | 骨干网络 | F1 | IoU | 精确率 | 召回率 |
|:------|:----:|:--------:|:--:|:---:|:-----:|:----:|
| FC-EF | 2018 | 简单 CNN | 77.20 | 62.83 | 76.14 | 78.29 |
| FC-Siam-diff | 2018 | 孪生 CNN | 80.02 | 66.69 | 78.38 | 81.73 |
| FC-Siam-conc | 2018 | 孪生 CNN | 77.64 | 63.43 | 77.80 | 77.49 |
| STANet | 2020 | ResNet + Attention | 87.26 | 77.38 | 84.54 | 90.13 |
| BIT | 2022 | ResNet + Transformer | 89.31 | 80.68 | 89.32 | 89.30 |
| **TransConvNeXt-CD (本作)** | **2026** | **ConvNeXt + CrossAttn** | **90.85** | **83.24** | **91.89** | **89.84** |

> 相比 BIT，本模型的 F1 提升 **+1.54%**，IoU 提升 **+2.56%**。注：对比范围限于能复现的已有方法，更新的方法可能取得不同结果。

---

## 📊 实验结果

### 测试集性能（128 对图像，8x TTA）

<div align="center">

| 指标 | 数值 | 可视化 |
|:------:|:-----:|:------------:|
| **F1 Score** | **90.85%** | <img src="https://img.shields.io/badge/90.85%25-brightgreen"> |
| **IoU（交并比）** | **83.24%** | <img src="https://img.shields.io/badge/83.24%25-success"> |
| **Precision（精确率）** | **91.89%** | <img src="https://img.shields.io/badge/91.89%25-brightgreen"> |
| **Recall（召回率）** | **89.84%** | <img src="https://img.shields.io/badge/89.84%25-brightgreen"> |

</div>

### 可视化结果

<div align="center">

![测试结果](assets/results.png)

*LEVIR-CD 测试集样本结果。从左到右：变化前图像 (T1) → 变化后图像 (T2) → 真实标签 → 预测结果（含每项指标）*

</div>

### 完整训练记录（150 轮）

<details open>
<summary><b>📈 点击展开训练历史 →</b></summary>

<br>

| 轮次 | F1 ↑ | IoU ↑ | 精确率 ↑ | 召回率 ↑ |
|:-----:|:----:|:-----:|:-----------:|:--------:|
| **1** | 0.4856 | 0.3206 | 0.3349 | 0.8828 |
| **5** | 0.8044 | 0.6727 | 0.7838 | 0.8260 |
| **10** | 0.8547 | 0.7462 | 0.7958 | 0.9229 |
| **15** | 0.8596 | 0.7538 | 0.7999 | 0.9289 |
| **20** | 0.8367 | 0.7192 | 0.8242 | 0.8495 |
| **25** | 0.8676 | 0.7662 | 0.8374 | 0.9001 |
| **30** | 0.8783 | 0.7830 | 0.8419 | 0.9181 |
| **35** | 0.8960 | 0.8116 | 0.8834 | 0.9089 |
| **40** | 0.8998 | 0.8179 | 0.8949 | 0.9048 |
| **45** | 0.8989 | 0.8163 | 0.8797 | 0.9189 |
| **50** | 0.8803 | 0.7862 | 0.8922 | 0.8687 |
| **55** | 0.8785 | 0.7833 | 0.8713 | 0.8858 |
| **60** | 0.8879 | 0.7983 | 0.8744 | 0.9017 |
| **65** | 0.8835 | 0.7913 | 0.8789 | 0.8881 |
| **70** | 0.8955 | 0.8107 | 0.8881 | 0.9029 |
| **75** | 0.8942 | 0.8087 | 0.8776 | 0.9115 |
| **80** | 0.9008 | 0.8194 | 0.8976 | 0.9039 |
| **85** | 0.9007 | 0.8193 | 0.8858 | 0.9160 |
| **90** | 0.9018 | 0.8212 | 0.9009 | 0.9027 |
| **92 👑** | **0.9078** | **0.8312** | **0.9177** | **0.8982** |
| **100** | 0.9052 | 0.8268 | 0.9061 | 0.9043 |
| **105** | 0.9057 | 0.8276 | 0.9063 | 0.9050 |
| **120** | 0.8982 | 0.8152 | 0.8972 | 0.8993 |
| **130** | 0.8975 | 0.8140 | 0.8839 | 0.9115 |
| **140** | 0.8995 | 0.8173 | 0.9052 | 0.8938 |
| **150** | 0.9035 | 0.8239 | 0.9036 | 0.9033 |

</details>

---

## 🏗️ 模型架构

<details>
<summary><b>📐 点击查看架构详情 →</b></summary>
<br>

```
┌────────────────────────────────────────────────────────────────────┐
│                        TransConvNeXt-CD                            │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   T1 图像 ──┐                                                     │
│               ├── ConvNeXt-Tiny 编码器                              │
│   T2 图像 ──┘       │      │      │      │                       │
│                      ▼      ▼      ▼      ▼                       │
│                [Stage1] [Stage2] [Stage3] [Stage4]                 │
│                  96通道  192通道  384通道  768通道                  │
│                    │       │       │       │                       │
│                    ▼       ▼       ▼       ▼                       │
│               SpatialDiff(96)    │  CrossAttention(768)            │
│                    │       │     │       │                         │
│                    ▼       ▼     ▼       ▼                         │
│               SpatialDiff(192)  │  Fusion(768)                     │
│                    │       │     │       │                         │
│                    ▼       ▼     ▼       ▼                         │
│               SpatialDiff(384)──Dec4──Dec3──Dec2──Dec1             │
│                                         │      │                   │
│                                         ▼      ▼                   │
│                                  DeepSup  DeepSup                  │
│                                         │      │                   │
│                                         ▼      ▼                   │
│                                  ┌───最终卷积 ───┐                │
│                                  │   变化图输出   │                │
│                                  └─────────────────┘              │
│                                                                    │
├────────────────────────────────────────────────────────────────────┤
│  编码器: ConvNeXt-Tiny (timm)  │  损失函数: BCE + Dice (混合)    │
│  注意力: Multihead (12 heads)  │  优化器: AdamW (lr=2e-4)        │
│  解码器: SE + 跳跃连接        │  调度器: CosineAnnealingWR      │
│  TTA: 8倍 (D4 二面体群)       │  精度: FP16 混合精度            │
└────────────────────────────────────────────────────────────────────┘
```

</details>

---

## 🚀 快速开始

### 环境安装

```bash
# 1. 克隆仓库
git clone https://github.com/Starry-Sky-Universe/TransConvNeXt-CD.git
cd TransConvNeXt-CD

# 2. 安装依赖
pip install -r requirements.txt

# 3. (可选) 下载预训练权重
git lfs pull
```

### 训练模型

```bash
# 默认训练 (150轮, batch_size=8)
python src/train.py --data /path/to/LEVIR

# 自定义配置
python src/train.py \
    --data /path/to/LEVIR \
    --save checkpoints/best_model.pth \
    --epochs 200 \
    --batch_size 16
```

### 评估测试

```bash
# 基础评估
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth

# 8x TTA + 可视化
python src/test.py \
    --data /path/to/LEVIR \
    --weight TransConv_SOTA_Best.pth \
    --visualize
```

---

## ⚙️ 超参数

| 设置 | 数值 | 说明 |
|:--------|:-----:|:------------|
| 训练图像尺寸 | 512 × 512 | 从 1024×1024 随机裁剪 |
| 验证/测试图像尺寸 | 1024 × 1024 | 全图推理 |
| 训练批次大小 | 8 | Tesla P100 (16GB) |
| 验证批次大小 | 4 | 全图 1024×1024 |
| 优化器 | AdamW | lr=2e-4, weight_decay=0.05 |
| 学习率调度 | CosineAnnealingWarmRestarts | T₀=15, T_mult=2 |
| 损失函数 | BCE + Dice | 混合损失 |
| 精度 | FP16 | 混合精度训练 |
| 迭代轮数 | 150 | ~6小时 (P100) |

---

## ⚖️ 许可证

本项目采用 **MIT 许可证**。

---

<div align="center">

<br>

## ⭐ 喜欢这个项目？给我们一个 Star！

[![GitHub Stars](https://img.shields.io/github/stars/Starry-Sky-Universe/TransConvNeXt-CD?style=social&label=Star%20this%20repo)](https://github.com/Starry-Sky-Universe/TransConvNeXt-CD)

<br>

*用 ❤️ 和 PyTorch 打造*

</div>