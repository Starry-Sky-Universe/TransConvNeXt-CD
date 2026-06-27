"""
TransConvNeXt-CD: Transformer-CNN Hybrid Network for Change Detection

Configuration module containing all hyperparameters and settings.
"""

import torch


class CFG:
    # Reproducibility
    seed = 42

    # Image sizes
    train_img_size = 512   # Training crop size
    val_img_size = 1024    # Validation full image size

    # Batch sizes
    train_batch_size = 8   # P100 can handle 8-16 at 512x512
    val_batch_size = 4     # Smaller for 1024x1024 to avoid OOM

    # Training
    epochs = 150
    lr = 2e-4
    min_lr = 1e-6
    weight_decay = 0.05
    num_workers = 2

    # Paths
    base_path = "/kaggle/input/levir-cd/LEVIR CD"
    model_path = "TransConv_SOTA_Best.pth"

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")