"""
TransConvNeXt-CD: Utility functions for training and evaluation.
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class HybridLoss(nn.Module):
    """BCE + Dice hybrid loss with deep supervision support."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target):
        probs = torch.sigmoid(pred)
        intersection = (probs * target).sum()
        union = probs.sum() + target.sum()
        return 1 - (2.0 * intersection + 1e-6) / (union + 1e-6)

    def forward(self, preds, target):
        if isinstance(preds, tuple):
            final, d4, d3 = preds
            t_d3 = F.interpolate(target, size=d3.shape[2:], mode="nearest")
            t_d4 = F.interpolate(target, size=d4.shape[2:], mode="nearest")

            l_main = self.bce(final, target) + self.dice_loss(final, target)
            l_d3 = self.bce(d3, t_d3) + self.dice_loss(d3, t_d3)
            l_d4 = self.bce(d4, t_d4) + self.dice_loss(d4, t_d4)
            return l_main + 0.4 * l_d3 + 0.2 * l_d4
        return self.bce(preds, target) + self.dice_loss(preds, target)


class Metrics:
    """Pixel-level change detection metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, pred, target, threshold=0.5):
        pred = (pred > threshold).float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()

    def get(self):
        f1 = 2 * self.tp / (2 * self.tp + self.fp + self.fn + 1e-8)
        iou = self.tp / (self.tp + self.fp + self.fn + 1e-8)
        precision = self.tp / (self.tp + self.fp + 1e-8)
        recall = self.tp / (self.tp + self.fn + 1e-8)
        return f1, iou, precision, recall


def predict_tta(model, t1, t2):
    """3x TTA (horizontal + vertical flip averaging)."""
    model.eval()
    out = torch.sigmoid(model(t1, t2))
    out += torch.flip(
        torch.sigmoid(model(torch.flip(t1, [3]), torch.flip(t2, [3]))), [3]
    )
    out += torch.flip(
        torch.sigmoid(model(torch.flip(t1, [2]), torch.flip(t2, [2]))), [2]
    )
    return out / 3.0


def predict_tta_8x(model, t1, t2):
    """8x TTA using D4 dihedral group (4 rotations x 2 flips)."""
    model.eval()
    preds = []

    for rot in range(4):
        for flip in [False, True]:
            t1_aug = torch.rot90(t1, rot, dims=[2, 3])
            t2_aug = torch.rot90(t2, rot, dims=[2, 3])
            if flip:
                t1_aug = torch.flip(t1_aug, [3])
                t2_aug = torch.flip(t2_aug, [3])

            out = torch.sigmoid(model(t1_aug, t2_aug))

            if flip:
                out = torch.flip(out, [3])
            out = torch.rot90(out, -rot, dims=[2, 3])
            preds.append(out)

    return torch.stack(preds).mean(dim=0)


def denormalize(tensor):
    """Denormalize image tensor for visualization."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return np.clip(img, 0, 1)