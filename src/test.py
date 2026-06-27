"""
TransConvNeXt-CD: Evaluation script for LEVIR-CD change detection.

Usage:
    python test.py --data /path/to/LEVIR --weight /path/to/model.pth
"""

import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from config import CFG
from dataset import LEVIR_Dataset
from model import TransConvNeXt_CD
from utils import seed_everything, Metrics, predict_tta_8x, denormalize


def calculate_single_metrics(pred, target):
    """Calculate per-image metrics."""
    pred_bin = (pred > 0.5).astype("float32")
    target_bin = target.astype("float32")
    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    p = tp / (tp + fp + 1e-8)
    r = tp / (tp + fn + 1e-8)
    return f1, iou, p, r


def evaluate(args):
    seed_everything(CFG.seed)
    print(f"[Info] Device: {CFG.device}")

    # Dataset
    test_ds = LEVIR_Dataset(args.data, "test")
    test_loader = DataLoader(
        test_ds,
        batch_size=CFG.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )
    print(f"[Info] Test samples: {len(test_ds)}")

    # Model
    model = TransConvNeXt_CD().to(CFG.device)
    model.load_state_dict(torch.load(args.weight, map_location=CFG.device))
    model.eval()

    # Evaluation
    met = Metrics()
    print("\n>>> Running evaluation with 8x TTA...")
    with torch.no_grad():
        for t1, t2, mask in tqdm(test_loader, desc="Testing"):
            t1, t2, mask = t1.to(CFG.device), t2.to(CFG.device), mask.to(CFG.device)
            pred = predict_tta_8x(model, t1, t2)
            met.update(pred, mask)

    f1, iou, precision, recall = met.get()
    print("\n" + "=" * 45)
    print("Test Set Metrics (8x TTA)")
    print("=" * 45)
    print(f"  IoU       (Intersection over Union): {iou:.4f}")
    print(f"  F1 Score  (F1 Score):                {f1:.4f}")
    print(f"  Precision (Precision):               {precision:.4f}")
    print(f"  Recall    (Recall):                  {recall:.4f}")
    print("=" * 45)

    # Visualization
    if args.visualize:
        print("\n>>> Generating visualization...")
        indices = random.sample(range(len(test_ds)), min(4, len(test_ds)))
        fig, axes = plt.subplots(len(indices), 4, figsize=(22, 22))

        with torch.no_grad():
            for row_idx, sample_idx in enumerate(indices):
                t1_t, t2_t, mask_t = test_ds[sample_idx]
                t1_in = t1_t.unsqueeze(0).to(CFG.device)
                t2_in = t2_t.unsqueeze(0).to(CFG.device)

                pred_t = predict_tta_8x(model, t1_in, t2_in)
                pred_np = pred_t[0, 0].cpu().numpy()
                mask_np = mask_t[0].cpu().numpy()
                pred_mask = (pred_np > 0.5).astype("float32")

                sf1, siou, sp, sr = calculate_single_metrics(pred_np, mask_np)

                img1 = denormalize(t1_t)
                img2 = denormalize(t2_t)

                ax = axes[row_idx]
                ax[0].imshow(img1)
                ax[0].axis("off")
                if row_idx == 0: ax[0].set_title("Image T1", fontsize=16)

                ax[1].imshow(img2)
                ax[1].axis("off")
                if row_idx == 0: ax[1].set_title("Image T2", fontsize=16)

                ax[2].imshow(mask_np, cmap="gray")
                ax[2].axis("off")
                if row_idx == 0: ax[2].set_title("Ground Truth", fontsize=16)

                ax[3].imshow(pred_mask, cmap="gray")
                ax[3].axis("off")
                color = "darkred" if siou < 0.5 else "darkgreen"
                ax[3].set_title(
                    f"Prediction\nIoU:{siou:.4f} F1:{sf1:.4f}\nP:{sp:.4f} R:{sr:.4f}",
                    fontsize=13, color=color,
                )

        plt.tight_layout()
        out_path = args.output or "results.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {out_path}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate TransConvNeXt-CD")
    parser.add_argument("--data", type=str, default=CFG.base_path, help="Dataset path")
    parser.add_argument("--weight", type=str, default=CFG.model_path, help="Model weight path")
    parser.add_argument("--visualize", action="store_true", help="Generate visualization")
    parser.add_argument("--output", type=str, default="results.png", help="Output path")
    args = parser.parse_args()
    evaluate(args)