"""
TransConvNeXt-CD: Training script for LEVIR-CD change detection.

Usage:
    python train.py --data /path/to/LEVIR --epochs 150 --batch_size 8
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from config import CFG
from dataset import LEVIR_Dataset
from model import TransConvNeXt_CD
from utils import seed_everything, HybridLoss, Metrics, predict_tta


def train_model(args):
    seed_everything(CFG.seed)

    print(f"[Info] Device: {CFG.device}")
    if torch.cuda.is_available():
        print(f"[Info] GPU: {torch.cuda.get_device_name(0)}")

    # Datasets
    train_ds = LEVIR_Dataset(args.data, "train")
    val_ds = LEVIR_Dataset(args.data, "val")
    print(f"[Info] Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
    )

    # Model
    model = TransConvNeXt_CD().to(CFG.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=args.min_lr
    )

    criterion = HybridLoss()
    scaler = torch.cuda.amp.GradScaler()
    met = Metrics()

    best_f1 = 0.0
    print(">>> Starting TransConvNeXt training (Patch Train / Full Val)...")

    for epoch in range(args.epochs):
        # Training
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for t1, t2, mask in pbar:
            t1, t2, mask = t1.to(CFG.device), t2.to(CFG.device), mask.to(CFG.device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(t1, t2)
                loss = criterion(preds, mask)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()

        # Validation
        met.reset()
        torch.cuda.empty_cache()
        model.eval()
        with torch.no_grad():
            for t1, t2, mask in val_loader:
                t1, t2, mask = t1.to(CFG.device), t2.to(CFG.device), mask.to(CFG.device)
                pred = predict_tta(model, t1, t2)
                met.update(pred, mask)

        f1, iou, precision, recall = met.get()
        print(
            f"[*] Epoch {epoch+1:3d} | F1: {f1:.4f} | IoU: {iou:.4f} "
            f"| Prec: {precision:.4f} | Rec: {recall:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), args.save)
            print(f">>> New best model saved! F1: {best_f1:.4f}")

    print(f"\nTraining complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {args.save}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TransConvNeXt-CD")
    parser.add_argument("--data", type=str, default=CFG.base_path, help="Dataset path")
    parser.add_argument("--save", type=str, default=CFG.model_path, help="Model save path")
    parser.add_argument("--epochs", type=int, default=CFG.epochs, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=CFG.train_batch_size, help="Batch size")
    parser.add_argument("--val_batch_size", type=int, default=CFG.val_batch_size, help="Val batch size")
    parser.add_argument("--lr", type=float, default=CFG.lr, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=CFG.min_lr, help="Min learning rate")
    parser.add_argument("--weight_decay", type=float, default=CFG.weight_decay, help="Weight decay")
    args = parser.parse_args()
    train_model(args)