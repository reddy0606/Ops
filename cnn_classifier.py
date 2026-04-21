"""
dl/cnn_classifier.py
────────────────────────────────────────────────────────
DEEP LEARNING — Custom CNN Image Classifier (PyTorch)

Architecture:
  Conv Block 1  →  Conv Block 2  →  Conv Block 3
  → GlobalAvgPool → FC(256) → Dropout → FC(num_classes)

Features:
  - Batch Normalisation after every conv layer
  - Dropout for regularisation
  - OneCycleLR learning rate scheduler
  - Early stopping to prevent overfitting
  - Saves best model checkpoint
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from pathlib import Path
import numpy as np

from utils.data_loader import load_cifar10
from utils.evaluator import evaluate_pytorch, TrainingHistory
from utils.visualizer import plot_training_curves, plot_confusion_matrix, plot_sample_predictions

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────
# Model Architecture
# ─────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU → (optional MaxPool)"""

    def __init__(self, in_ch, out_ch, pool=False):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))     # halve spatial dims
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class VisionCNN(nn.Module):
    """
    Custom CNN for CIFAR-10 (32×32 input).

    Layer flow:
      Input (3,32,32)
        → Block1 (64,32,32) → Pool → (64,16,16)
        → Block2 (128,16,16) → Pool → (128,8,8)
        → Block3 (256,8,8)  → Pool → (256,4,4)
        → GlobalAvgPool     → (256,1,1) → flatten (256,)
        → FC(256→128) → Dropout(0.4) → FC(128→10)
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.block1 = ConvBlock(3,   64,  pool=True)
        self.block2 = ConvBlock(64,  128, pool=True)
        self.block3 = ConvBlock(128, 256, pool=True)
        self.gap    = nn.AdaptiveAvgPool2d(1)   # Global Average Pool
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        return self.head(x)


# ─────────────────────────────────────────────────────
# Training loop (one epoch)
# ─────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scheduler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()     # OneCycleLR steps per batch

        total_loss += loss.item() * len(imgs)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += len(imgs)

    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * len(imgs)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += len(imgs)

    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────

def run_cnn_pipeline(epochs: int = 30, batch_size: int = 128, lr: float = 0.01):
    print("\n" + "="*55)
    print("  DEEP LEARNING — Custom CNN")
    print(f"  Device : {DEVICE}")
    print(f"  Epochs : {epochs}  |  Batch: {batch_size}  |  LR: {lr}")
    print("="*55)

    # 1. Data
    train_loader, val_loader, test_loader, class_names = load_cifar10(
        batch_size=batch_size, img_size=32
    )

    # 2. Model
    model     = VisionCNN(num_classes=len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer, max_lr=lr,
        steps_per_epoch=len(train_loader), epochs=epochs
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,}")

    # 3. Training with early stopping
    history   = TrainingHistory()
    best_acc  = 0.0
    patience  = 5
    no_improve = 0
    best_path  = OUTPUT_DIR / "cnn_best.pth"

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc  = train_one_epoch(model, train_loader, optimizer,
                                            criterion, scheduler, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        history.update(tr_loss, val_loss, tr_acc, val_acc)

        print(f"  Epoch {epoch:02d}/{epochs} | "
              f"Loss {tr_loss:.4f}/{val_loss:.4f} | "
              f"Acc {tr_acc:.1f}%/{val_acc:.1f}%", end="")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            no_improve = 0
            print("  ← best")
        else:
            no_improve += 1
            print()

        if no_improve >= patience:
            print(f"\n[CNN] Early stopping at epoch {epoch}")
            break

    # 4. Load best weights & evaluate on test set
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"\n[CNN] Test set evaluation (best model: val_acc={best_acc:.1f}%):")
    metrics = evaluate_pytorch(model, test_loader, DEVICE, class_names)

    # 5. Plots
    plot_training_curves(history, title="Custom CNN", save_as="cnn_training.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        title="CNN — Confusion Matrix", save_as="cnn_confusion.png"
    )

    # Sample predictions
    model.eval()
    imgs_batch, labels_batch = next(iter(test_loader))
    with torch.no_grad():
        preds = model(imgs_batch.to(DEVICE)).argmax(1).cpu().numpy()
    plot_sample_predictions(
        imgs_batch.numpy(), labels_batch.numpy(), preds,
        class_names, n=16, save_as="cnn_sample_preds.png"
    )

    print(f"\n[CNN] Done — best val acc: {best_acc:.1f}%")
    return {"Custom CNN": metrics["accuracy"]}


if __name__ == "__main__":
    run_cnn_pipeline(epochs=30, batch_size=128, lr=0.01)
