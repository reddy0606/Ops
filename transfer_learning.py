"""
dl/transfer_learning.py
────────────────────────────────────────────────────────
DEEP LEARNING — Transfer Learning with ResNet50

What is Transfer Learning?
  A model pre-trained on ImageNet (1.2M images, 1000 classes)
  already knows how to detect edges, textures, shapes, objects.
  We freeze those layers and only train the final classifier
  head on our dataset → fast, accurate, needs less data.

Strategy:
  Phase 1 — Feature extraction (freeze backbone, train head only)
  Phase 2 — Fine-tuning (unfreeze last ResNet block + head, low LR)
"""

import sys
sys.path.append("..")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.models as models
from pathlib import Path

from utils.data_loader import load_cifar10
from utils.evaluator import evaluate_pytorch, TrainingHistory
from utils.visualizer import plot_training_curves, plot_confusion_matrix

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────
# Build model: ResNet50 with custom head
# ─────────────────────────────────────────────────────

def build_resnet50(num_classes: int, freeze_backbone: bool = True):
    """
    Load pretrained ResNet50, replace the final FC layer.

    Args:
        num_classes:      Number of output classes
        freeze_backbone:  If True, only train the new head

    Returns:
        model with modified classifier head
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace final fully-connected layer
    in_features = model.fc.in_features   # 2048 for ResNet50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model


def unfreeze_last_block(model):
    """
    Unfreeze ResNet layer4 (deepest feature block) + head
    for Phase 2 fine-tuning.
    """
    for name, param in model.named_parameters():
        if "layer4" in name or "fc" in name:
            param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Transfer] Unfrozen layer4 + head → trainable params: {trainable:,}")


# ─────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train(train)
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total_loss += loss.item() * len(imgs)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(imgs)

    return total_loss / total, 100.0 * correct / total


# ─────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────

def run_transfer_learning(
    phase1_epochs: int = 10,
    phase2_epochs: int = 15,
    batch_size: int = 64,
):
    print("\n" + "="*55)
    print("  DEEP LEARNING — Transfer Learning (ResNet50)")
    print(f"  Device  : {DEVICE}")
    print(f"  Phase 1 : {phase1_epochs} epochs (head only)")
    print(f"  Phase 2 : {phase2_epochs} epochs (fine-tune layer4)")
    print("="*55)

    # CIFAR-10 images are 32×32 but ResNet expects 224×224
    train_loader, val_loader, test_loader, class_names = load_cifar10(
        batch_size=batch_size, img_size=224
    )

    model     = build_resnet50(num_classes=len(class_names), freeze_backbone=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history   = TrainingHistory()
    best_acc  = 0.0
    best_path = OUTPUT_DIR / "resnet50_best.pth"

    # ── Phase 1: Train head only ─────────────────────
    print(f"\n[Transfer] Phase 1 — Training head (backbone frozen)")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase1_epochs, eta_min=1e-5)

    for epoch in range(1, phase1_epochs + 1):
        tr_loss, tr_acc  = run_epoch(model, train_loader, optimizer, criterion, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader,  optimizer, criterion, DEVICE, train=False)
        history.update(tr_loss, val_loss, tr_acc, val_acc)
        scheduler.step()
        print(f"  [P1] Epoch {epoch:02d}/{phase1_epochs} | "
              f"Loss {tr_loss:.4f}/{val_loss:.4f} | Acc {tr_acc:.1f}%/{val_acc:.1f}%", end="")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print("  ← best")
        else:
            print()

    # ── Phase 2: Fine-tune layer4 ────────────────────
    print(f"\n[Transfer] Phase 2 — Fine-tuning (layer4 + head unfrozen)")
    unfreeze_last_block(model)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=phase2_epochs, eta_min=1e-6)

    for epoch in range(1, phase2_epochs + 1):
        tr_loss, tr_acc  = run_epoch(model, train_loader, optimizer, criterion, DEVICE, train=True)
        val_loss, val_acc = run_epoch(model, val_loader,  optimizer, criterion, DEVICE, train=False)
        history.update(tr_loss, val_loss, tr_acc, val_acc)
        scheduler.step()
        print(f"  [P2] Epoch {epoch:02d}/{phase2_epochs} | "
              f"Loss {tr_loss:.4f}/{val_loss:.4f} | Acc {tr_acc:.1f}%/{val_acc:.1f}%", end="")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print("  ← best")
        else:
            print()

    # ── Evaluate ──────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    print(f"\n[Transfer] Test set evaluation (best val_acc={best_acc:.1f}%):")
    metrics = evaluate_pytorch(model, test_loader, DEVICE, class_names)

    # ── Plots ─────────────────────────────────────────
    plot_training_curves(history, title="ResNet50 Transfer Learning",
                         save_as="transfer_training.png")
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        title="ResNet50 — Confusion Matrix",
        save_as="transfer_confusion.png"
    )

    print(f"\n[Transfer] Done — best val acc: {best_acc:.1f}%")
    return {"ResNet50": metrics["accuracy"]}


if __name__ == "__main__":
    run_transfer_learning(phase1_epochs=10, phase2_epochs=15)
