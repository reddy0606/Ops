"""
utils/evaluator.py
────────────────────────────────────────────────────────
Unified evaluation: accuracy, F1, AUC, confusion matrix.
Works for both sklearn (ML) and PyTorch (DL) models.
"""

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    roc_auc_score, confusion_matrix
)
from typing import Optional


# ─────────────────────────────────────────────────────
# ML Evaluator  (sklearn models)
# ─────────────────────────────────────────────────────

def evaluate_sklearn(y_true, y_pred, y_proba=None, class_names=None):
    """
    Evaluate an sklearn model's predictions.

    Args:
        y_true:      Ground truth labels
        y_pred:      Predicted labels
        y_proba:     Predicted probabilities (optional, for AUC)
        class_names: List of class name strings

    Returns:
        dict of metrics
    """
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted")
    cm  = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": round(acc * 100, 2),
        "f1_weighted": round(f1, 4),
        "confusion_matrix": cm,
    }

    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr",
                                average="weighted")
            metrics["auc_roc"] = round(auc, 4)
        except Exception:
            pass

    print("\n" + "─" * 50)
    print(f"  Accuracy  : {metrics['accuracy']}%")
    print(f"  F1 Score  : {metrics['f1_weighted']}")
    if "auc_roc" in metrics:
        print(f"  AUC-ROC   : {metrics['auc_roc']}")
    print("─" * 50)
    print(classification_report(
        y_true, y_pred,
        target_names=class_names if class_names else None
    ))

    return metrics


# ─────────────────────────────────────────────────────
# DL Evaluator  (PyTorch models)
# ─────────────────────────────────────────────────────

def evaluate_pytorch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names: Optional[list] = None,
):
    """
    Run inference on a DataLoader and return metrics.

    Args:
        model:       Trained PyTorch model
        loader:      DataLoader (test or val)
        device:      torch.device
        class_names: Optional list of class names

    Returns:
        dict of metrics
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_proba = np.array(all_probs)

    return evaluate_sklearn(y_true, y_pred, y_proba, class_names)


# ─────────────────────────────────────────────────────
# Training history tracker
# ─────────────────────────────────────────────────────

class TrainingHistory:
    """Records loss and accuracy per epoch for plotting."""

    def __init__(self):
        self.train_loss, self.val_loss = [], []
        self.train_acc,  self.val_acc  = [], []

    def update(self, tr_loss, val_loss, tr_acc, val_acc):
        self.train_loss.append(tr_loss)
        self.val_loss.append(val_loss)
        self.train_acc.append(tr_acc)
        self.val_acc.append(val_acc)

    def best_val_acc(self):
        return max(self.val_acc) if self.val_acc else 0.0

    def summary(self):
        print(f"  Best val accuracy : {self.best_val_acc():.2f}%")
        print(f"  Final train loss  : {self.train_loss[-1]:.4f}")
        print(f"  Final val loss    : {self.val_loss[-1]:.4f}")
