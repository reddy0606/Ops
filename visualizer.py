"""
utils/visualizer.py
────────────────────────────────────────────────────────
Plotting helpers: training curves, confusion matrix,
sample predictions, and metric comparison bar chart.
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────
# 1. Training curves (loss + accuracy per epoch)
# ─────────────────────────────────────────────────────

def plot_training_curves(history, title: str = "Training", save_as: str = "training.png"):
    """
    Plot loss and accuracy curves side-by-side.

    Args:
        history:  TrainingHistory object (has .train_loss, .val_loss etc.)
        title:    Plot title string
        save_as:  Filename inside /outputs/
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.train_loss) + 1)

    # Loss
    ax1.plot(epochs, history.train_loss, "b-o", markersize=4, label="Train loss")
    ax1.plot(epochs, history.val_loss,   "r-o", markersize=4, label="Val loss")
    ax1.set_title(f"{title} — Loss", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history.train_acc, "b-o", markersize=4, label="Train acc")
    ax2.plot(epochs, history.val_acc,   "r-o", markersize=4, label="Val acc")
    ax2.set_title(f"{title} — Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / save_as
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved → {path}")


# ─────────────────────────────────────────────────────
# 2. Confusion matrix heatmap
# ─────────────────────────────────────────────────────

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list,
    title: str = "Confusion Matrix",
    save_as: str = "confusion_matrix.png",
):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = OUTPUT_DIR / save_as
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved → {path}")


# ─────────────────────────────────────────────────────
# 3. Sample predictions grid
# ─────────────────────────────────────────────────────

def plot_sample_predictions(
    images: np.ndarray,
    true_labels: list,
    pred_labels: list,
    class_names: list,
    n: int = 16,
    save_as: str = "sample_predictions.png",
):
    """
    Show n sample images with true vs predicted label.
    Green title = correct, Red title = wrong.
    """
    n = min(n, len(images))
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i in range(n):
        img = np.transpose(images[i], (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)

        axes[i].imshow(img)
        axes[i].axis("off")
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = "green" if true_labels[i] == pred_labels[i] else "red"
        axes[i].set_title(f"T: {true_name}\nP: {pred_name}",
                          fontsize=8, color=color, fontweight="bold")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample Predictions (green=correct, red=wrong)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUTPUT_DIR / save_as
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved → {path}")


# ─────────────────────────────────────────────────────
# 4. Model comparison bar chart
# ─────────────────────────────────────────────────────

def plot_model_comparison(results: dict, save_as: str = "model_comparison.png"):
    """
    Bar chart comparing accuracy across all models.

    Args:
        results: {"Model Name": accuracy_float, ...}
                 e.g. {"SVM": 58.2, "CNN": 74.5, "ResNet50": 88.1}
    """
    names = list(results.keys())
    accs  = list(results.values())
    colors = ["#7F77DD", "#1D9E75", "#BA7517", "#D85A30", "#D4537E"]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, accs, color=colors[:len(names)],
                  edgecolor="white", linewidth=1.5, width=0.5)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_title("Model Comparison — Test Accuracy",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    path = OUTPUT_DIR / save_as
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Visualizer] Saved → {path}")
