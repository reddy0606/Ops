"""
utils/data_loader.py
────────────────────────────────────────────────────────
Handles dataset loading, preprocessing & augmentation.
Works with CIFAR-10 (default) or your own image folder.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import numpy as np
from pathlib import Path

# ── CIFAR-10 class names ──────────────────────────────
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# ─────────────────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────────────────

def get_transforms(mode: str = "train", img_size: int = 32):
    """
    Returns torchvision transforms for train or eval.

    Args:
        mode:     'train' (with augmentation) or 'eval'
        img_size: resize target (32 for CIFAR, 224 for ResNet)
    """
    mean = (0.4914, 0.4822, 0.4465)   # CIFAR-10 stats
    std  = (0.2470, 0.2435, 0.2616)

    if mode == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(img_size, padding=4),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # eval / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─────────────────────────────────────────────────────
# CIFAR-10 loader  (auto-downloads ~170 MB once)
# ─────────────────────────────────────────────────────

def load_cifar10(
    data_dir: str = "./data",
    batch_size: int = 64,
    img_size: int = 32,
    val_split: float = 0.1,
    num_workers: int = 2,
):
    """
    Downloads and returns CIFAR-10 DataLoaders.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    train_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=get_transforms("train", img_size)
    )
    test_ds = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=get_transforms("eval", img_size)
    )

    # Split train → train + val
    val_size   = int(len(train_ds) * val_split)
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(
        train_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"[DataLoader] Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    return train_loader, val_loader, test_loader, CIFAR10_CLASSES


# ─────────────────────────────────────────────────────
# Custom image folder loader  (your own dataset)
# ─────────────────────────────────────────────────────

def load_custom_dataset(
    root_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    val_split: float = 0.2,
    num_workers: int = 2,
):
    """
    Load your own dataset structured as:
        root_dir/
            class_a/  img1.jpg  img2.jpg ...
            class_b/  img1.jpg  img2.jpg ...

    Returns:
        train_loader, val_loader, class_names
    """
    full_ds = torchvision.datasets.ImageFolder(
        root=root_dir,
        transform=get_transforms("train", img_size)
    )
    val_size   = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(
        full_ds, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Val should use eval transforms (no augmentation)
    val_ds.dataset.transform = get_transforms("eval", img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)

    print(f"[Custom Dataset] Classes: {full_ds.classes}")
    print(f"[Custom Dataset] Train: {len(train_ds):,} | Val: {len(val_ds):,}")
    return train_loader, val_loader, full_ds.classes


# ─────────────────────────────────────────────────────
# Helper: get flat numpy arrays for ML (classical)
# ─────────────────────────────────────────────────────

def get_numpy_arrays(loader: DataLoader):
    """
    Flatten all batches into numpy arrays for sklearn ML models.

    Returns:
        X: (N, C*H*W) float32
        y: (N,)       int
    """
    X_list, y_list = [], []
    for imgs, labels in loader:
        X_list.append(imgs.numpy().reshape(len(imgs), -1))
        y_list.append(labels.numpy())
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    print(f"[get_numpy_arrays] X={X.shape}, y={y.shape}")
    return X, y
