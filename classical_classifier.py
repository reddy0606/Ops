"""
ml/classical_classifier.py
────────────────────────────────────────────────────────
MACHINE LEARNING — Classical Vision Classification

Pipeline:
  1. Load CIFAR-10 images
  2. Extract HOG (Histogram of Oriented Gradients) features
  3. Train SVM and Random Forest classifiers
  4. Evaluate and compare both models
  5. Save confusion matrix + metrics plot

Why HOG?  Raw pixels don't work well for classical ML.
HOG captures edge/gradient structure — much better features.
"""

import sys, time
sys.path.append("..")

import numpy as np
from pathlib import Path
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from joblib import dump, load
from skimage.feature import hog
from skimage.color import rgb2gray
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.evaluator import evaluate_sklearn
from utils.visualizer import plot_confusion_matrix, plot_model_comparison

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ─────────────────────────────────────────────────────
# Step 1: Load data as numpy arrays
# ─────────────────────────────────────────────────────

def load_data(n_train=10000, n_test=2000):
    """
    Load CIFAR-10 and return raw numpy arrays.
    Subsamples for speed (full = 50K train, 10K test).
    """
    print("[ML] Loading CIFAR-10...")
    tf = transforms.Compose([transforms.ToTensor()])

    train_ds = torchvision.datasets.CIFAR10("./data", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=tf)

    def to_numpy(ds, n):
        loader = DataLoader(ds, batch_size=n, shuffle=True)
        imgs, labels = next(iter(loader))
        # imgs: (N, 3, 32, 32) → (N, 32, 32, 3)
        return imgs.permute(0, 2, 3, 1).numpy(), labels.numpy()

    X_train, y_train = to_numpy(train_ds, n_train)
    X_test,  y_test  = to_numpy(test_ds,  n_test)
    print(f"[ML] Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────
# Step 2: HOG feature extraction
# ─────────────────────────────────────────────────────

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """
    Extract HOG features from a batch of RGB images.

    HOG = Histogram of Oriented Gradients:
    - Divides image into small cells
    - Computes gradient direction histogram per cell
    - Normalises across blocks for illumination invariance
    - Result: compact, discriminative feature vector (~324 dims)

    Args:
        images: (N, H, W, 3) float32 in [0, 1]
    Returns:
        features: (N, hog_dim) float32
    """
    print(f"[ML] Extracting HOG features from {len(images)} images...")
    features = []
    for img in images:
        gray = rgb2gray(img)           # → (32, 32)
        feat = hog(
            gray,
            orientations=9,            # 9 gradient bins
            pixels_per_cell=(4, 4),    # cell size
            cells_per_block=(2, 2),    # normalisation block
            feature_vector=True
        )
        features.append(feat)
    features = np.array(features, dtype=np.float32)
    print(f"[ML] HOG feature shape: {features.shape}")
    return features


# ─────────────────────────────────────────────────────
# Step 3: Train SVM
# ─────────────────────────────────────────────────────

def train_svm(X_train, y_train):
    """
    SVM with RBF kernel + PCA dimensionality reduction.

    Pipeline:
        StandardScaler → PCA(150) → SVC(rbf kernel)
    """
    print("\n[ML] Training SVM (this may take ~1-2 min)...")
    t0 = time.time()

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=150, random_state=42)),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=42))
    ])
    svm_pipe.fit(X_train, y_train)
    print(f"[ML] SVM trained in {time.time()-t0:.1f}s")
    return svm_pipe


# ─────────────────────────────────────────────────────
# Step 4: Train Random Forest
# ─────────────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    """
    Random Forest — ensemble of decision trees.
    Faster than SVM, interpretable feature importances.
    """
    print("\n[ML] Training Random Forest...")
    t0 = time.time()

    rf_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf",     RandomForestClassifier(
                       n_estimators=200,
                       max_depth=20,
                       min_samples_split=5,
                       n_jobs=-1,       # use all CPU cores
                       random_state=42
                   ))
    ])
    rf_pipe.fit(X_train, y_train)
    print(f"[ML] Random Forest trained in {time.time()-t0:.1f}s")
    return rf_pipe


# ─────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────

def run_ml_pipeline():
    print("\n" + "="*55)
    print("  MACHINE LEARNING — Image Classification")
    print("  Models: SVM  |  Random Forest")
    print("  Features: HOG (Histogram of Oriented Gradients)")
    print("="*55)

    # 1. Load
    X_train_raw, y_train, X_test_raw, y_test = load_data(
        n_train=10000, n_test=2000
    )

    # 2. Extract HOG features
    X_train = extract_hog_features(X_train_raw)
    X_test  = extract_hog_features(X_test_raw)

    results = {}

    # 3. SVM
    svm_model = train_svm(X_train, y_train)
    y_pred_svm   = svm_model.predict(X_test)
    y_proba_svm  = svm_model.predict_proba(X_test)
    print("\n[ML] SVM Results:")
    metrics_svm = evaluate_sklearn(y_test, y_pred_svm, y_proba_svm, CIFAR10_CLASSES)
    results["SVM"] = metrics_svm["accuracy"]
    plot_confusion_matrix(
        metrics_svm["confusion_matrix"],
        CIFAR10_CLASSES,
        title="SVM — Confusion Matrix",
        save_as="ml_svm_confusion.png"
    )

    # 4. Random Forest
    rf_model = train_random_forest(X_train, y_train)
    y_pred_rf   = rf_model.predict(X_test)
    y_proba_rf  = rf_model.predict_proba(X_test)
    print("\n[ML] Random Forest Results:")
    metrics_rf = evaluate_sklearn(y_test, y_pred_rf, y_proba_rf, CIFAR10_CLASSES)
    results["Random Forest"] = metrics_rf["accuracy"]
    plot_confusion_matrix(
        metrics_rf["confusion_matrix"],
        CIFAR10_CLASSES,
        title="Random Forest — Confusion Matrix",
        save_as="ml_rf_confusion.png"
    )

    # 5. Save best model
    best_name  = max(results, key=results.get)
    best_model = svm_model if best_name == "SVM" else rf_model
    model_path = OUTPUT_DIR / "best_ml_model.joblib"
    dump(best_model, model_path)
    print(f"\n[ML] Best model ({best_name}: {results[best_name]}%) saved → {model_path}")

    return results


if __name__ == "__main__":
    run_ml_pipeline()
