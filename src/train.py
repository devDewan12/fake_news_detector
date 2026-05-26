"""
============================================================
STEP 6 — TRAINING PIPELINE
============================================================
Module: train.py

6a. Load data/cleaned_data.csv.
6b. Compute X_meta, y via feature_engineering.
6c. Get / load BERT embeddings X_bert.
6d. 80/20 train-test split (stratified, random_state=42).
6e. Further split train 80/20 -> train / validation.
6f. FakeNewsDataset(Dataset) returning tensors.
6g. DataLoaders (batch_size=32, shuffle=True for train).
6h. Train FakeNewsClassifier for up to 20 epochs.
6i. Log train/val loss, acc, F1, ROC-AUC per epoch.
6j. Early stopping (patience=3 on val_loss).
6k. Save best weights -> models/best_fake_news_model.pt
6l. Train a baseline LogisticRegression on metadata only.
6m. Save training curves -> plots/training_curves.png

Class imbalance handled with compute_class_weight (Optimization O6).
Reproducibility seeds set everywhere (Optimization O1).
"""

from __future__ import annotations

import gc
import os
import random
from typing import Dict, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

# Local modules
from bert_embeddings import load_or_extract
from feature_engineering import engineer_features
from model import (
    EarlyStopping,
    FakeNewsClassifier,
    build_loss,
    build_optimizer,
    build_scheduler,
)

# ------------------------------------------------------------------ #
# Reproducibility (Optimization O1)
# ------------------------------------------------------------------ #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
for _d in (MODELS_DIR, PLOTS_DIR):
    os.makedirs(_d, exist_ok=True)

CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_data.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.pt")
BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
SPLIT_IDX_PATH = os.path.join(MODELS_DIR, "split_indices.joblib")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MAX_EPOCHS = 20


# ================================================================== #
# 6f — PyTorch Dataset
# ================================================================== #
class FakeNewsDataset(Dataset):
    """Returns ``(bert_vec, meta_vec, label)`` tensors."""

    def __init__(self, x_bert: np.ndarray, x_meta: np.ndarray,
                 y: np.ndarray) -> None:
        """Initialise the dataset.

        Args:
            x_bert: BERT embedding matrix ``(n, 768)``.
            x_meta: Metadata matrix ``(n, n_meta)``.
            y: Label vector ``(n,)``.
        """
        self.x_bert = torch.tensor(x_bert, dtype=torch.float32)
        self.x_meta = torch.tensor(x_meta, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.y)

    def __getitem__(self, idx: int):
        """Return one ``(bert, meta, label)`` triple."""
        return self.x_bert[idx], self.x_meta[idx], self.y[idx]


# ================================================================== #
# Epoch helpers
# ================================================================== #
def _run_epoch(model, loader, loss_fn, optimizer=None) -> Dict[str, float]:
    """Run one train (optimizer given) or eval (optimizer=None) epoch.

    Args:
        model: The classifier.
        loader: A DataLoader.
        loss_fn: The loss module.
        optimizer: If provided, runs in training mode.

    Returns:
        Dict with loss, accuracy, f1, and roc_auc for the epoch.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, n = 0.0, 0
    all_y, all_p = [], []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for xb, xm, yb in loader:
            xb, xm, yb = xb.to(DEVICE), xm.to(DEVICE), yb.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            preds = model(xb, xm)
            loss = loss_fn(preds, yb)
            if is_train:
                loss.backward()
                optimizer.step()

            bs = yb.size(0)
            total_loss += loss.item() * bs
            n += bs
            all_y.append(yb.detach().cpu().numpy())
            all_p.append(preds.detach().cpu().numpy())

    y_true = np.vstack(all_y).flatten()
    y_prob = np.vstack(all_p).flatten()
    y_pred = (y_prob >= 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "loss": total_loss / max(n, 1),
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": auc,
    }


def _plot_curves(history: Dict[str, list]) -> None:
    """Plot + save training/validation loss & accuracy (Step 6m).

    Args:
        history: Dict of per-epoch metric lists.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc")
    axes[1].set_title("Training vs Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "training_curves.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[OK] Training curves saved to {out}")


# ================================================================== #
# Main training routine
# ================================================================== #
def train(force_bert: bool = False) -> None:
    """Run the full Step-6 training pipeline.

    Args:
        force_bert: If True, recompute BERT embeddings even if cached.
    """
    # ---- 6a: load cleaned data
    if not os.path.exists(CLEANED_CSV):
        raise FileNotFoundError(
            f"{CLEANED_CSV} not found. Run data_preprocessing.py first."
        )
    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    print(f"[DATA] Loaded {len(df):,} rows from cleaned_data.csv")

    # ---- 6b: metadata features + labels
    x_meta, y = engineer_features(df, fit=True)
    print(f"[FEATS] X_meta shape: {x_meta.shape}")

    # ---- 6c: BERT embeddings (cached)
    texts = df["combined_text"].astype(str).tolist()
    x_bert = load_or_extract(texts, split="all", batch_size=16,
                             force=force_bert)
    print(f"[BERT] X_bert shape: {x_bert.shape}")

    # ---- 6d: 80/20 train-test split (stratified)
    idx = np.arange(len(y))
    idx_train_full, idx_test = train_test_split(
        idx, test_size=0.20, stratify=y, random_state=SEED
    )
    # ---- 6e: split train -> train/val (80/20)
    idx_train, idx_val = train_test_split(
        idx_train_full, test_size=0.20,
        stratify=y[idx_train_full], random_state=SEED
    )
    joblib.dump(
        {"train": idx_train, "val": idx_val, "test": idx_test},
        SPLIT_IDX_PATH,
    )
    print(f"[SPLIT] train={len(idx_train)} val={len(idx_val)} "
          f"test={len(idx_test)}")

    # ---- 6f / 6g: datasets + loaders
    train_ds = FakeNewsDataset(x_bert[idx_train], x_meta[idx_train],
                               y[idx_train])
    val_ds = FakeNewsDataset(x_bert[idx_val], x_meta[idx_val], y[idx_val])
    test_ds = FakeNewsDataset(x_bert[idx_test], x_meta[idx_test],
                              y[idx_test])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ---- Class weights (Optimization O6)
    classes = np.unique(y[idx_train])
    cw = compute_class_weight("balanced", classes=classes,
                              y=y[idx_train])
    weight_map = dict(zip(classes, cw))
    pos_weight = float(weight_map.get(1.0, 1.0) / weight_map.get(0.0, 1.0))
    print(f"[CLASS WEIGHTS] {weight_map}  -> pos_weight={pos_weight:.4f}")

    # ---- 6h: build + train model
    model = FakeNewsClassifier(bert_dim=x_bert.shape[1],
                               meta_dim=x_meta.shape[1]).to(DEVICE)
    loss_fn = build_loss(pos_weight=pos_weight)
    optimizer = build_optimizer(model, lr=1e-3)
    scheduler = build_scheduler(optimizer)
    stopper = EarlyStopping(patience=3)

    history = {k: [] for k in (
        "train_loss", "val_loss", "train_acc", "val_acc",
        "val_f1", "val_auc")}

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    for epoch in range(1, MAX_EPOCHS + 1):
        tr = _run_epoch(model, train_loader, loss_fn, optimizer)
        va = _run_epoch(model, val_loader, loss_fn, optimizer=None)
        scheduler.step(va["loss"])

        history["train_loss"].append(tr["loss"])
        history["val_loss"].append(va["loss"])
        history["train_acc"].append(tr["acc"])
        history["val_acc"].append(va["acc"])
        history["val_f1"].append(va["f1"])
        history["val_auc"].append(va["roc_auc"])

        # ---- 6i: epoch logging
        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"train_loss={tr['loss']:.4f} train_acc={tr['acc']:.4f} | "
            f"val_loss={va['loss']:.4f} val_acc={va['acc']:.4f} "
            f"val_f1={va['f1']:.4f} val_auc={va['roc_auc']:.4f}"
        )

        # ---- 6j / 6k: early stopping + best-model checkpoint
        improved = stopper.step(va["loss"])
        if improved:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"   [CKPT] New best model saved -> {BEST_MODEL_PATH}")
        if stopper.should_stop:
            print(f"   [EARLY STOP] No val_loss improvement for "
                  f"{stopper.patience} epochs. Stopping.")
            break

        gc.collect()  # Optimization O2

    # ---- 6m: training curves
    _plot_curves(history)

    # ---- 6l: baseline Logistic Regression (metadata only)
    print("\n[BASELINE] Training Logistic Regression on metadata only ...")
    baseline = LogisticRegression(max_iter=1000, class_weight="balanced",
                                  random_state=SEED)
    baseline.fit(x_meta[idx_train], y[idx_train])
    base_acc = accuracy_score(y[idx_test],
                              baseline.predict(x_meta[idx_test]))
    base_f1 = f1_score(y[idx_test],
                       baseline.predict(x_meta[idx_test]),
                       zero_division=0)
    joblib.dump(baseline, BASELINE_PATH)
    print(f"[BASELINE] test_acc={base_acc:.4f} test_f1={base_f1:.4f} "
          f"(saved to {BASELINE_PATH})")

    # Quick final test-set sanity check with the best model.
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    te = _run_epoch(model, test_loader, loss_fn, optimizer=None)
    print(f"\n[TEST] Multi-input model -> acc={te['acc']:.4f} "
          f"f1={te['f1']:.4f} auc={te['roc_auc']:.4f}")
    print("\n[DONE] Training pipeline complete.")


if __name__ == "__main__":
    train()
