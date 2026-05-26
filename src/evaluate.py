"""
============================================================
STEP 7 — EVALUATION
============================================================
Module: evaluate.py

7a. Load best model, run inference on the test set.
7b. Full classification report (precision/recall/F1, macro/weighted).
7c. Plots: normalized + raw confusion matrices, ROC curve (AUC),
    Precision-Recall curve, misinformation risk-score histogram by label.
7d. Multi-input vs baseline Logistic Regression comparison table.
7e. Top 10 highest-confidence correct predictions and
    Top 10 most-confused (most wrong) predictions.
7f. All plots saved to plots/.
"""

from __future__ import annotations

import os
import random
from typing import Tuple

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from bert_embeddings import load_or_extract
from feature_engineering import engineer_features
from model import FakeNewsClassifier

# Reproducibility (Optimization O1)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

CLEANED_CSV = os.path.join(DATA_DIR, "cleaned_data.csv")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.pt")
BASELINE_PATH = os.path.join(MODELS_DIR, "baseline_logreg.joblib")
SPLIT_IDX_PATH = os.path.join(MODELS_DIR, "split_indices.joblib")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_test_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                 pd.DataFrame]:
    """Reconstruct the test split arrays (Step 7a).

    Returns:
        ``(x_bert_test, x_meta_test, y_test, df_test)``.
    """
    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    x_meta, y = engineer_features(df, fit=False)
    texts = df["combined_text"].astype(str).tolist()
    x_bert = load_or_extract(texts, split="all", batch_size=16)

    splits = joblib.load(SPLIT_IDX_PATH)
    ti = splits["test"]
    return (x_bert[ti], x_meta[ti], y[ti],
            df.iloc[ti].reset_index(drop=True))


def _predict(model: FakeNewsClassifier, x_bert: np.ndarray,
             x_meta: np.ndarray, batch: int = 256) -> np.ndarray:
    """Batched inference (Optimization O2).

    Args:
        model: The loaded classifier.
        x_bert: Test BERT matrix.
        x_meta: Test metadata matrix.
        batch: Inference batch size.

    Returns:
        A 1-D array of predicted probabilities.
    """
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(x_bert), batch):
            xb = torch.tensor(x_bert[s:s + batch],
                              dtype=torch.float32).to(DEVICE)
            xm = torch.tensor(x_meta[s:s + batch],
                              dtype=torch.float32).to(DEVICE)
            out.append(model(xb, xm).cpu().numpy())
    return np.vstack(out).flatten()


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Save raw + normalized confusion-matrix heatmaps (Step 7c)."""
    cm = confusion_matrix(y_true, y_pred)
    cmn = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0],
                xticklabels=["True(0)", "Fake(1)"],
                yticklabels=["True(0)", "Fake(1)"])
    ax[0].set_title("Confusion Matrix (Raw)")
    ax[0].set_xlabel("Predicted")
    ax[0].set_ylabel("Actual")
    sns.heatmap(cmn, annot=True, fmt=".3f", cmap="Greens", ax=ax[1],
                xticklabels=["True(0)", "Fake(1)"],
                yticklabels=["True(0)", "Fake(1)"])
    ax[1].set_title("Confusion Matrix (Normalized)")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)


def _plot_roc_pr(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Save ROC and Precision-Recall curves (Step 7c).

    Returns:
        The ROC-AUC score.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax[0].set_title("ROC Curve")
    ax[0].set_xlabel("False Positive Rate")
    ax[0].set_ylabel("True Positive Rate")
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(rec, prec, color="purple")
    ax[1].set_title("Precision-Recall Curve")
    ax[1].set_xlabel("Recall")
    ax[1].set_ylabel("Precision")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_pr_curves.png"), dpi=130)
    plt.close(fig)
    return auc


def _plot_risk_hist(y_true: np.ndarray, y_prob: np.ndarray) -> None:
    """Save misinformation risk-score histogram by label (Step 7c)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6,
            label="True (0)", color="seagreen")
    ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6,
            label="Fake (1)", color="crimson")
    ax.axvline(0.5, color="black", linestyle="--", alpha=0.7,
               label="Decision threshold")
    ax.set_title("Misinformation Risk Score Distribution by Label")
    ax.set_xlabel("Predicted Risk Score")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "risk_score_distribution.png"),
                dpi=130)
    plt.close(fig)


def evaluate() -> None:
    """Run the full Step-7 evaluation pipeline."""
    if not os.path.exists(BEST_MODEL_PATH):
        raise FileNotFoundError(
            f"{BEST_MODEL_PATH} not found. Run train.py first."
        )

    x_bert, x_meta, y_true, df_test = _load_test_arrays()

    model = FakeNewsClassifier(bert_dim=x_bert.shape[1],
                               meta_dim=x_meta.shape[1]).to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))

    y_prob = _predict(model, x_bert, x_meta)
    y_pred = (y_prob >= 0.5).astype(int)

    # ---- 7b: classification report
    print("=" * 65)
    print("CLASSIFICATION REPORT — Multi-Input Model")
    print("=" * 65)
    print(classification_report(
        y_true, y_pred, target_names=["True (0)", "Fake (1)"],
        digits=4))

    # ---- 7c: plots
    _plot_confusion(y_true, y_pred)
    auc = _plot_roc_pr(y_true, y_prob)
    _plot_risk_hist(y_true, y_prob)
    print(f"[OK] Plots saved to {PLOTS_DIR}/  (ROC-AUC = {auc:.4f})")

    # ---- 7d: comparison vs baseline
    baseline = joblib.load(BASELINE_PATH)
    b_pred = baseline.predict(x_meta)
    b_prob = baseline.predict_proba(x_meta)[:, 1]
    comp = pd.DataFrame({
        "Model": ["Multi-Input (BERT+Meta)", "Baseline LogReg (Meta only)"],
        "Accuracy": [accuracy_score(y_true, y_pred),
                     accuracy_score(y_true, b_pred)],
        "F1 (Fake)": [f1_score(y_true, y_pred, zero_division=0),
                      f1_score(y_true, b_pred, zero_division=0)],
        "ROC-AUC": [auc, roc_auc_score(y_true, b_prob)],
    })
    print("\n" + "=" * 65)
    print("MODEL COMPARISON")
    print("=" * 65)
    print(comp.to_string(index=False))
    comp.to_csv(os.path.join(PLOTS_DIR, "model_comparison.csv"),
                index=False)

    # ---- 7e: top-10 confident-correct & most-confused
    correct = y_pred == y_true
    confidence = np.abs(y_prob - 0.5)
    df_res = df_test.copy()
    df_res["y_true"] = y_true
    df_res["y_prob"] = y_prob
    df_res["correct"] = correct
    df_res["confidence"] = confidence

    top_correct = (df_res[df_res["correct"]]
                   .sort_values("confidence", ascending=False)
                   .head(10))
    top_wrong = (df_res[~df_res["correct"]]
                 .sort_values("confidence", ascending=False)
                 .head(10))

    print("\nTOP 10 HIGHEST-CONFIDENCE CORRECT PREDICTIONS:")
    for _, r in top_correct.iterrows():
        print(f"  [{r['y_prob']:.3f}] true={int(r['y_true'])} | "
              f"{str(r['title'])[:70]}")

    print("\nTOP 10 MOST-CONFUSED (MOST WRONG) PREDICTIONS:")
    for _, r in top_wrong.iterrows():
        print(f"  [{r['y_prob']:.3f}] true={int(r['y_true'])} | "
              f"{str(r['title'])[:70]}")

    print("\n[DONE] Evaluation complete.")


if __name__ == "__main__":
    evaluate()
