"""
============================================================
STEP 9 — PREDICTION PIPELINE
============================================================
Module: predict.py

9a. predict_single_article(title, text, subject, date)
      - feature-engineers the article
      - extracts BERT embedding
      - loads saved model + scalers/encoders
      - returns the full explanation report (Section 8c)
      - prints a human-readable summary
9b. batch_predict(csv_file_path)
      - loads a CSV with the dataset columns
      - predicts every row
      - writes results CSV with added columns:
        misinformation_risk_score, risk_tier, prediction
      - prints summary statistics
"""

from __future__ import annotations

import os
import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from bert_embeddings import extract_bert_embeddings
from explainability import generate_explanation_report
from feature_engineering import engineer_single_article
from model import FakeNewsClassifier, risk_tier

# Reproducibility (Optimization O1)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fake_news_model.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_MODEL: FakeNewsClassifier = None


def _model(meta_dim: int) -> FakeNewsClassifier:
    """Load + cache the trained model.

    Args:
        meta_dim: Number of metadata features.

    Returns:
        The model in eval mode.
    """
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(BEST_MODEL_PATH):
            raise FileNotFoundError(
                f"{BEST_MODEL_PATH} not found. Train the model first."
            )
        _MODEL = FakeNewsClassifier(bert_dim=768, meta_dim=meta_dim)
        _MODEL.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        _MODEL.to(DEVICE).eval()
    return _MODEL


def predict_single_article(title: str, text: str, subject: str,
                           date: str) -> Dict:
    """Predict + explain a single article (Step 9a).

    Args:
        title: Article title.
        text: Article body (may be empty -> falls back to title).
        subject: Article subject/category.
        date: Publication date string.

    Returns:
        The full explanation report dict (see Step 8c).
    """
    report = generate_explanation_report({
        "title": title, "text": text,
        "subject": subject, "date": date,
    })

    # Human-readable summary.
    print("\n" + "=" * 60)
    print("FAKESHIELD — SINGLE ARTICLE ANALYSIS")
    print("=" * 60)
    print(f"Title              : {title[:80]}")
    print(f"Prediction         : {report['prediction']}")
    print(f"Risk Score         : {report['misinformation_risk_score'] * 100:.1f}%")
    print(f"Risk Tier          : {report['risk_tier']}")
    print(f"Credibility (heur.): {report['credibility_risk_score']:.4f}")
    print("Top suspicious words:")
    for item in report["top_suspicious_words"][:5]:
        if "word" in item:
            print(f"   - {item['word']}  (w={item['weight']})")
    print("Top metadata signals:")
    for item in report["top_metadata_signals"][:5]:
        if "feature" in item:
            print(f"   - {item['feature']}  (shap={item['shap_value']})")
    print("=" * 60)
    return report


def _fast_score(title: str, text: str, subject: str,
                date: str) -> float:
    """Score one row WITHOUT the (slow) SHAP/LIME layer.

    Used by batch_predict for speed.

    Args:
        title: Article title.
        text: Article body.
        subject: Article subject.
        date: Publication date string.

    Returns:
        The model's misinformation risk score in ``[0, 1]``.
    """
    x_meta = engineer_single_article(title, text, subject, date)
    model = _model(x_meta.shape[1])
    combined = (f"{title} [SEP] {text}"
                if str(text).strip() else str(title))
    emb = extract_bert_embeddings([combined], batch_size=1)
    with torch.no_grad():
        xb = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
        xm = torch.tensor(x_meta, dtype=torch.float32).to(DEVICE)
        return float(model(xb, xm).cpu().numpy().flatten()[0])


def batch_predict(csv_file_path: str,
                  out_path: str = None) -> pd.DataFrame:
    """Predict misinformation risk for every row of a CSV (Step 9b).

    Args:
        csv_file_path: Path to a CSV with title/text/subject/date.
        out_path: Optional output path. Defaults to
            ``<input>_predictions.csv``.

    Returns:
        The results DataFrame (also written to disk).
    """
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    for col in ("title", "text", "subject", "date"):
        if col not in df.columns:
            df[col] = ""

    scores, tiers, preds = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df),
                       desc="Batch predicting"):
        s = _fast_score(str(row["title"]), str(row["text"]),
                        str(row["subject"]), str(row["date"]))
        t, _e = risk_tier(s)
        scores.append(round(s, 4))
        tiers.append(t)
        preds.append("FAKE" if s >= 0.5 else "REAL")

    df["misinformation_risk_score"] = scores
    df["risk_tier"] = tiers
    df["prediction"] = preds

    if out_path is None:
        base, _ext = os.path.splitext(csv_file_path)
        out_path = f"{base}_predictions.csv"
    df.to_csv(out_path, index=False)

    # ---- summary statistics
    print("\n" + "=" * 55)
    print("BATCH PREDICTION SUMMARY")
    print("=" * 55)
    print(f"Total articles     : {len(df):,}")
    print(f"Predicted FAKE     : {(df['prediction'] == 'FAKE').sum():,}")
    print(f"Predicted REAL     : {(df['prediction'] == 'REAL').sum():,}")
    print(f"Mean risk score    : {np.mean(scores):.4f}")
    print("Risk tier breakdown:")
    print(df["risk_tier"].value_counts().to_string())
    print(f"\n[OK] Results saved to: {out_path}")
    print("=" * 55)
    return df


if __name__ == "__main__":
    if os.path.exists(BEST_MODEL_PATH):
        predict_single_article(
            title="BREAKING: Secret bombshell report EXPOSED online",
            text="Unnamed sources claim shocking viral truth was hidden.",
            subject="politicsNews",
            date="2017-05-12",
        )
    else:
        print("Train the model first (run train.py).")
