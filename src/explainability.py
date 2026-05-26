"""
============================================================
STEP 8 — EXPLAINABILITY LAYER
============================================================
Module: explainability.py

8a. SHAP on the metadata branch:
      - shap.KernelExplainer over a metadata-only surrogate.
      - SHAP values for 100 test instances.
      - shap_summary_plot.png + shap_beeswarm_plot.png.
      - explain_metadata_with_shap(article_metadata).
8b. LIME on the text branch:
      - lime.lime_text.LimeTextExplainer.
      - predict_proba_for_lime(texts): BERT on-the-fly + median metadata.
      - explain_text_with_lime(article_text, num_features=15).
      - plots/lime_example_explanation.html.
8c. generate_explanation_report(article) -> structured dict.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from bert_embeddings import extract_bert_embeddings, load_or_extract
from feature_engineering import (
    METADATA_FEATURE_NAMES,
    engineer_features,
    engineer_single_article,
)
from model import FakeNewsClassifier, risk_tier

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
SPLIT_IDX_PATH = os.path.join(MODELS_DIR, "split_indices.joblib")
MEDIAN_META_PATH = os.path.join(MODELS_DIR, "median_metadata.joblib")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Module-level model cache (loaded once, reused by LIME/SHAP).
_MODEL: FakeNewsClassifier = None
_MEDIAN_META: np.ndarray = None


def _load_model(meta_dim: int = len(METADATA_FEATURE_NAMES)
                ) -> FakeNewsClassifier:
    """Load + cache the trained classifier.

    Args:
        meta_dim: Number of metadata features.

    Returns:
        The model in eval mode on ``DEVICE``.
    """
    global _MODEL
    if _MODEL is None:
        _MODEL = FakeNewsClassifier(bert_dim=768, meta_dim=meta_dim)
        _MODEL.load_state_dict(
            torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        _MODEL.to(DEVICE).eval()
    return _MODEL


def _get_median_metadata() -> np.ndarray:
    """Return (and cache) a fixed median metadata vector.

    Used by LIME so that perturbed text is combined with a stable
    metadata context.

    Returns:
        A 1-D metadata vector of length ``len(METADATA_FEATURE_NAMES)``.
    """
    global _MEDIAN_META
    if _MEDIAN_META is not None:
        return _MEDIAN_META
    if os.path.exists(MEDIAN_META_PATH):
        _MEDIAN_META = joblib.load(MEDIAN_META_PATH)
        return _MEDIAN_META
    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    x_meta, _ = engineer_features(df, fit=False)
    _MEDIAN_META = np.median(x_meta, axis=0).astype(np.float32)
    joblib.dump(_MEDIAN_META, MEDIAN_META_PATH)
    return _MEDIAN_META


# ================================================================== #
# 8a — SHAP (metadata branch)
# ================================================================== #
def _metadata_branch_predict(x_meta: np.ndarray) -> np.ndarray:
    """Score articles using a FIXED median BERT vector + given metadata.

    This isolates the metadata branch's contribution for SHAP.

    Args:
        x_meta: Metadata matrix ``(n, n_meta)``.

    Returns:
        A 1-D array of predicted probabilities.
    """
    model = _load_model()
    # Use a zero BERT vector so SHAP attributes variation to metadata.
    bert_fixed = np.zeros((len(x_meta), 768), dtype=np.float32)
    with torch.no_grad():
        xb = torch.tensor(bert_fixed).to(DEVICE)
        xm = torch.tensor(x_meta.astype(np.float32)).to(DEVICE)
        return model(xb, xm).cpu().numpy().flatten()


def run_shap_analysis(n_background: int = 50,
                      n_explain: int = 100) -> None:
    """Compute SHAP values + save summary & beeswarm plots (Step 8a).

    Args:
        n_background: Size of the KernelExplainer background sample.
        n_explain: Number of test instances to explain.
    """
    import shap  # imported lazily to keep module import light

    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    x_meta, _ = engineer_features(df, fit=False)
    splits = joblib.load(SPLIT_IDX_PATH)
    test_meta = x_meta[splits["test"]]

    rng = np.random.RandomState(42)
    bg = test_meta[rng.choice(len(test_meta),
                              min(n_background, len(test_meta)),
                              replace=False)]
    sample = test_meta[rng.choice(len(test_meta),
                                  min(n_explain, len(test_meta)),
                                  replace=False)]

    explainer = shap.KernelExplainer(_metadata_branch_predict, bg)
    shap_values = explainer.shap_values(sample, nsamples=100)

    # Summary (global importance bar)
    plt.figure()
    shap.summary_plot(shap_values, sample,
                      feature_names=METADATA_FEATURE_NAMES,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_plot.png"),
                dpi=130, bbox_inches="tight")
    plt.close()

    # Beeswarm
    plt.figure()
    shap.summary_plot(shap_values, sample,
                      feature_names=METADATA_FEATURE_NAMES,
                      show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "shap_beeswarm_plot.png"),
                dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[OK] SHAP plots saved to {PLOTS_DIR}/")


def explain_metadata_with_shap(article_metadata: np.ndarray,
                               n_background: int = 50) -> Dict[str, float]:
    """Return SHAP values for a single article's metadata (Step 8a).

    Args:
        article_metadata: A scaled metadata vector ``(1, n_meta)`` or
            ``(n_meta,)``.
        n_background: Background sample size for KernelExplainer.

    Returns:
        Dict mapping feature name -> SHAP value.
    """
    import shap

    vec = np.asarray(article_metadata, dtype=np.float32).reshape(1, -1)
    df = pd.read_csv(CLEANED_CSV, parse_dates=["date"])
    x_meta, _ = engineer_features(df, fit=False)
    rng = np.random.RandomState(42)
    bg = x_meta[rng.choice(len(x_meta),
                           min(n_background, len(x_meta)),
                           replace=False)]
    explainer = shap.KernelExplainer(_metadata_branch_predict, bg)
    sv = explainer.shap_values(vec, nsamples=100)
    sv = np.asarray(sv).flatten()
    return dict(zip(METADATA_FEATURE_NAMES, sv.tolist()))


# ================================================================== #
# 8b — LIME (text branch)
# ================================================================== #
def predict_proba_for_lime(texts: List[str]) -> np.ndarray:
    """LIME-compatible probability function (Step 8b).

    Extracts BERT embeddings on the fly, combines them with a fixed
    median metadata vector, and returns ``[P(real), P(fake)]``.

    Args:
        texts: List of raw text strings (LIME perturbations).

    Returns:
        Array of shape ``(len(texts), 2)``.
    """
    model = _load_model()
    median_meta = _get_median_metadata()

    emb = extract_bert_embeddings(list(texts), batch_size=16)
    meta = np.tile(median_meta, (len(texts), 1)).astype(np.float32)

    with torch.no_grad():
        xb = torch.tensor(emb).to(DEVICE)
        xm = torch.tensor(meta).to(DEVICE)
        p_fake = model(xb, xm).cpu().numpy().flatten()

    p_fake = np.clip(p_fake, 1e-6, 1 - 1e-6)
    return np.column_stack([1.0 - p_fake, p_fake])


def explain_text_with_lime(article_text: str,
                           num_features: int = 15) -> List[tuple]:
    """Return the top contributing words for one article (Step 8b).

    Args:
        article_text: The raw article text.
        num_features: Number of words to return.

    Returns:
        A list of ``(word, weight)`` tuples. Positive weight ->
        pushes toward FAKE; negative -> pushes toward REAL.
    """
    from lime.lime_text import LimeTextExplainer

    explainer = LimeTextExplainer(class_names=["REAL", "FAKE"],
                                  random_state=42)
    exp = explainer.explain_instance(
        article_text or "",
        predict_proba_for_lime,
        num_features=num_features,
        num_samples=300,
        labels=(1,),
    )
    # Save an example HTML explanation (Step 8b last bullet).
    html_path = os.path.join(PLOTS_DIR, "lime_example_explanation.html")
    try:
        exp.save_to_file(html_path)
    except Exception:  # noqa: BLE001 - non-fatal
        pass
    return exp.as_list(label=1)


# ================================================================== #
# 8c — Combined explanation report
# ================================================================== #
def generate_explanation_report(article: Dict[str, str]) -> Dict:
    """Produce the structured explanation report (Step 8c).

    Args:
        article: Dict with keys ``title, text, subject, date``.

    Returns:
        Dict with keys:
            misinformation_risk_score, risk_tier, top_suspicious_words,
            top_metadata_signals, credibility_risk_score, prediction.
    """
    title = article.get("title", "")
    text = article.get("text", "")
    subject = article.get("subject", "unknown")
    date = article.get("date", "")

    # ---- metadata vector + raw credibility heuristic
    x_meta = engineer_single_article(title, text, subject, date)
    # credibility_risk_score is the LAST engineered feature (pre-scaling
    # value is recoverable; we report the scaled model-facing one here
    # plus a heuristic read for transparency).
    cred_idx = METADATA_FEATURE_NAMES.index("credibility_risk_score")
    credibility = float(x_meta[0, cred_idx])

    # ---- model score (full multi-input)
    model = _load_model(meta_dim=x_meta.shape[1])
    combined = f"{title} [SEP] {text}" if text.strip() else title
    emb = extract_bert_embeddings([combined], batch_size=1)
    with torch.no_grad():
        xb = torch.tensor(emb, dtype=torch.float32).to(DEVICE)
        xm = torch.tensor(x_meta, dtype=torch.float32).to(DEVICE)
        score = float(model(xb, xm).cpu().numpy().flatten()[0])

    tier, _emoji = risk_tier(score)
    prediction = "FAKE" if score >= 0.5 else "REAL"

    # ---- LIME suspicious words (guarded; explainability is optional)
    try:
        lime_words = explain_text_with_lime(combined, num_features=15)
        top_suspicious = [
            {"word": w, "weight": round(float(wt), 4)}
            for w, wt in lime_words[:10]
        ]
    except Exception as exc:  # noqa: BLE001
        top_suspicious = [{"error": f"LIME unavailable: {exc}"}]

    # ---- SHAP top-5 metadata signals (guarded)
    try:
        shap_map = explain_metadata_with_shap(x_meta)
        top_meta = sorted(shap_map.items(),
                          key=lambda kv: abs(kv[1]),
                          reverse=True)[:5]
        top_metadata = [
            {"feature": k, "shap_value": round(float(v), 4)}
            for k, v in top_meta
        ]
    except Exception as exc:  # noqa: BLE001
        top_metadata = [{"error": f"SHAP unavailable: {exc}"}]

    return {
        "misinformation_risk_score": round(score, 4),
        "risk_tier": tier,
        "top_suspicious_words": top_suspicious,
        "top_metadata_signals": top_metadata,
        "credibility_risk_score": round(credibility, 4),
        "prediction": prediction,
    }


if __name__ == "__main__":
    if os.path.exists(BEST_MODEL_PATH):
        demo = {
            "title": "SHOCKING: You won't believe this SECRET they EXPOSED",
            "text": "Sources say a bombshell report reveals the hidden truth.",
            "subject": "politicsNews",
            "date": "2017-06-01",
        }
        from pprint import pprint
        pprint(generate_explanation_report(demo))
    else:
        print("Train the model first (run train.py).")
