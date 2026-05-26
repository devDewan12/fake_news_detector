"""
============================================================
STEP 3 — FEATURE ENGINEERING
============================================================
Module: feature_engineering.py

Extracts the METADATA BRANCH features used by the multi-input model:

3a. Text statistical features (from `text`).
3b. Title statistical features (from `title`).
3c. Date / temporal features (from parsed `date`).
3d. Subject features (LabelEncoder, saved for inference).
3e. Custom `credibility_risk_score` heuristic (MinMax normalised).
3f. Combine all metadata features into a single matrix.
3g. Scale features with StandardScaler (saved with joblib).
3h. Return final metadata matrix X_meta and target y.

All fitted transformers (LabelEncoder, MinMaxScaler, StandardScaler)
are persisted to ``models/`` so they can be reused at inference time
(Optimization O3).
"""

from __future__ import annotations

import os
import re
import string
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ------------------------------------------------------------------ #
# Paths
# ------------------------------------------------------------------ #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SUBJECT_ENCODER_PATH = os.path.join(MODELS_DIR, "subject_encoder.joblib")
MINMAX_SCALER_PATH = os.path.join(MODELS_DIR, "credibility_minmax.joblib")
STD_SCALER_PATH = os.path.join(MODELS_DIR, "metadata_scaler.joblib")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "metadata_feature_names.joblib")

# ------------------------------------------------------------------ #
# Constants
# ------------------------------------------------------------------ #
CLICKBAIT_WORDS: List[str] = [
    "shocking", "you won't believe", "must see", "breaking",
    "exposed", "secret", "viral", "truth", "bombshell",
]

# A compact, dependency-free English stop-word list (avoids needing NLTK
# downloads, which keeps the pipeline offline-friendly — Optimization O4).
STOPWORDS: set = {
    "a", "an", "the", "and", "or", "but", "if", "while", "is", "are", "was",
    "were", "be", "been", "being", "to", "of", "in", "on", "for", "with",
    "as", "by", "at", "from", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "you", "i",
    "me", "my", "your", "our", "us", "not", "no", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "than",
    "then", "so", "such", "into", "about", "over", "after", "before", "up",
    "down", "out", "off", "again", "more", "most", "some", "any", "all",
    "who", "what", "when", "where", "why", "how", "which", "there", "here",
}

# The fixed ordered list of engineered metadata feature names.
METADATA_FEATURE_NAMES: List[str] = [
    # 3a — text statistical features
    "word_count", "char_count", "avg_word_length", "sentence_count",
    "exclamation_count", "question_count", "uppercase_ratio",
    "unique_word_ratio", "stopword_ratio", "digit_count",
    # 3b — title statistical features
    "title_word_count", "title_char_count", "title_uppercase_ratio",
    "title_exclamation_count", "title_has_question", "clickbait_score",
    # 3c — date / temporal features
    "pub_year", "pub_month", "pub_day", "pub_weekday", "is_weekend",
    # 3d — subject feature
    "subject_encoded",
    # 3e — custom heuristic
    "credibility_risk_score",
]


# ================================================================== #
# 3a. TEXT STATISTICAL FEATURES
# ================================================================== #
def _uppercase_ratio(text: str) -> float:
    """Return the ratio of uppercase letters to total letters."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    uppers = sum(1 for c in letters if c.isupper())
    return uppers / len(letters)


def extract_text_features(text: str) -> dict:
    """Extract 10 statistical features from an article body (Step 3a).

    Args:
        text: The raw article body text.

    Returns:
        A dict of the 10 text statistical features.
    """
    text = text if isinstance(text, str) else ""
    words = text.split()
    n_words = len(words)
    n_chars = len(text)

    avg_word_len = (
        np.mean([len(w) for w in words]) if n_words else 0.0
    )
    sentence_count = max(1, len(text.split(". "))) if text.strip() else 0
    unique_ratio = (len(set(w.lower() for w in words)) / n_words) if n_words else 0.0
    stop_ratio = (
        sum(1 for w in words if w.lower() in STOPWORDS) / n_words
        if n_words else 0.0
    )
    digit_count = sum(c.isdigit() for c in text)

    return {
        "word_count": float(n_words),
        "char_count": float(n_chars),
        "avg_word_length": float(avg_word_len),
        "sentence_count": float(sentence_count),
        "exclamation_count": float(text.count("!")),
        "question_count": float(text.count("?")),
        "uppercase_ratio": float(_uppercase_ratio(text)),
        "unique_word_ratio": float(unique_ratio),
        "stopword_ratio": float(stop_ratio),
        "digit_count": float(digit_count),
    }


# ================================================================== #
# 3b. TITLE STATISTICAL FEATURES
# ================================================================== #
def _clickbait_score(title: str) -> int:
    """Count how many clickbait trigger words/phrases appear in `title`."""
    low = title.lower()
    return sum(1 for w in CLICKBAIT_WORDS if w in low)


def extract_title_features(title: str) -> dict:
    """Extract 6 statistical features from the article title (Step 3b).

    Args:
        title: The raw article title.

    Returns:
        A dict of the 6 title statistical features.
    """
    title = title if isinstance(title, str) else ""
    words = title.split()
    return {
        "title_word_count": float(len(words)),
        "title_char_count": float(len(title)),
        "title_uppercase_ratio": float(_uppercase_ratio(title)),
        "title_exclamation_count": float(title.count("!")),
        "title_has_question": 1.0 if "?" in title else 0.0,
        "clickbait_score": float(_clickbait_score(title)),
    }


# ================================================================== #
# 3c. DATE / TEMPORAL FEATURES
# ================================================================== #
def extract_date_features(date_value: pd.Timestamp) -> dict:
    """Extract 5 temporal features from a parsed date (Step 3c).

    NaT dates fall back to safe zero/neutral values (Optimization O4).

    Args:
        date_value: A pandas Timestamp (or NaT).

    Returns:
        A dict of the 5 temporal features.
    """
    if pd.isna(date_value):
        return {
            "pub_year": 0.0, "pub_month": 0.0, "pub_day": 0.0,
            "pub_weekday": 0.0, "is_weekend": 0.0,
        }
    weekday = int(date_value.weekday())
    return {
        "pub_year": float(date_value.year),
        "pub_month": float(date_value.month),
        "pub_day": float(date_value.day),
        "pub_weekday": float(weekday),
        "is_weekend": 1.0 if weekday >= 5 else 0.0,
    }


# ================================================================== #
# 3d. SUBJECT FEATURE  (LabelEncoder, saved for inference)
# ================================================================== #
def fit_subject_encoder(subjects: pd.Series) -> LabelEncoder:
    """Fit and persist a LabelEncoder on the subject column (Step 3d).

    Args:
        subjects: Series of subject strings.

    Returns:
        The fitted LabelEncoder.
    """
    le = LabelEncoder()
    le.fit(subjects.astype(str).fillna("unknown"))
    joblib.dump(le, SUBJECT_ENCODER_PATH)
    return le


def transform_subject(subjects: pd.Series, le: LabelEncoder) -> np.ndarray:
    """Encode subjects, mapping unseen labels gracefully (Optimization O4).

    Args:
        subjects: Series of subject strings.
        le: A fitted LabelEncoder.

    Returns:
        A 1-D numpy array of encoded integers.
    """
    known = set(le.classes_)
    safe = subjects.astype(str).apply(
        lambda s: s if s in known else le.classes_[0]
    )
    return le.transform(safe).astype(float)


# ================================================================== #
# 3e. CREDIBILITY RISK SCORE  (heuristic, MinMax normalised)
# ================================================================== #
def compute_credibility_risk(df_feats: pd.DataFrame,
                             fit: bool = True) -> np.ndarray:
    """Compute the heuristic credibility risk score (Step 3e).

        risk = (uppercase_ratio*2) + (exclamation_count*0.5)
             + (clickbait_score*1.5) + (title_uppercase_ratio*2)
             + (1 - unique_word_ratio)*1.0

    The raw score is MinMax-scaled to [0, 1]; the scaler is saved when
    ``fit=True`` and re-loaded otherwise (Optimization O3 / O4).

    Args:
        df_feats: DataFrame already containing the component columns.
        fit: If True, fit + save the MinMaxScaler; else load it.

    Returns:
        A 1-D numpy array of normalised risk scores in [0, 1].
    """
    raw = (
        df_feats["uppercase_ratio"] * 2.0
        + df_feats["exclamation_count"] * 0.5
        + df_feats["clickbait_score"] * 1.5
        + df_feats["title_uppercase_ratio"] * 2.0
        + (1.0 - df_feats["unique_word_ratio"]) * 1.0
    ).values.reshape(-1, 1)

    if fit:
        scaler = MinMaxScaler()
        normed = scaler.fit_transform(raw)
        joblib.dump(scaler, MINMAX_SCALER_PATH)
    else:
        try:
            scaler = joblib.load(MINMAX_SCALER_PATH)
            normed = scaler.transform(raw)
        except Exception:  # noqa: BLE001 - fall back to a fresh scaler
            scaler = MinMaxScaler()
            normed = scaler.fit_transform(raw)
    return normed.flatten()


# ================================================================== #
# Row-level feature assembly
# ================================================================== #
def build_feature_frame(df: pd.DataFrame,
                         subject_encoder: LabelEncoder,
                         fit_credibility: bool = True) -> pd.DataFrame:
    """Build the full engineered-feature DataFrame for `df`.

    Args:
        df: Cleaned DataFrame with title/text/subject/date columns.
        subject_encoder: A fitted LabelEncoder for the subject column.
        fit_credibility: Whether to fit (True) or load (False) the
            MinMax scaler used by the credibility score.

    Returns:
        A DataFrame whose columns are exactly ``METADATA_FEATURE_NAMES``.
    """
    text_feats = df["text"].apply(extract_text_features).apply(pd.Series)
    title_feats = df["title"].apply(extract_title_features).apply(pd.Series)
    date_feats = df["date"].apply(extract_date_features).apply(pd.Series)

    feats = pd.concat([text_feats, title_feats, date_feats], axis=1)
    feats["subject_encoded"] = transform_subject(
        df.get("subject", pd.Series(["unknown"] * len(df))), subject_encoder
    )
    feats["credibility_risk_score"] = compute_credibility_risk(
        feats, fit=fit_credibility
    )
    # Guarantee column order matches METADATA_FEATURE_NAMES.
    return feats[METADATA_FEATURE_NAMES].astype(float)


# ================================================================== #
# 3f / 3g / 3h — combine, scale, return
# ================================================================== #
def engineer_features(df: pd.DataFrame,
                       fit: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """End-to-end metadata feature engineering (Steps 3f–3h).

    Args:
        df: Cleaned DataFrame (must contain a ``label`` column when y
            is required).
        fit: If True, fit + persist all transformers; else load them.

    Returns:
        A tuple ``(X_meta, y)`` where ``X_meta`` is the StandardScaler-
        scaled metadata matrix and ``y`` is the label array (or an
        empty array if no label column is present).
    """
    if fit:
        subject_encoder = fit_subject_encoder(
            df.get("subject", pd.Series(["unknown"] * len(df)))
        )
    else:
        subject_encoder = joblib.load(SUBJECT_ENCODER_PATH)

    feats = build_feature_frame(df, subject_encoder, fit_credibility=fit)

    if fit:
        scaler = StandardScaler()
        x_meta = scaler.fit_transform(feats.values)
        joblib.dump(scaler, STD_SCALER_PATH)
        joblib.dump(METADATA_FEATURE_NAMES, FEATURE_NAMES_PATH)
    else:
        scaler = joblib.load(STD_SCALER_PATH)
        x_meta = scaler.transform(feats.values)

    y = (
        df["label"].values.astype(np.float32)
        if "label" in df.columns else np.array([], dtype=np.float32)
    )
    return x_meta.astype(np.float32), y


def engineer_single_article(title: str, text: str, subject: str,
                             date: str) -> np.ndarray:
    """Engineer the metadata vector for ONE article (inference path).

    Loads previously-saved transformers; handles empty text by falling
    back to the title (Optimization O4).

    Args:
        title: Article title.
        text: Article body (may be empty).
        subject: Article subject/category.
        date: Publication date string.

    Returns:
        A scaled metadata vector of shape ``(1, n_features)``.
    """
    if not (text and str(text).strip()):
        text = title  # fallback to title-only

    row = pd.DataFrame([{
        "title": title or "",
        "text": text or "",
        "subject": subject or "unknown",
        "date": pd.to_datetime(date, errors="coerce"),
    }])
    subject_encoder = joblib.load(SUBJECT_ENCODER_PATH)
    feats = build_feature_frame(row, subject_encoder, fit_credibility=False)
    scaler = joblib.load(STD_SCALER_PATH)
    return scaler.transform(feats.values).astype(np.float32)


if __name__ == "__main__":
    # Smoke test against the cleaned data, if present.
    cleaned = os.path.join(PROJECT_ROOT, "data", "cleaned_data.csv")
    if os.path.exists(cleaned):
        _df = pd.read_csv(cleaned, parse_dates=["date"])
        _x, _y = engineer_features(_df, fit=True)
        print(f"X_meta shape: {_x.shape}")
        print(f"y shape     : {_y.shape}")
        print(f"Feature names ({len(METADATA_FEATURE_NAMES)}):")
        print(METADATA_FEATURE_NAMES)
    else:
        print("Run data_preprocessing.py first to create cleaned_data.csv")
