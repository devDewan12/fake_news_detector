"""
============================================================
STEP 4 — BERT EMBEDDINGS EXTRACTION
============================================================
Module: bert_embeddings.py

4a. Load pre-trained "bert-base-uncased" tokenizer + model.
4b. extract_bert_embeddings(texts, batch_size=16):
       - tokenize (max_length=256, padding="max_length", truncation=True)
       - forward pass under torch.no_grad()
       - take the [CLS] token (index 0) of last_hidden_state  -> 768-d
       - batched processing, returns np.ndarray (n_samples, 768)
4c. tqdm progress bar.
4d. CUDA if available, else CPU.
4e. Cache embeddings to data/bert_embeddings_{split}.npy.
4f. load_or_extract() — re-uses cached .npy when present.

Memory hygiene (Optimization O2): torch.no_grad(), explicit tensor
deletion + gc.collect() after every batch.
"""

from __future__ import annotations

import gc
import os
import random
from typing import List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# ------------------------------------------------------------------ #
# Reproducibility (Optimization O1)
# ------------------------------------------------------------------ #
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# ------------------------------------------------------------------ #
# Configuration
# ------------------------------------------------------------------ #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 256
EMBED_DIM = 768
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache the loaded tokenizer/model at module level so repeated calls
# (e.g. from LIME's predict_proba) do not reload BERT each time.
_TOKENIZER: Optional[AutoTokenizer] = None
_MODEL: Optional[AutoModel] = None


def _get_bert():
    """Lazily load + cache the BERT tokenizer and model (Step 4a).

    Returns:
        A tuple ``(tokenizer, model)`` with the model on ``DEVICE``
        in eval mode.
    """
    global _TOKENIZER, _MODEL
    if _TOKENIZER is None or _MODEL is None:
        try:
            _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
            _MODEL = AutoModel.from_pretrained(MODEL_NAME)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load '{MODEL_NAME}'. Ensure you have internet "
                f"access on first run (HuggingFace will cache the model "
                f"locally afterwards). Original error: {exc}"
            ) from exc
        _MODEL.to(DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL


def extract_bert_embeddings(texts: Sequence[str],
                            batch_size: int = 16) -> np.ndarray:
    """Extract [CLS] BERT embeddings for a list of texts (Step 4b).

    Args:
        texts: Sequence of raw strings.
        batch_size: Mini-batch size for the forward pass.

    Returns:
        A numpy array of shape ``(len(texts), 768)``.
    """
    tokenizer, model = _get_bert()
    texts = [t if isinstance(t, str) and t.strip() else "[PAD]" for t in texts]

    all_emb: List[np.ndarray] = []
    n = len(texts)
    for start in tqdm(range(0, n, batch_size),
                      desc="Extracting BERT embeddings",
                      unit="batch"):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(
            batch,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(DEVICE) for k, v in encoded.items()}

        with torch.no_grad():  # Optimization O2
            outputs = model(**encoded)
            # [CLS] token = index 0 of the last hidden state.
            cls = outputs.last_hidden_state[:, 0, :]
            all_emb.append(cls.cpu().numpy())

        # Aggressive memory hygiene (Optimization O2).
        del encoded, outputs, cls
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    return np.vstack(all_emb).astype(np.float32)


def _cache_path(split: str) -> str:
    """Return the .npy cache path for a given split name (Step 4e)."""
    return os.path.join(DATA_DIR, f"bert_embeddings_{split}.npy")


def load_or_extract(texts: Sequence[str],
                    split: str,
                    batch_size: int = 16,
                    force: bool = False) -> np.ndarray:
    """Load cached embeddings if present, else extract + cache (Step 4f).

    Args:
        texts: Sequence of raw strings.
        split: A short tag, e.g. ``"train"`` or ``"test"``.
        batch_size: Mini-batch size for extraction.
        force: If True, ignore the cache and re-extract.

    Returns:
        A numpy array of shape ``(len(texts), 768)``.
    """
    path = _cache_path(split)
    os.makedirs(DATA_DIR, exist_ok=True)

    if (not force) and os.path.exists(path):
        try:
            cached = np.load(path)
            if cached.shape[0] == len(texts):
                print(f"[CACHE] Loaded BERT embeddings from {path} "
                      f"-> shape {cached.shape}")
                return cached.astype(np.float32)
            print(f"[CACHE] Size mismatch ({cached.shape[0]} vs "
                  f"{len(texts)}); re-extracting.")
        except Exception as exc:  # noqa: BLE001
            print(f"[CACHE] Failed to load {path} ({exc}); re-extracting.")

    emb = extract_bert_embeddings(texts, batch_size=batch_size)
    np.save(path, emb)
    print(f"[CACHE] Saved BERT embeddings to {path} -> shape {emb.shape}")
    return emb


if __name__ == "__main__":
    demo = [
        "Breaking: scientists discover shocking truth about coffee.",
        "The central bank raised interest rates by 25 basis points today.",
    ]
    vecs = extract_bert_embeddings(demo, batch_size=2)
    print("Demo embedding shape:", vecs.shape)
    print("Device used         :", DEVICE)
