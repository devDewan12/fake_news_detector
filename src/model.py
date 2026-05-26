"""
============================================================
STEP 5 — MULTI-INPUT MODEL ARCHITECTURE
============================================================
Module: model.py

A multi-input neural network combining:

BRANCH A (BERT text, 768-d):
    Linear(768,256) -> BN -> ReLU -> Dropout(0.3)
    Linear(256,128) -> ReLU -> Dropout(0.2)

BRANCH B (metadata, n_meta-d):
    Linear(n_meta,64) -> BN -> ReLU -> Dropout(0.2)
    Linear(64,32) -> ReLU

FUSION:
    concat(128 + 32 = 160)
    Linear(160,64) -> ReLU -> Dropout(0.2)
    Linear(64,1) -> Sigmoid   (misinformation risk score in [0,1])

Also provides:
 - build_loss (BCELoss, supports pos_weight for class imbalance — O6)
 - build_optimizer (AdamW, weight_decay=1e-4)
 - build_scheduler (ReduceLROnPlateau on val_loss)
 - EarlyStopping (patience=3)
 - risk_tier() helper mapping a score to a human-readable tier.
"""

from __future__ import annotations

import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Reproducibility (Optimization O1)
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


class FakeNewsClassifier(nn.Module):
    """Multi-input (BERT + metadata) binary misinformation classifier."""

    def __init__(self, bert_dim: int = 768, meta_dim: int = 23) -> None:
        """Initialise the network.

        Args:
            bert_dim: Dimensionality of the BERT [CLS] vector (768).
            meta_dim: Number of engineered metadata features.
        """
        super().__init__()

        # ---------------- Branch A: BERT text ----------------
        self.bert_branch = nn.Sequential(
            nn.Linear(bert_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # ---------------- Branch B: Metadata -----------------
        self.meta_branch = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # ---------------- Fusion head ------------------------
        self.fusion = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_bert: torch.Tensor,
                x_meta: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x_bert: Tensor of shape ``(batch, bert_dim)``.
            x_meta: Tensor of shape ``(batch, meta_dim)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with values in ``[0, 1]``.
        """
        a = self.bert_branch(x_bert)
        b = self.meta_branch(x_meta)
        fused = torch.cat([a, b], dim=1)
        return self.fusion(fused)


# ================================================================== #
# 5b / 5c / 5d — loss, optimizer, scheduler factories
# ================================================================== #
def build_loss(pos_weight: Optional[float] = None) -> nn.Module:
    """Create the binary loss function (Step 5b, Optimization O6).

    BCELoss does not accept ``pos_weight`` directly, so when class
    weighting is requested we use a numerically-stable weighted BCE
    that operates on probabilities (the model already applies Sigmoid).

    Args:
        pos_weight: Optional weight for the positive (fake) class.

    Returns:
        A callable loss module ``loss(pred, target)``.
    """
    if pos_weight is None:
        return nn.BCELoss()

    pw = float(pos_weight)

    class WeightedBCE(nn.Module):
        """Weighted BCE over probabilities (class-imbalance aware)."""

        def __init__(self, weight: float) -> None:
            super().__init__()
            self.weight = weight

        def forward(self, pred: torch.Tensor,
                    target: torch.Tensor) -> torch.Tensor:
            eps = 1e-7
            pred = torch.clamp(pred, eps, 1.0 - eps)
            loss = -(
                self.weight * target * torch.log(pred)
                + (1.0 - target) * torch.log(1.0 - pred)
            )
            return loss.mean()

    return WeightedBCE(pw)


def build_optimizer(model: nn.Module,
                    lr: float = 1e-3) -> torch.optim.Optimizer:
    """Create the AdamW optimizer with weight decay (Step 5c).

    Args:
        model: The model whose parameters to optimise.
        lr: Initial learning rate.

    Returns:
        A configured ``torch.optim.AdamW`` instance.
    """
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Create a ReduceLROnPlateau scheduler on val_loss (Step 5d).

    Args:
        optimizer: The optimizer to schedule.

    Returns:
        A configured ``ReduceLROnPlateau`` scheduler.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )


# ================================================================== #
# 5e — Early stopping
# ================================================================== #
class EarlyStopping:
    """Early-stop training when val_loss stops improving (patience=3)."""

    def __init__(self, patience: int = 3, min_delta: float = 1e-4) -> None:
        """Initialise the early-stopper.

        Args:
            patience: Epochs to wait after the last improvement.
            min_delta: Minimum decrease in val_loss to count as
                an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        """Update internal state with the latest val_loss.

        Args:
            val_loss: The current epoch's validation loss.

        Returns:
            True if this epoch produced a new best model.
        """
        improved = val_loss < (self.best_loss - self.min_delta)
        if improved:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return improved


# ================================================================== #
# 5f — Risk tier helper
# ================================================================== #
def risk_tier(score: float) -> Tuple[str, str]:
    """Map a model score to a (tier_name, emoji) pair (Step 5f).

    Tiers:
        0.00 - 0.30  -> LOW       (Likely Real)
        0.30 - 0.60  -> MEDIUM    (Uncertain)
        0.60 - 0.85  -> HIGH      (Likely Fake)
        0.85 - 1.00  -> CRITICAL  (Very Likely Fake)

    Args:
        score: Model output probability in ``[0, 1]``.

    Returns:
        A tuple ``(tier_name, emoji)``.
    """
    if score < 0.30:
        return "LOW", "✅"
    if score < 0.60:
        return "MEDIUM", "⚠️"
    if score < 0.85:
        return "HIGH", "🔶"
    return "CRITICAL", "🚨"


if __name__ == "__main__":
    net = FakeNewsClassifier(bert_dim=768, meta_dim=23)
    xb = torch.randn(4, 768)
    xm = torch.randn(4, 23)
    out = net(xb, xm)
    print("Output shape:", out.shape)         # (4, 1)
    print("Sample scores:", out.detach().flatten().tolist())
    for s in (0.1, 0.45, 0.7, 0.95):
        print(s, "->", risk_tier(s))
