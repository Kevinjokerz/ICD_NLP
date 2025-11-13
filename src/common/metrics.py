"""
Metrics — Multi-Label ICD

Purpose
-------
Stateless metric helpers (no IO/logging) to avoid circular imports.

Conventions
-----------
- Shapes: (N, K)
- DTypes: y_true {0,1} (bool/uint8), y_scores float
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score

# =============================================================================
# Metrics
# =============================================================================

def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: int) -> float:
    """
    Compute Precision@k for a multi-label setting.

    For each sample (row), take the top-k predicted labels by score,
    compute the fraction of them that are actually true, and average
    this precision over all samples.

    Parameters
    ----------
    y_true  : np.ndarray
        Shape (N, K), {0, 1}, Ground-truth binary indicator matrix.
    y_scores: np.ndarray
        Shape (N, K), float, Predicted confidence scores (higher = more confident).
    k       : int
        Number of top labels to consider per sample.

    Returns
    -------
    float
        Mean precision@k across all samples (0 <= value <= 1)
    """
    assert y_true.shape == y_scores.shape, f"[CHECK] Shape mismatch: {y_true.shape} vs {y_scores.shape}"
    if k < 1:
        raise ValueError("[CHECK] k must be >= 1")
    
    N, K = y_true.shape
    k = min(k, K)
    y_true_c = np.asarray(y_true, dtype=np.uint8)
    y_scores_f = np.asarray(y_scores, dtype=float)
    if not np.isfinite(y_scores_f).all():
        raise ValueError("[CHECK] y_scores contain NaN/Inf")
    
    topk_idx = np.argpartition(-y_scores_f, kth=k-1, axis=1)[:, :k]
    hits = y_true_c[np.arange(N)[:, None], topk_idx].sum(axis=1)

    precision = float(np.mean(hits / float(k)))

    return precision

def compute_f1s(y_true: np.ndarray, y_pred_bin: np.ndarray) -> Tuple[float, float]:
    """
    Compute Micro- and Macro-F1 scores using sklearn.

    Parameters
    ----------
    y_true  : np.ndarray
        Ground-truth binary labels, shape (N, K) values {0, 1}
    y_pred_bin : np.ndarray
        Binary predictions, same shape as y_true
    
    Returns
    -------
    micro :float
        Micro-average F1 score.
    macro : float
        Macro-average F1 score.

    Notes
    -----
    - zero_division=0 prevents NaN if a class has no positive predictions.
    """
    assert y_true.shape == y_pred_bin.shape, f"[CHECK] Shape mismatch: {y_true.shape} vs {y_pred_bin.shape}"
    y_true_c = np.asarray(y_true, dtype=np.uint8)
    y_pred_c = np.asarray(y_pred_bin, dtype=np.uint8)
    vals_true, vals_pred = np.unique(y_true_c), np.unique(y_pred_c)
    if (y_true_c < 0).any() or (y_true_c > 1).any() or (y_pred_c < 0).any() or (y_pred_c > 1).any():
        raise ValueError("[CHECK] inputs must be binary {0, 1}")
    micro = f1_score(y_true_c, y_pred_c, average="micro", zero_division=0)
    macro = f1_score(y_true_c, y_pred_c, average="macro", zero_division=0)

    return float(micro), float(macro)