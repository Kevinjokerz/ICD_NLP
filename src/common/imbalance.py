from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import logging

def compute_pos_weight(option: Optional[str], Y_tr: np.ndarray, logger: logging.Logger) -> Optional[torch.Tensor]:
    """
    Compute pos_weight for BCEWithLogitsLoss.

    Parameters
    ----------
    option : str or None
        None -> return None; "balanced" -> compute; else -> path to .npy
    Y_tr : np.ndarray, shape (N, K), dtype {uint8,bool}
        Multi-hot labels on train split.

    Returns
    -------
    torch.Tensor or None
        Float32 tensor [K] on CPU; caller moves to device when needed.

    Notes
    -----
    pos_weight[k] = N_neg_k / max(N_pos_k, 1), where N_neg_k = N - N_pos_k.
    """

    # --- Normalize option ---
    if option is None:
        return None
    
    opt = str(option).lower().strip()
    if opt in {"none", "null", "false", ""}:
        return None
    
    # --- Balanced from Y_tr ---
    if opt == "balanced":
        assert Y_tr.ndim == 2, f"[POS] Y_tr.ndim={Y_tr.ndim}, expected 2."
        N, K = Y_tr.shape
        pos = Y_tr.sum(axis=0).astype("float64")
        Nf = float(N)
        neg = Nf - pos
        pos[pos <= 0.0] = 1.0
        w = neg / pos
        w = np.clip(w, 1.0, 12.0).astype("float32")
        logger.info("[POS] pos_weight balanced: min=%.3f, max=%.3f", float(w.min()), float(w.max()))
        return torch.from_numpy(w)
    
    # --- Load from .npy path ---
    path = Path(option)
    if not path.exists():
        raise FileNotFoundError(f"[POS] pos_weight file not found: {path}")
    w = np.load(path)
    assert w.ndim == 1 and w.shape[0] == Y_tr.shape[1], \
        f"[POS] pos_weight shape mismatch: {w.shape} vs K={Y_tr.shape[1]}"
    
    w = w.astype("float32")
    logger.info(f"[POS] pos_weight loaded: min={w.min()}, max={w.max()}.")
    return torch.from_numpy(w)