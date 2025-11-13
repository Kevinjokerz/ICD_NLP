"""
Threshold Tuning Utilities - Multi-Label ICD

Purpose
-------
Tune global or per-label decision thresholds on a validation split and persist
them for reproducible evaluation across runs and machines.

Conventions
-----------
- Shapes: y_true, y_scores -> (N, K)
- Dtypes: y_true in {0, 1} (bool/uint8); y_scores float32 in [0, 1]
- Paths : artifacts under models/<run_name>/ relative to repo root.
- PHI   : never log raw TEXTs or identifiers; only counts/metrics.

Inputs
------
- y_true    : np.ndarray (N, K) in {0, 1}
- y_scores  : np.ndarray (N, K) probabilities or scores

Outputs
-------
- models/<run_name>/thresholds.json
    {
        "kind": "global" | "per_label",
        "value": <float or list[float]>,
        "meta": {
            "K": int | null,
            "notes": "tuned on val"
        }
    }
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import json
import logging
import numpy as np

from src.common.metrics import (
    compute_f1s
)
from sklearn.metrics import f1_score

from src.utils import (
    to_relpath
)

# =============================================================================
# Shape & dtype guard
# =============================================================================

def _ensure_shapes(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Validate shapes and coerce dtypes.

    Parameters
    ----------
    y_true  : np.ndarray
        Binary ground-truth matrix, expected shape (N, K).
    y_scores: np.ndarray
        Predicted probabilities/scores, expected shape (N, K).

    Returns
    -------
    y_true_c    : np.ndarray
        Binary labels coerced to uint8, shape (N, K).
    y_scores_c  : np.ndarray
        Scores coerced to float32, shape (N, K).
    N   : int
        Number of samples
    K   : int
        Number of labels
    
    Notes
    -----
    - Does not copy large arrays unnecessarily; only dtype coercions if needed.
    """
    if y_true.ndim != 2 or y_scores.ndim != 2:
        got = (getattr(y_true, "ndim", None), getattr(y_scores, "ndim", None))
        raise ValueError(f"[CHECK] shape mismatch: expected 2D arrays for y_true/y_scores (got ndim={got})")
    
    if y_true.shape != y_scores.shape:
        raise ValueError(f"[CHECK] shape mismatch: y_true.shape={y_true.shape} vs y_scores.shape={y_scores.shape}")
    
    if y_true.dtype not in (np.bool_, np.uint8):
        raise ValueError(f"[CHECK] y_true must be bool or uint8 (got {y_true.dtype})")
    
    if not np.issubdtype(y_scores.dtype, np.floating):
        raise ValueError(f"[CHECK] y_scores must be a floating dtype (got {y_scores.dtype})")
    
    if not np.isfinite(y_scores).all():
        raise ValueError(f"[CHECK] scores contain NaN/inf")
    
    if (y_scores < 0).any() or (y_scores > 1).any():
        raise ValueError("[CHECK] y_scores outside [0, 1]; expected probabilities")
    
    y_true_c = np.asarray(y_true, dtype=np.uint8)
    y_scores_c = np.asarray(y_scores, dtype=np.float32)

    N, K = y_true_c.shape
    return y_true_c, y_scores_c, N, K

# =============================================================================
# TApply thresholds (scalar or per label)
# =============================================================================

def apply_thresholds(y_scores: np.ndarray, thr: Union[float, np.ndarray]) -> np.ndarray:
    """
    Binarize scores with a scalar or per-label vector threshold.

    Paramaters
    ---------
    y_scores    : np.ndarray
        Predicted probabilities, shape (N, K) dtype float32.
    thr         : float or np.ndarray
        Global scalar or per-label vector with shape (K,)
    
    Returns
    -------
    y_bin   : np.ndarray
        Binary predictions, shape (N, K), dtype uint8 in {0, 1}
    
    Notes
    -----
    - Comparison is `>= thr`.
    - If per-label, broadcasting across rows is expected
    """
    if y_scores.ndim != 2:
        raise ValueError(f"[CHECK] y_scores must be 2D (got {y_scores.ndim})")
    N, K = y_scores.shape
    if not np.issubdtype(y_scores.dtype, np.floating):
        raise ValueError(f"[CHECK] y_scores must be a floating dtype (got {y_scores.dtype})")
    if not np.isfinite(y_scores).all():
        raise ValueError("[CHECK] y_scores contain NaN/inf")
    
    if (y_scores < 0).any() or (y_scores > 1).any():
        raise ValueError("[CHECK] y_scores outside [0, 1], expected probabilities.")
    
    if isinstance(thr, (float, int, np.floating, np.integer)):
        tau = float(thr)
        thr_arr = tau
    else:
        thr_arr = np.asarray(thr, dtype=np.float32)
        assert thr_arr.ndim == 1, f"[CHECK] per-label thr must be 1D (got {thr_arr.ndim}D)"
        assert thr_arr.shape[0] == K, f"[CHECK] thr shape (K,) mismatch (got {thr_arr.shape[0]} vs K={K})"
        thr_arr = thr_arr.reshape(1, K)

    y_bin = (y_scores >= thr_arr).astype(np.uint8)
    assert y_bin.shape == (N, K), f"[CHECK] output shape mismatch (got {y_bin.shape}, expected {(N, K)})"
    pos_rate = float(y_bin.mean())
    if not (0.0 <= pos_rate <= 1):
        raise RuntimeError("[CHECK] Invalid binarization result")
    
    return y_bin

# =============================================================================
# Global threshold tuner
# =============================================================================

def tune_global_thr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        grid: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Grid-search a single threshold tau* maximizing micro-F1 on the validation set.

    Parameters
    ----------
    y_true  : np.ndarray
        Binary ground-truth, shape (N, K).
    y_scores: np.ndarray
        Probabilities/scores, shape(N, K).
    grid    : np.ndarray, optional
        Candidate thresholds; default linspace[0.05 .. 0.95] step~0.05.
    logger  : logging.Logger, optional
        Logger for progess/selection info (No PHI).
    
    Returns
    -------
    tau_star : float
        Best scalar threshold by micro-F1 (tie-breaker: macro-F1).
    stats    : dict
        Summary with {"micro": float, "macro": float, "tau": float}
    
    Notes
    -----
    - Use `zero_division=0` for stability on rare labels.
    - Deterministic given (y_true, y_scores, grid).
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    y_true_c, y_scores_c, N, K = _ensure_shapes(y_true=y_true, y_scores=y_scores)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19).astype(np.float32)
    best_pair = (-1.0, -1.0)
    tau_star = 0.5
    for tau in grid:
        y_bin = apply_thresholds(y_scores_c, float(tau))
        micro, macro = compute_f1s(y_true_c, y_bin)
        if (micro, macro) > best_pair:
            best_pair = (micro, macro)
            tau_star = float(tau)
    stats = {
        "micro": float(best_pair[0]),
        "macro": float(best_pair[1]),
        "tau": tau_star
    }
    logger.info("[THR:global] tau=%.3f micro=%.4f macro=%.4f", tau_star, stats["micro"], stats["macro"])
    return tau_star, stats

# =============================================================================
# Per-label threshold tuner (with rare-label guard)
# =============================================================================

def tune_per_label_thr(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        base_tau: float = 0.5,
        min_count: int = 5,
        grid: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Independent grid-search per label i to maximize F1_i.
    - Label with < min_count positive fallback to base_tau.

    Returns
    -------
    tau_vec : np.ndarray
        Per-label threshold, shape (K, ), dtype float32.
    stats   : dict
        Summary F1 when applying tau_vec: {"micro": float, "macro": float}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    y_true_c, y_scores_c, N, K = _ensure_shapes(y_true=y_true, y_scores=y_scores)
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19).astype(np.float32)

    tau_vec = np.full((K,), float(base_tau), dtype=np.float32)
    for i in range(K):
        yi = y_true_c[:, i]
        si = y_scores_c[:, i]
        pos = int(yi.sum())
        if pos < min_count:
            # keep base_tau to avoid overfitting rare labels
            continue
        best_f1 = -1.0
        best_tau = float(base_tau)
        for tau in grid:
            ybi = (si >= tau).astype(np.uint8)
            f1i = f1_score(yi, ybi, average="binary", zero_division=0)
            if f1i > best_f1:
                best_f1 = float(f1i)
                best_tau = float(tau)
        tau_vec[i] = best_tau
    yb_full = apply_thresholds(y_scores_c, tau_vec)
    micro, macro = compute_f1s(y_true_c, yb_full)
    stats = {
        "micro": float(micro),
        "macro": float(macro),
    }
    logger.info("[THR:per_label] micro=%.4f macro=%.4f", stats["micro"], stats["macro"])
    return tau_vec, stats

# =============================================================================
# I/O - persist & load thresholds
# =============================================================================

def save_thresholds(
        models_dir: Union[str, Path],
        run_name: str,
        thr: Union[float, np.ndarray],
        *,
        kind: str = "global",           # {"global", "per_label"}
        logger: Optional[logging.Logger] = None,
) -> Path:
    """
    Persist thresholds to `models/<run_name>/thresholds.json`.

    Parameters
    ----------
    models_dir   : str or Path
        Base models directory, e.g., "models".
    run_name    : str
        Subdirectory name for this run.
    thr         : float or np.ndarray
        Scalar tau or per-label tau vector.
    kind        : {"global", "per_label"}, default="global"
        Thresholding scheme.
    logger      : logging.Logger, optional
        Logger for artifact path messages.
    
    Returns
    -------
    out_path    : Path
        Path to the saved JSON file.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if kind not in {"global", "per_label"}:
        raise ValueError(f"[CHECK] kind must be 'global' or 'per_label' (got {kind!r})")
    out_dir = Path(models_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    thr_arr = None
    if isinstance(thr, (list, tuple, np.ndarray)):
        thr_arr = np.asarray(thr, dtype=np.float32)
    if thr_arr is not None and thr_arr.ndim == 1:
        value = thr_arr.tolist()
        K = int(thr_arr.shape[0])
    else:
        value = float(thr)
        K = None
    
    obj = {
        "kind": str(kind),
        "value": value,
        "meta": {
            "K": K,
            "notes": "tuned on val"
        }
    }
    out_path = out_dir / "thresholds.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    
    logger.info("[THR:save] %s", to_relpath(out_path))
    return out_path
    
def load_thresholds(
        models_dir: Union[str, Path],
        run_name: str,
        *,
        logger: Optional[logging.Logger] = None,
) -> Tuple[str, Union[float, np.ndarray]]:
    """
    Load thresholds from `models/<run_name>/thresholds.json`, or return a safe default.

    Parameters
    ----------
    models_dir  : str or Path
        Base models directory.
    run_name    : str
        Subdirectory name for this run.
    
    Returns
    -------
    kind : str
        "global" or "per_label" (or "global" if defaulting).
    thr  : float or np.ndarray
        Loaded threshold(s); default is 0.5 if file missing.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    in_path = Path(models_dir) / run_name / "thresholds.json"
    if not in_path.exists():
        logger.warning("[THR:load] thresholds.json is missing, default thr=0.5 -> %s", to_relpath(in_path))
        return "global", 0.5
    
    with open(in_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    kind = obj.get("kind", "global")
    val = obj.get("value", 0.5)
    thr = np.asarray(val, dtype=np.float32) if isinstance(val, list) else float(val)

    if kind == "global" and isinstance(val, list):
        logger.warning("[THR:load] kind='global' but got vector value; using vector as-is")
    if kind == "per_label" and not isinstance(val, list):
        logger.warning("[THR:load] kind='per_label' but got scalar value; using scalar as-is")

    logger.info("[THR:load] kind=%s from %s", kind, to_relpath(in_path))
    return kind, thr