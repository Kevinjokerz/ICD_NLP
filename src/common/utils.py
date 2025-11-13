"""
baseline_utils.py - Shared helpers for ICD-9 Baselines

Purpose
-------
Reusable utilities to:
- read_splits + label_map
- convert LABELS -> multi-hot Y
- compute Micro/Macro-F1 and Precision@k
- append metrics to a CSV for experiment tracking

Conventions
-----------
- Reproducibility via src.utils.set_seed
- Portability: relative paths only; no PHI in logs
- Metrics-first: Micro/Macro-F1, P@8/15

Inputs
------
- data/splits/{train, val, test}.jsonl      # keys: text, labels, subject_id, hadm_id
- data/processed/label_map.json             # {"<ICD>: idx}

Outputs
-------
- reports/baseline_metrics                  # one row per evaluation
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Iterable, Union
import logging
import json
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score
from src.utils import (
    set_seed, setup_logger, read_jsonl, 
    ensure_parent_dir, to_relpath, to_list_str
)

from src.common.metrics import (
    compute_f1s, precision_at_k
)

from src.common.thresholds import (
    tune_global_thr,
    tune_per_label_thr,
    apply_thresholds,
    save_thresholds,
    load_thresholds,
)

# =============================================================================
# Label space & data loading
# =============================================================================

def load_label_map(path: Union[Path, str] = Path("data/processed/label_map.json"), logger:Optional[logging.Logger] = None) -> Dict[str, int]:
    """
    Load mapping {ICD -> idx} and validate contiguity 0..K-1.

    Parameters
    ----------
    path    : Union[Path, str], default="data/processed/label_map.json"
        Path to the JSON file containing the label map in the form ``{"ICD9_CODE": index}``.
    logger  : logging.Logger, optional
        Existing logger for audit trail. If None, a module-level logger is created.

    Returns
    -------
    dict[str, int]
        Dictionary mapping ICD9 codes to integer indices.
        Keys are strings (e.g., "4019"), values are contiguous int [0, K-1]

    Notes
    -----
    - Ensure labels indices are contiguous 0..K-1 for stable multi-hot encoding (Y[:,idx]).
    - Used consistently across all baselines
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    label_map_p  = Path(path)
    if not label_map_p.exists():
        raise FileNotFoundError(f"[CHECK] file not existed: {to_relpath(label_map_p)}")
    
    with label_map_p.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    indices = list(mapping.values())
    K = len(indices)
    assert len(set(indices)) == K, "[LABEL_MAP] Duplicate indices detected"

    min_idx, max_idx = min(indices), max(indices)
    assert min_idx == 0 and max_idx == (K - 1), f"[LABEL_MAP] Non-contiguous indices: min={min_idx}, max={max_idx}, expected 0..{K-1}"

    logger.info(f"[LABEL_MAP] Loaded {K} labels (0..{K-1})")
    return mapping

def read_split(jsonl_path: Union[Path, str], logger: Optional[logging.Logger]=None) -> pd.DataFrame:
    """
    Read a split JSONL and normalize columns for baseline training.

    Parameters
    ----------
    jsonl_path  : Union[Path, str]
        Path to JSONL with keys: SUBJECT_ID, HADM_ID, TEXT, LABELS (list[str])
        Backward-compatible: accepts aliases like 'labels', 'codes', 'LABELS_ID'.

    Returns
    -------
    df  : pd.DataFrame
        DataFrame with columns:
            - SUBJECT_ID : int64
            - HADM_ID    : int64
            - TEXT       : str
            - LABELS     : list[str]
    
    Notes
    -----
    - PHI safety: never log raw TEXT.
    - This function only validate schema; no label map applied here
    """

    if logger is None:
        logger = logging.getLogger(__name__)
    
    split_path = Path(jsonl_path)
    if not split_path.exists():
        raise FileNotFoundError(f"[CHECK] File not found: {to_relpath(split_path)}")
    
    df = read_jsonl(split_path)
    df.columns = [c.upper() for c in df.columns]
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    missing = required - set(df.columns)
    assert not missing, f"[CHECK] Missing required columns: {missing}"

    df["SUBJECT_ID"] = pd.to_numeric(df["SUBJECT_ID"], errors="raise").astype("int64")
    df["HADM_ID"] = pd.to_numeric(df["HADM_ID"], errors="raise").astype("int64")
    df["TEXT"] = df["TEXT"].astype(str)

    df["LABELS"] = df["LABELS"].map(lambda x: to_list_str(x, remove_dots=True))
    assert df["LABELS"].map(lambda x: isinstance(x, list)).all(), "[CHECK] LABELS must be list[str]"

    assert df["SUBJECT_ID"].notna().all() and df["HADM_ID"].notna().all(), "[CHECK] Null IDs"
    assert df["LABELS"].map(len).gt(0).any(), "[CHECK] All rows have empty LABELS"

    df = df.reset_index(drop=True)

    logger.info(f"[READ SPLIT] {to_relpath(split_path)} -> rows={len(df):,}, cols={list(df.columns)}")
    lbl_nonempty = int((df["LABELS"].map(len) > 0).sum())
    logger.info(f"[LABELS] non_empty_rows={lbl_nonempty:,} / {len(df):,}")

    return df[["SUBJECT_ID", "HADM_ID","TEXT", "LABELS"]]

def labels_to_multi_hot(labels_col: Iterable[List[str]], 
                        label_map: Dict[str, int],
                        logger: Optional[logging.Logger] = None
) -> np.ndarray:
    """
    Convert list[str] labels per row -> multi-hot matrix Y (N x K) with value in {0, 1}.

    Parameters
    ----------
    labels_col  : iterable of list[str]
        Each element is the list of ICD codes for one sample.
    label_map   : {code: idx}
        Global mapping {code: contiguous index 0..K-1}
    logger      : logging.Logger, optional
        For audit trail; if None, uses module logger

    Returns
    -------
    np.ndarray
        Multi-hot binary matrix of shape (N, K), dtype=uint8
    
    Notes
    -----
    - Unknown codes (not in `label_map`) are ignored but counted in warnings.
    - PHI safety: codes are not logged verbatim.
    - Deterministic; does not modify input list
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    N = len(labels_col)
    K = len(label_map)
    Y = np.zeros((N, K), dtype=np.uint8)
    norm = lambda s: str(s).strip().upper()

    unknow_total = 0
    for i, codes in enumerate(labels_col):
        for c in codes:
            idx = label_map.get(norm(c))
            if idx is not None:
                Y[i, idx] = 1
            else:
                unknow_total += 1
    
    if unknow_total > 0:
        logger.warning(f"[LABELS] {unknow_total:,} unknown codes ignored during encoding")
    
    assert Y.sum() > 0, f"[CHECK] Empty multi-hot matrix (no positive labels)"
    assert Y.shape == (N, K), f"[CHECK] Y mismatch: expected {(N, K)} got {Y.shape}"

    logger.info(f"[ENCODE] Multi-hot built -> shape={Y.shape}, density={Y.mean():.4f}")
    return Y

def prepare_split(jsonl_path: Union[Path, str], 
                  label_map_path: Union[Path, str] = Path("data/processed/label_map.json"),
                  logger: Optional[logging.Logger]= None
) -> Tuple[List[str], np.ndarray, pd.DataFrame, Dict[str, int]]:
    """
    Load split and return (texts, Y_true, meta, label_map).

    Parameters
    ----------
    jsonl_path  : Path | str
        Path to split JSONL (columns: SUBJECT_ID, HADM_ID, TEXT, LABELS).
    label_map_path : Path | str, default="data/processed/label_map.json"
        JSON mapping {"ICD": idx} with contiguous indices 0..K-1.
    logger      : logging.Logger, optional
        Logger for audit trail.

    Returns
    -------
    texts   : list[str]
        Model input (len = N)
    Y       : np.ndarray
        Multi-hot binary matrix, shape (N, K), dtype=uint8
    meta    : pd.DataFrame
        columns: SUBJECT_ID, HADM_ID
    lm      : dict[str, int]
        Label map
    
    Notes
    -----
    - PHI-safe: no raw TEXT logged
    - Deterministic: No RNG used here
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    split_p = Path(jsonl_path)
    if not split_p.exists():
        raise FileNotFoundError(f"[CHECK] File not found: {to_relpath(split_p)}")
    
    label_map_p = Path(label_map_path)
    if not label_map_p.exists():
        raise FileNotFoundError(f"[CHECK] Label map not found: {to_relpath(label_map_p)}")
    
    # Load & validate (read_split already normalizes schema)
    df = read_split(split_p, logger=logger)
    lm = load_label_map(label_map_p, logger=logger)

    #Build Y and texts/meta
    Y = labels_to_multi_hot(df["LABELS"], lm, logger=logger)
    texts = df["TEXT"].astype(str).tolist()
    meta = df[["SUBJECT_ID", "HADM_ID"]].copy()

    assert len(texts) == Y.shape[0], f"[CHECK] texts vs Y rows: {len(texts)} != {Y.shape[0]}"
    assert Y.shape[1] == len(lm), f"[CHECK] Y cols vs label_map: {Y.shape[1]} != {len(lm)}"
    assert meta["SUBJECT_ID"].dtype =="int64" and meta["HADM_ID"].dtype == "int64", "[CHECK] meta dtypes must be int64"

    logger.info(f"[PREP] split={to_relpath(split_p)} -> N={len(texts):,}, K={len(lm):,}, density={Y.mean():.4f}")
    return texts, Y, meta, lm

def evaluate_probs(y_true: np.ndarray, 
                   y_scores: np.ndarray,
                   *,
                   ks: Tuple[int, int]=(8, 15),
                   thr: float=0.5
) -> dict:
    """
    Evaluate probabilistic model outputs against ground truth labels.

    Parameters
    ----------
    y_true  : np.ndarray
        Ground-truth binary labels, shape (N, K), dtype={0, 1}
    y_scores: np.ndarray
        Predicted probabilities or confidence scores, shape (N, K), float32/float64.
    ks      : Tuple[int, int], default=(8, 15)
        Values of k for computing Precision@K.
    thr     : float
        Threshold for converting scores to binary predictions.
    
    Returns
    -------
    metrics : dict[str, float]
        Dictionary with keys:
        - "micro_f1" : Micro-average F1 score.
        - "macro_f1" : Macro-average F1 score.
        - "p@8"      : Precision@8
        - "p@15"     : Precision@15
        - "thr"      : threshold used
    
    Notes
    -----
    - PHI-safe: does not log raw labels or scores.
    - Deterministic: no randomness, reproducible evaluation.
    - Intended for use in baseline and neural model evaluation.
    """

    assert y_true.shape == y_scores.shape, f"[CHECK] Shape mismatch: {y_true.shape} vs. {y_scores.shape}"

    y_bin = (y_scores >= thr).astype(np.uint8)
    micro, macro = compute_f1s(y_true, y_bin)

    p8 = precision_at_k(y_true, y_scores, ks[0])
    p15 = precision_at_k(y_true, y_scores, ks[1])

    metrics = {
        "micro_f1": float(micro),
        "macro_f1": float(macro),
        f"p@{ks[0]}": float(p8),
        f"p@{ks[1]}": float(p15),
        "thr" : float(thr)
    }

    return metrics

# =============================================================================
# Logging & Saving
# =============================================================================

def make_logger(name: str = "baseline") ->logging.Logger:
    """
    Create a logger
    """
    return setup_logger(name, log_dir=Path("logs"))

def set_global_seed(seed: int = 42) -> None:
    """
    set all RNG seeds (numpy/torch if available)
    """
    set_seed(seed)

def append_metrics_row(csv_path: Union[Path, str], row: dict) -> None:
    """
    Append one metrics row to a CSV file (create header if missing).

    Parameters
    ----------
    csv_path    : Path | str
        Destination CSV file, e.g, "reports/baseline_results.csv"
        Parent directory is created automatically if it does not exist.
    row         : dict
        Dictionaray of metrics and metadata to append as one row.
        Expected keys (recommended):
        ["run_name", "split", "N", "K",
        "micro_f1", "macro_f1", "p@8", "p@15", "thr"]
    
    Notes
    -----
    - Deterministic and idempotent (appending same row twice adds duplicates, not overwrite).
    - Use UTF-8 encoding and newline="\n" for portability.
    - This function is typically called at the end of a training/eval run.
    """

    csv_path = Path(csv_path)
    ensure_parent_dir(csv_path)
    df = pd.DataFrame([row])
    file_exists = csv_path.exists()

    df.to_csv(csv_path, mode="a" if file_exists else "w", header=not file_exists, index=False, encoding="utf-8")
    df_tail = pd.read_csv(csv_path).tail(1)

    assert set(row.keys()).issubset(df_tail.columns), f"[CHECK] Missing keys: {set(row.keys()) - set(df_tail.columns)}"

# =============================================================================
# Convenience (called by model runners)
# =============================================================================

def evaluate_and_save(
        run_name: str,
        split_name: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        *,
        thr: float = 0.5,
        threshold_policy: str = "fixed",        # {"fixed", "global_val", "per_label_val", "use_saved"}
        metrics_csv: Union[Path, str] = Path("reports/baseline_metrics.csv"),
        min_pos: int = 10,
        thresholds_dir: Union[str, Path] = Path("models"),
        logger: Optional[logging.Logger] = None,
) -> dict:
    """
    Compute evaluation metrics from y_scores, (optionally) tune/apply thresholds,
    append a CSV row, and return a compact metrics dict.

    Parameters
    ----------
    run_name    : str
        Experiment/model identifier (e.g., "tfidf_lr_v1").
    split_name  : str
        Data split name: "val", "test".
    y_true      : np.ndarray
        Ground-truth binary labels, shape (N, K), dtype={0, 1}
    y_scores    : np.ndarray
        Predicted probabilities/scores, shape (N, K), float32/float64
    metrics_csv : Union[Path, str], default="reports/baseline_metrics.csv"
        Path to an aggregated metrics CSV (appended per call).
    thr         : float, default=0.5
        Threshold for binarizing y_scores (used for F1 metrics).
    threshold_policy : {"fixed", "global_val", "per_label_val", "use_saved"}
        Threshold tuning policy
    min_pos     : int, default=10
        Minimum number of positives/labels to allow per-label-tuning;
        if less than fallback to global threshold.
    thresholds_dir: Path | str
        Root directory to save/read thresholds (default="models/<run_name>/thresholds.json")
    logger      : logging.Logger, optional
        Logger for audit trail.
    
    Returns
    -------
    dict
        Metrics dictionary as produced by `evaluate_probs`, with keys:
        {"run_name", "split", "N", "K", "micro_f1", "macro_f1", "p@8", "p@15",
        "thr_kind", "thr_policy", "thr_notes"}
    
    Notes
    -----
    - Deterministic: no randomness here.
    - PHI-safe: does not log raw predictions or text
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    assert isinstance(y_true, np.ndarray) and isinstance(y_scores, np.ndarray), "[CHECK] y_true, y_scores must be np.ndarray"
    assert y_true.ndim == 2 and y_scores.ndim == 2, \
        f"[CHECK] expect y_true, y_scores to be 2D array (got y_true.ndim={y_true.ndim}, y_scores.ndim={y_scores.ndim})"
    assert y_true.shape == y_scores.shape, f"[CHECK] shape mismatch: {y_true.shape} vs. {y_scores.shape}"
    assert y_true.dtype in (np.bool_, np.uint8, np.int8, np.int16, np.int32, np.int64), f"[CHECK] y_true must be binary dtype (got {y_true.dtype})"
    assert np.isfinite(y_scores).all(), "[CHECK] y_scores has NaN/Inf"
    assert (y_scores >= 0).all() and (y_scores <=1).all(), "[CHECK] y_scores must be in [0, 1] (expect probability)"

    split = str(split_name).strip().lower()
    policy = str(threshold_policy).strip().lower()
    assert split in {"val", "test"}, f"[CHECK] unknown split_name={split}"
    assert policy in {"fixed", "global_val", "per_label_val", "use_saved"}, f"[CHECK] unknown policy={policy}"


    N, K = int(y_true.shape[0]), int(y_true.shape[1])
    logger.info(f"[EVAL:init] run={run_name}, split={split}, policy={policy}, N={N:,}, K={K:,}")

    # --- thresholds: select/apply ---
    models_dir = Path(thresholds_dir)
    bin_pred: np.ndarray
    thr_kind, thr_notes = "fixed", ""

    if policy == "fixed":
        bin_pred = (y_scores >= float(thr)).astype(np.uint8, copy=False)
        thr_kind = "fixed"
        thr_notes = f"t={float(thr):.2f}"
    elif split == "val" and policy == "global_val":
        best_t, _stats = tune_global_thr(y_true, y_scores, logger=logger)
        bin_pred = (y_scores >= float(best_t)).astype(np.uint8, copy=False)
        thr_kind = "global"
        save_thresholds(models_dir=models_dir, run_name=run_name, thr=float(best_t), kind=thr_kind, logger=logger)
        thr_notes = f"t={float(best_t):.2f}"
    elif split == "val" and policy == "per_label_val":
        base_t, _stats = tune_global_thr(y_true=y_true, y_scores=y_scores, logger=logger)
        t_vec, pl_stats = tune_per_label_thr(y_true=y_true, y_scores=y_scores, base_tau=base_t, min_count=int(min_pos), logger=logger)
        bin_pred = apply_thresholds(y_scores=y_scores, thr=t_vec)
        thr_kind = "per_label"
        save_thresholds(models_dir=models_dir, run_name=run_name, thr=t_vec, kind=thr_kind, logger=logger)
        thr_notes = f"base={float(base_t):.2f} tuned={int((np.asarray(t_vec).ndim==1))}D"
    elif split == "test" and policy == "use_saved":
        kind, thr = load_thresholds(models_dir=models_dir, run_name=run_name, logger=logger)
        bin_pred = apply_thresholds(y_scores=y_scores, thr=thr)
        thr_kind = str(kind)
        thr_notes = "loaded"
    else:
        bin_pred = (y_scores >= float(thr)).astype(np.uint8, copy=False)
        thr_kind = "fixed"
        thr_notes = f"t={float(thr):.2f} [fallback]"
    
    assert bin_pred.shape == y_true.shape and bin_pred.dtype == np.uint8, "[CHECK] invalid bin_pred"

    micro_f1, macro_f1 = compute_f1s(y_true, bin_pred)
    p_at_8 = precision_at_k(y_true=y_true, y_scores=y_scores, k=8)
    p_at_15 = precision_at_k(y_true=y_true, y_scores=y_scores, k=15)

    row = {
        "run_name": run_name,
        "split": split,
        "N": N,
        "K": K,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
        "p@8": p_at_8,
        "p@15": p_at_15,
        "thr_kind": thr_kind,
        "thr_policy": policy,
        "thr_notes": thr_notes
    }

    Path(metrics_csv).parent.mkdir(parents=True, exist_ok=True)
    append_metrics_row(csv_path=metrics_csv, row=row)

    logger.info("[EVAL:done] %s | micro=%.4f | macro=%.4f | p@8=%.4f | p@15=%.4f | kind=%s %s",
                split, row["micro_f1"], row["macro_f1"], row["p@8"], row["p@15"], row["thr_kind"], row["thr_notes"])
    
    return row
