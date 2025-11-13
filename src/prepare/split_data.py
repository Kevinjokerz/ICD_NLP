"""
split_data.py - Split

Purpose
-------
Provide multiple splitting strategies for MIMIC-III ICD coding:
- subject_random   : random by SUBJECT_ID (no leakage)
- hadm_stratified : iterative stratified split at row level (multi-label)
- subject_stratified : iterative multi-label stratification at SUBJECT_ID level

Conventions
-----------
- Reproducible (single RNG seed)
- PHI-safe logging (no TEXT)
- project-relative paths
- 

Inputs
------
- data/processed/notes_icd.jsonl        # keys: subject_id, hadm_id, text, labels


Outputs
-------
- data/splits/{train, val, test}.jsonl
- data/splits/{train, val, test}_ids.txt    # SUBJECT_ID list (subject) or row indices (hadm_stratified)
- reports/split_manifest.json               # {strategy, seed, sizes, label_stats}
- reports/split_stats.txt                   # human-readable counts
"""

from __future__ import annotations
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, Optional, Union, List
import argparse
import json
import numpy as np
import pandas as pd
from scipy import sparse
import logging
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
from scipy.sparse import csr_matrix

from utils import (
    setup_logger, set_seed, read_jsonl, export_jsonl,
    to_relpath
)

from src.prepare.encode_labels import (
    load_label_map
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Subject-levl or admission-level stratified split for MIMIC ICD-9")

    parser.add_argument("--jsonl_full", type=Path, default=Path("data/processed/notes_icd.jsonl"), help="Full JSONL with TEXT + LABELS for joining back after split")
    parser.add_argument("--out_dir", type=Path, default=Path("data/splits"), help="Directory to save split JSONL files.")
    parser.add_argument("--reports_dir", type=Path, default=Path("reports"), help="Directory for QC and manifest outputs.")
    parser.add_argument("--strategy", type=str, choices=["subject_random", "hadm_stratified", "subject_stratified"], default="subject_stratified", help="Splitting strategy (default=subject_stratified)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed for reproducibility")
    parser.add_argument("--val_frac", type=float, default=0.15, help="Validation fraction (of total subjects).")
    parser.add_argument("--test_frac", type=float, default=0.15, help="Test fraction (of total subjects).")
    parser.add_argument("--parquet_in", type=Path, default=Path("data/processed/notes_icd_encoded.parquet"), help="Encoded parquet with SUBJECT_ID, HADM_ID, LABELS_IDX.")
    parser.add_argument("--label_map", type=Path, default=Path("data/processed/label_map.json"), help="JSON file mapping ICD code -> label index.")
    parser.add_argument("--top_m_preview", type=int, default=20, help="Top-M labels to preview in QC report.")



    return parser.parse_args()

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _load_min_columns(parquet_path: Union[str, Path], logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Read minimal columns from encoded parquet

    Parameters
    ----------
    parquet_path : str or Path
        Path to the encoded parquet file containing at least
        ['SUBJECT_ID', 'HADM_ID', 'LABELS_IDX'].
    logger      : logging.Logger, optional
        Existing logger for audit trail. Create a default logger if None.
    
    Returns
    -------
    pd.DataFrame
        Columns:
            - SUBJECT_ID : int64
            - HADM_ID    : int64
            - LABELS_IDX : list[int]
    
    Raises
    ------
    AssertionError
        If required columns are missing or invalid data types found.

    Notes
    -----
    - PHI-safe: Does not read or log any TEXT data.
    - Deterministic: Always returns columns in the same order.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    parquet_path = Path(parquet_path)
    assert parquet_path.exists(), f"Parquet file not found: {parquet_path}"

    df = pd.read_parquet(parquet_path, columns=["SUBJECT_ID", "HADM_ID", "LABELS_IDX"])

    required = {"SUBJECT_ID", "HADM_ID", "LABELS_IDX"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    
    df = df.dropna(subset=["SUBJECT_ID", "HADM_ID"])
    df = df.astype({"SUBJECT_ID": "int64", "HADM_ID": "int64"})
    df["LABELS_IDX"] = coerce_labels_idx_columns(df["LABELS_IDX"])
    df = df.reset_index(drop=True)
    assert all(isinstance(v, list) for v in df["LABELS_IDX"]), "LABELS_IDX must be list[int] per row"

    logger.info(f"[READ] {parquet_path.name} -> rows={len(df):,}, cols={list(df.columns)}")

    return df

def build_subject_level_csr(df: pd.DataFrame, K: int, logger: Optional[logging.Logger] = None) -> Tuple[pd.DataFrame, csr_matrix]:
    """
    Aggregate LABELS_IDX by SUBJECT_ID -> subject-level multi-hot CSR.

    Parameters
    ----------
    df  : pd.DataFrame
        Columns: SUBJECT_ID: int64, LABELS_IDX: list[int]
    K   : int
        Number of labels (len(label_map))
    
    Returns
    -------
    subj_df : pd.DataFrame
        One row per unique subject (col: SUBJECT_ID)
    Y_subj  : csr_matrix
        Shape (N_subjects, K), dtype=bool
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    required = {"SUBJECT_ID", "LABELS_IDX"}
    missing = required - set(df.columns)
    assert not missing, f"Missing required columns: {missing}"
    assert all(isinstance(v, list) for v in df["LABELS_IDX"]), "LABELS_IDX must be list[int] per row"

    groups = df.groupby("SUBJECT_ID", sort=True)["LABELS_IDX"].apply(list)
    subj_ids = groups.index.to_numpy(dtype=np.int64)
    N_subjects = len(subj_ids)
    assert N_subjects > 0

    # Build CSR indptr/indices with union per subject
    indptr: List[int] = [0]
    indices_buf: List[np.ndarray] = []
    for lsts in groups:
        merged = set()
        # lsts is a list of lists[int]
        for arr in lsts:
            merged.update(int(v) for v in arr)
        idx = np.fromiter(sorted(merged), dtype=np.int32)
        indices_buf.append(idx)
        indptr.append(indptr[-1] + len(idx))

    indices = np.concatenate(indices_buf) if indices_buf else np.array([], dtype=np.int32)
    data = np.ones(indices.shape[0], dtype=bool)
    indptr_arr = np.asarray(indptr, dtype=np.int64)

    if indices.size > 0:
        assert indices.min() >= 0, "Label index must be >=0"
        assert indices.max() < K, "Label index out of range"

    Y_subj = csr_matrix((data, indices, indptr_arr), shape=(N_subjects, K), dtype=bool)
    assert Y_subj.shape == (N_subjects, K)
    
    density = Y_subj.nnz / float(max(1, N_subjects*K))
    logger.info(f"[CSR] subjects={N_subjects:,}, K={K:,}, nnz={Y_subj.nnz:,}, density={density:.6f}")

    subj_df = pd.DataFrame({"SUBJECT_ID": subj_ids})
    return subj_df, Y_subj


def _build_temp_multi_hot(df: pd.DataFrame, logger=None) -> np.ndarray:
    """
    Build a temporary multi-hot matrix Y for stratification.

    Parameters
    ----------
    df  : pd.DataFrame
        Must contain columns LABELS as list[str]
    
    Returns
    -------
    np.ndarray
        Y with shape [N, K], dtype=uint8, where K = number of unique codes in df.
    
    Notes
    -----
    - This is only for splitting; training uses label_map from TRAIN-only later.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if "LABELS" not in df.columns:
        raise ValueError(f"Required column is missing: 'LABELS'")

    labels_col = df["LABELS"].apply(
        lambda v: v if isinstance(v, list) else ([] if pd.isna(v) else [v])
    ).values
    assert all(isinstance(x, list) for x in labels_col), "[CHECK] LABELS must be list."
    codes_set = sorted({str(c).strip().upper() for L in labels_col for c in L})
    if len(codes_set) == 0:
        raise ValueError("[CHECK] LABELS are empty across all rows; cannot stratify.")

    code2idx: Dict[str, int] = {code:i for i, code in enumerate(codes_set)}
    K = len(codes_set)
    N = len(df)
    Y = np.zeros((N, K), dtype=np.uint8)

    _norm = lambda x: str(x).strip().upper()
    for row, codes in enumerate(labels_col):
        for c in codes:
            c_norm = _norm(c)
            if c_norm in code2idx:
                Y[row, code2idx[c_norm]] = 1

    mem_MB = Y.nbytes / (1024**2)
    if mem_MB > 500:
        logger.warning(f"[WARN] Y~{mem_MB:.1f} MB -> using sparse build")
        Y_sparse = sparse.lil_matrix((N, K), dtype=np.uint8)
        for row, codes in enumerate(labels_col):
            for c in codes:
                c_norm = _norm(c)
                idx = code2idx.get(c_norm)
                if idx is not None:
                    Y_sparse[row, idx] = 1
        Y = Y_sparse.toarray()
        mem_MB = Y.nbytes / (1024**2)

    assert Y.shape == (N, K), f"[CHECK] Y shape: {Y.shape}, expected=({N}, {K})"
    assert Y.dtype == np.uint8, f"[CHECK] dtype={Y.dtype}, expected=uint8"
    assert Y.sum() > 0, "[CHECK] empty Y (no positive)"

    logger.info(f"[INFO] Y shape={Y.shape}, size~{mem_MB:.1f} MB")

    return Y

def _iterative_stratified_indices(
        Y: np.ndarray, *, seed: int, val_frac: float, test_frac: float, logger=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split indices into (train, val, test) using iterative multilabel stratification.

    Parameters
    ----------
    Y       : np.ndarray
        Multi-hot labels, shape [N, K]  dtype {0, 1}
    seed    : int
        Random seed for reproducibility.
    val_frac : float
        Fraction for validation set (0, 1).
    test_frac : float
        Fraction for test set (0, 1).
    
    Returns
    -------
    (idx_train, idx_val, idx_test) : tuple of 1D np.ndarray[int]
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    assert 0 < val_frac < 1 and 0 < test_frac < 1 and (val_frac + test_frac) < 1
    assert Y.ndim == 2, "[CHECK] Y must be 2D"
    Y = (Y > 0).astype(np.uint8)
    N = Y.shape[0]
    holdout = val_frac + test_frac
    n_splits = max(3, int(round(1.0 / holdout)))       # e.g., val+test = 0.3 -> ~3~4
    n_splits = min(n_splits, max(2, N - 1))

    # -- Iterative stratified KFold ---
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    X_dummy = np.zeros((N, 1), dtype=np.int8)
    for train_idx, holdout_idx in mskf.split(X_dummy, Y):
        break

    # -- Split holdout into val/test by proportion
    rng = np.random.default_rng(seed=seed)
    perm = rng.permutation(holdout_idx)
    assert len(perm) >= 2, "[SPLIT] holdout too small; increase n_splits or adjust val/test frac"

    n_hold = len(perm)
    n_val = int(round(val_frac / holdout * n_hold))
    n_val = min(max(1, n_val), n_hold - 1)
    idx_val = np.sort(perm[:n_val])
    idx_test = np.sort(perm[n_val:])
    idx_train = np.sort(train_idx)

    # --- Sanity checks ---
    assert len(set(idx_train) & set(idx_val)) == 0
    assert len(set(idx_train) & set(idx_test)) == 0
    assert len(set(idx_val) & set(idx_test)) == 0
    assert len(set(idx_train)) + len(set(idx_val)) + len(set(idx_test)) == N
    assert Y.ndim == 2 and Y.shape[0] == N, "[CHECK] Y rows changed unexpectedly"

    logger.info(f"[SPLIT] n_splits={n_splits} -> train={len(idx_train)} val={len(idx_val)} test={len(idx_test)}")

    return idx_train.astype(np.int64), idx_val.astype(np.int64), idx_test.astype(np.int64)

def _rowfreq_from_Y(Y_subj: csr_matrix, logger: Optional[logging.Logger]=None) -> np.ndarray:
    """
    Per-label counts from subject-level CSR.

    Parameters
    ----------
    Y_subj  : csr_matrix
        Shape (N_subjects, K), dtype=bool
    
    Returns
    -------
    np.ndarray
        Shape [K], dtype=float64 (counts per label)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    count = np.asarray(Y_subj.sum(axis=0)).ravel().astype(np.float64)
    assert count.ndim == 1, f"Count n-dimension mismatch: got {count.ndim}, expected 1"

    if count.sum() == 0:
        logger.warning("[WARN] _rowfreq_from_Y: all-zero label counts!")
    return count

def _js_divergence_counts(p_cnt: np.ndarray, q_cnt: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence from **count** vectors (auto-normalized to prob).

    Parameters
    ----------
    p_cnt, q_cnt   : np.ndarray
        Label count vector of equal length.
    eps     : float, default=1e-12
        Numerical stability constant

    Returns
    -------
    float
        Jensen-Shannon divergence (symmmetric, non-negative, <= log(2)).
    """
    assert p_cnt.shape == q_cnt.shape, f"Shape mismatch: {p_cnt.shape} vs {q_cnt.shape}"

    # Normalize to probabilities
    p = p_cnt / max(p_cnt.sum(), eps)
    q = q_cnt / max(q_cnt.sum(), eps)
    m = 0.5 * (p + q)

    kl = lambda a, b: float(np.sum(a * np.log((a + eps) / (b + eps))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def split_subjects_iterative(Y: csr_matrix, 
                             *,
                             seed: int, 
                             val_frac: float, 
                             test_frac: float, 
                             trials : int = 5,
                             logger: Optional[logging.Logger] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Subject-level iterative stratification with multi-trial **best-of-JSD** selection.

    Parameters
    ---------
    Y   : csr_matrix
        Shape (N_subjects, K), dtype=bool, Subject-level multi-hot labels.
    seed: int
        Global RNG seed.
    val_frac    : float
        Fraction of ALL subjects for validation.
    test_frac   : float
        Fraction of ALL subjects for test.
    trials      : int, default=5
        Number of resamplings; pick split with minimal worst-case JSD vs global.

    Returns
    -------
    (tr_idx, va_idx, te_idx)  : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices into subject axis (dtype=int64)

    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    N, K = Y.shape
    assert 0.0 < val_frac < 1.0 and 0.0 < test_frac < 1.0 and (val_frac + test_frac) < 1.0, "Require 0 < val_frac, test_frac and val_frac + test_frac < 1."
    assert trials >= 1, "trials must be >= 1"
    if hasattr(Y, "toarray"):
        Y_dense = Y.toarray().astype(np.uint8, copy=False)
    else:
        Y_dense = np.asarray(Y, dtype=np.uint8, order="C")

    X_dummy = np.zeros((N, 1), dtype=np.int8)

    global_cnt = _rowfreq_from_Y(Y, logger=logger)

    best_score = float("inf")
    best_split: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    for t in range(trials):
        # Stage 1 - Choose TEST (absolute fraction over ALL)
        m_test = MultilabelStratifiedShuffleSplit(n_splits=1, 
                    test_size=test_frac, random_state=seed + 1000 * t)
        trva_idx, te_idx = next(m_test.split(X_dummy, Y_dense))

        # Stage 2 - within TR+VAL pool, carve out VAL (conditional fraction)
        val_cond = val_frac / (1.0 - test_frac)
        m_val = MultilabelStratifiedShuffleSplit(n_splits=1, 
                    test_size=val_cond, random_state=seed + 2000 * t)
        tr_rel, val_rel = next(m_val.split(X_dummy[trva_idx], Y_dense[trva_idx]))
        tr_idx, va_idx = trva_idx[tr_rel], trva_idx[val_rel]

        # --- disjointness quick check ---
        assert len(set(tr_idx) & set(va_idx)) == 0
        assert len(set(tr_idx) & set(te_idx)) == 0
        assert len(set(va_idx) & set(te_idx)) == 0

        # --- evaluate JSD parity ---
        tr_cnt = _rowfreq_from_Y(Y[tr_idx], logger=logger)
        va_cnt = _rowfreq_from_Y(Y[va_idx], logger=logger)
        te_cnt = _rowfreq_from_Y(Y[te_idx], logger=logger)

        js_tr = _js_divergence_counts(global_cnt, tr_cnt)
        js_va = _js_divergence_counts(global_cnt, va_cnt)
        js_te = _js_divergence_counts(global_cnt, te_cnt)
        score = max(js_tr, js_va, js_te)

        if score < best_score:
            best_score = score
            best_split = (tr_idx, va_idx, te_idx)
        logger.debug(f"[TRY {t + 1}/{trials}] maxJSD={score:.6f} "
                     f"(tr={js_tr:.6f}, va={js_va:.6f}, te={js_te:.6f})")
    
    # Finalize
    assert best_split is not None, "No valid split found"
    tr_idx, va_idx, te_idx = best_split

    # Final disjointness
    assert len(set(tr_idx) & set(va_idx)) == 0
    assert len(set(tr_idx) & set(te_idx)) == 0
    assert len(set(va_idx) & set(te_idx)) == 0

    logger.info(f"[SPLIT] subjects: train={len(tr_idx):,} ({len(tr_idx)/N:.1%}), "
                f"val={len(va_idx):,} ({len(va_idx)/N:.1%}), "
                f"test={len(te_idx):,} ({len(te_idx)/N:.1%}); "
                f"trials={trials}, seed={seed}")

    return tr_idx.astype(np.int64), va_idx.astype(np.int64), te_idx.astype(np.int64)

def label_freq_from_rows(rows: pd.DataFrame, K: int) -> np.ndarray:
    """
    Count per-label frequency from row-level LABELS_IDX.

    Parameters
    ----------
    rows    : pd.DataFrame
        Must contain columns 'LABELS_IDX' as list[int] per row.
    K       : int
        Number of labels.
    
    Returns
    -------
    np.ndarray
        Shape [K], dtype=int64, where freq[i] is the number if rows containing label i.
    """
    assert "LABELS_IDX" in rows.columns, "[CHECK] Missing required column: 'LABELS_IDX'."
    freq = np.zeros(K, dtype=np.int64)

    for lst in rows["LABELS_IDX"]:
        if lst:
            idx = np.asarray(lst, dtype=np.int32)

            # Guard against out-of-range
            idx = idx[(idx >= 0) & (idx < K)]
            np.add.at(freq, idx, 1)
    assert freq.shape == (K,), f"[SHAPE] freq.shape={freq.shape}, expected=({K},)"
    assert (freq >=  0).all(), "[VALUE] Negative counts detected in label frequency array"
    return freq

def coerce_labels_idx_cell(x: Any) -> List[int]:
    """
    Normalize a single LABELS_IDX cell to list[int].
    
    Accepts: list/tuple/set/np.ndarray,JSON string/bytes like"[1, 2]",
    empty string -> [], None/NaN -> [].
    Raises TypeError on unsupported types; ValueError on bad JSON.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []

    if isinstance(x, list):
        return [int(v) for v in x]
    if isinstance(x, (tuple, set)):
        return [int(v) for v in x]
    if isinstance(x, np.ndarray):
        return [int(v) for v in x.tolist()]
    
    if isinstance(x, (bytes, bytearray)):
        try:
            arr = json.loads(x.decode("utf-8"))
            if not isinstance(arr, Iterable):
                raise ValueError("JSON must decode to an iterable")
            return [int(v) for v in arr]
        except Exception as e:
            raise TypeError(f"LABELS_IDX bytes not JSON-decodable: {x!r}") from e
        
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                if not isinstance(arr, Iterable):
                    raise ValueError("JSON must decode to an iterable")
                return[int(v) for v in arr]
            except Exception as e:
                raise TypeError(f"LABELS_IDX string not valid JSON list: {s[:120]}...") from e
    
    raise TypeError(f"Unsupported LABELS_IDX cell type {type(x)}; sample={str(x)[:80]!r}")

def coerce_labels_idx_columns(series: pd.Series) -> pd.Series:
    """
    Normalize an entire LABELS_IDX column to list[int] per row
    """
    out = series.map(coerce_labels_idx_cell)

    bad_row = next(
        (i for i, lst in enumerate(out)
          if not isinstance(lst, list) or not all(isinstance(v, (int, np.integer)) for v in lst)),
          None
    )

    assert bad_row is None, f"[CHECK] Non-list or non-int found in LABELS_IDX at row {bad_row}"

    if any((len(lst) and min(lst) < 0) for lst in out):
        raise ValueError("[CHECK] Negative label index detected in LABELS_IDX")
    
    return out
            

def write_manifest_and_qc(df_all: pd.DataFrame, splits: Dict[str, pd.DataFrame],
                          K: int, out_dir: Path, top_m: int = 20,
                          *, logger: Optional[logging.Logger] = None,
                          idx_to_code: Optional[Dict[int, str]] = None
) -> None:
    """
    Save split sizes, JSD vs global, and Top-M label preview CSV (per split).

    Parameters
    ----------
    df_all  : pd.DataFrame
        Full encoded rows with LABELS_IDX.
    splits  : Dict[str, pd.DataFrame]
        Keys like {"train", "val", "test"}; values contain SUBJECT_ID, HADM_ID, LABELS_IDX,...
    K       : int
        Number of labels (len(label_map))
    out_dir : Path
        Parent directory of split JSONL files.
    top_m   : int, default=20
        Number of most frequent labels (global) to preview in CSV.
    logger  : logging.Logger, optional
        For diagnostics; if None a module logger is used.
    idx_to_code : dict[int, str], optional
        If provided, add LABEL_CODE column to preview.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    rep_dir = (out_dir / "reports")
    rep_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = rep_dir / "split_manifest.json"
    preview_path = rep_dir / "top_labels_preview.csv"

    # Sizes
    stats = {
        name: {
            "rows": int(len(d)),
            "subjects": int(d["SUBJECT_ID"].nunique()),
            "hadm": int(d["HADM_ID"].nunique())
        }
        for name, d in splits.items()
    }
    
    # Label parity
    global_freq = label_freq_from_rows(df_all, K=K)
    per_freq = {n: label_freq_from_rows(d, K) for n, d in splits.items()}
    jsd = {n: _js_divergence_counts(global_freq, per_freq[n]) for n in splits}

    # Write manifest (JSON)
    manifest = {"sizes": stats, "jsd_vs_global": jsd}
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    logger.info(f"[QC] manifest -> {to_relpath(manifest_path)}")

    # 5) Build Top-M label preview (by global frequency)
    top_m = max(1, min(int(top_m), K))
    idx_order = np.argsort(global_freq)[::-1][:top_m]

    split_order = [n for n in ["train", "val", "test"] if n in splits] + [n for n in splits.keys() if n not in {"train", "val", "test"}]
    rows = []
    for i in idx_order:
        r = {
            "LABEL_IDX": int(i),
            "GLOBAL": int(global_freq[i])
        }
        if idx_to_code is not None and i in idx_to_code:
            r["LABEL_CODE"] = idx_to_code[i]
        for name in split_order:
            r[name.upper()] = int(per_freq[name][i])
        rows.append(r)
    
    df_preview = pd.DataFrame(
        rows, columns=(["LABEL_IDX", "LABEL_CODE"] if idx_to_code else ["LABEL_IDX"])
        + ["GLOBAL"] + [n.upper() for n in split_order]
    )
    df_preview.to_csv(preview_path, index=False)
    logger.info(f"[QC] top-M preview -> {to_relpath(preview_path)}")

    assert df_preview.shape[0] == top_m, f"[SHAPE] preview_rows={df_preview.shape[0]}, expected={top_m}"
    assert all(freq.shape == (K,) for freq in per_freq.values()), f"[SHAPE] per_freq shapes not all == ({K},)"

    jsd_vals = np.array(list(jsd.values()), dtype=float)
    assert np.isfinite(jsd_vals).all(), "[VALUE] Non-finite JSD detected"

# =============================================================================
# Main Orchestration
# =============================================================================

def main() -> None:
    args = parse_args()
    assert 0.0 < args.val_frac < 1.0 and 0.0 < args.test_frac < 1.0 and (args.val_frac + args.test_frac) < 1.0, \
        "[CHECK] Require 0 < val_frac, test_frac and val_frac + test_frac < 1"

    logger = setup_logger("split", log_dir=Path("logs"))
    set_seed(args.seed)

    # 1) Load dataset
    df = read_jsonl(args.jsonl_full, logger=logger)
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    missing = required - set(df.columns)
    assert not missing, f"[CHECK] Missing required columns: {missing}"

    assert df["SUBJECT_ID"].notna().all(), "[CHECK] Null SUBJECT_ID values"
    assert df["HADM_ID"].notna().all(), "[CHECK] Null HADM_ID values"

    assert df["LABELS"].map(lambda x: isinstance(x, list)).all(), "[CHECK] LABELS must be list[str]"
    assert df["LABELS"].map(len).gt(0).any(), "[CHECK] No LABELS present in any row"

    # 2) Strategy
    if args.strategy == "subject_random":
        # --- subject-level random split ---
        subs = df["SUBJECT_ID"].dropna().astype(int).unique().tolist()
        rng = np.random.default_rng(seed=args.seed)
        rng.shuffle(subs)

        n = len(subs)
        n_val = max(1, int(args.val_frac * n)) if n >= 3 else int(args.val_frac * n)
        n_test = max(1, int(args.test_frac * n)) if n >= 3 else int(args.test_frac * n)
        n_train = max(0, n - n_val - n_test)

        # split subject set
        train_set = set(subs[:n_train])
        val_set = set(subs[n_train:n_train + n_val])
        test_set = set(subs[n_train + n_val:])

        # boolean masks by SUBJECT_ID
        part = df["SUBJECT_ID"].map(lambda s: "train" if s in train_set else ("val" if s in val_set else "test"))
        idx_train = np.where(part.eq("train"))[0]
        idx_val = np.where(part.eq("val"))[0]
        idx_test = np.where(part.eq("test"))[0]

        ids_artifacts = {
            "train_ids" : sorted(list(train_set)),        # SUBJECT_IDs
            "val_ids"   : sorted(list(val_set)),
            "test_ids"  : sorted(list(test_set)),
            "ids_kind"  : "SUBJECT_ID"
        }

    elif args.strategy == "hadm_stratified":
        # --- hadm_stratified (multi-label, row-level) ---
        Y = _build_temp_multi_hot(df, logger=logger)

        idx_train, idx_val, idx_test = _iterative_stratified_indices(
            Y, seed=args.seed, val_frac=args.val_frac,
            test_frac=args.test_frac, logger=logger
        )

        ids_artifacts = {
            "train_ids" : idx_train.tolist(),
            "val_ids"   : idx_val.tolist(),
            "test_ids"  : idx_test.tolist(),
            "ids_kind"  : "ROW_INDEX"
        }
    else:
        df_enc = _load_min_columns(args.parquet_in, logger=logger)
        label_map = load_label_map(args.label_map)
        K = int(len(label_map))
        idx_to_code = {idx: code for code, idx in label_map.items()}

        subj_df, Y_subj = build_subject_level_csr(df=df_enc, K=K, logger=logger)
        tr_idx, va_idx, te_idx = split_subjects_iterative(
            Y_subj, seed=args.seed, val_frac=args.val_frac, test_frac=args.test_frac,trials=5, logger=logger
        )
        tr_subj = set(subj_df.iloc[tr_idx, 0].tolist())
        va_subj = set(subj_df.iloc[va_idx, 0].tolist())
        te_subj = set(subj_df.iloc[te_idx, 0].tolist())

        m = df["SUBJECT_ID"].astype(int)
        idx_train = np.where(m.isin(tr_subj))[0]
        idx_val = np.where(m.isin(va_subj))[0]
        idx_test = np.where(m.isin(te_subj))[0]

        assert set(df.iloc[idx_train]["SUBJECT_ID"]).isdisjoint(df.iloc[idx_val]["SUBJECT_ID"])
        assert set(df.iloc[idx_train]["SUBJECT_ID"]).isdisjoint(df.iloc[idx_test]["SUBJECT_ID"])
        assert set(df.iloc[idx_val]["SUBJECT_ID"]).isdisjoint(df.iloc[idx_test]["SUBJECT_ID"])
        assert set(df.iloc[idx_train]["HADM_ID"]).isdisjoint(df.iloc[idx_val]["HADM_ID"])
        assert set(df.iloc[idx_train]["HADM_ID"]).isdisjoint(df.iloc[idx_test]["HADM_ID"])
        assert set(df.iloc[idx_val]["HADM_ID"]).isdisjoint(df.iloc[idx_test]["HADM_ID"])

        ids_artifacts = {
            "train_ids": sorted(list(tr_subj)),
            "val_ids": sorted(list(va_subj)),
            "test_ids": sorted(list(te_subj)),
            "ids_kind": "SUBJECT_ID"
        }
    
    # 3) Save splits
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_jsonl(df.iloc[idx_train], out_dir / "train.jsonl")
    export_jsonl(df.iloc[idx_val], out_dir / "val.jsonl")
    export_jsonl(df.iloc[idx_test], out_dir / "test.jsonl")

    # 4) Save IDs to lists (for reproducibility)
    for split, ids in [("train", "train_ids"), ("val", "val_ids"), ("test", "test_ids")]:
        p = out_dir / f"{split}_ids.txt"
        with p.open("w", encoding="utf-8") as f:
            for v in ids_artifacts[ids]:
                f.write(f"{v}\n")

    # 5) Reports: manifest + human-readable stats
    manifest = {
        "strategy"  : args.strategy,
        "seed"      : int(args.seed),
        "jsonl_full"  : to_relpath(args.jsonl_full),
        "out_dir"   : to_relpath(out_dir),
        "N_rows"    : len(df),
        "rows"       : {
            "train": int(len(idx_train)),
            "val": int(len(idx_val)), 
            "test": int(len(idx_test))
        },
        "ids_kind"  : ids_artifacts["ids_kind"]
    }

    rep_dir = Path(args.reports_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    with (rep_dir / "split_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    
    with (rep_dir / "split_stats.txt").open("w", encoding="utf-8") as f:
        f.write("Split stats\n-----------\n")
        f.write(f"Rows: train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}\n")
    
    if args.strategy == "subject_stratified":
        tr_rows = df_enc[df_enc["SUBJECT_ID"].isin(tr_subj)]
        va_rows = df_enc[df_enc["SUBJECT_ID"].isin(va_subj)]
        te_rows = df_enc[df_enc["SUBJECT_ID"].isin(te_subj)]
        splits_df = {
            "train": tr_rows,
            "val": va_rows,
            "test": te_rows
        }
        write_manifest_and_qc(df_all=df_enc, splits=splits_df, K=K, out_dir=out_dir, top_m=args.top_m_preview,
                              logger=logger, idx_to_code=idx_to_code)
    
    logger.info(f"[SPLIT] rows -> train={len(idx_train)}, val={len(idx_val)}, test={len(idx_test)}, total={len(df)}")
    logger.info(f"[SAVE] splits -> {to_relpath(out_dir)}; manifest -> {to_relpath(rep_dir / 'split_manifest.json')}")
    logger.info("[DONE] split complete.")

if __name__ == "__main__":
    main()
        
