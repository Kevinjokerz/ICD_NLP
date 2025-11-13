"""
Encode Multi-Hot Labels

Purpose
-------
Convert ICD code lists (strings) to multi-hot vector using a label map.
Saves:
- Parquet: Row metadata + LABELS_IDX (list[int])
- CSR .npz (or dense .npy): multi-hot matrix for training
- QA text report with basic stats and previews

Conventions
-----------
- Reproducibility: fixed seed; portable, project-relative paths.
- PHI safety: never log raw PHI; only synthetic previews if needed.

Inputs
------
- data/processed/notes_icd.jsonl
    Keys per line: subject_id, hadm_id, text, labels (list[str])
- data/processed/label_map.json
    Mapping: {"<ICD>": int_index}

Outputs
-------
- data/processed/notes_icd_encoded.parquet
    Columns: SUBJECT_ID, HADM_ID, TEXT, LABELS_IDX (list[int])
- data/processed/Y_multi_hot.npz (CSR) or data/processed/Y_dense.npy
- reports/encoding_summary.txt

Example
-------
>>>

"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import argparse
import json
import logging

import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq
from collections import Counter

from src.utils import (
    setup_logger, set_seed, ensure_parent_dir, read_jsonl, to_relpath
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with paths and flags.
    """
    
    parser = argparse.ArgumentParser(
        description="Encode ICD labels to multi-hot vectors."
    )

    parser.add_argument(
        "--jsonl_in",
        type=Path,
        required=False,
        default=Path("data/processed/notes_icd.jsonl"),
        help="Input JSONL with SUBJECT_ID/HADM_ID/TEXT/LABELS."
    )
    parser.add_argument(
        "--label_map",
        type=Path,
        required=False,
        default=Path("data/processed/label_map.json"),
        help="JSON mapping {code: idx}."
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        required=False,
        default=Path("data/processed"),
        help="Output directory for encoded parquet and Y_* file.",
    )
    parser.add_argument(
        "--reports_dir",
        type=Path,
        required=False,
        default=Path("reports"),
        help="Directory to save encoding_summary.txt"
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Store dense Y (*.npy) instead of CSR (*.npz).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_label_map(path: Path, logger=None) -> Dict[str, int]:
    """
    Load {ICD code: index} mapping.

    Parameters
    ----------
    path    :   Path
        Path to label_map.json
    
    Returns
    -------
    dict[str, int]
        Code-to-index mapping
    
    Raises
    ------
    ValueError
        If indices are not 0..K-1 with no duplicates.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        mapping = json.load(f)

    indices = list(mapping.values())
    k = len(indices)

    if len(set(indices)) != k:
        raise ValueError("[LABEL_MAP] Duplicate indices detected")
    
    if min(indices) != 0 or max(indices) != k - 1:
        raise ValueError(f"[LABEL_MAP] Indices not contiguous: min={min(indices)}, max={max(indices)}, expect 0..{k-1}")
    
    logger.info(f"[LABEL_MAP] Loaded {k:,} labels (min={min(indices)}, max={max(indices)})")
    return mapping

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def codes_to_indices(
        codes: Iterable[str],
        label_map: Dict[str, int]
) -> Tuple[List[int], List[str]]:
    """
    Map ICD codes to label indices and collect unknowns.

    Parameters
    ----------
    codes   : Iterable[str]
        Label codes in the row.
    label_map: dict[str, int]
        Global mapping.

    Returns
    -------
    (idx_list, unknowns) : (list[int], list[str])
        Indices for known codes (dedup not enforced here) and unknown codes.
    """
    known_indices: List[int] = []
    unknown_indices: List[str] = []

    for code in codes:
        norm = (str(code).strip().upper())
        if norm in label_map:
            known_indices.append(label_map[norm])
        else:
            unknown_indices.append(norm)
    
    return known_indices, unknown_indices

def build_csr_from_index_lists(
        rows_indices: List[List[int]],
        num_labels: int
) -> sparse.csr_matrix:
    """
    Build a CSR binary matrix from per-row index lists.

    Parameters
    ----------
    rows_indices : list[list[int]]
        For each row, a sorted deduplicated list of indices.
    num_labels  : int
        Total number of labels (K)
    
    Returns
    -------
    scipy.sparse.csr_matrix
        Shape(N, K), dtype=uint8
    
    Notes
    -----
    - Expects indices within [0, K-1]
    """
    N = len(rows_indices)
    lengths = [len(r) for r in rows_indices]
    indptr = np.zeros(N + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(lengths)

    count = int(indptr[-1])
    indices = np.fromiter((j for row in rows_indices for j in row), dtype=np.int64, count=count)
    data = np.ones_like(indices, dtype=np.uint8)

    Y = sparse.csr_matrix((data, indices, indptr), shape=(N, num_labels), dtype=np.uint8)
    assert Y.shape == (N, num_labels), f"Unexpected shape: {Y.shape}"
    assert Y.nnz == len(indices), "Mismatch: number of nonzero values"
    return Y

# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_kv(label: str, value: str, *, width: int = 22) -> str:
    """Align 'label: value' for readability"""
    return f"{label:<{width}}: {value}"

def fmt_int(x: int) -> str:
    """Format integer with thousands separator."""
    return f"{x:,}"

def fmt_float(x: float, *, places: int=6) -> str:
    """Format float with fixed decimal places."""
    return f"{x:.{places}f}"

def write_summary_report(
        path: Path,
        *,
        n_rows: int,
        n_labels: int,
        n_ones: int,
        density: float,
        avg_labels: float,
        unknown_rate: float,
        samples: List[str],
) -> None:
    """
    Write a human-readable summary report. (PHI-safe)

    Parameters
    ----------
    path    : Path
        Destination path for the text report.
    n_rows  : int
        Dataset size.
    n_labels: int
        label space size (K)
    n_ones  : int
        total positive entries.
    density : float
        nnz / (N*K)
    avg_labels  : float
        Average labels per row (nnz / N).
    unknown_rate: float
        Unknown codes / (known + unknown), in [0, 1]
    samples     : List[str]
        Up to 2-3 preview lines (IDs + codes/idx), no raw TEXT.

    Notes
    -----
    - Do not print PHI (no raw TEXT).
    - Fixed section header for grep-ability.
    - Numbers have thousand separators; float have fixed precision.
    """

    assert n_rows >= 0 and n_labels >= 0 and n_ones >= 0
    assert (n_rows == 0 or n_labels == 0) or (0.0 <= density <= 1.0)
    assert 0.0 <= unknown_rate <= 1.0

    out_dir = Path(path)
    ensure_parent_dir(out_dir)

    text = "\n".join([
        "Encode Labels - QA summary",
        "-" * 28,
        "",
        "[Dataset]",
        fmt_kv("Rows (N)",      fmt_int(n_rows)),
        fmt_kv("Labels (K)",    fmt_int(n_labels)),
        fmt_kv("Total Ones (nnz)", fmt_int(n_ones)),
        "",
        "[Sparsity]",
        fmt_kv("Density nnz/(N*K)", fmt_float(density, places=8)),
        fmt_kv("Avg labels/rows",   fmt_float(avg_labels, places=4)),
        fmt_kv("Unknown code rate", fmt_float(unknown_rate, places=6)),
        "",
        "[Samples]",
        *([ "(no samples)"] if not samples else [f"- {s}" for s in samples[:3]]),
        "",
        "[Meta]",
        fmt_kv("Generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        fmt_kv("Report Path", str(to_relpath(out_dir))),
    ]) + "\n"
    
    with out_dir.open("w", encoding="utf-8") as f:
        f.write(text)
    
    size = out_dir.stat().st_size
    if size == 0:
        raise IOError(f"Report write verification failed: {out_dir} is empty")

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    """Run multi-hot encoding pipeline."""

    args = parse_args()

    logger = setup_logger("encode", log_dir=Path("logs"))
    set_seed(args.seed)

    df = read_jsonl(args.jsonl_in, logger=logger)
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    missing = required - set(df.columns)
    assert not missing, f"[CHECK] missing columns: {missing}"
    assert df["SUBJECT_ID"].notna().all() and df["HADM_ID"].notna().all(), "[CHECK] NA IDs"
    assert df["LABELS"].map(lambda x: isinstance(x, list)).all(), "[CHECK] LABELS must be list[str]"

    label_map = load_label_map(args.label_map, logger=logger)
    k = len(label_map)
    logger.info(f"[LOAD] rows={len(df):,} labels(K)={k:,}")

    # 2) Map codes -> indices per row
    rows_idx: List[List[int]] = []
    unknown_all: List[str] = []
    for codes in df["LABELS"]:
        idx_list, unknowns = codes_to_indices(codes, label_map)
        idx_list = sorted(set(idx_list))
        rows_idx.append(idx_list)
        unknown_all.extend(unknowns)
    
    unknown_ctr = Counter(unknown_all)
    unk_path = args.reports_dir / "unknown_code_top.txt"
    ensure_parent_dir(unk_path)
    with open(unk_path, "w", encoding="utf-8") as f:
        for code, cnt in unknown_ctr.most_common(50):
            f.write(f"{code}\t{cnt}\n")
    logger.info(f"[REPORT] {to_relpath(unk_path)} (top unknown codes)")
    
    # 3) Build Y (CSR or dense)
    if not args.dense:
        Y = build_csr_from_index_lists(rows_idx, k)
        n_ones = int(Y.nnz)
        assert Y.shape == (len(df), k), f"Y shape mismatch: {Y.shape} vs {(len(df), k)}"
    else:
        Y = np.zeros((len(df), k), dtype=np.uint8)
        for i, idxs in enumerate(rows_idx):
            if idxs: Y[i, idxs] = 1
        n_ones = int(Y.sum())

    # 4) Save encoded table (LABELS_IDX) alongside text/ids
    out_parquet = args.out_dir / "notes_icd_encoded.parquet"
    df_enc = df[["SUBJECT_ID", "HADM_ID", "TEXT"]].copy()
    df_enc["LABELS_IDX"] = rows_idx

    schema = pa.schema([
        pa.field("SUBJECT_ID", pa.int64()),
        pa.field("HADM_ID"  ,  pa.int64()),
        pa.field("TEXT"     ,  pa.string()),
        pa.field("LABELS_IDX", pa.list_(pa.int32()), nullable=False),
    ])

    arr_subj    = pa.array(df_enc["SUBJECT_ID"].tolist(), type=pa.int64())
    arr_hadm    = pa.array(df_enc["HADM_ID"].tolist(), type=pa.int64())
    arr_text    = pa.array(df_enc["TEXT"].astype(str).tolist(), type=pa.string())
    arr_labels  = pa.array([[int(x) for x in xs] for xs in df_enc["LABELS_IDX"]], type=pa.list_(pa.int32()))

    table = pa.Table.from_arrays([arr_subj, arr_hadm, arr_text, arr_labels], schema=schema)
    ensure_parent_dir(out_parquet)
    pq.write_table(table, out_parquet, compression="zstd")

    assert all((not r) or (0 <= r[0] and r[-1] < k) for r in rows_idx), "[CHECK] index out of bounds"

    # 5) Save Y matrix
    if not args.dense:
        out_y = args.out_dir / "Y_multi_hot.npz"
        ensure_parent_dir(out_y)
        sparse.save_npz(out_y, Y)
    else:
        out_y = args.out_dir / "Y_dense.npy"
        ensure_parent_dir(out_y)
        np.save(out_y, Y, allow_pickle=False)
    
    # 6) Validate on 2-3 samples (LOG + report)
    n = len(df)
    density = (n_ones / (n * k)) if  (n > 0 and k > 0) else 0.0
    avg_labels = (n_ones / n) if n > 0 else 0.0
    known_count = sum(len(r) for r in rows_idx)
    denom = (known_count + len(unknown_all)) or 1
    unknown_rate = len(unknown_all) / denom

    idxs = np.linspace(0, max(0, n - 1), num=min(3, n), dtype=int)
    sample_lines: List[str] = []
    for i in idxs:
        hadm = int(df["HADM_ID"].iat[i])
        codes = df["LABELS"].iat[i]
        idxs_i = rows_idx[i]
        sample_lines.append(f"HADM={hadm} codes={codes} idx={idxs_i}")
    
    report_path = args.reports_dir / "encoding_summary.txt"
    ensure_parent_dir(report_path)
    logger.info(f"[DEBUG] report_path={report_path} exists={report_path.exists()} suffix={report_path.suffix!r}")

    write_summary_report(
        report_path,
        n_rows=n,
        n_labels=k,
        n_ones=n_ones,
        density=density,
        avg_labels=avg_labels,
        unknown_rate=unknown_rate,
        samples=sample_lines,
    )
    if unknown_rate > 0.05:
        logger.warning(f"[WARN] unknown_rate={unknown_rate:.4f} - consider larger K")
    logger.info(f"[SAVE] table: {to_relpath(out_parquet)}")
    logger.info(f"[SAVE] labels: {to_relpath(out_y)}")
    logger.info(f"[REPORT] {to_relpath(report_path)}")
    logger.info(f"[STATS] N={n:,} K={k:,} nnz={n_ones:,} density={density:.8f} avg/row={avg_labels:.4f} unknown_rate={unknown_rate:.6f}")
    logger.info("[DONE] Multi-hot encoding complete.")

if __name__ == "__main__":
    main()