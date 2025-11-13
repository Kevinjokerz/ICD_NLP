"""
Build ICD9 Label Map

Purpose
-------
- Build a mapping {ICD9_CODE -> contiguous index} using TRAIN-first frequencies.
- Two selection modes:
    1) fixed_k      : use a user-provided K (Top-K)
    2) target_cov   : choose smallest K achieving target coverage on TRAIN.
- Emits a human readable summary with a coverage grid and top-10 preview.

Conventions
-----------
- Reproducibility (fixed seed), portability (relative path), structured logging.
- No PHI in logs (never print TEXT)

Inputs
------
- Preferred : data/splits/train.jsonl (SUBJECT_ID, HADM_ID, TEXT, LABELS:list[str])
- Fallback  : data/processed/label_frequency.csv (ICD9_CODE, COUNT[, PCT])

Outputs
-------
- data/processed/label_map.json         # JSON {"ICD9": idx}
- reports/label_map_summary.txt         # coverage grid + top-10 preview

Example
-------
# Fixed K
$ python -m src.prepare.build_label_map \
    --mode fixed_k --k 400 \
    --out_json data/processed/label_map.json \
    --summary reports/label_map_summary.txt \
    --strict_70

# Target coverage (auto-pick smallest K such that coverage >= 0.70)
$ python -m src.prepare.build_label_map \
    --mode target_cov --target_coverage 0.70 \
    --train_jsonl data/splits/train.jsonl \
    --out_json data/processed/label_map.json \
    --summary reports/label_map_summary.txt \
    --strict_70
"""

from __future__ import annotations
from pathlib import Path
import argparse
import json
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from collections import Counter
import logging

from src.utils import (
    setup_logger, load_config, read_jsonl,
    ensure_parent_dir, set_seed, to_relpath, read_table
)

COVERAGE_GRID: Tuple[int, ...] = (50, 100, 200, 400, 600, 800, 1000)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for building a Top-K label map.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments including freq_csv, k, out_json, summary, seed.

    Notes
    -----
    - Defaults match repo directory structure.
    - All paths are relative to project root.
    """
    parser = argparse.ArgumentParser("Build ICD9 Label Map")
    parser.add_argument("--mode", choices=["fixed_k", "target_cov"], required=True)

    # IO
    parser.add_argument("--train_jsonl", default="data/splits/train.jsonl")
    parser.add_argument("--freq_csv", type=str, default="data/processed/label_frequency.csv")
    parser.add_argument("--out_json", type=str, default="data/processed/label_map.json")
    parser.add_argument("--summary", type=str, default="reports/label_map_summary.txt")

    # Selection
    parser.add_argument("--k", type=int, default=None)                      # used by fixed k, or as lower bound in target_csv
    parser.add_argument("--target_coverage", type=float, default=None)      # used by target_cov
    parser.add_argument("--strict_70", action="store_true")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()

def _load_freq(train_jsonl: str, freq_csv: str, logger=None) -> pd.DataFrame:
    """
    Load TRAIN-first label frequencies; fallback to CSV.

    Parameters
    ----------
    train_jsonl : str
        Path to TRAIN jsonl with LABELS:list[str]
    freq_csv    : str
        Fallback label frequency table.
    
    Returns
    -------
    pd.DataFrame
        Columns: ["ICD9_CODE", "COUNT", "PCT"] sorted by COUNT desc.
    
    Notes
    -----
    - No TEXT is logged or returned
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    train = Path(train_jsonl)
    if train.exists():
        df = read_jsonl(train_jsonl)
        assert "LABELS" in df.columns, "LABELS missing in TRAIN jsonl"

        # Robust flatten in case of None rows
        flat = [str(c).strip().upper() for row in df["LABELS"] for c in (row or [])]
        ctr = Counter(flat)
        freq = pd.DataFrame({
            "ICD9_CODE" : list(ctr.keys()),
            "COUNT"     : list(ctr.values())
        })

        if freq.empty:
            logger.warning("[READ] TRAIN-first yielded 0 labels - check LABELS in train.jsonl")

        # Coerce & normalize
        freq["ICD9_CODE"] = freq["ICD9_CODE"].astype(str).str.upper()
        freq["COUNT"] = pd.to_numeric(freq["COUNT"], errors="coerce").fillna(0).astype(int)
        total = max(1, int(freq["COUNT"].sum()))
        freq["PCT"] = freq["COUNT"] / total

        # Always sort COUNT desc, code asc
        freq = freq.sort_values(["COUNT", "ICD9_CODE"], ascending=[False, True]).reset_index(drop=True)
        assert freq["COUNT"].is_monotonic_decreasing, "COUNT must be non-increasing after sort"

        logger.info(f"[READ] TRAIN-first -> {len(freq):,} labels from {to_relpath(train_jsonl)}")
        return freq
    
    # Fallback CSV
    csvp = Path(freq_csv)
    if not csvp.exists():
        raise FileNotFoundError(f"Neither TRAIN jsonl nor freq CSV found: {train} | {csvp}")
    
    freq = read_table(freq_csv)

    required = {"ICD9_CODE", "COUNT"}
    missing = [c for c in required if c not in freq.columns]
    assert not missing, f"Missing required columns: {missing}"

    # Coerce & normalize (ensure uppercases codes for consistency)
    freq["ICD9_CODE"] = freq["ICD9_CODE"].astype(str).str.upper()
    freq["COUNT"] = pd.to_numeric(freq["COUNT"], errors="coerce").fillna(0).astype(int)

    if "PCT" not in freq.columns:
        total = max(1, int(freq["COUNT"].sum()))
        freq["PCT"] = freq["COUNT"] / total

    freq = freq.sort_values(["COUNT", "ICD9_CODE"], ascending=[False, True]).reset_index(drop=True)
    assert freq["COUNT"].is_monotonic_decreasing, "COUNT must be non-increasing after sort"

    logger.info(f"[READ] CSV-fallback -> {len(freq):,} labels from {to_relpath(freq_csv)}")
    return freq

def select_codes(
        df_freq : pd.DataFrame, *,
        mode    : str,
        k       : Optional[int],
        target_coverage : Optional[float],
        logger  : Optional[logging.Logger] = None
) -> Tuple[List[str], int, float]:
    """
    Select codes either by fixed K or minimal K meeting target coverage.

    Parameters
    ----------
    df_freq : pd.DataFrame
        Sorted frequency table (COUNT desc).
    mode    : {"fixed_k", "target_cov"}
        Selection mode.
    k       : int | None
        Top-K (required for fixed_k; lower bound for target_cov if provided)
    target_coverage : float | None
        Coverage in [0, 1] used by target_cov mode.

    Returns
    -------
    (codes, k_eff, coverage) : (list[str], int, float)
        Selected codes, effective K, and TRAIN instance coverage at K.
    
    Notes
    -----
    - Uses cumulative sum to compute coverage curve effeciently.
    - Monotonicity: coverage is non-decreasing with respect to K.
    """
    assert df_freq["COUNT"].is_monotonic_decreasing, "COUNT must be non-increasing"

    # Robust coercions
    if not len(df_freq):
        return [], 0, 0.0
    mode = str(mode).lower()
    if k is not None:
        if not isinstance(k, int):
            raise TypeError("k must be int when provided")
        if k < 0:
            raise ValueError("k must be >= 0")
        
    counts = df_freq["COUNT"].to_numpy(dtype=np.int64)
    total = int(counts.sum()) if counts.size else 1
    csum = counts.cumsum()

    if mode == "fixed_k":
        assert k is not None and k > 0, "K must be > 0 for fixed_k"
        k_eff = min(int(k), len(df_freq))
        coverage = float(csum[k_eff - 1] / total) if k_eff > 0 else 0.0
    elif mode == "target_cov":
        assert target_coverage is not None and 0.0 <= target_coverage <= 1.0, "target_coverage must be in [0, 1]"
        threshold = target_coverage * total
        k_cov = int((csum >= threshold).argmax() + 1) if counts.size else 0
        k_eff = k_cov if k is None else max(k_cov, int(k))
        k_eff = min(k_eff, len(df_freq))
        coverage = float(csum[k_eff - 1] / total) if k_eff > 0 else 0.0
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    codes = df_freq["ICD9_CODE"].head(k_eff).tolist()
    assert len(codes) == k_eff, "codes must equal k_eff"

    if logger:
        logger.info(
            f"[SELECT] mode={mode:<10} k_eff={k_eff:<5d} "
            f"coverage={coverage:.4f} total_labels={len(df_freq):,}"
        )
    return codes, k_eff, coverage

def render_summary(
        df_freq : pd.DataFrame,
        k_eff   : int,
        coverage: float
) -> str:
    """
    Render a human-readable summary including coverage grid and top-10 preview.
    """
    counts = df_freq["COUNT"].to_numpy(dtype=np.int64)
    csum = counts.cumsum()
    total = int(counts.sum()) if counts.size else 1

    ks = sorted(set(COVERAGE_GRID + (k_eff,)))
    lines = []
    lines.append(f"Top-K label map (K={k_eff})")
    lines.append(f"Instance coverage : {coverage:.4f}")
    lines.append(f"Total labels      : {len(df_freq):,}")
    lines.append("")
    lines.append("Coverage grid (TRAIN):")
    for kk in ks:
        if counts.size == 0 or kk <= 0:
            cov = 0.0
        else:
            cov = float(csum[min(kk, len(csum)) - 1] / max(1 , total))
        lines.append(f"  K={kk:<5d}, coverage={cov:.4f}")
    lines.append("")
    cols = [c for c in ["ICD9_CODE", "COUNT", "PCT"] if c in df_freq.columns]
    lines.append("Top-10 preview:")
    if len(cols) and len(df_freq):
        lines.append(df_freq[cols].head(10).to_string(index=False))
    else:
        lines.append("(no labels to preview)")
    return "\n".join(lines)

def save_label_map(codes: List[str], path_json: str, logger: Optional[logging.Logger]=None) -> None:
    """
    Save mapping {ICD9_CODE -> idx} with contiguous indices 0..K-1.

    Parameters
    ----------
    codes   : list[str]
        Ordered list of ICD9 codes (Top-K selection).
    path_json: str
        Destination file path for JSON.
    logger  : logging.Logger, optional
        Active logger for structured reporting
    
    Notes
    -----
    - Ensure contiguous indices an round-trip JSON correctness.
    - Uses ensure_parent_dir() for portable relative writes.
    """
    mapping = {code: i for i, code in enumerate(codes)}
    assert sorted(mapping.values()) == list(range(len(codes))), "Indices not contiguous"

    ensure_parent_dir(path_json)
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, sort_keys=True, indent=2)
    
    reload = json.load(open(path_json, "r", encoding="utf-8"))
    assert reload == mapping, "JSON round-trip mismatch"

    if logger:
        logger.info(f"[SAVE] label_map written -> {to_relpath(path_json)}")

def _save_summary(text: str, path_txt: str, logger: Optional[logging.Logger] = None) -> None:
    """
    Save human-readable summary to text file.

    Parameters
    ----------
    text    : str
        Summary content.
    path_txt: str
        Destination file path.
    logger  : logging.Logger, optional
        Active logger for structured reporting

    Notes
    -----
    - Uses ensure_parent_dir() for safe relative path creation.
    - Append a trailing newline for POSIX compatibility.
    """
    ensure_parent_dir(path_txt)
    with open(path_txt, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    
    if logger:
        first_line = text.strip().splitlines()[0] if text else ""
        logger.info(f"[SAVE] summary written -> {path_txt}")
        logger.info(f"[HEAD] {first_line[:80]}")

def main() -> None:
    """
    CLI entry point for building the label map and coverage summary.

    Notes
    -----
    - Deterministic given fixed seed.
    - Logs relative paths only (no PHI leakage).
    - Raises ValueError if --strict_70 and coverage < 0.70
    """

    args = parse_args()
    logger = setup_logger("labelmap")
    set_seed(args.seed)
    logger.info(
        f"[INIT] mode={args.mode} | k={args.k} | target_cov={args.target_coverage} |"
        f"train={to_relpath(args.train_jsonl)} | csv={to_relpath(args.freq_csv)}"
    )

    df = _load_freq(args.train_jsonl, args.freq_csv, logger=logger)
    assert df["COUNT"].is_monotonic_decreasing, "COUNT must be non-increasing"

    codes, k_eff, cov = select_codes(
        df, mode=args.mode, k=args.k, 
        target_coverage=args.target_coverage, logger=logger
    )

    if args.strict_70 and cov < 0.70:
        raise ValueError(f"Coverage={cov:.4f} < 0.70 on TRAIN")
    elif cov < 0.70:
        logger.warning(f"[WARN] Coverage={cov:.4f} < 0.70 - consider larger K or different mode")
    
    save_label_map(codes, args.out_json, logger=logger)
    summary = render_summary(df, k_eff=k_eff, coverage=cov)
    _save_summary(summary, args.summary, logger=logger)

    logger.info(f"[DONE] Label_map.json + summary written -> K={k_eff} | coverage={cov:.4f}")
    print("OK")


if __name__ == "__main__":
    main()
    