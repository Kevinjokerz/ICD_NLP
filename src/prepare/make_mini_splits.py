"""
make_mini_split.py - Subject-Aware Mini-subset builder

Purpose
-------
Create smaller subject-aware splits from full data/splits for quick
smoke testing of TF-IDF/LR and neural baselines.

Conventions
-----------
- Reproducible: fixed SEED; logged manifest with subject counts & JSD parity.
- PHI-safe: no raw TEXT in logs or reports.
- Portability: relative paths only

Inputs
------
data/splits/{train, val, test}.jsonl
    Full subject-stratified splits.
Each row:
    SUBJECT_ID, HADM_ID, TEXT, LABELS

Outputs
-------
data/splits_mini/{train, val, test}.jsonl
reports/mini_manifest.json      # seed, subject counts, JSD vs. full
"""

from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import logging
from typing import Optional, Dict, Tuple
from src.utils import (
    read_jsonl, export_jsonl, setup_logger, set_seed, to_relpath, ensure_parent_dir
)
from src.prepare.split_data import label_freq_from_rows, _js_divergence_counts, coerce_labels_idx_columns

# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        splits_dir, out_dir, reports_dir, train_subj, val_subj, test_subj, seed
    """
    args = argparse.ArgumentParser("Subject-aware mini-split generator")
    args.add_argument("--splits_dir", type=str, default="data/splits", help="Source dir with {train, val, test}.jsonl")
    args.add_argument("--out_dir", type=str, default="data/splits_mini", help="Destination dir for mini splits")
    args.add_argument("--reports_dir", type=str, default="reports", help="Where to write mini_manifest.json")
    args.add_argument("--train_subj", type=int, default=150)
    args.add_argument("--val_subj", type=int, default=40)
    args.add_argument("--test_subj", type=int, default=40)
    args.add_argument("--seed", type=int, default=42)

    return args.parse_args()

# =============================================================================
# Helpers
# =============================================================================

def sample_subjects(df: pd.DataFrame, n: int, *, seed: int, logger: Optional[logging.Logger]) -> pd.DataFrame:
    """
    Select a subject-aware subset of rows.

    Parameters
    ----------
    df  : pd.DataFrame
        Columns: SUBJECT_ID, HADM_ID, TEXT, LABELS
    n   : int
        Number of unique subjects to keep
    seed: int
        RNG seed
    logger: logging.Logger, optional
    
    Returns
    -------
    pd.DataFrame
        Filtered rows containing only the chosen SUBJECT_ID
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # --- Validate schema ---
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    missing = required - set(df.columns)
    assert not missing, f"[CHECK] Missing required columns: {missing}"

    # --- Unique SUBJECT IDs ---
    subs = df["SUBJECT_ID"].astype("int64").unique()
    n_subs_total = len(subs)
    if n_subs_total == 0:
        raise ValueError("[CHECK] No SUBJECT_IDs found in input DataFrame")

    # --- Sample subset deterministically ---
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(subs)
    keep = set(subs[:min(n, subs.shape[0])])
    n_keep = len(keep)

    # --- Filter ---
    out = df[df["SUBJECT_ID"].isin(keep)].reset_index(drop=True)
    n_rows = len(out)

    # --- Checks ---
    assert out["SUBJECT_ID"].nunique() <= n_keep, "[CHECK] Subject count mismatch"
    assert out["LABELS"].map(len).gt(0).any(), "[CHECK] Empty labels after sampling"
    
    # --- Logging ---
    logger.info(
        f"[MINI] Selected {n_keep} / {n_subs_total:,} subjects"
        f"({(n_keep/n_subs_total)*100:.1f}%) -> {n_rows:,} rows"
    )
    logger.info(
        f"[MINI] Density preview: "
        f"mean_labels_per_row={out['LABELS'].map(len).mean():.2f}, "
        f"median={out['LABELS'].map(len).median():.1f}"
    )
    return out

def _freq_codes(df: pd.DataFrame, logger: Optional[logging.Logger]=None) -> Dict[str, int]:
    """
    Flatten LABELS to a frequency dict for JSD parity.

    Parameters
    ----------
    df  : pd.DataFrame
        Must contain columns `LABELS` (each element list[str])
    logger : logging.Logger, optional
        If provided, logs summary count for audit.

    Returns
    -------
    Dict[str, int]
        Mapping {normalized_code: frequency}
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    assert {"LABELS"}.issubset(df.columns), "[CHECK] Missing required column: 'LABELS'"

    counts: Dict[str, int] = {}
    norm = lambda c: str(c).strip().upper().replace(".", "")
    for codes in df["LABELS"]:
        if not isinstance(codes, (list, tuple, set)):
            continue
        for c in codes:
            c_norm = norm(c)
            if not c_norm:
                continue
            counts[c_norm] = counts.get(c_norm, 0) + 1
    
    n_unique = len(counts)
    n_total = sum(counts.values())
    assert len(counts) > 0, "[CHECK] No labels counted - possibly empty LABELS column"

    if logger:
        logger.info(f"[MINI] Counted {n_unique:,} unique label codes ({n_total:,} total occurences)")
    return counts


def _align_counts(a: Dict[str, int], b: Dict[str, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align two code -> count dicts to same index order.

    Parameters
    ----------
    a, b : Dict[str, int]
        Label -> count mappings (e.g., from _freq_codes())

    Returns
    -------
    (pa, pb)    : np.ndarray, np.ndarray
        Count vectors with the same ordering of labels
    """
    keys = sorted(set(a) | set(b))

    pa = np.array([a.get(k, 0) for k in keys], dtype=np.float32)
    pb = np.array([b.get(k, 0) for k in keys], dtype=np.float32)

    assert pa.shape == pb.shape, "[CHECK] Shape mismatch between count vectors"
    assert pa.size > 0, "[CHECK] Empty alignment: no overlapping or unique keys"

    return pa, pb

# =============================================================================
# Orchestration
# =============================================================================

def main() -> None:
    args = parse_args()
    logger = setup_logger("mini")
    set_seed(args.seed)

    in_dir = Path(args.splits_dir)
    out_dir = Path(args.out_dir)
    reports_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Load full splits ---
    tr = read_jsonl(in_dir / "train.jsonl")
    va = read_jsonl(in_dir / "val.jsonl")
    te = read_jsonl(in_dir / "test.jsonl")
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    for name, d in (("train", tr), ("val", va), ("test", te)):
        missing = required - set(d.columns)
        assert not missing, f"[CHECK] {name}: missing columns {sorted(missing)}"
    
    # --- Subject-aware sample ---
    tr_m = sample_subjects(tr, args.train_subj, seed=args.seed + 1, logger=logger)
    va_m = sample_subjects(va, args.val_subj, seed=args.seed + 2, logger=logger)
    te_m = sample_subjects(te, args.test_subj, seed=args.seed + 3, logger=logger)

    # --- Checks ---
    assert set(tr_m["SUBJECT_ID"]).isdisjoint(va_m["SUBJECT_ID"])
    assert set(tr_m["SUBJECT_ID"]).isdisjoint(te_m["SUBJECT_ID"])
    assert set(va_m["SUBJECT_ID"]).isdisjoint(te_m["SUBJECT_ID"])
    for name, d in (("train_mini", tr_m), ("val_mini", va_m), ("test_mini", te_m)):
        assert d["LABELS"].map(len).gt(0).any(), f"[CHECK] {name}: all LABELS empty"
    
    # --- Label parity ---
    g_tr, g_va, g_te = _freq_codes(tr), _freq_codes(va), _freq_codes(te)
    m_tr, m_va, m_te = _freq_codes(tr_m), _freq_codes(va_m), _freq_codes(te_m)
    pa, pb = _align_counts(g_tr, m_tr)
    js_tr = _js_divergence_counts(pa, pb)
    pa, pb = _align_counts(g_va, m_va)
    js_va = _js_divergence_counts(pa, pb)
    pa, pb = _align_counts(g_te, m_te)
    js_te = _js_divergence_counts(pa, pb)

    if max(js_tr, js_va, js_te) > 0.15:
        logger.warning("[MINI] JSD > 0.15 — mini distribution drifts from full; consider raising quotas or changing seed.")


    # --- Save mini splits ---
    export_jsonl(tr_m, out_dir / "train.jsonl")
    export_jsonl(va_m, out_dir / "val.jsonl")
    export_jsonl(te_m, out_dir / "test.jsonl")

    # --- Manifest ---
    mani = {
        "seed": int(args.seed),
        "in_dir": str(to_relpath(in_dir)),
        "out_dir": str(to_relpath(out_dir)),
        "subjects": {
            "train": int(tr_m["SUBJECT_ID"].nunique()),
            "val": int(va_m["SUBJECT_ID"].nunique()),
            "test": int(te_m["SUBJECT_ID"].nunique())
        },
        "rows": {
            "train": int(len(tr_m)),
            "val": int(len(va_m)),
            "test": int(len(te_m))
        },
        "jsd_vs_full": {
            "train": float(js_tr),
            "val": float(js_va),
            "test": float(js_te)
        }
    }

    ensure_parent_dir(reports_dir / "mini_manifest.json")
    with (reports_dir / "mini_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(mani, f, indent=2, ensure_ascii=False)
    logger.info(f"[MINI DONE] {mani['subjects']} rows={mani['rows']} JSD={mani['jsd_vs_full']}")

if __name__ == "__main__":
    main()