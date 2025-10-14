"""
Prepare Data Pipeline for Automated ICD-9 Coding (MIMIC-III).

This module loads raw NOTEEVENTS + DIAGNOSES_ICD, filters discharge notes,
group ICD codes per HADM_ID, merges note-label pairs, and write a portable
CSV with JSON-encoded label lists. All paths are sanitized to be relative to
the project root to avoid absolute path leakage.

Conventions
-----------
- Reproducibility: fixed SEED; avoid nondeterministic transforms.
- Single-responsibility: helpers do one things; compose in main().
- Portability: CSV/JSON contain only relative paths; no drive letters.
- PHI safety: MIMIC is de-identified; do not add external identifiers.

Outputs
-------
- data/processed/notes_icd.csv
    Columns: SUBJECT_ID(int64), HADM_ID(int64), TEXT(str), LABELS(json list[str])

Author: Kevin Vo
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
from typing import Optional
import json
import pandas as pd
import numpy as np
from .utils import (
    load_config, setup_logger, set_seed,
    read_csv, save_csv_with_json_labels,
    save_table, export_jsonl, to_relpath,
    ensure_parent_dir
)

def parse_args() -> argparse.Namespace:
    """
    Parse CLI arguments for the end-to-end pipeline.

    Returns
    -------
    argparse.Namespace
            Parsed arguments:
            - note_path    : str (required)
            - diag_path    : str (required)
            - out_path     : str (required)
            - max_rows     : Optional[int]
            - min_note_len : int
            - join_policy  : {"longest", "concat"}
            - concat_sep   : str
            - encoding     : str
    """
    parser = argparse.ArgumentParser(description="Prepare ICD data.")

    # --- Required arguments ---
    parser.add_argument("--note_path", type=str, required=True, help="Path to NOTEEVENTS.csv.gz (relative to project root)")
    parser.add_argument("--diag_path", type=str, required=True, help="Path to DIAGNOSES_ICD.csv.gz (relative to project root)")
    parser.add_argument("--out_path", type=str, required=True, help="Output directory for processed files (will contain merged CSV + JSON)")

    # --- Optional arguments ---
    parser.add_argument("--max_rows", type=int, default=-1, help="Maximum number of rows to load; -1 reads the entire file")
    parser.add_argument("--sample_intersection", action="store_true", help="Sample on the intersection of HADM_ID between notes and ICD.")
    parser.add_argument("--max_hadm", type=int, help="Cap number of admissions from the intersection; 0 means no cap.")
    parser.add_argument("--min_note_len", type=int, help="Minimum character length for discharge notes to include.")
    parser.add_argument("--join_policy", type=str, choices=["longest", "concat"], help="Policy to merge multiple notes per admission")
    parser.add_argument("--concat_sep", type=str, help="Separator used when concatenating multiple notes (if join_policy='concat').")
    parser.add_argument("--encoding", type=str, help="Text file encoding (default: utf-8)")
    parser.add_argument("--seed", type=int, help="Random seed override for reproducibility.")
    return parser.parse_args()

# =============================================================================
# Note Processing (single-responsibility helpers)
# =============================================================================

def filter_to_discharge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to discharge summaries only if CATEGORY column exists
    
    Parameters
    ----------
    df  :   pd.DataFrame
        Input dataframe (typically NOTEEVENTS subset)
    
    Returns
    -------
    pd.DataFrame
        Filtered DataFrame. If no 'CATEGORY' column exists, return original.
    
    Notes
    -----
    - Case-insensitive match on "discharge summary".
    - Does *not* modify input in-place (returns a copy view).
    - Logs the number of remaining rows for reproducibility.
    """
    out = df.copy()

    if "CATEGORY" in df.columns:
        mask = out["CATEGORY"].astype(str).str.strip().str.lower().eq("discharge summary")
        before, after = len(out), int(mask.sum())
        out = out.loc[mask]
        logging.info(f"[NOTES] Discharge only: {after:,}/{before:,} rows kept")
    else:
        logging.warning("[NOTES] 'CATEGORY' columns missing - returning input unfiltered")
    
    return out

def select_note_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select only the core note columns: SUBJECT_ID, HADM_ID, TEXT.

    Parameters
    ----------
    df  :   pd.DataFrame
        Raw or filtered NOTEEVENTS DataFrame.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['SUBJECT_ID', 'HADM_ID', 'TEXT'].
        Rows with missing keys or text are dropped.
    
    Notes
    -----
    - Ensures portability and clean schema for downstream join.
    - Raises AssertionError if any required column is missing.
    - Does not modify input in-place (returns a new DataFrame).
    """
    required = ['SUBJECT_ID', 'HADM_ID', 'TEXT']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    out = df[required].copy()
    before = len(out)
    out = out.dropna(subset=required)
    after = len(out)

    logging.info(f"[NOTES] dropped {before - after:,} rows with missing IDs/TEXT")
    return out

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize discharge note text minimally:
    unify line breaks, collapse excessive whitespace, and strip edges.

    Parameters
    ----------
    df  :   pd.DataFrame
        DataFrame with a TEXT column.
    
    Returns
    -------
    pd.DataFrame
        Copy of df with normalized TEXT.

    Notes
    -----
    - MIMIC-III is already de-identified; this step performs *no* PHI removal.
    - Non-destructive: returns a copy; originally df is unmodified.
    - Normalization includes:
        * Replace Windows/Mac line breaks with Unix '\n'
        * Collapse multiple spaces/tabs to one
        * Strip leading/trailing whitespace
    """
    if 'TEXT' not in df.columns:
        raise KeyError("Missing required column: 'TEXT'")
    
    out = df.copy()
    before = len(out)

    # Normalize line breaks and collapse repeated whitespace
    out['TEXT'] = (
        out['TEXT']
        .astype(str)
        .str.replace(r"\r\n|\r", "\n", regex=True)
        .str.replace(r"[ \t]+", " ", regex=True)
        .str.strip()
    )

    after = out["TEXT"].notna().sum()
    logging.info(f"[NOTES] normalize_text: {before:,} rows processed; {after:,} non-empty TEXT")

    return out

def drop_short_notes(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """
    Drop notes with length < min_len characters.

    Parameters:
    df      :   pd.DataFrame
        Input DataFrame containing a TEXT column.
    min_len :    int
        Minimum number of characters required to keep a note.

    Returns
    -------
    pd.DataFrame
        Copy of df filtered by length of TEXT.
    
    Notes
    -----
    - Non destructive (returns a new DataFrame)
    - Assumes TEXT has been normalized to str upstream.
    """
    if 'TEXT' not in df.columns:
        raise KeyError("Missing required column: 'TEXT'")
    
    out = df.copy()
    before = len(out)

    # TEXT should already be string; keep a defensive cast
    mask = out['TEXT'].astype(str).str.len().ge(min_len)
    out = out.loc[mask]

    kept = len(out)
    logging.info(f"[NOTES] min_len={min_len} kept={kept:,} dropped={before - kept:,}")
    return out

def enforce_note_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure SUBJECT_ID/HADM_ID are int64 and TEXT is object/str.

    Parameters
    ----------
    df  :   pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Copy of df with enforced dtypes:
        - SUBJECT_ID: int64
        - HADM_ID   : int64
        - TEXT      : object (string-like)
    
    Notes
    -----
    - Upstream steps should have dropped NA in keys.
    - If any non-numeric IDs slip through, this will raise; that's desired.
    """
    required = ["SUBJECT_ID", "HADM_ID", "TEXT"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    out = df.copy()
    out['SUBJECT_ID'] = out['SUBJECT_ID'].astype("int64")
    out['HADM_ID']    = out['HADM_ID'].astype("int64")

    dtypes_str = {k: str(v) for k, v in out.dtypes.to_dict().items()}
    logging.info(f"[NOTES] enforce_note_dtypes: {dtypes_str}")
    return out

def load_and_filter_notes(note_path: str,
                          max_rows: Optional[int],
                          min_note_len: int,
                          encoding: str = "utf-8") -> pd.DataFrame:
    """
    Load NOTEEVENTS and return a clean subset for discharge summaries.

    Returns
    -------
    pd.DataFrame
        Columns: SUBJECT_ID(int64), HADM_ID(int64), TEXT(str)
    """
    nrows = None if (max_rows is None or max_rows < 0) else int(max_rows)
    use_cols = ["SUBJECT_ID", "HADM_ID", "TEXT", "CATEGORY"]
    try:
        df = read_csv(note_path, nrows=nrows, usecols=use_cols, encoding=encoding)
        logging.info(f"[NOTES] read with usecols={use_cols}")
    except ValueError:
        fallback_usecols = ["SUBJECT_ID", "HADM_ID", "TEXT"]
        df = read_csv(note_path, nrows=nrows, usecols=fallback_usecols, encoding=encoding)
        logging.warning(f"[NOTES] CATEGORY missing: use fallback usecols={fallback_usecols}")
    
    logging.info(f"[NOTES] raw shape={df.shape}")
    
    df = filter_to_discharge(df)
    if "CATEGORY" in df.columns and df["CATEGORY"].notna().any():
        assert len(df) > 0, "[NOTES] no discharge summaries after filter - check input file"
    logging.info(f"[NOTES] after dischare filter: {df.shape}")

    df = select_note_columns(df)
    df = normalize_text(df)
    df = drop_short_notes(df, min_len=min_note_len)
    df = enforce_note_dtypes(df)

    logging.info(f"[NOTES] final shape={df.shape}")
    return df.copy()

# =============================================================================
# ICD Processing (single-responsibility helpers)
# =============================================================================

def select_icd_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select the core ICD columns (HADM_ID, ICD9_CODE) and drop rows with missing values.

    Parameter
    ---------
    df  :   pd.DataFrame
        Raw DIAGNOSES_ICD dataframe.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['HADM_ID', 'ICD9_CODE'] and no missing entries

    Notes
    -----
    - Raise AssertionError if required columns are missing.
    - Non-destructive (return a copy).
    - Logs number of rows dropped for audit/repoducibility.
    """
    required = ["HADM_ID", "ICD9_CODE"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    out = df[required].copy()
    before = len(out)
    out = out.dropna(subset=required)
    after = len(out)

    logging.info(f"[ICD] dropped {before - after:,} rows with missing HADM_ID/ICD9_CODE")
    return out

def cast_icd_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast ICD9_CODE to string to preserve leading zeros and strip whitespace.
    
    Parameters
    ----------
    df  :   pd.DataFrame
        DataFrame containing the 'ICD9_CODE' column.

    Returns
    -------
    pd.DataFrame
        Copy of df with ICD9_CODE cast to string.
    
    Notes
    -----
    - Preserves leading zeros (e.g., '0419' -> '0419').
    - Non-destructive (return a copy).
    - Log summary stats for reproducibility.
    """
    if "ICD9_CODE" not in df.columns:
        raise KeyError("Missing required column: 'ICD9_CODE'")

    out = df.copy()
    before = len(out)
    out['ICD9_CODE'] = (
        out['ICD9_CODE']
        .astype(str)
        .str.strip()
        .str.upper()
    )

    after = out["ICD9_CODE"].notna().sum()
    logging.info(f"[ICD] cast_icd_to_str: {before:,} rows processed; {after:,} non-NA codes")
    return out


def enforce_icd_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure HADM_ID is int64.

    Parameter
    ---------
    df  :   pd.DataFrame
        DataFrame containing the 'HADM_ID' column.

    Return
    ------
    pd.DataFrame
        Copy of DataFrame of HADM_ID cast to int64

    Notes
    -----
    - Raises KeyError if 'HADM_ID' columns is missing.
    - Non-destructive (returns a copy).
    - Logs number of rows processed for reproducibility.
    """
    if "HADM_ID" not in df.columns:
        raise KeyError("Missing required column: 'HADM_ID'")
    
    out = df.copy()
    before = len(out)

    out["HADM_ID"] = out["HADM_ID"].astype("int64")

    after = out["HADM_ID"].notna().sum()
    logging.info(f"[ICD] enforce_icd_dtypes: {before:,} rows processed; {after:,} non-NA IDs")
    return out

def aggregate_labels_by_hadm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group ICD9 codes by HADM_ID into unique, sorted list.

    Parameter
    ---------
    df  :   pd.DataFrame
        Input DataFrame containing at least ['HADM_ID', 'ICD9_CODE']

    Returns
    -------
    pd.DataFrame
        Columns: ['HADM_ID', 'LABELS']
        - HADM_ID   : int64
        - LABELS : list[str]
    ~
    Notes
    -----
    - Removes duplicates and sorts codes alphabetically within each admission.
    - Non-destructive (returns a copy)
    - Logs group count for audit/reproducibility.
    """
    required = ['HADM_ID', 'ICD9_CODE']
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    before = len(df)
    grouped = (
        df.groupby("HADM_ID", sort=True)["ICD9_CODE"]
        .apply(lambda s: sorted(set(s.dropna().tolist())))
    )
    out = grouped.reset_index().rename(columns={"ICD9_CODE": "LABELS"})

    assert all(isinstance(x, list) for x in out["LABELS"]), "LABELS column must contain list objects"
    logging.info(f"[ICD] aggregate_labels_by_hadm: {before:,} rows -> {len(out):,} unique HADM_IDs")
    return out

def group_icd_by_hadm(diag_path: str,
                      max_rows: Optional[int],
                      encoding: str='utf-8') -> pd.DataFrame:
    """
    Load DIAGNOSES_ICD and produce HADM_ID -> LABELS mapping.

    Parameters
    ----------
    diag_path   :   str
        Path to DIAGNOSES_ICD.csv.gz
    max_rows    :   Optional[str]
        Maximum number of rows to read (for sampling/debugging)
    encoding    :   str, default='utf-8'
        File encoding

    Returns
    -------
    pd.DataFrame
        Columns: ['HADM_ID', 'LABELS']
        - HADM_ID : int64
        - LABELS  : list[str]
    
    Notes
    -----
    - Read only required columns (HADM_ID, ICD9_CODE) to save memory.
    - Preserves leading zeros in ICD9_CODE via dtypes='str'.
    - Raises AssertionError if grouping yields no admissions.
    """
    nrows = None if (max_rows is None or max_rows <0) else int(max_rows)
    usecols = ["HADM_ID", "ICD9_CODE"]
    df = read_csv(diag_path, nrows=nrows, usecols=usecols, encoding=encoding, dtype="str")
    logging.info(f"[ICD] raw-shape={df.shape}")

    df = select_icd_columns(df)
    df = cast_icd_to_str(df)
    df = enforce_icd_dtypes(df)
    df = aggregate_labels_by_hadm(df)

    assert len(df) > 0, "[ICD] no admission after grouping - check DIAGNOSES_ICD subset"
    logging.info(f"[ICD] grouped shape={df.shape}")
    return df.copy()

# =============================================================================
# Admission Selection (Intersection & Sampling)
# =============================================================================

def intersect_and_sample_by_hadm(
        notes_df: pd.DataFrame,
        icd_df: pd.DataFrame,
        *,
        max_hadm: int,
        seed: int,
        logger: Optional[logging.Logger] = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Intersect notes & ICD on HADM_ID, then optionally sample up to `max_hadm`
    admissions deterministically, and finally filter both frames.

    Parameters
    ----------
    notes_df : pd.DataFrame
        Columns ['SUBJECT_ID', 'HADM_ID', 'TEXT'].
    icd_df  : pd.DataFrame
        Columns ['HADM_ID', 'LABELS'].
    max_hadm: int
        If > 0, cap the number of admissions to sample; 0; keep all.
    seed    : int
        RNG seed for reproducibility
    logger  : Optional[logging.Logger]
        If provided, logs count before/after for audit
    
    Returns
    -------
    (pd.DataFrame, pd.DataFrame)
        Filtered (notes_df, icd_df) restricted to the same HADM_ID set
    
    Notes
    -----
    - Solves top-N row mismatch by sampling on admissions, not rows.
    - Does not coerce dtypes; dtypes enforcement happens downstream.
    """

    # Preconditions
    if "HADM_ID" not in notes_df.columns:
        raise KeyError("Missing required column 'HADM_ID' in notes_df.")
    if "HADM_ID" not in icd_df.columns:
        raise KeyError("Missing required column 'HADM_ID' in icd_columns.")
    
    # 1) Intersection (IDs only)
    hadm_notes = pd.Index(notes_df["HADM_ID"].unique())
    hadm_icd = pd.Index(icd_df["HADM_ID"].unique())
    inter = hadm_notes.intersection(hadm_icd).sort_values()

    if inter.empty:
        raise ValueError("No intersecting admissions found.")
    
    # 2) Deterministic sampling (IDs only)
    if max_hadm > 0 and len(inter) > max_hadm:
        rng = np.random.RandomState(seed)
        chosen_ids = pd.Index(rng.choice(inter.values, size=max_hadm, replace=False))
    else:
        chosen_ids = inter
    
    # 3) Filter both frames by chosen IDs
    chosen = set(chosen_ids.tolist())
    notes_f = notes_df[notes_df["HADM_ID"].isin(chosen)]
    icd_f = icd_df[icd_df["HADM_ID"].isin(chosen)]

    if logger:
        logger.info(f"[SAMPLE] inter_hadm={len(inter):,} -> kept={len(chosen_ids):,} hadm"
                    f"| notes_rows={len(notes_f):,} icd_rows={len(icd_f):,}")
    
    assert notes_f["HADM_ID"].nunique() == len(chosen_ids)
    assert icd_f["HADM_ID"].nunique() == len(chosen_ids)

    return notes_f, icd_f
    

# =============================================================================
# Merge Policies
# =============================================================================

def dedupe_notes_keep_longest(df: pd.DataFrame) -> pd.DataFrame:
    """
    For admission with multiple notes, keep the longest TEXT.

    Parameters
    ----------
    df  :   pd.DataFrame
        Must contain ["SUBJECT_ID", "HADM_ID", "TEXT"] columns.

    Returns
    -------
    pd.DataFrame
        One row per HADM_ID (the longest note kept).
    
    Notes
    -----
    - Non-destructive (returns a copy).
    - Useful when join_policy='longest' in merging state.
    - Logs number of admissions deduplicated.
    """
    required = ["SUBJECT_ID", "HADM_ID", "TEXT"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    out = df.copy()
    before = len(out)

    out["_len"] = out["TEXT"].astype(str).str.len()
    out = out.sort_values("_len", ascending=False)
    out = out.drop_duplicates(subset=["HADM_ID"], keep="first").drop(columns=["_len"])

    after = len(out)
    logging.info(f"[NOTES] dedupe_notes_keep_longest: {before:,} -> {after:,} unique admissions.")
    return out

def dedupe_notes_concat(df: pd.DataFrame, sep: str = "\n\n") -> pd.DataFrame:
    """
    For admissions with multiple notes, concatenate TEXT strings.

    Parameters
    ----------
    df  :   pd.DataFrame
        Must contain ['SUBJECT_ID', 'HADM_ID', 'TEXT'] columns.
    sep :   str
        Separator between concatenated notes.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with one row per (SUBJECT_ID, HADM_ID),
        TEXT being the concatenation of aall notes for that admission.

    Notes
    -----
    - Non-destructive (returns a new DataFrame).
    - Stable order: preserves chronological order if input is sorted.
    - Logs number of admission deduplicated.
    """
    required = ["SUBJECT_ID", "HADM_ID", "TEXT"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    out = df.copy()
    before = len(out)

    out["TEXT"] = out["TEXT"].astype(str)
    out = (
        out.sort_values(["HADM_ID"], inplace=False)
        .groupby(["SUBJECT_ID", "HADM_ID"], sort=False)["TEXT"]
        .agg(lambda s: sep.join(list(s)))
        .reset_index(name="TEXT")
    )
    after = len(out)

    logging.info(f"[NOTES] dedupe_notes_concat: {before:,} -> {after:,} unique admissions")
    return out

def merge_notes_labels(
        notes_df: pd.DataFrame,
        icd_df  : pd.DataFrame,
        join_policy: str="longest",
        concat_sep: str="\n\n",
) -> pd.DataFrame:
    """
    Merge notes with ICD labels on HADM_ID.

    Parameters
    ----------
    notes_df    :   pd.DataFrame
    icd_df      :   pd.DataFrame
    join_policy :   {"longest", "concat"}
    concat_sep  :   str

    Returns
    -------
    pd.DataFrame
        Columns: ['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']
        - SUBJECT_ID: int64
        - HADM_ID   : int64
        - TEXT      : str
        - LABELS    : list[str]
    """
    policy = join_policy.strip().lower()
    assert policy in {"longest", "concat"}, f"Unknown join_policy={policy!r}"

    # Logging pre-stats
    n_notes_before = len(notes_df)
    hadm_notes_before = notes_df["HADM_ID"].nunique()
    hadm_icd = icd_df["HADM_ID"].nunique()

    # Apply dedupe policy
    if policy == "longest":
        notes_df = dedupe_notes_keep_longest(notes_df)
        assert notes_df["HADM_ID"].is_unique, "[NOTES] HADM_ID not unique after 'longest'"
    else:
        notes_df = dedupe_notes_concat(notes_df, sep=concat_sep)

    assert len(notes_df) > 0, "[MERGE] no notes left after dedupe"
    
    # Merge
    merged = notes_df.merge(icd_df, on="HADM_ID", how="inner")

    # Enforce dtypes on note columns (SUBJECT_ID/HADM_ID/TEXT)
    merged = enforce_note_dtypes(merged)

    # Invariants & validation
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    assert required.issubset(merged.columns), f"Missing columns: {required - set(merged.columns)}"
    assert len(merged) > 0, "[MERGE] empty result - check input/policy"
    assert merged["LABELS"].map(lambda x: isinstance(x, list)).all(), "[MERGE] LABELS must be list[str]"

    # Reproducible order
    merged = merged.sort_values(["HADM_ID", "SUBJECT_ID"]).reset_index(drop=True)

    # Logging post-stats
    logging.info(f"[MERGE] notes={n_notes_before:,} hadm_notes={hadm_notes_before:,} "
                 f"hadm_icd={hadm_icd} -> merged_rows={len(merged):,} "
                 f"merged_hadm={merged['HADM_ID'].nunique():,}")

    return merged[["SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"]]

# =============================================================================
# Reporting / Validation
# =============================================================================

def label_frequency(merged: pd.DataFrame, topk: int=10) -> pd.DataFrame:
    """
    Compute ICD9 label frequency to guide top-K selection.

    Parameters
    ----------
    merged  :   pd.DataFrame
        Merged dataFrame containing a "LABELS" column of list[str]
    topk    :   int, default=10
        Number of most frequent codes to returns

    Returns
    -------
    pd.DataFrame
        Columns: ['ICD9_CODE', 'COUNT', 'PCT']

    Notes
    -----
    - Non-destructive (returns a new DataFrame)
    - Safe even if LABELS column is empty or missing.
    - Logs summary statistics for audit/reproducibility.
    """
    if "LABELS" not in merged.columns:
        raise KeyError("Missing required column: 'LABELS'")
    
    if merged.empty:
        logging.warning("[MERGE] label_frequency called on empty DataFrame")
        return pd.DataFrame(columns=["ICD9_CODE", "COUNT", "PCT"])
    
    # Explode list -> individual rows
    x = merged[["LABELS"]].explode("LABELS").dropna(subset=["LABELS"])
    vc = x["LABELS"].value_counts()

    out = (
        vc.rename_axis("ICD9_CODE")
        .reset_index(name="COUNT")
        .assign(PCT=lambda d: d["COUNT"] / d["COUNT"].sum())
    )
    logging.info(f"[REPORT] label_frequency: compute {len(out):,} unique labels; showing top-{topk}")
    return out.head(topk)

def basic_report(merged: pd.DataFrame) -> None:
    """
    Minimal text-based summary report (no plots).

    Prints
    ------
    - Unique SUBJECT_ID and HADM_ID counts
    - Text length statistics (chars, tokens)
    - Labels-per-admission statistics
    - Top-10 ICD9 label frequency

    Notes
    -----
    - Designed for quick sanity checks after merge.
    - Non-destructive; prints only.
    - Logs summary header for reproducibility.
    """
    if merged.empty:
        logging.warning("[REPORT] basic_report called on an empty DataFrame.")
        return
    
    n_subj = merged["SUBJECT_ID"].nunique()
    n_hadm = merged["HADM_ID"].nunique()
    lens = merged["TEXT"].astype(str).str.len()
    tok = merged["TEXT"].astype(str).str.split().map(len)
    lab_counts = merged["LABELS"].map(len)

    logging.info(f"[REPORT] Generating summary for {len(merged):,} rows")
    print("=" * 60)
    print(f"[REPORT] subjects={n_subj:,} hadm={n_hadm:,}")
    print(f"[REPORT] text chars: mean={lens.mean():.1f}, median={lens.median():.1f}")
    print(f"[REPORT] tokens: mean={tok.mean():.1f}, median={tok.median():.1f}")
    print(f"[REPORT] labels/admission min={lab_counts.min()}, mean={lab_counts.mean():.2f}, max={lab_counts.max()}")
    print("-" * 60)
    print(label_frequency(merged, topk=10))
    print("=" * 60)

def quick_invariants(merged: pd.DataFrame, expect_unique_hadm: bool=True) -> None:
    """
    Invariant checks after merge; raises AssertionError on failure.

    Parameters
    ----------
    merged  :   pd.DataFrame
        Final merged DataFrame containing SUBJECT_ID, HADM_ID, TEXT, LABELS.
    expect_unique_hadm  :   bool
        If True, requires unique HADM_ID (when using 'longest' policy)

    Raises
    ------
    AssertionError
        If any invariant fails
    
    Notes
    -----
    - Ensures schema and non-empty TEXT.
    - Validate LABELS column JSON-serializability.
    - Logs success for reproducibility.
    """
    required = {"SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"}
    missing = [c for c in required if c not in merged.columns]
    assert not missing, f"[CHECK] Missing required columns: {missing}"

    if merged.empty:
        raise AssertionError("[CHECK] merged DataFrame is empty")

    if expect_unique_hadm:
        assert merged["HADM_ID"].is_unique, "[CHECK] HADM_ID not unique under 'longest' policy"

    assert merged["TEXT"].astype(str).str.len().ge(1).all(), "[CHECK] Empty TEXT entries detected"

    try:
        _ = [json.dumps(lbls) for lbls in merged["LABELS"].head(5)]
    except Exception as e:
        raise AssertionError(f"[CHECK] LABELS not JSON-serializable: {e}")
    
    logging.info(f"[CHECK] quick_invariants passed ({len(merged):,} rows, unique HADM_ID={merged['HADM_ID'].is_unique})")

# =============================================================================
# Audit / Artifacts
# =============================================================================



# =============================================================================
# Main Orchestration
# =============================================================================

def main() -> None:
    """
    Orchestrate the end-to-end preparation pipeline:

    1) Load config (CLI + .env via utils.config)
    2) Set seed
    3) Load & Filter notes
    4) Load & group ICD labels
    5) Merge notes & labels (policy-based)
    6) Validate + basic report
    7) Save CSV with JSON-encoded LABELS (path-safe)
    """

    # --- Args & Config ---
    args = parse_args()
    cfg = load_config(args)
    logger = setup_logger(name="prep", log_dir=cfg.log_dir)
    set_seed(cfg.seed)

    logger.info(
        f"[INIT] seed={cfg.seed} enc={cfg.encoding} join={cfg.join_policy}"
        f"min_note_len={cfg.min_note_len} max_rows={args.max_rows}"
    )

    notes = load_and_filter_notes(
        note_path=args.note_path,
        max_rows=args.max_rows,
        min_note_len=cfg.min_note_len,
        encoding=cfg.encoding,
    )

    icds = group_icd_by_hadm(
        diag_path=args.diag_path,
        max_rows=args.max_rows,
        encoding=cfg.encoding,
    )

    # --- Admission Selection (optional) ---
    if cfg.sample_intersection:
        notes, icds = intersect_and_sample_by_hadm(
            notes_df=notes,
            icd_df=icds,
            max_hadm=cfg.max_hadm,
            seed=cfg.seed,
            logger=logger,
        )
    merged = merge_notes_labels(
        notes_df=notes,
        icd_df=icds,
        join_policy=cfg.join_policy,
        concat_sep=cfg.concat_sep,
    )

    quick_invariants(merged, expect_unique_hadm=(cfg.join_policy == "longest"))
    assert merged["LABELS"].map(len).ge(1).all(), "[CHECK] empty label list exist"
    basic_report(merged)

    out_p = Path(args.out_path)
    ensure_parent_dir(out_p / "._touch")

    save_table(merged, out_p, fmt="csv")
    save_table(merged, out_p, fmt="parquet")
    export_jsonl(merged, out_p / "notes_icd.jsonl")

    lf = label_frequency(merged, topk=1000)
    lf_path = out_p / "label_frequency.csv"
    lf.to_csv(lf_path, index=False, encoding="utf-8")
    logger.info(f"[SAVE] label_frequency -> {to_relpath(lf_path)}")

    logger.info(f"[DONE] shape={merged.shape} dtypes={ {k:str(v) for k, v in merged.dtypes.items()} }")
    logger.info(f"[SAVE] merged -> {to_relpath(out_p)}")

if __name__ == "__main__":
    main()