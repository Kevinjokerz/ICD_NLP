"""
I/O Utilities

Purpose
-------
Portable, PHI-safe helpers to read/write tables (CSV/Parquet) and export JSONL
for training. Adds guard rails against absolute-path leakage and normalizes
LABELS serialization.

Convention
----------
- Reproducibility: deterministic; minimal side effects; fixed logging format.
- Portability: use project-relative paths in logs; forbid absolute path leaks.
- Single responsibility: every function perform one single task
- Logging: INFO lines on READ/SAVE with row/col counts via to rel_path().

Inputs
------
- CSV/Parquet files of discharge notes with ICD labels.
- Expected schema (core):
    * SUBJECT_ID    : int64
    * HADM_ID       : int64
    * TEXT          : str
    * LABELS        : list[str] (Parquet) | JSON string of list[str] (CSV)

Outputs
-------
- CSV with JSON-encoded LABELS per row
- Parquet with LABELS as list[str]
- JSONL for training:
    {"text": str, "labels": list[str], "hadm_id": int, "subject_id": int}

Example
-------
>>> from pathlib import Path
>>> import pandas as pd
>>> df = pd.DataFrame({
...     "SUBJECT_ID":[1], "HADM_ID":[100], "TEXT":["note"],
...     "LABELS"[["4019", "25000"]]
...})
>>> # Save parquet then read back
>>> save_table(df, Path("data/processed"), fmt="parquet")
>>> df2 = read_table("data/processed/notes_icd.parquet")
>>> assert isinstance(df2.iloc[0, "LABELS"], list)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Union, Any
import logging, json
import numpy as np
import pandas as pd
from .paths import get_project_root, looks_absolute, to_relpath, safe_join_rel
from .sanitize import sanitize_path_columns, assert_no_abs_in_df, assert_no_abs_in_json_series, sanitize_json_like
import math
import pyarrow as pa
import pyarrow.parquet as pq
PARQUET_KW = {
    "engine": "pyarrow",
    "compression": "zstd",
    "index": False
}

def ensure_parent_dir(path: Union[str, Path]) -> None:
    """
    Ensure that the parent directory of a file path exists (create if missing).
    
    Parameters
    ----------
    path    :   str | Path
        Target file path. Its parent directory will be created if it doesn't exist.
    
    Returns
    -------
    None

    Examples
    --------
    >>> ensure_parent_dir("data/processed/notes_icd.csv")
    """
    if path is None:
        raise ValueError("Path cannot be None.")
    
    p = Path(path)
    parent = p.parent if p.suffix or p.name else p

    try:
        parent.mkdir(parents=True, exist_ok=True)
        logging.debug(f"[MKDIR] ensured parent dir: {parent}")
    except Exception as e:
        raise OSError(f"Failed to create parent directory for {path!r}: {e}") from e

def _normalize_nrows(nrows: Optional[int]) -> Optional[int]:
    """
    Normalize 'nrows' to None or int >= 0. Negative -> None

    Parameters
    ----------
    nrows : int | None

    Returns
    -------
    Optional[int]
        None means "read all".
    
    Examples
    --------
    >>> assert _normalize_nrows(None) is None
    >>> assert _normalize_nrows(-1) is None
    >>> assert _normalize_nrows(5) == 5
    """
    if nrows is None:
        return None
    n = int(nrows)
    return None if n < 0 else n


def read_csv(path: Union[str, Path], 
             nrows: Optional[int] = None, 
             usecols: Optional[list[str]] = None, 
             encoding: str = "utf-8", 
             dtype: Optional[Union[str, dict]] = None,
             on_bad_lines: str = "warn"
) -> pd.DataFrame:
    """
    Thin wrapper for pd.read_csv with consistent defaults.
    
    Parameters
    ----------
    path    :   str | Path
        Path to CSV file.
    nrows   :   Optional[int]
        Number of rows to read (for sampling/debugging)
    usecols :   Optional[list[str]]
        Columns to read (subset)
    encoding:   str
        File encoding (default='utf-8')
    dtype   :   Optional[str]
        Force dtype for all columns (e.g., 'str' for text-heavy data)
    
    Returns
    -------
    pd.DataFrame
        loaded DataFrame
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    
    n = _normalize_nrows(nrows)
    df = pd.read_csv(
        p, nrows=n, usecols=usecols or None, 
        encoding=encoding, encoding_errors="replace", 
        low_memory=False, dtype=dtype, 
        on_bad_lines=("error" if on_bad_lines == "error" else "skip")
    )
    logging.info(f"[READ] {to_relpath(p)} -> {len(df)} rows, {df.shape[1]} cols")
    assert isinstance(df, pd.DataFrame)
    return df

def _label_to_json(x: Any) -> str:
    """
    Robust convert a LABELS cell to a JSON string representing list[str].

    Accept
    - list/tuple/np.ndarray/pd.Series   -> list[str] then dumps
    - JSON-looking string "[...]"       -> pass through
    - None / NA scalar                  -> "[]"
    - other scalars (e.g., "4019)       -> wrap as single-element list
    """
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        seq = x.tolist() if hasattr(x, "tolist") else list(x)
        labels = [str(v) for v in seq]
        return json.dumps(labels, ensure_ascii=False)

    if x is None:
        return "[]"
    
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                _ = json.loads(s)
                return s
            except Exception:
                return json.dumps([s], ensure_ascii=False)
        return json.dumps([s], ensure_ascii=False)
    
    try:
        if pd.isna(x):
            return "[]"
    except Exception:
        return "[]"
    
    return json.dumps([str(x)], ensure_ascii=False)

def save_csv_with_json_labels(df: pd.DataFrame, outpath: Union[str, Path]) -> None:
    """
    Save merged DataFrame to CSV; JSON-encode LABELS and enforce path safety.
    
    Parameters
    ----------
    df  :   pd.DataFrame
        Data to save. Must contains a "LABELS" columns (JSON-like per row)
    out_path : str | Path
        Destination CSV path (prefer relative to project root).
    """
    root = get_project_root().resolve()

    # --- Validate required column ---
    if "LABELS" not in df.columns:
        raise KeyError("Missing required column: 'LABELS'")
    
    # --- Sanitize path-like columns + assert no absolute ---
    safe_df = sanitize_path_columns(df, root)
    assert_no_abs_in_df(safe_df)

    # --- Normalize LABELS: sanitize JSON-like, then dumps
    tmp = safe_df.copy()
    
    tmp["LABELS"] = tmp["LABELS"].map(_label_to_json)

    # --- Ensure destination under project root (no absolute leak) ---
    outpath = Path(outpath)
    if outpath.suffix == "":
        outpath = outpath / "notes_icd.csv"
    if looks_absolute(str(outpath)):
        rel = to_relpath(outpath, root=root)
        dest = safe_join_rel(rel, root=root)
    else:
        dest = safe_join_rel(outpath, root=root)
    
    # --- Create parent dir & write csv ---
    ensure_parent_dir(dest)
    tmp.to_csv(dest, index=False, encoding="utf-8")

    # --- Post-save sanity check ---
    assert_no_abs_in_json_series(tmp["LABELS"])

    # --- Log friendly path ---
    logging.info(f"[SAVE] {to_relpath(dest, root)} rows={len(tmp):,} cols:{tmp.shape[1]}")

def _decide_outfile(
    base: Union[str, Path],
    fmt: str,
    default_stem: str = "notes_icd",
) -> Path:
    """
    Decide the output file path from a directory or an explicit file path.

    Parameters
    ----------
    base    : Union[str, Path]
    fmt     : str
    default_stem    : str, default="notes_icd"

    Returns
    -------
    Path

    Notes
    -----
    - If 'base' has a suffix, return as-is.
    - Else, append 'default_stem' with proper extension based on 'fmt'.
    """
    p = Path(base)
    fmt_l = fmt.strip().lower()
    if fmt_l not in {'csv', 'parquet'}:
        raise ValueError(f"Unknown format: {fmt!r}")
    if p.suffix:
        return p
    ext = ".csv" if fmt_l.lower() == "csv" else ".parquet"
    dest = p / f"{default_stem}{ext}"
    logging.debug(f"[PATH] decided output: {dest}")
    return dest

def _labels_to_list_any(v: Any) -> list[str]:
    """
    Coerce a heterogeneous LABELS cell into list[str].

    Parameters
    ----------
    v   : Any
        Accepts list/tuple/set/np.ndarray/Series/str/scalar/None/NaN.
    
    Returns
    -------
    list[str]
        Normalized list of strings.
    
    Notes
    -----
    - str: try json.loads(s) -> list; if fail, wrap as [s]
    - None/NaN: []
    - sequence-like: cast to list of str.
    - other scalars: [str(v)]
    """
    if isinstance(v, str):
        try:
            return [str(x) for x in json.loads(v)]
        except Exception:
            return [v]
    if v is None:
        return []
    
    try:
        if isinstance(v, float) and math.isnan(v):
            return []
    except Exception:
        pass

    if isinstance(v, (list, tuple, set, np.ndarray, pd.Series)):
        seq = v.tolist() if hasattr(v, "tolist") else list(v)
        return [str(x) for x in seq]
    try:
        if pd.isna(v):
            return []
    except Exception:
        pass
    
    return [str(v)]

        

def save_table(
        df  :   pd.DataFrame,
        outpath: Union[str, Path],
        *,
        fmt :   str,
) -> None:
    """
    Save DataFrame as 'csv' or 'parquet' with LABELS handling.

    Parameters
    ----------
    df  : pd.DataFrame
        Must contain SUBJECT_ID, HADM_ID, TEXT, LABELS.
        LABELS list[str] (parquet) | JSON string (csv)
    outpath: str | Path
        Directory or explicit file path
    fmt : {"csv", "parquet"}
        Output format.
    
    Returns
    -------
    None

    Examples
    --------
    >>> save_table(df, "data/processed", fmt="csv")
    >>> save_table(df, "data/processed", fmt="parquet")
    """
    fmt_l = fmt.strip().lower()
    if fmt_l not in {"csv", "parquet"}:
        raise ValueError(f"fmt must be 'csv' or 'parquet', got: {fmt!r}")
    
    dest = _decide_outfile(outpath, fmt_l, default_stem="notes_icd")
    ensure_parent_dir(dest)

    if fmt_l == "csv":
        assert "LABELS" in df.columns, "Missing required column: 'LABELS'"
        return save_csv_with_json_labels(df, dest)
    elif fmt_l == "parquet":
        # --- Normalize entire LABELS column ---
        df = df.assign(LABELS=df["LABELS"].map(_labels_to_list_any))
        df = df.assign(LABELS=df["LABELS"].map(lambda xs: [] if xs is None else [str(x) for x in xs]))
        assert df["LABELS"].map(lambda x: isinstance(x, list)).all(), "LABELS must be list[str] befor parquet."

        schema = pa.schema([
            pa.field("SUBJECT_ID", pa.int64()),
            pa.field("HADM_ID"   , pa.int64()),
            pa.field("TEXT"      , pa.string()),
            pa.field("LABELS"    , pa.list_(pa.string()), nullable=False),
        ])

        # Convert pandas -> Arrow arrays explicitly to avoid mixed types
        arr_subj = pa.array(df["SUBJECT_ID"].tolist(), type=pa.int64())
        arr_hadm = pa.array(df["HADM_ID"].tolist(), type=pa.int64())
        arr_text = pa.array(df["TEXT"].astype(str).tolist(), type=pa.string())
        arr_labels = pa.array(df["LABELS"].tolist(), type=pa.list_(pa.string()))

        table = pa.Table.from_arrays(
            [arr_subj, arr_hadm, arr_text, arr_labels],
            schema=schema
        )
        
        pq.write_table(table, dest, compression=PARQUET_KW["compression"])
        logging.info(f"[SAVE] {to_relpath(dest)} rows={len(df):,} cols:{df.shape[1]}")

        tbl = pq.read_table(dest)
        labels_py = tbl.column("LABELS").to_pylist()
        assert all(isinstance(x, list) for x in labels_py), f"Round-trip: LABELS not list after Arrow read (n_bad={sum(not isinstance(x, list) for x in labels_py)})"
    
def read_table(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a table by suffix (.parquet or .csv).

    Parameters
    ----------
    path : str | Path
        file path with suffix
    
    Returns
    -------
    pd.DataFrame
        For parquet: LABELS preserved as list[str]
        for csv: LABELS read as string; caller may json.loads later.
    
    Raises
    ------
    FileNotFoundError
        If `path` does not exist.
    ValueError
        If suffix unsupported
    
    Examples
    --------
    >>> df = read_table("data/processed/note_icd.parquet")
    """
    p = Path(path)
    assert p.exists(), f"File not found: {p}"
    suf = p.suffix.lower()
    if suf == ".parquet":
        tbl = pq.read_table(p)
        cols = {name: tbl.column(name).to_pylist() for name in tbl.schema.names}
        df = pd.DataFrame(cols)
        df["SUBJECT_ID"] = pd.Series(df["SUBJECT_ID"], dtype="int64")
        df["HADM_ID"] = pd.Series(df["HADM_ID"], dtype="int64")
        assert df["LABELS"].map(lambda x: isinstance(x, list)).all()
        return df
    elif suf == ".csv":
        return pd.read_csv(p, low_memory=False)
    else:
        raise ValueError(f"Unsupported format: {suf}")
    
def export_jsonl(df: pd.DataFrame, jsonl_path: Union[str, Path]) -> None:
    """
    Write JSONL for training
    {"text": str, "labels": list[str], "hadm_id": int, "subject_id": int}

    Parameters
    ----------
    df  : pd.DataFrame
        requires SUBJECT_ID, HADM_ID, TEXT, LABELS; no NaN in TEXT/IDs.
        LABELS may be list[str] or JSON string; will be coerced to list[str].
    jsonl_path: str | Path
        Destination file (project-relative preferred).

    Returns
    -------
    None

    Notes
    -----
    - UTF-8 with ensure_ascii=False; one record per line.
    - Forbids NaN in JSON payload; set allow_nan=False.

    Examples
    --------
    >>> export_jsonl(df.head(3), "data/processed/notes_icd.jsonl")
    """
    required = ["SUBJECT_ID", "HADM_ID", "TEXT", "LABELS"]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"Missing required columns: {missing}"

    dest = Path(jsonl_path)
    ensure_parent_dir(dest)
    with open(dest, "w", encoding="utf-8") as f:
        for s, h, t, lbls in zip(df["SUBJECT_ID"], df["HADM_ID"], df["TEXT"], df["LABELS"]):
            if isinstance(lbls, str):
                lbls = json.loads(lbls)
            labels = [str(x) for x in (lbls or [])]
            rec = {"text": str(t), "labels": labels, "hadm_id": int(h), "subject_id": int(s)}
            line = json.dumps(rec, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            f.write(line + '\n')
    logging.info(f"[SAVE] {to_relpath(dest)} lines={len(df):,}")

def read_jsonl(path: Path, *, nrows: Optional[int] = None, logger=None) -> pd.DataFrame:
    """
    Read JSONL into a pandas DataFrame

    Parameters
    ----------
    path    : Path
        path to .jsonl file
    nrows   : int, optional
        Limit rows for sampling/debug (None = all).
    
    Returns
    -------
    pd.DataFrame
        Columns inferred from JSON keys.
    
    Notes
    -----
    - Each line must be a valid JSON object.
    - Automatically casts 'labels' to list[str] if possible.
    - Logs row/column counts but never prints raw TEXT.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    
    if logger is None:
        logger = logging.getLogger(__name__)

    records = []
    with p.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError as e:
                logger.warning(f"[SKIP] line {i}: {e}")
                continue
    
    if not records:
        logger.warning(f"[EMPTY] {path}")
        return pd.DataFrame()

    df = pd.DataFrame.from_records(records)

    df.columns = [c.upper() for c in df.columns]

    for col in ("SUBJECT_ID", "HADM_ID"):
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass

    if "LABELS" in df.columns:
        df["LABELS"] = df["LABELS"].apply(
            lambda x: x if isinstance(x, list) else []
        )

    logger.info(
        f"[READ] {p.as_posix()} -> rows={len(df):,}, cols={df.shape[1]}"
    )
    
    return df