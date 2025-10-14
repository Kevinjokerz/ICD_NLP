from __future__ import annotations
from pathlib import Path
from typing import Optional, Any
import logging, json
import pandas as pd
import pandas.api.types as pdt
from .paths import get_project_root, looks_absolute, to_relpath

ABS_HINTS = (":\\", ":/", "\\\\")   # Window drive, URL-like, UNC shares

def sanitize_json_like(obj, root: Optional[Path] = None):
    """
    Recursively convert absolute string paths within nested lists/dicts to
    project-root-relative POSIX paths.

    Parameters
    ----------
    obj  : Any
        Arbitrary Python object (str/list/dict/primitive).
    root : Optional[Path]
        Base directory for relative path conversion (defaults to project root).
    
    Returns
    --------
    Any
        Sanitized object with absolute paths relative POSIX paths.
    """
    base = Path(root or get_project_root()).resolve()

    # --- String case ---
    if isinstance(obj, str) and looks_absolute(obj):
        try:
            return to_relpath(obj, base)
        except Exception as e:
            logging.warning(f"[WARN] sanitize_json_like: failed to relative {obj!r} ({e}).")
            return obj  # fallback: keep original
        
    # --- List case ---
    if isinstance(obj, list):
        return [sanitize_json_like(v, base) for v in obj]
    
    # --- Dict case ---
    if isinstance(obj, dict):
        return{k: sanitize_json_like(v, base) for k, v in obj.items()}
    
    # --- primitive open ---
    return obj

def assert_no_abs_in_df(df: pd.DataFrame) -> None:
    """
    Assert that suspected path columns do not contain absolute paths.

    Parameters
    ----------
    df  : pd.DataFrame
    """
    for c in df.columns:
        if any(k in c.lower() for k in ["path", "file", "dir", "paths", "files", "dirs"]):
            bad = df[c].map(lambda x: looks_absolute(str(x)) if pd.notna(x) else False)
            assert not bad.any(), f"Absolute path leakage in column '{c}'"

def assert_no_abs_in_json_series(series: pd.Series, n: int = 50) -> None:
    """
    Assert that a Series of JSON strings (or objects) does not contain absolute paths.
    
    Parameters
    ----------
    series  :   pd.Series
        Series of JSON-serializable objects or JSON strings.
    n       :   int, optional
        Number of rows to sample for validation (default=50)
    
    Raises
    ------
    AssertionError
        If an absolute or unsafe path pattern is detected in any sampled record.
    """
    sample = series.head(n)
    for v in sample:
        if isinstance(v, str):
            try:
                obj = json.loads(v)
            except json.JSONDecodeError:
                continue
        else:
            obj = v
        
        try:
            s = json.dumps(obj)
        except Exception:
            continue
        
        s_lower = s.lower().strip()
        # --- Heuristic checks for absolute path leakage ---
        if s_lower.startswith("/") or s_lower.startswith("\\"):
            raise AssertionError("Absolute path leakage detected (Unix/Windows roots).")
        if "://" in s_lower[:12]:
            raise AssertionError("Absolute path leakage detected (URL scheme)")
        
        for h in ABS_HINTS:
            idx = s_lower.find(h)
            if 0 <= idx <= 10:
                raise AssertionError(f"Absolute path leakage detected (pattern '{h}') in JSON payload.")

def sanitize_path_columns(df: pd.DataFrame, root: Optional[Path]=None) -> pd.DataFrame:
    """
    Detect columns that likely store file system paths and convert to relative.

    Heuristics: column name contains 'path', 'file', or 'dir' (case-insensitive).

    Parameters
    ----------
    df  : pd.DataFrame
    root: Optional[Path]

    Returns
    -------
    pd.DataFrame
        New DataFrame with sanitized path-like columns.
    """
    base = Path(root or get_project_root()).resolve()

    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in ["path", "file", "dir", "paths", "files", "dirs"])
        and (pdt.is_string_dtype(df[c]) or df[c].dtype == object)
    ]
    
    out = df.copy()

    def _fix_cell(x):
        if pd.isna(x):
            return x
        sx = str(x)
        if looks_absolute(sx):
            try:
                return to_relpath(sx, base)
            except Exception as e:
                logging.warning(f"[SANITIZE] Could not relativize {sx!r}: {e}")
                return sx
        return sx
    
    for c in candidates:
        out[c] = out[c].map(_fix_cell)
    
    logging.debug(f"[SANITIZED] columns={candidates}")
    assert_no_abs_in_df(out)
    for c in df.columns:
        if c not in candidates:
            assert df[c].dtype == out[c].dtype, f"dtype changed for non-path column '{c}'"

    return out