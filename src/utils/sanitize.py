from __future__ import annotations
from pathlib import Path
from typing import Optional, Any, Iterable, List
import logging, json
import pandas as pd
import pandas.api.types as pdt
from .paths import get_project_root, looks_absolute, to_relpath
import numpy as np

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

def to_list_str(x: Any, *, uppercase: bool = True, remove_dots: bool = False,
                parse_json_list: bool = True, split_commas: bool=True, drop_empty: bool = True
) -> List[str]:
    """
    Robust converter for label-like fields -> list[str] (ICD-safe)

    Parameters
    ---------
    x       : Any
        Input object: list/tuple/set/ndarray/str/bytes/None/NaN/JSON-list string.
    uppercase   : bool, default=True
        Normalize tokens to uppercase
    remove_dots : bool, default=False
        Remove '.' from tokens (e.g., '250.00' -> '25000') if label_map uses no dots.
    parse_json_list : bool, default=True
        If x is a string that looks like a JSON list, parse it.
    split_commas: bool, default=True
        If x is a plain string with commas, split by ',' into tokens.
    drop_empty  : bool, default=True
        Drop empty tokens after stripping.

    Returns
    -------
    list[str]
        Cleaned tokens, PHI-safe, ready to match label_map keys.
    
    Notes
    -----
    - Do NOT log raw tokens here.
    - Keep behavior deterministic; document any normalization that affects matching.
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    
    if isinstance(x, (list, tuple, set, np.ndarray)):
        iter_tokens= list(x)
    elif isinstance(x, (bytes, bytearray)):
        s = x.decode("utf-8", errors="ignore")
        if parse_json_list and s.strip().startswith("[") and s.strip().endswith("]"):
            try:
                arr = json.loads(s)
                iter_tokens = list(arr) if isinstance(arr, Iterable) else [s]
            except Exception:
                iter_tokens = [s]
        elif split_commas and ("," in s):
            iter_tokens = s.split(",")
        else:
            iter_tokens = [s]
    elif isinstance(x, str):
        s = x.strip()
        if s == "":
            return []
        if parse_json_list and s.startswith("[") and s.endswith("]"):
            try:
                arr = json.loads(s)
                iter_tokens = list(arr) if isinstance(arr, Iterable) else [s]
            except Exception:
                iter_tokens = [s]
        elif split_commas and ("," in s):
            iter_tokens = s.split(",")
        else:
            iter_tokens = [s]
    else:
        iter_tokens = [x]
    
    out: List[str] = []
    for tok in iter_tokens:
        t = str(tok).strip()
        if remove_dots:
            t = t.replace(".", "")
        if uppercase:
            t = t.upper()
        if drop_empty and t == "":
            continue
        out.append(t)

    return out
        
            