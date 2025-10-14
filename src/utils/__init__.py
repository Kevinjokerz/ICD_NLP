"""
Unified public API for `src/utils`.

Purpose
-------
Exposes only *stable* utilities used across the pipeline:
- repro         : seeds & determinism
- paths         : project-root relative helpers (no absolute leak)
- sanitize      : JSON/path sanitization + assertions
- io            : CSV/Parquet/JSONL I/O with path-safety
- logger        : opt-in setup (no side effects on import)
- config        : CLI + .env configuration loader

Notes
-----
- Keep this surface minimal and stable to avoid breaking import elsewhere
- Do NOT initialize logging/config on import
"""
# --- Reproducibility ---------------------------------------------------------
from .repro import DEFAULT_SEED, set_seed

# --- Paths -------------------------------------------------------------------
from .paths import get_project_root, looks_absolute, to_relpath, safe_join_rel

# --- Sanitization ------------------------------------------------------------
from .sanitize import (
    sanitize_json_like, 
    sanitize_path_columns,
    assert_no_abs_in_df, 
    assert_no_abs_in_json_series,
)

# --- I/O ---------------------------------------------------------------------
from .io import (
    ensure_parent_dir, 
    read_csv, 
    save_csv_with_json_labels,
    save_table,
    read_table,
    export_jsonl,
)

# --- Logging -----------------------------------------------------------------
from .logger import setup_logger

# --- Config ------------------------------------------------------------------
from .config import PrepConfig, load_config

# --- Public API (explicit) ---------------------------------------------------
__all__ = [
    # repro
    "DEFAULT_SEED", "set_seed",

    # paths
    "get_project_root", "looks_absolute", "to_relpath", "safe_join_rel",

    # sanitize
    "sanitize_json_like", "sanitize_path_columns",
    "assert_no_abs_in_df", "assert_no_abs_in_json_series",

    # io
    "ensure_parent_dir", "read_csv", "save_csv_with_json_labels",

    # logger
    "setup_logger",

    # config
    "PrepConfig", "load_config"
]