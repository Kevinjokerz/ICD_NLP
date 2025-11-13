from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import os
from dotenv import load_dotenv

# =============================================================================
# Dataclass definition
# =============================================================================

@dataclass(frozen=True)
class PrepConfig:
    """
    Configuration for the prepare_data pipeline

    Resolution order
    ----------------
    1) CLI args (Highest priority)
    2) Environment (.env)
    3) Hard-coded default (lowest priority)

    Notes
    -----
    - Avoid leaking absolute paths; store and return only relative or user-provided paths.
    """
    seed: int                           # Random seed for preproducibility
    encoding: str                       # Character encoding used when reading CSV files.
    join_policy: str                    # Policy for handliing multiple notes per HADM_ID
    concat_sep: str                     # Separator used when concatenating multiple notes (if join_policy = 'concat')
    min_note_len: int                   # Minimum number of characters required to keep a note
    sample_intersection: bool           # Whether to restrict data to the intersection of HADM_IDs between notes and ICD tables
    max_hadm: int                       # Maximum number of admissions to keep after intersection sampling (0 = no limit)
    log_dir: str                        # Relative directory path for saving log files
    reports_dir: str                    # Relative directory path for saving sumary or metadata reports

# =============================================================================
# Helpers
# =============================================================================

def _as_bool(value: str) -> bool:
    """Robust boolean parser for environment values"""
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y", "on"}

def _decode_escapes(s: str) -> str:
    """Interpret escaped characters like '\\n' from .env files."""
    return s.encode("utf-8").decode("unicode_escape")

# =============================================================================
# Main loader
# =============================================================================

def load_config(cli_args: Any) -> PrepConfig:
    """
    Load configuration from CLI + .env with proper precedence.

    Parameters
    ----------
    cli_args    :   argparse.Namespace
        Parsed CLI arguments from prepare_data.py

    Returns
    -------
    PrepConfig
        Immutable configuration object for the pipeline.
    """
    load_dotenv()

    env = {
        "SEED": int(os.getenv("SEED", "42")),
        "ENCODING": os.getenv("ENCODING", "utf-8"),
        "JOIN_POLICY": os.getenv("JOIN_POLICY", "concat"),
        "CONCAT_SEP": _decode_escapes(os.getenv("CONCAT_SEP", "\\n\\n")),
        "MIN_NOTE_LEN": int(os.getenv("MIN_NOTE_LEN", "50")),
        "SAMPLE_INTERSECTION": _as_bool(os.getenv("SAMPLE_INTERSECTION", "false")),
        "MAX_HADM": int(os.getenv("MAX_HADM", "0")),
        "LOG_DIR": os.getenv("LOG_DIR", "logs"),
        "REPORTS_DIR": os.getenv("REPORTS_DIR", "reports")
    }

    seed = getattr(cli_args, "seed", None)
    seed = env["SEED"] if seed is None else int(seed)

    encoding = getattr(cli_args, "encoding", None) or env["ENCODING"]
    join_policy = getattr(cli_args, "join_policy", None) or env["JOIN_POLICY"]
    concat_sep = getattr(cli_args, "concat_sep", None) or env["CONCAT_SEP"]
    min_note_len = getattr(cli_args, "min_note_len", None)
    min_note_len = env["MIN_NOTE_LEN"] if min_note_len is None else int(min_note_len)

    if hasattr(cli_args, "sample_intersection"):
        sample_intersection = bool(cli_args.sample_intersection)
    else:
        sample_intersection = env["SAMPLE_INTERSECTION"]

    max_hadm = getattr(cli_args, "max_hadm", None)
    max_hadm = env["MAX_HADM"] if max_hadm is None else int(max_hadm)

    log_dir = env["LOG_DIR"]
    reports_dir = env["REPORTS_DIR"]

    valid_policies = {"longest", "concat"}
    if join_policy not in valid_policies:
        raise ValueError(f"JOIN_POLICY must be one of {valid_policies}, got {join_policy!r}")
    if max_hadm < 0:
        raise ValueError(f"MAX_HADM must be >= 0, got {max_hadm}")
    if min_note_len < 1:
        raise ValueError(f"MIN_NOTE_LEN must be >= 1, got {min_note_len}")

    return PrepConfig(
        seed=seed,
        encoding=encoding,
        join_policy=join_policy,
        concat_sep=concat_sep,
        min_note_len=min_note_len,
        sample_intersection=sample_intersection,
        max_hadm=max_hadm,
        log_dir=log_dir,
        reports_dir=reports_dir,
    )