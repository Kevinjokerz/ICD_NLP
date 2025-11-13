"""
prepare - Data Preparation Package (ICD-Coding Project)

Purpose
-------
Provides deterministic and PHI-safe preprocessing utilities for:
- Merging MIMIC notes with ICD codes
- Building label maps
- Encoding multi-hot vectors
- Splitting subject-level data

Modules
-------
- prepare_data      : Clean and join NOTEEVENTS + DIAGNOSES_ICD
- build_label_map   : Create top-K ICD frequency map
- encode_labels     : Covert labels -> indices -> multi-hot matrix
- split_data        : Subject-aware stratified split

Convention
----------
- Reproducible (seed fixed)
- Relative paths only
- No PHI in logs
"""
from pathlib import Path

from .prepare_data import main as prepare_notes_icd
from .build_label_map import main as build_label_map
from .encode_labels import main as encode_labels
from .split_data import main as split_dataset

__all__ = [
    "prepare_notes_icd",
    "build_label_map",
    "encode_labels",
    "split_dataset",
]

# --- Package metadata ---
__version__ = "0.1.0"
__author__ = "Kevin Vo"
__license__ = "MIT"

import logging

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

logger.info("[INIT] prepare package loaded (v%s)", __version__)

_pkg_root = Path(__file__).resolve().parent
for f in ("prepare_data.py", "build_label_map.py", "encode_labels.py", "split_data.py"):
    assert (_pkg_root / f).exists(), f"Missing expected module: {f}"