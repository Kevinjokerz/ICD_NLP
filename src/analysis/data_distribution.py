"""
data_distribution.py — Label & Note Distribution + Long-Tail Plots

Purpose
-------
Summarize dataset statistics and visualize long-tail ICD distribution:
- Split-level stats (N, K, label density, labels-per-note)
- Long-tail label frequency (rank vs count, log-scale)
- Coverage curve: % of all label assignments covered by top-k labels
- Histogram of labels per note

Conventions
-----------
- PHI-safe: never log raw TEXT.
- Paths are relative to repo root.
- Figures saved under reports/figures/.

Inputs
------
- data/splits/{train,val,test}.jsonl
- data/processed/label_map.json

Outputs
-------
- Console stats
- PNG plots under reports/figures/
- (Optional) reports/data_distribution.csv

Author: Kevin Vo
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger
from src.utils.repro import set_seed
from src.common.utils import prepare_split
from src.utils import to_relpath

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot long-tail label frequency from train split."
    )


    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--val", default="data/splits/val.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--label_map", default="data/processed/label_map.json")

    parser.add_argument("--out_dir", default="reports/figures", help="Directory to save PNG plots.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    
    return parser

# =============================================================================
# Core helper
# =============================================================================
def plot_labels_per_note_hist(labels_per_note: np.ndarray,
                              split_name: str,
                              out_dir: Path,
                              logger: logging.Logger) -> None:
    """
    Plot histogram of number of labels per note for a split.

    Parameters
    ----------
    labels_per_note : np.ndarray
        Shape (N,), number of labels per discharge summary.
    split_name : str
        "train", "val", or "test".
    out_dir : Path
        Directory to save PNG.
    logger : logging.Logger
    """
    assert labels_per_note.ndim == 1
    max_labels = int(labels_per_note.max())
    bins = np.arange(0, max_labels + 2)

    fig, ax = plt.subplots()
    ax.hist(labels_per_note, bins=bins, density=False)
    ax.hist(labels_per_note, bins=bins, density=False)
    ax.set_ylabel("Number of notes")
    ax.set_title(f"Distribution of labels per note ({split_name})")

    fig.tight_layout()
    out_path = out_dir / f"labels_per_note_{split_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    logger.info("[PLOT] labels-per-note %s -> %s", split_name, to_relpath(out_path))

def plot_longtail_for_split(split_name: str,
                            split_path: Path,
                            label_map_path: Path,
                            out_dir: Path,
                            logger: logging.Logger) -> None:
    """
    Load one split, compute label frequencies, and plot long-tail curve.

    Parameters
    ----------
    split_name : {"train","val","test"}
    split_path : Path
        JSONL file for the split.
    label_map_path : Path
        Path to label_map.json for consistency.
    out_dir : Path
        Directory to save PNG.
    logger : logging.Logger
    """
    if not split_path.exists():
        logger.warning("[SKIP] %s split not found: %s",
                        split_name, to_relpath(split_path))
        return

    texts, Y, meta, label_map = prepare_split(
        jsonl_path=split_path,
        label_map_path=label_map_path,
        logger=logger
    )

    assert Y.ndim == 2
    N, K = Y.shape

    # --- Histogram labels per note (bell-ish) ---
    labels_per_note = Y.sum(axis=1).astype("int32")
    plot_labels_per_note_hist(labels_per_note, split_name, out_dir, logger)


    density = float(Y.mean())
    logger.info(
        "[%s] N=%d, K=%d, density=%.4f",
        split_name.upper(), N, K, density,
    )

    counts = Y.sum(axis=0).astype("int64")
    assert counts.shape == (K,)
    sorted_count = np.sort(counts)[::-1]
    ranks = np.arange(1,  len(sorted_count) + 1)


    fig, ax = plt.subplots()
    ax.plot(ranks, sorted_count)
    ax.set_xlabel("Label rank (sorted by frequency)")
    ax.set_ylabel("Number of notes with this label")
    ax.set_title(f"Long-tail ICD label frequency ({split_name})")
    ax.set_yscale("log")
    fig.tight_layout()

    # --- Save PNG ---
    out_path = out_dir / f"longtail_{split_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("[PLOT] %s -> %s", split_name, out_path.as_posix())


# =============================================================================
# Orchestration
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    logger = setup_logger(name="longtail")
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    set_seed(int(args.seed))

    train_p = Path(args.train)
    val_p   = Path(args.val)
    test_p  = Path(args.test)
    lm_p    = Path(args.label_map)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_path in [
        ("train", train_p),
        ("val",   val_p),
        ("test",  test_p),
    ]:
        plot_longtail_for_split(split_name, split_path, lm_p, out_dir, logger)




if __name__ == "__main__":
    main()