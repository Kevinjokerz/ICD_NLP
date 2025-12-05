from __future__ import annotations
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
from typing import List, Optional
import argparse
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.utils.logger import setup_logger
from src.utils.repro import set_seed
from src.utils import to_relpath

RUN_NAMES = [
    "first_run_tfidf",           # TF-IDF baseline run_name
    "cnn_len1024_ep80_pw_bal",   # TextCNN run_name
    "bigru_full_ep80_nsota",     # BiGRU run_name
    "bert_full_med_bs8_lr3e5",   # ClinicalBERT run_name
    "bert_chunked_clinical_bs16_lr3e5_ep7_pw12",
]

DISPLAY_NAMES = [
    "TF-IDF LR",
    "TextCNN",
    "BiGRU",
    "ClinicalBERT",
    "chunked ClinicalBERT",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simple bar plots for Micro/Macro-F1 and Precision@k."
    )
    parser.add_argument("--baseline_csv", default="reports/baseline_metrics.csv")
    parser.add_argument("--bert_csv", default="reports/bert_metrics.csv")
    parser.add_argument("--chunked_bert_csv", default="reports/bert_chunked_metrics.csv")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out_dir", default="reports/figures")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING"])
    
    return parser

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)

def load_all_metrics(baseline_csv: str,
                     bert_csv: str,
                     chunked_bert_csv: str,
                     logger: logging.Logger) -> pd.DataFrame:
    """
    Read baseline + BERT metrics CSVs and concatenate.
    """
    paths = [baseline_csv, bert_csv, chunked_bert_csv]
    frames: List[pd.DataFrame] = []

    for p in paths:
        csv_path = Path(p)
        if not csv_path.exists():
            logger.warning("[SKIP] metrics CSV not found: %s", to_relpath(csv_path))
            continue

        df = pd.read_csv(csv_path)
        df.columns = [str(c).strip() for c in df.columns]
        frames.append(df)
        logger.info("[READ] %s rows=%d", to_relpath(csv_path), len(df))

    if not frames:
        raise ValueError("No metrics CSVs loaded.")
    
    df_all = pd.concat(frames, ignore_index=True)
    logger.info("Columns in metrics: %s", list(df_all.columns))
    return df_all


def collect_metrics_for_runs(df_all: pd.DataFrame,
                             split: str,
                             run_names: List[str],
                             logger: logging.Logger) -> pd.DataFrame:
    """
    Filter to a given split and pick the last row for each run_name.

    Returns a DataFrame with one row per run_name, in the same order as run_names.
    """
    df_split = df_all[df_all["split"] == split].copy()

    if df_split.empty:
        raise ValueError(f"No rows for split='{split}'")
    
    rows = []
    for rn in run_names:
        df_run = df_split[df_split["run_name"] == rn]
        if df_run.empty:
            raise ValueError(f"No rows for run_name='{rn}' and split='{split}'")
        row = df_run.iloc[-1]
        rows.append(row)

    df_sel = pd.DataFrame(rows)
    logger.info("Selected runs for split=%s:", split)
    logger.info(df_sel[["run_name","micro_f1","macro_f1","p@8","p@15"]])
    return df_sel


def plot_metric_bar(values: np.ndarray,
                    labels: List[str],
                    metric_name: str,
                    out_path: Path,
                    logger: logging.Logger) -> None:
    """
    Plot a single bar chart for one metric across models.
    """

    assert len(values) == len(labels)
    fig, ax = plt.subplots(figsize=(4, 4))
    x = np.arange(len(labels))
    ax.bar(x, values)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name)

    for xi, v in zip(x, values):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("[PLOT] %s -> %s", metric_name, to_relpath(out_path))

def run(args: argparse.Namespace) -> None:
    logger = setup_logger(name="simple_metrics")
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    set_seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- load + filter ---
    df_all = load_all_metrics(args.baseline_csv, args.bert_csv, args.chunked_bert_csv ,logger)
    df_sel = collect_metrics_for_runs(df_all, args.split, RUN_NAMES, logger)

    assert {"micro_f1","macro_f1","p@8","p@15"} <= set(df_sel.columns)

    micro = df_sel["micro_f1"].to_numpy()
    macro = df_sel["macro_f1"].to_numpy()
    p8    = df_sel["p@8"].to_numpy()
    p15   = df_sel["p@15"].to_numpy()

    # --- 4 plots ---
    plot_metric_bar(micro, DISPLAY_NAMES, "Micro-F1", out_dir / "micro_f1_bar.png", logger)
    plot_metric_bar(macro, DISPLAY_NAMES, "Macro-F1", out_dir / "macro_f1_bar.png", logger)
    plot_metric_bar(p8, DISPLAY_NAMES, "Precision@8", out_dir / "p8_bar.png", logger)
    plot_metric_bar(p15, DISPLAY_NAMES, "Precision@15", out_dir / "p15_bar.png", logger)

def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
