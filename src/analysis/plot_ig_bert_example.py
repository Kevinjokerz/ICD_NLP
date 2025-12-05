"""
plot_ig_bert_example.py — Research-Grade Utility

Purpose
-------
Visualize token-level Integrated Gradients (IG) explanations for a single
(BERT-chunked) ICD prediction example, using the JSONL output produced by
src.explain.ig_bert.

Conventions
-----------
- Uses the same IG JSONL format as ig_bert.py.
- Only operates on pre-exported, PHI-safe IG outputs.
- All paths are relative to the project root (no absolute paths).

Inputs
------
- IG JSONL file produced by ig_bert.py. Each line is a JSON object with:
  - "tokens": list[str]
  - "ig_scores": list[float]
  - "target_label_idx": int
  - "pred_prob": float
  - "meta": dict (subject_id, hadm_id, icd_code, case_type, ... )
  - "example_id": str

Outputs
-------
- A simple bar-chart PNG highlighting the top-k tokens by IG score for a
  single explanation example.

Example
-------
>>> # TODO[test]: example CLI to generate one PNG
>>> # python -m src.explain.plot_ig_bert_example \\
>>> #   --ig-jsonl reports/xai/ig_bert_chunked/ig_..._split_val.jsonl \\
>>> #   --example-id 11446_165408_427 \\
>>> #   --top-k 30 \\
>>> #   --out-png reports/figures/ig/11446_165408_427.png

Author: Kevin Vo
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.utils.logger import setup_logger
from src.utils.paths import to_relpath

def load_ig_jsonl(path: Path, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    Load IG explanations from a JSONL file.

    Parameters
    ----------
    path : Path
        Path to the ig_*.jsonl file produced by ig_bert.py.
    logger: logging.Logger
        Logger to use.

    Returns
    -------
    list of dict
        A list of parsed JSON objects. Each object should contain
        at least: "tokens", "ig_scores", "target_label_idx",
        "pred_prob", "meta", and "example_id".

    Raises
    ------
    FileNotFoundError
        If the given path does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"[LOAD] IG JSONL file '{to_relpath(path)}' does not exist.")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue

            row = json.loads(s)

            assert "tokens" in row, "'tokens' does not exist in IG JSONL file."
            assert "ig_scores" in row, "'ig_scores' does not exist in IG JSONL file."
            rows.append(row)


    logger.info(f"[LOAD] Load {len(rows)} IG explanations.")
    return rows

def filter_rows(
        rows: List[Dict[str, Any]],
        *,
        example_id: Optional[str]=None,
        doc_index: Optional[int]=None,
        icd_code: Optional[str]=None,
        case_type: Optional[str]=None,
        logger: Optional[logging.Logger]=None,
) -> List[Dict[str, Any]]:
    """
       Filter IG rows according to example_id, doc_index, icd_code, and/or case_type.

       Parameters
       ----------
       rows : sequence of dict
           List (or iterable) of IG explanation objects loaded from JSONL.
       example_id : str, optional
           If provided, only rows with row["example_id"] == example_id are kept.
       doc_index : int, optional
           If provided (and example_id is None), only rows with
           row["meta"]["doc_index"] == doc_index are kept.
       icd_code : str, optional
           If provided, only rows with row["meta"]["icd_code"] == icd_code are kept.
       case_type : str, optional
           If provided, only rows with row["meta"]["case_type"] == case_type
           are kept. Typically one of {"TP", "FP", "FN"}.
       logger : logging.Logger, optional
            Logger to use.

       Returns
       -------
       list of dict
           Filtered subset of rows.

       Raises
       ------
       ValueError
           If no rows remain after filtering.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    filtered: List[Dict[str, Any]] = list(rows)

    # --- primary selector: example_id vs doc_index ---
    if example_id is not None:
        filtered = [r for r in filtered if r["example_id"] == example_id]
    elif doc_index is not  None:
        filtered = [r for r in filtered if r["meta"]["doc_index"] == doc_index]

    # --- secondary selectors: icd_code, case_type ---
    if icd_code is not None:
        filtered = [r for r in filtered if r["meta"]["icd_code"] == icd_code]
    if case_type is not None:
        filtered = [r for r in filtered if r["meta"]["case_type"] == case_type]

    if len(filtered) == 0:
        raise ValueError(f"No rows remain after filtering.")

    logger.info(f"[FILTER] {len(filtered)} rows remain after filtering.")
    return filtered


def choose_example(
    rows: Sequence[Dict[str, Any]],
    strategy: str = "first",
    logger: Optional[logging.Logger]=None,
) -> Dict[str, Any]:
    """
       Choose a single IG row to visualize.

       Parameters
       ----------
       rows : sequence of dict
           Filtered IG rows (e.g., output from filter_rows()).
       strategy : {"first", "max_prob"}, default="first"
           Strategy for choosing one row when multiple candidates remain.
           - "first": return rows[0].
           - "max_prob": return the row with maximum meta["pred_probs_label"].
       logger: logging.Logger, optional
        Logger to use.

       Returns
       -------
       dict
           A single IG explanation row.

       Raises
       ------
       ValueError
           If `rows` is empty.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not rows:
        raise ValueError(f"No rows remain after filtering.")

    chosen: Dict[str, Any] = {}
    if strategy == "first":
        chosen = rows[0]
    elif strategy == "max_prob":
        prob_list: List[float] = [float(r["meta"]["pred_probs_label"]) for r in rows]
        idx = int(np.argmax(prob_list))
        chosen = rows[idx]
    else:
        raise ValueError(f"Strategy {strategy} not supported.")

    logger.info(
        f"[CHOOSE] strategy={strategy} | example_id={chosen.get('example_id')})"
        f"icd_code={chosen.get('meta', {}).get('icd_code')} | "
        f"prob={chosen.get('meta', {}).get('pred_probs_label')}"
    )
    return chosen

def prepare_topk_tokens(
    tokens: Sequence[str],
    scores: Sequence[float],
    top_k: int,
) -> Tuple[List[str], np.ndarray]:
    """
    Compute top-k tokens by IG score.

    Parameters
    ----------
    tokens : sequence of str
        Tokens from the BERT chunk. Expected length up to max_len (e.g. 512).
    scores : sequence of float
        Corresponding IG scores per token (same length as `tokens`).
    top_k : int
        Number of tokens to select for visualization.

    Returns
    -------
    top_tokens : list of str
        Top-k tokens sorted by descending IG score.
    top_scores : np.ndarray
        Top-k scores sorted by descending IG score, shape (k,),
        where k = min(top_k, len(tokens)).

    Raises
    ------
    ValueError
        If lengths of tokens and scores differ.

    Notes
    -----
    - If top_k exceeds the number of tokens, it will be clipped.
    """

    if len(tokens) != len(scores):
        raise ValueError(f"Length of tokens and scores differ.")

    scores = np.asarray(scores, dtype=np.float32)
    n_tokens = len(tokens)

    if top_k <= 0:
        raise ValueError(f"top_k={top_k} must be greater than 0.")
    k = min(top_k, n_tokens)

    idx_sorted = np.argsort(-scores)
    idx_top = idx_sorted[:k]

    top_tokens = [tokens[i] for i in idx_top]
    top_scores = scores[idx_top]

    return top_tokens, top_scores

def plot_token_importance(
    tokens: Sequence[str],
    scores: Sequence[float],
    *,
    title: str = "",
    out_png: Optional[Path] = None,
    max_token_len: int = 20,
) -> None:
    """
    Plot a horizontal bar chart of token importance (IG scores).

    Parameters
    ----------
    tokens : sequence of str
        Tokens to plot (usually the output of prepare_topk_tokens()).
    scores : sequence of float
        IG scores corresponding to `tokens`.
    title : str, default=""
        Title to display above the plot (e.g. "ICD=V433, case=TP").
    out_png : Path, optional
        If provided, the figure will be saved to this path instead of
        being shown interactively.
    max_token_len : int, default=20
        Maximum length of token strings to display (longer ones will
        be truncated for readability).

    Returns
    -------
    None
    """
    assert len(tokens) == len(scores), "Length of tokens and scores differ."
    tokens_trunc = [
        (t if len(t) < max_token_len else t[:max_token_len - 1] + "...") for t in tokens
    ]
    scores_arr = np.asarray(scores, np.float32)
    y_pos = np.arange(len(tokens_trunc))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(y_pos, scores_arr, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(tokens_trunc)
    ax.yaxis.set_inverted(True)
    ax.set_title(title)
    ax.set_xlabel("IG score")
    ax.set_ylabel("token")
    plt.tight_layout()

    if out_png is not None:
        out_png.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_png)
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes:
        - ig_jsonl: Path to IG JSONL file.
        - example_id: Optional[str]
        - doc_index: Optional[int]
        - icd_code: Optional[str]
        - case_type: Optional[str]
        - top_k: int
        - out_png: Optional[Path]
        - strategy: str
    """
    parser = argparse.ArgumentParser(
        description="Visualize token-level IG for single ICD prediction example."
    )

    parser.add_argument("--ig_jsonl", type=Path, default="reports/xai/ig_bert_chunked/ig_bert_chunked_clinical_bs16_lr3e5_ep7_pw12_split_val.jsonl", help="Path to IG JSONL file.")
    parser.add_argument("--example_id", type=str, help="Example ID.")
    parser.add_argument("--doc_index", type=int, help="Document index.")
    parser.add_argument("--icd_code", type=str, help="ICD code.")
    parser.add_argument("--case_type", type=str, choices=["TP", "FP", "FN"],help="Case type.")
    parser.add_argument("--strategy", type=str, choices=["first", "max_prob"], default="first", help="Strategy.")
    parser.add_argument("--top_k", type=int, default=30, help="Number of tokens to select for visualization.")
    parser.add_argument("--out_png", type=Path, help="Path to output figure.")
    return parser.parse_args()

def main() -> None:
    """
    CLI entry point for plotting a single IG explanation example.

    Notes
    -----
    - This function wires together the other utilities:
      1) load_ig_jsonl()
      2) filter_rows()
      3) choose_example()
      4) prepare_topk_tokens()
      5) plot_token_importance()
    """
    logger = setup_logger("plot_ig_bert_example")
    args = parse_args()
    ig_path = args.ig_jsonl
    out_png = args.out_png

    rows = load_ig_jsonl(ig_path, logger=logger)

    filtered_rows = filter_rows(
        rows=rows,
        example_id=args.example_id,
        doc_index=args.doc_index,
        icd_code=args.icd_code,
        case_type=args.case_type,
        logger=logger,
    )

    chosen = choose_example(filtered_rows, strategy=args.strategy, logger=logger)

    meta = chosen["meta"]
    title = f"ICD={meta['icd_code']} | case={meta['case_type']} | prob={meta['pred_probs_label']:.3f}"

    SPECIAL = {"[CLS]", "[SEP]", "[PAD]", "("}
    tokens = []
    scores = []
    for t, s in zip(chosen["tokens"], chosen["ig_scores"]):
        if t in SPECIAL:
            continue
        tokens.append(t)
        scores.append(s)

    top_tokens, top_scores = prepare_topk_tokens(
        tokens=tokens,
        scores=scores,
        top_k=args.top_k,
    )

    plot_token_importance(
        tokens=top_tokens,
        scores=top_scores,
        title=title,
        out_png=out_png,
    )

if __name__ == "__main__":
    main()












