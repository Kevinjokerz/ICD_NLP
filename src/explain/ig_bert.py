"""
ig_bert.py — Integrated Gradients for ClinicalBERT ICD model.

Purpose
-------
Compute token-level attribution scores for a trained ClinicalBERT ICD
classifier using Integrated Gradients (IG).

Conventions
-----------
- No PHI in logs: only counts, shapes, and IDs (SUBJECT_ID, HADM_ID).
- Splits: JSONL with columns: SUBJECT_ID, HADM_ID, TEXT, LABELS.
- Labels: indices 0..K-1 from label_map.json.
- Output: JSONL with tokens + IG scores per (example, label).

Inputs
------
- --split       : path to split JSONL (val/test).
- --label-map   : path to label_map.json.
- --models-dir  : directory with model runs (models/<run_name>/).
- --run         : run name (subdir under models_dir).
- --output-dir  : directory to save IG explanations.

Outputs
-------
- ig_<run>_split_<name>.jsonl
  Each line: {"example_id", "tokens", "ig_scores", "target_label_idx",
              "pred_prob", "meta": {...}}

Example
-------
>>> python -m src.explain.ig_bert \\
...   --split data/splits/test.jsonl \\
...   --label_map data/processed/label_map.json \\
...   --models_dir models \\
...   --run_name bert_chunked_clinical_R1 \\
...   --output_dir reports/xai/ig_bert \\
...   --num_examples 50 --ig_steps 32 --baseline_strategy pad
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import argparse
import json
import logging

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from src.utils.logger import setup_logger
from src.utils.repro import set_seed
from src.utils.paths import to_relpath
from src.common.utils import prepare_split
from src.models.BERT.bert_model import ClinicalBERTModel
from src.common.thresholds import load_thresholds
from src.train_eval_models.train_bert_chunked import (
    chunk_texts,
    aggregate_chunk_probs,
)

from captum.attr import LayerIntegratedGradients

@dataclass
class IGConfig:
    """
    Configuration for BERT + Integrated Gradients explanations.
    """

    split_path: Path
    label_map_path: Path
    models_dir: Path
    run_name: str
    output_dir: Path

    num_examples: int = 50
    max_len: int = 512
    ig_steps: int = 32
    baseline_strategy: str = 'pad'      # {'pad', "cls"}
    target_top_k_labels: int = 3
    doc_stride: int = 128
    batch_size: int = 32
    agg: str = "max"

    device: str = "cuda"
    seed: int = 42

    save_meta: bool = True

    def model_run_dir(self) -> Path:
        """
        Path to model run directory containing best checkpoint.
        """

        return Path(self.models_dir) / self.run_name


def load_label_map(path: Path, logger: logging.Logger) -> Dict[str, int]:
    """
    Load ICD label_map (code -> index).
    """

    if not path.is_file():
        raise FileNotFoundError(f"[CHECK] label_map not found: {to_relpath(path)}")

    with path.open("r", encoding="utf-8") as f:
        mapping: Dict[str, int] = json.load(f)

    logger.info(f"Loaded label_map from {to_relpath(path)}, K={len(mapping)}")
    return mapping

def select_examples(
    texts: Sequence[str],
    Y: np.ndarray,
    meta: pd.DataFrame,
    cfg: IGConfig,
    logger: logging.Logger,
) -> Tuple[List[str], np.ndarray, pd.DataFrame]:
    """
    Subsample examples for IG explanations.

    Current policy: random sampling.
    """
    n_total = len(texts)
    n_sel = min(cfg.num_examples, n_total)

    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(n_total, size=n_sel, replace=False)

    assert idx.shape == (n_sel,)
    assert np.issubdtype(idx.dtype, np.integer)

    texts_sel = [texts[i] for i in idx]
    Y_sel = Y[idx].copy()
    meta_sel = meta.iloc[idx].copy()

    logger.info(f"[SAMPLE] selected {n_sel}/{n_total} examples for IG explanations.")
    return texts_sel, Y_sel, meta_sel

def load_model_and_tokenizer(
    cfg: IGConfig,
    logger: logging.Logger,
) -> Tuple[ClinicalBERTModel, PreTrainedTokenizerBase]:
    """
    Load ClinicalBERTModel + HF tokenizer and move to cfg.device.

    Parameters
    ----------
    cfg
    logger

    Returns
    -------
    Tuple[ClinicalBERTModel, Any]
    """

    model_run_dir: Path = cfg.model_run_dir()
    if not model_run_dir.exists():
        raise ValueError(f"[CHECK] model_run_dir not found: {to_relpath(model_run_dir)}")

    logger.info("[LOAD] ClinicalBERTModel from %s", to_relpath(model_run_dir))

    model = ClinicalBERTModel.load(model_run_dir, logger=logger)

    hf_name = getattr(model.cfg, "model_name_or_path", None)
    if hf_name is None:
        raise ValueError("[CHECK] model.cfg.model_name_or_path is missing")

    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    device = torch.device(cfg.device)
    model = model.to_device(device, logger=logger)
    model.eval()

    return model, tokenizer

def encode_text(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    max_len: int
) -> Dict[str, torch.Tensor]:
    """
    Tokenize a single note into HF-style tensors.

    Returns
    -------
    batch : dict
        {
            "input_ids"      : LongTensor [1, L],
            "attention_mask" : LongTensor [1, L]
        }
    """
    enc = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    assert enc["input_ids"].shape == (1, max_len)

    batch = {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
    }
    return batch

def build_baseline_ids(
    input_ids: torch.Tensor,
    pad_token_id: int,
    cls_token_id: int,
    cfg: IGConfig
) -> torch.Tensor:
    """
    Construct baseline token IDs for IG (same shape as input_ids).
    """
    if cfg.baseline_strategy == "pad":
        baseline_ids = torch.full_like(input_ids, fill_value=pad_token_id)
    elif cfg.baseline_strategy == "cls":
        baseline_ids = torch.full_like(input_ids, fill_value=cls_token_id)
    else:
        raise ValueError(f"[CHECK] Unknown baseline_strategy: {cfg.baseline_strategy}")

    return baseline_ids

def forward_logits_for_ig(
    model: ClinicalBERTModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Wrapper used by Captum to obtain logits from ClinicalBERT.

    Notes
    -----
    - Must be compatible with LayerIntegratedGradients forward_func.
    - Should return logits [B, K] for the target label index.
    """
    logits = model.net(input_ids=input_ids, attention_mask=attention_mask)
    assert logits.ndim == 2, f"Expected [B, K], got {logits.shape}"
    return logits

def compute_ig_for_label(
    model: ClinicalBERTModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    target_label_idx: int,
    cfg: IGConfig,
) -> Dict[str, Any]:
    """
    Run Integrated Gradients for a single text and label index.

    Returns
    -------
    explanation : dict
        {
            "tokens": list[str],
            "ig_scores": list[float],
            "target_label_idx": int,
            "pred_prob": float,
            "meta": {...}
        }
    """

    device = next(model.parameters()).device

    encoded = encode_text(tokenizer, text, cfg.max_len)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id

    baseline_ids = build_baseline_ids(input_ids, pad_token_id, cls_token_id, cfg).to(device)

    emb_layer = model.net.encoder.embeddings

    lig = LayerIntegratedGradients(
        lambda ids: forward_logits_for_ig(model, ids, attention_mask.expand_as(ids)),
        emb_layer,
    )

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_label_idx,
        n_steps=cfg.ig_steps,
    )
    assert attributions.ndim == 3, f"Expected [B, L, H], got {attributions.shape}"
    attr_token = attributions.squeeze(0)
    token_scores = attr_token.abs().sum(dim=-1)
    token_scores = token_scores * attention_mask[0].to(token_scores.dtype)

    tokens: List[str] = tokenizer.convert_ids_to_tokens(
        input_ids[0].detach().cpu().tolist()
    )

    scores: List[float] = token_scores.detach().cpu().tolist()

    assert len(tokens) == len(scores), f"{len(tokens)=} != {len(scores)=}"

    logits = forward_logits_for_ig(model, input_ids, attention_mask)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred_prob = float(probs[target_label_idx])

    explanation: Dict[str, Any] = {
        "tokens": tokens,
        "ig_scores": scores,
        "target_label_idx": target_label_idx,
        "pred_prob": pred_prob,
        "meta": {
            "ig_steps": int(cfg.ig_steps),
            "baseline_strategy": cfg.baseline_strategy,
        },
    }

    return explanation

def save_explanations_jsonl(
    rows: List[Dict[str, Any]],
    dest: Path,
    logger: logging.Logger,
) -> None:
    """
    Save explanations as JSONL (one dict per line).
    """
    dest.parent.mkdir(exist_ok=True, parents=True)

    with dest.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=False)
            f.write("\n")

    logger.info("[SAVE] IG explanations saved to %s (rows=%d)", dest, len(rows))

def compute_chunk_probs(
    model: ClinicalBERTModel,
    encoding: Dict[str, np.ndarray],
    cfg: IGConfig,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Run model over all chunks and return chunk-level probs [N_chunks, K].

    Parameters
    ----------
    encodings : dict
        {
            "input_ids": np.ndarray [N_chunks, L],
            "attention_mask": np.ndarray [N_chunks, L],
        }

    Returns
    -------
    probs_chunks : np.ndarray
        Chunk-level probabilities, shape (N_chunks, K).
    """
    device = next(model.parameters()).device

    input_ids_np = encoding["input_ids"]
    attn_np = encoding["attention_mask"]

    assert input_ids_np.shape == attn_np.shape
    N_chunks, L = input_ids_np.shape

    input_ids = torch.from_numpy(input_ids_np).long()
    attention_mask = torch.from_numpy(attn_np).long()

    all_probs: List[np.ndarray] = []

    for start in range(0, N_chunks, cfg.batch_size):
        end = min(start + cfg.batch_size, N_chunks)

        batch_id = input_ids[start:end].to(device)
        batch_mask = attention_mask[start:end].to(device)

        with torch.no_grad():
            logits = forward_logits_for_ig(model, input_ids=batch_id, attention_mask=batch_mask)
        probs_batch = torch.sigmoid(logits).detach().cpu().numpy()

        all_probs.append(probs_batch)

    probs_chunks = np.concatenate(all_probs, axis=0)
    logger.info("[IG] computed chunk-level probs: shape=%s", probs_chunks.shape)
    return probs_chunks

def compute_ig_for_chunk(
    model: ClinicalBERTModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids_1d: torch.Tensor,
    attention_mask_1d: torch.Tensor,
    target_label_idx: int,
    cfg: IGConfig,
) -> Dict[str, Any]:
    """
    Run IG on a pre-tokenized chunk (1D input_ids, attention_mask).

    Notes
    -----
    - input_ids_1d, attention_mask_1d have shape [L].
    """
    assert input_ids_1d.ndim == 1 and attention_mask_1d.ndim == 1

    device = next(model.parameters()).device

    input_ids = input_ids_1d.unsqueeze(0).to(device)
    attention_mask = attention_mask_1d.unsqueeze(0).to(device)

    pad_token_id = tokenizer.pad_token_id
    cls_token_id = tokenizer.cls_token_id

    baseline_ids = build_baseline_ids(input_ids, pad_token_id, cls_token_id, cfg).to(device)

    lig = LayerIntegratedGradients(
        lambda ids: forward_logits_for_ig(model, ids, attention_mask.expand_as(ids)),
        model.net.encoder.embeddings,
    )

    attributions = lig.attribute(
        inputs=input_ids,
        baselines=baseline_ids,
        target=target_label_idx,
        n_steps=cfg.ig_steps,
    )
    assert attributions.ndim == 3, f"Expected [B, L, H], got {attributions.shape}"

    attr_token = attributions.squeeze(0)
    token_scores = attr_token.abs().sum(dim=-1)
    token_scores = token_scores * attention_mask[0].to(token_scores.dtype)

    tokens: List[str] = tokenizer.convert_ids_to_tokens(
        input_ids[0].detach().cpu().tolist()
    )
    scores: List[float] = token_scores.detach().cpu().tolist()

    assert len(tokens) == len(scores), f"{len(tokens)=} != {len(scores)=}"

    logits = forward_logits_for_ig(model, input_ids, attention_mask)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    pred_prob = float(probs[target_label_idx])

    explanation: Dict[str, Any] = {
        "tokens": tokens,
        "ig_scores": scores,
        "target_label_idx": target_label_idx,
        "pred_prob": pred_prob,
        "meta": {
            "ig_steps": int(cfg.ig_steps),
            "baseline_strategy": cfg.baseline_strategy,
        }
    }

    return explanation



def run_ig_for_split(
    cfg: IGConfig,
    logger: logging.Logger,
) -> None:
    """
    Orchestrate full IG pipeline over a subset of the split.
    """
    set_seed(cfg.seed)

    # ---- Load split ----
    texts, Y, meta, label_map = prepare_split(cfg.split_path, cfg.label_map_path, logger=logger)
    idx2code = {idx: code for code, idx in label_map.items()}
    texts_sel, Y_sel, meta_sel = select_examples(texts, Y, meta, cfg, logger=logger)
    N_docs_sel, K = Y_sel.shape

    # ---- Load model + tokenizer ----
    model, tokenizer = load_model_and_tokenizer(cfg, logger=logger)
    device = next(model.parameters()).device

    # ---- Load thresholds ----
    kind, thr = load_thresholds(
        models_dir=cfg.models_dir,
        run_name=cfg.run_name,
        logger=logger,
    )
    if kind != "per_label":
        logger.warning(
            "[IG] Expected per-label thresholds but got kind=%s; "
            "still using returned thr", kind
        )
    if isinstance(thr, np.ndarray):
        assert thr.shape == (K,), f"[IG] thr shape: {thr.shape} != (K={K},)"

    # ---- Chunk the "selected" docs using same logic as training ----
    encodings, doc_index = chunk_texts(
        texts=texts_sel,
        tokenizer=tokenizer,
        max_len=cfg.max_len,
        doc_stride=cfg.doc_stride,
    )

    # ---- Run model on all chunks ----
    probs_chunks = compute_chunk_probs(model, encodings, cfg, logger=logger)

    # ---- Aggregate chunk probs -> doc probs ----
    P_docs = aggregate_chunk_probs(
        probs=probs_chunks,
        doc_index=doc_index,
        agg=cfg.agg,
    )
    assert P_docs.shape == (N_docs_sel, K)

    yhat_docs = (P_docs >= thr).astype(np.uint8)

    explanations: List[Dict[str, Any]] = []

    for d in range(N_docs_sel):
        text = texts_sel[d]
        y_row = Y_sel[d]
        yhat_row = yhat_docs[d]
        probs_doc = P_docs[d]

        meta_row = meta_sel.iloc[d]

        tp_idx = np.where((y_row == 1) & (yhat_row == 1))[0]
        fp_idx = np.where((y_row == 0) & (yhat_row == 1))[0]
        fn_idx = np.where((y_row == 1) & (yhat_row == 0))[0]

        label_indices = np.concatenate([tp_idx, fp_idx, fn_idx])
        if label_indices.size == 0:
            continue
        scores = probs_doc[label_indices]
        order = np.argsort(-scores)
        label_indices = label_indices[order]
        label_indices = label_indices[: cfg.target_top_k_labels]

        doc_mask = (doc_index == d)
        doc_chunk_idx = np.where(doc_mask)[0]

        for label_idx in label_indices:
            label_idx = int(label_idx)

            if y_row[label_idx] == 1 and yhat_row[label_idx] == 1:
                case_type = "TP"
            elif y_row[label_idx] == 0 and yhat_row[label_idx] == 1:
                case_type = "FP"
            elif y_row[label_idx] == 1 and yhat_row[label_idx] == 0:
                case_type = "FN"
            else:
                case_type = "TN"

            chunk_probs_label = probs_chunks[doc_chunk_idx, label_idx]
            best_local = int(chunk_probs_label.argmax())
            best_chunk_idx = int(doc_chunk_idx[best_local])

            chunk_ids_1d = torch.from_numpy(
                encodings["input_ids"][best_chunk_idx]
            ).long()
            chunk_mask_1d = torch.from_numpy(
                encodings["attention_mask"][best_chunk_idx]
            ).long()

            exp = compute_ig_for_chunk(
                model=model,
                tokenizer=tokenizer,
                input_ids_1d=chunk_ids_1d,
                attention_mask_1d=chunk_mask_1d,
                target_label_idx=label_idx,
                cfg=cfg,
            )

            if cfg.save_meta:
                exp["meta"].update(
                    {
                        "doc_index": int(d),
                        "best_chunk_index": int(best_chunk_idx),
                        "subject_id": int(meta_row["SUBJECT_ID"]),
                        "hadm_id": int(meta_row["HADM_ID"]),
                        "case_type": case_type,
                        "y_true_label": int(y_row[label_idx]),
                        "y_pred_label": int(yhat_row[label_idx]),
                        "pred_probs_label": float(probs_doc[label_idx]),
                        "chunk_prob_label": float(chunk_probs_label[best_local]),
                        "icd_code": idx2code[label_idx],
                    }
                )
                exp["example_id"] = f"{int(meta_row['SUBJECT_ID'])}_{int(meta_row['HADM_ID'])}_{label_idx}"

            explanations.append(exp)

    out_jsonl = cfg.output_dir / f"ig_{cfg.run_name}_split_{cfg.split_path.stem}.jsonl"
    save_explanations_jsonl(explanations, out_jsonl, logger=logger)


# =============================================================================
# CLI
# =============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for IG explanations.
    """
    parser = argparse.ArgumentParser(
        description="Integrated Gradients explanations for ClinicalBERT ICD model."
    )

    # ---- IO Path ----
    parser.add_argument("--split", type=str, required=True, help="Pah to split JSONL (val/test).")
    parser.add_argument("--label_map", type=str, required=True, help="Path to label_map.json.")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing model runs (models/...).")
    parser.add_argument("--run_name", type=str, required=True, help="Run name (subdir under model_dir).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save IG explanations (JSONL).")

    # ---- IG / runtime knobs ----
    parser.add_argument("--num_examples", type=int, default=50)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--ig_steps", type=int, default=32)
    parser.add_argument("--baseline_strategy", type=str, default="pad", choices=["pad", "cls"])
    parser.add_argument("--target_top_k_labels", type=int, default=3)
    parser.add_argument("--doc_stride", type=int, default=128, help="Doc stride used for chunking (must match training).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for running model on chunks.")
    parser.add_argument("--agg", type=str, default="max", choices=["mean", "max"], help="Aggregation strategy used for chunking (must match training).")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser

def parse_args(argv: Optional[Sequence[str]] = None) -> IGConfig:
    """
    Parse CLI args -> IGConfig dataclass.
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = IGConfig(
        split_path=Path(args.split),
        label_map_path=Path(args.label_map),
        models_dir=Path(args.models_dir),
        run_name=str(args.run_name),
        output_dir=Path(args.output_dir),
        num_examples=int(args.num_examples),
        max_len=int(args.max_len),
        ig_steps=int(args.ig_steps),
        baseline_strategy=str(args.baseline_strategy),
        target_top_k_labels=int(args.target_top_k_labels),
        doc_stride=int(args.doc_stride),
        batch_size=int(args.batch_size),
        agg=str(args.agg),
        device=str(args.device),
        seed=int(args.seed),
    )
    return cfg

def main(argv: Optional[Sequence[str]] = None) -> None:
    """
    Entry point for IG CLI.
    """

    cfg = parse_args(argv)
    logger = setup_logger(name="ig_bert")

    logger.info("[INIT] IG for run=%s split=%s", cfg.run_name, cfg.split_path)
    run_ig_for_split(cfg, logger)

if __name__ == "__main__":
    main()