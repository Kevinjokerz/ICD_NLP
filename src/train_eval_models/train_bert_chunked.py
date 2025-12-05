"""
train_bert_chunked.py — Chunked ClinicalBERT Training (ICD Multi-Label)

Purpose
-------
Handle long discharge summaries by splitting into wordpiece chunks and
aggregating chunk-level predictions back to document-level labels.

Key ideas
---------
- Sliding-window chunking at tokenizer level (max_len, doc_stride).
- Training on chunk-level batches (labels copied from parent doc).
- Evaluation at document level: aggregate P(chunk | doc) -> P(doc).

Inputs
------
- data/splits/{train,val,test}.jsonl   # TEXT, LABELS, SUBJECT_ID, HADM_ID
- data/processed/label_map.json
- models/<run_name>/...                # for saving checkpoints / thresholds

Outputs
-------
- models/<run_name>/best.ckpt
- models/<run_name>/thresholds.json
- reports/bert_chunked_metrics.csv
"""

from __future__ import annotations
import  argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from src.utils.repro import set_seed
from src.utils.logger import setup_logger
from src.common.utils import prepare_split, evaluate_and_save
from src.common.imbalance import compute_pos_weight
from src.models.BERT.bert_model import ClinicalBERTModel
from src.utils.paths import to_relpath
from src.config.run_args import from_kwargs


# =============================================================================
# Chunking helpers
# =============================================================================

def chunk_texts(
    texts: Sequence[str],
    tokenizer: AutoTokenizer,
    *,
    max_len: int,
    doc_stride: int,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Tokenize + chunk each document into multiple segments.

    Parameters
    ----------
    texts : Sequence[str]
        Original discharge summaries (len = N_docs).
    tokenizer : AutoTokenizer
        HF tokenizer compatible with ClinicalBERT.
    max_len : int
        Max wordpiece length per chunk (including special tokens).
    doc_stride : int
        Overlap between neighboring chunks in wordpiece space.

    Returns
    -------
    encodings : dict[str, np.ndarray]
        HF-style arrays for all chunks:
        - "input_ids"      : int64 [N_chunks, L]
        - "attention_mask" : int64 [N_chunks, L]
    doc_index : np.ndarray
        Shape [N_chunks], each entry is parent document index in [0, N_docs).
    """

    encoded = tokenizer(
        list(texts),
        max_length=max_len,
        truncation=True,
        padding="max_length",
        return_overflowing_tokens=True,
        stride=doc_stride,
        return_attention_mask=True,
    )

    input_ids_list = encoded["input_ids"]
    attention_mask_list = encoded["attention_mask"]
    mapping_list = encoded["overflow_to_sample_mapping"]

    input_ids_np = np.asarray(input_ids_list, dtype=np.int64)
    attention_mask_np = np.asarray(attention_mask_list, dtype=np.int64)

    doc_index_np = np.asarray(mapping_list, dtype=np.int64)

    encodings = {
        "input_ids": input_ids_np,
        "attention_mask": attention_mask_np,
    }

    assert input_ids_np.shape[0] == doc_index_np.shape[0], (
        f"[CHUNK_TEXTS] mismatch: N-chunks(input_ids)={input_ids_np.shape[0]} "
        f"!= N_chunks(doc_index)={doc_index_np.shape[0]}"
    )

    assert input_ids_np.shape == attention_mask_np.shape, (
        f"[CHUNK_TEXTS] shape mismatch: "
        f"input_ids={input_ids_np.shape}, "
        f"attention_mask={attention_mask_np.shape}"
    )

    assert input_ids_np.shape[1] == max_len, (
        f"[CHUNK_TEXTS] unexpected seq length"
    )

    return encodings, doc_index_np

class ChunkedICDClinicalBERTDataset(Dataset):
    """
    Dataset over *chunks* instead of full documents.

    Each item:
        - "input_ids"      : LongTensor [L]
        - "attention_mask" : LongTensor [L]
        - "labels"         : FloatTensor [K]
        - "doc_index"      : LongTensor []  (parent document id)
    """

    def __init__(
        self,
        encodings: Dict[str, np.ndarray],
        doc_index: np.ndarray,
        Y_docs: np.ndarray,
    ) -> None:
        super().__init__()

        self.encodings = encodings
        self.doc_index = doc_index
        self.Y_docs = Y_docs

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]
        N_chunks, L = input_ids.shape

        # ---- encodings shape checks ----
        assert input_ids.ndim == 2, "[DATASET] input_ids must be 2D [N_chunks, L]"
        assert attention_mask.ndim == 2, "[DATASET] attention_mask must be 2D"
        assert input_ids.shape == attention_mask.shape, (
            f"[DATASET] input_ids and attention_mask shape mismatch: "
            f"{input_ids.shape} vs {attention_mask.shape}"
        )

        # ---- doc_index shape checks ----
        assert doc_index.shape == (N_chunks,), (
            f"[DATASET] doc_index must be shape (N_chunks,), got {doc_index.shape}"
        )

        # ---- Y_docs shape checks ----
        assert Y_docs.ndim == 2, "[DATASET] Y_docs must be 2D [N_docs, K]"
        N_docs, K = Y_docs.shape

        assert doc_index.max() < N_docs, (
            f"[DATASET] doc_index.max()={doc_index.max()} >= N_docs={N_docs}"
        )

        assert doc_index.min() >= 0, "[DATASET] doc_index has negative indices"

    def __len__(self) -> int:
        N_chunks = self.encodings["input_ids"].shape[0]
        return N_chunks

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        doc_id = self.doc_index[idx]

        input_ids_np = self.encodings["input_ids"][idx]
        attention_mask_np = self.encodings["attention_mask"][idx]
        labels_np = self.Y_docs[doc_id]

        input_ids_tensor = torch.as_tensor(input_ids_np, dtype=torch.long)
        attention_mask_tensor = torch.as_tensor(attention_mask_np, dtype=torch.long)
        labels_tensor = torch.as_tensor(labels_np, dtype=torch.float32)

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "labels": labels_tensor,
            "doc_id": int(doc_id)
        }


def aggregate_chunk_probs(
    probs: np.ndarray,
    doc_index: np.ndarray,
    *,
    agg: str = "max"
) -> np.ndarray:
    """
    Aggregate chunk-level probabilities to document-level.

    Parameters
    ----------
    probs : np.ndarray
        Chunk probabilities, shape (N_chunks, K) in [0,1].
    doc_index : np.ndarray
        Parent doc id for each chunk, shape (N_chunks,).
    agg : {"max", "mean"}
        Aggregation rule over chunks of the same document.

    Returns
    -------
    P_docs : np.ndarray
        Document-level probabilities, shape (N_docs, K).

    Notes
    -----
    - This is used at eval time (val/test) and optionally during thr tuning.
    """

    assert doc_index.ndim == 1, "[AGGREGATE_CHUNK_PROBS] doc_index must be 1D"
    assert probs.ndim == 2, "[AGGREGATE_CHUNK_PROBS] probs must be 2D [N_chunks, K]"
    assert doc_index.shape[0] == probs.shape[0], (
        "[AGGREGATE_CHUNK_PROBS] len(doc_index) must match N_chunks"
    )
    assert agg in {"max", "mean"}

    N_docs = int(doc_index.max()) + 1
    K = probs.shape[1]
    P_docs = np.zeros((N_docs, K), dtype=probs.dtype)

    for d in range(N_docs):
        mask = (doc_index == d)
        chunk_probs = probs[mask, :]

        if agg == "max":
            P_docs[d] = chunk_probs.max(axis=0)
        else:
            P_docs[d] = chunk_probs.mean(axis=0)

    return P_docs

# =============================================================================
# Dataloaders
# =============================================================================

def _prepare_chunked_split(
    split_path: Optional[Path],
    split_name: str,
    *,
    label_map_path: Path,
    tokenizer,
    max_len: int,
    doc_stride: int,
    logger: logging.Logger,
):
    """
    Helper (top-level, not nested) to build one chunked Dataset.

    Returns
    -------
    dataset : ChunkedICDClinicalBERTDataset or None
    Y_docs  : np.ndarray or None
    doc_idx : np.ndarray or None
    """
    if split_path is None:
        logger.info(f"[CHUNKED] {split_name}: path is None -> skipping.")
        return None, None, None

    logger.info(f"[CHUNKED] Loading {split_name} split from {to_relpath(split_path)}.")

    texts, Y_docs, meta, label_map = prepare_split(
        jsonl_path=split_path,
        label_map_path=label_map_path,
        logger=logger
    )
    logger.info(
        f"[CHUNKED] {split_name}: N_docs={len(texts)}, K={Y_docs.shape[1]}"
    )

    encodings, doc_index = chunk_texts(
        texts=texts,
        tokenizer=tokenizer,
        max_len=max_len,
        doc_stride=doc_stride,
    )

    logger.info(
        f"[CHUNKED] {split_name}: N_chunks={encodings['input_ids'].shape[0]}, "
        f"L={encodings['attention_mask'].shape[1]}"
    )

    dataset = ChunkedICDClinicalBERTDataset(
        encodings=encodings,
        doc_index=doc_index,
        Y_docs=Y_docs,
    )

    return dataset, Y_docs, doc_index

def build_chunked_dataloaders(
    train_path: Path,
    val_path: Optional[Path],
    test_path: Optional[Path],
    label_map_path: Path,
    tokenizer_name: str,
    max_len: int,
    doc_stride: int,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger
) -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Prepare chunked DataLoaders for train/val/test.

    Returns
    -------
    dl_tr, dl_val, dl_te,
    Y_tr_docs, Y_val_docs, Y_te_docs,
    doc_idx_val, doc_idx_test
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    ds_tr, Y_tr_docs, doc_idx_tr = _prepare_chunked_split(
        split_path=train_path,
        split_name="train",
        label_map_path=label_map_path,
        tokenizer=tokenizer,
        max_len=max_len,
        doc_stride=doc_stride,
        logger=logger,
    )

    assert ds_tr is not None and Y_tr_docs is not None

    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Val ----
    ds_val, Y_val_docs, doc_idx_val = _prepare_chunked_split(
        split_path=val_path,
        split_name="val",
        label_map_path=label_map_path,
        tokenizer=tokenizer,
        max_len=max_len,
        doc_stride=doc_stride,
        logger=logger,
    )
    if ds_val is not None:
        dl_va = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        dl_va = None
        Y_val_docs = None
        doc_idx_val = None

    # ---- Test ----
    ds_te, Y_te_docs, doc_idx_te = _prepare_chunked_split(
        split_path=test_path,
        split_name="test",
        label_map_path=label_map_path,
        tokenizer=tokenizer,
        max_len=max_len,
        doc_stride=doc_stride,
        logger=logger,
    )
    if ds_te is not None:
        dl_te = DataLoader(
            ds_te,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
    else:
        dl_te = None
        Y_te_docs = None
        doc_idx_te = None

    return (
        dl_tr,
        dl_va,
        dl_te,
        Y_tr_docs,
        Y_val_docs,
        Y_te_docs,
        doc_idx_val,
        doc_idx_te,
    )

# =============================================================================
# Train + eval orchestration
# =============================================================================

def save_probs_npz(
    run_dir: Path,
    split_name: str,
    y_docs: np.ndarray,
    p_docs: np.ndarray,
    doc_index: np.ndarray,
    logger: logging.Logger,
) -> None:
    """
    Save document-level predictions for later analysis.

    Artifacts format is parallel to train_bert:
    - {run_dir}/{split_name}_doc_preds.npz
      - "y_docs"   : [N_docs, K] multi-hot ground truth
      - "p_docs"   : [N_docs, K] probabilities
      - "doc_index": [N_chunks]  mapping chunks -> doc (for debugging)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = run_dir / f"{split_name}_doc_preds.npz"

    logger.info(f"[ARTIFACT] Saving {split_name} predictions to {to_relpath(out_path)}...")
    np.savez(out_path, y_docs=y_docs, p_docs=p_docs, doc_index=doc_index)

def eval_split_chunked(
    *,
    model,
    loader: Optional[DataLoader],
    y_docs: Optional[np.ndarray],
    doc_index: Optional[np.ndarray],
    split_name: str,
    run_name: str,
    agg: str,
    ks: list[int],
    metrics_csv: Path,
    run_dir: Path,
    thresholds_dir: Path,
    thr: float,
    threshold_policy: str,
    min_pos: int,
    device: torch.device,
    logger: logging.Logger,
) -> None:
    """
    Eval helper cho chunked BERT, parallel với eval logic trong train_bert:
    - predict chunk-level probs
    - aggregate -> document level
    - evaluate_and_save(...)
    - save_probs_npz(...)
    """
    if loader is None or y_docs is None or doc_index is None:
        logger.info(f"[EVAL:{split_name}] no data -> skipping")
        return

    logger.info(f"[EVAL:{split_name}] predicting chunk-level probabilities...")
    P_chunks = model.predict_proba_loader(loader)

    logger.info(f"[EVAL:{split_name}] aggregating chunks -> documents (agg={agg}) ...")
    P_docs = aggregate_chunk_probs(P_chunks, doc_index, agg=agg)

    assert P_docs.shape == y_docs.shape, (
        f"[BERT-CHUNKED:{split_name}] prob/label shape mismatch: "
        f"P_docs={P_docs.shape} vs Y={y_docs.shape}."
    )

    if split_name == "test":
        policy = "use_saved"
    else:
        policy = threshold_policy

    logger.info(f"[EVAL:{split_name}] evaluating at document level ...")
    metrics = evaluate_and_save(
        run_name=run_name,
        split_name=split_name,
        y_true=y_docs,
        y_scores=P_docs,
        thr=thr,
        threshold_policy=policy,
        metrics_csv=metrics_csv,
        min_pos=min_pos,
        thresholds_dir=thresholds_dir,
        logger=logger,
    )

    save_probs_npz(
        run_dir=run_dir,
        split_name=split_name,
        y_docs=y_docs,
        p_docs=P_docs,
        doc_index=doc_index,
        logger=logger,
    )

def train_and_eval(args: argparse.Namespace) -> None:
    """
    Full pipeline for chunked ClinicalBERT:
    - seed & logger
    - build chunked DataLoaders
    - train ClinicalBERTModel
    - eval (val/test) at document level
    """

    common_args, eval_args, train_args = from_kwargs(**vars(args))

    set_seed(common_args.seed)
    models_dir = Path(train_args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(name=f"bert_chunked.{common_args.run}")
    logger.setLevel(common_args.log_level)
    logger.info(f"[BERT-CHUNKED] common=%s eval=%s train=%s",
                common_args, eval_args, train_args)

    device = torch.device(args.device)
    logger.info("[BERT-CHUNKED] device=%s", device)

    # ---- 4) Build chunked dataloaders (common_args + model-specific) ----
    (
        dl_tr,
        dl_val,
        dl_te,
        Y_tr_docs,
        Y_va_docs,
        Y_te_docs,
        doc_idx_va,
        doc_idx_te,
    ) = build_chunked_dataloaders(
        train_path=Path(common_args.train),
        val_path=Path(common_args.val) if common_args.val else None,
        test_path=Path(common_args.test) if common_args.test else None,
        label_map_path=Path(common_args.label_map),
        tokenizer_name=args.model_name_or_path,
        max_len=args.max_len,
        doc_stride=args.doc_stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logger=logger,
    )

    if Y_va_docs is not None and doc_idx_va is not None:
        Y_va_chunks = Y_va_docs[doc_idx_va]
    else:
        Y_va_chunks = None

    K = Y_tr_docs.shape[1]

    # ---- 5) Imbalance ----
    pos_w = compute_pos_weight(train_args.pos_weight, Y_tr_docs, logger=logger)

    # ---- 6) Build model (model-specific args) ----
    model = ClinicalBERTModel().build(
        num_labels=K,
        model_name_or_path=args.model_name_or_path,
        max_len=args.max_len,
        p_drop=args.p_drop,
        freeze_bert=args.freeze_bert,
        use_pooler=args.use_pooler,
    )
    model.to_device(device, logger=logger)

    # ---- 7) Train config: mix common/eval/train ----
    train_cfg: Dict[str, Any] = dict(
        # common
        run=common_args.run,
        train=common_args.train,
        val=common_args.val,
        test=common_args.test,
        label_map=common_args.label_map,
        models_dir=train_args.models_dir,

        # eval
        thr=eval_args.thr,
        threshold_policy=eval_args.threshold_policy,
        min_pos=eval_args.min_pos,
        ks=eval_args.ks,

        # train
        epochs=train_args.epochs,
        lr=train_args.lr,
        weight_decay=train_args.weight_decay,
        amp=train_args.amp,
        grad_accum=train_args.grad_accum,
        clip=train_args.clip,
        log_every=train_args.log_every,
        pos_weight=pos_w,

        # extra for BERT
        y_val=Y_va_chunks,
        metrics_csv=str(metrics_path)
    )

    model.fit_loader(dl_train=dl_tr, dl_val=dl_val, logger=logger, **train_cfg)

    out_dir = models_dir / common_args.run
    model.save(out_dir, protocol=common_args.protocol, logger=logger)

    eval_split_chunked(
        model=model,
        loader=dl_val,
        y_docs=Y_va_docs,
        doc_index=doc_idx_va,
        split_name="val",
        run_name=common_args.run,
        agg=args.agg,
        ks=list(eval_args.ks),
        metrics_csv=metrics_path,
        run_dir=out_dir,
        thresholds_dir=models_dir,
        thr=eval_args.thr,
        threshold_policy=eval_args.threshold_policy,
        min_pos=eval_args.min_pos,
        device=device,
        logger=logger,
    )

    eval_split_chunked(
        model=model,
        loader=dl_te,
        y_docs=Y_te_docs,
        doc_index=doc_idx_te,
        split_name="test",
        run_name=common_args.run,
        agg=args.agg,
        ks=list(eval_args.ks),
        metrics_csv=metrics_path,
        run_dir=out_dir,
        thresholds_dir=models_dir,
        thr=eval_args.thr,
        threshold_policy=eval_args.threshold_policy,
        min_pos=eval_args.min_pos,
        device=device,
        logger=logger,
    )

# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for chunked ClinicalBERT training.

    Parameters
    ----------
    argv : list[str] | None
        Optional override of sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes used across this script.
    """
    parser = argparse.ArgumentParser(
        description="Train/eval chunked ClinicalBERT model ICD classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---------------------------------------------------------------------
    # Common run/data paths  (map thẳng vào CommonArgs / TrainArgs)
    # ---------------------------------------------------------------------
    parser.add_argument("--run", type=str, default="bert_chunked_run1", help="run name for logging and output dirs.")
    parser.add_argument("--train", type=str, default="data/splits/train.jsonl",help="path to train split (JSONL).")
    parser.add_argument("--val", type=str, default="data/splits/val.jsonl",help="path to validation split (JSONL).")
    parser.add_argument("--test", type=str, default="data/splits/test.jsonl",help="path to test split (JSONL).")
    parser.add_argument("--label_map", type=str, default="data/processed/label_map.json",help="path to label_map.json.")
    parser.add_argument("--models_dir", type=str, default="models",help="Root directory to save model checkpoints and thresholds.")
    parser.add_argument("--metrics_csv", type=str, default="reports/bert_chunked_metrics.csv",help="CSV file to append metrics rows into.")

    # ---------------------------------------------------------------------
    # BERT / tokenizer + chunking
    # ---------------------------------------------------------------------
    parser.add_argument("--model_name_or_path", type=str, default="emilyalsentzer/Bio_ClinicalBERT", help="HF model checkpoint for Clinical/Bio BERT.")
    parser.add_argument("--max_len", type=int, default=512, help="max wordpiece length per chunk (including special tokens).")
    parser.add_argument("--doc_stride", type=int, default=128, help="Overlap (in wordpieces) between neighboring chunks.")
    parser.add_argument("--agg", type=str, default="max", choices=["max", "mean"], help="How to aggregate chunk probabilities back to document level.")
    parser.add_argument("--p_drop", type=float, default=0.1, help="Dropout probability before the classification head.")
    parser.add_argument("--freeze_bert", action="store_true", help="Freeze encoder weights (feature-extractor mode).")
    parser.add_argument("--use_pooler", action="store_true", help="Use encoder.pooler_output instead of mean pooling.")

    # ---------------------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------------------
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1.0, help="Gradient clipping max L2 norm.")
    parser.add_argument("--amp", action="store_true", help="Enable automatic mixed precision.")
    parser.add_argument("--log_every", type=int, default=50, help="Log training loss every N steps.")

    # ---------------------------------------------------------------------
    # Thresholds / metrics
    # ---------------------------------------------------------------------
    parser.add_argument("--threshold_policy", type=str, default="per_label_val", choices=["fixed", "global_val", "per_label_val"], help="Threshold selection strategy during evaluation.")
    parser.add_argument("--thr", type=float, default=0.5, help="Fixed threshold if threshold_policy='fixed'.")
    parser.add_argument("--min_pos", type=int, default=10, help="Minimum positives per label for tuning.")
    parser.add_argument("--ks", type=int, nargs="+", default=[8, 15], help="Values of k for Precision@k.")
    parser.add_argument("--pos_weight", default=None, help="Optional: 'balanced' or path to .npy of shape [K].")

    # ---------------------------------------------------------------------
    # Misc (CommonArgs + runtime)
    # ---------------------------------------------------------------------
    parser.add_argument("--device", type=str, default="cuda", help="Torch device string, e.g. 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--protocol", type=int, default=4, help="Pickle protocol for torch.save (model.save).")
    parser.add_argument("--log_level", dest="log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging verbosity level.")


    return parser.parse_args(argv)

def main(argv: Optional[List[str]] = None) -> None:
    """
    Entrypoint for CLI usage.
    """
    args = parse_args(argv)
    train_and_eval(args)

if __name__ == "__main__":
    main()




