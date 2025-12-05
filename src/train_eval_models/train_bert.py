"""
train_bert_model.py — ClinicalBERT Training Script (ICD Multi-Label)

Purpose
-------
Train a ClinicalBERT-style encoder on MIMIC-III discharge summaries for
multi-label ICD-9 classification. Uses the unified baseline utilities
(prepare_split, evaluate_and_save) and NeuralBaseModel-style wrapper
(ClinicalBERTModel).

Conventions
-----------
- Splits: JSONL files with columns: SUBJECT_ID, HADM_ID, TEXT, LABELS.
- Labels: multi-hot Y with shape (N, K), dtype={uint8, bool}.
- No PHI in logs, only counts/metrics/paths (relative).
- Metrics: Micro/Macro-F1, Precision@8/15, thresholds saved under models/.

Inputs
------
- data/splits/train.jsonl
- data/splits/val.jsonl
- data/splits/test.jsonl (optional but recommended)
- data/processed/label_map.json

Outputs
-------
- models/<run_name>/best.ckpt
- models/<run_name>/thresholds.json
- reports/bert_metrics.csv (one row per split/run)

Example
-------
>>> # CLI (example)
>>> python -m src.train_bert_model \\
...   --run-name clinicalbert_v1 \\
...   --train-jsonl data/splits/train.jsonl \\
...   --val-jsonl data/splits/val.jsonl \\
...   --test-jsonl data/splits/test.jsonl \\
...   --label-map data/processed/label_map.json \\
...   --model-name emilyalsentzer/Bio_ClinicalBERT \\
...   --batch-size 8 --max-len 512 --epochs 3 --lr 2e-5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


from src.utils.repro import set_seed
from src.utils.logger import setup_logger
from src.common.utils import prepare_split, evaluate_and_save
from src.models.BERT.bert_model import ClinicalBERTModel
from src.common.imbalance import compute_pos_weight

# =============================================================================
# Dataset + DataLoader builder
# =============================================================================
class ICDClinicalBERTDataset(Dataset):
    """
    Minimal Dataset wrapper for ClinicalBERT.

    Each item is a dict with keys:
        - "input_ids"      : LongTensor [L]
        - "attention_mask" : LongTensor [L]
        - (optional) "labels" : FloatTensor [K]
    """
    def __init__(
            self,
            texts: Sequence[str],
            labels: Optional[np.ndarray],
            tokenizer: AutoTokenizer,
            max_len: int
    ) -> None:
        super().__init__()
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = int(max_len)

        if labels is not None:
            assert labels.ndim == 2, (
                "[ICDClinicalBERTDataset] labels must be 2D [N, K]; "
                f"got shape={labels.shape}."
            )
            assert labels.shape[0] == len(self.texts), (
                "[ICDClinicalBERTDataset] labels/text length mismatch: "
                f"N_text={len(self.texts)} vs N_labels={labels.shape[0]}."
            )
    
    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one tokenized example.

        Returns
        -------
        batch : dict
            {
                "input_ids"      : LongTensor [L],
                "attention_mask" : LongTensor [L],
                "labels"         : FloatTensor [K]
            }
        """
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }
        
        if self.labels is not None:
            y = torch.as_tensor(self.labels[idx], dtype=torch.float32)
            item["labels"] = y

        return item

def build_dataloaders(
    train_path: Path,
    val_path: Optional[Path],
    test_path: Optional[Path],
    label_map_path: Path,
    tokenizer_name: str,
    max_len: int,
    batch_size: int,
    num_workers: int,
    logger: logging.Logger,
)  -> Tuple[
    DataLoader,
    Optional[DataLoader],
    Optional[DataLoader],
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Read splits via `prepare_split` and return DataLoaders + targets.

    Returns
    -------
    dl_train, dl_val, dl_test, Y_tr, Y_val, Y_te
    """
    logger.info("[BERT] loading tokenizer %s", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    X_tr, Y_tr, meta_tr, label_map = prepare_split(
        train_path, label_map_path, logger=logger
    )
    K = Y_tr.shape[1]
    logger.info("[BERT] train: N=%d, K=%d", len(X_tr), K)

    ds_train = ICDClinicalBERTDataset(X_tr, Y_tr, tokenizer, max_len)
    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # --- val ---
    dl_val: Optional[DataLoader] = None
    Y_val: Optional[np.ndarray] = None
    if val_path is not None:
        X_val, Y_val, meta_val, _ = prepare_split(
            val_path, label_map_path, logger=logger
        )
        assert Y_val.shape[1] == K, (
            "[BERT] val K mismatch: "
            f"train K={K} vs val K={Y_val.shape[1]}."
        )
        ds_val = ICDClinicalBERTDataset(X_val, Y_val, tokenizer, max_len)
        dl_val = DataLoader(
            ds_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("[BERT] val:   N=%d", len(X_val))
    
    # --- test ---
    dl_test: Optional[DataLoader] = None
    Y_te: Optional[np.ndarray] = None
    if test_path is not None:
        X_te, Y_te, meta_te, _ = prepare_split(
            test_path, label_map_path, logger=logger
        )
        assert Y_te.shape[1] == K, (
            "[BERT] test K mismatch: "
            f"train K={K} vs test K={Y_te.shape[1]}."
        )
        ds_test = ICDClinicalBERTDataset(X_te, Y_te, tokenizer, max_len)
        dl_test = DataLoader(
            ds_test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        logger.info("[BERT] test:  N=%d", len(X_te))

    return dl_train, dl_val, dl_test, Y_tr, Y_val, Y_te

# =============================================================================
# Training + Evaluation orchestration
# =============================================================================

def train_and_eval(args: argparse.Namespace) -> None:
    """
    Full pipeline: seed, DataLoaders, train ClinicalBERT, eval val/test.
    """
    set_seed(args.seed)
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_csv)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger(
        name=f"bert.{args.run}"
    )
    logger.info("[BERT] args = %s", vars(args))

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("[BERT] CUDA requested but not available; falling back to CPU.")
        device = torch.device("cpu")

    # --- DataLoaders ---
    dl_train, dl_val, dl_test, Y_tr, Y_val, Y_te = build_dataloaders(
        train_path=Path(args.train),
        val_path=Path(args.val) if args.val else None,
        test_path=Path(args.test) if args.test else None,
        label_map_path=Path(args.label_map),
        tokenizer_name=args.model_name_or_path,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        logger=logger,
    )
    K = Y_tr.shape[1]
    # --- class imbalance: pos_weight for BCEWithLogitsLoss ---
    pos_w = compute_pos_weight(getattr(args, "pos_weight", None), Y_tr, logger)
    if pos_w is not None:
        logger.info(
            "[IMB] Using pos_weight (K=%d) range=[%.2f, %.2f]",
            pos_w.numel(),
            float(pos_w.min()),
            float(pos_w.max()),
        )
    else:
        logger.info("[IMB] pos_weight disabled (None).")

    model = ClinicalBERTModel().build(
        num_labels=K,
        model_name_or_path=args.model_name_or_path,
        max_len=args.max_len,
        p_drop=args.p_drop,
        freeze_bert=args.freeze_bert,
        use_pooler=args.use_pooler,
    )
    model.to_device(device, logger=logger)

    train_cfg: Dict[str, Any] = dict(
        run=args.run,
        train=args.train,
        val=args.val,
        test=args.test,
        label_map=args.label_map,
        models_dir=args.models_dir,

        # eval / thresholds
        thr=args.thr,
        threshold_policy=args.threshold_policy,
        min_pos=args.min_pos,
        ks=tuple(args.ks),

        # training hyperparams
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        amp=args.amp,
        clip=args.clip,
        grad_accum=args.grad_accum,
        log_every=args.log_every,
        pos_weight=pos_w,
        # extra for ClinicalBERTModel.fit_loader
        y_val=Y_val,
    )

    logger.info("[BERT] starting training...")
    model.fit_loader(dl_train=dl_train, dl_val=dl_val, **train_cfg)

    # --- save checkpoint ---
    out_dir = models_dir / args.run
    model.save(out_dir, protocol=args.protocol, logger=logger)

    # --- evaluation ---
    def _eval_split(
        split_name: str,
        dl: Optional[DataLoader],
        Y: Optional[np.ndarray],
    ) -> None:
        if dl is None or Y is None or Y.size == 0:
            logger.warning("[BERT] skip %s: no data.", split_name)
            return

        logger.info("[BERT] scoring %s split...", split_name)
        P = model.predict_proba_loader(dl)
        assert P.shape == Y.shape, (
            f"[BERT:{split_name}] prob/label shape mismatch: "
            f"P={P.shape} vs Y={Y.shape}."
        )
        if split_name == "test":
            test_policy = "use_saved"
        else:
            test_policy = args.threshold_policy
            
        metrics = evaluate_and_save(
            args.run,          # run_name
            split_name,        # split_name
            Y,                 # y_true
            P,                 # y_scores
            thr=args.thr,
            threshold_policy=test_policy,
            metrics_csv=metrics_path,
            min_pos=args.min_pos,
            thresholds_dir=models_dir,
            logger=logger,
        )

    # val + test metrics
    if dl_val is not None and Y_val is not None:
        _eval_split("val", dl_val, Y_val)
    if dl_test is not None and Y_te is not None:
        _eval_split("test", dl_test, Y_te)

# =============================================================================
# CLI
# =============================================================================

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments for BERT training.

    Parameters
    ----------
    argv : list[str] | None
        Optional override of sys.argv[1:].

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes used across this script.
    """
    p = argparse.ArgumentParser(
        description="Train/evaluate ClinicalBERT ICD classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- I/O paths ---
    p.add_argument("--run", required=True, help="Run name (subdir under models_dir).")
    p.add_argument("--train", required=True, help="Train split JSONL.")
    p.add_argument("--val", required=False, default="", help="Val split JSONL.")
    p.add_argument("--test", required=False, default="", help="Test split JSONL.")
    p.add_argument("--label-map", required=True, help="Label map JSON (code -> idx).")
    p.add_argument("--models-dir", default="models", help="Root directory for model checkpoints / thresholds.")
    p.add_argument("--metrics-csv", default="reports/bert_metrics.csv", help="CSV to append metrics rows into.")

    # --- BERT / tokenizer ---
    p.add_argument("--model-name-or-path", default="emilyalsentzer/Bio_ClinicalBERT", help="HF model checkpoint for Clinical/Bio BERT.")
    p.add_argument("--max-len", type=int, default=512, help="Max wordpiece length.")
    p.add_argument("--p-drop", type=float, default=0.1, help="Dropout before head.")
    p.add_argument("--freeze-bert", action="store_true", help="Freeze encoder weights (feature extractor mode).")
    p.add_argument("--use-pooler", action="store_true", help="Use encoder.pooler_output instead of mean pooling.")

    # --- optimization ---
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--clip", type=float, default=1.0, help="Gradient clip (max norm).")
    p.add_argument("--amp", action="store_true", help="Enable mixed precision (torch.autocast + GradScaler).")
    p.add_argument("--log-every", type=int, default=50, help="Log training loss every N steps.")

    # --- thresholds / metrics ---
    p.add_argument("--threshold-policy", type=str, default="per_label_val", choices=["fixed", "global_val", "per_label_val"], help="Threshold selection strategy during eval.")
    p.add_argument("--thr", type=float, default=0.5, help="Fixed threshold if threshold_policy='fixed'.")
    p.add_argument("--min-pos", type=int, default=10, help="Min positives per label for tuning (per_label/global).")
    p.add_argument("--ks", type=int, nargs="+", default=[8, 15], help="Values of k for Precision@k.")
    p.add_argument("--pos_weight", default=None, help="Optional: 'balanced' or path to .npy of shape [K].")
    
    # misc
    p.add_argument("--device", default="cuda", help="Device string for torch.device, e.g. 'cuda' or 'cpu'.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--protocol", type=int, default=4, help="Pickle protocol for torch.save (model.save).",
    )

    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    """
    Entrypoint for CLI execution.

    Notes
    -----
    - For unit tests, call `train_and_eval(parse_args(custom_argv))`.
    """
    args = parse_args(argv)
    train_and_eval(args)


if __name__ == "__main__":
    main()
