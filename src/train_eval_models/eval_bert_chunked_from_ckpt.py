"""
eval_bert_chunked_from_ckpt.py — Eval-only for an existing chunked ClinicalBERT run.

Purpose
-------
- Reload best.ckpt for a given run.
- Rebuild chunked DataLoaders (train/val/test splits).
- Run eval_split_chunked() for val and test to:
  * compute micro/macro-F1, P@k
  * write rows into metrics CSV
  * save val/test doc-level predictions as .npz
"""

from pathlib import Path
import torch

from src.train_eval_models.train_bert_chunked import (
    build_chunked_dataloaders,
    eval_split_chunked
)

from src.models.BERT.bert_model import ClinicalBERTModel
from src.utils.logger import setup_logger


def main() -> None:

    run_name = "bert_chunked_clinical_bs16_lr3e5_ep7_pw12"

    train_path = Path("data/splits/train.jsonl")
    val_path = Path("data/splits/val.jsonl")
    test_path = Path("data/splits/test.jsonl")
    label_map = Path("data/processed/label_map.json")

    models_dir = Path("models")
    run_dir = models_dir / run_name
    metrics_csv = Path("reports/bert_chunked_metrics.csv")

    tokenizer_name = "emilyalsentzer/Bio_ClinicalBERT"
    max_len        = 512
    doc_stride     = 128
    batch_size     = 16
    num_workers    = 0
    agg            = "max"
    ks             = [8, 15]
    thr            = 0.5
    threshold_policy = "per_label_val"
    min_pos        = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = setup_logger(name=f"bert_chunked.eval_only.{run_name}")
    logger.info(f"[EVAL-ONLY] run={run_name}, device={device}")

    # ---------- Build chunked DataLoaders ----------
    (
        dl_tr,
        dl_va,
        dl_te,
        Y_tr_docs,
        Y_va_doc,
        Y_te_docs,
        doc_idx_va,
        doc_idx_te,
    ) = build_chunked_dataloaders(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        label_map_path=label_map,
        tokenizer_name=tokenizer_name,
        max_len=max_len,
        doc_stride=doc_stride,
        batch_size=batch_size,
        num_workers=num_workers,
        logger=logger,
    )

    model = ClinicalBERTModel.load(run_dir, logger=logger)
    model.to_device(device, logger=logger)

    if dl_va is not None and Y_va_doc is not None:
        eval_split_chunked(
            model=model,
            loader=dl_va,
            y_docs=Y_va_doc,
            doc_index=doc_idx_va,
            split_name="val",
            run_name=run_name,
            agg=agg,
            ks=ks,
            metrics_csv=metrics_csv,
            run_dir=run_dir,
            thresholds_dir=models_dir,
            thr=thr,
            threshold_policy=threshold_policy,
            min_pos=min_pos,
            device=device,
            logger=logger,
        )

    if dl_te is not None and Y_te_docs is not None:
        eval_split_chunked(
            model=model,
            loader=dl_te,
            y_docs=Y_te_docs,
            doc_index=doc_idx_te,
            split_name="test",
            run_name=run_name,
            agg=agg,
            ks=ks,
            metrics_csv=metrics_csv,
            run_dir=run_dir,
            thresholds_dir=models_dir,
            thr=thr,
            threshold_policy=threshold_policy,
            min_pos=min_pos,
            device=device,
            logger=logger,
        )

if __name__ == "__main__":
    main()