"""
train_seq_model.py - CLI Runner for CNN/BiGRU (ICD-9 Multi-Label)

Purpose
-------
Provide a single CLI to train/evaluate TextCNN or BiGRU on
MIMIC-III discharge summaries. Emphasizes PHI-safe logging, reproducibility, and
metrics-first evaluation with configurable threshold policies and model presets.


Conventions
-----------
- Reproducibility: deterministic seed; log library versions.
- Portability: relative paths only; artifacts under models/ and reports/.
- Safety: never print raw text; only log counts/shapes.


Inputs
------
- data/splits/train.jsonl : JSONL with columns TEXT (str), LABELS (list[str]), ids
- data/splits/val.jsonl   : same schema (optional)
- data/splits/test.jsonl  : same schema (optional)
- data/processed/label_map.json : {"ICD_CODE": int_idx, ...}

Outputs
-------
- models/<run>/best.ckpt            : checkpoint (state_dict + config)
- models/<run>/thresholds.json      : tuned thresholds (by policy)
- reports/baseline_metrics.csv      : appended rows per split
"""

from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
from collections import Counter

from src.utils.logger import setup_logger
from src.utils.repro import set_seed as set_global_seed
from src.utils.paths import to_relpath
from src.common.utils import prepare_split, evaluate_and_save
from src.common.thresholds import load_thresholds, save_thresholds
from src.common.metrics import compute_f1s, precision_at_k
from src.common.imbalance import compute_pos_weight

from src.common.textdata import Tokenizer, Vocab, NotesDataset, pad_collate
from src.models.cnn.cnn_model import TextCNNModel
from src.models.bigru.bigru_model import BiGRUModel

# =============================================================================
# CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for CNN/BiGRU runner.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser with all options.
    """
    parser = argparse.ArgumentParser(
        description="Run CNN/BiGRU pipeline for ICD-9 multi-label classification (MIMIC-III)"
    )

    # --- Model Selection & run info
    parser.add_argument("--model", choices=["cnn", "bigru"], required=True,
                        help="choose sequence model: 'cnn' or 'bigru'.")
    parser.add_argument("--run", default="seq_run_v1", help="Run name used for artifacts.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="torch.device string, e.g., 'cuda' or 'cpu'.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed for reproducibility.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    # --- Data IO
    parser.add_argument("--train", default="data/splits/train.jsonl")
    parser.add_argument("--val", default="data/splits/val.jsonl")
    parser.add_argument("--test", default="data/splits/test.jsonl")
    parser.add_argument("--label_map", default="data/processed/label_map.json")

    # --- Dataloading / Sequence length
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    # --- Optim / training
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")

    # --- Class imbalance
    parser.add_argument("--pos_weight", default=None,
                        help="Optional: 'balanced' or path to .npy of shape [K].")
    
    # --- Thresholds / Evaluation
    parser.add_argument("--threshold_policy", default="fixed",
                        choices=["fixed", "global_val", "per_label_val", "use_saved"])
    parser.add_argument("--thr", type=float, default=0.5)
    parser.add_argument("--min_pos", type=int, default=10)

    # --- Artifact location
    parser.add_argument("--models_dir", default="models")
    parser.add_argument("--metrics_csv", default="reports/baseline_metrics.csv")

    # --- Preset (applies to model build hparams)
    parser.add_argument("--preset", default="strong",
                        choices=["fast", "strong", "max"],
                        help="Quick template of model hparams (overridable by explicit flags).")

    # --- CNN hparams (map to TextCNNModel.build kwargs)
    g_cnn = parser.add_argument_group("CNN hparams")
    g_cnn.add_argument("--cnn_emb_dim", type=int, default=None)
    g_cnn.add_argument("--cnn_num_filters", type=int, default=None)
    g_cnn.add_argument("--cnn_kernel_sizes", type=str, default=None,
                       help='Comma list, e.g., "2,3,4,5"')
    g_cnn.add_argument("--cnn_p_drop", type=float, default=None)
    g_cnn.add_argument("--cnn_emb_drop", type=float, default=None)
    g_cnn.add_argument("--cnn_conv_drop", type=float, default=None)

    # --- BiGRU hparams (map to BiGRUModel.build kwargs)
    g_gru = parser.add_argument_group("BiGRU hparams")
    g_gru.add_argument("--gru_emb_dim", type=int, default=None)
    g_gru.add_argument("--gru_hid", type=int, default=None)
    g_gru.add_argument("--gru_layers", type=int, default=None)
    g_gru.add_argument("--gru_p_drop", type=float, default=None)
    g_gru.add_argument("--gru_emb_drop", type=float, default=None)
    g_gru.add_argument("--gru_gru_drop", type=float, default=None)

    return parser

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse args for external calling (texts) or CLI.

    Parameters
    ----------
    argv : list of str, optional
        Args list; None to read from sys.argv.
    
    Returns
    -------
    argparse.Namespace
    """
    parser = build_parser()
    return parser.parse_args(argv)

# =============================================================================
# Runtime / Data helpers
# =============================================================================
def configure_runtime(args: argparse.Namespace) -> Tuple[logging.Logger, torch.device, Path, Path]:
    """
    Configure logger, seed, device, and artifact dirs.

    Returns
    -------
    logger  : logging.Logger
    device  : torch.device
    models_dir : Path
    metrics_csv : Path
    """

    logger = setup_logger(name=f"{args.model}.{args.run}")
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    set_global_seed(int(args.seed))

    req_dev = str(args.device)
    if req_dev.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("[RUNTIME] CUDA requested but not available; falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(req_dev)
    
    models_dir = Path(args.models_dir) / args.run
    models_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = Path(args.metrics_csv)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[RUNTIME] device={device}, torch={torch.__version__}")
    logger.info(f"[PATH] models_dir={to_relpath(models_dir)}, metrics_csv={to_relpath(metrics_csv)}")
    return logger, device, models_dir, metrics_csv

def load_splits(args: argparse.Namespace, 
                logger: logging.Logger,
) -> Tuple[List[str], np.ndarray, List[str], Optional[np.ndarray], List[str], Optional[np.ndarray], Dict[str, int]]:
    """
    Load train/val/test splits and label map, PHI-safe.

    Returns
    -------
    tr_texts, Y_tr, va_texts, Y_va, te_texts, Y_te, label_map.

    Notes
    -----
    - Y_* must be np.ndarray [N, K], dtype {uint8/bool}, consistent K across splits.
    - Do NOT log raw text; only counts/shapes.
    """

    tr_texts, Y_tr, _meta_tr, label_map = prepare_split(
        args.train, label_map_path=args.label_map, logger=logger
    )

    if Path(args.val).exists():
        va_texts, Y_va, _meta_va, _ = prepare_split(
            args.val, label_map_path=args.label_map, logger=logger
        )
    else:
        va_texts, Y_va = [], None
    
    if Path(args.test).exists():
        te_texts, Y_te, _meta_te, _ = prepare_split(
            args.test, label_map_path=args.label_map, logger=logger
        )
    else:
        te_texts, Y_te = [], None

    assert Y_tr.ndim == 2, f"[LOAD_SPLIT] Y_train must be 2D; got ndim={Y_tr.ndim}"
    assert (Y_tr.dtype in (np.bool_, bool, np.uint8)), \
        f"[LOAD_SPLIT] Y_train dtype must be bool/uint8; got {Y_tr.dtype}"
    
    K = int(Y_tr.shape[1])
    if Y_va is not None:
        assert Y_va.shape[1] == K, f"[LOAD_SPLIT] Val labels must have same K={K}"
    if Y_te is not None:
        assert Y_te.shape[1] == K, f"[LOAD_SPLIT] Test labels must have same K={K}"
    
    logger.info(f"[LOAD_SPLIT] N_tr={len(tr_texts)}, N_va={len(va_texts)}, N_te={len(te_texts)}, K={K}")    
    return tr_texts, Y_tr, va_texts, Y_va, te_texts, Y_te, label_map

def build_vocab_from_train(
        train_texts: List[str],
        *,
        tok: Tokenizer,
        max_size: int = 50_000,
        min_freq: int = 2,
) -> Vocab:
    """
    Build Vocab from train texts only (deterministic).

    Parameters
    ----------
    train_texts : list[str]
        De-identified notes for training.
    tok : Tokenizer
        PHI-safe tokenizer instance.
    max_size : int
        Maximum vocabulary size (including special tokens).
    min_freq : int
        Minimum frequency to keep a token.
    
    Returns
    -------
    Vocab
        With attributes: pad_idx: int, unk_idx: int; __len__ -> vocab_size.
    """
    if not isinstance(train_texts, list) or len(train_texts) == 0:
        raise ValueError("[VOCAB] train_texts must be non-empty list[str].")
    if max_size < 1000:
        raise ValueError(f"[VOCAB] max_size too small: {max_size}")
    if min_freq < 1:
        raise ValueError(f"[VOCAB] min_freq must be >= 1; got {min_freq}.")
    
    # --- Count tokens over train ---
    ctr = Counter()
    for s in train_texts:
        toks = tok(s)
        ctr.update(toks)

    # --- Build vocab from Counter ---
    vocab = Vocab(max_size=max_size, min_freq=min_freq, specials=["<PAD>", "<UNK>"])
    vocab.fit_from_counter(ctr)

    assert isinstance(vocab.pad_idx, int) and isinstance(vocab.unk_idx, int), \
        f"[VOCAB] pad_idx/unk_idx must be ints."
    assert vocab.pad_idx != vocab.unk_idx, "[VOCAB] pad_idx and unk_idx must be differ."
    assert len(vocab) <= max_size, f"[VOCAB] size={len(vocab)} exceed max_size={max_size}."
    
    return vocab
    

def make_dataloader(
        texts: List[str],
        Y: np.ndarray,
        *,
        tok: Tokenizer,
        vocab: Vocab,
        max_len: int,
        batch_size: int,
        num_workers: int,
        for_rnn: bool
) -> DataLoader:
    """
    Create DataLoader yielding dicts needed by CNN/BiGRU.

    Parameters
    ----------
    texts : list[str]
    Y : np.ndarray
        Multi-hot labels [N, K] dtype=uint8/bool.
    for_rnn : bool
        True -> sort_for_rnn & return_index for BiGRU; False for CNN.

    Returns
    -------
    torch.utils.data.DataLoader
        Batch dict keys:
        - 'tokens' : torch.int64 [B, L]
        - 'lengths': torch.int64 [B]
        - 'labels' : torch.float32 [B, K]
        - 'unsort_idx' : torch.int64 [B] (only when for_rnn=True)
    """

    ds = NotesDataset(texts, Y, tok=tok, vocab=vocab, max_len=max_len)

    collate_fn = partial(
        pad_collate,
        pad_idx=vocab.pad_idx,
        sort_for_rnn=for_rnn,
        return_index=for_rnn,
        return_labels=True
    )

    gen = torch.Generator()
    gen.manual_seed(42)

    if num_workers > 0:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
            generator=gen,
            persistent_workers=True if num_workers > 0 else False,
            prefetch_factor=2 if num_workers > 0 else None,
        )
    else:
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
            generator=gen,
        )

    d = next(iter(dl))
    assert d["tokens"].dtype == torch.int64, f"[DATALOADER] tokens dtype must be int64; got {d['tokens'].dtype}."
    assert d["lengths"].dtype == torch.int64, f"[DATALOADER] lengths dtype must be int64; got {d['lengths'].dtype}."
    assert d['labels'].dtype == torch.float32, f"[DATALOADER] labels dtype must be float32; got {d['labels'].dtype}."
    if for_rnn:
        assert "unsort_idx" in d, "[DATALOADER] missing 'unsort_idx' for RNN mode."
    assert d["tokens"].shape[0] == d["labels"].shape[0], \
        f"[DATALOADER] batch B mismatch: tokens B={d['tokens'].shape[0]} vs labels B={d['labels'].shape[0]}."
    
    return dl

# =============================================================================
# Model hparams presets
# =============================================================================

def _parse_tuple_of_ints(csv: Optional[str]) -> Optional[tuple]:
    """
    Convert CSV like '3,4,5' -> (3, 4, 5). Returns None if csv is None.

    Returns
    -------
    tuple or None
    """
    if csv is None:
        return None
    vals = tuple(int(x.strip()) for x in csv.split(",") if x.strip())
    assert len(vals) >= 1 and all(v > 0 for v in vals), f"[HPARAMS] invalid tuple: {csv}"
    return vals

def preset_defaults(model: str, preset: str) -> dict:
    """
    Return default hparams by model & preset (can be overridden by CLI flags). 

    Notes
    -----
    - seq_len = 512, batch_size=64 (AMP on).
    - CNN uses wider filter bank; BiGRU uses 2 layers + dropout.
    """
    if model == "cnn":
        if preset == "fast":
            return dict(emb_dim=200, num_filters=128, kernel_sizes=(3, 4, 5),
                        p_drop=0.4, emb_drop=0.1, conv_drop=0.0)
        if preset == "max":
            return dict(emb_dim=300, num_filters=512, kernel_sizes=(2, 3, 4, 5, 7),
                        p_drop=0.6, emb_drop=0.15, conv_drop=0.1)
        return dict(emb_dim=300, num_filters=256, kernel_sizes=(2, 3, 4, 5),
                        p_drop=0.5, emb_drop=0.1, conv_drop=0.05)
    
    # BiGRU
    if preset == "fast":
        return dict(emb_dim=200, hid=256, layers=1, p_drop=0.4, emb_drop=0.1, gru_drop=0.1)
    if preset == "max":
        return dict(emb_dim=300, hid=384, layers=2, p_drop=0.5, emb_drop=0.15, gru_drop=0.3)
    return dict(emb_dim=300, hid=256, layers=2, p_drop=0.5, emb_drop=0.1, gru_drop=0.2)

def parse_model_hparams(args: argparse.Namespace) -> dict:
    """
    Merge preset defaults with explicit CLI overrides, outputting kwargs
    that match the target model's build(...) signature.

    Returns
    -------
    dict
        For CNN: {emb_dim,num_filters,kernel_sizes,p_drop,emb_drop,conv_drop}
        For BiGRU: {emb_dim,hid,layers,p_drop,emb_drop,gru_drop}

    Notes
    -----
    - Explicit CLI flags override preset values.
    """

    base = preset_defaults(args.model, args.preset)

    if args.model == "cnn":
        ks = _parse_tuple_of_ints(args.cnn_kernel_sizes)
        out = dict(
            emb_dim=args.cnn_emb_dim if args.cnn_emb_dim is not None else base["emb_dim"],
            num_filters=args.cnn_num_filters if args.cnn_num_filters is not None else base["num_filters"],
            kernel_sizes=ks if ks is not None else base["kernel_sizes"],
            p_drop=args.cnn_p_drop if args.cnn_p_drop is not None else base["p_drop"],
            emb_drop=args.cnn_emb_drop if args.cnn_emb_drop is not None else base["emb_drop"],
            conv_drop=args.cnn_conv_drop if args.cnn_conv_drop is not None else base.get("conv_drop", 0.0),
        )
        return out
    
    out = dict(
        emb_dim=args.gru_emb_dim if args.gru_emb_dim is not None else base["emb_dim"],
        hid=args.gru_hid if args.gru_hid is not None else base["hid"],
        layers=args.gru_layers if args.gru_layers is not None else base["layers"],
        p_drop=args.gru_p_drop if args.gru_p_drop is not None else base["p_drop"],
        emb_drop=args.gru_emb_drop if args.gru_emb_drop is not None else base["emb_drop"],
        gru_drop=args.gru_gru_drop if args.gru_gru_drop is not None else base["gru_drop"],
    )
    return out

# =============================================================================
# Model init / class imbalance
# =============================================================================
def init_model_by_name(
        name: str,
        *,
        vocab_size: int,
        num_labels: int,
        pad_idx: int,
        device: torch.device,
        logger: logging.Logger,
        hparams: dict,
):
    """
    Instantiate model with hyperparameters and move to device.

    Parameters
    ----------
    name : {"cnn","bigru"}
        Which model to instantiate.

    Returns
    -------
    model : TextCNNModel | BiGRUModel
        Exposes: build(...), to_device(...), fit_loader(...),
        predict_proba_loader(...), save(dir, protocol, logger)
    """
    if name == "cnn":
        m = TextCNNModel().build(vocab_size=vocab_size, num_labels=num_labels, pad_idx=pad_idx, **hparams)
    else:
        m = BiGRUModel().build(vocab_size=vocab_size, num_labels=num_labels, pad_idx=pad_idx, **hparams)
    
    m.to_device(device, logger=logger)
    logger.info(f"[MODEL] {name} cfg={hparams}")
    return m

# =============================================================================
# Train / Evaluate — pos_weight forced to None (implement later)
# =============================================================================

def train_model(
        model,
        dl_tr: DataLoader,
        dl_va: Optional[DataLoader],
        *,
        Y_va: Optional[np.ndarray],
        args: argparse.Namespace,
        models_dir: Path,
        logger: logging.Logger,
        pos_weight: Optional[torch.Tensor] = None
) -> None:
    """
    Fit the model via model.fit_loader.

    Notes
    -----
    - y_val is required when threshold_policy in {"global_val","per_label_val"}.
    """
    if args.threshold_policy in {"global_val", "per_label_val"} and Y_va is None:
        raise ValueError("[TRAIN] threshold_policy requires --val split with labels.")
        
    train_cfg = dict(
        run=args.run,
        thr=float(args.thr),
        threshold_policy=str(args.threshold_policy),
        min_pos=int(args.min_pos),
        epochs=int(args.epochs),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        grad_accum=int(args.grad_accum),
        clip=float(args.clip),
        amp=bool(args.amp),
        models_dir=models_dir,
        pos_weight=pos_weight,
        y_val=Y_va if (dl_va is not None and args.threshold_policy in {"global_val", "per_label_val"}) else None,
    )

    model.fit_loader(dl_tr, dl_val=dl_va, **train_cfg)
    model.save(models_dir, protocol=4, logger=logger)


def evaluate_split(
        split_name: str,
        model,
        texts: List[str],
        Y: np.ndarray,
        *,
        tok: Tokenizer,
        vocab: Vocab,
        args: argparse.Namespace,
        logger: logging.Logger,
        metrics_csv: Path,
        models_dir: Path,
        for_rnn: bool,
        threshold_policy: Optional[str] = None
) -> None:
    """
    Run predict_proba on a split and append metrics row via evaluate_and_save.

    Notes
    -----
    P is np.float32 [N, K] with scores in [0, 1].
    When testing, set --threshold-policy use_saved to reuse tuned thresholds.
    """

    if not texts:
        return
    
    dl = make_dataloader(texts, Y, tok=tok, vocab=vocab,
                         max_len=args.max_len, batch_size=args.batch_size,
                         num_workers=args.num_workers, for_rnn=for_rnn)
    P = model.predict_proba_loader(dl)
    assert P.shape == Y.shape, f"[EVAL:{split_name}] P {getattr(P,'shape', None)} vs Y {Y.shape}"

    policy = threshold_policy if threshold_policy is not None else args.threshold_policy
    evaluate_and_save(
        run_name=args.run, split_name=split_name,
        y_true=Y, y_scores=P,
        thr=float(args.thr), threshold_policy=str(policy),
        metrics_csv=metrics_csv, min_pos=int(args.min_pos),
        thresholds_dir=models_dir, logger=logger
    )


# =============================================================================
# Orchestration
# =============================================================================

def run(args: argparse.Namespace) -> None:
    """
     Orchestrate end-to-end pipeline.
    """
    logger, device, models_dir, metrics_csv = configure_runtime(args)

    # --- Load data ---
    tr_texts, Y_tr, va_texts, Y_va, te_texts, Y_te, label_map = load_splits(args, logger)
    K = int(Y_tr.shape[1])

    # --- Tokenizer & Vocab
    tok = Tokenizer(anonymize_numbers=True)
    vocab = build_vocab_from_train(tr_texts, tok=tok, max_size=50_000, min_freq=2)
    pad_idx = vocab.pad_idx
    V = len(vocab)

    # --- Class imbalance ---
    pos_w = compute_pos_weight(getattr(args, "pos_weight", None), Y_tr, logger)
    if pos_w is not None:
        pos_w = pos_w.to(device)
        logger.info(
            f"[IMB] Using pos_weight (K=%d) range=[%.2f, %.2f]",
            pos_w.numel(), float(pos_w.min()), float(pos_w.max())
        )
    else:
        logger.info("[IMB] pos_weight disabled (None).")

    # --- DataLoaders
    dl_tr = make_dataloader(tr_texts, Y_tr, tok=tok, vocab=vocab,
                            max_len=args.max_len, batch_size=args.batch_size,
                            num_workers=args.num_workers, for_rnn=(args.model=="bigru"))
    dl_va = make_dataloader(va_texts, Y_va, tok=tok, vocab=vocab,
                            max_len=args.max_len, batch_size=args.batch_size,
                            num_workers=args.num_workers, for_rnn=(args.model=="bigru")) if va_texts else None
    
    # --- Model hparams ---
    hparams = parse_model_hparams(args)
    logger.info(f"[HPARAMS] {hparams}")

    # --- Model ---
    model = init_model_by_name(args.model,
                               vocab_size=V, num_labels=K, pad_idx=pad_idx,
                               device=device, logger=logger, hparams=hparams)
    
    # --- Train ---
    train_model(model, dl_tr, dl_va, Y_va=Y_va, args=args,
                models_dir=models_dir, logger=logger, pos_weight=pos_w)
    
    # --- Evaluate ---
    if va_texts:
        evaluate_split("val", model, va_texts, Y_va, tok=tok, vocab=vocab,
                       args=args, logger=logger, metrics_csv=metrics_csv,
                       models_dir=models_dir, for_rnn=(args.model=="bigru"))
    if te_texts:
        if args.threshold_policy in {"global_val", "per_label_val"}:
            test_policy = "use_saved"
        else:
            test_policy = args.threshold_policy

        evaluate_split("test", model, te_texts, Y_te, tok=tok, vocab=vocab,
                    args=args, logger=logger, metrics_csv=metrics_csv,
                    models_dir=models_dir, for_rnn=(args.model=="bigru"), threshold_policy=test_policy)
    
    return


def main() -> None:
    """
    CLI entry point.
    """
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()