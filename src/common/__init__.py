"""
Helpers Package - shared helpers for all models

Purpose
-------
Shared components for all baseline models (TF-IDF, CNN/GRU, BERT, Longformer).

Submodules
----------
- protocol.py       : Common interface (fit / predict_proba / save /load)
- utils.py          : I/O, evaluation, split loading
- vectorizers.py    : TF-IDF word/char factories
- thresholds.py     : threshold tuning + apply + IO
"""

from .utils import (
    load_label_map, 
    read_split,
    labels_to_multi_hot, 
    evaluate_probs,
    append_metrics_row,
    prepare_split,
    evaluate_and_save,
)

from .metrics import (
    precision_at_k,
    compute_f1s,
)

from .vectorizers import (
    VectorizeConfig,
    make_vectorizers,
    fit_transform_splits,
    save_vectorizers,
    load_vectorizers,
    transform_texts,
)

from .thresholds import (
    apply_thresholds,
    tune_global_thr,
    tune_per_label_thr, 
    save_thresholds,
    load_thresholds,
)

from .textdata import (
    Tokenizer,
    Vocab,
    NotesDataset,
    pad_collate,
)

from .imbalance import (
    compute_pos_weight,
)

__all__ = [

    # --- utils ---
    "load_label_map", 
    "read_split",
    "labels_to_multi_hot", 
    "evaluate_probs",
    "append_metrics_row",
    "prepare_split",
    "evaluate_and_save",

    # vectorizers (public surface)
    "VectorizeConfig", 
    "make_vectorizers",
    "fit_transform_splits", 
    "save_vectorizers", 
    "load_vectorizers",
    "transform_texts",

    # metrics
    "precision_at_k",
    "compute_f1s",

    # thresholds
    "apply_thresholds",
    "tune_global_thr",
    "tune_per_label_thr", 
    "save_thresholds",
    "load_thresholds",

    # textdata
    "Tokenizer",
    "Vocab",
    "NotesDataset",
    "pad_collate",

    # imbalance
    "compute_pos_weight",
]