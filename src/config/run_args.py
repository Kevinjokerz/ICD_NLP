"""
run_args.py — Shared run/eval/train argument dataclasses (PHI-safe)

Purpose
-------
Centralize shared args across models (TF-IDF, BiGRU, BERT, ...).

Conventions
-----------
- Frozen dataclasses for immutability.
- No PHI, relative paths only.

Outputs
-------
- CommonArgs, EvalArgs, TrainArgs instances
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any, Tuple


@dataclass(frozen=True)
class CommonArgs:
    """Common run-time configuration."""
    run: str = "run_v1"
    train: str = "data/splits/train.jsonl"
    val: Optional[str] = "data/splits/val.jsonl"
    test: Optional[str] = "data/splits/test.jsonl"
    label_map: str = "data/processed/label_map.json"
    protocol: int = 4
    seed: int = 42
    log_level: str = "INFO"


@dataclass(frozen=True)
class EvalArgs:
    """Evaluation configuration (fixed threshold for fairness)."""
    thr: float = 0.5                    # fixed threshold policy
    threshold_policy: str = "fixed"     # {"fixed", "global_val", "per_label_val"}
    ks: Tuple[int, int] = (8, 15)       # Precision@k (reporting handled in evaluator)
    min_pos: int = 10                   # per-label tuning guard


@dataclass(frozen=True)
class TrainArgs:
    epochs: int = 6
    lr: float = 2e-3
    weight_decay: float = 0.0
    amp: bool = True
    grad_accum: int = 1
    clip: Optional[float] = 1.0
    log_every: int = 50
    pos_weight: Optional[Any] = None
    models_dir: str = "models"

def from_kwargs(**kw) -> Tuple[CommonArgs, EvalArgs, TrainArgs]:
    """
    Construct validated (CommonArgs, EvalArgs, TrainArgs) from kwargs.

    Notes
    - Only coerce shared fields; model-specific args live in each model file.
    """

    c = CommonArgs(
        run = str(kw.get("run", CommonArgs.run)),
        train = str(kw.get("train", CommonArgs.train)),
        val = kw.get("val", CommonArgs.val),
        test = kw.get("test", CommonArgs.test),
        label_map = str(kw.get("label_map", CommonArgs.label_map)),
        protocol = int(kw.get("protocol", CommonArgs.protocol)),
        seed = int(kw.get("seed", CommonArgs.seed)),
        log_level = str(kw.get("log_level", CommonArgs.log_level)),
    )

    e = EvalArgs(
        thr = float(kw.get("thr", EvalArgs.thr)),
        threshold_policy = str(kw.get("threshold_policy", EvalArgs.threshold_policy)),
        ks = tuple(kw.get("ks", EvalArgs.ks)),
        min_pos = int(kw.get("min_pos", EvalArgs.min_pos)),
    )

    t = TrainArgs(
        epochs = int(kw.get("epochs", TrainArgs.epochs)),
        lr = float(kw.get("lr", TrainArgs.lr)),
        weight_decay = float(kw.get("weight_decay", TrainArgs.weight_decay)),
        amp = bool(kw.get("amp", TrainArgs.amp)),
        grad_accum = int(kw.get("grad_accum", TrainArgs.grad_accum)),
        clip = kw.get("clip", TrainArgs.clip),
        log_every = int(kw.get("log_every", TrainArgs.log_every)),
        pos_weight = kw.get("pos_weight", TrainArgs.pos_weight),
        models_dir = str(kw.get("models_dir", kw.get("model_dir", TrainArgs.models_dir))),
    )

    assert t.epochs >= 1 and t.grad_accum >= 1
    assert 1e-7 <= t.lr <= 1
    assert e.threshold_policy in {"fixed", "global_val", "per_label_val"}
    assert 0.0 < e.thr < 1.0
    assert isinstance(e.ks, tuple) and len(e.ks) == 2 and all(int(k) > 0 for k in e.ks), f"[RUN_ARGS] ks must be a tuple of two positive ints; got {e.ks}"

    return (c, e, t)

__all__ = ["CommonArgs", "EvalArgs", "TrainArgs", "from_kwargs"]