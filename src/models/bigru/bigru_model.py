"""
bigru_model.py - BiGRU text classifier

Purpose
-------
BiGRU encoder + linear head -> K logits per notes. Train via DataLoader

Conventions
-----------
- PAD id = 0; right-padding; lengths provided (and usually sorted desc for pack())
- Loss: BCEWithLogitsLoss (trainer/hook)
- Determinism: seed set in runner.

Inputs
------
tokens  : LongTensor [B, L]
lengths : LongTensor [B]
labels  : Float/UInt8 [B, K] (train)

Outputs
-------
logits  : FloatTensor [B, K]
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Iterable, List
from pathlib import Path
import logging
import numpy as np
from tqdm.auto import tqdm
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.metrics import compute_f1s
from src.common.thresholds import (
    tune_global_thr, tune_per_label_thr,
    save_thresholds, apply_thresholds
)
from src.utils import set_seed as to_relpath
from src.config.run_args import from_kwargs


# =========================
# Config dataclasses
# =========================


@dataclass(frozen=True)
class BiGRUArgs:
    """
    Configuration for BiGRU encoder.

    Parameters
    ----------
    vocab_size  : int
        Size of token vocabulary (including PAD/UNK/etc.)
    num_labels  : int
        Number of output labels K.
    pad_idx     : int, default=0
        Index used for padding tokens.
    emb_dim     : int, default=200
        Embedding dimension.
    hid         : int, default=256
        Hidden size for each direction in GRU.
    layers      : int, default=1
        Number of stacked GRU layers.
    p_drop      : float, default=0.5
        Dropout probability before the linear head.
    """
    vocab_size: int
    num_labels: int
    pad_idx : int = 0
    emb_dim: int = 200
    hid : int = 256
    layers: int = 1
    p_drop: float = 0.5
    emb_drop: float = 0.1
    gru_drop: float = 0.2


# -----------------------------------------------------------------------------
# Encoder
# -----------------------------------------------------------------------------
class BiGRUEncoder(nn.Module):
    """
    BiGRU encoder with mean-free pooling over final states (concat).

    Notes
    -----
    - Uses `pack_padded_sequence` for efficiency on variable lengths.
    - Exposes a final fully connected layerr to produce K logits.
    - Keep `enforce_sorted=True` for maximum speed if upstream collate sorts.
    - All tensor stay in fp32; AMP handled by caller.
    """
    def __init__(self, cfg: BiGRUArgs) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.emb_drop = nn.Dropout(cfg.emb_drop)
        self.gru = nn.GRU(
            cfg.emb_dim, 
            cfg.hid, 
            num_layers=cfg.layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=(cfg.gru_drop if cfg.layers > 1 else 0.0),
        )
        self.drop = nn.Dropout(cfg.p_drop)
        self.fc = nn.Linear(2 * cfg.hid, cfg.num_labels)
    
    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        tokens  : torch.LongTensor
            Shape [B, L], token ids (PAD = cfg.pad_idx).
        lengths : torch.LongTensor
            Shape [B], actual sequence lengths (sorted desc if enforce_sorted=True).
        
        Returns
        -------
        torch.FloatTensor
            Logits with shape [B, K]. (No `sigmoid` inside.)
        """
        assert isinstance(tokens, torch.Tensor), "[BiGRU:forward] `tokens` must be a torch.Tensor (LongTensor ids)."
        assert isinstance(lengths, torch.Tensor), "[BiGRU:forward] `lengths` must be a torch.Tensor (LongTensor sizes)."

        assert tokens.dtype == torch.long, f"[BiGRU:forward] tokens.dtype must be torch.long (int64); got {tokens.dtype}"
        assert lengths.dtype == torch.long, f"[BiGRU:forward] lengths.dtype must be torch.long (int64); got {lengths.dtype}"

        assert tokens.ndim == 2, f"[BiGRU:forward] `tokens` must be 2D [B, L]; got dim={tokens.ndim} shape={tuple(tokens.shape)}."
        assert lengths.ndim ==1, f"[BiGRU:forward] `lengths` must be 1D [B]; got dim={lengths.ndim} shape={tuple(lengths.shape)}."

        B, L = tokens.shape
        assert lengths.numel() == B, f"[BiGRU:forward] batch size mismatch: tokens B={B} but lengths has {lengths.numel()} elements."

        min_len = int(lengths.min().item())
        max_len = int(lengths.max().item())
        assert min_len >= 1, f"[BiGRU:forward] all lengths must be >= 1; found min={min_len}."
        assert max_len <= L, f"[BiGRU:forward] lengths must be <= sequence length L={L}; found max={max_len}."

        emb_dev = self.emb.weight.device
        assert tokens.device == emb_dev, (
            f"[BiGRU:forward] device mismatch: tokens on {tokens.device}, embedding on {emb_dev}. "
            f"Call `model.to_device(device)` and move batches with `.to(model.device)`."
        )

        assert bool(torch.all(lengths[:-1] >= lengths[1:]).item()), (
            "[BiGRU:forward] `lengths` must be sorted in descending order when enforce_sorted=True. "
            "Either sort in `collate_fn` or set enforce_sorted=False."
        )

        x = self.emb_drop(self.emb(tokens))
        x = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=True
        )
        _, h_n = self.gru(x)
        h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        out = self.fc(self.drop(h_last))

        assert out.shape == (B, self.cfg.num_labels), f"[BiGRU:forward] logits shape must be (B, K)=({B}, {self.cfg.num_labels}); got {tuple(out.shape)}"
        return out
    
# -----------------------------------------------------------------------------
# Model wrapper (runner‑friendly)
# -----------------------------------------------------------------------------
class BiGRUModel(nn.Module):
    """
    Minimal wrapper exposing build / to_device / fit_loader / predict_proba_loader / save / load.

    This class keeps runner-friendly hooks and **does not** log PHI. It follows
    the project `ProbModel` spirit (fit/predict_proba), but operates on
    DataLoaders to avoid duplicating dataset/featurization logic.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: Optional[BiGRUEncoder] = None
        self.cfg: Optional[BiGRUArgs] = None
        self.K: Optional[int] = None
        self.device: torch.device = torch.device("cpu")
        self.best_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Build / device
    # ------------------------------------------------------------------

    def build(self, *, vocab_size: int, num_labels: int, pad_idx: int, **hparams: Any) -> "BiGRUModel":
        """
        Instantiate encoder with config.

        Parameters
        ----------
        vocab_size  : int
            Vocabulary size.
        num_labels  : int
            Number of labels (K) for the classification head.
        pad_idx     : int
            Index used for PAD in tokens.
        **hparams   : Any
            Additional overrides for :class:`BiGRUArgs` fields.
        
        Returns
        -------
        BiGRUModel
            Self, for chaining.
        """
        assert vocab_size > 0 and num_labels > 0
        assert 0 <= pad_idx < vocab_size, f"[BiGRU:build] pad_idx={pad_idx} must be in [0, {vocab_size - 1}]"
        self.cfg = BiGRUArgs(vocab_size=vocab_size, num_labels=num_labels, pad_idx=pad_idx, **hparams)
        self.net = BiGRUEncoder(self.cfg)
        self.K = int(num_labels)

        assert isinstance(self.net, nn.Module), f"[BiGRU:build] net must be nn.Module."
        return self
    
    def to_device(self, device: torch.device, logger: Optional[logging.Logger] = None) -> "BiGRUModel":
        """
        Move model to device and remember it (cpu/cuda).

        Parameters
        ----------
        device  : torch.device
            Target device.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        assert self.net is not None, "[BiGRU:to_device] call build() before to_device()."

        device = torch.device(device)
        self.device = device
        super().to(device)
        pdev = next(self.parameters()).device

        if pdev.type == "cuda":
            idx = pdev.index if pdev.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory / (1024**3)
            logger.info(f"[BiGRU:device] {pdev} | {name} | CC {cc} | {mem_gb:.1f} GB")
        else:
            logger.warning(f"[BiGRU:device] {pdev} (CPU)")

        same_index = (device.index is None) or (pdev.index == device.index)
        assert (pdev.type != "cuda") or same_index, \
            f"[BiGRU:to_device] expected {device}, got {pdev}"
        return self
    
    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_proba_loader(self, dl: Iterable[dict]) -> np.ndarray:
        """
        Score an iterable of batches and return probabilities.

        Parameters
        ----------
        dl : Iterable[dict]
            DataLoader yielding dicts with keys {"tokens", "lengths"[, "unsort_idx"]}.
            See `pad_collate(sort_for_rnn=True, return_index=True)` for more details.

        Returns
        -------
        np.ndarray
            Probabilities with shape (N_total, K) float32 in [0, 1]. Rows are
            automatically unsorted back to original order if `unsort_idx` is 
            provided by the collate.
        """
        assert self.net is not None, "[BiGRU:predict] call build()/to_device() before predict."
        self.eval()
        prob_list: List[torch.Tensor] = []

        for i, batch in enumerate(dl):
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)
            logits = self.net(tokens, lengths)

            if i == 0:
                assert logits.shape[1] == int(self.K or logits.shape[1]), f"[BiGRU:predict] logits K={logits.shape[1]} != self.K={self.K}"

            probs = torch.sigmoid(logits)
            if "unsort_idx" in batch:
                idx = batch["unsort_idx"]
                
                if not torch.is_tensor(idx):
                    raise TypeError("[BiGRU:predict] `unsort_idx` must be a torch.LongTensor.")
                
                idx = idx.to(self.device, non_blocking=True)
                assert idx.dtype == torch.long and idx.shape[0] == probs.shape[0]
                probs = probs.index_select(0, idx)
            prob_list.append(probs.cpu())
        
        if not prob_list:
            K = int(self.K or (self.cfg.num_labels if self.cfg else 0))
            return np.zeros((0, K), dtype=np.float32)
        
        return torch.cat(prob_list, dim=0).numpy().astype(np.float32)
    
    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit_loader(self, dl_train: Iterable[dict], dl_val: Optional[Iterable[dict]] = None, **train_cfg) -> "BiGRUModel":
        """
        Train with AdamW + BCEWithLogitsLoss (+ AMP) over DataLoaders.
        Threshold selection handled via EvalArgs policy.

        Parameters
        ----------
        dl_train    : Iterable[dict]
            Training DataLoader yielding {tokens, lengths, labels}
        dl_val      : Iterable[dict]
            Validation DataLoader (same schema) for early-stopping.
        **train_cfg : Any
            Hyperparameters: {epochs: int, lr: float, weight_decay: float,
            amp: bool, clip: float|None, grad_accum: int >= 1, run_name: str,
            pos_weight: Optional[torch.Tensor] (shape [K])}
        
        Returns
        -------
        BiGRUModel
            Self after training. Best state is cached in memory and used by `save()`.
        
        Notes
        -----
        - Thresholding/metrics are handled outside by shared evaluators.
        - To reproduce: pin seeds before calling; log versions in runner.
        """

        common, evalargs, trainargs = from_kwargs(**train_cfg)

        # --- Hyperparameters / default ---
        amp = bool(trainargs.amp) and (self.device.type == "cuda")
        lr = float(trainargs.lr)
        wd = float(trainargs.weight_decay)
        grad_accum = int(trainargs.grad_accum)
        clip_val = trainargs.clip
        epochs = int(trainargs.epochs)
        log_every = int(trainargs.log_every)
        pos_weight = trainargs.pos_weight

        # --- Selection Policy ---
        threshold_policy = str(evalargs.threshold_policy)
        thr_fixed = float(evalargs.thr)
        min_pos = int(evalargs.min_pos)
        run_name = str(common.run)
        models_dir = str(trainargs.models_dir)

        assert threshold_policy in {"fixed", "global_val", "per_label_val"}, f"[BiGRU:fit_loader] unknown threshold_policy={threshold_policy}"
        assert 0.0 < thr_fixed < 1.0, "[BiGRU:fit_loader] thr must be in (0, 1)"

        # --- Setup ---
        assert self.net is not None, "[BiGRU:fit] call build()/to_device() before fit."
        optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler(enabled=amp)

        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if pos_weight.device != self.device:
                pos_weight = pos_weight.to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        assert pos_weight is None or pos_weight.numel() == int(self.K), \
            f"[BiGRU:fit_loader] pos_weight size {getattr(pos_weight, 'numel', lambda: -1)()} != K={self.K}"

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=3) if dl_val is not None else None

        best_metric = -1.0
        self.best_state = None

        needs_val_target = (dl_val is not None) and (threshold_policy in {"global_val", "per_label_val"})
        if needs_val_target and ("y_val" not in train_cfg):
            raise ValueError(
                "[BiGRU:fit_loader] y_val is required when threshold_policy in {'global_val', 'per_label_val'}. "
                "Pass y_val=(N_val, K) from runner, or set threshold_policy='fixed'."
            )

        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0.0

            # --- train bar ---
            pbar = tqdm(enumerate(dl_train, start=1), 
                        total=(len(dl_train) if hasattr(dl_train, "__len__") else None),
                        desc=f"train epoch {epoch}/{epochs}", leave=False, dynamic_ncols=True)
            
            optim.zero_grad(set_to_none=True)
            last_step = 0
            for step, batch in pbar:
                # Move batch to device
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True).float()

                # Forward (+AMP)
                with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=amp):
                    logits = self.net(tokens, lengths)
                    loss = criterion(logits, labels)

                scaler.scale(loss / max(1, grad_accum)).backward()
                last_step = step

                if step % grad_accum == 0:
                    if amp:
                        scaler.unscale_(optim)
                    if clip_val is not None:
                        nn.utils.clip_grad_norm_(self.parameters(), max_norm=float(clip_val))
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad(set_to_none=True)
                
                total_loss += float(loss.detach().item())
                if step % log_every == 0:
                    avg = total_loss / step
                    lr0 = optim.param_groups[0]["lr"]
                    pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{lr0:.2e}")
            
            if last_step % grad_accum != 0:
                if amp:
                    scaler.unscale_(optim)
                if clip_val is not None:
                    nn.utils.clip_grad_norm_(self.parameters(), max_norm=float(clip_val))
                scaler.step(optim)
                scaler.update()
                optim.zero_grad(set_to_none=True)
            
            val_metric = None
            if dl_val is not None:
                self.eval()
                vbar = tqdm(dl_val, desc=f"val epoch {epoch}", leave=False, dynamic_ncols=True)

                prob_list: List[np.ndarray] = []
                for batch in vbar:
                    with torch.no_grad():
                        tokens = batch["tokens"].to(self.device, non_blocking=True)
                        lengths = batch["lengths"].to(self.device, non_blocking=True)

                        logits = self.net(tokens, lengths)
                        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                        if "unsort_idx" in batch:
                            probs = probs[batch["unsort_idx"].cpu().numpy()]
                        prob_list.append(probs)
                
                P_val = np.concatenate(prob_list, axis=0) if prob_list else np.zeros((0, int(self.K or 0)), dtype=np.float32)
                Y_val = train_cfg["y_val"]
                if torch.is_tensor(Y_val):
                    Y_val = Y_val.detach().cpu().numpy()
                Y_val = np.asarray(Y_val)
                assert Y_val.ndim == 2 and Y_val.shape[1] == int(self.K), f"[BiGRU:fit_loader] y_val shape invalid: {Y_val.shape}, expected (*, {self.K})"
                if Y_val.dtype != np.uint8:
                    Y_val = (Y_val > 0.5).astype(np.uint8)
                assert P_val.shape == Y_val.shape == (Y_val.shape[0], int(self.K)), f"[BiGRU:fit_loader] val shape mismatch: {P_val.shape} vs {Y_val.shape}"

                if threshold_policy == "fixed":
                    thr_kind, thr_val = "global", thr_fixed
                elif threshold_policy == "global_val":
                    thr_kind, thr_val = "global", float(tune_global_thr(Y_val, P_val)[0])
                elif threshold_policy == "per_label_val":
                    thr_kind, thr_val = "per_label", tune_per_label_thr(Y_val, P_val, min_count=min_pos)[0]

                Yhat_val = apply_thresholds(P_val, thr_val)
                micro, macro = compute_f1s(Y_val, Yhat_val)
                val_metric = float(micro)

                if scheduler is not None and val_metric is not None:
                    scheduler.step(val_metric)
                
                if val_metric is not None and val_metric > best_metric:
                    best_metric = val_metric
                    self.best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                    save_thresholds(models_dir, run_name, kind=thr_kind, thr=thr_val)
                
        if self.best_state is None:
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

        return self
    
    def save(self, path: Path, *, protocol: int = 4, logger: Optional[logging.Logger] = None) -> None:
        """
        Persist checkpoint and config under `path`.

        Artifacts
        ---------
        - `best.ckpt` : torch.save({state_dict, config, K, versions})

        Notes
        -----
        - Uses relative path only; creates parents.
        - Expects `self.best_state` or current `state_dict()`.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        # --- state_dict to CPU (even if currently on CUDA) ---
        sd = self.best_state if self.best_state is not None else self.state_dict()
        sd_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

        # --- config to plain dict ---
        cfg_dict = None
        if self.cfg is not None:
            try:
                cfg_dict = asdict(self.cfg)
            except Exception:
                cfg_dict = dict(self.cfg.__dict__)

        payload = {
            "state_dict": sd_cpu,
            "config": cfg_dict,
            "K": int(self.K or 0),
            "versions": {
                "torch": torch.__version__,
                "numpy": np.__version__,
                "python": sys.version.split()[0],
            },
            "meta": {
                "model_class": type(self).__name__,
                "cfg_class": type(self.cfg).__name__ if self.cfg is not None else None,
            },
        }

        torch.save(payload, out / "best.ckpt", pickle_protocol=protocol)

        logger.info(f"[SAVE] best.ckpt -> {to_relpath(out)} (K={payload['K']})")

    @classmethod
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> "BiGRUModel":
        """
        Reload from `path/best.ckpt` and return a ready model on CPU.

        Returns
        -------
        BiGRUModel
            Model with weights loaded and `K`/`cfg` restored.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        ckpt_path = Path(path) / "best.ckpt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"[LOAD] checkpoint not found: {to_relpath(ckpt_path)}")

        obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        cfg_dict = obj.get("config", None)
        if cfg_dict is None:
            raise ValueError("[LOAD] missing `config` in checkpoint; cannot rebuild model.")
        m = cls().build(**cfg_dict)

        sd = obj.get("state_dict", None)
        if sd is None:
            raise ValueError("[LOAD] missing `state_dict` in checkpoint.")
        
        m.load_state_dict(sd, strict=True)
        saved_K = int(obj.get("K", 0))
        if saved_K <= 0:
            saved_K = int(cfg_dict.get("num_labels", 0))
        assert saved_K == m.K, f"[LOAD] K mismatch: saved {saved_K} vs model {m.K}"

        m.to_device(torch.device("cpu"))
        m.eval()

        vers = obj.get("versions", {})
        vt = vers.get("torch")
        if vt and vt != torch.__version__:
            logger.warning(f"[LOAD] torch version: saved={vt} current={torch.__version__}")
        vn = vers.get("numpy")
        if vn and vn != np.__version__:
            logger.warning(f"[LOAD] numpy version: saved={vn} current={np.__version__}")
        
        logger.info(f"[LOAD] restored from {to_relpath(ckpt_path)} (K={m.K})")
        return m





