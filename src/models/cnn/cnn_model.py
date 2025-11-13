"""
cnn_model.py - TextCNN text classifier (ICD multi-label)

Purpose
-------
CNN encoder + linear head -> K logits. Train via DataLoaders (NeuralBaseModel API).

Conventions
-----------
- PAD id = 0; right padding; lengths optional (ignored).
- Loss: BCEWithLogitsLoss (set in trainer).
- Determinism: global seed fixed by runner; no PHI logs.

Inputs
------
tokens  : LongTensor [B, L]
labels  : Float/UInt8 [B, K] (train)

Outputs
-------
logits  : FloatTensor [B, K]
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Iterable, List, Tuple
from pathlib import Path
import logging
import sys
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.metrics import compute_f1s
from src.common.thresholds import tune_global_thr, tune_per_label_thr, save_thresholds, apply_thresholds
from src.config.run_args import from_kwargs
from src.utils import to_relpath

# =========================
# Config dataclass
# =========================
@dataclass(frozen=True)
class TextCNNArgs:
    """
    Architecture config for TextCNN

    Parameters
    ----------
    vocab_size  : int
        Size of token vocabulary (including PAD/UNK).
    num_labels  : int
        Number of output labels K.
    pad_idx     : int, default=0
        Index used for PAD in tokens.
    emb_dim     : int, default=200
        Embedding dimension.
    num_filters : int, default=100
        Number of filters per kernel size.
    kernel_sizes: Tuple[int, ...], default=(3, 4, 5)
        List of kernel widths for 1D conv.
    p_drop      : float, default=0.5
        Dropout before linear head.
    emb_drop    : float, default=0.1
        Dropout on embeddings.
    conv_drop   : float, default=0.0
        Dropout after pooled features (optional).
    """
    vocab_size: int
    num_labels: int
    pad_idx: int = 0
    emb_dim: int = 200
    num_filters: int = 100
    kernel_sizes: Tuple[int, ...] = (3, 4, 5)
    p_drop: float = 0.5
    emb_drop: float = 0.1
    conv_drop: float = 0.0

# -------------------------
# Encoder
# -------------------------
class TextCNNEncoder(nn.Module):
    """
    Embedding -> Conv1d[k] x m -> ReLU -> global-max-pool -> concat -> dropout -> FC(K).
    """
    def __init__(self, cfg: TextCNNArgs) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.emb_drop = nn.Dropout(cfg.emb_drop)

        with torch.no_grad():
            self.emb.weight[cfg.pad_idx].fill_(0)

        pads = [k // 2 for k in cfg.kernel_sizes]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=cfg.emb_dim, out_channels=cfg.num_filters, kernel_size=k, padding=p)
            for k, p in zip(cfg.kernel_sizes, pads)
        ])

        self.drop = nn.Dropout(cfg.p_drop)
        self.fc = nn.Linear(len(cfg.kernel_sizes) * cfg.num_filters, cfg.num_labels)

        for conv in self.convs:
            nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu")
            nn.init.zeros_(conv.bias)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: Optional[torch.LongTensor] = None) -> torch.Tensor:
        """
        Forward pass of the TextCNN encoder.

        Parameters
        ----------
        tokens : torch.LongTensor, shape [B, L]
            Right-padded token IDs (PAD at `cfg.pad_idx`). `dtype` must be `torch.long`.
        lengths : Optional[torch.LongTensor], shape [B], default=None
            Original (unpadded) sequence lengths. Kept for API parity.
        
        Returns
        -------
        logits : torch.FloatTensor, shape [B, K]
            Unnormalized scores for each of K labels. Apply `torch.sigmoid(logits)` outside
            to obtain probabilities.

        Notes
        -----
        - Pipeline: Embedding -> Conv1d[k] x m -> ReLU -> GlobalMaxPool -> Concat -> Dropout -> Linear(K).
        - Padding policy is "SAME-ish" via `padding = k // 2`, so L' = L for each conv branch.
        - PHI-safety: inputs are token IDs only; no raw text is logged.
        """
        assert tokens.ndim == 2, f"[CNN:forward] tokens must be 2D (got {tokens.ndim}D)"
        assert tokens.dtype == torch.long, f"[CNN:forward] tokens must be LongTensor (got {tokens.dtype})"
        assert tokens.size(1) >= max(self.cfg.kernel_sizes), \
            f"[CNN:forward] L={tokens.size(1)} < max_kernel={max(self.cfg.kernel_sizes)}"
        
        x = self.emb(tokens)
        x = self.emb_drop(x)
        x = x.transpose(1, 2)

        pooled = []
        for conv in self.convs:
            h = F.relu(conv(x))
            m = F.max_pool1d(h, kernel_size=h.shape[-1]).squeeze(-1)
            pooled.append(m)

        z = torch.cat(pooled, dim=1)
        z = F.dropout(z, p=self.cfg.conv_drop, training=self.training)
        logits = self.fc(self.drop(z))

        assert logits.shape == (tokens.size(0), self.cfg.num_labels), (
            f"[CNN:forward] Output shape mismatch: "
            f"got {tuple(logits.shape)}, expected ({tokens.size(0)}, {self.cfg.num_labels})."
        )
        assert logits.dtype in (torch.float16, torch.float32, torch.float64), (
            f"[CNN:forward] `logits` must be floating point, got dtype={logits.dtype}"
        )

        return logits

# -------------------------
# Model wrapper
# -------------------------

class TextCNNModel(nn.Module):
    """
    Runner friendly wrapper exposing: build / to_device / fit_loader / predict_proba_loader / save / load
    """
    def __init__(self, ):
        super().__init__()
        self.net: Optional[TextCNNEncoder] = None
        self.cfg: Optional[TextCNNArgs] = None
        self.K: Optional[int] = None
        self.device: torch.device = torch.device("cpu")
        self.best_state: Optional[Dict[str, Any]] = None


    # ------------------------------------------------------------------
    # Build / device
    # ------------------------------------------------------------------
    def build(self, *, vocab_size: int, num_labels: int, pad_idx: int, **hparams: Any) -> "TextCNNModel":
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
            Additional overrides for :class:`TextCNNArgs` fields.
        
        Returns
        -------
        TextCNNModel
            Self, for chaining.
        """
        assert vocab_size > 0 and num_labels > 0
        assert 0 <= pad_idx < vocab_size, f"[CNN:build] pad_idx={pad_idx} must be in [0, {vocab_size - 1}]"
        self.cfg = TextCNNArgs(vocab_size=vocab_size, num_labels=num_labels, pad_idx=pad_idx, **hparams)
        self.net = TextCNNEncoder(self.cfg)
        self.K = int(num_labels)
        
        assert isinstance(self.net, nn.Module), f"[CNN:build] net must be nn.Module"
        return self
    
    def to_device(self, device: torch.device, logger: Optional[logging.Logger] = None) -> "TextCNNModel":
        """
        Move model to device and remember it (cpu/cuda).

        Parameters
        ----------
        device  : torch.device
            Target device.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        assert self.net is not None, "[CNN:to_device] call build() before to_device()"

        device = torch.device(device)
        self.device = device
        super().to(device)
        pdev = next(self.parameters()).device

        if pdev.type == "cuda":
            idx = pdev.index if pdev.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory / (1024 ** 3)
            logger.info(f"[CNN:device] {pdev} | {name} | CC {cc} | {mem_gb:.1f} GB")
        else:
            logger.warning(f"[CNN:device] {pdev} (CPU)")
        
        same_index = (device.index is None) or (pdev.index == device.index)
        assert (pdev.type != "cuda") or same_index, \
            f"[CNN:to_device] expected {device}, got {pdev}"

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
        assert self.net is not None, "[CNN:predict] call build()/to_device() before predict."
        self.eval()
        prob_list: List[torch.Tensor] = []

        for i, batch in enumerate(dl):
            tokens = batch["tokens"].to(self.device, non_blocking=True)
            lengths = batch["lengths"].to(self.device, non_blocking=True)
            logits = self.net(tokens, lengths)

            if i == 0:
                assert logits.shape[1] == int(self.K or logits.shape[1]), \
                    f"[CNN:predict] logits K={logits.shape[1]} != self.K={self.K}"
            
            probs = torch.sigmoid(logits)
            if "unsort_idx" in batch:
                idx = batch["unsort_idx"]

                if not torch.is_tensor(idx):
                    raise TypeError("[CNN:predict] `unsort_idx` must be torch.LongTensor.")
                
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

    def fit_loader(self, dl_train: Iterable[dict], dl_val: Optional[Iterable[dict]] = None, **train_cfg) -> "TextCNNModel":
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
        TextCNNModel
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

        # --- Select policy ---
        threshold_policy = str(evalargs.threshold_policy)
        thr_fixed = float(evalargs.thr)
        min_pos = int(evalargs.min_pos)
        run_name = str(common.run)
        models_dir = str(trainargs.models_dir)

        assert threshold_policy in {"fixed", "global_val", "per_label_val"}, \
            f"[CNN:fit] unknown threshold_policy={threshold_policy}"
        assert 0.0 < thr_fixed < 1.0, "[CNN:fit] thr must be in (0, 1)"

        # --- Setup ---
        assert self.net is not None, "[CNN:fit] call build()/to_device() before fit."
        optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler(enabled=amp)

        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            if pos_weight.device != self.device:
                pos_weight = pos_weight.to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if pos_weight is not None else nn.BCEWithLogitsLoss()
        assert pos_weight is None or pos_weight.numel() == int(self.K), \
            f"[CNN:fit] pos_weight size {getattr(pos_weight, 'numel', lambda: -1)()} != K={self.K}"
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=2) if dl_val is not None else None

        best_metric = -1.0
        self.best_state = None

        needs_val_target = (dl_val is not None) and (threshold_policy in {"global_val", "per_label_val"})
        if needs_val_target and ("y_val" not in train_cfg):
            raise ValueError(
                "[CNN:fit] y_val is required when threshold_policy in {'global_val', 'per_label_val'}. "
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
                # --- Move batch to device ---
                tokens = batch["tokens"].to(self.device, non_blocking=True)
                lengths = batch["lengths"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                # --- Forward (+AMP) ---
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
            assert Y_val.ndim == 2 and Y_val.shape[1] == int(self.K), f"[CNN:fit] y_val shape invalid: {Y_val.shape}, expected (*, {self.K})"
            if Y_val.dtype != np.uint8:
                Y_val = (Y_val > 0.5).astype(np.uint8)
            assert P_val.shape == Y_val.shape == (Y_val.shape[0], int(self.K)), f"[CNN:fit] val shape mismatch: {P_val.shape} vs {Y_val.shape}"

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
        - Expected `self.best_state` or current `state_dict()`.
        """
        if logger is None:
            logger = logging.getLogger(__name__)
        
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
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
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> "TextCNNModel":
        """
        Reload from `path/best.ckpt` and return a ready model on CPU.

        Returns
        -------
        TextCNNModel
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


