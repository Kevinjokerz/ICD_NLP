"""
clinical_bert_model.py — ClinicalBERT text classifier (ICD multi-label)

Purpose
-------
BERT-family encoder (e.g., ClinicalBERT) + linear head → K logits for ICD-9 codes.
Trains via DataLoaders, similar interface to TextCNNModel / BiGRUModel.

Conventions
-----------
- Input batches: de-identified notes already tokenized via HF tokenizer.
- Multi-label loss: BCEWithLogitsLoss (set in trainer).
- Determinism: seeds controlled by runner; PHI never logged.

Inputs
------
- Batch dict from DataLoader:
    input_ids      : LongTensor [B, L] (padded)
    attention_mask : LongTensor [B, L] (1 = real token, 0 = pad)
    labels         : FloatTensor/UInt8 [B, K] (train)
    (optional) unsort_idx : LongTensor [B] from collate, for restoring order

Outputs
-------
- logits  : FloatTensor [B, K]
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Any, Dict, Iterable, List, Tuple
from pathlib import Path
import logging
import sys
import time
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer, AutoConfig

from src.common.metrics import compute_f1s
from src.common.thresholds import (
    tune_global_thr,
    tune_per_label_thr,
    save_thresholds,
    apply_thresholds,
)
from src.config.run_args import from_kwargs
from src.utils import to_relpath



# =========================
# Config dataclass
# =========================

@dataclass(frozen=True)
class ClinicalBERTArgs:
    """
    Configuration for ClinicalBERT encoder + classification head.

    Parameters
    ----------
    model_name_or_path : str
        HF checkpoint id or local path (e.g., "emilyalsentzer/Bio_ClinicalBERT").
    num_labels         : int
        Number of ICD labels K.
    max_len            : int, default=512
        Max sequence length (in wordpieces) for tokenizer.
    p_drop             : float, default=0.1
        Dropout before classification head.
    freeze_bert        : bool, default=False
        If True, freeze encoder (feature extractor mode).
    use_pooler         : bool, default=False
        If True, use encoder.pooler_output; else mean-pool last_hidden_state.
    """
    model_name_or_path: str
    num_labels: int
    max_len: int = 512
    p_drop: float = 0.1
    freeze_bert: bool = False
    use_pooler: bool = False


# -------------------------
# Encoder
# -------------------------

class ClinicalBERTEncoder(nn.Module):
    """
    HF encoder wrapper: [CLS] / pooled / mean-pooled representation → linear head.

    Pipeline (suggested)
    --------------------
    - AutoModel.from_pretrained(model_name_or_path)
    - forward(input_ids, attention_mask)
    - take:
        * pooler_output (if cfg.use_pooler), or
        * mean over last_hidden_state masked by attention_mask
    - dropout + linear → logits [B, K]
    """
    def __init__(self, cfg: ClinicalBERTArgs) -> None:
        super().__init__()
        self.cfg = cfg

        config = AutoConfig.from_pretrained(cfg.model_name_or_path)
        self.encoder = AutoModel.from_pretrained(cfg.model_name_or_path, config=config)

        hidden_size = self.encoder.config.hidden_size
        assert hidden_size > 0, (
            f"[ClinicalBERT:__init__] hidden_size must be > 0; got {hidden_size}."
        )
        assert cfg.num_labels > 0, (
            f"[ClinicalBERT:__init__] num_labels must be > 0; got {cfg.num_labels}."
        )
        self.drop = nn.Dropout(cfg.p_drop)
        self.fc = nn.Linear(hidden_size, cfg.num_labels)

        self.use_pooler = cfg.use_pooler

        if cfg.freeze_bert:
            for p in self.encoder.parameters():
                p.requires_grad = False


    def _mean_pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool last hidden states over non-padding tokens.

        Parameters
        ----------
        last_hidden_state : FloatTensor [B, L, H]
        attention_mask    : Long/BoolTensor [B, L]

        Returns
        -------
        pooled : FloatTensor [B, H]
        """
        assert isinstance(last_hidden_state, torch.Tensor), (
            "[ClinicalBERT:mean_pool] `last_hidden_state` must be a torch.Tensor."
        )
        assert isinstance(attention_mask, torch.Tensor), (
            "[ClinicalBERT:mean_pool] `attention_mask` must be a torch.Tensor."
        )

        assert last_hidden_state.ndim == 3, (
            f"[ClinicalBERT:mean_pool] last_hidden_state must be 3D [B, L, H]; "
            f"got dim={last_hidden_state.ndim} shape={tuple(last_hidden_state.shape)}."
        )

        assert attention_mask.ndim == 2, (
            f"[ClinicalBERT:mean_pool] attention_mask must be 2D [B, L]; "
            f"got dim={attention_mask.ndim} shape={tuple(attention_mask.shape)}."
        )

        B, L, H = last_hidden_state.shape
        assert attention_mask.shape == (B, L), (
            "[ClinicalBERT:mean_pool] attention_mask shape mismatch: "
            f"expected (B, L)=({B}, {L}); got {tuple(attention_mask.shape)}"
        )

        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)  # [B, L, 1]
        masked = last_hidden_state * mask
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = masked.sum(dim=1) / denom
        return pooled


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        """
        Forward pass of ClinicalBERT encoder.

        Parameters
        ----------
        input_ids : LongTensor [B, L]
        attention_mask : LongTensor/BoolTensor [B, L]

        Returns
        -------
        logits : FloatTensor [B, K]
            Unnormalized scores for each of K labels.
        """
        assert isinstance(input_ids, torch.Tensor), (
            "[ClinicalBERT:forward] `input_ids` must be a torch.Tensor."
        )
        assert isinstance(attention_mask, torch.Tensor), (
            "[ClinicalBERT:forward] `attention_mask` must be a torch.Tensor."
        )

        assert input_ids.ndim == 2, (
            f"[ClinicalBERT:forward] input_ids must be 2D [B, L]; "
            f"got dim={input_ids.ndim} shape={tuple(input_ids.shape)}."
        )
        assert attention_mask.ndim == 2, (
            f"[ClinicalBERT:forward] attention_mask must be 2D [B, L]; "
            f"got dim={attention_mask.ndim} shape={tuple(attention_mask.shape)}."
        )
        assert input_ids.shape == attention_mask.shape, (
            "[ClinicalBERT:forward] input_ids and attention_mask must have the same "
            f"shape; got input_ids={tuple(input_ids.shape)} "
            f"attention_mask={tuple(attention_mask.shape)}."
        )

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if self.use_pooler and getattr(outputs, "pooler_output", None) is not None:
            h = outputs.pooler_output               # [B, H]
        else:
            h = self._mean_pool(outputs.last_hidden_state, attention_mask)
        
        h = self.drop(h)
        logits = self.fc(h)

        expected = (input_ids.size(0), self.cfg.num_labels)
        assert logits.shape == expected, (
            "[ClinicalBERT:forward] logits shape mismatch: "
            f"expected {expected}, got {tuple(logits.shape)}."
        )

        assert logits.dtype in (torch.float16, torch.float32, torch.float64), (
            f"[ClinicalBERT:forward] logits dtype must be float, got {logits.dtype}."
        )
        return logits


# -------------------------
# Model wrapper
# -------------------------

class ClinicalBERTModel(nn.Module):
    """
    Runner-friendly wrapper exposing:
    - build(...)
    - to_device(...)
    - predict_proba_loader(...)
    - fit_loader(...)
    - save(...)
    - load(...)

    Mirrors TextCNNModel/BiGRUModel but works with HF-style batches:
    {input_ids, attention_mask, labels[, unsort_idx]}.
    """
    def __init__(self) -> None:
        super().__init__()
        self.net: Optional[ClinicalBERTEncoder] = None
        self.cfg: Optional[ClinicalBERTArgs] = None
        self.K: Optional[int] = None
        self.device: torch.device = torch.device("cpu")
        self.best_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Build / device
    # ------------------------------------------------------------------
    def build(self, *, num_labels: int, model_name_or_path: str, **hparams: Any) -> "ClinicalBERTModel":
        """
        Instantiate encoder with config.

        Parameters
        ----------
        num_labels : int
            Number of labels K for classification head.
        model_name_or_path : str
            HF model id or directory.
        **hparams
            Extra ClinicalBERTArgs fields: max_len, p_drop, freeze_bert, use_pooler, ...

        Returns
        -------
        ClinicalBERTModel
            Self, for chaining.
        """
        assert num_labels > 0, (
            "[ClinicalBERT:build] `num_labels` must be > 0, "
            f"got {num_labels}"
        )

        if self.net is not None:
            raise RuntimeError(
                "[ClinicalBERT:build] model already built; create a new instance instead."
            )
        
        self.cfg = ClinicalBERTArgs(
            model_name_or_path=model_name_or_path,
            num_labels=num_labels,
            **hparams,
        )
        self.net = ClinicalBERTEncoder(self.cfg)
        self.K = int(num_labels)

        return self

    def to_device(self, device: torch.device, logger: Optional[logging.Logger] = None) -> "ClinicalBERTModel":
        """
        Move model to device and remember it (cpu/cuda).

        Parameters
        ----------
        device : torch.device
            Target device.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        assert self.net is not None, "[ClinicalBERT:to_device] call build() before to_device()."

        device = torch.device(device)
        self.device = device
        super().to(device)

        pdev = next(self.parameters()).device

        if device.type == "cuda":
            idx = pdev.index if pdev.index is not None else torch.cuda.current_device()
            name = torch.cuda.get_device_name(idx)
            props = torch.cuda.get_device_properties(idx)
            cc = f"{props.major}.{props.minor}"
            mem_gb = props.total_memory / (1024**3)
            logger.info(f"[ClinicalBERT:device] {pdev} | {name} | CC {cc} | {mem_gb:.1f} GB")
        else:
            logger.warning(f"[ClinicalBERT:device] {pdev} (CPU)")

        same_index = (device.index is None) or (pdev.index == device.index)
        assert (pdev.type != "cuda") or same_index, (
            f"[ClinicalBERT:to_device] expected {device}, got {pdev}."
        )
        
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
            DataLoader yielding dicts with keys:
            - "input_ids"      : LongTensor [B, L]
            - "attention_mask" : LongTensor/BoolTensor [B, L]
            - (optional) "unsort_idx" : LongTensor [B]

        Returns
        -------
        np.ndarray
            Probabilities with shape (N_total, K) float32 in [0, 1].
            Rows are unsorted back to original order if `unsort_idx` is given.
        """
        assert self.net is not None, "[ClinicalBERT:predict] call build()/to_device() before predict()."
        self.eval()

        prob_list: List[torch.Tensor] = []

        for i, batch in enumerate(dl):

            assert "input_ids" in batch and "attention_mask" in batch, (
                "[ClinicalBERT:predict] batch must contain 'input_ids' and 'attention_mask' keys; "
                f"got keys={list(batch.keys())}."
            )

            # move tensors to self.device
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)

            logits = self.net(ids, mask)

            if i == 0:
                assert logits.shape[1] == int(self.K or logits.shape[1]), (
                    "[ClinicalBERT:predict] logits K mismatch: "
                    f"got {logits.shape[1]}, expected {self.K}"
                )
            probs = torch.sigmoid(logits)

            if "unsort_idx" in batch:
                unsort_idx = batch["unsort_idx"].to(self.device, non_blocking=True)
                assert unsort_idx.ndim == 1 and unsort_idx.numel() == probs.size(0), (
                    "[ClinicalBERT:predict] unsort_idx must be [B], "
                    f"got shape={tuple(unsort_idx.shape)} for batch B={probs.size(0)}"
                )
                probs = probs[unsort_idx]
            prob_list.append(probs.cpu())
        
        if not prob_list:
            K = int(self.K or (self.cfg.num_labels if self.cfg else 0))
            return np.zeros((0, K), dtype=np.float32)
        
        return torch.cat(prob_list, dim=0).numpy().astype(np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit_loader(
        self,
        dl_train: Iterable[dict],
        dl_val: Optional[Iterable[dict]] = None,
        **train_cfg: Any,
    ) -> "ClinicalBERTModel":
        """
        Train with AdamW + BCEWithLogitsLoss (+ AMP) over DataLoaders.
        Threshold selection handled via EvalArgs policy.

        Parameters
        ----------
        dl_train : Iterable[dict]
            Training DataLoader yielding {input_ids, attention_mask, labels}
        dl_val   : Iterable[dict], optional
            Validation DataLoader (same schema) for early-stopping / thr tuning.
        **train_cfg : Any
            Hyperparameters, parsed via `from_kwargs`:
            - epochs, lr, weight_decay, amp, clip, grad_accum, log_every
            - run (run_name), models_dir
            - thr, threshold_policy, min_pos, ks, ...

        Returns
        -------
        ClinicalBERTModel
            Self after training. Best state cached in `self.best_state`.
        """
        common, evalargs, trainargs = from_kwargs(**train_cfg)

        amp = bool(trainargs.amp) and (self.device.type == "cuda")
        lr = float(trainargs.lr)
        wd = float(trainargs.weight_decay)
        grad_accum = int(trainargs.grad_accum)
        clip_val = trainargs.clip
        epochs = int(trainargs.epochs)
        log_every = int(trainargs.log_every)
        pos_weight = trainargs.pos_weight
        threshold_policy = str(evalargs.threshold_policy)
        thr_fixed = float(evalargs.thr)
        min_pos = int(evalargs.min_pos)
        run_name = str(common.run)
        models_dir = str(trainargs.models_dir)

        assert 0.0 < thr_fixed < 1.0, (
            "[ClinicalBERT:fit] `thr` must be in (0, 1); "
            f"got {thr_fixed}."
        )

        assert self.net is not None, "[ClinicalBERT:fit] call build()/to_device() before fit()."
        assert threshold_policy in  {"fixed", "global_val", "per_label_val"}, (
            f"[ClinicalBERT:fit] unknown threshold_policy={threshold_policy}"
        )

        optim = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.amp.GradScaler(enabled=amp)

        if pos_weight is not None:
            if not torch.is_tensor(pos_weight):
                pos_weight = torch.as_tensor(pos_weight, dtype=torch.float32)
            pos_weight = pos_weight.to(self.device)
            assert pos_weight.numel() == int(self.K or 0), (
                "[ClinicalBERT:fit] pos_weight must have length K; "
                f"got {pos_weight.numel()} vs K={self.K}."
            )
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()

        scheduler = None
        if dl_val is not None:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="max", factor=0.5, patience=1)
        
        needs_val_target = (dl_val is not None) and (threshold_policy in {"global_val", "per_label_val"})
        if needs_val_target and ("y_val" not in train_cfg):
            raise ValueError(
                "[ClinicalBERT:fit] y_val is required when threshold_policy in {'global_val', 'per_label_val'}. "
                "Pass y_val=(N_val, K) from runner, or set threshold_policy='fixed'."
            )

        best_metric = -1.0
        self.best_state = None
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
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True).float()

                with torch.autocast(device_type=self.device.type, dtype=torch.float32, enabled=amp):
                    logits = self.net(input_ids, attention_mask)
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
                        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

                        logits = self.net(input_ids, attention_mask)
                        probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
                        if "unsort_idx" in batch:
                            probs = probs[batch["unsort_idx"].cpu().numpy()]
                        prob_list.append(probs)
                
                P_val = np.concatenate(prob_list, axis=0) if prob_list else np.zeros((0, int(self.K or 0)), dtype=np.float32)
                Y_val = train_cfg["y_val"]
                if torch.is_tensor(Y_val):
                    Y_val = Y_val.detach().cpu().numpy()
                Y_val = np.asarray(Y_val)
                assert Y_val.ndim == 2 and Y_val.shape[1] == int(self.K), f"[ClinicalBERT:fit] y_val shape invalid: {Y_val.shape}, expected (*, {self.K})"
                if Y_val.dtype != np.uint8:
                    Y_val = (Y_val > 0.5).astype(np.uint8)
                assert P_val.shape == Y_val.shape == (Y_val.shape[0], int(self.K)), f"[ClinicalBERT:fit] val shape mismatch: {P_val.shape} vs {Y_val.shape}"

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


    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: Path, *, protocol: int = 4, logger: Optional[logging.Logger] = None) -> None:
        """
        Persist checkpoint and config under `path`.

        Artifacts
        ---------
        - `best.ckpt` : torch.save({state_dict, config, K, versions, meta})

        Notes
        -----
        - Uses relative path only; creates parents.
        - Expects `self.best_state` or current `state_dict()`.
        """
        if logger is None:
            logger = logging.getLogger(__name__)

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)

        sd = self.best_state if self.best_state is not None else self.state_dict()
        sd_cpu = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}

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
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> "ClinicalBERTModel":
        """
        Reload from `path/best.ckpt` and return a ready model on CPU.

        Returns
        -------
        ClinicalBERTModel
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

