"""
protocol.py - Unified Model Interface (ProbModel)

Purpose
-------
Define a reproducible API so that all models
(TF-IDF baselines, CNN/GRU, BERT, Longformer, etc.) can be trained,
evaluated, and saved using an identical interface.

Conventions
-----------
- Deterministic: honors global random seed via utils.seed.
- PHI-safety: models operate on pre-sanitized text; never log raw notes.
- Inputs: List[str] of de-identified text.
- Labels: Y (N, K) multi-hot np.ndarray, dtype={uint8, bool}, values in {0, 1}.
- Outputs: probability matrix (N, K) float32 in [0, 1].
- Artifacts: save() must persist both model weights and preprocessing state.
"""

from __future__ import annotations
from typing import Protocol, List, Optional, Any, runtime_checkable
from pathlib import Path
import numpy as np
import logging

@runtime_checkable
class ProbModel(Protocol):
    """
    Standard interface for probabilistic multi-label models.

    Notes
    -----
    - Implementations must be deterministic given the global seed.
    - predict_proba() must return probabilities aligned with K from fit().
    """

    def fit(self, texts: List[str], Y: np.ndarray, logger: Optional[logging.Logger]=None) -> "ProbModel":
        """
        Train the model on texts and multi-hot labels.

        Parameters
        ----------
        texts   : list[str]             # N items
        Y       : np.ndarray            # shape (N, K), dtype {uint8, bool}
        logger  : logging.Logger, optional

        Returns
        -------
        ProbModel
            self (to allow chaining).
        """
        ...

    def predict_proba(self, texts: List[str], logger: Optional[logging.Logger]=None) -> np.ndarray:
        """
        Return predicted probabilities.

        Parameters
        ----------
        texts   : list[str]         # N' items

        Returns
        -------
        np.ndarray
            shape (N', K), dtype float32 in [0, 1]
        """
        ...
    
    def save(self, path: Path, *, protocol: int = 4, logger: Optional[logging.Logger]=None) -> None:
        """
        Persist models weights, preprocessing objects, and metadata.
        """
        ...
    
    @classmethod
    def load(cls, path: Path, logger: Optional[logging.Logger]=None) -> "ProbModel":
        """
        Reload a previously saved model.
        """
        ...

@runtime_checkable
class NeuralBaseModel(Protocol):
    """
    Interface for neural models that train from DataLoaders (not raw text).

    Notes
    -----
    - Use BCEWithLogitsLoss internally; sigmoid applied at interface to return prob.
    - Must be device-aware and AMP-capable (optional).
    """
    def build(self, *, vocab_size: int, num_labels: int, pad_idx: int,
              **hparams: Any) -> "NeuralBaseModel":
        ...
    
    def fit_loader(self, dl_train: Any, dl_val: Optional[Any] = None, **train_cfg: Any) -> "NeuralBaseModel":
        ...
    
    def predict_proba_loader(self, dl: Any) -> np.ndarray:
        ...
    
    def to_device(self, device: Any) -> "NeuralBaseModel":
        ...
    
    def save(self, path: Path, *, protocol: int = 4, logger: Optional[logging.Logger] = None) -> None:
        ...
    
    @classmethod
    def load(cls, path: Path, logger: Optional[logging.Logger] = None) -> "NeuralBaseModel":
        ...


