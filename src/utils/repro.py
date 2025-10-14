from __future__ import annotations
import os
import random
import logging
import numpy as np

DEFAULT_SEED: int = 42

def set_seed(seed: int | None = None) -> int:
    """
    Set global random seeds for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed to set.
    """
    if seed is None:
        seed = DEFAULT_SEED
    
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    logging.info(f"[seed] Global random seed set to {seed}")