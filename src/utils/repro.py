from __future__ import annotations
import os
import random
import logging
import numpy as np
from pathlib import Path

DEFAULT_SEED: int = 42

def set_seed(seed: int | np.integer | str | Path | None = None) -> int:
    """
    Set global random seeds for reproducibility.

    Parameters
    ----------
    seed: int
        Random seed to set.
    """
    if seed is None:
        seed = DEFAULT_SEED
    else:
        try:
            seed_norm = int(seed)
        except Exception as e:
            logging.warning(f"[seed] Invalid seed {type(seed).__name__}={seed!r}; "
                            f"falling back to {DEFAULT_SEED}. err={e}")
            seed_norm = DEFAULT_SEED
    
    os.environ["PYTHONHASHSEED"] = str(seed_norm)
    random.seed(seed_norm)
    np.random.seed(seed_norm)
    try:
        import torch

        torch.manual_seed(seed_norm)
        torch.cuda.manual_seed(seed_norm)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    logging.info(f"[seed] Global random seed set to {seed_norm}")