from __future__ import annotations
import logging
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = 'prep', log_dir: str = "logs", level: int = logging.INFO) -> logging.Logger:
    """
    Create a dual-handler logger that writes both to console and to a file.
    Files logs are timestamped unnder logs/ for auditing and reproducibility.

    Parameters
    ----------
    name    :    str
        Loggeer name (e.g., "prep")
    log_dir :   str
        Relative directory to write logs (portable)
    
    Returns
    -------
    logging.Logger
        Configured logger instance with no duplicate handlers.
    
    Notes
    -----
    - Does not leak absolute system paths in the log message.
    - Intended to be called once at the entrypoint (main()).
    """
    # Ensure log directory exist    
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Timestamped file path
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(log_dir) / f"{name}_{ts}.log"

    # --- Formatter ---
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        "%Y-%m-%d %H:%M:%S",
    )

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(level)

    # --- File handler ---
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(level)

    # --- Attach both handlers ---
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)
    root.setLevel(level)
    
    logger = logging.getLogger(name)
    logger.propagate = True
    logger.setLevel(level)
    logger.info(f"[INIT] logging to {log_path.as_posix()}")
    
    return logger