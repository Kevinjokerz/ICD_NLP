from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union
import os
import logging

ABS_HINTS = (":\\", ":/", "\\\\")   # Window drive, URL-like, UNC shares
FORBID_PREFIXES = ("/",)            # Unix absolute root

def get_project_root() -> Path:
    """
    Resolve the project root directory.

    Resolution order:
    1) Environment variable `ICD_PROJ_ROOT`
    2) Traverse upwards from this file to find a folder containing 'data' and 'src'
    3) Fallback to current working directory

    Returns
    -------
    Path
        Project root path.
    """
    root_env = os.getenv("ICD_PROJ_ROOT")
    if root_env:
        return Path(root_env).resolve()
    
    start = Path(__file__).resolve().parent
    for p in (start, *start.parents):
        data_dir = p / "data"
        src_dir = p / "src"
        if data_dir.is_dir() and src_dir.is_dir():
            return p
    
    return Path.cwd().resolve()


def looks_absolute(path_str: str) -> bool:
    """
    Heuristic detector of absolute path strings.

    Parameters
    ----------
    path_str : str

    Returns
    -------
    bool
    """
    s = (path_str or "").strip()

    if s.startswith(FORBID_PREFIXES):
        return True
    
    for hint in ABS_HINTS:
        idx = s.find(hint)
        if 0 <= idx <= 3:
            return True
        
    return False

def to_relpath(path_like: str | Path, root: Optional[Path] = None) -> str:
    """
    Convert a path to a POSIX-style relative path from project root.

    If the path is outside the project root, falls back to the filename
    and log a warning instead of raising an exception.

    Parameters
    ----------
    path_like : str | Path
        Path to convert.

    root      : Optional[Path]
        Base directory (default to project root via `get_project_root()`).

    Returns
    -------
    str
        POSIX relative path.
    """
    base = Path(root or get_project_root()).resolve()
    p_in = Path(path_like)

    try:
        rel = p_in.relative_to(base)
        return rel.as_posix()
    except Exception:
        pass

    p_abs = (p_in if p_in.is_absolute() else (base / p_in)).resolve()
    try:
        rel = p_abs.relative_to(base)
        return rel.as_posix() 
    except ValueError:
        logging.warning(f"[WARN] {p_abs} is outside project root {base}; using basename only.")
        return Path(p_abs.name).as_posix()


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

def safe_join_rel(*parts: Union[str, Path], root: Optional[Path] = None) -> Path:
    """
    Join relative parts under the project root and return a normalized, resolved Path.

    Safety guarantees
    -----------------
    - Rejects absolute or URL-like inputs (Windows drive, UNC, scheme://).
    - Prevents parent-traversal escapes (...).
    - Ensures the resolved path remains inside the project root.

    Parameters
    ----------
    parts : str | Path
        Path components (relative to project root).
    root  : Optional[Path]
        Project root (default to get_project_root()).
    
    Returns
    -------
    Path
        Resolved path strictly inside project root.

    Raises
    ------
    ValueError
        If an input part is absolute/URL-like, contains traversal,
        or the resolved path escapes the project root.
    """
    base = Path(root or get_project_root()).resolve()
    
    clean_parts: Iterable[Path] = []
    for part in parts:
        p = Path(part)

        # reject absolute or url-like (e.g., c:\, //server, https://)
        s = str(p).strip()
        low = s.lower()
        if p.is_absolute() or low.startswith(("/", "\\"))  or ":\\" in low[:4] or "://" in low[:8]:
            raise ValueError(f"Absolute/URL-like path not allowed: {part!r}")
        
        # prevent sneaky parent traversal in components
        if any(seg == ".." for seg in p.parts):
            raise ValueError(f"Parent-traversal '..' not allowed: {part!r}")
        
        clean_parts = (*clean_parts, p)
    
    candidate = (base.joinpath(*clean_parts)).resolve()

    # final guard: must still be inside base after resolution/symlinks
    if not _is_relative_to(candidate, base):
        raise ValueError(f"Resolved path escaped project root: {candidate} (root={base})")
    
    return candidate