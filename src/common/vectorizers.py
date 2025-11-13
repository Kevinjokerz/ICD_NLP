"""
vectorizers.py - TF-IDF Feature Builders

Purpose
-------
Factories and helpers to create word/char TF-IDF features and combine them
for downstream linear baselines

Conventions
-----------
- Reproducibility: deterministic vectorizers; fixed params in config.
- Portability: only relative paths
- PHI-safe: never log raw TEXT
- Metrics-first: shapes/dtypes check upfront.

Inputs
------
- texts: Sequence[str] (discharge notes), never logged verbatim.

Outputs
-------
- scipy.sparse.csr_matrix features
- Pickled vectorizers in models/ (optional)
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Union
from scipy import sparse as sp
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import sys
import sklearn

from src.utils import setup_logger, ensure_parent_dir, to_relpath

# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class VectorizeConfig:
    """
    Immutable configuration for TF-IDF vectorizers.

    Parameters
    vec_mode    : {"word", "char", "both", "none"}
        Which analyzers to instantiate.
    word_max_features : int
        Vocabulary cap for the word analyzer.
    char_max_features : int
        Vocabulary cap for the char analyzer.
    word_ngram  : Tuple[int, int]
        (min_n, max_n) for word n-grams.
    char_ngram  : Tuple[int, int]
        (min_n, max_n) for char n_grams.
    min_df      : float | int
        min doc freq shared by analyzer (override per analyzer if needed).
    lowercase   : bool
        Lowercase toggle for both analyzers.
    strip_accents : Optional[str]
        "unicode" recommended.
    sublinear_tf : bool
        Use sublinear tf scaling.
    """
    vec_mode: str = "both"
    word_max_features: int = 50_000
    char_max_features: int = 30_000
    word_ngram: Tuple[int, int] = (1, 2)
    char_ngram: Tuple[int, int] = (3, 5)
    min_df: Union[float, int] = 2
    lowercase: bool = True
    strip_accents: Optional[str] = "unicode"
    sublinear_tf: bool = True
    dtype: np.dtype = np.float32

# =============================================================================
# Factories
# =============================================================================

def make_vectorizers(
    cfg: VectorizeConfig
) -> Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    """
    Factoring return {vec_word, vec_char} according to vec_mode.

    Parameters
    ----------
    cfg : VectorizeConfig
    
    Returns
    -------
    (vec_word, vec_char) : Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]
        Word-level and char-level vectorizers (None if not used).
    """
    assert isinstance(cfg, VectorizeConfig)
    assert cfg.vec_mode in {"word", "char", "both", "none"}, f"[CHECK] Unknown vectorize mode: {cfg.vec_mode}"

    create_word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=cfg.word_ngram, max_features=cfg.word_max_features,
        min_df=cfg.min_df, lowercase=cfg.lowercase, strip_accents=cfg.strip_accents,
        sublinear_tf=cfg.sublinear_tf, dtype=cfg.dtype
    )

    create_char_vec = TfidfVectorizer(
        analyzer="char", ngram_range=cfg.char_ngram, max_features=cfg.char_max_features,
        min_df=cfg.min_df, lowercase=cfg.lowercase, strip_accents=cfg.strip_accents,
        sublinear_tf=cfg.sublinear_tf, dtype=cfg.dtype
    )

    if cfg.vec_mode == "word":
        result = (create_word_vec, None)
    elif cfg.vec_mode == "char":
        result = (None, create_char_vec)
    elif cfg.vec_mode == "both":
        result = (create_word_vec, create_char_vec)
    else:
        result = (None, None)
    
    assert len(result) == 2 and all([x is None or hasattr(x, "fit")] for x in result), "[CHECK] Unexpected tuple length"
    return result
    
# =============================================================================
# Fit & Transform helpers
# =============================================================================

def _fit_or_transform(vec: TfidfVectorizer, 
                      texts: Sequence[str], 
                      *, 
                      fit: bool, 
                      logger: Optional[logging.Logger]=None,
                      to_dtype: Optional[Union[str, np.dtype]] = "float32",
) -> sp.csr_matrix:
    """
    Internal helper to call fit_transform/transform.

    Parameters
    ----------
    vec     : TfidfVectorizer
    texts   : Sequence[str]
    fit     : bool
    logger  : logging.Logger, optional

    Returns
    -------
    X   : sp.csr_matrix
        CSR TF-IDF matrix.
    
    Notes
    -----
    - Never logs raw texts.
    - May log N only.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    assert vec is not None, "Vectorizer is None"
    assert hasattr(vec, "fit") and hasattr(vec, "transform"), "[CHECK] Invalid vectorizer API"
    
    texts = list(texts) if not isinstance(texts, list) else texts
    assert all(isinstance(t, str) for t in texts), "texts must be Sequence[str]"

    if fit and len(texts) == 0:
        raise ValueError("[CHECK] Empty training texts; cannot fit TF-IDF.")
    if not fit and len(texts) == 0:
        assert hasattr(vec, "vocabulary_"), "[CHECK] Vectorizer must be fitted before transform([])"
        k = len(vec.vocabulary_)
        return sp.csr_matrix((0, k), dtype=(np.dtype(to_dtype) if to_dtype else None))
    
    try:
        X = vec.fit_transform(texts) if fit else vec.transform(texts)
    except Exception as e:
        raise RuntimeError(f"[TFIDF] {'fit' if fit else 'transform'} failed: {e}") from e
    
    assert sp.isspmatrix_csr(X), "Expected CSR matrix"
    if to_dtype is not None:
        X = X.astype(np.dtype(to_dtype), copy=False)

    if X.shape[1] == 0:
        logger.debug("[TFIDF] produced 0 features - check min_df/ngram/config")
    logger.debug(f"[TFIDF] fit={fit} N={len(texts)} K={X.shape[1]} nnz={X.nnz}")
    return X

def hstack_features(
        vec_word: Optional[TfidfVectorizer], 
        vec_char: Optional[TfidfVectorizer], 
        texts: Sequence[str], 
        *, 
        fit: bool,
        to_dtype: Optional[Union[str, np.dtype]] = "float32",
        logger: Optional[logging.Logger] = None
) -> sp.csr_matrix:
    """
    Build design matrix by concatenating word +char TF-IDF blocks.

    Parameters
    ----------
    vec_word    : Optional[TfidfVectorizer]
    vec_char    : Optional[TfidfVectorizer]
    texts       : Sequence[str]
    fit         : bool
        If True, fit on texts; else transform using fitted vocabularies.
    
    Returns
    -------
    X : sp.csr_matrix
        Concatenated features (word|char|both). Valid even if both None (N x 0)

    Notes
    -----
    - Column order is [word_block | char_block]
    - Safe when one of analyzer is None.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    assert isinstance(texts, (list, tuple)), "[CHECK] texts must be a Sequence[str]"
    N = len(texts)

    _dtype = (np.dtype(to_dtype) if to_dtype is not None else None)
    Xw = _fit_or_transform(vec_word, texts=texts, fit=fit, logger=logger, to_dtype=_dtype) if vec_word else sp.csr_matrix((N, 0), dtype=_dtype)
    Xc = _fit_or_transform(vec_char, texts=texts, fit=fit, logger=logger, to_dtype=_dtype) if vec_char else sp.csr_matrix((N, 0), dtype=_dtype)

    X = sp.hstack([Xw, Xc], format="csr")
    assert sp.isspmatrix_csr(X), "[CHECK] Expected CSR after hstack"
    assert X.shape[0] == N, f"[CHECK] Shape mismatch"

    if X.shape[1] == 0:
        logger.debug("[HSTACK] produced 0 features - check min_df/ngram/config")
    logger.debug(f"[HSTACK] X.shape={X.shape}, nnz={X.nnz}")

    return X

def fit_on_texts(vec_word: Optional[TfidfVectorizer],
                 vec_char: Optional[TfidfVectorizer],
                 texts: Sequence[str],
                 *,
                 to_dtype: Optional[Union[str, np.dtype]] = "float32",
                 logger: Optional[logging.Logger] = None,
) -> sp.csr_matrix:
    """
    Fit vectorizers on train and return concatenated train matrix.

    Returns
    -------
    X_tr : sp.csr_matrix
        Shape (N_train, K_total) with float32 dtype
    """
    return hstack_features(vec_word=vec_word, vec_char=vec_char, texts=texts, fit=True, to_dtype=to_dtype, logger=logger)

def transform_texts(vec_word: Optional[TfidfVectorizer],
                    vec_char: Optional[TfidfVectorizer],
                    texts   : Sequence[str],
                    *,
                    to_dtype: Optional[Union[str, np.dtype]] = "float32",
                    logger: Optional[logging.Logger] = None,
) -> sp.csr_matrix:
    """
    Transform texts using fitted vectorizers.

    Returns
    -------
    X   : sp.csr_matrix
        Shape (N, K_total) with float32 dtype.
    """
    if len(texts) > 0:
        if vec_word is not None:
            assert hasattr(vec_word, "vocabulary_"), "[CHECK] word vec not fitted."
        if vec_char is not None:
            assert hasattr(vec_char, "vocabulary_"), "[CHECK] char vec not fitted"
    return hstack_features(vec_word=vec_word, vec_char=vec_char, texts=texts, fit=False, to_dtype=to_dtype, logger=logger)

def fit_transform_splits(vec_word: Optional[TfidfVectorizer],
                        vec_char: Optional[TfidfVectorizer],
                        texts_tr: Sequence[str],
                        texts_va: Optional[Sequence[str]] = None,
                        texts_te: Optional[Sequence[str]] = None,
                        *,
                        to_dtype: Optional[Union[str, np.dtype]] = "float32",
                        logger: Optional[logging.Logger] = None,
) -> Dict[str, sp.csr_matrix]:
    """
    Convenience: fit on train, transform val/test.

    Returns
    -------
    mats : dict[str, sp.csr_matrix]
        Subset of {"train", "val", "test"} mapping to CSR matrices (float32).
    
    Notes
    -----
    - Train is always present. Val/Test optional.
    - K_total must be identical across splits.
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    X_tr = fit_on_texts(vec_word, vec_char, list(texts_tr), to_dtype=to_dtype, logger=logger)
    K = X_tr.shape[1]

    mats: Dict[str, sp.csr_matrix] = {"train": X_tr}

    if texts_va is not None:
        X_va = transform_texts(vec_word, vec_char, list(texts_va), to_dtype=to_dtype, logger=logger)
        assert X_va.shape[1] == K, "[CHECK] VAL feature dim K mismatch vs TRAIN"
        mats["val"] = X_va
    
    if texts_te is not None:
        X_te = transform_texts(vec_word, vec_char, list(texts_te), to_dtype=to_dtype, logger=logger)
        assert X_te.shape[1] == K, "[CHECK] TEST feature dim K mismatch vs TRAIN"
        mats["test"] = X_te

    logger.debug(f"[SPLITS]" + ", ".join(f"{k}={v.shape}" for k, v in mats.items()))
    return mats

# =============================================================================
# Persistence
# =============================================================================

def save_vectorizers(vec_word: Optional[TfidfVectorizer],
                     vec_char: Optional[TfidfVectorizer],
                     *,
                     dir_path: Union[str, Path],
                     run_name: str,
                     protocol: Union[int, None] = None,
                     logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Save vectorizers to models/{run_name}/ as pickles.

    Returns
    -------
    (p_word, p_char) : tuple[Optional[Path], Optional[Path]]
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    out_dir = Path(dir_path) / run_name
    ensure_parent_dir(out_dir / "placeholder")

    proto = pickle.HIGHEST_PROTOCOL if protocol is None else int(protocol)
    p_word: Optional[Path] = None
    p_char: Optional[Path] = None

    if vec_word is not None:
        p_word = out_dir / "tfidf_word.pkl"
        with open(p_word, "wb") as f:
            pickle.dump(vec_word, f, protocol=proto)

    if vec_char is not None:
        p_char = out_dir / "tfidf_char.pkl"
        with open(p_char, "wb") as f:
            pickle.dump(vec_char, f, protocol=proto)
    
    meta = {
        "python": sys.version,
        "pickle_protocol": proto,
        "sklearn": sklearn.__version__
    }
    with open(out_dir / "tfidf_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    
    try:
        logger.debug(
            "[TFIDF:save] word=%s char=%s proto=%s",
            (to_relpath(p_word)) if p_word else "-",
            (to_relpath(p_char)) if p_char else "-",
            proto,
        )
    except Exception:
        logger.debug(f"[TFIDF:save] proto={proto}")

    return p_word, p_char

def load_vectorizers(*,
                     dir_path: Union[str, Path],
                     run_name: str,
                     logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]:
    """
    Load vectorizers back from models/{run_name}/.

    Returns
    -------
    (vec_word, vec_char): Tuple[Optional[TfidfVectorizer], Optional[TfidfVectorizer]]
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    base = Path(dir_path) / run_name
    p_word = base / "tfidf_word.pkl"
    p_char = base / "tfidf_char.pkl"

    vec_word: Optional[TfidfVectorizer] = None
    vec_char: Optional[TfidfVectorizer] = None

    if p_word.exists():
        with open(p_word, "rb") as f:
            vec_word = pickle.load(f)
    
    if p_char.exists():
        with open(p_char, "rb") as f:
            vec_char = pickle.load(f)
    
    assert vec_word is None or hasattr(vec_word, "transform"), "[CHECK] Invalid word vectorizer"
    assert vec_char is None or hasattr(vec_char, "transform"), "[CHECK] Invalid char vectorizer"
    
    logger.debug("[TFIDF:load] word=%s char=%s",
                 to_relpath(p_word) if p_word.exists() else "-",
                 to_relpath(p_char) if p_char.exists() else "-")
    
    return vec_word, vec_char

# =============================================================================
# Tiny smoke-check (PHI-safe)
# =============================================================================

def _tiny_smoke_test() -> None:
    """
    Run a miniature end-to-end check with synthetic texts.
    """

    logger = setup_logger("vectorizers", level=logging.DEBUG)
    texts = ["ab pain and cough", "chest pain, cough cough", "no issues reported"]
    cfg = VectorizeConfig(vec_mode="both", word_max_features=1_000, char_max_features=500, min_df=1)
    vw, vc = make_vectorizers(cfg)
    mats = fit_transform_splits(vw, vc, texts_tr=texts, texts_va=texts[:2], logger=logger)
    assert "train" in mats and mats["train"].shape[0] == len(texts)
    assert mats["train"].dtype == np.float32
    if "val" in mats:
        assert mats["val"].shape[1] == mats["train"].shape[1]

    cfg_w = VectorizeConfig(vec_mode="word", min_df=1)
    vw, vc = make_vectorizers(cfg_w)
    mats_w = fit_transform_splits(vw, vc, texts_tr=texts, texts_va=[], logger=logger)
    assert mats_w["train"].shape[1] >= 1
    if "val" in mats_w:
        assert mats_w["val"].shape == (0, mats_w["train"].shape[1])
    
    cfg_c = VectorizeConfig(vec_mode="char", min_df=1, char_ngram=(10, 12))
    vw, vc = make_vectorizers(cfg_c)
    mats_c = fit_transform_splits(vw, vc, texts_tr=texts, logger=logger)
    assert mats_c["train"].shape[0] == len(texts)
    assert sp.isspmatrix_csr(mats_c["train"])
    logger.debug("[SMOKE] OK: both/word/char cases passed.")

if __name__ ==  "__main__":
    _tiny_smoke_test()

