"""
textdata.py - Tokenizer, Vocab, Dataset, Collate (PHI-safe)

Purpose
-------
Provide minimal text preprocessing utilities shared by CNN/BiGRU
- Whitespace tokenizer (lowercase)
- Frequency-capped vocabulary with PAD/UNK
- Torch Dataset (notes -> token ids, lengths, multi-hot labels)
- Collate (pad-to-batch-max)

Convention
----------
- PHI-safety: never log raw text
- Reproducibility: vocab built on train only; deterministic ordering.
- Portability: relative paths only.

Inputs
------
- texts : list[str] de-identified notes
- Y     : np.ndarray, shape (N, K), dtype {uint8, bool}

Outputs
-------
- tokens : torch.int64 [B, L]
- lengths: torch.int64 [B]
- labels : torch.float32 [B, K]
"""

from __future__ import annotations
from typing import List, Sequence, Dict, Tuple
from collections import Counter
import numpy as np
import torch
from torch.utils.data import Dataset
import unicodedata
import regex as re

class Tokenizer:
    """
    Minimal, PHI-safe tokenizer.

    Notes
    -----
    - Lowercase + Unicode NFKC normalize.
    - Remove unicode control chars (C*), collapse whitespace.
    - Strip leading/trailing punctuation/symbols; keep internal hyphens.
    - Optionally anonymize digit run (>=2) as <NUM>.
    - Never logs input text.
    """
    def __init__(self, *, anonymize_numbers: bool = True) -> None:
        self.anonymize_numbers = bool(anonymize_numbers)
        self.MULTISPACE = re.compile(r"\s+")
        self.EDGE_PUNCT = re.compile(r"^[\p{P}\p{S}]+|[\p{P}\p{S}]+$", flags=re.UNICODE)
        self.DIGITS = re.compile(r"\d{2,}")
        self.NUM_TOKEN = "<NUM>"

    def _remove_control_chars(self, text: str) -> str:
        out = ''.join(ch if unicodedata.category(ch)[0] != "C" else " " for ch in text)
        return out
    
    def _strip_edge_punct(self, tok: str) -> str:
        tok = self.EDGE_PUNCT.sub("", tok)
        return tok

    def __call__(self, text: str) -> List[str]:
        if not text:
            return []
        
        # --- lowercase + NFKC + remove control chars ---
        s = str(text).lower()
        s = unicodedata.normalize("NFKC", s)
        s = self._remove_control_chars(s)

        # --- collapse whitespace ---
        s = self.MULTISPACE.sub(" ", s).strip()

        # --- whitespace split ---
        raw_tokens = s.split()

        # --- per-token cleanup ---
        tokens: List[str] = []
        for t in raw_tokens:
            t = self._strip_edge_punct(t)
            if not t:
                continue

            # anonymize numbers
            if self.anonymize_numbers and self.DIGITS.search(t):
                t = self.NUM_TOKEN
            
            if t != self.NUM_TOKEN and not any(c.isalnum() for c in t):
                continue

            tokens.append(t)
        
        assert isinstance(tokens, list), f"[CHECK] tokens must be a list"
        assert all(isinstance(t, str) and len(t) > 0 for t in tokens), f"[TOKENIZER:type] Found non-str or empty token in output: {tokens}"

        return tokens

class Vocab:
    """
    Frequency-based vocabulary with PAD/UNK.

    Parameters
    ----------
    max_size : int
        Upper bound on vocabulary size (including specials).
    min_freq : int
        Minimum frequency threshold (keep tokens with freq >= min_freq).
    specials : List[str]
        Order matters. Recommended: ["<PAD>", "<UNK>"].

    Attributes
    ----------
    stoi : Dict[str, int]
        String -> index.
    itos : List[str]
        Index -> str.
    pad_idx : int
    unk_idx : int
    """
    def __init__(
            self,
            *,
            max_size: int = 50_000,
            min_freq: int = 1,
            specials: List[str] = ["<PAD>", "<UNK>"]
    ) -> None:
        self.max_size = int(max_size)
        self.min_freq = int(min_freq)

        assert len(specials) >= 2 and specials[0] == "<PAD>" and specials[1] == "<UNK>"
        self.specials = list(specials)
        self.stoi: Dict[str, int] = {}
        self.itos: List[str] = []
        self.pad_idx: int = 0
        self.unk_idx: int = 1
    
    def fit_from_counter(self, ctr: Counter) -> None:
        """
        Build vocab from a token Counter (train split only).

        Notes
        -----
        - Deterministic: sort by (-freq, token).
        - Enforce max_size and min_freq.
        - Ensure specials at the front (PAD=0, UNK=1).
        """
        self.itos = list(self.specials)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.pad_idx, self.unk_idx = 0, 1

        # --- Filter & sort candidates (not specials) ---
        toks = [(t, f) for t, f in ctr.items() if f >= self.min_freq and t not in self.stoi]
        toks = sorted(toks, key=lambda x: (-x[1], x[0]))

        # --- ensure space for <NUM> if anonymizer emits it ---
        reserve = 1 if "<NUM>" not in self.stoi else 0
        remaining = max(0, self.max_size - len(self.itos) - reserve)

        # --- take the top rermaining tokens and append to itos ---
        keep = toks[:remaining]
        for t, _ in keep:
            self.stoi[t] = len(self.itos)
            self.itos.append(t)

        # --- Ensure  <NUM> exists if tokenizer emits it ---
        if reserve == 1:
            self.stoi["<NUM>"] = len(self.itos)
            self.itos.append("<NUM>")
        
        assert len(self.itos) <= self.max_size, f"[VOCAB:cap] len(itos)={len(self.itos)} exceed max_size={self.max_size}"
        assert self.itos[0] == "<PAD>" and self.itos[1] == "<UNK>", f"[VOCAB:specials] First two must be ['<PAD>', '<UNK>']; got {self.itos[:2]}"

    def encode(self, tokens: List[str]) -> List[int]:
        """
        Map tokens to ids (OOV -> UNK)

        Returns
        -------
        List[int]
            Tokens ids, len = len(tokens)
        """
        return [self.stoi.get(t, self.unk_idx) for t in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """
        Map ids back to tokens (out-of-range -> <UNK>).
        """
        return [self.itos[i] if 0 <= i < len(self.itos) else self.itos[self.unk_idx] for i in ids]
    
    def __len__(self) -> int:
        return len(self.itos)

def pad_collate(
        batch: List[Dict[str, torch.Tensor]], 
        pad_idx: int,
        *,
        sort_for_rnn: bool = False,
        return_index: bool = False,
        return_labels: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Pad sequences in a batch to max length.

    Inputs
    ------
    batch : list of dict
        Each dict has {"tokens": LongTensor [L_i], "labels": FloatTensor/Bool [K]}.

    Returns
    -------
    tokens  : LongTensor [B, L_max] padded with pad_idx
    lengths : LongTensor [B]
    labels  : FloatTensor [B, K]
    """
    # --- gather ---
    seqs = [ex["tokens"] for ex in batch]
    lbls = [ex["labels"] for ex in batch] if return_labels else None

    # --- length ---
    lengths = torch.tensor([int(x.size(0)) for x in seqs], dtype=torch.long)

    # --- pad to L_max ---
    tokens = torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=int(pad_idx)
    )
    
    out: Dict[str, torch.Tensor] = {}
    out["tokens"] = tokens
    out["lengths"] = lengths

    # --- sort for BiGRU + pack_padded_sequence ---    
    if return_labels:
        labels = torch.stack(lbls, dim=0).to(dtype=torch.float32)
        out["labels"] = labels
    
    if sort_for_rnn:
        lengths, sort_idx = lengths.sort(dim=0, descending=True)
        tokens = tokens.index_select(0, sort_idx)
        if return_labels:
            out["labels"] = out["labels"].index_select(0, sort_idx)

        out["tokens"] = tokens
        out["lengths"] = lengths

        if return_index:
            unsort_idx = torch.empty_like(sort_idx)
            unsort_idx[sort_idx] = torch.arange(tokens.size(0), device=sort_idx.device)
            out["sort_idx"] = sort_idx
            out["unsort_idx"] = unsort_idx
    
    assert tokens.dtype == torch.long, f"[COLLATE] token.dtype={tokens.dtype} expected torch.long"
    assert lengths.dtype == torch.long, f"[COLLATE] lengths.dtype={lengths.dtype} expected torch.long"
        
    if return_labels:
        assert tokens.size(0) == lengths.size(0) == out["labels"].size(0), \
            f"[COLLATE] batch dims mismatch: tokens={tokens.size()}, lengths={lengths.size()}, labels={out['labels'].size()}"

    return out

class NotesDataset(Dataset):
    """
    Torch Dataset for notes + multi-label targets.

    Parameters
    ----------
    texts   : Sequence[str]
        De-identified notes; length N.
    Y       : np.ndarray
        Multi-hot matrix (N, K), dtype {uint8, bool}.
    tok     : Tokenizer
    vocab   : Vocab
    max_len : int
        Clip/pad length per example.
    
    Returns (per item)
    ------------------
    dict
        - "tokens": torch.LongTensor [L_i] (no padding here)
        - "labels": torch.FloatTensor [K]
    

    Notes
    -----
    - No padding inside Dataset; padding is handled by `pad_collate`.
    - Never log raw text. Keep only derived numeric tensors.
    """
    def __init__(self, texts: Sequence[str], Y: np.ndarray, 
                 tok: Tokenizer, vocab: Vocab, *, max_len: int = 512
    ) -> None:
        # --- Validate inputs ---
        assert isinstance(texts, (list, tuple)) and len(texts) > 0, "[DATASET] texts must be non-empty Sequence[str]"
        Y = np.asarray(Y)
        assert Y.ndim == 2, f"[DATASET] Y.ndim={Y.ndim} expected 2"
        N, K = Y.shape
        assert N == len(texts), f"[DATASET] len(texts)={len(texts)} != Y.shape[0]={N}"
        assert Y.dtype in (np.uint8, np.bool_, bool), f"[DATASET] Y.dtype={Y.dtype} must be uint8/bool"
        if Y.dtype is not np.uint8:
            Y = Y.astype(np.uint8, copy=False)
        assert set(np.unique(Y)) <= {0, 1}, "[DATASET] Y must be multi-hot in {0, 1}"

        # --- store ---
        self.texts = list(texts)
        self.Y = np.ascontiguousarray(Y, dtype=np.uint8)
        self.tok = tok
        self.vocab = vocab
        self.max_len = int(max_len)
        self.K = int(self.Y.shape[1])

    def __len__(self) -> int:
        txts_len = len(self.texts)
        return txts_len

    def _encode_one(self, text: str) -> Tuple[List[int], int]:
        """
        Tokenize -> map to ids -> clip to max_len. No padding here.

        Returns
        -------
        ids : list[int] length L_i <= max_len
        L   : int, unclipped length after clipping (=len(ids))
        """
        toks = self.tok(text or "")
        ids = [ self.vocab.stoi.get(t, self.vocab.unk_idx) for t in toks ]
        if not ids:
            ids = [ self.vocab.unk_idx ]
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        
        return ids, len(ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        ids, _L = self._encode_one(text)
        x = torch.tensor(ids, dtype=torch.long)
        y = torch.from_numpy(self.Y[idx]).to(dtype=torch.float32)
        return {"tokens": x, "labels": y}
