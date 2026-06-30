from __future__ import annotations

"""POA-style exemplar inference.

This module provides a first pass at a partial-order-alignment-style consensus
baseline for cluster exemplar inference.

Backends
--------
- ``backend='pyabpoa'`` uses the optional ``pyabpoa`` package when available.
- ``backend='progressive'`` uses a lightweight pure-Python progressive
  consensus fallback that works on arbitrary token sequences and weighted
  :class:`SequenceBag` objects.
- ``backend='auto'`` tries ``pyabpoa`` first and falls back to the progressive
  implementation when the package is unavailable or the token vocabulary is too
  large for the safe single-character encoding used here.

The fallback is not a full POA implementation, but it is close enough to serve
as a useful pass-2 benchmark baseline and avoids making the whole workflow
unusable on machines without the external package installed.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import string
import numpy as np

try:  # pragma: no cover - optional dependency
    import pyabpoa as _pyabpoa  # type: ignore
except Exception:  # pragma: no cover
    _pyabpoa = None

from .bags import SequenceBag
from .medoid import edit_distance, infer_medoid_sequence
from .preprocess import prepare_sequences_from_bag

SAFE_ASCII_ALPHABET = list(string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation)


@dataclass
class POAResult:
    exemplar: List[Any]
    backend: str
    seed_index: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _POAColumn:
    token_weights: Dict[Any, float]
    gap_weight: float = 0.0


@dataclass
class _ProgressiveConsensusResult:
    exemplar: List[Any]
    columns: List[_POAColumn]
    seed_index: int
    total_weight: float
    order: List[int]


def _choose_seed_index(bag: SequenceBag, *, seed_mode: str = 'medoid') -> int:
    if bag.n_unique == 0:
        return -1
    mode = str(seed_mode).lower().strip()
    if mode == 'medoid':
        res = infer_medoid_sequence(bag, return_result=True)
        return int(res.candidate_index)
    if mode in {'highest_weight', 'weight'}:
        return int(np.argmax(np.asarray(bag.weights, dtype=float)))
    if mode in {'longest', 'max_len'}:
        lens = np.asarray([len(seq) for seq in bag.sequences], dtype=int)
        return int(np.argmax(lens))
    raise ValueError("seed_mode must be 'medoid', 'highest_weight', or 'longest'")


def _order_indices(bag: SequenceBag, seed_index: int, *, order_mode: str = 'closest_first') -> List[int]:
    if bag.n_unique == 0:
        return []
    order = [i for i in range(bag.n_unique) if i != seed_index]
    mode = str(order_mode).lower().strip()
    if mode in {'input', 'original'}:
        return [seed_index] + order
    seed = bag.sequences[seed_index]
    if mode in {'closest_first', 'closest', 'nearest'}:
        order.sort(key=lambda i: (edit_distance(seed, bag.sequences[i]), -float(bag.weights[i]), i))
    elif mode in {'heaviest_first', 'weight'}:
        order.sort(key=lambda i: (-float(bag.weights[i]), edit_distance(seed, bag.sequences[i]), i))
    else:
        raise ValueError("order_mode must be 'closest_first', 'heaviest_first', or 'input'")
    return [seed_index] + order


def _nw_ops(
    seq_a: Sequence[Any],
    seq_b: Sequence[Any],
    *,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
) -> List[str]:
    """Needleman–Wunsch alignment path between token sequences.

    Returns a list of ops over ``{'M', 'D', 'I'}`` where:
    - ``M``: advance in both sequences,
    - ``D``: token from ``seq_a`` aligned to a gap in ``seq_b``,
    - ``I``: gap in ``seq_a`` aligned to a token from ``seq_b``.
    """
    a = list(seq_a)
    b = list(seq_b)
    m, n = len(a), len(b)
    dp = np.zeros((m + 1, n + 1), dtype=float)
    trace = np.zeros((m + 1, n + 1), dtype=np.int8)  # 0 diag, 1 up(D), 2 left(I)

    for i in range(1, m + 1):
        dp[i, 0] = dp[i - 1, 0] + gap_score
        trace[i, 0] = 1
    for j in range(1, n + 1):
        dp[0, j] = dp[0, j - 1] + gap_score
        trace[0, j] = 2

    for i in range(1, m + 1):
        ai = a[i - 1]
        for j in range(1, n + 1):
            bj = b[j - 1]
            diag = dp[i - 1, j - 1] + (match_score if ai == bj else mismatch_score)
            up = dp[i - 1, j] + gap_score
            left = dp[i, j - 1] + gap_score
            best = diag
            code = 0
            if up > best:
                best = up
                code = 1
            if left > best:
                best = left
                code = 2
            dp[i, j] = best
            trace[i, j] = code

    ops: List[str] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and trace[i, j] == 0:
            ops.append('M')
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or trace[i, j] == 1):
            ops.append('D')
            i -= 1
        else:
            ops.append('I')
            j -= 1
    ops.reverse()
    return ops


def _column_consensus_token(col: _POAColumn, *, gap_vote_threshold: float = 1.0) -> Optional[Any]:
    if not col.token_weights:
        return None
    tok, tok_w = max(col.token_weights.items(), key=lambda kv: (float(kv[1]), str(kv[0])))
    if float(tok_w) >= float(gap_vote_threshold) * float(col.gap_weight):
        return tok
    return None


def _current_rep_sequence(columns: Sequence[_POAColumn]) -> List[Any]:
    rep: List[Any] = []
    for col in columns:
        if not col.token_weights:
            rep.append(None)
            continue
        tok = max(col.token_weights.items(), key=lambda kv: (float(kv[1]), str(kv[0])))[0]
        rep.append(tok)
    return rep


def _progressive_consensus(
    bag: SequenceBag,
    *,
    seed_mode: str = 'medoid',
    order_mode: str = 'closest_first',
    gap_vote_threshold: float = 1.0,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
) -> _ProgressiveConsensusResult:
    if bag.n_unique == 0:
        return _ProgressiveConsensusResult(exemplar=[], columns=[], seed_index=-1, total_weight=0.0, order=[])

    seed_index = _choose_seed_index(bag, seed_mode=seed_mode)
    order = _order_indices(bag, seed_index, order_mode=order_mode)
    first = order[0]
    seed_seq = list(bag.sequences[first])
    seed_w = float(bag.weights[first])
    columns = [_POAColumn(token_weights={tok: seed_w}, gap_weight=0.0) for tok in seed_seq]
    total_prev_weight = seed_w

    for idx in order[1:]:
        seq = list(bag.sequences[idx])
        w = float(bag.weights[idx])
        rep = _current_rep_sequence(columns)
        ops = _nw_ops(rep, seq, match_score=match_score, mismatch_score=mismatch_score, gap_score=gap_score)

        new_cols: List[_POAColumn] = []
        col_ptr = 0
        tok_ptr = 0
        for op in ops:
            if op == 'M':
                col = columns[col_ptr]
                tok = seq[tok_ptr]
                col.token_weights[tok] = col.token_weights.get(tok, 0.0) + w
                new_cols.append(col)
                col_ptr += 1
                tok_ptr += 1
            elif op == 'D':
                col = columns[col_ptr]
                col.gap_weight += w
                new_cols.append(col)
                col_ptr += 1
            elif op == 'I':
                tok = seq[tok_ptr]
                new_cols.append(_POAColumn(token_weights={tok: w}, gap_weight=total_prev_weight))
                tok_ptr += 1
            else:  # pragma: no cover
                raise RuntimeError(f'Unexpected alignment op: {op!r}')
        columns = new_cols
        total_prev_weight += w

    exemplar = [tok for tok in (_column_consensus_token(col, gap_vote_threshold=gap_vote_threshold) for col in columns) if tok is not None]
    return _ProgressiveConsensusResult(
        exemplar=list(exemplar),
        columns=columns,
        seed_index=seed_index,
        total_weight=total_prev_weight,
        order=order,
    )


def _expand_bag_for_external_poa(
    bag: SequenceBag,
    *,
    repeat_mode: str = 'weights',
    target_total_repeats: int = 96,
    max_total_repeats: int = 192,
    denoise_rounds: int = 0,
    denoise_seed: int = 0,
) -> List[List[Any]]:
    if bag.n_unique == 0:
        return []
    mode = str(repeat_mode).lower().strip()
    target = max(1, min(int(target_total_repeats), int(max_total_repeats)))
    if mode in {'weights', 'weight'}:
        base = np.asarray(bag.weights, dtype=float)
    elif mode in {'counts', 'count'}:
        base = np.asarray(bag.counts, dtype=float)
    elif mode in {'none', 'unique', 'dedup'}:
        base = np.ones(bag.n_unique, dtype=float)
        target = bag.n_unique
    else:
        raise ValueError("repeat_mode must be 'weights', 'counts', or 'none'")

    base = np.maximum(base, 0.0)
    if float(base.sum()) <= 0:
        base = np.ones_like(base, dtype=float)
    probs = base / float(base.sum())
    raw = probs * float(target)
    reps = np.floor(raw).astype(int)
    residual = target - int(reps.sum())
    if residual > 0:
        frac = raw - reps
        order = np.argsort(-frac)
        for idx in order[:residual]:
            reps[int(idx)] += 1
    reps = np.maximum(reps, 1)

    out: List[List[Any]] = []
    for seq, rep in zip(bag.sequences, reps):
        for _ in range(int(rep)):
            out.append(list(seq))
    if len(out) > int(max_total_repeats):
        out = out[: int(max_total_repeats)]
    return out


def _encode_tokens_to_safe_ascii(seqs: Sequence[Sequence[Any]]) -> Tuple[List[str], Dict[str, Any]]:
    vocab: List[Any] = []
    seen: Dict[Any, None] = {}
    for seq in seqs:
        for tok in seq:
            if tok not in seen:
                seen[tok] = None
                vocab.append(tok)
    if len(vocab) > len(SAFE_ASCII_ALPHABET):
        raise ValueError(
            f'pyabpoa safe token encoder currently supports at most {len(SAFE_ASCII_ALPHABET)} unique tokens; '
            f'got {len(vocab)}.'
        )
    tok_to_char = {tok: SAFE_ASCII_ALPHABET[i] for i, tok in enumerate(vocab)}
    char_to_tok = {ch: tok for tok, ch in tok_to_char.items()}
    strings = [''.join(tok_to_char[tok] for tok in seq) for seq in seqs]
    return strings, char_to_tok


def _infer_with_pyabpoa(
    bag: SequenceBag,
    *,
    repeat_mode: str = 'weights',
    target_total_repeats: int = 96,
    max_total_repeats: int = 192,
    denoise_rounds: int = 0,
    denoise_seed: int = 0,
    match: int = 2,
    mismatch: int = 4,
    gap_open1: int = 4,
    gap_ext1: int = 2,
    aln_mode: str = 'g',
) -> POAResult:
    if _pyabpoa is None:  # pragma: no cover
        raise ImportError('pyabpoa is not installed')
    expanded = _expand_bag_for_external_poa(
        bag,
        repeat_mode=repeat_mode,
        target_total_repeats=target_total_repeats,
        max_total_repeats=max_total_repeats,
    )
    if not expanded:
        return POAResult(exemplar=[], backend='pyabpoa', metadata={'n_input_sequences': 0})

    strings, char_to_tok = _encode_tokens_to_safe_ascii(expanded)
    aligner = _pyabpoa.msa_aligner(
        aln_mode=aln_mode,
        match=int(match),
        mismatch=int(mismatch),
        gap_open1=int(gap_open1),
        gap_ext1=int(gap_ext1),
    )
    res = aligner.msa(strings, out_cons=True, out_msa=False)
    consensus_str = '' if not getattr(res, 'cons_seq', None) else str(res.cons_seq[0])
    exemplar = [char_to_tok[ch] for ch in consensus_str]
    return POAResult(
        exemplar=exemplar,
        backend='pyabpoa',
        metadata={
            'n_input_sequences': len(strings),
            'token_vocab_size': len(char_to_tok),
            'repeat_mode': repeat_mode,
            'target_total_repeats': int(target_total_repeats),
            'max_total_repeats': int(max_total_repeats),
        },
    )


def infer_poa_sequence(
    bag: SequenceBag,
    *,
    backend: str = 'auto',
    repeat_mode: str = 'weights',
    target_total_repeats: int = 96,
    max_total_repeats: int = 192,
    denoise_rounds: int = 0,
    denoise_seed: int = 0,
    seed_mode: str = 'medoid',
    order_mode: str = 'closest_first',
    gap_vote_threshold: float = 1.0,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_score: int = -1,
    match: int = 2,
    mismatch: int = 4,
    gap_open1: int = 4,
    gap_ext1: int = 2,
    aln_mode: str = 'g',
    return_result: bool = False,
) -> POAResult | List[Any]:
    if bag.n_unique == 0:
        out = POAResult(exemplar=[], backend='empty', metadata={'n_input_sequences': 0})
        return out if return_result else out.exemplar

    backend_req = str(backend).lower().strip()
    if backend_req not in {'auto', 'pyabpoa', 'progressive'}:
        raise ValueError("backend must be 'auto', 'pyabpoa', or 'progressive'")

    working_bag = bag
    if int(denoise_rounds) > 0:
        prepared = prepare_sequences_from_bag(
            bag,
            repeat_mode=repeat_mode,
            target_total_sequences=target_total_repeats,
            max_total_sequences=max_total_repeats,
            denoise_rounds=int(denoise_rounds),
            denoise_seed=int(denoise_seed),
        )
        working_bag = SequenceBag.from_sequences(prepared, deduplicate=False, name=bag.name, metadata=dict(getattr(bag, 'metadata', {})))
        repeat_mode = 'unique'
        target_total_repeats = max(1, working_bag.n_unique)
        max_total_repeats = max(1, working_bag.n_unique)

    if backend_req in {'auto', 'pyabpoa'} and _pyabpoa is not None:
        try:
            out = _infer_with_pyabpoa(
                working_bag,
                repeat_mode=repeat_mode,
                target_total_repeats=target_total_repeats,
                max_total_repeats=max_total_repeats,
                match=match,
                mismatch=mismatch,
                gap_open1=gap_open1,
                gap_ext1=gap_ext1,
                aln_mode=aln_mode,
            )
            return out if return_result else out.exemplar
        except Exception as e:
            if backend_req == 'pyabpoa':
                raise
            fail_msg = f'{type(e).__name__}: {e}'
        else:  # pragma: no cover
            fail_msg = None
    else:
        fail_msg = None if backend_req != 'pyabpoa' else 'pyabpoa unavailable'

    prog = _progressive_consensus(
        working_bag,
        seed_mode=seed_mode,
        order_mode=order_mode,
        gap_vote_threshold=gap_vote_threshold,
        match_score=match_score,
        mismatch_score=mismatch_score,
        gap_score=gap_score,
    )
    out = POAResult(
        exemplar=list(prog.exemplar),
        backend='progressive',
        seed_index=int(prog.seed_index),
        metadata={
            'n_input_sequences': int(working_bag.n_unique),
            'total_weight': float(working_bag.total_weight),
            'denoise_rounds': int(denoise_rounds),
            'order_mode': order_mode,
            'seed_mode': seed_mode,
            'gap_vote_threshold': float(gap_vote_threshold),
            'fallback_reason': fail_msg,
        },
    )
    return out if return_result else out.exemplar


def infer_cluster_poa_exemplars(
    bags: Mapping[int, SequenceBag],
    *,
    return_details: bool = False,
    **kwargs: Any,
) -> Dict[int, POAResult] | Dict[int, List[Any]]:
    results = {int(c): infer_poa_sequence(bag, return_result=True, **kwargs) for c, bag in bags.items()}
    if return_details:
        return results
    return {int(c): list(res.exemplar) for c, res in results.items()}


__all__ = [
    'POAResult',
    'infer_poa_sequence',
    'infer_cluster_poa_exemplars',
]
