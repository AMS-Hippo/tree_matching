from __future__ import annotations

"""Sequence preprocessing helpers for exemplar inference.

These helpers are intentionally simple and dependency-light. They support two
operations that showed up repeatedly in the old exemplar notebooks:

1. Materialize a weighted :class:`SequenceBag` into an explicit list of sequences
   for methods that expect repeated observations.
2. Optionally denoise the resulting sequence list via repeated pairwise edit
   alignment / LCS-style consensus rounds.
"""

from typing import Any, List, Sequence

import numpy as np

from .bags import SequenceBag


def expand_bag_sequences(
    bag: SequenceBag,
    *,
    repeat_mode: str = 'weights',
    target_total_sequences: int = 96,
    max_total_sequences: int = 192,
) -> List[List[Any]]:
    """Materialize a bag into an explicit sequence list.

    Parameters
    ----------
    repeat_mode:
        ``'weights'`` uses bag weights, ``'counts'`` uses raw duplicate counts,
        and ``'unique'`` keeps one copy of each unique sequence.
    """
    if bag.n_unique == 0:
        return []
    mode = str(repeat_mode).lower().strip()
    target = max(1, min(int(target_total_sequences), int(max_total_sequences)))
    if mode in {'weights', 'weight'}:
        base = np.asarray(bag.weights, dtype=float)
    elif mode in {'counts', 'count'}:
        base = np.asarray(bag.counts, dtype=float)
    elif mode in {'unique', 'none', 'dedup'}:
        base = np.ones(bag.n_unique, dtype=float)
        target = bag.n_unique
    else:
        raise ValueError("repeat_mode must be 'weights', 'counts', or 'unique'")

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
    if len(out) > int(max_total_sequences):
        out = out[: int(max_total_sequences)]
    return out


def pair_align_consensus(a: Sequence[Any], b: Sequence[Any]) -> List[Any]:
    """Return a simple pairwise consensus using edit-distance alignment.

    Matches are kept, disagreements are dropped, and pure insertions/deletions
    are ignored. This is the same conservative consensus gadget used in the old
    MSA workflow notebooks.
    """
    a = list(a)
    b = list(b)
    n, m = len(a), len(b)

    dp = np.zeros((n + 1, m + 1), dtype=int)
    bt = np.zeros((n + 1, m + 1), dtype=np.int8)

    for i in range(1, n + 1):
        dp[i, 0] = i
        bt[i, 0] = 1
    for j in range(1, m + 1):
        dp[0, j] = j
        bt[0, j] = 2

    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            sub = dp[i - 1, j - 1] + (0 if ai == bj else 1)
            dele = dp[i - 1, j] + 1
            ins = dp[i, j - 1] + 1
            best = min(sub, dele, ins)
            dp[i, j] = best
            if best == sub:
                bt[i, j] = 0
            elif best == dele:
                bt[i, j] = 1
            else:
                bt[i, j] = 2

    i, j = n, m
    out: List[Any] = []
    while i > 0 or j > 0:
        move = bt[i, j]
        if move == 0:
            ai = a[i - 1]
            bj = b[j - 1]
            if ai == bj:
                out.append(ai)
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return out


def denoise_round(paths: Sequence[Sequence[Any]], *, seed: int = 0) -> List[List[Any]]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(paths), dtype=int)
    rng.shuffle(idx)

    nxt: List[List[Any]] = []
    k = 0
    while k + 1 < len(idx):
        x = paths[int(idx[k])]
        y = paths[int(idx[k + 1])]
        nxt.append(pair_align_consensus(x, y))
        k += 2
    if k < len(idx):
        nxt.append(list(paths[int(idx[k])]))
    return nxt


def denoise_paths(paths: Sequence[Sequence[Any]], *, rounds: int = 2, seed: int = 0) -> List[List[Any]]:
    cur = [list(x) for x in paths]
    for r in range(int(rounds)):
        cur = denoise_round(cur, seed=int(seed) + r)
    return cur


def prepare_sequences_from_bag(
    bag: SequenceBag,
    *,
    repeat_mode: str = 'weights',
    target_total_sequences: int = 96,
    max_total_sequences: int = 192,
    denoise_rounds: int = 0,
    denoise_seed: int = 0,
) -> List[List[Any]]:
    seqs = expand_bag_sequences(
        bag,
        repeat_mode=repeat_mode,
        target_total_sequences=target_total_sequences,
        max_total_sequences=max_total_sequences,
    )
    if int(denoise_rounds) > 0 and seqs:
        seqs = denoise_paths(seqs, rounds=int(denoise_rounds), seed=int(denoise_seed))
    return [list(seq) for seq in seqs]


__all__ = [
    'expand_bag_sequences',
    'pair_align_consensus',
    'denoise_round',
    'denoise_paths',
    'prepare_sequences_from_bag',
]
