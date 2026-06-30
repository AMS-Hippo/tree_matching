from __future__ import annotations

"""Extraction of matched subsequence bags from pairwise tree matches."""

from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

from embedder.tree_cluster import extract_seqs, extract_toy_alpha_seqs
from .bags import SequenceBag



def _extract_pair_sequences(
    pairs: Sequence[Tuple[int, int]],
    G: Any,
    H: Any,
    *,
    phi_name: str,
    seq_mode: str,
) -> Tuple[List[Any], List[Any]]:
    mode = str(seq_mode).lower().strip()
    if mode in {"alpha", "toy_alpha"}:
        return extract_toy_alpha_seqs(pairs, G, H, phi_name=phi_name)
    if mode in {"raw", "label", "labels"}:
        return extract_seqs(pairs, G, H, phi_name=phi_name)
    raise ValueError("seq_mode must be 'raw' or 'alpha'")



def extract_sequences_by_cluster(
    matches: Mapping[Tuple[int, int], Sequence[Tuple[int, int]]],
    labels: Sequence[int],
    tree_list: Sequence[Any],
    *,
    phi_name: str = "label",
    seq_mode: str = "alpha",
) -> Dict[int, List[List[Any]]]:
    """Compatibility-style extraction returning the old list-of-lists format."""
    gt = [int(x) for x in labels]
    out: Dict[int, List[List[Any]]] = {int(c): [] for c in sorted(set(gt))}
    for (i, j), pairs in matches.items():
        if gt[i] != gt[j]:
            continue
        seq_i, seq_j = _extract_pair_sequences(pairs, tree_list[i], tree_list[j], phi_name=phi_name, seq_mode=seq_mode)
        out[int(gt[i])].extend([list(seq_i), list(seq_j)])
    return out



def extract_sequence_bags_by_cluster(
    matches: Mapping[Tuple[int, int], Sequence[Tuple[int, int]]],
    labels: Sequence[int],
    tree_list: Sequence[Any],
    *,
    phi_name: str = "label",
    seq_mode: str = "alpha",
    deduplicate: bool = True,
    rebalance_by_tree: bool = True,
) -> Dict[int, SequenceBag]:
    """Build one weighted :class:`SequenceBag` per cluster.

    Each matched within-cluster pair contributes two observed sequences, one from
    each tree.  When ``rebalance_by_tree=True`` these observations are weighted so
    that each original tree contributes roughly total weight 1, regardless of how
    many pairwise matches it participates in.
    """
    gt = [int(x) for x in labels]
    clusters = sorted(set(gt))
    obs_by_cluster: Dict[int, List[Tuple[Tuple[Any, ...], int]]] = {c: [] for c in clusters}

    for (i, j), pairs in matches.items():
        if gt[i] != gt[j]:
            continue
        c = int(gt[i])
        seq_i, seq_j = _extract_pair_sequences(pairs, tree_list[i], tree_list[j], phi_name=phi_name, seq_mode=seq_mode)
        obs_by_cluster[c].append((tuple(seq_i), int(i)))
        obs_by_cluster[c].append((tuple(seq_j), int(j)))

    out: Dict[int, SequenceBag] = {}
    for c in clusters:
        observations = obs_by_cluster[c]
        if not observations:
            out[c] = SequenceBag.from_sequences([], name=f"cluster_{c}", metadata={"cluster": c, "seq_mode": seq_mode})
            continue
        source_counts = Counter(src for _seq, src in observations)
        weights = [1.0 / max(source_counts[src], 1) if rebalance_by_tree else 1.0 for _seq, src in observations]
        bag = SequenceBag.from_sequences(
            [seq for seq, _src in observations],
            weights=weights,
            name=f"cluster_{c}",
            metadata={
                "cluster": c,
                "seq_mode": seq_mode,
                "rebalance_by_tree": rebalance_by_tree,
                "n_source_trees": len(source_counts),
            },
            deduplicate=deduplicate,
        )
        out[c] = bag
    return out
