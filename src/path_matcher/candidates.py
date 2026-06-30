
"""
Candidate generation / pruning helpers.

These are shared by beam search and sparse candidate alignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


def build_label_index(labels: Sequence[Any]) -> Dict[Any, List[int]]:
    """
    Build an inverted index: label -> list of node indices where that label occurs.
    """
    index: Dict[Any, List[int]] = {}
    for i, lab in enumerate(labels):
        index.setdefault(lab, []).append(i)
    return index


def select_subset(
    items: Sequence[int],
    k: Optional[int],
    *,
    mode: str = "first",
    rng: Optional[np.random.Generator] = None,
) -> List[int]:
    """
    Select up to k items from an input sequence.

    Parameters
    ----------
    items:
        Sequence of ints (typically node ids), assumed deterministic order.
    k:
        Max number to return. If None, returns all items.
    mode:
        "first"  -> first k items
        "last"   -> last k items
        "random" -> random sample without replacement (deterministic if rng is seeded)
        "spread" -> approximately evenly spaced selection across the list

    Returns
    -------
    list[int]
    """
    if k is None or k >= len(items):
        return list(items)

    if k <= 0:
        return []

    mode = mode.lower()
    if mode == "first":
        return list(items[:k])
    if mode == "last":
        return list(items[-k:])

    if mode == "random":
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(items), size=k, replace=False)
        idx.sort()
        return [items[int(i)] for i in idx]

    if mode == "spread":
        if k == 1:
            return [items[len(items) // 2]]
        pos = np.linspace(0, len(items) - 1, num=k)
        idx = np.unique(np.round(pos).astype(int))
        out = [items[int(i)] for i in idx[:k]]
        if len(out) < k:
            out.extend(items[: (k - len(out))])
        return out[:k]

    raise ValueError("mode must be one of: 'first', 'last', 'random', 'spread'")


@dataclass(frozen=True)
class CandidateIndex:
    """
    Candidate index for quick retrieval of candidate nodes given a label.
    """
    label_to_nodes: Mapping[Any, Sequence[int]]

    def candidates_for_label(self, label: Any) -> Sequence[int]:
        return self.label_to_nodes.get(label, ())


def make_candidate_index_from_labels(
    labels_H: Sequence[Any],
    *,
    max_per_label: Optional[int] = None,
    select_mode: str = "first",
    seed: int = 0,
) -> CandidateIndex:
    """
    Build a CandidateIndex from H's labels, optionally truncating each label bucket.
    """
    index = build_label_index(labels_H)

    rng = np.random.default_rng(seed)
    if max_per_label is not None:
        index = {
            lab: select_subset(nodes, max_per_label, mode=select_mode, rng=rng)
            for lab, nodes in index.items()
        }

    return CandidateIndex(label_to_nodes=index)
