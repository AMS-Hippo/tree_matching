from __future__ import annotations

"""Weighted, deduplicated bags of cluster sequences.

The old workflow stored a plain list of noisy sequences for each cluster.  This
module keeps that view available, but adds a small structured representation with
weights and counts so that a later exemplar method can balance contributions from
individual trees and avoid O(n^2) duplicate domination.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


TokenSeq = Sequence[Any]



def _freeze_sequence(seq: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(seq)


@dataclass
class SequenceBag:
    sequences: List[Tuple[Any, ...]]
    weights: np.ndarray
    counts: np.ndarray
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sequences = [tuple(seq) for seq in self.sequences]
        self.weights = np.asarray(self.weights, dtype=float)
        self.counts = np.asarray(self.counts, dtype=int)
        if len(self.sequences) != int(self.weights.shape[0]) or len(self.sequences) != int(self.counts.shape[0]):
            raise ValueError("SequenceBag fields must have the same length")

    @classmethod
    def from_sequences(
        cls,
        sequences: Sequence[TokenSeq],
        *,
        weights: Optional[Sequence[float]] = None,
        name: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
        deduplicate: bool = True,
    ) -> "SequenceBag":
        seqs = [tuple(seq) for seq in sequences]
        if weights is None:
            ws = np.ones(len(seqs), dtype=float)
        else:
            ws = np.asarray(list(weights), dtype=float)
            if ws.shape != (len(seqs),):
                raise ValueError("weights must match the number of sequences")

        if not deduplicate:
            counts = np.ones(len(seqs), dtype=int)
            return cls(sequences=list(seqs), weights=ws, counts=counts, name=name, metadata=dict(metadata or {}))

        order: Dict[Tuple[Any, ...], int] = {}
        uniq: List[Tuple[Any, ...]] = []
        uniq_weights: List[float] = []
        uniq_counts: List[int] = []
        for seq, w in zip(seqs, ws):
            idx = order.get(seq)
            if idx is None:
                order[seq] = len(uniq)
                uniq.append(seq)
                uniq_weights.append(float(w))
                uniq_counts.append(1)
            else:
                uniq_weights[idx] += float(w)
                uniq_counts[idx] += 1
        return cls(
            sequences=uniq,
            weights=np.asarray(uniq_weights, dtype=float),
            counts=np.asarray(uniq_counts, dtype=int),
            name=name,
            metadata=dict(metadata or {}),
        )

    @property
    def n_unique(self) -> int:
        return len(self.sequences)

    @property
    def n_observations(self) -> int:
        return int(np.sum(self.counts)) if self.counts.size else 0

    @property
    def total_weight(self) -> float:
        return float(np.sum(self.weights)) if self.weights.size else 0.0

    def mean_length(self) -> float:
        if not self.sequences:
            return 0.0
        lens = np.asarray([len(seq) for seq in self.sequences], dtype=float)
        return float(np.average(lens, weights=np.maximum(self.weights, 1e-12)))

    def median_length(self) -> float:
        if not self.sequences:
            return 0.0
        expanded = []
        for seq, count in zip(self.sequences, self.counts):
            expanded.extend([len(seq)] * int(count))
        return float(np.median(np.asarray(expanded, dtype=float))) if expanded else 0.0

    def raw_sequences(self, *, repeat_by_counts: bool = True) -> List[List[Any]]:
        out: List[List[Any]] = []
        for seq, count in zip(self.sequences, self.counts):
            reps = int(count) if repeat_by_counts else 1
            for _ in range(reps):
                out.append(list(seq))
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "n_unique": self.n_unique,
            "n_observations": self.n_observations,
            "total_weight": self.total_weight,
            "mean_length": self.mean_length(),
            "median_length": self.median_length(),
        }
