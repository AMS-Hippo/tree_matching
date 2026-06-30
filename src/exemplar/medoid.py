from __future__ import annotations

"""Simple center-string / medoid exemplar baseline."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from .bags import SequenceBag



def edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    a = list(seq_a)
    b = list(seq_b)
    m, n = len(a), len(b)
    if m == 0:
        return n
    if n == 0:
        return m
    prev = list(range(n + 1))
    cur = [0] * (n + 1)
    for i in range(1, m + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, cur = cur, prev
    return int(prev[n])


@dataclass
class MedoidResult:
    exemplar: List[Any]
    objective: float
    candidate_index: int
    pairwise_costs: np.ndarray



def infer_medoid_sequence(
    bag: SequenceBag,
    *,
    length_penalty: float = 0.0,
    return_result: bool = False,
) -> MedoidResult | List[Any]:
    if bag.n_unique == 0:
        out = MedoidResult(exemplar=[], objective=0.0, candidate_index=-1, pairwise_costs=np.zeros((0, 0), dtype=float))
        return out if return_result else out.exemplar

    seqs = bag.sequences
    weights = np.asarray(bag.weights, dtype=float)
    m = len(seqs)
    D = np.zeros((m, m), dtype=float)
    for i in range(m):
        for j in range(i + 1, m):
            d = float(edit_distance(seqs[i], seqs[j]))
            D[i, j] = d
            D[j, i] = d

    if length_penalty > 0:
        target_len = float(bag.median_length())
        penalties = np.asarray([abs(len(seq) - target_len) for seq in seqs], dtype=float)
    else:
        penalties = np.zeros(m, dtype=float)

    objectives = D.dot(weights) + float(length_penalty) * penalties
    best = int(np.argmin(objectives))
    result = MedoidResult(
        exemplar=list(seqs[best]),
        objective=float(objectives[best]),
        candidate_index=best,
        pairwise_costs=D,
    )
    return result if return_result else result.exemplar



def infer_cluster_exemplars(
    bags: Mapping[int, SequenceBag],
    *,
    method: str = "medoid",
    return_details: bool = False,
    **kwargs: Any,
) -> Dict[int, List[Any]] | Dict[int, Any]:
    mode = str(method).lower().strip()
    if mode == "medoid":
        if return_details:
            return {int(c): infer_medoid_sequence(bag, return_result=True, **kwargs) for c, bag in bags.items()}
        return {int(c): list(infer_medoid_sequence(bag, **kwargs)) for c, bag in bags.items()}
    if mode == "poa":
        from .poa import infer_cluster_poa_exemplars

        return infer_cluster_poa_exemplars(bags, return_details=return_details, **kwargs)
    if mode in {"likelihood", "em", "simple_ascent"}:
        from .likelihood import infer_cluster_likelihood_exemplars

        return infer_cluster_likelihood_exemplars(bags, return_details=return_details, **kwargs)
    raise NotImplementedError("method must currently be 'medoid', 'poa', or 'likelihood'")
