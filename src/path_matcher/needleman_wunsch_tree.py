
"""
Core implementation of Algorithm 1 ("Basic Matching Algorithm") from the paper.

Algorithm 1 computes
  A[u,v] = max{
      A[ancG(u), v],
      A[u, ancH(v)],
      w(ϕG(u), ϕH(v)) + A[ancG(u), ancH(v)]
  }
and stores the argmax choice in C[u,v] ∈ {1,2,3} (ties broken in favor of larger option).

It then starts from ℓ = argmax_{u,v} A[u,v] and traces back, producing a matched path.

This module implements the same logic using a "shifted" DP table so that:
- DP row/col 0 corresponds to the virtual index -1 in the paper.
- Actual nodes u∈{0,...,n-1} correspond to DP row u+1 (and similarly for v).

This eliminates special boundary cases at the root.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np

from .tree_data import TreeData


WeightFn = Callable[[Any, Any], float]


def id_match(a: Any, b: Any) -> float:
    """Default weight: 1 if labels are equal, else 0."""
    return 1.0 if a == b else 0.0


@dataclass(frozen=True)
class AlignmentResult:
    """
    Output of Algorithm 1.

    `path_internal` uses internal indices 0..n-1 / 0..m-1 (after any reordering).
    The caller can map these back to original ids via TreeData.orig_index.
    """
    path_internal: List[Tuple[int, int]]
    score: float
    end_internal: Tuple[int, int]  # (u*, v*) where A achieves its maximum
    A: Optional[np.ndarray] = None
    C: Optional[np.ndarray] = None


def align_trees_algorithm1(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    dtype: Any = np.float32,
    return_matrices: bool = False,
    prefer_match_on_tie: bool = True,
) -> AlignmentResult:
    """
    Compute best-matching path and score between two trees using Algorithm 1.

    Parameters
    ----------
    G, H:
        Trees in TreeData form. Must satisfy TreeData invariants (root at 0, parent<child).
    w:
        Weight function w(ϕG(u), ϕH(v)) -> score. If None, uses id_match.
    dtype:
        Float dtype for DP matrix A (np.float32 recommended for large problems).
    return_matrices:
        If True, include A and C matrices in the result (useful for debugging).
        Warning: A and C are O(nm) memory.
    prefer_match_on_tie:
        If True, break ties as in the original implementation: option 3
        (diagonal/match) over option 2 over option 1. If False, prefer skip
        moves on ties, which avoids many zero-score diagonal transitions when
        non-matches have weight 0.

    Returns
    -------
    AlignmentResult
    """
    if w is None:
        w_fn = id_match
        w_is_id = True
    else:
        w_fn = w
        w_is_id = (w is id_match)

    n, m = G.n, H.n

    A = np.zeros((n + 1, m + 1), dtype=dtype)
    C = np.zeros((n + 1, m + 1), dtype=np.uint8)

    ancG = G.ancestors_shifted()
    ancH = H.ancestors_shifted()

    labelsG = G.label
    labelsH = H.label

    for U in range(1, n + 1):
        u = U - 1
        ancU = int(ancG[u])
        lab_u = labelsG[u]

        for V in range(1, m + 1):
            v = V - 1
            ancV = int(ancH[v])

            opt1 = A[ancU, V]
            opt2 = A[U, ancV]

            if w_is_id:
                w_uv = 1.0 if (lab_u == labelsH[v]) else 0.0
            else:
                w_uv = float(w_fn(lab_u, labelsH[v]))

            opt3 = w_uv + A[ancU, ancV]

            if prefer_match_on_tie:
                # Tie-break in favor of larger option: 3 > 2 > 1.
                if opt3 >= opt2 and opt3 >= opt1:
                    A[U, V] = opt3
                    C[U, V] = 3
                elif opt2 >= opt1:
                    A[U, V] = opt2
                    C[U, V] = 2
                else:
                    A[U, V] = opt1
                    C[U, V] = 1
            else:
                # Tie-break toward skips: 1 > 2 > 3.
                if opt1 >= opt2 and opt1 >= opt3:
                    A[U, V] = opt1
                    C[U, V] = 1
                elif opt2 >= opt3:
                    A[U, V] = opt2
                    C[U, V] = 2
                else:
                    A[U, V] = opt3
                    C[U, V] = 3

    # ℓ = argmax over actual nodes: argmax A[1:,1:].
    sub = A[1:, 1:]
    flat = int(np.argmax(sub))
    U_star = flat // m + 1
    V_star = flat % m + 1
    score = float(sub.flat[flat])

    # Traceback.
    path_rev: List[Tuple[int, int]] = []
    U, V = U_star, V_star
    while U != 0 and V != 0:
        choice = int(C[U, V])
        if choice == 3:
            path_rev.append((U - 1, V - 1))

        if choice == 1:
            U = int(ancG[U - 1])
        elif choice == 2:
            V = int(ancH[V - 1])
        elif choice == 3:
            U = int(ancG[U - 1])
            V = int(ancH[V - 1])
        else:
            raise RuntimeError(f"Invalid traceback state C[{U},{V}]={choice}")

    path_rev.reverse()

    if return_matrices:
        return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=A, C=C)
    return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=None, C=None)
