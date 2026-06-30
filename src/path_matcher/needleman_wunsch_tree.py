
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



def _require_path_template(H: TreeData) -> None:
    """
    Validate that H is a rooted path in internal TreeData order.

    For a path template, the internal parent array must be exactly
    [-1, 0, 1, ..., m-2]. This is the natural root-to-leaf order used by the
    shifted DP tables in this module.
    """
    parent = np.asarray(H.parent, dtype=np.int64)
    if parent.shape != (H.n,):
        raise ValueError("Template parent array has invalid shape")
    if H.n == 0:  # defensive; TreeData itself rejects empty trees
        raise ValueError("Template cannot be empty")
    if int(parent[0]) != -1:
        raise ValueError("Template path must have parent[0] == -1")
    if H.n > 1:
        expected = np.arange(H.n - 1, dtype=np.int64)
        if not np.array_equal(parent[1:], expected):
            raise ValueError(
                "mode='template_repeat' expects H, the second input, to be a single root-to-leaf path "
                "with internal parent array [-1, 0, 1, ..., m-2]."
            )


def _align_tree_to_repeating_template_penalized(
    G: TreeData,
    H: TreeData,
    *,
    w_fn: WeightFn,
    w_is_id: bool,
    dtype: Any,
    return_matrices: bool,
    prefer_match_on_tie: bool,
    repeat_penalty: float,
) -> AlignmentResult:
    """
    Penalized template-repeat alignment.

    This helper uses two DP states so that the linear repeat penalty is applied
    only to the second and later consecutive use of the same template vertex.
    It is used when repeat_penalty > 0. The unpenalized case below keeps the
    one-matrix implementation as close as possible to align_trees_algorithm1.
    """
    n, m = G.n, H.n

    # F[U,V] is the usual best score using tree ancestors up to U and template
    # prefix up to V. R[U,V] is the best score whose last matched template state
    # is exactly V, so matching V again is a true repetition.
    F = np.zeros((n + 1, m + 1), dtype=dtype)
    R = np.full((n + 1, m + 1), -np.inf, dtype=dtype)
    CF = np.zeros((n + 1, m + 1), dtype=np.uint8)
    CR = np.zeros((n + 1, m + 1), dtype=np.uint8)

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

            if w_is_id:
                w_uv = 1.0 if (lab_u == labelsH[v]) else 0.0
            else:
                w_uv = float(w_fn(lab_u, labelsH[v]))

            # Exact-last-template state R.
            opt_skip_exact = R[ancU, V]
            opt_first = w_uv + F[ancU, ancV]
            opt_repeat = w_uv - repeat_penalty + R[ancU, V]

            if prefer_match_on_tie:
                # Tie-break toward matching: repeat > first > skip-exact.
                if opt_repeat >= opt_first and opt_repeat >= opt_skip_exact:
                    R[U, V] = opt_repeat
                    CR[U, V] = 3
                elif opt_first >= opt_skip_exact:
                    R[U, V] = opt_first
                    CR[U, V] = 2
                else:
                    R[U, V] = opt_skip_exact
                    CR[U, V] = 1
            else:
                # Tie-break toward skips, then non-repeat starts, then repeats.
                if opt_skip_exact >= opt_first and opt_skip_exact >= opt_repeat:
                    R[U, V] = opt_skip_exact
                    CR[U, V] = 1
                elif opt_first >= opt_repeat:
                    R[U, V] = opt_first
                    CR[U, V] = 2
                else:
                    R[U, V] = opt_repeat
                    CR[U, V] = 3

            # Prefix/local state F.
            opt1 = F[ancU, V]  # skip tree vertex u
            opt2 = F[U, ancV]  # skip template vertex v
            opt3 = R[U, V]     # use template vertex v at least once

            if prefer_match_on_tie:
                # Tie-break in favor of larger option: 3 > 2 > 1.
                if opt3 >= opt2 and opt3 >= opt1:
                    F[U, V] = opt3
                    CF[U, V] = 3
                elif opt2 >= opt1:
                    F[U, V] = opt2
                    CF[U, V] = 2
                else:
                    F[U, V] = opt1
                    CF[U, V] = 1
            else:
                # Tie-break toward skips: 1 > 2 > 3.
                if opt1 >= opt2 and opt1 >= opt3:
                    F[U, V] = opt1
                    CF[U, V] = 1
                elif opt2 >= opt3:
                    F[U, V] = opt2
                    CF[U, V] = 2
                else:
                    F[U, V] = opt3
                    CF[U, V] = 3

    sub = F[1:, 1:]
    flat = int(np.argmax(sub))
    U_star = flat // m + 1
    V_star = flat % m + 1
    score = float(sub.flat[flat])

    path_rev: List[Tuple[int, int]] = []
    U, V = U_star, V_star
    state = 0  # 0 = F, 1 = R
    while U != 0 and V != 0:
        if state == 0:
            choice = int(CF[U, V])
            if choice == 1:
                U = int(ancG[U - 1])
            elif choice == 2:
                V = int(ancH[V - 1])
            elif choice == 3:
                state = 1
            else:
                raise RuntimeError(f"Invalid traceback state CF[{U},{V}]={choice}")
        else:
            choice = int(CR[U, V])
            if choice == 1:
                U = int(ancG[U - 1])
            elif choice == 2:
                path_rev.append((U - 1, V - 1))
                U = int(ancG[U - 1])
                V = int(ancH[V - 1])
                state = 0
            elif choice == 3:
                path_rev.append((U - 1, V - 1))
                U = int(ancG[U - 1])
                # V stays fixed: this is the template self-loop.
            else:
                raise RuntimeError(f"Invalid traceback state CR[{U},{V}]={choice}")

    path_rev.reverse()

    if return_matrices:
        # A is the public score table. C is the prefix-state choice table; the
        # exact-last-template choice table CR is intentionally internal.
        return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=F, C=CF)
    return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=None, C=None)


def align_tree_to_repeating_template(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    dtype: Any = np.float32,
    return_matrices: bool = False,
    prefer_match_on_tie: bool = True,
    repeat_penalty: float = 0.0,
) -> AlignmentResult:
    """
    Align a tree path in G to a path template H with repeatable template states.

    This is the asymmetric template-repeat variant of Algorithm 1. H must be a
    single rooted path. The tree-side path is still consumed in ancestor order,
    but a match transition may keep the same template vertex V rather than
    consuming its parent/previous template vertex.

    The unpenalized recurrence is

        A[U,V] = max(
            A[ancG(U), V],                  # skip tree vertex U
            A[U, ancH(V)],                  # skip template vertex V
            w(U,V) + A[ancG(U), V],         # match U to reusable template V
        )

    where H is a path, so ancH(V) is simply the previous template position. The
    traceback for the match transition moves to (ancG(U), V), leaving V fixed.

    Parameters
    ----------
    G:
        The ordinary labelled tree.
    H:
        The path template. Its internal parent array must be [-1, 0, 1, ...].
    w:
        Weight function w(ϕG(u), ϕH(v)) -> score. If None, uses id_match.
    repeat_penalty:
        Optional linear penalty for a second and later consecutive use of the
        same template vertex. A run of length r against one template state pays
        repeat_penalty * (r - 1). The default 0.0 gives the one-matrix recurrence
        above. Positive penalties use an equivalent two-state DP so the first
        use of each template state is not penalized.

    Returns
    -------
    AlignmentResult
    """
    _require_path_template(H)
    repeat_penalty = float(repeat_penalty)
    if repeat_penalty < 0.0:
        raise ValueError("repeat_penalty must be nonnegative")

    if w is None:
        w_fn = id_match
        w_is_id = True
    else:
        w_fn = w
        w_is_id = (w is id_match)

    if repeat_penalty > 0.0:
        return _align_tree_to_repeating_template_penalized(
            G,
            H,
            w_fn=w_fn,
            w_is_id=w_is_id,
            dtype=dtype,
            return_matrices=return_matrices,
            prefer_match_on_tie=prefer_match_on_tie,
            repeat_penalty=repeat_penalty,
        )

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

            # Template-repeat match: consume the tree vertex but leave the
            # template vertex fixed. This is the only recurrence difference from
            # align_trees_algorithm1, whose match term uses A[ancU, ancV].
            opt3 = w_uv + A[ancU, V]

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

    # Traceback. For choice 3, V is not consumed: the template state is reused.
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
            # V stays fixed: this is the template self-loop.
        else:
            raise RuntimeError(f"Invalid traceback state C[{U},{V}]={choice}")

    path_rev.reverse()

    if return_matrices:
        return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=A, C=C)
    return AlignmentResult(path_internal=path_rev, score=score, end_internal=(U_star - 1, V_star - 1), A=None, C=None)
