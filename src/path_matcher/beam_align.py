
"""
Beam / pruned variant of Algorithm 1.

There isn't a single canonical "beam search" for this DP because the DP's state space is
a 2D grid (u,v) on two trees, and the paper's exact algorithm fills all n*m states.

The variant implemented here is a practical, *row-wise beam + candidate pruning* approach:

- For each node u in G (i.e., each DP row), we construct a candidate set of columns V:
    * V's coming from the BEAM of u's parent row (these carry forward high-scoring states),
    * plus V's suggested by a candidate generator for label ϕ_G(u) in H (e.g., same-label hits).

- We compute DP values A[u,v] only for those candidate columns (and whatever additional
  ancestor cells are required as dependencies).

- We then keep only the top `beam_width` columns in that row as the beam for descendants.

This is a standard pattern in sequence alignment / Viterbi-style problems:
keep only a small set of promising states per time step (here: per u / row).

Notes
-----
1) This is an *approximation*. If the true optimum relies on a state (u,v) that never
   enters the beam or candidate set, it may be missed.

2) The DP recurrence itself is unchanged from Algorithm 1. Boundary/root issues are
   handled via the same "shifted index" convention used in the exact implementation.

3) By default, we do *not* allocate dense A,C matrices; we store only the computed
   cells in per-row dictionaries.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import heapq
import math
import numpy as np

from .tree_data import TreeData
from .needleman_wunsch_tree import AlignmentResult, WeightFn, id_match
from .candidates import CandidateIndex, make_candidate_index_from_labels, select_subset


# Candidate function signature for custom candidate generation:
# return a list/sequence of v indices (0..m-1) in H that are plausible matches for u.
CandidateFn = Callable[[int, Any, TreeData], Sequence[int]]

# Optional predicate to decide whether to evaluate a match at (u,v).
# If provided and predicate returns False, option 3 is disabled at that cell.
MatchPredicate = Callable[[Any, Any], bool]


def align_trees_beam(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    beam_width: int = 200,
    candidate_fn: Optional[CandidateFn] = None,
    max_candidates_per_label: Optional[int] = 200,
    max_candidates_per_u: Optional[int] = None,
    candidate_select_mode: str = "first",
    seed: int = 0,
    match_predicate: Optional[MatchPredicate] = None,
    prefer_match_on_tie: bool = True,
) -> AlignmentResult:
    """
    Beam/pruned approximate matching for Algorithm 1.

    Parameters
    ----------
    G, H:
        Trees in TreeData form.
    w:
        Weight function w(ϕ_G(u), ϕ_H(v)). If None, uses id_match (1 if labels equal else 0).
    beam_width:
        Keep at most this many columns per row as the "beam" passed to descendants.
        Must be >= 1.
    candidate_fn:
        Optional custom candidate generator. If not provided, candidates are generated
        by indexing H's labels (exact-equality buckets), optionally truncated.
        Signature: candidate_fn(u_index, label_u, H) -> Sequence[v_index]
    max_candidates_per_label:
        Only used when candidate_fn is None. Caps the number of H nodes retained per label.
        Use None to keep all occurrences (may be huge for frequent labels).
    max_candidates_per_u:
        Optional additional cap applied per u after per-label truncation.
    candidate_select_mode:
        How to truncate candidates ("first", "last", "random", "spread").
    seed:
        RNG seed used for candidate selection when candidate_select_mode="random".
    match_predicate:
        Optional quick predicate on labels (label_u, label_v) -> bool.
        If provided and predicate returns False, option 3 is disabled (treated as -inf) at that cell.
    prefer_match_on_tie:
        If True, break ties as in the paper: prefer option 3 over 2 over 1.
        If False, prefer skipping on ties (reduces 0-weight "matches" when w can be 0).

    Returns
    -------
    AlignmentResult
    """
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")

    if w is None:
        w_fn = id_match
        w_is_id = True
    else:
        w_fn = w
        w_is_id = (w is id_match)

    n, m = G.n, H.n
    labelsG = G.label
    labelsH = H.label

    ancG = G.ancestors_shifted()
    ancH = H.ancestors_shifted()

    rng = np.random.default_rng(seed)

    cand_index: Optional[CandidateIndex]
    if candidate_fn is None:
        cand_index = make_candidate_index_from_labels(
            labelsH,
            max_per_label=max_candidates_per_label,
            select_mode=candidate_select_mode,
            seed=seed,
        )
    else:
        cand_index = None

    # Sparse DP storage: per-row dictionaries keyed by shifted column V (0..m).
    A_rows: List[Dict[int, float]] = [dict() for _ in range(n + 1)]
    C_rows: List[Dict[int, int]] = [dict() for _ in range(n + 1)]

    # Boundary (virtual row/col):
    A_rows[0][0] = 0.0
    C_rows[0][0] = 0

    # Beam columns per row (shifted V). Always include 0 as a safe fallback state.
    beam_cols: List[List[int]] = [[] for _ in range(n + 1)]
    beam_cols[0] = [0]

    best_score: float = 0.0
    best_U: int = 0
    best_V: int = 0

    def _tie_break(opt1: float, opt2: float, opt3: float) -> Tuple[float, int]:
        if prefer_match_on_tie:
            if opt3 >= opt2 and opt3 >= opt1:
                return opt3, 3
            if opt2 >= opt1:
                return opt2, 2
            return opt1, 1

        # Prefer skipping on ties.
        if opt1 >= opt2 and opt1 >= opt3:
            return opt1, 1
        if opt2 >= opt3:
            return opt2, 2
        return opt3, 3

    def ensure_cell(U: int, V: int) -> float:
        nonlocal best_score, best_U, best_V

        if V in A_rows[U]:
            return A_rows[U][V]

        stack: List[Tuple[int, int, int]] = [(U, V, 0)]  # (U,V,state)

        while stack:
            uU, vV, state = stack.pop()

            if vV in A_rows[uU]:
                continue

            if uU == 0 or vV == 0:
                A_rows[uU][vV] = 0.0
                C_rows[uU][vV] = 0
                continue

            u = uU - 1
            v = vV - 1
            aU = int(ancG[u])
            aV = int(ancH[v])

            if state == 0:
                stack.append((uU, vV, 1))
                stack.append((aU, vV, 0))
                stack.append((uU, aV, 0))
                stack.append((aU, aV, 0))
                continue

            opt1 = A_rows[aU][vV]
            opt2 = A_rows[uU][aV]
            base = A_rows[aU][aV]

            if w_is_id:
                w_uv = 1.0 if (labelsG[u] == labelsH[v]) else 0.0
            else:
                if match_predicate is not None and not bool(match_predicate(labelsG[u], labelsH[v])):
                    w_uv = -math.inf
                else:
                    w_uv = float(w_fn(labelsG[u], labelsH[v]))

            opt3 = w_uv + base

            val, choice = _tie_break(opt1, opt2, opt3)
            A_rows[uU][vV] = float(val)
            C_rows[uU][vV] = int(choice)

            if uU != 0 and vV != 0 and val > best_score:
                best_score = float(val)
                best_U = int(uU)
                best_V = int(vV)

        return A_rows[U][V]

    for U in range(1, n + 1):
        u = U - 1
        parentU = int(ancG[u])  # shifted parent row

        cand_cols_set = set(beam_cols[parentU])  # shifted V values

        if candidate_fn is None:
            assert cand_index is not None
            v_list = cand_index.candidates_for_label(labelsG[u])
        else:
            v_list = candidate_fn(u, labelsG[u], H)

        if max_candidates_per_u is not None:
            v_list = select_subset(v_list, max_candidates_per_u, mode=candidate_select_mode, rng=rng)

        for v in v_list:
            if v < 0 or v >= m:
                raise ValueError(f"candidate_fn returned v={v} outside [0, {m-1}]")
            cand_cols_set.add(int(v) + 1)

        cand_cols_set.add(0)

        for V in cand_cols_set:
            ensure_cell(U, int(V))

        items = [(score, V) for V, score in A_rows[U].items() if V != 0]
        if items:
            top = heapq.nlargest(beam_width, items, key=lambda x: x[0])
            beam = [0] + [int(V) for (_, V) in top]
        else:
            beam = [0]
        beam_cols[U] = beam

    if best_U == 0 or best_V == 0:
        return AlignmentResult(path_internal=[], score=0.0, end_internal=(0, 0), A=None, C=None)

    path_rev: List[Tuple[int, int]] = []
    U, V = best_U, best_V
    while U != 0 and V != 0:
        choice = int(C_rows[U].get(V, 0))
        if choice == 0:
            break

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
            raise RuntimeError(f"Invalid traceback choice {choice} at (U,V)=({U},{V})")

    path_rev.reverse()
    return AlignmentResult(
        path_internal=path_rev,
        score=float(best_score),
        end_internal=(best_U - 1, best_V - 1),
        A=None,
        C=None,
    )


def align_trees_beam_symmetric(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    beam_width: int = 200,
    candidate_fn: Optional[CandidateFn] = None,
    max_candidates_per_label: Optional[int] = 200,
    max_candidates_per_u: Optional[int] = None,
    candidate_select_mode: str = "first",
    seed: int = 0,
    match_predicate: Optional[MatchPredicate] = None,
    prefer_match_on_tie: bool = True,
) -> AlignmentResult:
    """
    Symmetric-by-default beam search: run beam alignment in both directions and keep the better score.

    Why this exists:
      - The exact Algorithm 1 is symmetric (up to swapping G and H),
      - but the row-wise beam approximation is not.
    """
    res_fwd = align_trees_beam(
        G, H,
        w=w,
        beam_width=beam_width,
        candidate_fn=candidate_fn,
        max_candidates_per_label=max_candidates_per_label,
        max_candidates_per_u=max_candidates_per_u,
        candidate_select_mode=candidate_select_mode,
        seed=seed,
        match_predicate=match_predicate,
        prefer_match_on_tie=prefer_match_on_tie,
    )

    res_rev = align_trees_beam(
        H, G,
        w=w,
        beam_width=beam_width,
        candidate_fn=candidate_fn,
        max_candidates_per_label=max_candidates_per_label,
        max_candidates_per_u=max_candidates_per_u,
        candidate_select_mode=candidate_select_mode,
        seed=seed,
        match_predicate=match_predicate,
        prefer_match_on_tie=prefer_match_on_tie,
    )

    if res_rev.score > res_fwd.score:
        # res_rev.path_internal is in (u_H, v_G); swap to (u_G, v_H)
        swapped_path = [(v, u) for (u, v) in res_rev.path_internal]
        end_uH, end_vG = res_rev.end_internal
        return AlignmentResult(
            path_internal=swapped_path,
            score=res_rev.score,
            end_internal=(end_vG, end_uH),
            A=None,
            C=None,
        )

    return res_fwd
