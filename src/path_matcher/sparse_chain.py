"""
Exact maximum-weight chain search on a sparse set of candidate node pairs.

The tree-path matching objective is a maximum-weight chain in the product of
strict ancestor orders.  If C is a set of candidate pairs, then for each
(u, v) in C,

    D(u, v) = weight(u, v) + max D(u', v'),

where u' is a strict ancestor of u and v' is a strict ancestor of v.  The
maximum over C is therefore exact relative to C.  It is also exact for the full
matching problem whenever C contains every positive-weight node pair.

This module evaluates that recurrence in O((n + m) + K log m) time after
candidate generation, where K is the number of positive candidate pairs.  A
DFS of the first tree keeps precisely its strict-ancestor states active.  Each
active state ending at v' is inserted over the DFS interval of the strict
descendants of v' in the second tree.  Point queries then recover the best
compatible predecessor for a new endpoint v.

The implementation is deliberately separate from ``sparse_align.py``.  That
module computes the original skip-closure DP at selected cells.  This module
works directly on the candidate-pair partial order and is normally preferable
when the positive candidate set is genuinely sparse.
"""

from __future__ import annotations

from math import isfinite
from time import perf_counter
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np

from .diagnostics import MatchDiagnostics
from .needleman_wunsch_tree import AlignmentResult
from .sparse_align import SparseCandidateConfig, generate_sparse_candidates
from .sparse_preprocess import (
    PreprocessedTree,
    _build_tree_structure,
    warn_if_weight_override_differs,
)
from .tree_data import TreeData


ScoredCandidate = Tuple[int, float]

def _is_better(
    score: float,
    length: int,
    state_id: int,
    current_score: float,
    current_length: int,
    current_state_id: int,
) -> bool:
    """Deterministic order: score, then length, then earlier state creation."""
    if score != current_score:
        return score > current_score
    if length != current_length:
        return length > current_length
    if current_state_id < 0:
        return state_id >= 0
    if state_id < 0:
        return False
    return state_id < current_state_id


class _RollbackRangeMaxPointQuery:
    """
    Range insertion / point maximum with rollback.

    A candidate state ending at H-node v contributes to every strict descendant
    of v.  We cover that DFS interval by O(log m) segment-tree nodes.  Each node
    stores the best currently active state among ranges covering it.  DFS in G
    makes insertions properly nested, so changed values can be restored from a
    simple history stack on subtree exit.
    """

    def __init__(self, n_points: int, *, collect_counters: bool = False) -> None:
        if n_points < 1:
            raise ValueError("n_points must be positive")
        size = 1
        while size < n_points:
            size <<= 1
        self.size = size
        n_nodes = 2 * size
        self.score = [0.0] * n_nodes
        self.length = [0] * n_nodes
        self.state_id = [-1] * n_nodes
        self.history: List[Tuple[int, float, int, int]] = []
        self.collect_counters = bool(collect_counters)
        self.point_queries = 0
        self.interval_insertions = 0
        self.node_update_attempts = 0
        self.successful_node_updates = 0
        self.max_history_size = 0

    def checkpoint(self) -> int:
        return len(self.history)

    def rollback(self, checkpoint: int) -> None:
        if checkpoint < 0 or checkpoint > len(self.history):
            raise ValueError("invalid rollback checkpoint")
        while len(self.history) > checkpoint:
            node, old_score, old_length, old_state_id = self.history.pop()
            self.score[node] = old_score
            self.length[node] = old_length
            self.state_id[node] = old_state_id

    def _update_node(self, node: int, score: float, length: int, state_id: int) -> None:
        if self.collect_counters:
            self.node_update_attempts += 1
        if not _is_better(
            score,
            length,
            state_id,
            self.score[node],
            self.length[node],
            self.state_id[node],
        ):
            return
        self.history.append((node, self.score[node], self.length[node], self.state_id[node]))
        if self.collect_counters:
            self.successful_node_updates += 1
            self.max_history_size = max(self.max_history_size, len(self.history))
        self.score[node] = score
        self.length[node] = length
        self.state_id[node] = state_id

    def add_interval(self, left: int, right: int, score: float, length: int, state_id: int) -> None:
        """Insert one state on the inclusive point interval [left, right]."""
        if left > right:
            return
        if self.collect_counters:
            self.interval_insertions += 1
        if left < 0 or right >= self.size:
            raise IndexError("segment-tree interval is outside the point domain")

        left += self.size
        right += self.size
        while left <= right:
            if left & 1:
                self._update_node(left, score, length, state_id)
                left += 1
            if not (right & 1):
                self._update_node(right, score, length, state_id)
                right -= 1
            left >>= 1
            right >>= 1

    def query_point(self, point: int) -> Tuple[float, int, int]:
        """Return (score, length, state_id) of the best range covering point."""
        if point < 0 or point >= self.size:
            raise IndexError("segment-tree query point is outside the point domain")
        if self.collect_counters:
            self.point_queries += 1

        node = point + self.size
        best_score = 0.0
        best_length = 0
        best_state_id = -1
        while node >= 1:
            if _is_better(
                self.score[node],
                self.length[node],
                self.state_id[node],
                best_score,
                best_length,
                best_state_id,
            ):
                best_score = self.score[node]
                best_length = self.length[node]
                best_state_id = self.state_id[node]
            node >>= 1
        return best_score, best_length, best_state_id


def _canonicalize_scored_candidates(
    treeG: TreeData,
    treeH: TreeData,
    scored_candidates: Sequence[Sequence[ScoredCandidate]],
    tinH: np.ndarray,
) -> List[List[ScoredCandidate]]:
    """Validate, discard nonpositive pairs, and deduplicate each (u, v)."""
    if len(scored_candidates) != treeG.n:
        raise ValueError(
            f"Expected scored_candidates to have length {treeG.n}; got {len(scored_candidates)}"
        )

    out: List[List[ScoredCandidate]] = []
    for u, row in enumerate(scored_candidates):
        best_by_v: dict[int, float] = {}
        for item in row:
            if len(item) != 2:
                raise ValueError(f"Candidate for G-node {u} must be a (v, weight) pair")
            v = int(item[0])
            if v < 0 or v >= treeH.n:
                raise ValueError(f"Candidate ({u}, {v}) is outside the H-node range")
            weight = float(item[1])
            if not isfinite(weight):
                raise ValueError(f"Candidate ({u}, {v}) has non-finite weight {weight}")
            # Empty matching is allowed and arbitrary ancestor jumps are allowed,
            # so a nonpositive pair can never improve an optimum.
            if weight <= 0.0:
                continue
            old = best_by_v.get(v)
            if old is None or weight > old:
                best_by_v[v] = weight
        row_out = sorted(best_by_v.items(), key=lambda pair: (int(tinH[pair[0]]), pair[0]))
        out.append([(int(v), float(weight)) for v, weight in row_out])
    return out


def align_scored_sparse_chain(
    treeG: TreeData,
    treeH: TreeData,
    scored_candidates: Sequence[Sequence[ScoredCandidate]],
    *,
    childrenG: Optional[Sequence[Sequence[int]]] = None,
    tinH: Optional[np.ndarray] = None,
    toutH: Optional[np.ndarray] = None,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> AlignmentResult:
    """
    Compute the maximum-weight valid matching from pre-scored candidate pairs.

    Parameters
    ----------
    scored_candidates:
        ``scored_candidates[u]`` contains ``(v, weight(u,v))`` entries.  The
        algorithm is exact relative to these entries.  Duplicate pairs keep the
        largest weight; nonpositive entries are ignored.
    childrenG, tinH, toutH:
        Optional cached structural indices.  ``PreprocessedTree`` supplies these
        in the public wrapper below.
    """
    total_start = perf_counter()
    if diagnostics is not None:
        diagnostics.reset(algorithm="sparse_chain_scored", n_g=treeG.n, n_h=treeH.n)
        diagnostics.candidate_pairs_generated = sum(len(row) for row in scored_candidates)
        diagnostics.positive_candidate_pairs = 0

    preprocessing_start = perf_counter()
    if childrenG is None:
        childrenG_use, _, _ = _build_tree_structure(treeG)
    else:
        if len(childrenG) != treeG.n:
            raise ValueError("childrenG must have one row per G-node")
        childrenG_use = tuple(tuple(int(v) for v in row) for row in childrenG)

    if tinH is None or toutH is None:
        _, tinH_use, toutH_use = _build_tree_structure(treeH)
    else:
        tinH_use = np.asarray(tinH, dtype=np.int64)
        toutH_use = np.asarray(toutH, dtype=np.int64)
        if tinH_use.shape != (treeH.n,) or toutH_use.shape != (treeH.n,):
            raise ValueError("tinH and toutH must have shape (H.n,)")

    candidates = _canonicalize_scored_candidates(treeG, treeH, scored_candidates, tinH_use)
    positive_count = sum(len(row) for row in candidates)

    active = _RollbackRangeMaxPointQuery(treeH.n, collect_counters=diagnostics is not None)

    if diagnostics is not None:
        diagnostics.preprocessing_seconds = perf_counter() - preprocessing_start
        diagnostics.positive_candidate_pairs = positive_count
        diagnostics.extra["candidate_pairs_after_deduplication"] = positive_count

    state_u: List[int] = []
    state_v: List[int] = []
    state_score: List[float] = []
    state_length: List[int] = []
    state_prev: List[int] = []

    best_state_id = -1
    best_score = 0.0
    best_length = 0

    # (u, exiting, checkpoint).  On entry we first score every pair at u,
    # then activate them together.  This prevents pairs sharing the same G-node
    # from being used as predecessors of one another.
    search_start = perf_counter()
    stack: List[Tuple[int, bool, int]] = [(0, False, 0)]
    while stack:
        u, exiting, checkpoint = stack.pop()
        if exiting:
            active.rollback(checkpoint)
            continue

        checkpoint = active.checkpoint()
        new_state_ids: List[int] = []

        for v, weight in candidates[u]:
            pred_score, pred_length, pred_state_id = active.query_point(int(tinH_use[v]))
            score = pred_score + weight
            length = pred_length + 1
            state_id = len(state_score)

            state_u.append(u)
            state_v.append(v)
            state_score.append(score)
            state_length.append(length)
            state_prev.append(pred_state_id)
            new_state_ids.append(state_id)

            if _is_better(
                score,
                length,
                state_id,
                best_score,
                best_length,
                best_state_id,
            ):
                best_score = score
                best_length = length
                best_state_id = state_id

        # Only after all candidates at u have been scored do they become
        # available to strict descendants of u in G.
        for state_id in new_state_ids:
            v = state_v[state_id]
            left = int(tinH_use[v]) + 1
            right = int(toutH_use[v])
            if left <= right:
                active.add_interval(
                    left,
                    right,
                    state_score[state_id],
                    state_length[state_id],
                    state_id,
                )

        stack.append((u, True, checkpoint))
        for child in reversed(childrenG_use[u]):
            stack.append((int(child), False, 0))

    if diagnostics is not None:
        diagnostics.search_seconds = perf_counter() - search_start
        diagnostics.extra["segment_tree_point_queries"] = active.point_queries
        diagnostics.extra["segment_tree_interval_insertions"] = active.interval_insertions
        diagnostics.extra["segment_tree_node_update_attempts"] = active.node_update_attempts
        diagnostics.extra["segment_tree_successful_node_updates"] = active.successful_node_updates
        diagnostics.extra["segment_tree_peak_rollback_history"] = active.max_history_size
        diagnostics.extra["candidate_states_created"] = len(state_score)

    traceback_start = perf_counter()
    if best_state_id < 0:
        if diagnostics is not None:
            diagnostics.traceback_seconds = perf_counter() - traceback_start
            diagnostics.result_score = 0.0
            diagnostics.result_length = 0
            diagnostics.total_seconds = perf_counter() - total_start
        return AlignmentResult(path_internal=[], score=0.0, end_internal=(0, 0), A=None, C=None)

    path_rev: List[Tuple[int, int]] = []
    state_id = best_state_id
    while state_id >= 0:
        path_rev.append((state_u[state_id], state_v[state_id]))
        state_id = state_prev[state_id]
    path_rev.reverse()

    end = path_rev[-1]
    if diagnostics is not None:
        diagnostics.traceback_seconds = perf_counter() - traceback_start
        diagnostics.result_score = float(best_score)
        diagnostics.result_length = len(path_rev)
        diagnostics.total_seconds = perf_counter() - total_start
    return AlignmentResult(
        path_internal=path_rev,
        score=float(best_score),
        end_internal=(int(end[0]), int(end[1])),
        A=None,
        C=None,
    )


def align_trees_sparse_chain(
    G: PreprocessedTree,
    H: PreprocessedTree,
    *,
    candidates: Optional[Sequence[Sequence[int]]] = None,
    cfg: Optional[SparseCandidateConfig] = None,
    w: Optional[Any] = None,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> AlignmentResult:
    """
    Sparse product-poset alignment using bucket-generated candidate pairs.

    The result is always exact relative to ``candidates``.  When candidates are
    generated internally, the default is ``SparseCandidateConfig.exhaustive()``.
    The result then equals the full tree-matching optimum provided that:

    1. every positive-score label pair shares a blocking key;
    2. preprocessing used ``max_nodes_per_key=None``; and
    3. the exhaustive candidate configuration is not replaced by a capped one.

    A capped ``cfg`` remains useful as an approximate candidate generator; the
    chain optimization itself is still exact on the generated candidate set.
    If ``w`` overrides the preprocessing weight while candidates are generated
    internally, the function warns and continues; global candidate completeness
    may no longer hold.
    """
    total_start = perf_counter()
    candidate_start = perf_counter()
    generated_internally = candidates is None
    if candidates is None:
        warn_if_weight_override_differs(G, H, w, stacklevel=3)
        cfg_use = SparseCandidateConfig.exhaustive() if cfg is None else cfg
        candidates = generate_sparse_candidates(G, H, cfg=cfg_use)

    if len(candidates) != G.tree.n:
        raise ValueError(f"Expected candidates to have length {G.tree.n}; got {len(candidates)}")

    w_fn = G.w if w is None else w
    if not callable(w_fn):
        raise TypeError("w must be callable")

    scored_candidates: List[List[ScoredCandidate]] = []
    labelsG = G.tree.label
    labelsH = H.tree.label
    score_evaluations = 0
    for u, row in enumerate(candidates):
        scored_row: List[ScoredCandidate] = []
        for v_raw in row:
            v = int(v_raw)
            if v < 0 or v >= H.tree.n:
                raise ValueError(f"Candidate ({u}, {v}) is outside the H-node range")
            weight = float(w_fn(labelsG[u], labelsH[v]))
            if diagnostics is not None:
                score_evaluations += 1
            scored_row.append((v, weight))
        scored_candidates.append(scored_row)
    candidate_seconds = perf_counter() - candidate_start

    childrenG = getattr(G, "children", None)
    tinH = getattr(H, "tin", None)
    toutH = getattr(H, "tout", None)
    core_diagnostics = MatchDiagnostics() if diagnostics is not None else None
    result = align_scored_sparse_chain(
        G.tree,
        H.tree,
        scored_candidates,
        childrenG=childrenG,
        tinH=tinH,
        toutH=toutH,
        diagnostics=core_diagnostics,
    )
    if diagnostics is not None:
        assert core_diagnostics is not None
        copied = core_diagnostics.copy()
        diagnostics.__dict__.clear()
        diagnostics.__dict__.update(copied.__dict__)
        diagnostics.algorithm = "sparse_chain"
        diagnostics.candidate_generation_seconds += candidate_seconds
        diagnostics.node_pair_score_evaluations = score_evaluations
        diagnostics.candidate_pairs_generated = score_evaluations
        diagnostics.total_seconds = perf_counter() - total_start
        diagnostics.extra["candidates_generated_internally"] = generated_internally
    return result


__all__ = [
    "ScoredCandidate",
    "align_scored_sparse_chain",
    "align_trees_sparse_chain",
]
