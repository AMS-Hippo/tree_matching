"""
Beam search with local skip/match transitions (Algorithm 6 in the project PDF).

The search state is a pair of current tree nodes, together with an accumulated
matching score and a predecessor pointer.  From terminal pair ``(x, y)`` the
algorithm may

- advance from ``x`` to one of its children while keeping ``y`` fixed;
- advance from ``y`` to one of its children while keeping ``x`` fixed; or
- advance in both trees and match the selected child pair.

States are processed by product depth ``depth_G(x) + depth_H(y)``.  Within each
layer, states with the same terminal pair are merged by accumulated score, and
only the top ``beam_width`` states are expanded.  Setting ``child_cap=None``
uses all children, matching the full local expansion rule in the PDF.  A
positive integer cap gives its capped local expansion rule, with children
ordered by internal node index.

This is deliberately a simple baseline implementation.  It does not use the
partial-matching heuristics or lookahead machinery in ``beam_align.py``.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

from .diagnostics import MatchDiagnostics
from .needleman_wunsch_tree import AlignmentResult, WeightFn, id_match
from .tree_data import TreeData


_SENTINEL = -1


@dataclass(frozen=True, slots=True)
class _Proposal:
    """Endpoint-pruned state waiting to be admitted to a product layer."""

    u: int
    v: int
    score: float
    prev_id: int
    match_u: int
    match_v: int
    order: int
    existing_state_id: int = -1


def _build_children(tree: TreeData) -> Tuple[Tuple[int, ...], ...]:
    children: List[List[int]] = [[] for _ in range(tree.n)]
    for node in range(1, tree.n):
        children[int(tree.parent[node])].append(node)
    return tuple(tuple(nodes) for nodes in children)


def _build_depth(tree: TreeData) -> List[int]:
    # Real roots have augmented depth 1 because the PDF places a sentinel at
    # depth 0 above each root.
    depth = [0] * tree.n
    depth[0] = 1
    for node in range(1, tree.n):
        depth[node] = depth[int(tree.parent[node])] + 1
    return depth


def _traceback(
    best_state_id: int,
    prev: List[int],
    match_u: List[int],
    match_v: List[int],
) -> List[Tuple[int, int]]:
    path_rev: List[Tuple[int, int]] = []
    state_id = int(best_state_id)
    while state_id > 0:
        u = int(match_u[state_id])
        if u >= 0:
            path_rev.append((u, int(match_v[state_id])))
        state_id = int(prev[state_id])
    path_rev.reverse()
    return path_rev


def align_trees_local_beam(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    beam_width: int = 200,
    child_cap: Optional[int] = None,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> AlignmentResult:
    """
    Approximate tree-path matching with Algorithm 6's local transitions.

    Parameters
    ----------
    G, H:
        Trees in ``TreeData`` form.
    w:
        Nonnegative match score ``w(label_G, label_H)``.  If ``None``, equal
        labels receive score 1 and all other pairs receive 0.
    beam_width:
        Maximum number of endpoint-pruned states expanded in each product
        layer.  If this is at least the number of terminal pairs in every
        layer, the full local expansion recovers the exact DP score.
    child_cap:
        ``None`` uses every child.  A positive integer ``R`` uses only the
        first ``R`` children of each terminal node, in internal node order.
        This is the capped local expansion rule from the PDF.

    Notes
    -----
    Match transitions with zero score are still generated, as in Algorithm 6,
    but zero-score pairs are omitted from the returned path because deleting
    them preserves both validity and score.
    """
    total_start = perf_counter()
    if diagnostics is not None:
        diagnostics.reset(algorithm="beam_local", n_g=G.n, n_h=H.n)
        diagnostics.candidate_pairs_generated = 0
        diagnostics.positive_candidate_pairs = 0
        diagnostics.states_generated = 0
        diagnostics.states_after_endpoint_pruning = 0
        diagnostics.states_retained = 0
        diagnostics.peak_frontier_size = 0
        diagnostics.layers_processed = 0

    beam_width = int(beam_width)
    if beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if child_cap is not None:
        child_cap = int(child_cap)
        if child_cap < 1:
            raise ValueError("child_cap must be >= 1, or None")

    preprocessing_start = perf_counter()
    if w is None:
        w_fn: WeightFn = id_match
        w_is_id = True
    else:
        if not callable(w):
            raise TypeError("w must be callable")
        w_fn = w
        w_is_id = w is id_match

    children_G = _build_children(G)
    children_H = _build_children(H)
    depth_G = _build_depth(G)
    depth_H = _build_depth(H)
    max_layer = max(depth_G) + max(depth_H)

    if diagnostics is not None:
        diagnostics.preprocessing_seconds = perf_counter() - preprocessing_start
        diagnostics.extra["beam_width"] = beam_width
        diagnostics.extra["child_cap"] = child_cap

    def children_g(node: int) -> Tuple[int, ...]:
        nodes = (0,) if node == _SENTINEL else children_G[node]
        return nodes if child_cap is None else nodes[:child_cap]

    def children_h(node: int) -> Tuple[int, ...]:
        nodes = (0,) if node == _SENTINEL else children_H[node]
        return nodes if child_cap is None else nodes[:child_cap]

    def layer_of(u: int, v: int) -> int:
        du = 0 if u == _SENTINEL else depth_G[u]
        dv = 0 if v == _SENTINEL else depth_H[v]
        return du + dv

    # Only proposals that win endpoint pruning are kept in a layer.  They are
    # materialized as permanent traceback states only if admitted to the beam.
    layers: List[Dict[Tuple[int, int], _Proposal]] = [dict() for _ in range(max_layer + 1)]
    layers[0][(_SENTINEL, _SENTINEL)] = _Proposal(
        u=_SENTINEL,
        v=_SENTINEL,
        score=0.0,
        prev_id=-1,
        match_u=_SENTINEL,
        match_v=_SENTINEL,
        order=0,
        existing_state_id=0,
    )

    # Permanent state pool.  State 0 is the initial sentinel state.
    terminal_u: List[int] = [_SENTINEL]
    terminal_v: List[int] = [_SENTINEL]
    scores: List[float] = [0.0]
    prev: List[int] = [-1]
    matched_u: List[int] = [_SENTINEL]
    matched_v: List[int] = [_SENTINEL]

    best_state_id = 0
    best_score = 0.0
    generation_order = 1
    candidate_generation_seconds = 0.0

    def offer(
        *,
        u: int,
        v: int,
        score: float,
        prev_id: int,
        match: Optional[Tuple[int, int]],
    ) -> None:
        nonlocal generation_order
        if diagnostics is not None:
            diagnostics.states_generated += 1
        layer = layer_of(u, v)
        if layer > max_layer:
            raise RuntimeError("generated state exceeds the maximum product layer")
        key = (u, v)
        old = layers[layer].get(key)
        if old is None or score > old.score:
            if match is None:
                mu = mv = _SENTINEL
            else:
                mu, mv = match
            layers[layer][key] = _Proposal(
                u=u,
                v=v,
                score=float(score),
                prev_id=int(prev_id),
                match_u=int(mu),
                match_v=int(mv),
                order=generation_order,
            )
        generation_order += 1

    search_start = perf_counter()
    for layer in range(max_layer + 1):
        pending = layers[layer]
        if not pending:
            continue
        layers[layer] = {}

        if diagnostics is not None:
            diagnostics.layers_processed += 1
            diagnostics.states_after_endpoint_pruning += len(pending) - (1 if layer == 0 else 0)

        # TopB after endpoint pruning, with deterministic endpoint/order ties.
        admitted = heapq.nsmallest(
            beam_width,
            pending.values(),
            key=lambda p: (-p.score, p.u, p.v, p.order),
        )

        if diagnostics is not None:
            diagnostics.states_retained += len(admitted) - (1 if layer == 0 else 0)
            diagnostics.peak_frontier_size = max(diagnostics.peak_frontier_size, len(admitted))

        frontier: List[int] = []
        for proposal in admitted:
            if proposal.existing_state_id >= 0:
                state_id = proposal.existing_state_id
            else:
                state_id = len(scores)
                terminal_u.append(int(proposal.u))
                terminal_v.append(int(proposal.v))
                scores.append(float(proposal.score))
                prev.append(int(proposal.prev_id))
                # Canonicalize away zero-weight matched pairs.  The transition
                # is still present; only the reported matching omits the pair.
                if proposal.match_u >= 0:
                    delta = float(proposal.score) - float(scores[proposal.prev_id])
                    if delta > 0.0:
                        matched_u.append(int(proposal.match_u))
                        matched_v.append(int(proposal.match_v))
                    else:
                        matched_u.append(_SENTINEL)
                        matched_v.append(_SENTINEL)
                else:
                    matched_u.append(_SENTINEL)
                    matched_v.append(_SENTINEL)

            frontier.append(state_id)
            if scores[state_id] > best_score:
                best_score = float(scores[state_id])
                best_state_id = int(state_id)

        if layer == max_layer:
            continue

        expansion_start = perf_counter() if diagnostics is not None else 0.0
        for state_id in frontier:
            x = int(terminal_u[state_id])
            y = int(terminal_v[state_id])
            current_score = float(scores[state_id])
            next_G = children_g(x)
            next_H = children_h(y)

            # Skip in H: advance in G only.
            for u in next_G:
                offer(u=u, v=y, score=current_score, prev_id=state_id, match=None)

            # Skip in G: advance in H only.
            for v in next_H:
                offer(u=x, v=v, score=current_score, prev_id=state_id, match=None)

            # Match: advance in both trees.
            for u in next_G:
                label_u = G.label[u]
                for v in next_H:
                    if diagnostics is not None:
                        diagnostics.candidate_pairs_generated += 1
                        diagnostics.node_pair_score_evaluations += 1
                    if w_is_id:
                        delta = 1.0 if label_u == H.label[v] else 0.0
                    else:
                        delta = float(w_fn(label_u, H.label[v]))
                    if diagnostics is not None and delta > 0.0:
                        diagnostics.positive_candidate_pairs += 1
                    offer(
                        u=u,
                        v=v,
                        score=current_score + delta,
                        prev_id=state_id,
                        match=(u, v),
                    )
        if diagnostics is not None:
            candidate_generation_seconds += perf_counter() - expansion_start

    search_elapsed = perf_counter() - search_start
    if diagnostics is not None:
        diagnostics.candidate_generation_seconds = candidate_generation_seconds
        diagnostics.search_seconds = max(0.0, search_elapsed - candidate_generation_seconds)
        diagnostics.extra["permanent_state_pool_size"] = len(scores)
        diagnostics.extra["maximum_product_layer"] = max_layer

    traceback_start = perf_counter()
    if best_state_id == 0:
        if diagnostics is not None:
            diagnostics.traceback_seconds = perf_counter() - traceback_start
            diagnostics.result_score = 0.0
            diagnostics.result_length = 0
            diagnostics.total_seconds = perf_counter() - total_start
        return AlignmentResult(path_internal=[], score=0.0, end_internal=(0, 0), A=None, C=None)

    path = _traceback(best_state_id, prev, matched_u, matched_v)
    if diagnostics is not None:
        diagnostics.traceback_seconds = perf_counter() - traceback_start
        diagnostics.result_score = float(best_score)
        diagnostics.result_length = len(path)
        diagnostics.total_seconds = perf_counter() - total_start
    return AlignmentResult(
        path_internal=path,
        score=float(best_score),
        end_internal=(int(terminal_u[best_state_id]), int(terminal_v[best_state_id])),
        A=None,
        C=None,
    )
