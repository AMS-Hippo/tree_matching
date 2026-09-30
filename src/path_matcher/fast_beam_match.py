"""Specialized score-only partial-matching beam search.

This module accelerates the deterministic Algorithm-7-style beam used by the
``beam_partial_score`` benchmark preset for the two score families supported by
:mod:`path_matcher.fast_match`:

``equality``
    A node pair has positive score when the two integerized labels agree.

``overlap``
    A node pair has score equal to the largest positive weight of a token shared
    by the two node labels.

The implementation is deliberately narrower than :mod:`beam_align`.  It does
not support Python callbacks, random exploration, rarity bonuses, future-score
bonuses, or lookahead.  In exchange, each tree can be encoded and structurally
indexed once, candidate generation runs in a Numba-compatible kernel, and beam
layers are reduced with vectorized endpoint pruning.

Candidate semantics
-------------------
For each live state, positive tokens are scanned from largest to smallest
weight.  For a token, the first ``max_nodes_per_token_side`` strict descendants
in each tree are paired.  The first ``expansion_width`` distinct node pairs are
retained.  In equality mode this is the encoded counterpart of the generic
score-only beam.  In overlap mode the token postings provide a more direct
candidate generator than grouping nodes by their complete token sets; a pair
seen through several tokens receives the largest shared-token weight because
higher-weight tokens are scanned first.

The returned path is always a valid ancestor-consistent matching.  With a beam
and expansion budget large enough to retain every positive feasible pair, the
method recovers the exact optimum for the supported nonnegative scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .diagnostics import MatchDiagnostics
from .fast_match import (
    EncodedTree,
    EncodedTreeEquality,
    EncodedTreeOverlap,
    FastLabelEncoder,
    LabelGetter,
)
from .igraph_io import igraph_to_treedata
from .needleman_wunsch_tree import AlignmentResult
from .sparse_preprocess import _build_tree_structure
from .tree_data import TreeData

try:
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover - exercised when numba is unavailable
    HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def deco(fn):
            return fn

        return deco


@dataclass(frozen=True)
class FastBeamPreparedTree:
    """Encoded tree plus reusable structural and token-posting indices."""

    encoded: EncodedTree
    children_offsets: np.ndarray
    children_nodes: np.ndarray
    depth: np.ndarray
    height: np.ndarray
    tin: np.ndarray
    tout: np.ndarray
    token_offsets: np.ndarray
    token_nodes: np.ndarray
    token_tins: np.ndarray
    node_token_offsets: np.ndarray
    node_token_ids: np.ndarray
    encoder_fingerprint: Optional[Tuple[str, int]] = None

    @property
    def tree(self) -> TreeData:
        return self.encoded.tree

    @property
    def mode(self) -> str:
        if isinstance(self.encoded, EncodedTreeEquality):
            return "equality"
        if isinstance(self.encoded, EncodedTreeOverlap):
            return "overlap"
        raise TypeError(f"Unsupported encoded tree type {type(self.encoded)!r}")


def _looks_like_treedata(x: Any) -> bool:
    return hasattr(x, "parent") and hasattr(x, "label") and hasattr(x, "orig_index")


def _as_treedata(
    G: Any,
    *,
    phi_name: str,
    order: str,
    ts_field: Optional[str],
    strict_tree: bool,
) -> TreeData:
    if isinstance(G, TreeData) or _looks_like_treedata(G):
        return G  # type: ignore[return-value]
    return igraph_to_treedata(
        G,
        phi_name=phi_name,
        order=order,
        ts_field=ts_field,
        strict_tree=strict_tree,
    )


def _flatten_children(children: Tuple[Tuple[int, ...], ...]) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(children) + 1, dtype=np.int64)
    flat: List[int] = []
    for u, row in enumerate(children):
        flat.extend(int(v) for v in row)
        offsets[u + 1] = len(flat)
    return offsets, np.asarray(flat, dtype=np.int32)


def _depth_height(tree: TreeData, children: Tuple[Tuple[int, ...], ...]) -> Tuple[np.ndarray, np.ndarray]:
    n = tree.n
    parent = np.asarray(tree.parent, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int32)
    for u in range(1, n):
        depth[u] = np.int32(int(depth[int(parent[u])]) + 1)

    height = np.ones(n, dtype=np.int32)
    for u in range(n - 1, -1, -1):
        if children[u]:
            height[u] = np.int32(1 + max(int(height[v]) for v in children[u]))
    return depth, height


def prepare_fast_beam_encoded(
    encoded: EncodedTree,
    *,
    token_count: Optional[int] = None,
    encoder_fingerprint: Optional[Tuple[str, int]] = None,
) -> FastBeamPreparedTree:
    """Build reusable DFS, child, depth, height, and token-posting indices."""

    children, tin, tout = _build_tree_structure(encoded.tree)
    child_offsets, child_nodes = _flatten_children(children)
    depth, height = _depth_height(encoded.tree, children)

    max_token = -1
    if isinstance(encoded, EncodedTreeEquality):
        if encoded.label_ids.size:
            max_token = int(np.max(encoded.label_ids))
    elif isinstance(encoded, EncodedTreeOverlap):
        if encoded.flat_token_ids.size:
            max_token = int(np.max(encoded.flat_token_ids))
    else:
        raise TypeError(f"Unsupported encoded tree type {type(encoded)!r}")

    if token_count is None:
        token_count_use = max_token + 1
    else:
        token_count_use = int(token_count)
        if token_count_use < max_token + 1:
            raise ValueError(
                f"token_count={token_count_use} is too small for maximum token id {max_token}"
            )
    if token_count_use < 0:
        raise ValueError("token_count must be nonnegative")

    postings: List[List[int]] = [[] for _ in range(token_count_use)]
    if isinstance(encoded, EncodedTreeEquality):
        for u, raw in enumerate(encoded.label_ids):
            token = int(raw)
            if token >= 0:
                postings[token].append(int(u))
    else:
        for u in range(encoded.tree.n):
            start = int(encoded.offsets[u])
            stop = int(encoded.offsets[u + 1])
            for raw in encoded.flat_token_ids[start:stop]:
                postings[int(raw)].append(int(u))

    token_offsets = np.zeros(token_count_use + 1, dtype=np.int64)
    flat_nodes: List[int] = []
    flat_tins: List[int] = []
    for token, nodes in enumerate(postings):
        nodes.sort(key=lambda u: int(tin[u]))
        flat_nodes.extend(nodes)
        flat_tins.extend(int(tin[u]) for u in nodes)
        token_offsets[token + 1] = len(flat_nodes)

    if isinstance(encoded, EncodedTreeEquality):
        node_token_offsets = np.zeros(encoded.tree.n + 1, dtype=np.int64)
        node_tokens: List[int] = []
        for u, raw in enumerate(encoded.label_ids):
            token = int(raw)
            if token >= 0:
                node_tokens.append(token)
            node_token_offsets[u + 1] = len(node_tokens)
        node_token_ids = np.asarray(node_tokens, dtype=np.int32)
    else:
        node_token_offsets = np.asarray(encoded.offsets, dtype=np.int64)
        node_token_ids = np.asarray(encoded.flat_token_ids, dtype=np.int32)

    return FastBeamPreparedTree(
        encoded=encoded,
        children_offsets=child_offsets,
        children_nodes=child_nodes,
        depth=depth,
        height=height,
        tin=np.asarray(tin, dtype=np.int32),
        tout=np.asarray(tout, dtype=np.int32),
        token_offsets=token_offsets,
        token_nodes=np.asarray(flat_nodes, dtype=np.int32),
        token_tins=np.asarray(flat_tins, dtype=np.int32),
        node_token_offsets=node_token_offsets,
        node_token_ids=node_token_ids,
        encoder_fingerprint=encoder_fingerprint,
    )


def _validate_weight_array(weight_by_id: np.ndarray, token_count: int) -> np.ndarray:
    weights = np.asarray(weight_by_id, dtype=np.float64)
    if weights.ndim != 1:
        raise ValueError("weight_by_id must be one-dimensional")
    if len(weights) < int(token_count):
        raise ValueError(
            f"weight_by_id has length {len(weights)}, but prepared trees use {token_count} tokens"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("weight_by_id must contain only finite values")
    return weights


def _default_token_order(
    G: FastBeamPreparedTree,
    H: FastBeamPreparedTree,
    weight_by_id: np.ndarray,
    *,
    min_match_score: float,
    token_tie_keys: Optional[Sequence[str]] = None,
) -> np.ndarray:
    n_tokens = min(
        len(weight_by_id),
        len(G.token_offsets) - 1,
        len(H.token_offsets) - 1,
    )
    positive: List[int] = []
    for token in range(n_tokens):
        if float(weight_by_id[token]) <= float(min_match_score):
            continue
        if int(G.token_offsets[token + 1]) == int(G.token_offsets[token]):
            continue
        if int(H.token_offsets[token + 1]) == int(H.token_offsets[token]):
            continue
        positive.append(token)

    denom = np.log(max(2.0, float(G.tree.n * H.tree.n)))

    def rarity(token: int) -> float:
        freq_g = int(G.token_offsets[token + 1] - G.token_offsets[token])
        freq_h = int(H.token_offsets[token + 1] - H.token_offsets[token])
        return float(
            max(
                0.0,
                np.log(max(1.0, float(G.tree.n * H.tree.n) / float(freq_g * freq_h))) / denom,
            )
        )

    if token_tie_keys is None:
        positive.sort(
            key=lambda t: (float(weight_by_id[t]), rarity(t), int(t)),
            reverse=True,
        )
    else:
        positive.sort(
            key=lambda t: (
                float(weight_by_id[t]),
                rarity(t),
                str(token_tie_keys[t]) if t < len(token_tie_keys) else str(t),
            ),
            reverse=True,
        )
    return np.asarray(positive, dtype=np.int32)


@njit(cache=True)
def _lower_bound_range(arr: np.ndarray, start: int, stop: int, value: int) -> int:
    lo = int(start)
    hi = int(stop)
    while lo < hi:
        mid = (lo + hi) // 2
        if int(arr[mid]) < int(value):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _encoded_pair_score(
    mode_overlap: bool,
    u: int,
    v: int,
    node_offsets_g: np.ndarray,
    node_tokens_g: np.ndarray,
    node_offsets_h: np.ndarray,
    node_tokens_h: np.ndarray,
    weight_by_id: np.ndarray,
    fallback_weight: float,
) -> float:
    if not mode_overlap:
        return float(fallback_weight)
    i = int(node_offsets_g[u])
    i_stop = int(node_offsets_g[u + 1])
    j = int(node_offsets_h[v])
    j_stop = int(node_offsets_h[v + 1])
    best = 0.0
    while i < i_stop and j < j_stop:
        a = int(node_tokens_g[i])
        b = int(node_tokens_h[j])
        if a == b:
            value = float(weight_by_id[a])
            if value > best:
                best = value
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return best


@njit(cache=True)
def _generate_partial_candidates_numba(
    frontier_u: np.ndarray,
    frontier_v: np.ndarray,
    token_order: np.ndarray,
    token_offsets_g: np.ndarray,
    token_nodes_g: np.ndarray,
    token_tins_g: np.ndarray,
    tin_g: np.ndarray,
    tout_g: np.ndarray,
    token_offsets_h: np.ndarray,
    token_nodes_h: np.ndarray,
    token_tins_h: np.ndarray,
    tin_h: np.ndarray,
    tout_h: np.ndarray,
    node_offsets_g: np.ndarray,
    node_tokens_g: np.ndarray,
    node_offsets_h: np.ndarray,
    node_tokens_h: np.ndarray,
    mode_overlap: bool,
    weight_by_id: np.ndarray,
    max_nodes_per_token_side: int,
    expansion_width: int,
    max_token_types_per_expansion: int,
):
    """Generate score-ordered distinct descendant pairs for one complete frontier."""

    frontier_size = int(frontier_u.shape[0])
    max_out = frontier_size * int(expansion_width)
    out_u = np.empty(max_out, dtype=np.int32)
    out_v = np.empty(max_out, dtype=np.int32)
    out_delta = np.empty(max_out, dtype=np.float64)
    out_parent = np.empty(max_out, dtype=np.int32)
    out_serial = np.empty(max_out, dtype=np.int64)

    out_count = 0
    token_ranges_scanned = 0
    posting_pair_attempts = 0
    duplicate_pairs_skipped = 0

    for parent_index in range(frontier_size):
        terminal_g = int(frontier_u[parent_index])
        terminal_h = int(frontier_v[parent_index])

        if terminal_g < 0:
            lo_tin_g = 0
            hi_tin_g = int(tin_g.shape[0])
        else:
            lo_tin_g = int(tin_g[terminal_g]) + 1
            hi_tin_g = int(tout_g[terminal_g]) + 1

        if terminal_h < 0:
            lo_tin_h = 0
            hi_tin_h = int(tin_h.shape[0])
        else:
            lo_tin_h = int(tin_h[terminal_h]) + 1
            hi_tin_h = int(tout_h[terminal_h]) + 1

        state_start = out_count
        state_count = 0
        state_token_types_scanned = 0

        for token_index in range(token_order.shape[0]):
            if (
                int(max_token_types_per_expansion) > 0
                and state_token_types_scanned >= int(max_token_types_per_expansion)
                and state_count > 0
            ):
                break
            token = int(token_order[token_index])
            token_ranges_scanned += 1
            state_token_types_scanned += 1

            g0 = int(token_offsets_g[token])
            g1 = int(token_offsets_g[token + 1])
            h0 = int(token_offsets_h[token])
            h1 = int(token_offsets_h[token + 1])

            ga = _lower_bound_range(token_tins_g, g0, g1, lo_tin_g)
            gb = _lower_bound_range(token_tins_g, ga, g1, hi_tin_g)
            if ga >= gb:
                continue
            ha = _lower_bound_range(token_tins_h, h0, h1, lo_tin_h)
            hb = _lower_bound_range(token_tins_h, ha, h1, hi_tin_h)
            if ha >= hb:
                continue

            take_g = min(int(max_nodes_per_token_side), gb - ga)
            take_h = min(int(max_nodes_per_token_side), hb - ha)
            weight = float(weight_by_id[token])

            stop_state = False
            for i in range(take_g):
                u = int(token_nodes_g[ga + i])
                for j in range(take_h):
                    v = int(token_nodes_h[ha + j])
                    posting_pair_attempts += 1

                    duplicate = False
                    for old in range(state_start, out_count):
                        if int(out_u[old]) == u and int(out_v[old]) == v:
                            duplicate = True
                            break
                    if duplicate:
                        duplicate_pairs_skipped += 1
                        continue

                    out_u[out_count] = u
                    out_v[out_count] = v
                    out_delta[out_count] = _encoded_pair_score(
                        mode_overlap,
                        u,
                        v,
                        node_offsets_g,
                        node_tokens_g,
                        node_offsets_h,
                        node_tokens_h,
                        weight_by_id,
                        weight,
                    )
                    out_parent[out_count] = parent_index
                    out_serial[out_count] = out_count
                    out_count += 1
                    state_count += 1
                    if state_count >= int(expansion_width):
                        stop_state = True
                        break
                if stop_state:
                    break
            if stop_state:
                break

        # The generic score-only beam sorts the retained candidates by
        # decreasing score and then increasing node ids before creating states.
        # Reproduce that order with a tiny insertion sort (the segment has at
        # most ``expansion_width`` entries).
        for pos in range(state_start + 1, out_count):
            key_u = int(out_u[pos])
            key_v = int(out_v[pos])
            key_delta = float(out_delta[pos])
            key_parent = int(out_parent[pos])
            j = pos - 1
            while j >= state_start:
                before = False
                old_delta = float(out_delta[j])
                old_u = int(out_u[j])
                old_v = int(out_v[j])
                if key_delta > old_delta:
                    before = True
                elif key_delta == old_delta:
                    if key_u < old_u or (key_u == old_u and key_v < old_v):
                        before = True
                if not before:
                    break
                out_u[j + 1] = out_u[j]
                out_v[j + 1] = out_v[j]
                out_delta[j + 1] = out_delta[j]
                out_parent[j + 1] = out_parent[j]
                j -= 1
            out_u[j + 1] = key_u
            out_v[j + 1] = key_v
            out_delta[j + 1] = key_delta
            out_parent[j + 1] = key_parent
        for pos in range(state_start, out_count):
            out_serial[pos] = pos

    return (
        out_u[:out_count],
        out_v[:out_count],
        out_delta[:out_count],
        out_parent[:out_count],
        out_serial[:out_count],
        token_ranges_scanned,
        posting_pair_attempts,
        duplicate_pairs_skipped,
    )


def _endpoint_prune(
    cand_u: np.ndarray,
    cand_v: np.ndarray,
    cand_scores: np.ndarray,
    cand_parent: np.ndarray,
    cand_serial: np.ndarray,
    *,
    n_h: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cand_u.size == 0:
        return cand_u, cand_v, cand_scores, cand_parent, cand_serial

    keys = cand_u.astype(np.int64) * int(n_h) + cand_v.astype(np.int64)
    # Primary key: terminal pair.  Within a terminal pair, keep the largest
    # accumulated score; if tied, keep the earliest generated candidate.
    order = np.lexsort((cand_serial, -cand_scores, keys))
    sorted_keys = keys[order]
    first = np.empty(len(order), dtype=bool)
    first[0] = True
    if len(order) > 1:
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep = order[first]
    return cand_u[keep], cand_v[keep], cand_scores[keep], cand_parent[keep], cand_serial[keep]


def _traceback(
    best_state: int,
    state_u: Sequence[int],
    state_v: Sequence[int],
    state_prev: Sequence[int],
) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    sid = int(best_state)
    while sid > 0:
        out.append((int(state_u[sid]), int(state_v[sid])))
        sid = int(state_prev[sid])
    out.reverse()
    return out


def align_trees_fast_partial_beam_prepared(
    G: FastBeamPreparedTree,
    H: FastBeamPreparedTree,
    *,
    weight_by_id: np.ndarray,
    beam_width: int = 200,
    expansion_width: int = 64,
    max_nodes_per_token_side: int = 8,
    max_token_types_per_expansion: int = 2048,
    min_match_score: float = 0.0,
    max_length: Optional[int] = None,
    token_order: Optional[np.ndarray] = None,
    token_tie_keys: Optional[Sequence[str]] = None,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> AlignmentResult:
    """Run the prepared deterministic score-only partial-matching beam."""

    total_start = perf_counter()
    if G.mode != H.mode:
        raise TypeError(f"Prepared modes disagree: G is {G.mode!r}, H is {H.mode!r}")
    if (
        G.encoder_fingerprint is not None
        and H.encoder_fingerprint is not None
        and G.encoder_fingerprint != H.encoder_fingerprint
    ):
        raise ValueError(
            "Prepared trees were produced by incompatible FastLabelEncoder mappings. "
            "Fit one encoder on the full corpus and prepare both trees with that encoder."
        )
    if int(beam_width) < 1:
        raise ValueError("beam_width must be >= 1")
    if int(expansion_width) < 1:
        raise ValueError("expansion_width must be >= 1")
    if int(max_nodes_per_token_side) < 1:
        raise ValueError("max_nodes_per_token_side must be >= 1")
    if int(max_token_types_per_expansion) < 1:
        raise ValueError("max_token_types_per_expansion must be >= 1")

    token_count = min(len(G.token_offsets), len(H.token_offsets)) - 1
    weights = _validate_weight_array(weight_by_id, token_count)
    if diagnostics is not None:
        diagnostics.reset(
            algorithm=f"fast_beam_partial_{G.mode}",
            n_g=G.tree.n,
            n_h=H.tree.n,
        )
        diagnostics.candidate_pairs_generated = 0
        diagnostics.positive_candidate_pairs = 0
        diagnostics.states_generated = 0
        diagnostics.states_after_endpoint_pruning = 0
        diagnostics.states_retained = 0
        diagnostics.peak_frontier_size = 1
        diagnostics.layers_processed = 0

    prep_start = perf_counter()
    if token_order is None:
        token_order_use = _default_token_order(
            G,
            H,
            weights,
            min_match_score=float(min_match_score),
            token_tie_keys=token_tie_keys,
        )
    else:
        token_order_use = np.asarray(token_order, dtype=np.int32)
        if token_order_use.ndim != 1:
            raise ValueError("token_order must be one-dimensional")
    if max_length is None:
        lmax = int(min(int(G.height[0]), int(H.height[0])))
    else:
        lmax = int(max_length)
        if lmax < 0:
            raise ValueError("max_length must be nonnegative")
    if diagnostics is not None:
        diagnostics.preprocessing_seconds = perf_counter() - prep_start
        diagnostics.extra.update(
            {
                "beam_width": int(beam_width),
                "expansion_width": int(expansion_width),
                "max_nodes_per_token_side": int(max_nodes_per_token_side),
                "max_token_types_per_expansion": int(max_token_types_per_expansion),
                "positive_token_types": int(len(token_order_use)),
                "maximum_matching_length": int(lmax),
                "candidate_semantics": "descending_token_weight_first_descendants",
                "numba_available": bool(HAVE_NUMBA),
            }
        )

    # State 0 is the sentinel.  Only beam-retained states are made permanent;
    # with score-only ranking the best generated state is necessarily retained.
    state_u: List[int] = [-1]
    state_v: List[int] = [-1]
    state_score: List[float] = [0.0]
    state_prev: List[int] = [-1]

    frontier_u = np.asarray([-1], dtype=np.int32)
    frontier_v = np.asarray([-1], dtype=np.int32)
    frontier_score = np.asarray([0.0], dtype=np.float64)
    frontier_global = np.asarray([0], dtype=np.int64)

    best_state = 0
    best_score = 0.0
    candidate_seconds = 0.0
    token_ranges_scanned = 0
    posting_pair_attempts = 0
    duplicate_pairs_skipped = 0

    search_start = perf_counter()
    for _layer in range(lmax):
        if diagnostics is not None:
            diagnostics.layers_processed = int(diagnostics.layers_processed or 0) + 1

        cg_start = perf_counter()
        (
            cand_u,
            cand_v,
            cand_delta,
            cand_parent,
            cand_serial,
            ranges_here,
            attempts_here,
            duplicates_here,
        ) = _generate_partial_candidates_numba(
            frontier_u,
            frontier_v,
            token_order_use,
            G.token_offsets,
            G.token_nodes,
            G.token_tins,
            G.tin,
            G.tout,
            H.token_offsets,
            H.token_nodes,
            H.token_tins,
            H.tin,
            H.tout,
            G.node_token_offsets,
            G.node_token_ids,
            H.node_token_offsets,
            H.node_token_ids,
            bool(G.mode == "overlap"),
            weights,
            int(max_nodes_per_token_side),
            int(expansion_width),
            int(max_token_types_per_expansion),
        )
        candidate_seconds += perf_counter() - cg_start
        token_ranges_scanned += int(ranges_here)
        posting_pair_attempts += int(attempts_here)
        duplicate_pairs_skipped += int(duplicates_here)

        generated = int(cand_u.size)
        if diagnostics is not None:
            diagnostics.candidate_pairs_generated = int(diagnostics.candidate_pairs_generated or 0) + generated
            diagnostics.positive_candidate_pairs = int(diagnostics.positive_candidate_pairs or 0) + generated
            diagnostics.states_generated = int(diagnostics.states_generated or 0) + generated
            diagnostics.node_pair_score_evaluations += generated
        if generated == 0:
            break

        cand_scores = frontier_score[cand_parent] + cand_delta
        pu, pv, ps, pp, pserial = _endpoint_prune(
            cand_u,
            cand_v,
            cand_scores,
            cand_parent,
            cand_serial,
            n_h=H.tree.n,
        )
        if diagnostics is not None:
            diagnostics.states_after_endpoint_pruning = int(
                diagnostics.states_after_endpoint_pruning or 0
            ) + int(len(pu))
        if len(pu) == 0:
            break

        # Score-only beam priority.  Earlier generated candidates win exact ties.
        rank = np.lexsort((pserial, -ps))
        keep = rank[: min(int(beam_width), len(rank))]
        next_u = pu[keep].astype(np.int32, copy=False)
        next_v = pv[keep].astype(np.int32, copy=False)
        next_score = ps[keep].astype(np.float64, copy=False)
        next_parent = pp[keep]

        base = len(state_score)
        next_global = np.arange(base, base + len(keep), dtype=np.int64)
        for i in range(len(keep)):
            state_u.append(int(next_u[i]))
            state_v.append(int(next_v[i]))
            state_score.append(float(next_score[i]))
            state_prev.append(int(frontier_global[int(next_parent[i])]))

        frontier_u = next_u
        frontier_v = next_v
        frontier_score = next_score
        frontier_global = next_global

        if diagnostics is not None:
            diagnostics.states_retained = int(diagnostics.states_retained or 0) + int(len(keep))
            diagnostics.peak_frontier_size = max(
                int(diagnostics.peak_frontier_size or 0), int(len(keep))
            )

        if len(keep) > 0 and float(next_score[0]) > best_score:
            best_score = float(next_score[0])
            best_state = int(next_global[0])

    search_elapsed = perf_counter() - search_start
    traceback_start = perf_counter()
    path = _traceback(best_state, state_u, state_v, state_prev)
    traceback_elapsed = perf_counter() - traceback_start

    if diagnostics is not None:
        diagnostics.candidate_generation_seconds = float(candidate_seconds)
        diagnostics.search_seconds = max(0.0, float(search_elapsed - candidate_seconds))
        diagnostics.traceback_seconds = float(traceback_elapsed)
        diagnostics.total_seconds = perf_counter() - total_start
        diagnostics.result_score = float(best_score)
        diagnostics.result_length = len(path)
        diagnostics.extra.update(
            {
                "token_ranges_scanned": int(token_ranges_scanned),
                "posting_pair_attempts": int(posting_pair_attempts),
                "duplicate_pairs_skipped": int(duplicate_pairs_skipped),
                "permanent_state_pool_size": int(len(state_score)),
            }
        )

    end = path[-1] if path else (-1, -1)
    return AlignmentResult(path_internal=path, score=float(best_score), end_internal=end)


class FastBeamTreePathMatcher:
    """Prepared encoded score-only partial-matching beam.

    This first accelerated beam implementation supports ``search='partial'``
    only.  The parameter is retained so that a compatible local-beam kernel can
    be added later without changing the high-level API.

    Typical repeated-comparison workflow
    ------------------------------------
    >>> matcher = FastBeamTreePathMatcher(mode="equality", beam_width=200)
    >>> matcher.fit_encoder(all_trees)
    >>> prepared = [matcher.prepare_tree(T) for T in all_trees]
    >>> path, score = matcher.predict_prepared(prepared[i], prepared[j])
    """

    def __init__(
        self,
        *,
        mode: str = "equality",
        search: str = "partial",
        beam_width: int = 200,
        expansion_width: int = 64,
        max_nodes_per_token_side: int = 8,
        max_token_types_per_expansion: int = 2048,
        min_match_score: float = 0.0,
        max_length: Optional[int] = None,
        label_getter: LabelGetter = None,
        token_weights: Optional[Mapping[Any, float]] = None,
        default_weight: float = 1.0,
        phi_name: str = "label",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        encoder: Optional[FastLabelEncoder] = None,
        collect_diagnostics: bool = False,
    ) -> None:
        mode = mode.lower().strip()
        if mode not in {"equality", "overlap"}:
            raise ValueError("mode must be 'equality' or 'overlap'")
        search = search.lower().strip()
        if search != "partial":
            raise ValueError("The current fast beam supports search='partial' only")
        if int(beam_width) < 1:
            raise ValueError("beam_width must be >= 1")
        if int(expansion_width) < 1:
            raise ValueError("expansion_width must be >= 1")
        if int(max_nodes_per_token_side) < 1:
            raise ValueError("max_nodes_per_token_side must be >= 1")
        if int(max_token_types_per_expansion) < 1:
            raise ValueError("max_token_types_per_expansion must be >= 1")
        if encoder is not None and encoder.mode != mode:
            raise ValueError(f"encoder.mode={encoder.mode!r} does not match matcher mode={mode!r}")

        self.mode = mode
        self.search = search
        self.beam_width = int(beam_width)
        self.expansion_width = int(expansion_width)
        self.max_nodes_per_token_side = int(max_nodes_per_token_side)
        self.max_token_types_per_expansion = int(max_token_types_per_expansion)
        self.min_match_score = float(min_match_score)
        self.max_length = max_length
        self.label_getter = label_getter
        self.token_weights = dict(token_weights or {})
        self.default_weight = float(default_weight)
        self.phi_name = phi_name
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = bool(strict_tree)
        self.collect_diagnostics = bool(collect_diagnostics)
        self.encoder = encoder or FastLabelEncoder(
            mode=mode,
            label_getter=label_getter,
            token_weights=self.token_weights,
            default_weight=self.default_weight,
        )

        self.preparedG_: Optional[FastBeamPreparedTree] = None
        self.preparedH_: Optional[FastBeamPreparedTree] = None
        self.last_diagnostics_: Optional[MatchDiagnostics] = None
        self.last_fit_diagnostics_: Optional[MatchDiagnostics] = None

    def _to_tree(self, G: Any) -> TreeData:
        return _as_treedata(
            G,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
        )

    def fit_encoder(self, trees: Sequence[Any]) -> "FastBeamTreePathMatcher":
        td_trees = [self._to_tree(T) for T in trees]
        self.encoder.fit_from_trees(td_trees)
        return self

    def _token_tie_keys(self) -> List[str]:
        # ``beam_align._safe_label_key`` represents hashable scalar labels as
        # ("hash", label), and its deterministic label-pair order uses repr.
        # Mirroring that convention makes equality-mode candidate ties agree
        # with the generic score-only beam whenever the candidate budgets agree.
        return [repr(("hash", token)) for token in self.encoder.id_to_token]

    def prepare_tree(self, G: Any) -> FastBeamPreparedTree:
        tree = self._to_tree(G)
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([tree])
        assert self.encoder.weight_by_id is not None
        encoded = self.encoder.transform_tree(tree)
        return prepare_fast_beam_encoded(
            encoded,
            token_count=len(self.encoder.weight_by_id),
            encoder_fingerprint=self.encoder.fingerprint,
        )

    def fit(self, G: Any, H: Any) -> "FastBeamTreePathMatcher":
        start = perf_counter()
        self.last_fit_diagnostics_ = None
        treeG = self._to_tree(G)
        treeH = self._to_tree(H)
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([treeG, treeH])
        assert self.encoder.weight_by_id is not None
        fingerprint = self.encoder.fingerprint
        self.preparedG_ = prepare_fast_beam_encoded(
            self.encoder.transform_tree(treeG),
            token_count=len(self.encoder.weight_by_id),
            encoder_fingerprint=fingerprint,
        )
        self.preparedH_ = prepare_fast_beam_encoded(
            self.encoder.transform_tree(treeH),
            token_count=len(self.encoder.weight_by_id),
            encoder_fingerprint=fingerprint,
        )
        if self.collect_diagnostics:
            diag = MatchDiagnostics(
                algorithm="fast_beam_partial_fit",
                n_g=treeG.n,
                n_h=treeH.n,
            )
            diag.input_preprocessing_seconds = perf_counter() - start
            diag.total_seconds = diag.input_preprocessing_seconds
            self.last_fit_diagnostics_ = diag
        return self

    def predict(
        self,
        G: Optional[Any] = None,
        H: Optional[Any] = None,
    ) -> Tuple[List[Tuple[int, int]], float]:
        total_start = perf_counter()
        self.last_diagnostics_ = None
        supplied = G is not None or H is not None
        input_start = total_start
        if supplied:
            if G is None or H is None:
                raise ValueError("Either provide both G and H, or provide neither")
            treeG = self._to_tree(G)
            treeH = self._to_tree(H)
            if not self.encoder.is_fitted:
                self.encoder.fit_from_trees([treeG, treeH])
            assert self.encoder.weight_by_id is not None
            fingerprint = self.encoder.fingerprint
            preparedG = prepare_fast_beam_encoded(
                self.encoder.transform_tree(treeG),
                token_count=len(self.encoder.weight_by_id),
                encoder_fingerprint=fingerprint,
            )
            preparedH = prepare_fast_beam_encoded(
                self.encoder.transform_tree(treeH),
                token_count=len(self.encoder.weight_by_id),
                encoder_fingerprint=fingerprint,
            )
        else:
            if self.preparedG_ is None or self.preparedH_ is None:
                raise RuntimeError("Must call fit(G,H) before predict() if no inputs are provided")
            preparedG = self.preparedG_
            preparedH = self.preparedH_
        input_seconds = perf_counter() - input_start if supplied else 0.0
        return self._predict_prepared(
            preparedG,
            preparedH,
            input_preprocessing_seconds=input_seconds,
            total_start=total_start,
        )

    def predict_prepared(
        self,
        G: FastBeamPreparedTree,
        H: FastBeamPreparedTree,
    ) -> Tuple[List[Tuple[int, int]], float]:
        self.last_diagnostics_ = None
        return self._predict_prepared(
            G,
            H,
            input_preprocessing_seconds=0.0,
            total_start=perf_counter(),
        )

    def _predict_prepared(
        self,
        G: FastBeamPreparedTree,
        H: FastBeamPreparedTree,
        *,
        input_preprocessing_seconds: float,
        total_start: float,
    ) -> Tuple[List[Tuple[int, int]], float]:
        if not self.encoder.is_fitted or self.encoder.weight_by_id is None:
            raise RuntimeError("FastLabelEncoder is not fitted")
        fingerprint = self.encoder.fingerprint
        for name, prepared in (("G", G), ("H", H)):
            if (
                prepared.encoder_fingerprint is not None
                and prepared.encoder_fingerprint != fingerprint
            ):
                raise ValueError(
                    f"Prepared tree {name} was encoded with a different FastLabelEncoder "
                    "mapping (or before the current encoder was refitted). Refit one encoder "
                    "on the full corpus and prepare every tree again."
                )

        diagnostics = MatchDiagnostics() if self.collect_diagnostics else None
        result = align_trees_fast_partial_beam_prepared(
            G,
            H,
            weight_by_id=self.encoder.weight_by_id,
            beam_width=self.beam_width,
            expansion_width=self.expansion_width,
            max_nodes_per_token_side=self.max_nodes_per_token_side,
            max_token_types_per_expansion=self.max_token_types_per_expansion,
            min_match_score=self.min_match_score,
            max_length=self.max_length,
            token_tie_keys=self._token_tie_keys(),
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            diagnostics.input_preprocessing_seconds += float(input_preprocessing_seconds)
            diagnostics.total_seconds = perf_counter() - total_start
            self.last_diagnostics_ = diagnostics.copy()

        path_orig = [
            (int(G.tree.orig_index[u]), int(H.tree.orig_index[v]))
            for u, v in result.path_internal
        ]
        return path_orig, float(result.score)


__all__ = [
    "HAVE_NUMBA",
    "FastBeamPreparedTree",
    "FastBeamTreePathMatcher",
    "prepare_fast_beam_encoded",
    "align_trees_fast_partial_beam_prepared",
]
