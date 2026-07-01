"""
Beam search over partial tree-path matchings.

This module implements the beam-search formulation in which a live state is a
valid partial matching, not a row/column subset of the dynamic-programming
matrix.  A state stores the last matched pair, the accumulated score, the
matching length, and a predecessor pointer.  Expansions append a feasible
descendant pair, so skipped tree nodes are handled implicitly.

The default expansion rule is intentionally heuristic but not merely a
placeholder:

- candidate pairs are generated from positive-scoring label-pair buckets;
- descendant queries use preorder intervals and label indexes rather than
  scanning full trees;
- local candidate ranking combines match score, label-pair rarity, gap/balance
  penalties, a small continuation estimate, optional subtree-sketch lookahead,
  and optional seeded exploration;
- frontier priority combines accumulated score with a conservative remaining
  height bound and, when enabled, a precomputed subtree-compatibility lookahead.

Users can override the expansion rule, the candidate heuristic, and/or the beam
priority.  The exact DP implementation remains in ``needleman_wunsch_tree.py``;
this module never forms the dense |G| x |H| DP table.
"""

from __future__ import annotations

import bisect
import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .tree_data import TreeData
from .needleman_wunsch_tree import AlignmentResult, WeightFn, id_match


# Legacy per-G-node candidate callback.  It is still supported: for every
# selected descendant u below the current G terminal, the callback may return
# candidate v nodes in H, which are then filtered to descendants of the current
# H terminal.
CandidateFn = Callable[[int, Any, TreeData], Sequence[int]]

# Optional quick admissibility predicate on raw labels.
MatchPredicate = Callable[[Any, Any], bool]

# Optional callable exposed to custom expansion rules.  It returns the current
# built-in lookahead estimate for a terminal pair, or 0.0 when lookahead is off.
BeamLookaheadScoreFn = Callable[[int, int], float]


@dataclass(frozen=True, slots=True)
class BeamHeuristicStats:
    """Read-only arrays and constants useful to custom beam heuristics."""

    depthG: np.ndarray
    depthH: np.ndarray
    heightG: np.ndarray
    heightH: np.ndarray
    max_match_score: float
    nG: int
    nH: int
    lookahead_enabled: bool = False
    lookahead_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class BeamCandidateContext:
    """
    Context passed to a custom candidate heuristic.

    The callable should return a larger-is-better priority for adding pair
    ``(u, v)`` to the candidate expansion set of the current state.
    """

    G: TreeData
    H: TreeData
    last_u: int
    last_v: int
    u: int
    v: int
    label_u: Any
    label_v: Any
    match_score: float
    current_score: float
    current_length: int
    depth_gap: int
    balance_gap: int
    rarity: float
    future_bound: float
    lookahead_score: float
    default_priority: float
    stats: BeamHeuristicStats
    rng: np.random.Generator


@dataclass(frozen=True, slots=True)
class BeamStateContext:
    """
    Context passed to a custom beam priority function.

    The callable should return the priority rho(z) used for top-B frontier
    truncation.  The score of the returned matching is always the accumulated
    matching score, not this priority.
    """

    G: TreeData
    H: TreeData
    last_u: int
    last_v: int
    score: float
    length: int
    future_bound: float
    lookahead_score: float
    default_priority: float
    stats: BeamHeuristicStats
    rng: np.random.Generator


@dataclass(frozen=True, slots=True)
class BeamExpansionContext:
    """
    Context passed to a custom expansion function.

    A custom expansion function should return an iterable of ``(u, v)`` pairs,
    or ``(u, v, score)`` triples.  Pairs must be strict descendants of
    ``last_u`` and ``last_v``; the implementation validates this.
    """

    G: TreeData
    H: TreeData
    last_u: int
    last_v: int
    score: float
    length: int
    layer: int
    stats: BeamHeuristicStats
    rng: np.random.Generator
    lookahead_score_fn: Optional[BeamLookaheadScoreFn] = None


CandidateHeuristic = Callable[[BeamCandidateContext], float]
PriorityFn = Callable[[BeamStateContext], float]
ExpansionFn = Callable[[BeamExpansionContext], Iterable[Union[Tuple[int, int], Tuple[int, int, float]]]]

# Public aliases with names that read naturally from TreePathMatcher.
BeamCandidateHeuristicFn = CandidateHeuristic
BeamPriorityFn = PriorityFn
BeamExpansionFn = ExpansionFn


def default_candidate_heuristic(ctx: BeamCandidateContext) -> float:
    """Return the built-in candidate priority for a proposed extension."""
    return float(ctx.default_priority)


def default_beam_priority(ctx: BeamStateContext) -> float:
    """Return the built-in frontier priority for a partial-matching state."""
    return float(ctx.default_priority)


@dataclass(frozen=True, slots=True)
class _LabelBucket:
    nodes: Tuple[int, ...]
    tins: Tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _TreeIndex:
    children: Tuple[Tuple[int, ...], ...]
    depth: np.ndarray
    height: np.ndarray
    tin: np.ndarray
    tout: np.ndarray
    preorder: Tuple[int, ...]
    label_key: Tuple[Any, ...]
    label_to_nodes: Mapping[Any, _LabelBucket]
    label_by_key: Mapping[Any, Any]
    max_height: int


@dataclass(frozen=True, slots=True)
class _LabelPair:
    key_g: Any
    key_h: Any
    score: float
    rarity: float
    static_priority: float


@dataclass(frozen=True, slots=True)
class _ScoredCandidate:
    u: int
    v: int
    match_score: float
    priority: float


@dataclass(frozen=True, slots=True)
class _LookaheadSketches:
    # ``below[node]`` summarizes labels in the strict descendants of ``node``.
    # ``sentinel`` summarizes all real nodes and is used for the initial state.
    below: Tuple[Tuple[Tuple[Any, float], ...], ...]
    sentinel: Tuple[Tuple[Any, float], ...]
    # ``below_chunks`` uses fixed-length downward label shingles.  It is a
    # deliberately heuristic way to notice that larger path-like chunks may be
    # available below a candidate pair.
    below_chunks: Tuple[Tuple[Tuple[Tuple[Any, ...], float], ...], ...]
    sentinel_chunks: Tuple[Tuple[Tuple[Any, ...], float], ...]
    chunk_size: int


@dataclass(slots=True)
class _LookaheadIndex:
    idxG: _TreeIndex
    idxH: _TreeIndex
    sketchesG: _LookaheadSketches
    sketchesH: _LookaheadSketches
    w_fn: WeightFn
    w_is_id: bool
    match_predicate: Optional[MatchPredicate]
    min_match_score: float
    max_match_score: float
    label_weight: float
    chunk_weight: float
    cache: Dict[int, float]


@dataclass(frozen=True, slots=True)
class _BeamParams:
    beam_width: int
    expansion_width: Optional[int]
    max_label_pair_scan: int
    max_label_pairs_per_expansion: Optional[int]
    max_nodes_per_label_side: int
    descendant_select_mode: str
    random_fraction: float
    min_match_score: float
    rarity_weight: float
    gap_penalty: float
    balance_penalty: float
    candidate_future_weight: float
    priority_future_weight: float
    priority_length_weight: float
    lookahead_enabled: bool
    lookahead_weight: float
    lookahead_sketch_size: int
    lookahead_depth_discount: float
    lookahead_chunk_size: int
    lookahead_label_weight: float
    lookahead_chunk_weight: float
    max_descendant_nodes_for_legacy_candidate_fn: int


def _safe_label_key(label: Any) -> Any:
    """Return a stable dictionary key for possibly unhashable labels."""

    try:
        hash(label)
    except TypeError:
        return ("repr", repr(label))
    return ("hash", label)


def _build_tree_index(T: TreeData) -> _TreeIndex:
    n = T.n
    parent = np.asarray(T.parent, dtype=np.int64)

    children_lists: List[List[int]] = [[] for _ in range(n)]
    for u in range(1, n):
        p = int(parent[u])
        children_lists[p].append(u)
    children: Tuple[Tuple[int, ...], ...] = tuple(tuple(c) for c in children_lists)

    depth = np.zeros(n, dtype=np.int32)
    for u in range(1, n):
        depth[u] = depth[int(parent[u])] + 1

    height = np.ones(n, dtype=np.int32)
    for u in range(n - 1, -1, -1):
        if children[u]:
            height[u] = 1 + max(int(height[c]) for c in children[u])

    tin = np.empty(n, dtype=np.int32)
    tout = np.empty(n, dtype=np.int32)
    preorder: List[int] = []
    counter = 0
    stack: List[Tuple[int, bool]] = [(0, False)]
    while stack:
        u, exiting = stack.pop()
        if exiting:
            tout[u] = counter
            continue
        tin[u] = counter
        preorder.append(u)
        counter += 1
        stack.append((u, True))
        # Reverse child order so traversal is deterministic and respects the
        # natural child order in children_lists.
        for c in reversed(children[u]):
            stack.append((c, False))

    label_key: List[Any] = [_safe_label_key(lab) for lab in T.label]
    label_by_key: Dict[Any, Any] = {}
    tmp_nodes: Dict[Any, List[int]] = {}
    for u, key in enumerate(label_key):
        label_by_key.setdefault(key, T.label[u])
        tmp_nodes.setdefault(key, []).append(u)

    # Store every bucket in preorder order; descendant queries then become
    # binary searches in the bucket's tin list.
    label_to_nodes: Dict[Any, _LabelBucket] = {}
    for key, nodes in tmp_nodes.items():
        nodes_sorted = sorted(nodes, key=lambda node: int(tin[node]))
        tins_sorted = tuple(int(tin[node]) for node in nodes_sorted)
        label_to_nodes[key] = _LabelBucket(nodes=tuple(nodes_sorted), tins=tins_sorted)

    return _TreeIndex(
        children=children,
        depth=depth,
        height=height,
        tin=tin,
        tout=tout,
        preorder=tuple(preorder),
        label_key=tuple(label_key),
        label_to_nodes=label_to_nodes,
        label_by_key=label_by_key,
        max_height=int(height[0]),
    )


def _strict_descendant_interval(index: _TreeIndex, terminal: int) -> Tuple[int, int]:
    """Return preorder-tin interval [lo, hi) for strict descendants."""

    if terminal < 0:
        return 0, len(index.preorder)
    return int(index.tin[terminal]) + 1, int(index.tout[terminal])


def _is_strict_descendant(index: _TreeIndex, terminal: int, node: int) -> bool:
    if node < 0 or node >= len(index.preorder):
        return False
    if terminal < 0:
        return True
    return int(index.tin[terminal]) < int(index.tin[node]) < int(index.tout[terminal])


def _descendants_for_label(index: _TreeIndex, key: Any, terminal: int) -> Tuple[int, ...]:
    bucket = index.label_to_nodes.get(key)
    if bucket is None:
        return ()
    lo_tin, hi_tin = _strict_descendant_interval(index, terminal)
    lo = bisect.bisect_left(bucket.tins, lo_tin)
    hi = bisect.bisect_left(bucket.tins, hi_tin)
    if lo >= hi:
        return ()
    return bucket.nodes[lo:hi]


def _all_descendants(index: _TreeIndex, terminal: int) -> Tuple[int, ...]:
    lo, hi = _strict_descendant_interval(index, terminal)
    if lo >= hi:
        return ()
    return index.preorder[lo:hi]


def _select_nodes(
    nodes: Sequence[int],
    k: Optional[int],
    *,
    mode: str,
    rng: np.random.Generator,
    random_fraction: float,
) -> List[int]:
    """Select a small deterministic/stochastic subset from an ordered node list."""

    n = len(nodes)
    if k is None or k >= n:
        return list(nodes)
    if k <= 0:
        return []

    mode = mode.lower()
    if mode == "first":
        return list(nodes[:k])
    if mode == "last":
        return list(nodes[-k:])
    if mode == "random":
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
        return [int(nodes[int(i)]) for i in idx]
    if mode == "spread":
        if k == 1:
            return [int(nodes[n // 2])]
        pos = np.linspace(0, n - 1, num=k)
        idx = np.unique(np.round(pos).astype(int))
        out = [int(nodes[int(i)]) for i in idx[:k]]
        if len(out) < k:
            for node in nodes:
                node_i = int(node)
                if node_i not in out:
                    out.append(node_i)
                    if len(out) == k:
                        break
        return out
    if mode == "mixed":
        random_fraction = min(1.0, max(0.0, float(random_fraction)))
        random_k = int(round(k * random_fraction))
        det_k = max(0, k - random_k)
        out: List[int] = []
        seen: set[int] = set()

        for node in _select_nodes(nodes, det_k, mode="spread", rng=rng, random_fraction=0.0):
            if node not in seen:
                seen.add(node)
                out.append(node)

        remaining = [int(node) for node in nodes if int(node) not in seen]
        if random_k > 0 and remaining:
            take = min(random_k, len(remaining))
            idx = rng.choice(len(remaining), size=take, replace=False)
            idx.sort()
            for i in idx:
                node = remaining[int(i)]
                if node not in seen:
                    seen.add(node)
                    out.append(node)

        # Fill any duplicate/rounding gaps deterministically.
        if len(out) < k:
            for node in nodes:
                node_i = int(node)
                if node_i not in seen:
                    seen.add(node_i)
                    out.append(node_i)
                    if len(out) == k:
                        break
        return out[:k]

    raise ValueError("descendant_select_mode must be one of: 'first', 'last', 'random', 'spread', 'mixed'")


def _score_label_pair(
    label_g: Any,
    label_h: Any,
    *,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
) -> float:
    if match_predicate is not None and not bool(match_predicate(label_g, label_h)):
        return -math.inf
    if w_is_id:
        return 1.0 if label_g == label_h else 0.0
    return float(w_fn(label_g, label_h))


def _build_label_pairs(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    *,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
    min_match_score: float,
    rarity_weight: float,
    max_label_pair_scan: int,
    rng: np.random.Generator,
) -> List[_LabelPair]:
    """Precompute positive-scoring label pairs for default expansion."""

    keysG = list(idxG.label_by_key.keys())
    keysH = list(idxH.label_by_key.keys())
    freqG = {key: len(bucket.nodes) for key, bucket in idxG.label_to_nodes.items()}
    freqH = {key: len(bucket.nodes) for key, bucket in idxH.label_to_nodes.items()}
    denom = math.log(max(2.0, float(G.n * H.n)))

    pairs: Dict[Tuple[Any, Any], _LabelPair] = {}

    def add_pair(key_g: Any, key_h: Any) -> None:
        pair_key = (key_g, key_h)
        if pair_key in pairs:
            return
        label_g = idxG.label_by_key[key_g]
        label_h = idxH.label_by_key[key_h]
        score = _score_label_pair(
            label_g,
            label_h,
            w_fn=w_fn,
            w_is_id=w_is_id,
            match_predicate=match_predicate,
        )
        if not math.isfinite(score) or score <= min_match_score:
            return
        fg = max(1, int(freqG[key_g]))
        fh = max(1, int(freqH[key_h]))
        # Normalized rarity in [roughly 0, 1]: rare label-pair buckets are more
        # informative, especially in noisy planted-path settings.
        rarity = max(0.0, math.log(max(1.0, float(G.n * H.n) / float(fg * fh))) / denom)
        static_priority = float(score) + float(rarity_weight) * rarity
        pairs[pair_key] = _LabelPair(
            key_g=key_g,
            key_h=key_h,
            score=float(score),
            rarity=float(rarity),
            static_priority=float(static_priority),
        )

    if w_is_id:
        for key in keysG:
            if key in idxH.label_by_key:
                add_pair(key, key)
    else:
        total_unique_pairs = len(keysG) * len(keysH)
        if total_unique_pairs <= max_label_pair_scan:
            for key_g in keysG:
                for key_h in keysH:
                    add_pair(key_g, key_h)
        else:
            # Large label alphabets make an all-pairs label scan undesirable.
            # Always test same-key pairs first, then add a reproducible sample of
            # other label pairs.  For broad non-equality weights, callers should
            # either increase max_label_pair_scan or pass expansion_fn.
            for key_g in keysG:
                if key_g in idxH.label_by_key:
                    add_pair(key_g, key_g)

            budget = max(0, int(max_label_pair_scan) - len(pairs))
            if budget > 0 and keysG and keysH:
                sampled: set[Tuple[int, int]] = set()
                attempts = 0
                max_attempts = max(10 * budget, budget + 100)
                while len(sampled) < budget and attempts < max_attempts:
                    attempts += 1
                    i = int(rng.integers(0, len(keysG)))
                    j = int(rng.integers(0, len(keysH)))
                    if (i, j) in sampled:
                        continue
                    sampled.add((i, j))
                    add_pair(keysG[i], keysH[j])

    out = list(pairs.values())
    out.sort(key=lambda p: (p.static_priority, p.score, p.rarity, repr(p.key_g), repr(p.key_h)), reverse=True)
    return out


def _default_candidate_priority(
    *,
    match_score: float,
    rarity: float,
    depth_gap: int,
    balance_gap: int,
    future_bound: float,
    lookahead_score: float,
    params: _BeamParams,
) -> float:
    return float(
        match_score
        + params.rarity_weight * rarity
        + params.candidate_future_weight * future_bound
        + params.lookahead_weight * lookahead_score
        - params.gap_penalty * depth_gap
        - params.balance_penalty * balance_gap
    )


def _future_bound_for_pair(idxG: _TreeIndex, idxH: _TreeIndex, u: int, v: int, max_match_score: float) -> float:
    # height includes the current node; descendants that can still be appended
    # are bounded by height-1 on each side.
    remaining = min(max(0, int(idxG.height[u]) - 1), max(0, int(idxH.height[v]) - 1))
    return float(remaining) * float(max_match_score)


def _top_k_mass(mass: Mapping[Any, float], k: int) -> Tuple[Tuple[Any, float], ...]:
    """Return the largest positive mass entries in deterministic order."""

    if k <= 0 or not mass:
        return ()
    items = [(key, float(value)) for key, value in mass.items() if float(value) > 0.0]
    if not items:
        return ()
    items.sort(key=lambda item: (item[1], repr(item[0])), reverse=True)
    return tuple((key, value) for key, value in items[:k])


def _build_lookahead_sketches(
    index: _TreeIndex,
    *,
    sketch_size: int,
    depth_discount: float,
    chunk_size: int,
) -> _LookaheadSketches:
    """Build capped discounted descendant-label and path-chunk sketches.

    A strict child contributes label weight 1.0, a grandchild contributes
    ``depth_discount``, and so on.  The chunk sketch stores capped fixed-length
    downward label shingles.  This keeps precomputation close to
    O(|T| * sketch_size * chunk_size) for ordinary bounded-degree trees.
    """

    n = len(index.preorder)
    cap = int(sketch_size)
    q = max(1, int(chunk_size))
    discount = float(depth_discount)

    below: List[Tuple[Tuple[Any, float], ...]] = [() for _ in range(n)]
    below_chunks: List[Tuple[Tuple[Tuple[Any, ...], float], ...]] = [() for _ in range(n)]
    # prefixes[u][length] stores capped label sequences of exactly ``length``
    # starting at u.  Index 0 is unused.
    prefixes: List[List[Tuple[Tuple[Tuple[Any, ...], float], ...]]] = [
        [tuple() for _ in range(q + 1)] for _ in range(n)
    ]

    for u_raw in reversed(index.preorder):
        u = int(u_raw)

        label_mass: Dict[Any, float] = {}
        for c in index.children[u]:
            child_key = index.label_key[int(c)]
            label_mass[child_key] = label_mass.get(child_key, 0.0) + 1.0
            for key, value in below[int(c)]:
                label_mass[key] = label_mass.get(key, 0.0) + discount * float(value)
        below[u] = _top_k_mass(label_mass, cap)

        prefixes[u][1] = (((index.label_key[u],), 1.0),)
        for length in range(2, q + 1):
            prefix_mass: Dict[Tuple[Any, ...], float] = {}
            for c in index.children[u]:
                for seq, value in prefixes[int(c)][length - 1]:
                    prefix_mass[(index.label_key[u],) + tuple(seq)] = prefix_mass.get(
                        (index.label_key[u],) + tuple(seq), 0.0
                    ) + float(value)
            prefixes[u][length] = _top_k_mass(prefix_mass, cap)  # type: ignore[assignment]

        chunk_mass: Dict[Tuple[Any, ...], float] = {}
        for c in index.children[u]:
            c_int = int(c)
            for seq, value in prefixes[c_int][q]:
                chunk_mass[seq] = chunk_mass.get(seq, 0.0) + float(value)
            for seq, value in below_chunks[c_int]:
                chunk_mass[seq] = chunk_mass.get(seq, 0.0) + discount * float(value)
        below_chunks[u] = _top_k_mass(chunk_mass, cap)  # type: ignore[assignment]

    root_mass: Dict[Any, float] = {index.label_key[0]: 1.0}
    for key, value in below[0]:
        root_mass[key] = root_mass.get(key, 0.0) + discount * float(value)
    sentinel = _top_k_mass(root_mass, cap)

    root_chunk_mass: Dict[Tuple[Any, ...], float] = {}
    for seq, value in prefixes[0][q]:
        root_chunk_mass[seq] = root_chunk_mass.get(seq, 0.0) + float(value)
    for seq, value in below_chunks[0]:
        root_chunk_mass[seq] = root_chunk_mass.get(seq, 0.0) + discount * float(value)
    sentinel_chunks = _top_k_mass(root_chunk_mass, cap)  # type: ignore[assignment]

    return _LookaheadSketches(
        below=tuple(below),
        sentinel=sentinel,
        below_chunks=tuple(below_chunks),
        sentinel_chunks=sentinel_chunks,
        chunk_size=q,
    )


def _build_lookahead_index(
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    *,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
    min_match_score: float,
    max_match_score: float,
    sketch_size: int,
    depth_discount: float,
    chunk_size: int,
    label_weight: float,
    chunk_weight: float,
) -> _LookaheadIndex:
    return _LookaheadIndex(
        idxG=idxG,
        idxH=idxH,
        sketchesG=_build_lookahead_sketches(
            idxG,
            sketch_size=int(sketch_size),
            depth_discount=float(depth_discount),
            chunk_size=int(chunk_size),
        ),
        sketchesH=_build_lookahead_sketches(
            idxH,
            sketch_size=int(sketch_size),
            depth_discount=float(depth_discount),
            chunk_size=int(chunk_size),
        ),
        w_fn=w_fn,
        w_is_id=bool(w_is_id),
        match_predicate=match_predicate,
        min_match_score=float(min_match_score),
        max_match_score=float(max_match_score),
        label_weight=float(label_weight),
        chunk_weight=float(chunk_weight),
        cache={},
    )


def _lookahead_pair_key(lookahead: _LookaheadIndex, u: int, v: int) -> int:
    # Shift by +1 so that the sentinel terminal -1 can be cached.
    return (int(u) + 1) * (len(lookahead.idxH.preorder) + 1) + (int(v) + 1)


def _sketch_for_terminal(sketches: _LookaheadSketches, terminal: int) -> Tuple[Tuple[Any, float], ...]:
    if terminal < 0:
        return sketches.sentinel
    return sketches.below[int(terminal)]


def _chunk_sketch_for_terminal(
    sketches: _LookaheadSketches,
    terminal: int,
) -> Tuple[Tuple[Tuple[Any, ...], float], ...]:
    if terminal < 0:
        return sketches.sentinel_chunks
    return sketches.below_chunks[int(terminal)]


def _lookahead_height_cap(lookahead: _LookaheadIndex, u: int, v: int) -> float:
    if u < 0 or v < 0:
        remaining = min(int(lookahead.idxG.max_height), int(lookahead.idxH.max_height))
        return float(remaining) * float(lookahead.max_match_score)
    return _future_bound_for_pair(lookahead.idxG, lookahead.idxH, int(u), int(v), lookahead.max_match_score)


def _lookahead_label_overlap(
    lookahead: _LookaheadIndex,
    sketch_g: Tuple[Tuple[Any, float], ...],
    sketch_h: Tuple[Tuple[Any, float], ...],
) -> float:
    """Score capped descendant-label overlap between two terminals."""

    if not sketch_g or not sketch_h:
        return 0.0

    if lookahead.w_is_id and lookahead.match_predicate is None:
        raw = 0.0
        mass_h = {key: float(value) for key, value in sketch_h}
        for key_g, mass_g in sketch_g:
            mass = min(float(mass_g), mass_h.get(key_g, 0.0))
            if mass <= 0.0:
                continue
            label_g = lookahead.idxG.label_by_key[key_g]
            label_h = lookahead.idxH.label_by_key[key_g]
            score = _score_label_pair(
                label_g,
                label_h,
                w_fn=lookahead.w_fn,
                w_is_id=lookahead.w_is_id,
                match_predicate=lookahead.match_predicate,
            )
            if math.isfinite(score) and score > lookahead.min_match_score:
                raw += mass * float(score)
        return raw

    pair_scores: List[Tuple[float, str, str, Any, Any]] = []
    for key_g, _mass_g in sketch_g:
        label_g = lookahead.idxG.label_by_key[key_g]
        for key_h, _mass_h in sketch_h:
            label_h = lookahead.idxH.label_by_key[key_h]
            score = _score_label_pair(
                label_g,
                label_h,
                w_fn=lookahead.w_fn,
                w_is_id=lookahead.w_is_id,
                match_predicate=lookahead.match_predicate,
            )
            if math.isfinite(score) and score > lookahead.min_match_score:
                pair_scores.append((float(score), repr(key_g), repr(key_h), key_g, key_h))

    if not pair_scores:
        return 0.0

    raw = 0.0
    pair_scores.sort(reverse=True)
    rem_g: Dict[Any, float] = {key: float(value) for key, value in sketch_g}
    rem_h: Dict[Any, float] = {key: float(value) for key, value in sketch_h}
    for score, _repr_g, _repr_h, key_g, key_h in pair_scores:
        mass = min(rem_g.get(key_g, 0.0), rem_h.get(key_h, 0.0))
        if mass <= 0.0:
            continue
        raw += mass * float(score)
        rem_g[key_g] = rem_g.get(key_g, 0.0) - mass
        rem_h[key_h] = rem_h.get(key_h, 0.0) - mass
    return raw


def _chunk_pair_score(
    lookahead: _LookaheadIndex,
    seq_g: Tuple[Any, ...],
    seq_h: Tuple[Any, ...],
) -> float:
    """Return the path-chunk compatibility score for two equal-length shingles."""

    if len(seq_g) != len(seq_h):
        return 0.0
    if lookahead.w_is_id and lookahead.match_predicate is None:
        return float(len(seq_g)) * float(lookahead.max_match_score) if seq_g == seq_h else 0.0

    total = 0.0
    for key_g, key_h in zip(seq_g, seq_h):
        label_g = lookahead.idxG.label_by_key[key_g]
        label_h = lookahead.idxH.label_by_key[key_h]
        score = _score_label_pair(
            label_g,
            label_h,
            w_fn=lookahead.w_fn,
            w_is_id=lookahead.w_is_id,
            match_predicate=lookahead.match_predicate,
        )
        if not math.isfinite(score) or score <= lookahead.min_match_score:
            return 0.0
        total += float(score)
    return total


def _lookahead_chunk_overlap(
    lookahead: _LookaheadIndex,
    chunks_g: Tuple[Tuple[Tuple[Any, ...], float], ...],
    chunks_h: Tuple[Tuple[Tuple[Any, ...], float], ...],
) -> float:
    """Score capped overlap between fixed-length descendant path shingles."""

    if not chunks_g or not chunks_h:
        return 0.0

    if lookahead.w_is_id and lookahead.match_predicate is None:
        raw = 0.0
        mass_h = {seq: float(value) for seq, value in chunks_h}
        for seq_g, mass_g in chunks_g:
            mass = min(float(mass_g), mass_h.get(seq_g, 0.0))
            if mass <= 0.0:
                continue
            raw += mass * float(len(seq_g)) * float(lookahead.max_match_score)
        return raw

    pair_scores: List[Tuple[float, str, str, Tuple[Any, ...], Tuple[Any, ...]]] = []
    for seq_g, _mass_g in chunks_g:
        for seq_h, _mass_h in chunks_h:
            score = _chunk_pair_score(lookahead, tuple(seq_g), tuple(seq_h))
            if score > 0.0:
                pair_scores.append((float(score), repr(seq_g), repr(seq_h), tuple(seq_g), tuple(seq_h)))

    if not pair_scores:
        return 0.0

    raw = 0.0
    pair_scores.sort(reverse=True)
    rem_g: Dict[Tuple[Any, ...], float] = {tuple(seq): float(value) for seq, value in chunks_g}
    rem_h: Dict[Tuple[Any, ...], float] = {tuple(seq): float(value) for seq, value in chunks_h}
    for score, _repr_g, _repr_h, seq_g, seq_h in pair_scores:
        mass = min(rem_g.get(seq_g, 0.0), rem_h.get(seq_h, 0.0))
        if mass <= 0.0:
            continue
        raw += mass * float(score)
        rem_g[seq_g] = rem_g.get(seq_g, 0.0) - mass
        rem_h[seq_h] = rem_h.get(seq_h, 0.0) - mass
    return raw


def _lookahead_score(lookahead: Optional[_LookaheadIndex], u: int, v: int) -> float:
    """Estimate future match potential below terminal pair ``(u, v)``.

    This is intentionally non-admissible: it is a ranking feature, not a proof
    bound.  It combines two capped sketches: descendant-label mass and
    fixed-length downward path shingles.  The shingle component is the
    lightweight "look farther ahead" part; it can notice that a candidate opens
    access to a coherent future path chunk rather than only to isolated labels.
    """

    if lookahead is None or lookahead.max_match_score <= 0.0:
        return 0.0

    cache_key = _lookahead_pair_key(lookahead, u, v)
    cached = lookahead.cache.get(cache_key)
    if cached is not None:
        return float(cached)

    cap = _lookahead_height_cap(lookahead, int(u), int(v))
    if cap <= 0.0:
        lookahead.cache[cache_key] = 0.0
        return 0.0

    raw_label = 0.0
    if lookahead.label_weight > 0.0:
        sketch_g = _sketch_for_terminal(lookahead.sketchesG, int(u))
        sketch_h = _sketch_for_terminal(lookahead.sketchesH, int(v))
        raw_label = _lookahead_label_overlap(lookahead, sketch_g, sketch_h)

    raw_chunk = 0.0
    if lookahead.chunk_weight > 0.0 and lookahead.sketchesG.chunk_size > 0 and lookahead.sketchesH.chunk_size > 0:
        chunks_g = _chunk_sketch_for_terminal(lookahead.sketchesG, int(u))
        chunks_h = _chunk_sketch_for_terminal(lookahead.sketchesH, int(v))
        raw_chunk = _lookahead_chunk_overlap(lookahead, chunks_g, chunks_h)

    raw = float(lookahead.label_weight) * float(raw_label) + float(lookahead.chunk_weight) * float(raw_chunk)
    out = max(0.0, min(raw, float(cap)))
    lookahead.cache[cache_key] = out
    return out


def _push_candidate(
    heap: List[Tuple[float, int, _ScoredCandidate]],
    candidate: _ScoredCandidate,
    *,
    cap: Optional[int],
    counter: int,
) -> None:
    item = (float(candidate.priority), int(counter), candidate)
    if cap is None:
        heapq.heappush(heap, item)
        return
    if cap <= 0:
        return
    if len(heap) < cap:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0] or (item[0] == heap[0][0] and item[1] < heap[0][1]):
        heapq.heapreplace(heap, item)


def _candidate_from_pair(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    stats: BeamHeuristicStats,
    params: _BeamParams,
    *,
    u: int,
    v: int,
    last_u: int,
    last_v: int,
    current_score: float,
    current_length: int,
    match_score: float,
    rarity: float,
    candidate_heuristic: Optional[CandidateHeuristic],
    lookahead: Optional[_LookaheadIndex],
    rng: np.random.Generator,
) -> Optional[_ScoredCandidate]:
    if not math.isfinite(match_score) or match_score <= params.min_match_score:
        return None

    if last_u < 0:
        gap_g = int(idxG.depth[u])
        step_g = int(idxG.depth[u]) + 1
    else:
        gap_g = max(0, int(idxG.depth[u]) - int(idxG.depth[last_u]) - 1)
        step_g = max(1, int(idxG.depth[u]) - int(idxG.depth[last_u]))

    if last_v < 0:
        gap_h = int(idxH.depth[v])
        step_h = int(idxH.depth[v]) + 1
    else:
        gap_h = max(0, int(idxH.depth[v]) - int(idxH.depth[last_v]) - 1)
        step_h = max(1, int(idxH.depth[v]) - int(idxH.depth[last_v]))

    depth_gap = int(gap_g + gap_h)
    balance_gap = int(abs(step_g - step_h))
    future_bound = _future_bound_for_pair(idxG, idxH, u, v, stats.max_match_score)
    lookahead_score = _lookahead_score(lookahead, u, v)
    default_priority = _default_candidate_priority(
        match_score=match_score,
        rarity=rarity,
        depth_gap=depth_gap,
        balance_gap=balance_gap,
        future_bound=future_bound,
        lookahead_score=lookahead_score,
        params=params,
    )

    if candidate_heuristic is not None:
        ctx = BeamCandidateContext(
            G=G,
            H=H,
            last_u=last_u,
            last_v=last_v,
            u=u,
            v=v,
            label_u=G.label[u],
            label_v=H.label[v],
            match_score=float(match_score),
            current_score=float(current_score),
            current_length=int(current_length),
            depth_gap=depth_gap,
            balance_gap=balance_gap,
            rarity=float(rarity),
            future_bound=float(future_bound),
            lookahead_score=float(lookahead_score),
            default_priority=float(default_priority),
            stats=stats,
            rng=rng,
        )
        priority = float(candidate_heuristic(ctx))
    else:
        priority = default_priority

    if not math.isfinite(priority):
        return None
    return _ScoredCandidate(u=int(u), v=int(v), match_score=float(match_score), priority=float(priority))


def _generate_default_expansion(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    stats: BeamHeuristicStats,
    label_pairs: Sequence[_LabelPair],
    params: _BeamParams,
    *,
    last_u: int,
    last_v: int,
    current_score: float,
    current_length: int,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
    candidate_heuristic: Optional[CandidateHeuristic],
    lookahead: Optional[_LookaheadIndex],
    rng: np.random.Generator,
) -> List[_ScoredCandidate]:
    heap: List[Tuple[float, int, _ScoredCandidate]] = []
    seen: set[int] = set()
    cap = params.expansion_width
    counter = 0
    scanned = 0

    for label_pair in label_pairs:
        if params.max_label_pairs_per_expansion is not None and scanned >= params.max_label_pairs_per_expansion:
            # If no candidates at all have been found within the normal scan
            # budget, keep scanning.  This avoids a brittle failure mode in
            # deep subtrees whose labels are not among the globally top buckets.
            if heap:
                break
        scanned += 1

        nodes_g = _descendants_for_label(idxG, label_pair.key_g, last_u)
        if not nodes_g:
            continue
        nodes_h = _descendants_for_label(idxH, label_pair.key_h, last_v)
        if not nodes_h:
            continue

        sel_g = _select_nodes(
            nodes_g,
            params.max_nodes_per_label_side,
            mode=params.descendant_select_mode,
            rng=rng,
            random_fraction=params.random_fraction,
        )
        sel_h = _select_nodes(
            nodes_h,
            params.max_nodes_per_label_side,
            mode=params.descendant_select_mode,
            rng=rng,
            random_fraction=params.random_fraction,
        )

        for u in sel_g:
            for v in sel_h:
                key = int(u) * H.n + int(v)
                if key in seen:
                    continue
                seen.add(key)

                # Re-score raw labels to support custom weights and labels that
                # share a fallback repr key.  For ordinary equality matching this
                # is cheap and deterministic.
                match_score = _score_label_pair(
                    G.label[u],
                    H.label[v],
                    w_fn=w_fn,
                    w_is_id=w_is_id,
                    match_predicate=match_predicate,
                )
                cand = _candidate_from_pair(
                    G,
                    H,
                    idxG,
                    idxH,
                    stats,
                    params,
                    u=int(u),
                    v=int(v),
                    last_u=last_u,
                    last_v=last_v,
                    current_score=current_score,
                    current_length=current_length,
                    match_score=match_score,
                    rarity=label_pair.rarity,
                    candidate_heuristic=candidate_heuristic,
                    lookahead=lookahead,
                    rng=rng,
                )
                if cand is not None:
                    _push_candidate(heap, cand, cap=cap, counter=counter)
                    counter += 1

    out = [item[2] for item in heap]
    out.sort(key=lambda c: (c.priority, c.match_score, -c.u, -c.v), reverse=True)
    return out


def _generate_legacy_candidate_fn_expansion(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    stats: BeamHeuristicStats,
    params: _BeamParams,
    *,
    last_u: int,
    last_v: int,
    current_score: float,
    current_length: int,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
    candidate_fn: CandidateFn,
    candidate_heuristic: Optional[CandidateHeuristic],
    lookahead: Optional[_LookaheadIndex],
    rng: np.random.Generator,
) -> List[_ScoredCandidate]:
    descendants_g = _all_descendants(idxG, last_u)
    selected_g = _select_nodes(
        descendants_g,
        params.max_descendant_nodes_for_legacy_candidate_fn,
        mode=params.descendant_select_mode,
        rng=rng,
        random_fraction=params.random_fraction,
    )

    heap: List[Tuple[float, int, _ScoredCandidate]] = []
    seen: set[int] = set()
    cap = params.expansion_width
    counter = 0

    for u in selected_g:
        v_list = candidate_fn(int(u), G.label[int(u)], H)
        for v_raw in v_list:
            v = int(v_raw)
            if v < 0 or v >= H.n:
                raise ValueError(f"candidate_fn returned v={v} outside [0, {H.n - 1}]")
            if not _is_strict_descendant(idxH, last_v, v):
                continue
            key = int(u) * H.n + v
            if key in seen:
                continue
            seen.add(key)
            match_score = _score_label_pair(
                G.label[int(u)],
                H.label[v],
                w_fn=w_fn,
                w_is_id=w_is_id,
                match_predicate=match_predicate,
            )
            # Use a neutral rarity value because the legacy callback may use an
            # arbitrary blocking scheme unrelated to labels.
            cand = _candidate_from_pair(
                G,
                H,
                idxG,
                idxH,
                stats,
                params,
                u=int(u),
                v=v,
                last_u=last_u,
                last_v=last_v,
                current_score=current_score,
                current_length=current_length,
                match_score=match_score,
                rarity=0.0,
                candidate_heuristic=candidate_heuristic,
                lookahead=lookahead,
                rng=rng,
            )
            if cand is not None:
                _push_candidate(heap, cand, cap=cap, counter=counter)
                counter += 1

    out = [item[2] for item in heap]
    out.sort(key=lambda c: (c.priority, c.match_score, -c.u, -c.v), reverse=True)
    return out


def _generate_custom_expansion(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    stats: BeamHeuristicStats,
    params: _BeamParams,
    *,
    last_u: int,
    last_v: int,
    current_score: float,
    current_length: int,
    layer: int,
    w_fn: WeightFn,
    w_is_id: bool,
    match_predicate: Optional[MatchPredicate],
    expansion_fn: ExpansionFn,
    candidate_heuristic: Optional[CandidateHeuristic],
    lookahead: Optional[_LookaheadIndex],
    rng: np.random.Generator,
) -> List[_ScoredCandidate]:
    ctx = BeamExpansionContext(
        G=G,
        H=H,
        last_u=last_u,
        last_v=last_v,
        score=float(current_score),
        length=int(current_length),
        layer=int(layer),
        stats=stats,
        rng=rng,
        lookahead_score_fn=(lambda u, v: _lookahead_score(lookahead, int(u), int(v))) if lookahead is not None else None,
    )
    raw_candidates = expansion_fn(ctx)

    heap: List[Tuple[float, int, _ScoredCandidate]] = []
    seen: set[int] = set()
    cap = params.expansion_width
    counter = 0

    for raw in raw_candidates:
        if len(raw) == 2:  # type: ignore[arg-type]
            u_raw, v_raw = raw  # type: ignore[misc]
            provided_score: Optional[float] = None
        elif len(raw) == 3:  # type: ignore[arg-type]
            u_raw, v_raw, score_raw = raw  # type: ignore[misc]
            provided_score = float(score_raw)
        else:
            raise ValueError("expansion_fn must return (u, v) pairs or (u, v, score) triples")

        u = int(u_raw)
        v = int(v_raw)
        if u < 0 or u >= G.n or v < 0 or v >= H.n:
            raise ValueError(f"expansion_fn returned pair ({u}, {v}) outside tree bounds")
        if not _is_strict_descendant(idxG, last_u, u) or not _is_strict_descendant(idxH, last_v, v):
            raise ValueError(
                f"expansion_fn returned infeasible pair ({u}, {v}) for terminal ({last_u}, {last_v}); pairs must be strict descendants"
            )
        key = u * H.n + v
        if key in seen:
            continue
        seen.add(key)

        if provided_score is None:
            match_score = _score_label_pair(
                G.label[u],
                H.label[v],
                w_fn=w_fn,
                w_is_id=w_is_id,
                match_predicate=match_predicate,
            )
        else:
            match_score = float(provided_score)

        cand = _candidate_from_pair(
            G,
            H,
            idxG,
            idxH,
            stats,
            params,
            u=u,
            v=v,
            last_u=last_u,
            last_v=last_v,
            current_score=current_score,
            current_length=current_length,
            match_score=match_score,
            rarity=0.0,
            candidate_heuristic=candidate_heuristic,
            lookahead=lookahead,
            rng=rng,
        )
        if cand is not None:
            _push_candidate(heap, cand, cap=cap, counter=counter)
            counter += 1

    out = [item[2] for item in heap]
    out.sort(key=lambda c: (c.priority, c.match_score, -c.u, -c.v), reverse=True)
    return out


def _state_priority(
    G: TreeData,
    H: TreeData,
    idxG: _TreeIndex,
    idxH: _TreeIndex,
    stats: BeamHeuristicStats,
    params: _BeamParams,
    *,
    last_u: int,
    last_v: int,
    score: float,
    length: int,
    priority_fn: Optional[PriorityFn],
    lookahead: Optional[_LookaheadIndex],
    rng: np.random.Generator,
) -> float:
    if last_u < 0 or last_v < 0:
        future_bound = float(min(idxG.max_height, idxH.max_height)) * float(stats.max_match_score)
    else:
        future_bound = _future_bound_for_pair(idxG, idxH, last_u, last_v, stats.max_match_score)
    lookahead_score = _lookahead_score(lookahead, last_u, last_v)
    default_priority = float(
        score
        + params.priority_future_weight * future_bound
        + params.lookahead_weight * lookahead_score
        + params.priority_length_weight * length
    )
    if priority_fn is None:
        return default_priority
    ctx = BeamStateContext(
        G=G,
        H=H,
        last_u=int(last_u),
        last_v=int(last_v),
        score=float(score),
        length=int(length),
        future_bound=float(future_bound),
        lookahead_score=float(lookahead_score),
        default_priority=float(default_priority),
        stats=stats,
        rng=rng,
    )
    priority = float(priority_fn(ctx))
    if not math.isfinite(priority):
        return -math.inf
    return priority


def _traceback(best_state_id: int, last_u: List[int], last_v: List[int], prev: List[int]) -> List[Tuple[int, int]]:
    path_rev: List[Tuple[int, int]] = []
    sid = int(best_state_id)
    while sid > 0:
        path_rev.append((int(last_u[sid]), int(last_v[sid])))
        sid = int(prev[sid])
    path_rev.reverse()
    return path_rev


def _align_trees_beam_once(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn],
    params: _BeamParams,
    candidate_fn: Optional[CandidateFn],
    expansion_fn: Optional[ExpansionFn],
    candidate_heuristic: Optional[CandidateHeuristic],
    priority_fn: Optional[PriorityFn],
    match_predicate: Optional[MatchPredicate],
    seed: int,
    max_length: Optional[int],
) -> AlignmentResult:
    if w is None:
        w_fn: WeightFn = id_match
        w_is_id = True
    else:
        w_fn = w
        w_is_id = w is id_match

    idxG = _build_tree_index(G)
    idxH = _build_tree_index(H)
    rng = np.random.default_rng(seed)

    label_pairs = _build_label_pairs(
        G,
        H,
        idxG,
        idxH,
        w_fn=w_fn,
        w_is_id=w_is_id,
        match_predicate=match_predicate,
        min_match_score=params.min_match_score,
        rarity_weight=params.rarity_weight,
        max_label_pair_scan=params.max_label_pair_scan,
        rng=rng,
    )
    max_match_score = max((p.score for p in label_pairs), default=0.0)
    stats = BeamHeuristicStats(
        depthG=idxG.depth,
        depthH=idxH.depth,
        heightG=idxG.height,
        heightH=idxH.height,
        max_match_score=float(max_match_score),
        nG=G.n,
        nH=H.n,
        lookahead_enabled=bool(params.lookahead_enabled),
        lookahead_weight=float(params.lookahead_weight),
    )
    lookahead: Optional[_LookaheadIndex] = None
    if params.lookahead_enabled and max_match_score > 0.0:
        lookahead = _build_lookahead_index(
            idxG,
            idxH,
            w_fn=w_fn,
            w_is_id=w_is_id,
            match_predicate=match_predicate,
            min_match_score=params.min_match_score,
            max_match_score=float(max_match_score),
            sketch_size=params.lookahead_sketch_size,
            depth_discount=params.lookahead_depth_discount,
            chunk_size=params.lookahead_chunk_size,
            label_weight=params.lookahead_label_weight,
            chunk_weight=params.lookahead_chunk_weight,
        )

    if max_length is None:
        lmax = int(min(idxG.max_height, idxH.max_height))
    else:
        lmax = int(max_length)
        if lmax < 0:
            raise ValueError("max_length must be nonnegative")

    # State pool, using parallel lists rather than one Python object per state.
    last_u: List[int] = [-1]
    last_v: List[int] = [-1]
    scores: List[float] = [0.0]
    prev: List[int] = [-1]
    lengths: List[int] = [0]

    frontier: List[int] = [0]
    best_state = 0
    best_score = 0.0

    for layer in range(lmax):
        pruned_by_terminal: Dict[int, int] = {}

        for sid in frontier:
            lu = int(last_u[sid])
            lv = int(last_v[sid])
            current_score = float(scores[sid])
            current_length = int(lengths[sid])

            if expansion_fn is not None:
                candidates = _generate_custom_expansion(
                    G,
                    H,
                    idxG,
                    idxH,
                    stats,
                    params,
                    last_u=lu,
                    last_v=lv,
                    current_score=current_score,
                    current_length=current_length,
                    layer=layer,
                    w_fn=w_fn,
                    w_is_id=w_is_id,
                    match_predicate=match_predicate,
                    expansion_fn=expansion_fn,
                    candidate_heuristic=candidate_heuristic,
                    lookahead=lookahead,
                    rng=rng,
                )
            elif candidate_fn is not None:
                candidates = _generate_legacy_candidate_fn_expansion(
                    G,
                    H,
                    idxG,
                    idxH,
                    stats,
                    params,
                    last_u=lu,
                    last_v=lv,
                    current_score=current_score,
                    current_length=current_length,
                    w_fn=w_fn,
                    w_is_id=w_is_id,
                    match_predicate=match_predicate,
                    candidate_fn=candidate_fn,
                    candidate_heuristic=candidate_heuristic,
                    lookahead=lookahead,
                    rng=rng,
                )
            else:
                candidates = _generate_default_expansion(
                    G,
                    H,
                    idxG,
                    idxH,
                    stats,
                    label_pairs,
                    params,
                    last_u=lu,
                    last_v=lv,
                    current_score=current_score,
                    current_length=current_length,
                    w_fn=w_fn,
                    w_is_id=w_is_id,
                    match_predicate=match_predicate,
                    candidate_heuristic=candidate_heuristic,
                    lookahead=lookahead,
                    rng=rng,
                )

            for cand in candidates:
                new_score = current_score + float(cand.match_score)
                new_id = len(scores)
                last_u.append(int(cand.u))
                last_v.append(int(cand.v))
                scores.append(float(new_score))
                prev.append(int(sid))
                lengths.append(current_length + 1)

                terminal_key = int(cand.u) * H.n + int(cand.v)
                old_id = pruned_by_terminal.get(terminal_key)
                if old_id is None:
                    pruned_by_terminal[terminal_key] = new_id
                else:
                    # Local dominance: same terminal pair, keep larger accumulated score.
                    if new_score > scores[old_id]:
                        pruned_by_terminal[terminal_key] = new_id

        if not pruned_by_terminal:
            break

        pruned_states = list(pruned_by_terminal.values())
        for sid in pruned_states:
            if scores[sid] > best_score:
                best_score = float(scores[sid])
                best_state = int(sid)

        ranked: List[Tuple[float, float, int, int]] = []
        for sid in pruned_states:
            pri = _state_priority(
                G,
                H,
                idxG,
                idxH,
                stats,
                params,
                last_u=last_u[sid],
                last_v=last_v[sid],
                score=scores[sid],
                length=lengths[sid],
                priority_fn=priority_fn,
                lookahead=lookahead,
                rng=rng,
            )
            if math.isfinite(pri):
                # Deterministic tie-breaking: priority, then score, then earlier state id.
                ranked.append((float(pri), float(scores[sid]), -int(sid), int(sid)))

        if not ranked:
            break
        top = heapq.nlargest(params.beam_width, ranked, key=lambda x: (x[0], x[1], x[2]))
        frontier = [sid for (_pri, _score, _neg_sid, sid) in top]

    if best_state == 0:
        return AlignmentResult(path_internal=[], score=0.0, end_internal=(0, 0), A=None, C=None)

    path = _traceback(best_state, last_u, last_v, prev)
    return AlignmentResult(
        path_internal=path,
        score=float(best_score),
        end_internal=(int(last_u[best_state]), int(last_v[best_state])),
        A=None,
        C=None,
    )


def _validate_beam_params(params: _BeamParams) -> None:
    if params.beam_width < 1:
        raise ValueError("beam_width must be >= 1")
    if params.expansion_width is not None and params.expansion_width < 1:
        raise ValueError("expansion_width must be >= 1, or None for exhaustive generated expansions")
    if params.max_label_pair_scan < 1:
        raise ValueError("max_label_pair_scan must be >= 1")
    if params.max_label_pairs_per_expansion is not None and params.max_label_pairs_per_expansion < 1:
        raise ValueError("max_label_pairs_per_expansion must be >= 1, or None")
    if params.max_nodes_per_label_side < 1:
        raise ValueError("max_nodes_per_label_side must be >= 1")
    if not (0.0 <= params.random_fraction <= 1.0):
        raise ValueError("random_fraction must be in [0, 1]")
    if params.min_match_score < -math.inf:
        raise ValueError("min_match_score is invalid")
    if params.lookahead_weight < 0.0:
        raise ValueError("lookahead_weight must be nonnegative")
    if params.lookahead_sketch_size < 1:
        raise ValueError("lookahead_sketch_size must be >= 1")
    if not (0.0 <= params.lookahead_depth_discount <= 1.0):
        raise ValueError("lookahead_depth_discount must be in [0, 1]")
    if params.lookahead_chunk_size < 1:
        raise ValueError("lookahead_chunk_size must be >= 1")
    if params.lookahead_label_weight < 0.0:
        raise ValueError("lookahead_label_weight must be nonnegative")
    if params.lookahead_chunk_weight < 0.0:
        raise ValueError("lookahead_chunk_weight must be nonnegative")
    if params.max_descendant_nodes_for_legacy_candidate_fn < 1:
        raise ValueError("max_descendant_nodes_for_legacy_candidate_fn must be >= 1")


def align_trees_beam(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    beam_width: int = 200,
    expansion_width: Optional[int] = 64,
    candidate_fn: Optional[CandidateFn] = None,
    expansion_fn: Optional[ExpansionFn] = None,
    candidate_heuristic: Optional[CandidateHeuristic] = None,
    priority_fn: Optional[PriorityFn] = None,
    max_candidates_per_label: Optional[int] = None,
    max_candidates_per_u: Optional[int] = None,
    candidate_select_mode: str = "mixed",
    seed: int = 0,
    n_restarts: int = 1,
    match_predicate: Optional[MatchPredicate] = None,
    prefer_match_on_tie: bool = True,
    min_match_score: float = 0.0,
    max_label_pair_scan: int = 100_000,
    max_label_pairs_per_expansion: Optional[int] = 2_048,
    max_nodes_per_label_side: int = 8,
    random_fraction: float = 0.10,
    rarity_weight: float = 0.25,
    gap_penalty: float = 0.03,
    balance_penalty: float = 0.01,
    candidate_future_weight: float = 0.03,
    priority_future_weight: float = 0.20,
    priority_length_weight: float = 0.0,
    lookahead: bool = False,
    lookahead_weight: float = 0.35,
    lookahead_sketch_size: int = 32,
    lookahead_depth_discount: float = 0.85,
    lookahead_chunk_size: int = 3,
    lookahead_label_weight: float = 1.0,
    lookahead_chunk_weight: float = 0.5,
    max_length: Optional[int] = None,
) -> AlignmentResult:
    """
    Beam search over valid partial matchings.

    Parameters
    ----------
    G, H:
        Trees in ``TreeData`` form.
    w:
        Weight function ``w(label_G, label_H)``.  If ``None``, equality labels
        receive score 1 and all other pairs receive 0.  The beam only expands
        pairs with score strictly greater than ``min_match_score``.
    beam_width:
        Number of live partial matchings kept after each layer.
    expansion_width:
        Max number of candidate descendant pairs generated per live state.
        ``None`` disables this cap for the generated positive label-pair set;
        this can be useful for small exactness checks but is usually expensive.
    expansion_fn:
        Optional custom expansion rule ``N(x, y)``.  It receives a
        ``BeamExpansionContext`` and returns feasible pairs or scored triples.
    candidate_heuristic:
        Optional local heuristic for ranking candidate descendant pairs.  It
        receives a ``BeamCandidateContext``.
    priority_fn:
        Optional frontier priority ``rho(z)``.  It receives a
        ``BeamStateContext``.
    candidate_fn:
        Backward-compatible per-G-node candidate callback.  Prefer
        ``expansion_fn`` for new code.
    n_restarts:
        Number of seeded stochastic restarts.  The best returned score is kept.
    lookahead:
        If True, precompute discounted subtree label sketches and add their
        estimated future compatibility to the default candidate and frontier
        priorities.  This is deliberately heuristic, not an admissible A* bound.
    lookahead_weight:
        Weight of the precomputed lookahead score in the built-in priorities.
    lookahead_chunk_size:
        Length of fixed downward label shingles used by the path-chunk part of
        the lookahead sketch.
    lookahead_label_weight, lookahead_chunk_weight:
        Relative weights of the discounted descendant-label overlap and exact
        fixed-length path-shingle overlap before the result is capped by the
        remaining path-height estimate.
    prefer_match_on_tie:
        Accepted for API compatibility with the exact matcher; this search does
        not form DP cells, so there is no DP tie-break to apply.

    Returns
    -------
    AlignmentResult
        ``path_internal`` contains a valid sequence of matched internal node
        pairs; ``score`` is the accumulated match score.
    """
    del max_candidates_per_label  # retained for API compatibility; unused by this algorithm
    del prefer_match_on_tie       # retained for API compatibility; no DP tie-break here

    legacy_g_cap = max_candidates_per_u if max_candidates_per_u is not None else max(64, int(beam_width))
    params = _BeamParams(
        beam_width=int(beam_width),
        expansion_width=None if expansion_width is None else int(expansion_width),
        max_label_pair_scan=int(max_label_pair_scan),
        max_label_pairs_per_expansion=None
        if max_label_pairs_per_expansion is None
        else int(max_label_pairs_per_expansion),
        max_nodes_per_label_side=int(max_nodes_per_label_side),
        descendant_select_mode=str(candidate_select_mode),
        random_fraction=float(random_fraction),
        min_match_score=float(min_match_score),
        rarity_weight=float(rarity_weight),
        gap_penalty=float(gap_penalty),
        balance_penalty=float(balance_penalty),
        candidate_future_weight=float(candidate_future_weight),
        priority_future_weight=float(priority_future_weight),
        priority_length_weight=float(priority_length_weight),
        lookahead_enabled=bool(lookahead),
        lookahead_weight=float(lookahead_weight) if bool(lookahead) else 0.0,
        lookahead_sketch_size=int(lookahead_sketch_size),
        lookahead_depth_discount=float(lookahead_depth_discount),
        lookahead_chunk_size=int(lookahead_chunk_size),
        lookahead_label_weight=float(lookahead_label_weight),
        lookahead_chunk_weight=float(lookahead_chunk_weight),
        max_descendant_nodes_for_legacy_candidate_fn=int(legacy_g_cap),
    )
    _validate_beam_params(params)

    if n_restarts < 1:
        raise ValueError("n_restarts must be >= 1")

    best: Optional[AlignmentResult] = None
    for restart in range(int(n_restarts)):
        res = _align_trees_beam_once(
            G,
            H,
            w=w,
            params=params,
            candidate_fn=candidate_fn,
            expansion_fn=expansion_fn,
            candidate_heuristic=candidate_heuristic,
            priority_fn=priority_fn,
            match_predicate=match_predicate,
            seed=int(seed) + restart,
            max_length=max_length,
        )
        if best is None or res.score > best.score:
            best = res

    assert best is not None
    return best


def align_trees_beam_symmetric(
    G: TreeData,
    H: TreeData,
    *,
    w: Optional[WeightFn] = None,
    beam_width: int = 200,
    expansion_width: Optional[int] = 64,
    candidate_fn: Optional[CandidateFn] = None,
    expansion_fn: Optional[ExpansionFn] = None,
    candidate_heuristic: Optional[CandidateHeuristic] = None,
    priority_fn: Optional[PriorityFn] = None,
    max_candidates_per_label: Optional[int] = None,
    max_candidates_per_u: Optional[int] = None,
    candidate_select_mode: str = "mixed",
    seed: int = 0,
    n_restarts: int = 1,
    match_predicate: Optional[MatchPredicate] = None,
    prefer_match_on_tie: bool = True,
    min_match_score: float = 0.0,
    max_label_pair_scan: int = 100_000,
    max_label_pairs_per_expansion: Optional[int] = 2_048,
    max_nodes_per_label_side: int = 8,
    random_fraction: float = 0.10,
    rarity_weight: float = 0.25,
    gap_penalty: float = 0.03,
    balance_penalty: float = 0.01,
    candidate_future_weight: float = 0.03,
    priority_future_weight: float = 0.20,
    priority_length_weight: float = 0.0,
    lookahead: bool = False,
    lookahead_weight: float = 0.35,
    lookahead_sketch_size: int = 32,
    lookahead_depth_discount: float = 0.85,
    lookahead_chunk_size: int = 3,
    lookahead_label_weight: float = 1.0,
    lookahead_chunk_weight: float = 0.5,
    max_length: Optional[int] = None,
) -> AlignmentResult:
    """
    Run the partial-matching beam in both directions and keep the better score.

    The new beam formulation is already much less row-order-dependent than the
    previous row-wise beam.  This wrapper is retained for compatibility and for
    stochastic/asymmetric custom heuristics, but it roughly doubles work.
    """

    kwargs = dict(
        w=w,
        beam_width=beam_width,
        expansion_width=expansion_width,
        candidate_fn=candidate_fn,
        expansion_fn=expansion_fn,
        candidate_heuristic=candidate_heuristic,
        priority_fn=priority_fn,
        max_candidates_per_label=max_candidates_per_label,
        max_candidates_per_u=max_candidates_per_u,
        candidate_select_mode=candidate_select_mode,
        seed=seed,
        n_restarts=n_restarts,
        match_predicate=match_predicate,
        prefer_match_on_tie=prefer_match_on_tie,
        min_match_score=min_match_score,
        max_label_pair_scan=max_label_pair_scan,
        max_label_pairs_per_expansion=max_label_pairs_per_expansion,
        max_nodes_per_label_side=max_nodes_per_label_side,
        random_fraction=random_fraction,
        rarity_weight=rarity_weight,
        gap_penalty=gap_penalty,
        balance_penalty=balance_penalty,
        candidate_future_weight=candidate_future_weight,
        priority_future_weight=priority_future_weight,
        priority_length_weight=priority_length_weight,
        lookahead=lookahead,
        lookahead_weight=lookahead_weight,
        lookahead_sketch_size=lookahead_sketch_size,
        lookahead_depth_discount=lookahead_depth_discount,
        lookahead_chunk_size=lookahead_chunk_size,
        lookahead_label_weight=lookahead_label_weight,
        lookahead_chunk_weight=lookahead_chunk_weight,
        max_length=max_length,
    )
    res_fwd = align_trees_beam(G, H, **kwargs)

    # User-supplied expansion/candidate/heuristic functions are usually written
    # for the original orientation.  Do not silently apply them to swapped
    # trees; the single forward run above is the meaningful result.
    if expansion_fn is not None or candidate_fn is not None or candidate_heuristic is not None or priority_fn is not None:
        return res_fwd

    kwargs_rev = dict(kwargs)
    if w is not None:
        kwargs_rev["w"] = lambda label_h, label_g: w(label_g, label_h)
    if match_predicate is not None:
        kwargs_rev["match_predicate"] = lambda label_h, label_g: match_predicate(label_g, label_h)
    res_rev = align_trees_beam(H, G, **kwargs_rev)

    if res_rev.score > res_fwd.score:
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
