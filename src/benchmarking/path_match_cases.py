from __future__ import annotations

"""Synthetic tree-pair regimes for path-matching benchmarks.

The existing :mod:`path_matcher.planted_path_sampler` implements the planted-path
model used by the paper and returns ``igraph`` objects.  This module complements
that code with a small ``TreeData``-native generator designed for algorithmic
benchmarks.  It exposes the dimensions that matter most for the matching
implementations:

* narrow, medium-depth, wide, and path-shaped trees;
* tree--tree and tree--path comparisons;
* uniform and Zipf/heavy-tailed symbol frequencies;
* scalar labels and variable-size token sets;
* equality, max-overlap, and weighted-Jaccard scores.

The generator is deterministic given the regime seed and instance index.  It is
not intended to replace the paper's statistical model; it is a controlled
benchmark fixture whose ground-truth planted correspondence is known.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from path_matcher.bucketable_weight import BucketableWeight
from path_matcher.tree_data import TreeData


ScoreMode = str


def _as_sorted_token_tuple(label: Any) -> Tuple[str, ...]:
    if label is None:
        return ()
    if isinstance(label, str):
        return (label,)
    if isinstance(label, (tuple, list, set, frozenset, np.ndarray)):
        return tuple(sorted({str(x) for x in label}))
    return (str(label),)


@dataclass(frozen=True)
class ScoreModel:
    """A score model shared by generic, sparse, and specialized matchers."""

    mode: ScoreMode
    token_weights: Mapping[str, float] = field(default_factory=dict)
    default_weight: float = 1.0

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().strip()
        if mode not in {"equality", "overlap", "jaccard"}:
            raise ValueError("score mode must be 'equality', 'overlap', or 'jaccard'")
        object.__setattr__(self, "mode", mode)
        weights = {str(k): float(v) for k, v in dict(self.token_weights).items()}
        if any((not np.isfinite(v)) or v < 0.0 for v in weights.values()):
            raise ValueError("token weights must be finite and nonnegative")
        if (not np.isfinite(float(self.default_weight))) or float(self.default_weight) < 0.0:
            raise ValueError("default_weight must be finite and nonnegative")
        object.__setattr__(self, "token_weights", weights)
        object.__setattr__(self, "default_weight", float(self.default_weight))

    @property
    def fast_mode(self) -> Optional[str]:
        """Return the compatible specialized matcher mode, when one exists."""
        return self.mode if self.mode in {"equality", "overlap"} else None

    def weight_of(self, token: str) -> float:
        return float(self.token_weights.get(str(token), self.default_weight))

    def score(self, a: Any, b: Any) -> float:
        if self.mode == "equality":
            if a != b:
                return 0.0
            return self.weight_of(str(a))

        aa = set(_as_sorted_token_tuple(a))
        bb = set(_as_sorted_token_tuple(b))
        inter = aa.intersection(bb)
        if not inter:
            return 0.0
        if self.mode == "overlap":
            return max(self.weight_of(tok) for tok in inter)

        union = aa.union(bb)
        denom = sum(self.weight_of(tok) for tok in union)
        if denom <= 0.0:
            return 0.0
        numer = sum(self.weight_of(tok) for tok in inter)
        return float(numer / denom)

    def blocking_keys(self, label: Any) -> Iterable[str]:
        if self.mode == "equality":
            return (str(label),)
        return _as_sorted_token_tuple(label)

    def bucketable_weight(self) -> BucketableWeight:
        """Return an exact blocking wrapper for this score model."""
        return BucketableWeight(w=self.score, keyer=self.blocking_keys)

    def as_metadata(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "default_weight": self.default_weight,
            "token_weights": dict(self.token_weights),
        }


@dataclass(frozen=True)
class PairRegime:
    """Configuration for one family of synthetic matching instances."""

    name: str
    n_g: int
    n_h: int
    shape_g: str = "medium"
    shape_h: str = "medium"
    comparison: str = "tree_tree"
    depth_g: Optional[int] = None
    depth_h: Optional[int] = None

    alphabet_size: int = 20
    symbol_distribution: str = "uniform"
    zipf_exponent: float = 1.3
    symbols_per_node: Any = 1

    score_mode: str = "equality"
    weight_mode: str = "uniform"
    max_token_weight: float = 8.0
    reserved_token_weight: float = 4.0

    planted_length: int = 10
    planted_token_policy: str = "background"
    planted_distinct_tokens: Optional[int] = None
    planted_tail_fraction: float = 0.2
    p_obs_g: float = 0.9
    p_obs_h: float = 0.9
    planted_position_mode: str = "even"
    planted_path_selection: str = "random_leaf"

    n_instances: int = 1
    seed: int = 0
    expected_algorithm: Optional[str] = None
    notes: str = ""

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> "PairRegime":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        unknown = sorted(set(spec).difference(fields))
        if unknown:
            raise ValueError(f"Unknown PairRegime fields: {unknown}")
        return cls(**dict(spec))

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("regime name must be non-empty")
        if int(self.n_g) < 1 or int(self.n_h) < 1:
            raise ValueError("n_g and n_h must be positive")
        comparison = str(self.comparison).lower().strip()
        if comparison not in {"tree_tree", "tree_path"}:
            raise ValueError("comparison must be 'tree_tree' or 'tree_path'")
        object.__setattr__(self, "comparison", comparison)

        shape_g = str(self.shape_g).lower().strip()
        shape_h = str(self.shape_h).lower().strip()
        allowed_shapes = {"path", "narrow", "medium", "balanced", "wide"}
        if shape_g not in allowed_shapes or shape_h not in allowed_shapes:
            raise ValueError(f"tree shapes must be one of {sorted(allowed_shapes)}")
        if comparison == "tree_path":
            shape_h = "path"
        object.__setattr__(self, "shape_g", shape_g)
        object.__setattr__(self, "shape_h", shape_h)

        if int(self.alphabet_size) < 1:
            raise ValueError("alphabet_size must be positive")
        dist = str(self.symbol_distribution).lower().strip()
        if dist not in {"uniform", "zipf"}:
            raise ValueError("symbol_distribution must be 'uniform' or 'zipf'")
        object.__setattr__(self, "symbol_distribution", dist)
        if float(self.zipf_exponent) <= 0.0:
            raise ValueError("zipf_exponent must be positive")

        score_mode = str(self.score_mode).lower().strip()
        if score_mode not in {"equality", "overlap", "jaccard"}:
            raise ValueError("score_mode must be 'equality', 'overlap', or 'jaccard'")
        object.__setattr__(self, "score_mode", score_mode)
        if score_mode == "equality" and _max_symbols_per_node(self.symbols_per_node) != 1:
            raise ValueError("equality regimes require exactly one symbol per node")

        weight_mode = str(self.weight_mode).lower().strip()
        if weight_mode not in {"uniform", "inverse_sqrt"}:
            raise ValueError("weight_mode must be 'uniform' or 'inverse_sqrt'")
        object.__setattr__(self, "weight_mode", weight_mode)

        if int(self.planted_length) < 1:
            raise ValueError("planted_length must be positive")
        if int(self.planted_length) > min(int(self.n_g), int(self.n_h)):
            raise ValueError("planted_length cannot exceed either tree size")
        policy = str(self.planted_token_policy).lower().strip()
        if policy not in {"background", "tail", "reserved"}:
            raise ValueError("planted_token_policy must be 'background', 'tail', or 'reserved'")
        object.__setattr__(self, "planted_token_policy", policy)
        if not (0.0 < float(self.planted_tail_fraction) <= 1.0):
            raise ValueError("planted_tail_fraction must be in (0,1]")
        for name, value in (("p_obs_g", self.p_obs_g), ("p_obs_h", self.p_obs_h)):
            if not (0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must lie in [0,1]")
        position_mode = str(self.planted_position_mode).lower().strip()
        if position_mode not in {"even", "random"}:
            raise ValueError("planted_position_mode must be 'even' or 'random'")
        object.__setattr__(self, "planted_position_mode", position_mode)
        path_selection = str(self.planted_path_selection).lower().strip()
        if path_selection not in {"spine", "random_leaf", "random_deep_leaf"}:
            raise ValueError(
                "planted_path_selection must be 'spine', 'random_leaf', or 'random_deep_leaf'"
            )
        object.__setattr__(self, "planted_path_selection", path_selection)
        if int(self.n_instances) < 1:
            raise ValueError("n_instances must be positive")

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SyntheticPair:
    regime_name: str
    instance_index: int
    G: TreeData
    H: TreeData
    score_model: ScoreModel
    truth_pairs: Tuple[Tuple[int, int], ...]
    planted_nodes_g: Tuple[int, ...]
    planted_nodes_h: Tuple[int, ...]
    template_tokens: Tuple[str, ...]
    metadata: Mapping[str, Any]

    @property
    def truth_score(self) -> float:
        return float(sum(self.score_model.score(self.G.label[u], self.H.label[v]) for u, v in self.truth_pairs))


def _max_symbols_per_node(spec: Any) -> int:
    if isinstance(spec, (int, np.integer)):
        return int(spec)
    if not isinstance(spec, Mapping):
        raise TypeError("symbols_per_node must be an int or a mapping")
    kind = str(spec.get("kind", "fixed")).lower().strip()
    if kind == "fixed":
        return int(spec.get("value", 1))
    if kind == "categorical":
        values = [int(x) for x in spec.get("values", [1])]
        if not values:
            raise ValueError("categorical symbols_per_node needs non-empty values")
        return max(values)
    if kind == "poisson":
        return int(spec.get("max", max(1, int(np.ceil(float(spec.get("mean", 1.0)) * 3.0)))))
    raise ValueError("symbols_per_node kind must be fixed, categorical, or poisson")


def _sample_symbol_count(spec: Any, rng: np.random.Generator, alphabet_size: int) -> int:
    if isinstance(spec, (int, np.integer)):
        value = int(spec)
    else:
        cfg = dict(spec)
        kind = str(cfg.get("kind", "fixed")).lower().strip()
        if kind == "fixed":
            value = int(cfg.get("value", 1))
        elif kind == "categorical":
            values = np.asarray([int(x) for x in cfg.get("values", [1])], dtype=int)
            probs = np.asarray(cfg.get("probs", np.ones(len(values))), dtype=float)
            if values.size == 0 or probs.shape != values.shape:
                raise ValueError("categorical symbols_per_node needs aligned values/probs")
            if np.any(values < 1) or np.any(probs < 0.0) or float(probs.sum()) <= 0.0:
                raise ValueError("invalid categorical symbols_per_node")
            probs = probs / probs.sum()
            value = int(rng.choice(values, p=probs))
        elif kind == "poisson":
            mean = float(cfg.get("mean", 1.0))
            lo = int(cfg.get("min", 1))
            hi = int(cfg.get("max", max(lo, int(np.ceil(mean * 3.0)))))
            if mean < 0.0 or lo < 1 or hi < lo:
                raise ValueError("invalid poisson symbols_per_node")
            value = int(np.clip(rng.poisson(mean), lo, hi))
        else:
            raise ValueError("symbols_per_node kind must be fixed, categorical, or poisson")
    if value < 1:
        raise ValueError("each node must have at least one symbol")
    return min(value, int(alphabet_size))


def _background_distribution(regime: PairRegime) -> Tuple[List[str], np.ndarray]:
    alphabet = [f"s{i}" for i in range(int(regime.alphabet_size))]
    if regime.symbol_distribution == "uniform":
        probs = np.ones(len(alphabet), dtype=float)
    else:
        ranks = np.arange(1, len(alphabet) + 1, dtype=float)
        probs = np.power(ranks, -float(regime.zipf_exponent))
    probs /= probs.sum()
    # ``Generator.choice(..., replace=False, p=...)`` is unusually strict
    # about accumulated floating-point normalization for large alphabets.
    # Make the final entry the exact residual and avoid passing ``p`` at all
    # for the uniform case below.
    if len(probs) > 1:
        probs[-1] = max(0.0, 1.0 - float(probs[:-1].sum()))
        probs /= probs.sum()
    return alphabet, probs


def _token_weights(
    regime: PairRegime,
    background_alphabet: Sequence[str],
    background_probs: np.ndarray,
    reserved_tokens: Sequence[str],
) -> Dict[str, float]:
    if regime.weight_mode == "uniform":
        out = {str(tok): 1.0 for tok in background_alphabet}
    else:
        pmax = float(np.max(background_probs))
        out = {
            str(tok): min(
                float(regime.max_token_weight),
                float(np.sqrt(pmax / max(float(p), 1e-15))),
            )
            for tok, p in zip(background_alphabet, background_probs)
        }
    for tok in reserved_tokens:
        out[str(tok)] = float(regime.reserved_token_weight)
    return out


def _default_depth(n: int, shape: str, planted_length: int) -> int:
    if shape == "path":
        return n - 1
    minimum = max(0, planted_length - 1)
    if shape == "narrow":
        proposal = max(minimum, int(round(2.0 * np.sqrt(n))), 12)
    elif shape in {"medium", "balanced"}:
        proposal = max(minimum, int(round(2.0 * np.log2(max(n, 2)))), 8)
    else:  # wide
        proposal = max(minimum, 6)
    return min(n - 1, proposal)


def _allocate_level_sizes(
    n: int,
    depth: int,
    shape: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if depth < 0 or depth >= n:
        raise ValueError("depth must satisfy 0 <= depth < n")
    if depth == 0 and n != 1:
        raise ValueError("a tree with more than one node must have depth at least 1")
    if shape == "path":
        if depth != n - 1:
            raise ValueError("path shape requires depth=n-1")
        return np.ones(n, dtype=int)

    sizes = np.ones(depth + 1, dtype=int)
    extra = n - (depth + 1)
    if extra <= 0:
        return sizes

    d = np.arange(1, depth + 1, dtype=float)
    if shape == "narrow":
        weights = np.ones(depth, dtype=float)
    elif shape in {"medium", "balanced"}:
        # Gradual growth towards the leaves without producing an extreme star.
        weights = np.exp(np.linspace(0.0, np.log(8.0), depth))
    else:  # wide
        weights = np.full(depth, 0.05, dtype=float)
        weights[0] = 1.0
        if depth >= 2:
            weights[1] = 0.45
        if depth >= 3:
            weights[2] = 0.20
    weights = weights / weights.sum()
    sizes[1:] += rng.multinomial(extra, weights)
    return sizes


def _randomize_node_order_within_levels(
    parent: np.ndarray,
    levels: Sequence[Sequence[int]],
    spine: Sequence[int],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Randomize sibling order while preserving parent-before-child indexing.

    The local beam's capped expansion uses child order.  Synthetic structure
    generation must therefore avoid tying a structurally special branch to low
    node indices.  We independently permute the node ids inside every depth
    level, then remap the parent array and the guaranteed spine.  Since parents
    remain in earlier levels, the resulting ``TreeData`` ordering is still
    topological.
    """

    n = int(parent.size)
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[0] = 0
    next_id = 1
    for level_nodes in levels[1:]:
        shuffled = rng.permutation(np.asarray(level_nodes, dtype=np.int64))
        for old_raw in shuffled:
            old_to_new[int(old_raw)] = next_id
            next_id += 1
    if next_id != n:
        raise RuntimeError("internal node-order randomization error")

    remapped_parent = np.full(n, -1, dtype=np.int64)
    for old in range(1, n):
        remapped_parent[int(old_to_new[old])] = int(old_to_new[int(parent[old])])
    remapped_spine = tuple(int(old_to_new[int(node)]) for node in spine)
    return remapped_parent, remapped_spine


def _build_tree(
    *,
    n: int,
    shape: str,
    depth: Optional[int],
    planted_length: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Tuple[int, ...], Dict[str, Any]]:
    if depth is None:
        depth = _default_depth(n, shape, planted_length)
    depth = int(depth)
    if shape == "path":
        depth = n - 1
    depth = max(depth, planted_length - 1)
    if depth >= n:
        raise ValueError(f"requested depth {depth} is not possible for n={n}")

    level_sizes = _allocate_level_sizes(n, depth, shape, rng)
    parent = np.full(n, -1, dtype=np.int64)
    levels: List[List[int]] = [[0]]
    next_node = 1
    spine = [0]
    child_counts = np.zeros(n, dtype=np.int64)

    for level in range(1, depth + 1):
        count = int(level_sizes[level])
        nodes = list(range(next_node, next_node + count))
        next_node += count
        previous = levels[level - 1]

        # The first node at every level forms a guaranteed root-to-depth spine.
        parent[nodes[0]] = spine[-1]
        child_counts[spine[-1]] += 1
        spine.append(nodes[0])

        for j, node in enumerate(nodes[1:], start=1):
            if shape in {"medium", "balanced"}:
                # Prefer currently under-used parents; random jitter prevents a
                # completely deterministic complete k-ary tree.
                counts = child_counts[np.asarray(previous, dtype=int)].astype(float)
                priorities = 1.0 / (1.0 + counts)
                priorities *= rng.uniform(0.9, 1.1, size=len(previous))
                priorities /= priorities.sum()
                p = int(rng.choice(previous, p=priorities))
            elif shape == "wide":
                p = int(previous[j % len(previous)])
            else:  # narrow: distribute small side branches along the prior level
                p = int(previous[j % len(previous)])
            parent[node] = p
            child_counts[p] += 1
        levels.append(nodes)

    if next_node != n:
        raise RuntimeError("internal level allocation error")

    # Randomize ids inside each depth level.  This preserves the generated
    # shape but prevents child-index-based algorithms from receiving an
    # accidental advantage from the construction order.
    parent, spine_randomized = _randomize_node_order_within_levels(
        parent,
        levels,
        spine,
        rng,
    )

    metadata = {
        "n": n,
        "depth": depth,
        "max_width": int(max(level_sizes)),
        "level_sizes": [int(x) for x in level_sizes],
        "max_out_degree": int(child_counts.max(initial=0)),
        "mean_nonleaf_out_degree": float(
            child_counts[child_counts > 0].mean() if np.any(child_counts > 0) else 0.0
        ),
        "node_order_randomized_within_levels": True,
    }
    return parent, spine_randomized, metadata


def _sample_background_labels(
    *,
    n: int,
    alphabet: Sequence[str],
    probs: np.ndarray,
    score_mode: str,
    symbols_per_node: Any,
    rng: np.random.Generator,
) -> List[Any]:
    labels: List[Any] = []
    choice_probs = None if np.allclose(probs, probs[0], rtol=1e-12, atol=1e-15) else probs
    for _ in range(n):
        k = 1 if score_mode == "equality" else _sample_symbol_count(symbols_per_node, rng, len(alphabet))
        chosen = rng.choice(len(alphabet), size=k, replace=False, p=choice_probs)
        toks = tuple(sorted(str(alphabet[int(i)]) for i in chosen))
        labels.append(toks[0] if score_mode == "equality" else toks)
    return labels


def _choose_template_tokens(
    regime: PairRegime,
    alphabet: Sequence[str],
    probs: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[List[str], List[str]]:
    L = int(regime.planted_length)
    distinct = regime.planted_distinct_tokens
    if distinct is None:
        distinct = min(L, max(2, int(round(np.sqrt(L)))))
    distinct = max(1, min(int(distinct), L))

    reserved: List[str] = []
    if regime.planted_token_policy == "reserved":
        reserved = [f"plant{i}" for i in range(distinct)]
        pool = reserved
        pool_probs = np.ones(len(pool), dtype=float) / len(pool)
    elif regime.planted_token_policy == "tail":
        tail_n = max(1, int(np.ceil(len(alphabet) * float(regime.planted_tail_fraction))))
        pool = list(alphabet[-tail_n:])
        pool_probs = np.asarray(probs[-tail_n:], dtype=float).copy()
        pool_probs /= pool_probs.sum()
    else:
        pool = list(alphabet)
        pool_probs = np.asarray(probs, dtype=float)

    choice_probs = None if np.allclose(pool_probs, pool_probs[0], rtol=1e-12, atol=1e-15) else pool_probs
    if len(pool) >= distinct:
        base = [str(x) for x in rng.choice(pool, size=distinct, replace=False, p=choice_probs)]
    else:
        base = [str(x) for x in rng.choice(pool, size=distinct, replace=True, p=choice_probs)]

    # Repeat the selected vocabulary in a shuffled order.  Repetition makes the
    # ordering problem nontrivial while preserving controlled symbol rarity.
    seq = [base[i % len(base)] for i in range(L)]
    rng.shuffle(seq)
    return seq, reserved


def _select_planted_path(
    parent: np.ndarray,
    fallback_spine: Sequence[int],
    minimum_length: int,
    mode: str,
    rng: np.random.Generator,
) -> Tuple[int, ...]:
    """Select a root-to-leaf path without privileging child index order."""

    if mode == "spine":
        return tuple(int(x) for x in fallback_spine)

    n = int(parent.size)
    child_count = np.zeros(n, dtype=np.int64)
    depth = np.zeros(n, dtype=np.int64)
    for node in range(1, n):
        p = int(parent[node])
        child_count[p] += 1
        depth[node] = depth[p] + 1
    eligible = [
        node for node in range(n)
        if child_count[node] == 0 and int(depth[node]) + 1 >= int(minimum_length)
    ]
    if not eligible:
        return tuple(int(x) for x in fallback_spine)
    if mode == "random_deep_leaf":
        deepest = max(int(depth[node]) for node in eligible)
        eligible = [node for node in eligible if int(depth[node]) == deepest]
    leaf = int(rng.choice(np.asarray(eligible, dtype=np.int64)))
    path: List[int] = []
    node = leaf
    while node >= 0:
        path.append(node)
        node = int(parent[node])
    path.reverse()
    return tuple(path)


def _choose_spine_positions(
    spine: Sequence[int],
    length: int,
    mode: str,
    rng: np.random.Generator,
) -> List[int]:
    if length > len(spine):
        raise ValueError("planted path length exceeds available spine")
    if length == len(spine):
        idx = np.arange(len(spine), dtype=int)
    elif mode == "random":
        idx = np.sort(rng.choice(len(spine), size=length, replace=False))
    else:
        # Integer rounding can repeat positions, so greedily enforce strictness.
        raw = np.linspace(0, len(spine) - 1, num=length)
        idx = np.rint(raw).astype(int)
        for i in range(1, len(idx)):
            idx[i] = max(idx[i], idx[i - 1] + 1)
        for i in range(len(idx) - 2, -1, -1):
            idx[i] = min(idx[i], idx[i + 1] - 1)
    return [int(spine[int(i)]) for i in idx]


def _insert_planted_token(label: Any, token: str, score_mode: str) -> Any:
    if score_mode == "equality":
        return str(token)
    toks = set(_as_sorted_token_tuple(label))
    toks.add(str(token))
    return tuple(sorted(toks))


def _tree_metrics(parent: np.ndarray) -> Dict[str, Any]:
    n = int(parent.size)
    depth = np.zeros(n, dtype=int)
    children = np.zeros(n, dtype=int)
    widths: Dict[int, int] = {}
    for i in range(n):
        if i > 0:
            p = int(parent[i])
            depth[i] = depth[p] + 1
            children[p] += 1
        widths[int(depth[i])] = widths.get(int(depth[i]), 0) + 1
    return {
        "n": n,
        "depth": int(depth.max(initial=0)),
        "max_width": int(max(widths.values(), default=0)),
        "max_out_degree": int(children.max(initial=0)),
        "mean_nonleaf_out_degree": float(children[children > 0].mean() if np.any(children > 0) else 0.0),
    }


def estimate_candidate_work(pair: SyntheticPair) -> Dict[str, int]:
    """Cheap bucket-count estimates for positive-pair discovery.

    ``posting_hits`` is exact for equality and an upper bound before
    deduplication for overlap/Jaccard labels.
    """

    def postings(labels: Sequence[Any], mode: str) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for label in labels:
            keys = (str(label),) if mode == "equality" else _as_sorted_token_tuple(label)
            for key in set(keys):
                out[key] = out.get(key, 0) + 1
        return out

    g = postings(pair.G.label, pair.score_model.mode)
    h = postings(pair.H.label, pair.score_model.mode)
    hits = sum(int(c) * int(h.get(k, 0)) for k, c in g.items())
    return {
        "dense_cells": int(pair.G.n * pair.H.n),
        "posting_hits": int(hits),
        "shared_keys": int(sum(1 for k in g if k in h)),
    }


def generate_synthetic_pair(regime: PairRegime, instance_index: int = 0) -> SyntheticPair:
    """Generate one deterministic synthetic tree pair for ``regime``."""

    seed_seq = np.random.SeedSequence([int(regime.seed), int(instance_index)])
    rng = np.random.default_rng(seed_seq)

    parent_g, spine_g, structure_g = _build_tree(
        n=int(regime.n_g),
        shape=regime.shape_g,
        depth=regime.depth_g,
        planted_length=int(regime.planted_length),
        rng=rng,
    )
    parent_h, spine_h, structure_h = _build_tree(
        n=int(regime.n_h),
        shape=regime.shape_h,
        depth=regime.depth_h,
        planted_length=int(regime.planted_length),
        rng=rng,
    )
    candidate_path_g = _select_planted_path(
        parent_g, spine_g, int(regime.planted_length), regime.planted_path_selection, rng
    )
    candidate_path_h = _select_planted_path(
        parent_h, spine_h, int(regime.planted_length), regime.planted_path_selection, rng
    )

    alphabet, probs = _background_distribution(regime)
    template_tokens, reserved_tokens = _choose_template_tokens(regime, alphabet, probs, rng)
    token_weights = _token_weights(regime, alphabet, probs, reserved_tokens)
    score_model = ScoreModel(
        mode=regime.score_mode,
        token_weights=token_weights,
        default_weight=1.0,
    )

    labels_g = _sample_background_labels(
        n=int(regime.n_g),
        alphabet=alphabet,
        probs=probs,
        score_mode=regime.score_mode,
        symbols_per_node=regime.symbols_per_node,
        rng=rng,
    )
    labels_h = _sample_background_labels(
        n=int(regime.n_h),
        alphabet=alphabet,
        probs=probs,
        score_mode=regime.score_mode,
        symbols_per_node=regime.symbols_per_node,
        rng=rng,
    )

    plant_g = _choose_spine_positions(
        candidate_path_g,
        int(regime.planted_length),
        regime.planted_position_mode,
        rng,
    )
    plant_h = _choose_spine_positions(
        candidate_path_h,
        int(regime.planted_length),
        regime.planted_position_mode,
        rng,
    )
    obs_g = rng.random(int(regime.planted_length)) < float(regime.p_obs_g)
    obs_h = rng.random(int(regime.planted_length)) < float(regime.p_obs_h)

    observed_nodes_g: List[int] = []
    observed_nodes_h: List[int] = []
    truth_pairs: List[Tuple[int, int]] = []
    for i, tok in enumerate(template_tokens):
        ug = int(plant_g[i])
        vh = int(plant_h[i])
        if bool(obs_g[i]):
            labels_g[ug] = _insert_planted_token(labels_g[ug], tok, regime.score_mode)
            observed_nodes_g.append(ug)
        if bool(obs_h[i]):
            labels_h[vh] = _insert_planted_token(labels_h[vh], tok, regime.score_mode)
            observed_nodes_h.append(vh)
        if bool(obs_g[i]) and bool(obs_h[i]):
            truth_pairs.append((ug, vh))

    G = TreeData(
        parent=parent_g,
        label=tuple(labels_g),
        orig_index=np.arange(int(regime.n_g), dtype=np.int64),
    )
    H = TreeData(
        parent=parent_h,
        label=tuple(labels_h),
        orig_index=np.arange(int(regime.n_h), dtype=np.int64),
    )

    metadata: Dict[str, Any] = {
        "regime": regime.as_dict(),
        "instance_seed_entropy": [int(regime.seed), int(instance_index)],
        "tree_g": {**structure_g, **_tree_metrics(parent_g)},
        "tree_h": {**structure_h, **_tree_metrics(parent_h)},
        "background_alphabet": list(alphabet),
        "background_probabilities": [float(x) for x in probs],
        "reserved_tokens": list(reserved_tokens),
        "candidate_planted_path_g": [int(x) for x in candidate_path_g],
        "candidate_planted_path_h": [int(x) for x in candidate_path_h],
        "observed_template_positions_g": [int(i) for i, flag in enumerate(obs_g) if flag],
        "observed_template_positions_h": [int(i) for i, flag in enumerate(obs_h) if flag],
    }

    pair = SyntheticPair(
        regime_name=regime.name,
        instance_index=int(instance_index),
        G=G,
        H=H,
        score_model=score_model,
        truth_pairs=tuple(truth_pairs),
        planted_nodes_g=tuple(observed_nodes_g),
        planted_nodes_h=tuple(observed_nodes_h),
        template_tokens=tuple(template_tokens),
        metadata=metadata,
    )
    metadata.update(estimate_candidate_work(pair))
    metadata["truth_score"] = pair.truth_score
    return pair


__all__ = [
    "PairRegime",
    "ScoreModel",
    "SyntheticPair",
    "estimate_candidate_work",
    "generate_synthetic_pair",
]
