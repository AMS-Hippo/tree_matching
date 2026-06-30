
from __future__ import annotations

"""
tree_cluster.py

Embedding / clustering utilities for collections of trees using the current
matching code.

This module is designed to play the role that ``gw_clustering.py`` played in
the older codebase, but with a more modular API and with a few scalable
alternatives to the full pairwise-similarity workflow.

Main ideas
----------
Given a collection of trees, we can:

1. score tree pairs with the current matcher stack,
2. normalize those scores,
3. feed the resulting similarities or feature vectors into UMAP,
4. cluster the UMAP embedding with HDBSCAN.

Three scoring regimes are supported:

- ``similarity_mode='full'``:
    compute the full pairwise similarity matrix.
- ``similarity_mode='landmarks'``:
    score every tree only against a subset of landmarks, then use the resulting
    tree-to-landmark score vectors as features for UMAP.
- ``similarity_mode='approx_knn'``:
    build a cheap sketch for each tree, find approximate candidate neighbours in
    sketch space, then compute exact matcher scores only on those candidate
    pairs. Missing similarities are left at 0.

The class-based API is intentionally similar in spirit to
``path_matcher.fast_match.FastTreePathMatcher``:

>>> emb = TreeClusterEmbedder(similarity_mode="full", matcher_kind="fast")
>>> emb.fit(some_trees)
>>> result = emb.predict()

There are also compatibility wrappers such as ``embed_and_cluster_sns(...)`` so
older notebooks can be ported with fairly small edits.

Notes
-----
- ``umap-learn`` and ``hdbscan`` are optional runtime dependencies. The module
  imports even if they are missing, but calling ``fit(...)`` will raise a clear
  error when those backends are required.
- ``pynndescent`` is optional. If available, it is used for approximate sketch
  kNN search; otherwise the code falls back to ``sklearn.neighbors``.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import math
import random
from collections import Counter, defaultdict

import numpy as np
from scipy import sparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

try:
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover
    hdbscan = None

try:
    import pynndescent  # type: ignore
except Exception:  # pragma: no cover
    pynndescent = None


ScoreLike = Union[float, int, np.floating, np.integer]
PairScore = Tuple[List[Tuple[int, int]], float]
NormalizerLike = Union[str, Callable[[np.ndarray], np.ndarray], None]

EPS = 1e-12


# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------

def _require_umap() -> Any:
    if umap is None:
        raise ImportError(
            "UMAP is required for TreeClusterEmbedder.fit(...), but `umap-learn` "
            "could not be imported in this environment."
        )
    return umap


def _require_hdbscan() -> Any:
    if hdbscan is None:
        raise ImportError(
            "HDBSCAN is required for TreeClusterEmbedder.fit(...), but `hdbscan` "
            "could not be imported in this environment."
        )
    return hdbscan


# ---------------------------------------------------------------------------
# Tree coercion / label extraction
# ---------------------------------------------------------------------------

def _looks_like_treedata(x: Any) -> bool:
    return hasattr(x, "parent") and hasattr(x, "label") and hasattr(x, "orig_index")


def _as_treedata(
    G: Any,
    *,
    phi_name: str,
    order: str,
    ts_field: Optional[str],
    strict_tree: bool,
) -> Any:
    if _looks_like_treedata(G):
        return G
    from path_matcher.igraph_io import igraph_to_treedata

    return igraph_to_treedata(
        G,
        phi_name=phi_name,
        order=order,
        ts_field=ts_field,
        strict_tree=strict_tree,
    )


def _default_extract(label: Any) -> Any:
    if isinstance(label, dict):
        for key in ("Labels", "labels", "label", "tokens", "token_set"):
            if key in label:
                return label[key]
    return label


def _is_scalar_token(x: Any) -> bool:
    return isinstance(x, (str, bytes, int, float, np.integer, np.floating))


def _extract_raw(label: Any, label_getter: Optional[Callable[[Any], Any]]) -> Any:
    return label_getter(label) if label_getter is not None else _default_extract(label)


def _normalize_tokens(label: Any, label_getter: Optional[Callable[[Any], Any]] = None) -> Tuple[Hashable, ...]:
    raw = _extract_raw(label, label_getter)
    if raw is None:
        return ()
    if _is_scalar_token(raw):
        toks = [raw]
    elif isinstance(raw, np.ndarray):
        toks = raw.tolist()
    else:
        try:
            toks = list(raw)
        except TypeError:
            toks = [raw]
    out: Dict[Hashable, None] = {}
    for tok in toks:
        hash(tok)
        out[tok] = None
    return tuple(out.keys())


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class TreeEmbeddingResult:
    tree_list: List[Any]
    tree_data_list: List[Any]
    gt_labels: Optional[np.ndarray]
    pred_labels: np.ndarray
    embedding: np.ndarray
    similarity_mode: str
    raw_scores: Optional[np.ndarray] = None
    normalized_scores: Optional[np.ndarray] = None
    distances: Optional[np.ndarray] = None
    features: Optional[np.ndarray] = None
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = field(default_factory=dict)
    self_scores: Optional[np.ndarray] = None
    landmark_indices: Optional[np.ndarray] = None
    candidate_pairs: Optional[List[Tuple[int, int]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_trees(self) -> int:
        return len(self.tree_list)

    def clustering_scores(self) -> Dict[str, float]:
        if self.gt_labels is None:
            return {"ari": math.nan, "nmi": math.nan}
        gt = np.asarray(self.gt_labels)
        pred = np.asarray(self.pred_labels)
        return {
            "ari": float(adjusted_rand_score(gt, pred)),
            "nmi": float(normalized_mutual_info_score(gt, pred)),
        }


# ---------------------------------------------------------------------------
# Input flattening / compatibility with old nested structure
# ---------------------------------------------------------------------------

def expand_tree_list(some_trees: Any) -> Tuple[List[Any], Optional[List[int]]]:
    """
    Flatten a few common dataset formats.

    Supported inputs
    ----------------
    1. Old-style nested structure:
         [{"trees": [[graph, Gamma], ...], ...}, ...]
       In that case the class labels are the outer-list indices.

    2. Tuple ``(tree_list, labels)``.

    3. Plain list of trees / graphs / TreeData objects.
       In this case class labels are returned as ``None``.
    """
    if isinstance(some_trees, tuple) and len(some_trees) == 2:
        tree_list = list(some_trees[0])
        labels = list(some_trees[1]) if some_trees[1] is not None else None
        return tree_list, labels

    if isinstance(some_trees, list) and some_trees and all(isinstance(x, dict) and "trees" in x for x in some_trees):
        tree_list: List[Any] = []
        class_list: List[int] = []
        for class_id, block in enumerate(some_trees):
            for item in block.get("trees", []):
                if isinstance(item, (list, tuple)) and len(item) >= 1:
                    tree = item[0]
                else:
                    tree = item
                tree_list.append(tree)
                class_list.append(class_id)
        return tree_list, class_list

    if isinstance(some_trees, (list, tuple)):
        return list(some_trees), None

    raise TypeError(
        "Unsupported dataset format. Expected either a plain tree list, "
        "(tree_list, labels), or the older nested `some_trees` structure."
    )


# ---------------------------------------------------------------------------
# Score normalizers
# ---------------------------------------------------------------------------

def _rowmax_scales(sims: np.ndarray, eps: float = EPS) -> np.ndarray:
    scales = np.max(np.asarray(sims, dtype=float), axis=1)
    return np.maximum(scales, eps)


def cos_normalize(sims: np.ndarray, eps: float = EPS) -> np.ndarray:
    sims = np.asarray(sims, dtype=float)
    scales = _rowmax_scales(sims, eps=eps)
    denom = np.sqrt(np.outer(scales, scales))
    return sims / np.maximum(denom, eps)


def jacc_normalize(sims: np.ndarray, eps: float = EPS) -> np.ndarray:
    sims = np.asarray(sims, dtype=float)
    scales = _rowmax_scales(sims, eps=eps)
    denom = np.add.outer(scales, scales) - sims
    return sims / np.maximum(denom, eps)


def dice_normalize(sims: np.ndarray, eps: float = EPS) -> np.ndarray:
    sims = np.asarray(sims, dtype=float)
    scales = _rowmax_scales(sims, eps=eps)
    denom = np.add.outer(scales, scales)
    return 2.0 * sims / np.maximum(denom, eps)


def overlap_normalize(sims: np.ndarray, eps: float = EPS) -> np.ndarray:
    sims = np.asarray(sims, dtype=float)
    scales = _rowmax_scales(sims, eps=eps)
    denom = np.minimum.outer(scales, scales)
    return sims / np.maximum(denom, eps)


def _normalize_from_scales(
    scores: np.ndarray,
    row_scales: np.ndarray,
    col_scales: np.ndarray,
    *,
    mode: str,
    eps: float = EPS,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    row_scales = np.maximum(np.asarray(row_scales, dtype=float), eps)
    col_scales = np.maximum(np.asarray(col_scales, dtype=float), eps)

    if mode in {"raw", "none"}:
        return scores.copy()
    if mode in {"cos", "cosine", "self_cos", "self_cosine"}:
        denom = np.sqrt(np.outer(row_scales, col_scales))
        return scores / np.maximum(denom, eps)
    if mode in {"jacc", "jaccard", "self_jacc", "self_jaccard"}:
        denom = np.add.outer(row_scales, col_scales) - scores
        return scores / np.maximum(denom, eps)
    if mode in {"dice", "self_dice"}:
        denom = np.add.outer(row_scales, col_scales)
        return 2.0 * scores / np.maximum(denom, eps)
    if mode in {"overlap", "self_overlap"}:
        denom = np.minimum.outer(row_scales, col_scales)
        return scores / np.maximum(denom, eps)
    raise ValueError(f"Unknown scale-based normalization mode: {mode!r}")


def normalize_similarity(
    scores: np.ndarray,
    *,
    normalizer: NormalizerLike = "jaccard",
    row_scales: Optional[np.ndarray] = None,
    col_scales: Optional[np.ndarray] = None,
    diag_source: str = "row_max",
    eps: float = EPS,
) -> np.ndarray:
    """
    Normalize a similarity matrix or a rectangular score table.

    ``diag_source='row_max'`` reproduces the older behaviour for square matrices.
    If explicit ``row_scales``/``col_scales`` are supplied, those are used instead.
    """
    scores = np.asarray(scores, dtype=float)
    if callable(normalizer):
        if row_scales is not None or col_scales is not None or scores.shape[0] != scores.shape[1]:
            raise ValueError("Callable normalizers are only supported for square matrices without explicit scales.")
        return np.asarray(normalizer(scores), dtype=float)

    if normalizer is None:
        return scores.copy()

    mode = str(normalizer).lower().strip()

    if row_scales is not None or col_scales is not None:
        if row_scales is None or col_scales is None:
            raise ValueError("Provide both row_scales and col_scales, or neither.")
        return _normalize_from_scales(scores, row_scales, col_scales, mode=mode, eps=eps)

    if scores.shape[0] != scores.shape[1]:
        raise ValueError("Rectangular score tables need explicit row_scales and col_scales.")

    if mode in {"raw", "none"}:
        return scores.copy()

    if diag_source == "row_max":
        if mode in {"cos", "cosine"}:
            return cos_normalize(scores, eps=eps)
        if mode in {"jacc", "jaccard"}:
            return jacc_normalize(scores, eps=eps)
        if mode == "dice":
            return dice_normalize(scores, eps=eps)
        if mode == "overlap":
            return overlap_normalize(scores, eps=eps)
        raise ValueError(f"Unknown row-max normalization mode: {mode!r}")

    if diag_source == "self":
        diag = np.diag(scores).copy()
        return _normalize_from_scales(scores, diag, diag, mode=mode, eps=eps)

    raise ValueError("diag_source must be 'row_max' or 'self'")


def to_diff(M: np.ndarray) -> np.ndarray:
    M = np.asarray(M, dtype=float)
    return np.sqrt(np.maximum(0.0, 1.0 - M))


# ---------------------------------------------------------------------------
# Cheap tree sketches for candidate search / landmarks
# ---------------------------------------------------------------------------

def build_tree_token_features(
    tree_data_list: Sequence[Any],
    *,
    label_getter: Optional[Callable[[Any], Any]] = None,
    hash_dim: Optional[int] = 256,
    binary: bool = False,
    l2_normalize: bool = True,
) -> np.ndarray:
    """
    Build a cheap sketch vector for each tree using token counts over node labels.

    These sketches are not the core method; they are just a lightweight proposal
    mechanism for landmark selection and approximate kNN candidate generation.
    """
    n = len(tree_data_list)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    if hash_dim is None:
        vocab: Dict[Hashable, int] = {}
        rows: List[List[int]] = []
        vals: List[List[float]] = []
        for tree in tree_data_list:
            counter: Dict[int, float] = {}
            for lab in tree.label:
                toks = _normalize_tokens(lab, label_getter=label_getter)
                for tok in toks:
                    idx = vocab.setdefault(tok, len(vocab))
                    counter[idx] = 1.0 if binary else counter.get(idx, 0.0) + 1.0
            rows.append(list(counter.keys()))
            vals.append(list(counter.values()))
        out = np.zeros((n, len(vocab)), dtype=np.float32)
        for i, (idxs, ws) in enumerate(zip(rows, vals)):
            if idxs:
                out[i, np.asarray(idxs, dtype=int)] = np.asarray(ws, dtype=np.float32)
    else:
        d = int(hash_dim)
        out = np.zeros((n, d), dtype=np.float32)
        for i, tree in enumerate(tree_data_list):
            for lab in tree.label:
                toks = _normalize_tokens(lab, label_getter=label_getter)
                for tok in toks:
                    h = hash(tok) % d
                    if binary:
                        out[i, h] = 1.0
                    else:
                        out[i, h] += 1.0

    if l2_normalize and out.size > 0:
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out = out / norms
    return out


def select_landmarks(
    features: np.ndarray,
    n_landmarks: int,
    *,
    mode: str = "random",
    random_state: int = 0,
) -> np.ndarray:
    n = int(features.shape[0])
    if n_landmarks <= 0:
        raise ValueError("n_landmarks must be positive")
    n_landmarks = min(int(n_landmarks), n)

    rng = np.random.default_rng(random_state)
    if mode == "random":
        idx = rng.choice(n, size=n_landmarks, replace=False)
        return np.sort(idx.astype(int))

    if mode in {"spread", "farthest"}:
        chosen = [int(rng.integers(0, n))]
        if n_landmarks == 1:
            return np.asarray(chosen, dtype=int)
        sq = np.sum(features * features, axis=1)
        min_d2 = np.full(n, np.inf, dtype=float)
        for _ in range(1, n_landmarks):
            c = chosen[-1]
            d2 = sq + sq[c] - 2.0 * features.dot(features[c])
            min_d2 = np.minimum(min_d2, d2)
            next_idx = int(np.argmax(min_d2))
            chosen.append(next_idx)
        return np.asarray(sorted(set(chosen)), dtype=int)

    raise ValueError("landmark mode must be 'random' or 'spread'")


def candidate_pairs_from_features(
    features: np.ndarray,
    *,
    top_k: int = 10,
    metric: str = "cosine",
    backend: str = "auto",
    random_state: int = 0,
    rule: str = "union",
    backend_kwargs: Optional[Mapping[str, Any]] = None,
) -> List[Tuple[int, int]]:
    n = int(features.shape[0])
    if n <= 1:
        return []
    n_nbrs = min(max(2, int(top_k) + 1), n)

    if backend == "auto":
        backend = "pynndescent" if pynndescent is not None else "sklearn"
    bk = dict(backend_kwargs or {})

    if backend == "pynndescent":
        if pynndescent is None:
            raise ImportError("backend='pynndescent' was requested but the package is not installed.")
        index = pynndescent.NNDescent(features, n_neighbors=n_nbrs, metric=metric, random_state=random_state, **bk)
        nbr_idx, _nbr_dist = index.neighbor_graph
    elif backend == "sklearn":
        nn = NearestNeighbors(n_neighbors=n_nbrs, metric=metric, **bk)
        nn.fit(features)
        nbr_idx = nn.kneighbors(features, return_distance=False)
    else:
        raise ValueError("backend must be 'auto', 'pynndescent', or 'sklearn'")

    rule_mode = str(rule).lower().strip()
    if rule_mode not in {"union", "mutual"}:
        raise ValueError("rule must be 'union' or 'mutual'")

    nbr_sets: List[set[int]] = []
    for i in range(n):
        cur = {int(j) for j in np.asarray(nbr_idx[i], dtype=int).tolist() if int(j) != i}
        nbr_sets.append(cur)

    pairs: set[Tuple[int, int]] = set()
    for i in range(n):
        for j in nbr_sets[i]:
            if rule_mode == 'mutual' and i not in nbr_sets[j]:
                continue
            a, b = (i, int(j)) if i < int(j) else (int(j), i)
            pairs.add((a, b))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Pair scoring backend
# ---------------------------------------------------------------------------

class _PairScorer:
    def __init__(
        self,
        raw_trees: Sequence[Any],
        tree_data_list: Sequence[Any],
        *,
        matcher_kind: str,
        matcher_kwargs: Optional[Mapping[str, Any]] = None,
        pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
    ) -> None:
        self.raw_trees = list(raw_trees)
        self.tree_data_list = list(tree_data_list)
        self.matcher_kind = str(matcher_kind).lower().strip()
        self.matcher_kwargs = dict(matcher_kwargs or {})
        self.pair_scorer = pair_scorer

        self._generic_matcher = None
        self._fast_matcher = None
        self._encoded = None
        self.n_score_calls = 0
        self.n_self_score_calls = 0

        if self.pair_scorer is not None:
            self.matcher_kind = "callable"
        elif self.matcher_kind == "fast":
            from path_matcher.fast_match import FastTreePathMatcher

            self._fast_matcher = FastTreePathMatcher(**self.matcher_kwargs)
            self._fast_matcher.fit_encoder(self.tree_data_list)
            self._encoded = [self._fast_matcher.encode_tree(T) for T in self.tree_data_list]
        elif self.matcher_kind == "generic":
            from path_matcher.matcher import TreePathMatcher

            self._generic_matcher = TreePathMatcher(**self.matcher_kwargs)
        else:
            raise ValueError("matcher_kind must be 'generic', 'fast', or you must pass pair_scorer=...")

    def score_pair(self, i: int, j: int) -> PairScore:
        self.n_score_calls += 1
        if self.pair_scorer is not None:
            out = self.pair_scorer(self.raw_trees[i], self.raw_trees[j])
            if isinstance(out, tuple) and len(out) == 2:
                pairs, score = out
                return list(pairs), float(score)
            return [], float(out)

        if self.matcher_kind == "fast":
            pairs, score = self._fast_matcher.predict_encoded(self._encoded[i], self._encoded[j])
            return list(pairs), float(score)

        # generic matcher
        pairs, score = self._generic_matcher.fit(self.raw_trees[i], self.raw_trees[j]).predict()
        return list(pairs), float(score)

    def self_score(self, i: int) -> float:
        self.n_self_score_calls += 1
        _pairs, score = self.score_pair(i, i)
        return float(score)


# ---------------------------------------------------------------------------
# Compatibility helpers around matched sequences
# ---------------------------------------------------------------------------

def _label_lookup(tree: Any, idx: int, *, phi_name: str = "label") -> Any:
    if _looks_like_treedata(tree):
        orig = np.asarray(tree.orig_index)
        hits = np.where(orig == int(idx))[0]
        if hits.size == 0:
            raise KeyError(f"Could not map original index {idx} back into TreeData.orig_index")
        return tree.label[int(hits[0])]

    if hasattr(tree, "vs"):
        return tree.vs[int(idx)][phi_name]

    raise TypeError("Unsupported tree type for label extraction")


def extract_seqs(pairs: Sequence[Tuple[int, int]], G: Any, H: Any, *, phi_name: str = "label") -> Tuple[List[Any], List[Any]]:
    resG = [_label_lookup(G, int(u), phi_name=phi_name) for (u, _v) in pairs]
    resH = [_label_lookup(H, int(v), phi_name=phi_name) for (_u, v) in pairs]
    return resG, resH


def extract_toy_alpha_seqs(pairs: Sequence[Tuple[int, int]], G: Any, H: Any, *, phi_name: str = "label") -> Tuple[List[str], List[str]]:
    seqG, seqH = extract_seqs(pairs, G, H, phi_name=phi_name)
    def _to_alpha(x: Any) -> str:
        if isinstance(x, str) and len(x) == 1 and x.isalpha():
            return x
        if isinstance(x, (int, np.integer)):
            return chr(ord('A') + int(x))
        s = str(x)
        if s.startswith("sym") and s[3:].isdigit():
            return chr(ord('A') + int(s[3:]))
        return s
    return [_to_alpha(x) for x in seqG], [_to_alpha(x) for x in seqH]


def extract_alpha_seqs_by_cluster(
    matches: Mapping[Tuple[int, int], Sequence[Tuple[int, int]]],
    gt_labels: Sequence[int],
    tree_list: Sequence[Any],
    *,
    phi_name: str = "label",
) -> Dict[int, List[List[str]]]:
    res: Dict[int, List[List[str]]] = {int(c): [] for c in np.unique(np.asarray(gt_labels, dtype=int)).tolist()}
    n = len(tree_list)
    gt = list(gt_labels)
    for i in range(n - 1):
        for j in range(i + 1, n):
            if gt[i] == gt[j] and (i, j) in matches:
                a, b = extract_toy_alpha_seqs(matches[(i, j)], tree_list[i], tree_list[j], phi_name=phi_name)
                res[int(gt[i])].extend([a, b])
    return res


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TreeClusterEmbedder:
    """
    Compute tree embeddings and HDBSCAN cluster labels from matcher-based scores.

    Parameters
    ----------
    similarity_mode:
        One of ``'full'``, ``'landmarks'``, or ``'approx_knn'``.
    matcher_kind:
        ``'generic'`` uses ``path_matcher.matcher.TreePathMatcher``.
        ``'fast'`` uses ``path_matcher.fast_match.FastTreePathMatcher``.
    matcher_kwargs:
        Passed to the selected matcher class.
    pair_scorer:
        Optional user callable ``f(G, H) -> score`` or ``(pairs, score)``.
        If supplied, it overrides ``matcher_kind``.
    normalizer:
        Similarity normalization. Strings include ``'raw'``, ``'cosine'``,
        ``'jaccard'``, ``'dice'``, and ``'overlap'``.
    diag_source:
        For square full matrices, use either ``'row_max'`` (old behaviour) or
        ``'self'`` (explicit self-match scores).
    phi_name, order, ts_field, strict_tree:
        Used when coercing igraph inputs to TreeData.
    n_landmarks:
        Number of landmarks when ``similarity_mode='landmarks'``.
    candidate_top_k:
        Candidate neighbour count for ``similarity_mode='approx_knn'``.
    sketch_hash_dim:
        Feature dimension for cheap hashed tree sketches.
    feature_metric:
        UMAP metric used when the input to UMAP is a feature matrix rather than a
        precomputed distance matrix.
    """

    def __init__(
        self,
        *,
        similarity_mode: str = "full",
        matcher_kind: str = "generic",
        matcher_kwargs: Optional[Mapping[str, Any]] = None,
        pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
        normalizer: NormalizerLike = "jaccard",
        diag_source: str = "row_max",
        phi_name: str = "label",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        label_getter: Optional[Callable[[Any], Any]] = None,
        n_landmarks: int = 32,
        landmark_mode: str = "random",
        candidate_top_k: int = 10,
        sketch_hash_dim: Optional[int] = 256,
        sketch_binary: bool = False,
        candidate_metric: str = "cosine",
        candidate_backend: str = "auto",
        candidate_rule: str = "union",
        candidate_backend_kwargs: Optional[Mapping[str, Any]] = None,
        feature_metric: str = "euclidean",
        distance_transform: str = "sqrt_one_minus",
        umap_kwargs: Optional[Mapping[str, Any]] = None,
        hdbscan_kwargs: Optional[Mapping[str, Any]] = None,
        return_matches: bool = True,
        compute_self_scores: Optional[bool] = None,
        random_state: int = 0,
        verbose: bool = False,
    ) -> None:
        mode = str(similarity_mode).lower().strip()
        if mode not in {"full", "landmarks", "approx_knn"}:
            raise ValueError("similarity_mode must be 'full', 'landmarks', or 'approx_knn'")
        self.similarity_mode = mode
        self.matcher_kind = str(matcher_kind).lower().strip()
        self.matcher_kwargs = dict(matcher_kwargs or {})
        self.pair_scorer = pair_scorer
        self.normalizer = normalizer
        self.diag_source = str(diag_source)
        self.phi_name = phi_name
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = strict_tree
        self.label_getter = label_getter
        self.n_landmarks = int(n_landmarks)
        self.landmark_mode = str(landmark_mode)
        self.candidate_top_k = int(candidate_top_k)
        self.sketch_hash_dim = sketch_hash_dim
        self.sketch_binary = bool(sketch_binary)
        self.candidate_metric = str(candidate_metric)
        self.candidate_backend = str(candidate_backend)
        self.candidate_rule = str(candidate_rule)
        self.candidate_backend_kwargs = dict(candidate_backend_kwargs or {})
        self.feature_metric = str(feature_metric)
        self.distance_transform = str(distance_transform)
        self.umap_kwargs = dict(umap_kwargs or {})
        self.hdbscan_kwargs = dict(hdbscan_kwargs or {})
        self.return_matches = bool(return_matches)
        self.compute_self_scores = compute_self_scores
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        self.tree_list_: List[Any] = []
        self.tree_data_list_: List[Any] = []
        self.gt_labels_: Optional[np.ndarray] = None
        self.result_: Optional[TreeEmbeddingResult] = None

    def _print(self, *msg: Any) -> None:
        if self.verbose:
            print(*msg)

    def _to_tree(self, G: Any) -> Any:
        return _as_treedata(
            G,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
        )

    def _resolve_compute_self_scores(self) -> bool:
        if self.compute_self_scores is not None:
            return bool(self.compute_self_scores)
        if self.similarity_mode in {"landmarks"}:
            return True
        if isinstance(self.normalizer, str):
            mode = self.normalizer.lower().strip()
            if mode.startswith("self_"):
                return True
        return self.diag_source == "self"

    def _run_umap(self, X: np.ndarray, *, metric: str) -> np.ndarray:
        umap_mod = _require_umap()
        kwargs = dict(self.umap_kwargs)
        kwargs.setdefault("random_state", self.random_state)
        kwargs.setdefault("n_components", 2)
        kwargs.setdefault("metric", metric)
        reducer = umap_mod.UMAP(**kwargs)
        return reducer.fit_transform(X)

    def _run_hdbscan(self, embedding: np.ndarray) -> np.ndarray:
        hdbscan_mod = _require_hdbscan()
        kwargs = dict(self.hdbscan_kwargs)
        clusterer = hdbscan_mod.HDBSCAN(**kwargs)
        clusterer.fit(embedding)
        return np.asarray(clusterer.labels_, dtype=int)

    def _distance_from_similarity(self, S: np.ndarray) -> np.ndarray:
        mode = self.distance_transform.lower().strip()
        if mode in {"sqrt_one_minus", "sqrt"}:
            return to_diff(S)
        if mode in {"one_minus", "linear"}:
            return np.maximum(0.0, 1.0 - np.asarray(S, dtype=float))
        raise ValueError("distance_transform must be 'sqrt_one_minus' or 'one_minus'")

    def _make_pair_scorer(self) -> _PairScorer:
        matcher_kwargs = dict(self.matcher_kwargs)
        matcher_kwargs.setdefault("phi_name", self.phi_name)
        matcher_kwargs.setdefault("order", self.order)
        matcher_kwargs.setdefault("ts_field", self.ts_field)
        matcher_kwargs.setdefault("strict_tree", self.strict_tree)
        return _PairScorer(
            self.tree_list_,
            self.tree_data_list_,
            matcher_kind=self.matcher_kind,
            matcher_kwargs=matcher_kwargs,
            pair_scorer=self.pair_scorer,
        )

    def _compute_self_scores(self, scorer: _PairScorer) -> np.ndarray:
        n = len(self.tree_list_)
        out = np.zeros(n, dtype=float)
        for i in range(n):
            out[i] = scorer.self_score(i)
        return out

    def _full_pairwise(self, scorer: _PairScorer) -> TreeEmbeddingResult:
        n = len(self.tree_list_)
        sims = np.zeros((n, n), dtype=float)
        matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for i in range(n):
            for j in range(i + 1, n):
                pairs, score = scorer.score_pair(i, j)
                sims[i, j] = score
                sims[j, i] = score
                if self.return_matches:
                    matches[(i, j)] = pairs

        self_scores = self._compute_self_scores(scorer) if self._resolve_compute_self_scores() else None
        if self_scores is not None:
            np.fill_diagonal(sims, self_scores)

        normalized = normalize_similarity(
            sims,
            normalizer=self.normalizer,
            diag_source=self.diag_source,
            row_scales=self_scores if self_scores is not None and self.diag_source == "self" else None,
            col_scales=self_scores if self_scores is not None and self.diag_source == "self" else None,
        )
        dists = self._distance_from_similarity(normalized)
        embedding = self._run_umap(dists, metric="precomputed")
        pred = self._run_hdbscan(embedding)

        return TreeEmbeddingResult(
            tree_list=self.tree_list_,
            tree_data_list=self.tree_data_list_,
            gt_labels=self.gt_labels_,
            pred_labels=pred,
            embedding=embedding,
            similarity_mode=self.similarity_mode,
            raw_scores=sims,
            normalized_scores=normalized,
            distances=dists,
            matches=matches,
            self_scores=self_scores,
            metadata={
                "normalizer": self.normalizer,
                "diag_source": self.diag_source,
                "n_score_calls": int(scorer.n_score_calls),
                "n_self_score_calls": int(scorer.n_self_score_calls),
                "n_nonself_score_calls": int(scorer.n_score_calls - scorer.n_self_score_calls),
            },
        )

    def _landmark_features(self, scorer: _PairScorer) -> TreeEmbeddingResult:
        features0 = build_tree_token_features(
            self.tree_data_list_,
            label_getter=self.label_getter,
            hash_dim=self.sketch_hash_dim,
            binary=self.sketch_binary,
        )
        landmark_idx = select_landmarks(
            features0,
            self.n_landmarks,
            mode=self.landmark_mode,
            random_state=self.random_state,
        )

        n = len(self.tree_list_)
        L = len(landmark_idx)
        raw = np.zeros((n, L), dtype=float)
        matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for i in range(n):
            for pos, j in enumerate(landmark_idx.tolist()):
                pairs, score = scorer.score_pair(i, int(j))
                raw[i, pos] = score
                if self.return_matches:
                    matches[(i, int(j))] = pairs

        self_scores = self._compute_self_scores(scorer)
        landmark_self = self_scores[landmark_idx]
        normalized = normalize_similarity(
            raw,
            normalizer=self.normalizer,
            row_scales=self_scores,
            col_scales=landmark_self,
            diag_source="self",
        )
        embedding = self._run_umap(normalized, metric=self.feature_metric)
        pred = self._run_hdbscan(embedding)

        return TreeEmbeddingResult(
            tree_list=self.tree_list_,
            tree_data_list=self.tree_data_list_,
            gt_labels=self.gt_labels_,
            pred_labels=pred,
            embedding=embedding,
            similarity_mode=self.similarity_mode,
            raw_scores=raw,
            normalized_scores=normalized,
            features=normalized,
            matches=matches,
            self_scores=self_scores,
            landmark_indices=landmark_idx,
            metadata={
                "normalizer": self.normalizer,
                "feature_metric": self.feature_metric,
                "landmark_mode": self.landmark_mode,
                "n_landmarks": int(L),
                "n_score_calls": int(scorer.n_score_calls),
                "n_self_score_calls": int(scorer.n_self_score_calls),
                "n_nonself_score_calls": int(scorer.n_score_calls - scorer.n_self_score_calls),
            },
        )

    def _approx_knn(self, scorer: _PairScorer) -> TreeEmbeddingResult:
        sketch = build_tree_token_features(
            self.tree_data_list_,
            label_getter=self.label_getter,
            hash_dim=self.sketch_hash_dim,
            binary=self.sketch_binary,
        )
        resolved_backend = self.candidate_backend if self.candidate_backend != 'auto' else ('pynndescent' if pynndescent is not None else 'sklearn')
        cand_pairs = candidate_pairs_from_features(
            sketch,
            top_k=self.candidate_top_k,
            metric=self.candidate_metric,
            backend=self.candidate_backend,
            random_state=self.random_state,
            rule=self.candidate_rule,
            backend_kwargs=self.candidate_backend_kwargs,
        )

        n = len(self.tree_list_)
        sims = np.zeros((n, n), dtype=float)
        matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for i, j in cand_pairs:
            pairs, score = scorer.score_pair(i, j)
            sims[i, j] = score
            sims[j, i] = score
            if self.return_matches:
                matches[(i, j)] = pairs

        self_scores = self._compute_self_scores(scorer) if self._resolve_compute_self_scores() else None
        if self_scores is not None:
            np.fill_diagonal(sims, self_scores)

        normalized = normalize_similarity(
            sims,
            normalizer=self.normalizer,
            diag_source=self.diag_source,
            row_scales=self_scores if self_scores is not None and self.diag_source == "self" else None,
            col_scales=self_scores if self_scores is not None and self.diag_source == "self" else None,
        )
        dists = self._distance_from_similarity(normalized)
        embedding = self._run_umap(dists, metric="precomputed")
        pred = self._run_hdbscan(embedding)

        return TreeEmbeddingResult(
            tree_list=self.tree_list_,
            tree_data_list=self.tree_data_list_,
            gt_labels=self.gt_labels_,
            pred_labels=pred,
            embedding=embedding,
            similarity_mode=self.similarity_mode,
            raw_scores=sims,
            normalized_scores=normalized,
            distances=dists,
            matches=matches,
            self_scores=self_scores,
            candidate_pairs=cand_pairs,
            metadata={
                "normalizer": self.normalizer,
                "diag_source": self.diag_source,
                "candidate_top_k": self.candidate_top_k,
                "candidate_metric": self.candidate_metric,
                "candidate_backend": self.candidate_backend,
                "resolved_candidate_backend": resolved_backend,
                "candidate_rule": self.candidate_rule,
                "n_candidate_pairs": int(len(cand_pairs)),
                "candidate_pair_fraction": float(len(cand_pairs)) / float(max(n * (n - 1) // 2, 1)),
                "n_score_calls": int(scorer.n_score_calls),
                "n_self_score_calls": int(scorer.n_self_score_calls),
                "n_nonself_score_calls": int(scorer.n_score_calls - scorer.n_self_score_calls),
            },
        )

    def fit(self, some_trees: Any) -> "TreeClusterEmbedder":
        tree_list, gt_labels = expand_tree_list(some_trees)
        self.tree_list_ = list(tree_list)
        try:
            self.tree_data_list_ = [self._to_tree(T) for T in self.tree_list_]
        except Exception:
            if self.pair_scorer is not None and self.similarity_mode == "full":
                # Full pairwise mode can operate purely on the user-supplied scorer.
                self.tree_data_list_ = list(self.tree_list_)
            else:
                raise
        self.gt_labels_ = None if gt_labels is None else np.asarray(gt_labels, dtype=int)

        scorer = self._make_pair_scorer()
        self._print(f"Scoring mode={self.similarity_mode!r} on n={len(self.tree_list_)} trees")

        if self.similarity_mode == "full":
            self.result_ = self._full_pairwise(scorer)
        elif self.similarity_mode == "landmarks":
            self.result_ = self._landmark_features(scorer)
        else:
            self.result_ = self._approx_knn(scorer)
        return self

    def predict(self) -> TreeEmbeddingResult:
        if self.result_ is None:
            raise RuntimeError("Call fit(...) before predict().")
        return self.result_

    def fit_predict(self, some_trees: Any) -> TreeEmbeddingResult:
        return self.fit(some_trees).predict()


# ---------------------------------------------------------------------------
# Convenience / compatibility wrappers
# ---------------------------------------------------------------------------

def compute_sims_and_matches(
    tree_list: Sequence[Any],
    *,
    phi_name: str = "label",
    matcher_kind: str = "generic",
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[Tuple[int, int]]]]:
    emb = TreeClusterEmbedder(
        similarity_mode="full",
        matcher_kind=matcher_kind,
        matcher_kwargs=matcher_kwargs,
        pair_scorer=pair_scorer,
        normalizer="raw",
        phi_name=phi_name,
        return_matches=True,
        compute_self_scores=False,
        umap_kwargs={"n_neighbors": 5},
        hdbscan_kwargs={"min_cluster_size": 2},
    )
    # Avoid requiring UMAP/HDBSCAN for this compatibility helper.
    tree_raw, _labels = expand_tree_list(list(tree_list))
    emb.tree_list_ = list(tree_raw)
    if pair_scorer is not None:
        emb.tree_data_list_ = list(emb.tree_list_)
    else:
        emb.tree_data_list_ = [emb._to_tree(T) for T in emb.tree_list_]
    scorer = emb._make_pair_scorer()
    n = len(emb.tree_list_)
    sims = np.zeros((n, n), dtype=float)
    matches: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            pairs, score = scorer.score_pair(i, j)
            sims[i, j] = score
            sims[j, i] = score
            matches[(i, j)] = pairs
    return sims, matches


def compute_sims_and_matches_and_sequences(
    tree_list: Sequence[Any],
    *,
    phi_name: str = "label",
    matcher_kind: str = "generic",
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
) -> Tuple[np.ndarray, Dict[Tuple[int, int], List[Tuple[int, int]]], Dict[Tuple[int, int], Tuple[List[Any], List[Any]]]]:
    sims, matches = compute_sims_and_matches(
        tree_list,
        phi_name=phi_name,
        matcher_kind=matcher_kind,
        matcher_kwargs=matcher_kwargs,
        pair_scorer=pair_scorer,
    )
    seqs = {
        key: extract_seqs(val, tree_list[key[0]], tree_list[key[1]], phi_name=phi_name)
        for key, val in matches.items()
    }
    return sims, matches, seqs


def embed_and_cluster_sns(
    some_trees: Any,
    *,
    normalizer: NormalizerLike = "jaccard",
    phi_name: str = "label",
    matcher_kind: str = "generic",
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
    similarity_mode: str = "full",
    title: Optional[str] = None,
    show_plot: bool = True,
    scatter_kwargs: Optional[Mapping[str, Any]] = None,
    **kwargs: Any,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], np.ndarray, List[Any], Optional[np.ndarray], np.ndarray]:
    """
    Compatibility wrapper in the spirit of the older ``gw_clustering.embed_and_cluster_sns``.

    Returns
    -------
    matches, embedding, tree_list, gt_labels, pred_labels
    """
    emb = TreeClusterEmbedder(
        similarity_mode=similarity_mode,
        matcher_kind=matcher_kind,
        matcher_kwargs=matcher_kwargs,
        pair_scorer=pair_scorer,
        normalizer=normalizer,
        phi_name=phi_name,
        **kwargs,
    )
    res = emb.fit_predict(some_trees)

    if show_plot:
        import matplotlib.pyplot as plt
        import seaborn as sns

        skw = dict(scatter_kwargs or {})
        if res.gt_labels is None:
            sns.scatterplot(x=res.embedding[:, 0], y=res.embedding[:, 1], **skw)
        else:
            sns.scatterplot(x=res.embedding[:, 0], y=res.embedding[:, 1], hue=res.gt_labels, palette="deep", **skw)
            plt.legend([], [], frameon=False)
        if title is None:
            title = f"Embedding based on {similarity_mode} scores with {normalizer} normalization"
        plt.title(title)
        plt.show()

    return res.matches, res.embedding, res.tree_list, res.gt_labels, res.pred_labels


def embed_and_cluster(
    some_trees: Any,
    *,
    normalizer: NormalizerLike = "jaccard",
    phi_name: str = "label",
    matcher_kind: str = "generic",
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    pair_scorer: Optional[Callable[[Any, Any], Union[ScoreLike, PairScore]]] = None,
    similarity_mode: str = "full",
    title: Optional[str] = None,
    show_plot: bool = True,
    **kwargs: Any,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], np.ndarray, List[Any], Optional[np.ndarray], np.ndarray]:
    emb = TreeClusterEmbedder(
        similarity_mode=similarity_mode,
        matcher_kind=matcher_kind,
        matcher_kwargs=matcher_kwargs,
        pair_scorer=pair_scorer,
        normalizer=normalizer,
        phi_name=phi_name,
        **kwargs,
    )
    res = emb.fit_predict(some_trees)

    if show_plot:
        import matplotlib.pyplot as plt

        if res.gt_labels is None:
            plt.scatter(res.embedding[:, 0], res.embedding[:, 1])
        else:
            plt.scatter(res.embedding[:, 0], res.embedding[:, 1], c=res.gt_labels)
        if title is None:
            title = f"Embedding based on {similarity_mode} scores with {normalizer} normalization"
        plt.title(title)
        plt.show()

    return res.matches, res.embedding, res.tree_list, res.gt_labels, res.pred_labels


__all__ = [
    "TreeEmbeddingResult",
    "TreeClusterEmbedder",
    "expand_tree_list",
    "compute_sims_and_matches",
    "compute_sims_and_matches_and_sequences",
    "extract_seqs",
    "extract_toy_alpha_seqs",
    "extract_alpha_seqs_by_cluster",
    "cos_normalize",
    "jacc_normalize",
    "dice_normalize",
    "overlap_normalize",
    "normalize_similarity",
    "to_diff",
    "build_tree_token_features",
    "select_landmarks",
    "candidate_pairs_from_features",
    "embed_and_cluster",
    "embed_and_cluster_sns",
]
