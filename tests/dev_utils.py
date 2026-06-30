"""tests/dev_utils.py

Utilities for developing and sanity-checking unit tests for the tree-path matcher.

Design goals
------------
- Use the *same graph format* as the project samplers: directed igraph.Graph trees.
- Keep everything easy to edit in one place (samplers, planting, scoring, plotting).
- Provide small, reproducible test cases (10–20 nodes) for notebook-based inspection.

Conventions
-----------
- Vertex attributes:
    - "label": str (categorical symbol)
    - "is_planted": int in {0,1} (ground-truth planted path indicator)
    - "weight": float (optional; used only for visualization / weighted-identity scoring)
- Edges are directed parent -> child.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import igraph as ig

# Project modules (these live in path_matcher/)
from path_matcher.planted_path_sampler import (
    TreePathSampler,
    build_pi,
    label_tree_with_planted_path,
)
from path_matcher.matcher import TreePathMatcher


EPS = 1e-9


# ---------------------------------------------------------------------
# Weight functions
# ---------------------------------------------------------------------

WeightMap = Dict[str, float]
WeightFn = Callable[[str, str], float]


def weighted_identity_w(weight_map: WeightMap) -> WeightFn:
    """w(a,b)=weight_map[a] * 1[a=b]. Missing labels get weight 0."""
    def w(a: str, b: str) -> float:
        if a != b:
            return 0.0
        return float(weight_map.get(str(a), 0.0))
    return w


def unweighted_identity_w(a: str, b: str) -> float:
    """w(a,b)=1[a=b]."""
    return 1.0 if a == b else 0.0


# ---------------------------------------------------------------------
# Simple narrow-tree sampler (spine + optional short side chains)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class NarrowSpineTreeSampler(TreePathSampler):
    """
    Narrow tree with a single long spine (0->1->...->L-1) plus small side chains.

    Parameters
    ----------
    spine_length:
        Number of spine vertices (path length is spine_length).
    stub_prob:
        Probability each spine vertex spawns *each* of its stubs.
    stubs_per_spine_vertex:
        Max number of stubs to attempt per spine vertex.
    stub_chain_length:
        Length of each stub chain (>=1). 1 means a single leaf child.
    """
    spine_length: int = 10
    stub_prob: float = 0.7
    stubs_per_spine_vertex: int = 1
    stub_chain_length: int = 1

    def sample(self, rng: np.random.Generator) -> Tuple[ig.Graph, List[int]]:
        L = int(self.spine_length)
        if L < 2:
            raise ValueError("spine_length must be >= 2")

        edges: List[Tuple[int, int]] = []
        # spine edges
        for i in range(L - 1):
            edges.append((i, i + 1))

        next_vid = L
        # stubs
        for i in range(L):
            for _ in range(int(self.stubs_per_spine_vertex)):
                if rng.random() >= float(self.stub_prob):
                    continue
                parent = i
                # chain of length stub_chain_length
                prev = parent
                for _k in range(int(self.stub_chain_length)):
                    edges.append((prev, next_vid))
                    prev = next_vid
                    next_vid += 1

        g = ig.Graph(n=next_vid, edges=edges, directed=True)
        spine_path = list(range(L))
        return g, spine_path


# ---------------------------------------------------------------------
# Helpers: graph inspection + scoring
# ---------------------------------------------------------------------

def _root_vertex(g: ig.Graph) -> int:
    roots = g.vs.select(_indegree_eq=0)
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root, found {len(roots)}")
    return int(roots[0].index)


def _is_leaf(g: ig.Graph, v: int) -> bool:
    return int(g.degree(int(v), mode="OUT")) == 0


def score_pairs(G: ig.Graph, H: ig.Graph, pairs: Sequence[Tuple[int, int]], w: WeightFn) -> float:
    """Compute sum w(label_G[u], label_H[v]) over a list of matched pairs."""
    labG = G.vs["label"]
    labH = H.vs["label"]
    s = 0.0
    for u, v in pairs:
        s += float(w(str(labG[int(u)]), str(labH[int(v)])))
    return float(s)


def filter_positive_pairs(G: ig.Graph, H: ig.Graph, pairs: Sequence[Tuple[int, int]], w: WeightFn, eps: float = EPS) -> List[Tuple[int, int]]:
    """Keep only pairs with strictly positive contribution under w."""
    labG = G.vs["label"]
    labH = H.vs["label"]
    out: List[Tuple[int, int]] = []
    for u, v in pairs:
        val = float(w(str(labG[int(u)]), str(labH[int(v)])))
        if val > eps:
            out.append((int(u), int(v)))
    return out


def index_by_label_multi(g: ig.Graph, label_attr: str = "label") -> Dict[str, List[int]]:
    """Map label -> list of vertex indices (allows duplicates)."""
    labels = [str(x) for x in g.vs[label_attr]]
    out: Dict[str, List[int]] = {}
    for i, lab in enumerate(labels):
        out.setdefault(lab, []).append(int(i))
    return out


def find_unique_label(g: ig.Graph, label: str, label_attr: str = "label") -> int:
    """Return the unique vertex index with the given label; raises if not exactly one."""
    label = str(label)
    idxs = index_by_label_multi(g, label_attr).get(label, [])
    if len(idxs) != 1:
        raise ValueError(f"Expected exactly one vertex with label {label!r}, found {len(idxs)}.")
    return int(idxs[0])


def index_by_label(g: ig.Graph, label_attr: str = "label") -> Dict[str, int]:
    """Map label -> vertex index, requiring that each label appears exactly once."""
    multi = index_by_label_multi(g, label_attr)
    out: Dict[str, int] = {}
    for lab, idxs in multi.items():
        if len(idxs) != 1:
            raise ValueError(f"Duplicate label {lab!r} in graph (indices {idxs}).")
        out[str(lab)] = int(idxs[0])
    return out


# ---------------------------------------------------------------------
# Core runner: call TreePathMatcher(method="exact")
# ---------------------------------------------------------------------

def run_exact_match(G: ig.Graph, H: ig.Graph, w: Optional[WeightFn]) -> Tuple[List[Tuple[int, int]], float]:
    """
    Run the baseline exact matcher on two igraph trees.

    Uses TreePathMatcher(...).fit(...).predict() (see path_matcher/matcher.py).
    """
    matcher = TreePathMatcher(method="exact", w=w)
    matcher.fit(G, H)
    pairs, score = matcher.predict()
    # ensure plain ints
    pairs2 = [(int(u), int(v)) for (u, v) in pairs]
    return pairs2, float(score)


# ---------------------------------------------------------------------
# Case result containers
# ---------------------------------------------------------------------

@dataclass
class DevMatchCaseResult:
    name: str
    seed: int

    G: ig.Graph
    H: ig.Graph

    # Ground truth
    truth_path_G: List[int]
    truth_path_H: List[int]
    truth_pairs: List[Tuple[int, int]]
    truth_score: float

    # Matcher output
    found_pairs: List[Tuple[int, int]]
    found_score: float
    found_pos_pairs: List[Tuple[int, int]]
    found_pos_score: float

    # Root/leaf inclusion flags (w.r.t. truth_path_* as a path segment in each tree)
    truth_includes_root_G: bool
    truth_includes_leaf_G: bool
    truth_includes_root_H: bool
    truth_includes_leaf_H: bool

    # For debugging / viz
    weight_map: WeightMap
    notes: str = ""


# ---------------------------------------------------------------------
# Case builders
# ---------------------------------------------------------------------

def _ensure_noise_symbols(k: int, *, prefix: str = "N") -> List[str]:
    k = int(k)
    if k <= 0:
        return [f"{prefix}0"]
    return [f"{prefix}{i}" for i in range(k)]


def _make_match_symbols(seg_len: int, *, prefix: str = "M") -> List[str]:
    seg_len = int(seg_len)
    if seg_len <= 0:
        raise ValueError("seg_len must be positive")
    return [f"{prefix}{i}" for i in range(seg_len)]


def _build_pi_noise_only(alphabet: Sequence[str], noise_symbols: Sequence[str], rng: np.random.Generator) -> np.ndarray:
    # Put mass only on noise symbols; match symbols get prob 0 (by omission).
    pi_dict = {str(s): 1.0 for s in noise_symbols}
    return build_pi(list(alphabet), pi=pi_dict, rng=rng)


def _plant_segment_deterministic(
    g: ig.Graph,
    segment: Sequence[int],
    *,
    alphabet: Sequence[str],
    pi: np.ndarray,
    planted_sequence: Sequence[str],
    rng: np.random.Generator,
    label_attr: str = "label",
    planted_attr: str = "is_planted",
) -> None:
    """
    Deterministically plant planted_sequence onto 'segment' by calling
    label_tree_with_planted_path with p_obs=1 and K=len(segment)=len(planted_sequence).

    This uses the project code-path (same attribute names, same semantics).
    """
    if len(segment) != len(planted_sequence):
        raise ValueError("segment length must equal planted_sequence length for deterministic planting.")
    label_tree_with_planted_path(
        g,
        list(map(int, segment)),
        alphabet=list(map(str, alphabet)),
        pi=np.asarray(pi, dtype=float),
        planted_sequence=list(map(str, planted_sequence)),
        p_obs=1.0,
        rng=rng,
        label_attr=label_attr,
        planted_attr=planted_attr,
    )


def _set_vertex_weights_from_label(g: ig.Graph, weight_map: WeightMap, label_attr: str = "label", weight_attr: str = "weight") -> None:
    labels = [str(x) for x in g.vs[label_attr]]
    g.vs[weight_attr] = [float(weight_map.get(lab, 0.0)) for lab in labels]


def case_root_leaf_inclusion(
    *,
    seed: int,
    seg_len: int = 6,
    include_root_G: bool = True,
    include_leaf_G: bool = True,
    include_root_H: bool = True,
    include_leaf_H: bool = True,
    # sampler hyperparameters (narrow sampler)
    stub_prob: float = 0.7,
    stubs_per_spine_vertex: int = 1,
    stub_chain_length: int = 1,
    noise_k: int = 2,
    weight_mode: str = "increasing",  # "increasing" or "constant"
) -> DevMatchCaseResult:
    """
    Root/leaf inclusion/exclusion test.

    We build two narrow trees with *possibly different* spine lengths so that the planted
    segment has length seg_len in both trees, while being anchored (or not) at the root/leaf
    according to the include_* flags.

    Planting is deterministic and uses the project's label_tree_with_planted_path.
    """
    seed = int(seed)
    rng_master = np.random.default_rng(seed)
    rngG = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    rngH = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

    seg_len = int(seg_len)
    if seg_len < 2:
        raise ValueError("seg_len should be >=2 for meaningful root/leaf tests.")

    # Match symbols (unique, deterministic)
    match_symbols = _make_match_symbols(seg_len, prefix="M")
    noise_symbols = _ensure_noise_symbols(noise_k, prefix="N")
    alphabet = list(match_symbols) + list(noise_symbols)

    # Noise-only π so match symbols never appear except on the planted segment.
    piG = _build_pi_noise_only(alphabet, noise_symbols, rngG)
    piH = _build_pi_noise_only(alphabet, noise_symbols, rngH)

    # Weight map for weighted-identity scoring
    weight_map: WeightMap = {}
    if weight_mode == "constant":
        for s in match_symbols:
            weight_map[str(s)] = 1.0
    else:
        # increasing weights (makes it easy to sanity-check sums)
        for i, s in enumerate(match_symbols):
            weight_map[str(s)] = float(i + 1)
    for s in noise_symbols:
        weight_map[str(s)] = 0.3

    w = weighted_identity_w(weight_map)

    # Choose spine lengths so that:
    #   segment length = seg_len
    #   segment includes root iff include_root_*
    #   segment includes leaf iff include_leaf_*
    def spine_len_for(include_root: bool, include_leaf: bool) -> int:
        return seg_len + (0 if include_root else 1) + (0 if include_leaf else 1)

    Lg = spine_len_for(include_root_G, include_leaf_G)
    Lh = spine_len_for(include_root_H, include_leaf_H)

    samplerG = NarrowSpineTreeSampler(
        spine_length=Lg,
        stub_prob=stub_prob,
        stubs_per_spine_vertex=stubs_per_spine_vertex,
        stub_chain_length=stub_chain_length,
    )
    samplerH = NarrowSpineTreeSampler(
        spine_length=Lh,
        stub_prob=stub_prob,
        stubs_per_spine_vertex=stubs_per_spine_vertex,
        stub_chain_length=stub_chain_length,
    )

    G, spineG = samplerG.sample(rngG)
    H, spineH = samplerH.sample(rngH)

    # Segment indices: offset head/tail by 1 if we want to exclude root/leaf
    startG = 0 if include_root_G else 1
    endG = (Lg - 1) if include_leaf_G else (Lg - 2)
    segG = spineG[startG : endG + 1]
    if len(segG) != seg_len:
        raise RuntimeError("Internal error: segG length mismatch")

    startH = 0 if include_root_H else 1
    endH = (Lh - 1) if include_leaf_H else (Lh - 2)
    segH = spineH[startH : endH + 1]
    if len(segH) != seg_len:
        raise RuntimeError("Internal error: segH length mismatch")

    # Plant labels on segG and segH
    _plant_segment_deterministic(G, segG, alphabet=alphabet, pi=piG, planted_sequence=match_symbols, rng=rngG)
    _plant_segment_deterministic(H, segH, alphabet=alphabet, pi=piH, planted_sequence=match_symbols, rng=rngH)

    _set_vertex_weights_from_label(G, weight_map)
    _set_vertex_weights_from_label(H, weight_map)

    # Ground truth pairs: match planted symbols by label (they are unique; noise labels may repeat)
    truth_pairs = [(find_unique_label(G, s, "label"), find_unique_label(H, s, "label")) for s in match_symbols]
    truth_score = float(sum(weight_map[s] for s in match_symbols))

    # Run matcher
    found_pairs, found_score = run_exact_match(G, H, w)
    found_pos_pairs = filter_positive_pairs(G, H, found_pairs, w)
    found_pos_score = score_pairs(G, H, found_pos_pairs, w)

    # Root/leaf inclusion flags w.r.t. the *truth segment*
    rootG = _root_vertex(G)
    rootH = _root_vertex(H)

    truth_includes_root_G = (int(segG[0]) == int(rootG))
    truth_includes_leaf_G = _is_leaf(G, int(segG[-1]))
    truth_includes_root_H = (int(segH[0]) == int(rootH))
    truth_includes_leaf_H = _is_leaf(H, int(segH[-1]))

    name = f"rootleaf(G r={include_root_G} l={include_leaf_G}; H r={include_root_H} l={include_leaf_H})"

    return DevMatchCaseResult(
        name=name,
        seed=seed,
        G=G,
        H=H,
        truth_path_G=list(map(int, segG)),
        truth_path_H=list(map(int, segH)),
        truth_pairs=truth_pairs,
        truth_score=truth_score,
        found_pairs=found_pairs,
        found_score=float(found_score),
        found_pos_pairs=found_pos_pairs,
        found_pos_score=float(found_pos_score),
        truth_includes_root_G=bool(truth_includes_root_G),
        truth_includes_leaf_G=bool(truth_includes_leaf_G),
        truth_includes_root_H=bool(truth_includes_root_H),
        truth_includes_leaf_H=bool(truth_includes_leaf_H),
        weight_map=weight_map,
        notes="weighted-identity; noise weights=0; match symbols unique",
    )


def case_no_matches(
    *,
    seed: int,
    spine_length_G: int = 10,
    spine_length_H: int = 10,
    stub_prob: float = 0.7,
    stubs_per_spine_vertex: int = 1,
    stub_chain_length: int = 1,
) -> DevMatchCaseResult:
    """
    Disjoint alphabets across the two trees -> score should be 0.

    Uses a weighted-identity w with positive weights, but symbols are disjoint so no pair matches.
    """
    seed = int(seed)
    rng_master = np.random.default_rng(seed)
    rngG = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))
    rngH = np.random.default_rng(int(rng_master.integers(0, 2**32 - 1)))

    samplerG = NarrowSpineTreeSampler(
        spine_length=int(spine_length_G),
        stub_prob=stub_prob,
        stubs_per_spine_vertex=stubs_per_spine_vertex,
        stub_chain_length=stub_chain_length,
    )
    samplerH = NarrowSpineTreeSampler(
        spine_length=int(spine_length_H),
        stub_prob=stub_prob,
        stubs_per_spine_vertex=stubs_per_spine_vertex,
        stub_chain_length=stub_chain_length,
    )

    G, spineG = samplerG.sample(rngG)
    H, spineH = samplerH.sample(rngH)

    # Disjoint alphabets: G uses G0.., H uses H0..
    alphabetG = [f"G{i}" for i in range(4)]
    alphabetH = [f"H{i}" for i in range(4)]

    piG = build_pi(alphabetG, rng=rngG)
    piH = build_pi(alphabetH, rng=rngH)

    # Label everything iid π; mark no planted path.
    G.vs["label"] = [str(x) for x in rngG.choice(alphabetG, size=G.vcount(), replace=True, p=piG)]
    H.vs["label"] = [str(x) for x in rngH.choice(alphabetH, size=H.vcount(), replace=True, p=piH)]
    G.vs["is_planted"] = [0] * G.vcount()
    H.vs["is_planted"] = [0] * H.vcount()

    # Positive weights, but no overlap across graphs
    weight_map: WeightMap = {s: 1.0 for s in alphabetG + alphabetH}
    w = weighted_identity_w(weight_map)

    _set_vertex_weights_from_label(G, weight_map)
    _set_vertex_weights_from_label(H, weight_map)

    truth_pairs: List[Tuple[int, int]] = []
    truth_score = 0.0

    found_pairs, found_score = run_exact_match(G, H, w)
    found_pos_pairs = filter_positive_pairs(G, H, found_pairs, w)
    found_pos_score = score_pairs(G, H, found_pos_pairs, w)

    rootG = _root_vertex(G)
    rootH = _root_vertex(H)

    return DevMatchCaseResult(
        name="no_matches_disjoint_alphabets",
        seed=seed,
        G=G,
        H=H,
        truth_path_G=[],
        truth_path_H=[],
        truth_pairs=truth_pairs,
        truth_score=float(truth_score),
        found_pairs=found_pairs,
        found_score=float(found_score),
        found_pos_pairs=found_pos_pairs,
        found_pos_score=float(found_pos_score),
        truth_includes_root_G=False,
        truth_includes_leaf_G=False,
        truth_includes_root_H=False,
        truth_includes_leaf_H=False,
        weight_map=weight_map,
        notes="Expected: empty positive match; score 0",
    )


@dataclass
class DevCompareResult:
    """
    For cases that compare two weight functions (e.g. weighted vs unweighted identity)
    on the same pair of trees.
    """
    name: str
    weighted: DevMatchCaseResult
    unweighted: DevMatchCaseResult


def case_weighted_vs_unweighted_blockswap(*, seed: int = 0) -> DevCompareResult:
    """
    Path-graphs where two matching blocks appear in opposite order, creating an LCS-style tradeoff.

    G: A0 A1 A2 B0 B1 B2 B3
    H: B0 B1 B2 B3 A0 A1 A2

    - Unweighted identity prefers B-block (length 4).
    - Weighted identity prefers A-block if weights(A) > weights(B total).
    """
    seed = int(seed)
    rng = np.random.default_rng(seed)

    A = [f"A{i}" for i in range(3)]
    B = [f"B{i}" for i in range(4)]
    labelsG = A + B
    labelsH = B + A

    # Build two directed path trees
    def make_path(labels: Sequence[str]) -> Tuple[ig.Graph, List[int]]:
        n = len(labels)
        edges = [(i, i + 1) for i in range(n - 1)]
        g = ig.Graph(n=n, edges=edges, directed=True)
        g.vs["label"] = list(map(str, labels))
        g.vs["is_planted"] = [0] * n
        return g, list(range(n))

    G, pathG = make_path(labelsG)
    H, pathH = make_path(labelsH)

    # Weighted identity: A-block very heavy, B-block light
    weight_map_weighted: WeightMap = {}
    for s in A:
        weight_map_weighted[s] = 10.0
    for s in B:
        weight_map_weighted[s] = 1.0
    w_weighted = weighted_identity_w(weight_map_weighted)

    _set_vertex_weights_from_label(G, weight_map_weighted)
    _set_vertex_weights_from_label(H, weight_map_weighted)

    # Expected truth for weighted: match A symbols
    idxG = index_by_label(G)
    idxH = index_by_label(H)
    truth_pairs_A = [(idxG[s], idxH[s]) for s in A]
    truth_score_A = float(sum(weight_map_weighted[s] for s in A))

    # Run weighted
    pairs_w, score_w = run_exact_match(G, H, w_weighted)
    pos_pairs_w = filter_positive_pairs(G, H, pairs_w, w_weighted)
    pos_score_w = score_pairs(G, H, pos_pairs_w, w_weighted)

    res_weighted = DevMatchCaseResult(
        name="blockswap_weighted_identity",
        seed=seed,
        G=G,
        H=H,
        truth_path_G=[idxG[s] for s in A],
        truth_path_H=[idxH[s] for s in A],
        truth_pairs=truth_pairs_A,
        truth_score=truth_score_A,
        found_pairs=pairs_w,
        found_score=float(score_w),
        found_pos_pairs=pos_pairs_w,
        found_pos_score=float(pos_score_w),
        truth_includes_root_G=(idxG[A[0]] == 0),
        truth_includes_leaf_G=_is_leaf(G, idxG[A[-1]]),
        truth_includes_root_H=(idxH[A[0]] == 0),
        truth_includes_leaf_H=_is_leaf(H, idxH[A[-1]]),
        weight_map=weight_map_weighted,
        notes="Expected optimal match: A-block (high weights)",
    )

    # Unweighted identity: count matches
    # Expected truth: match B symbols (length 4)
    truth_pairs_B = [(idxG[s], idxH[s]) for s in B]
    truth_score_B = float(len(B))  # unweighted
    pairs_u, score_u = run_exact_match(G, H, unweighted_identity_w)
    pos_pairs_u = filter_positive_pairs(G, H, pairs_u, unweighted_identity_w)
    pos_score_u = score_pairs(G, H, pos_pairs_u, unweighted_identity_w)

    # For plotting, reuse weight_map_weighted (so nodes still show weights).
    res_unweighted = DevMatchCaseResult(
        name="blockswap_unweighted_identity",
        seed=seed,
        G=G,
        H=H,
        truth_path_G=[idxG[s] for s in B],
        truth_path_H=[idxH[s] for s in B],
        truth_pairs=truth_pairs_B,
        truth_score=truth_score_B,
        found_pairs=pairs_u,
        found_score=float(score_u),
        found_pos_pairs=pos_pairs_u,
        found_pos_score=float(pos_score_u),
        truth_includes_root_G=(idxG[B[0]] == 0),
        truth_includes_leaf_G=_is_leaf(G, idxG[B[-1]]),
        truth_includes_root_H=(idxH[B[0]] == 0),
        truth_includes_leaf_H=_is_leaf(H, idxH[B[-1]]),
        weight_map=weight_map_weighted,
        notes="Expected optimal match: B-block (longer length)",
    )

    return DevCompareResult(
        name="weighted_vs_unweighted_blockswap",
        weighted=res_weighted,
        unweighted=res_unweighted,
    )


# ---------------------------------------------------------------------
# Pretty-print + plotting (matplotlib-only; no networkx)
# ---------------------------------------------------------------------

def summarize_case(res: DevMatchCaseResult) -> Dict[str, Union[str, int, float, bool]]:
    """Return a small dict suitable for a DataFrame row."""
    return {
        "name": res.name,
        "seed": res.seed,
        "nG": int(res.G.vcount()),
        "nH": int(res.H.vcount()),
        "truth_score": float(res.truth_score),
        "found_score": float(res.found_score),
        "found_pos_score": float(res.found_pos_score),
        "pos_len": int(len(res.found_pos_pairs)),
        "truth_len": int(len(res.truth_pairs)),
        "path_ok": (res.found_pos_pairs == res.truth_pairs),
        "score_ok": (abs(float(res.found_pos_score) - float(res.truth_score)) <= 1e-6),
        "rootG": res.truth_includes_root_G,
        "leafG": res.truth_includes_leaf_G,
        "rootH": res.truth_includes_root_H,
        "leafH": res.truth_includes_leaf_H,
    }


def _layout_tree_coords(g: ig.Graph) -> np.ndarray:
    """
    Compute 2D coords for a directed tree using Reingold–Tilford layout.
    Returns array of shape (n,2).
    """
    root = _root_vertex(g)
    try:
        layout = g.layout_reingold_tilford(root=[root], mode="out")
    except Exception:
        # fallback
        layout = g.layout("rt")
    coords = np.asarray(layout.coords, dtype=float)
    # normalize y to be negative depth-ish (matplotlib has y up)
    if coords.shape[1] >= 2:
        coords[:, 1] = -coords[:, 1]
    return coords[:, :2]


def _discrete_colors(labels: Sequence[str]) -> Dict[str, Tuple[float, float, float, float]]:
    """Assign a distinct color per label using matplotlib's tab10/tab20."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    uniq = sorted(set(map(str, labels)))
    cmap = cm.get_cmap("tab20", max(1, len(uniq)))
    out: Dict[str, Tuple[float, float, float, float]] = {}
    for i, lab in enumerate(uniq):
        out[lab] = cmap(i)
    return out


def plot_case(
    res: DevMatchCaseResult,
    *,
    title: Optional[str] = None,
    show_vertex_ids: bool = True,
    show_weights: bool = True,
    figsize: Tuple[float, float] = (12, 5),
) -> None:
    """
    Plot G and H side-by-side:
    - fill color by label
    - node text shows (id, weight) by default
    - node border:
        red = matched (positive under w)
        blue = truth but not matched
        black = otherwise
    - truth path edges colored blue
    """
    import matplotlib.pyplot as plt

    def plot_one(ax, g: ig.Graph, truth_path: Sequence[int], matched_vertices: Iterable[int], title: str) -> None:
        coords = _layout_tree_coords(g)
        labels = [str(x) for x in g.vs["label"]]
        weights = g.vs["weight"] if "weight" in g.vs.attributes() else [0.0] * g.vcount()
        color_map = _discrete_colors(labels)

        # edges
        for e in g.es:
            u = int(e.source)
            v = int(e.target)
            ax.plot([coords[u, 0], coords[v, 0]], [coords[u, 1], coords[v, 1]], linewidth=1.0, alpha=0.5)

        # truth path edges in blue
        truth_set = set(map(int, truth_path))
        for a, b in zip(truth_path[:-1], truth_path[1:]):
            ua = int(a)
            vb = int(b)
            ax.plot([coords[ua, 0], coords[vb, 0]], [coords[ua, 1], coords[vb, 1]], linewidth=2.5, alpha=0.9)

        # nodes
        matched_set = set(map(int, matched_vertices))

        for i in range(g.vcount()):
            lab = labels[i]
            face = color_map[lab]
            if i in matched_set:
                edgecolor = "red"
                lw = 3.0
            elif i in truth_set:
                edgecolor = "dodgerblue"
                lw = 3.0
            else:
                edgecolor = "black"
                lw = 1.0
            ax.scatter([coords[i, 0]], [coords[i, 1]], s=220, edgecolors=edgecolor, linewidths=lw, facecolors=[face], zorder=3)

            txt = ""
            if show_vertex_ids:
                txt += str(i)
            if show_weights:
                wv = float(weights[i]) if weights is not None else 0.0
                txt += f" ({wv:g})" if txt else f"{wv:g}"
            if txt:
                ax.text(coords[i, 0], coords[i, 1], txt, ha="center", va="center", fontsize=8, zorder=4)

        ax.set_title(title)
        ax.axis("off")

    matchG = [u for (u, _v) in res.found_pos_pairs]
    matchH = [v for (_u, v) in res.found_pos_pairs]

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_one(axes[0], res.G, res.truth_path_G, matchG, "G")
    plot_one(axes[1], res.H, res.truth_path_H, matchH, "H")

    if title is None:
        title = f"{res.name}\ntruth={res.truth_score:g}  found={res.found_pos_score:g}"
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
