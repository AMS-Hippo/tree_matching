"""subgraph_utils.py

Small helpers used by the `SYNTAX_subgraph_matching.ipynb` demo notebook.

These are intentionally *not* imported by the core DP module.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def add_monotone_timestamps(
    G: Any,
    *,
    ts_attr: str = "timestamp",
    root: int = 0,
    step: float = 1.0,
    jitter: float = 0.1,
    seed: Optional[int] = 0,
) -> None:
    """Attach a timestamp attribute to an igraph directed tree.

    The timestamps are strictly increasing along every root->leaf path.

    Notes
    -----
    - This is just a demo helper. Real applications should use your true timestamps.
    - Requires igraph.Graph input.
    """
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")
    if not G.is_directed():
        raise ValueError("G must be directed (parent -> child)")

    n = int(G.vcount())
    rng = np.random.default_rng(seed)

    ts = np.full(n, np.nan, dtype=float)
    ts[root] = 0.0

    # DFS stack
    stack = [root]
    while stack:
        u = stack.pop()
        t_u = float(ts[u])
        # children are OUT-neighbors in parent->child convention
        kids = G.neighbors(u, mode="OUT")
        # deterministic order before jitter so runs are reproducible
        kids = sorted(int(x) for x in kids)
        for v in kids:
            # ensure strictly increasing; add sibling-dependent offset for uniqueness
            ts[v] = t_u + float(step) + float(rng.uniform(0.0, jitter))
        # push after assignment
        for v in reversed(kids):
            stack.append(int(v))

    if not np.isfinite(ts).all():
        missing = np.where(~np.isfinite(ts))[0].tolist()
        raise ValueError(f"Failed to assign timestamps to some vertices: {missing[:20]}")

    G.vs[ts_attr] = ts.tolist()


def check_timestamps_consistent(G: Any, *, ts_attr: str = "timestamp") -> bool:
    """Return True iff parent timestamps are <= child timestamps on all edges."""
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")
    if ts_attr not in G.vs.attributes():
        raise KeyError(f"Missing vertex attribute '{ts_attr}'")

    ts = list(G.vs[ts_attr])
    for a, b in G.get_edgelist():
        if ts[a] is None or ts[b] is None:
            return False
        if ts[a] > ts[b]:
            return False
    return True


def pick_subtree_root(
    G: Any,
    *,
    root: int = 0,
    min_depth: int = 1,
    seed: Optional[int] = 0,
) -> int:
    """Pick a random vertex with depth >= min_depth (useful for subtree demos)."""
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")

    depths = G.distances(source=root)[0]
    candidates = [i for i, d in enumerate(depths) if d is not None and int(d) >= int(min_depth)]
    if not candidates:
        return int(root)
    rng = np.random.default_rng(seed)
    return int(rng.choice(candidates))


def pairs_to_vertices(pairs: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[int]]:
    """Split matched (u,v) pairs into per-tree vertex lists."""
    uu: List[int] = []
    vv: List[int] = []
    for u, v in pairs:
        uu.append(int(u))
        vv.append(int(v))
    return uu, vv


def order_pairs_by_timestamp(
    G: Any,
    pairs: Sequence[Tuple[int, int]],
    *,
    ts_attr: str = "timestamp",
    side: str = "G",
) -> List[Tuple[int, int]]:
    """Return pairs sorted by timestamps on one side (default: G).

    This is handy for giving a deterministic numbering in plots.
    """
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")
    if ts_attr not in G.vs.attributes():
        return list(pairs)

    ts = list(G.vs[ts_attr])
    if side.upper() == "G":
        return sorted(pairs, key=lambda p: float(ts[int(p[0])]))
    return list(pairs)


def annotate_match_indices(
    G: Any,
    matched_vertices: Sequence[int],
    *,
    label_attr: str = "match_idx",
    start: int = 1,
) -> None:
    """Set a per-vertex label attribute giving 1..k for matched vertices."""
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")

    n = int(G.vcount())
    lab = [""] * n
    for i, v in enumerate(matched_vertices, start=start):
        v = int(v)
        if 0 <= v < n:
            lab[v] = str(i)
    G.vs[label_attr] = lab


def to_float_timestamp_list(G: Any, *, ts_attr: str = "timestamp") -> List[float]:
    """Return timestamps as a float list (for passing timestamps externally)."""
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")
    if ts_attr not in G.vs.attributes():
        raise KeyError(f"Missing vertex attribute '{ts_attr}'")
    return [float(x) for x in G.vs[ts_attr]]


def subtree_size(G: Any, *, root: int) -> int:
    """Return number of vertices in the rooted subtree at `root` (igraph)."""
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")
    if not G.is_directed():
        raise ValueError("G must be directed (parent -> child)")

    verts = G.subcomponent(root, mode="OUT")  # descendants including root
    return int(len(verts))


def pick_small_subtree_root(
    G: Any,
    *,
    root: int = 0,
    min_depth: int = 1,
    max_size: int = 60,
    seed: Optional[int] = 0,
    max_tries: int = 2000,
) -> int:
    """Pick a vertex (depth>=min_depth) whose descendant subtree is <= max_size.

    Useful for demos of the unordered matcher, which can be much slower.

    Falls back to `pick_subtree_root` if no such vertex is found.
    """
    import igraph as ig  # type: ignore

    if not isinstance(G, ig.Graph):
        raise TypeError("G must be an igraph.Graph")

    depths = G.distances(source=root)[0]
    candidates = [i for i, d in enumerate(depths) if d is not None and int(d) >= int(min_depth)]
    if not candidates:
        return int(root)

    rng = np.random.default_rng(seed)

    for _ in range(int(max_tries)):
        u = int(rng.choice(candidates))
        if subtree_size(G, root=u) <= int(max_size):
            return u

    # fallback
    return pick_subtree_root(G, root=root, min_depth=min_depth, seed=seed)
