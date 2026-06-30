from __future__ import annotations

"""fast_subgraph_match.py

Fast exact DP for *timestamp-ordered* subtree matching.

This module is the subtree analogue of `path_matcher/fast_match.py`.

Problem (ordered rooted-tree alignment)
--------------------------------------
Given two rooted trees G and H (directed parent -> child) and a nonnegative
vertex-pair score w( label(u), label(v) ), we want a maximum-score set of
matched vertex pairs subject to:

  1. All matched pairs lie in chosen subtrees G[rG] and H[rH].
  2. Ancestor/descendant is preserved: if u is a descendant of a in G and
     v is matched to b in H, then v must be a descendant of b.
  3. *Child order* is preserved ("ordered trees"). Here the order comes from
     timestamps: children are ordered by increasing timestamp.

With child order, the DP runs in O(|G| |H|) time (more precisely,
O(|G||H| + sum_{u,v} deg(u)deg(v)) which factorizes to O(|G||H|)).

Specialized scoring families
----------------------------
To match the performance philosophy of `fast_match.py`, we provide specialized
implementations for two common scoring families:

  - mode='equality':   score is token_weight[t] if label_u == label_v else 0
  - mode='overlap' :   score is max(token_weight[t] for t in label_u ∩ label_v)

These avoid calling a Python weight function inside the DP.

Timestamps
----------
This matcher **relies on a left-to-right order of children**, derived from
timestamps.

Ways to supply timestamps:

  (A) If you pass igraph.Graph inputs, store timestamps as a vertex attribute
      and set `ts_field='your_attr'` (and optionally `order='timestamp'`).

  (B) If you have timestamps externally (array/dict), pass them via
      `timestamps=` to `.fit(...)` / `.predict(...)` / `.encode_tree(...)`.
      We will temporarily attach them and reorder internally.

  (C) If you pass TreeData, you can either:
        - provide it already in timestamp-topological order, or
        - pass `timestamps=` so we can reorder the TreeData.

API
---
The main user-facing class is `FastSubtreeMatcher`, mirroring
`path_matcher.fast_match.FastTreePathMatcher`.

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# -----------------------------------------------------------------------------
# Imports from the existing matcher package
# -----------------------------------------------------------------------------

try:  # package layout
    from path_matcher.tree_data import TreeData
    from path_matcher.igraph_io import igraph_to_treedata
    from path_matcher.fast_match import (
        HAVE_NUMBA,
        njit,  # type: ignore
        FastLabelEncoder,
        EncodedTreeEquality,
        EncodedTreeOverlap,
    )
except Exception:  # pragma: no cover
    # loose-file / alternate layout
    from tree_data import TreeData  # type: ignore
    from igraph_io import igraph_to_treedata  # type: ignore
    from fast_match import (  # type: ignore
        HAVE_NUMBA,
        njit,  # type: ignore
        FastLabelEncoder,
        EncodedTreeEquality,
        EncodedTreeOverlap,
    )


LabelGetter = Optional[Callable[[Any], Any]]


# -----------------------------------------------------------------------------
# Encoded trees for subtree matching
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodedSubtreeEquality:
    tree: TreeData
    label_ids: np.ndarray  # (n,) int32
    child_offsets: np.ndarray  # (n+1,) int32
    child_flat: np.ndarray  # (n-1,) int32


@dataclass(frozen=True)
class EncodedSubtreeOverlap:
    tree: TreeData
    offsets: np.ndarray  # (n+1,) int32   per-vertex label token offsets
    flat_token_ids: np.ndarray  # concatenation of per-vertex sorted unique token ids
    child_offsets: np.ndarray  # (n+1,) int32
    child_flat: np.ndarray  # (n-1,) int32


EncodedSubtree = Union[EncodedSubtreeEquality, EncodedSubtreeOverlap]


@dataclass(frozen=True)
class FastSubtreeAlignmentResult:
    pairs_internal: List[Tuple[int, int]]
    score: float
    roots_internal: Tuple[int, int]
    A: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Helpers: TreeData detection, (re)ordering via timestamps, and children arrays
# -----------------------------------------------------------------------------


def _looks_like_treedata(x: Any) -> bool:
    return hasattr(x, "parent") and hasattr(x, "label") and hasattr(x, "orig_index")


def _build_children_arrays(parent: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (child_offsets, child_flat) encoding the ordered children lists.

    Assumes TreeData ordering invariant: parent[i] < i for all i>0.

    With timestamp-topological ordering, children are already in timestamp order
    when sorted by internal index, so this construction yields the desired order.
    """
    parent = np.asarray(parent, dtype=np.int64)
    n = int(parent.shape[0])
    if n <= 0:
        raise ValueError("Tree is empty")

    deg = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        p = int(parent[i])
        deg[p] += 1

    offsets = np.zeros(n + 1, dtype=np.int32)
    # prefix sum
    s = 0
    for i in range(n):
        offsets[i] = np.int32(s)
        s += int(deg[i])
    offsets[n] = np.int32(s)

    flat = np.empty(max(0, n - 1), dtype=np.int32)
    cursor = offsets.copy()
    for child in range(1, n):
        p = int(parent[child])
        j = int(cursor[p])
        flat[j] = np.int32(child)
        cursor[p] = np.int32(j + 1)

    return offsets, flat


def _heap_topo_order_from_children(children: List[List[int]], key: Sequence[Any], root: int = 0) -> List[int]:
    """Heap-based topological order (parent before child), with key tie-break.

    This mirrors path_matcher.igraph_io._topological_order_tree(..., key=...).
    """
    import heapq

    n = len(children)
    heap: List[Tuple[Any, int]] = []
    heapq.heappush(heap, (key[root], root))
    order: List[int] = []
    while heap:
        _, u = heapq.heappop(heap)
        order.append(u)
        for v in children[u]:
            heapq.heappush(heap, (key[v], v))
    if len(order) != n:
        raise ValueError("Timestamp/topological ordering failed (unexpected)")
    return order


def reorder_treedata_by_timestamp(tree: TreeData, timestamps: Union[Sequence[Any], Mapping[Any, Any]]) -> TreeData:
    """Return a new TreeData reordered by timestamps (parent before child).

    Parameters
    ----------
    tree:
        Existing TreeData.
    timestamps:
        Either:
          - sequence of length >= max(orig_index)+1 addressed by original ids, or
          - mapping original_id -> timestamp.

    Notes
    -----
    The returned TreeData satisfies the usual invariant parent[i] < i, and
    children are ordered by increasing timestamp.
    """
    parent = np.asarray(tree.parent, dtype=np.int64)
    orig = np.asarray(tree.orig_index, dtype=np.int64)
    n = int(parent.shape[0])

    # Build children lists in the *old* internal indexing.
    children: List[List[int]] = [[] for _ in range(n)]
    for i in range(1, n):
        p = int(parent[i])
        children[p].append(i)

    # Key by old internal index.
    if isinstance(timestamps, Mapping):
        key = [timestamps.get(int(orig[i])) for i in range(n)]
    else:
        ts_seq = list(timestamps)
        key = [ts_seq[int(orig[i])] for i in range(n)]

    if any(k is None for k in key):
        raise ValueError("Missing timestamp for at least one vertex (None encountered)")

    order_old = _heap_topo_order_from_children(children, key, root=0)

    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[np.asarray(order_old, dtype=np.int64)] = np.arange(n, dtype=np.int64)

    new_parent = np.full(n, -1, dtype=np.int64)
    for new_i, old_i in enumerate(order_old):
        p_old = int(parent[old_i])
        new_parent[new_i] = -1 if p_old == -1 else int(old_to_new[p_old])

    # Remap labels and orig_index
    labels_new = [tree.label[old_i] for old_i in order_old]
    orig_new = orig[np.asarray(order_old, dtype=np.int64)]

    return TreeData(
        parent=new_parent.astype(np.int32, copy=False),
        label=labels_new,
        orig_index=orig_new.astype(np.int32, copy=False),
    )


def _as_treedata(
    G: Any,
    *,
    phi_name: str,
    order: str,
    ts_field: Optional[str],
    strict_tree: bool,
    timestamps: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
) -> TreeData:
    """Convert G -> TreeData, optionally forcing timestamp ordering."""
    if _looks_like_treedata(G):
        td: TreeData = G  # type: ignore[assignment]
        if timestamps is not None:
            td = reorder_treedata_by_timestamp(td, timestamps)
        return td

    # igraph path
    if timestamps is not None:
        # Attach timestamps under a temporary attribute name (avoid mutating the input).
        try:
            import igraph as ig  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError("igraph is required when passing non-TreeData inputs") from e

        if not isinstance(G, ig.Graph):  # type: ignore
            raise TypeError("When passing timestamps=, G must be a TreeData or igraph.Graph")

        n = int(G.vcount())
        if isinstance(timestamps, Mapping):
            ts_list = [timestamps.get(i) for i in range(n)]
        else:
            ts_seq = list(timestamps)
            if len(ts_seq) != n:
                raise ValueError(f"timestamps must have length {n} for this igraph input, got {len(ts_seq)}")
            ts_list = ts_seq

        if any(t is None for t in ts_list):
            raise ValueError("timestamps contains None values")

        tmp = "__tmp_ts__"
        G2 = G.copy()
        G2.vs[tmp] = ts_list
        return igraph_to_treedata(G2, phi_name=phi_name, order="timestamp", ts_field=tmp, strict_tree=strict_tree)

    # Standard conversion: user can pass ts_field/order to choose timestamp ordering.
    return igraph_to_treedata(G, phi_name=phi_name, order=order, ts_field=ts_field, strict_tree=strict_tree)


def _orig_to_internal(tree: TreeData) -> Dict[int, int]:
    orig = np.asarray(tree.orig_index, dtype=np.int64)
    return {int(o): int(i) for i, o in enumerate(orig)}


# -----------------------------------------------------------------------------
# Numba DP kernels
# -----------------------------------------------------------------------------


if HAVE_NUMBA:

    @njit(cache=True)
    def _dp_subtree_equality_numba(
        child_offG: np.ndarray,
        child_flatG: np.ndarray,
        idsG: np.ndarray,
        child_offH: np.ndarray,
        child_flatH: np.ndarray,
        idsH: np.ndarray,
        weight_by_id: np.ndarray,
        max_deg_H: int,
    ) -> np.ndarray:
        n = idsG.shape[0]
        m = idsH.shape[0]
        A = np.zeros((n, m), dtype=np.float32)

        prev = np.zeros(max_deg_H + 1, dtype=np.float32)
        curr = np.zeros(max_deg_H + 1, dtype=np.float32)

        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])
            idu = int(idsG[u])

            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])
                l = off_v_end - off_v

                # root score
                w_uv = 0.0
                if idu >= 0 and idu == int(idsH[v]):
                    w_uv = float(weight_by_id[idu])

                # weighted LCS over children lists
                for j in range(l + 1):
                    prev[j] = 0.0

                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    curr[0] = 0.0
                    for j in range(1, l + 1):
                        cv = int(child_flatH[off_v + (j - 1)])
                        opt1 = prev[j]
                        opt2 = curr[j - 1]
                        opt3 = prev[j - 1] + A[cu, cv]
                        val = opt1
                        if opt2 > val:
                            val = opt2
                        if opt3 > val:
                            val = opt3
                        curr[j] = val
                    tmp = prev
                    prev = curr
                    curr = tmp

                lcs_val = prev[l]
                M = w_uv + lcs_val

                # skip-root options
                best1 = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    val = float(A[cu, v])
                    if val > best1:
                        best1 = val

                best2 = 0.0
                for j_idx in range(off_v, off_v_end):
                    cv = int(child_flatH[j_idx])
                    val = float(A[u, cv])
                    if val > best2:
                        best2 = val

                val = M
                if best1 > val:
                    val = best1
                if best2 > val:
                    val = best2
                A[u, v] = val

        return A


    @njit(cache=True)
    def _score_overlap_vertex_pair_numba(
        offG: np.ndarray,
        flatG: np.ndarray,
        u: int,
        offH: np.ndarray,
        flatH: np.ndarray,
        v: int,
        weight_by_id: np.ndarray,
    ) -> float:
        i = int(offG[u])
        i_end = int(offG[u + 1])
        j = int(offH[v])
        j_end = int(offH[v + 1])
        best = 0.0
        while i < i_end and j < j_end:
            a = int(flatG[i])
            b = int(flatH[j])
            if a == b:
                w = float(weight_by_id[a])
                if w > best:
                    best = w
                i += 1
                j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        return best


    @njit(cache=True)
    def _dp_subtree_overlap_numba(
        child_offG: np.ndarray,
        child_flatG: np.ndarray,
        offG: np.ndarray,
        flatG: np.ndarray,
        child_offH: np.ndarray,
        child_flatH: np.ndarray,
        offH: np.ndarray,
        flatH: np.ndarray,
        weight_by_id: np.ndarray,
        max_deg_H: int,
    ) -> np.ndarray:
        n = child_offG.shape[0] - 1
        m = child_offH.shape[0] - 1
        A = np.zeros((n, m), dtype=np.float32)

        prev = np.zeros(max_deg_H + 1, dtype=np.float32)
        curr = np.zeros(max_deg_H + 1, dtype=np.float32)

        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])

            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])
                l = off_v_end - off_v

                w_uv = _score_overlap_vertex_pair_numba(offG, flatG, u, offH, flatH, v, weight_by_id)

                for j in range(l + 1):
                    prev[j] = 0.0

                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    curr[0] = 0.0
                    for j in range(1, l + 1):
                        cv = int(child_flatH[off_v + (j - 1)])
                        opt1 = prev[j]
                        opt2 = curr[j - 1]
                        opt3 = prev[j - 1] + A[cu, cv]
                        val = opt1
                        if opt2 > val:
                            val = opt2
                        if opt3 > val:
                            val = opt3
                        curr[j] = val
                    tmp = prev
                    prev = curr
                    curr = tmp

                lcs_val = prev[l]
                M = w_uv + lcs_val

                best1 = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    val = float(A[cu, v])
                    if val > best1:
                        best1 = val

                best2 = 0.0
                for j_idx in range(off_v, off_v_end):
                    cv = int(child_flatH[j_idx])
                    val = float(A[u, cv])
                    if val > best2:
                        best2 = val

                val = M
                if best1 > val:
                    val = best1
                if best2 > val:
                    val = best2
                A[u, v] = val

        return A

else:  # pragma: no cover

    def _dp_subtree_equality_numba(child_offG, child_flatG, idsG, child_offH, child_flatH, idsH, weight_by_id, max_deg_H):
        n = int(idsG.shape[0])
        m = int(idsH.shape[0])
        A = np.zeros((n, m), dtype=np.float32)
        prev = np.zeros(max_deg_H + 1, dtype=np.float32)
        curr = np.zeros(max_deg_H + 1, dtype=np.float32)
        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])
            idu = int(idsG[u])
            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])
                l = off_v_end - off_v
                w_uv = float(weight_by_id[idu]) if (idu >= 0 and idu == int(idsH[v])) else 0.0
                prev[: l + 1] = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    curr[0] = 0.0
                    for j in range(1, l + 1):
                        cv = int(child_flatH[off_v + (j - 1)])
                        opt1 = float(prev[j])
                        opt2 = float(curr[j - 1])
                        opt3 = float(prev[j - 1] + A[cu, cv])
                        curr[j] = max(opt1, opt2, opt3)
                    prev, curr = curr, prev
                M = w_uv + float(prev[l])
                best1 = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    best1 = max(best1, float(A[cu, v]))
                best2 = 0.0
                for j_idx in range(off_v, off_v_end):
                    cv = int(child_flatH[j_idx])
                    best2 = max(best2, float(A[u, cv]))
                A[u, v] = max(M, best1, best2)
        return A

    def _score_overlap_vertex_pair_numba(offG, flatG, u, offH, flatH, v, weight_by_id):
        i = int(offG[u]); i_end = int(offG[u + 1])
        j = int(offH[v]); j_end = int(offH[v + 1])
        best = 0.0
        while i < i_end and j < j_end:
            a = int(flatG[i]); b = int(flatH[j])
            if a == b:
                best = max(best, float(weight_by_id[a]))
                i += 1; j += 1
            elif a < b:
                i += 1
            else:
                j += 1
        return best

    def _dp_subtree_overlap_numba(child_offG, child_flatG, offG, flatG, child_offH, child_flatH, offH, flatH, weight_by_id, max_deg_H):
        n = int(child_offG.shape[0] - 1)
        m = int(child_offH.shape[0] - 1)
        A = np.zeros((n, m), dtype=np.float32)
        prev = np.zeros(max_deg_H + 1, dtype=np.float32)
        curr = np.zeros(max_deg_H + 1, dtype=np.float32)
        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])
            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])
                l = off_v_end - off_v
                w_uv = float(_score_overlap_vertex_pair_numba(offG, flatG, u, offH, flatH, v, weight_by_id))
                prev[: l + 1] = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    curr[0] = 0.0
                    for j in range(1, l + 1):
                        cv = int(child_flatH[off_v + (j - 1)])
                        curr[j] = max(float(prev[j]), float(curr[j - 1]), float(prev[j - 1] + A[cu, cv]))
                    prev, curr = curr, prev
                M = w_uv + float(prev[l])
                best1 = 0.0
                for i_idx in range(off_u, off_u_end):
                    cu = int(child_flatG[i_idx])
                    best1 = max(best1, float(A[cu, v]))
                best2 = 0.0
                for j_idx in range(off_v, off_v_end):
                    cv = int(child_flatH[j_idx])
                    best2 = max(best2, float(A[u, cv]))
                A[u, v] = max(M, best1, best2)
        return A


# -----------------------------------------------------------------------------
# Python-side scoring helpers (used in traceback)
# -----------------------------------------------------------------------------


def _score_uv_equality_py(idsG: np.ndarray, u: int, idsH: np.ndarray, v: int, weight_by_id: np.ndarray) -> float:
    idu = int(idsG[u])
    if idu >= 0 and idu == int(idsH[v]):
        return float(weight_by_id[idu])
    return 0.0


def _score_uv_overlap_py(offG: np.ndarray, flatG: np.ndarray, u: int, offH: np.ndarray, flatH: np.ndarray, v: int, weight_by_id: np.ndarray) -> float:
    i = int(offG[u])
    i_end = int(offG[u + 1])
    j = int(offH[v])
    j_end = int(offH[v + 1])
    best = 0.0
    while i < i_end and j < j_end:
        a = int(flatG[i])
        b = int(flatH[j])
        if a == b:
            w = float(weight_by_id[a])
            if w > best:
                best = w
            i += 1
            j += 1
        elif a < b:
            i += 1
        else:
            j += 1
    return best


# -----------------------------------------------------------------------------
# Traceback / reconstruction (Python; called for one pair at a time)
# -----------------------------------------------------------------------------


def _child_list(child_off: np.ndarray, child_flat: np.ndarray, u: int) -> np.ndarray:
    a = int(child_off[u])
    b = int(child_off[u + 1])
    return child_flat[a:b]


def _lcs_children_pairs(
    children_u: np.ndarray,
    children_v: np.ndarray,
    A: np.ndarray,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Compute weighted LCS on child lists and return (value, chosen child pairs)."""
    k = int(children_u.shape[0])
    l = int(children_v.shape[0])
    if k == 0 or l == 0:
        return 0.0, []

    F = np.zeros((k + 1, l + 1), dtype=np.float32)
    P = np.zeros((k + 1, l + 1), dtype=np.uint8)  # 1=up, 2=left, 3=diag(match)

    for i in range(1, k + 1):
        cu = int(children_u[i - 1])
        for j in range(1, l + 1):
            cv = int(children_v[j - 1])
            opt1 = F[i - 1, j]
            opt2 = F[i, j - 1]
            opt3 = F[i - 1, j - 1] + A[cu, cv]
            # tie-break: prefer diag, then left, then up
            if opt3 >= opt2 and opt3 >= opt1:
                F[i, j] = opt3
                P[i, j] = 3
            elif opt2 >= opt1:
                F[i, j] = opt2
                P[i, j] = 2
            else:
                F[i, j] = opt1
                P[i, j] = 1

    pairs_rev: List[Tuple[int, int]] = []
    i, j = k, l
    while i > 0 and j > 0:
        p = int(P[i, j])
        if p == 3:
            pairs_rev.append((int(children_u[i - 1]), int(children_v[j - 1])))
            i -= 1
            j -= 1
        elif p == 2:
            j -= 1
        else:
            i -= 1
    pairs_rev.reverse()
    return float(F[k, l]), pairs_rev


def _reconstruct_matching(
    encG: EncodedSubtree,
    encH: EncodedSubtree,
    A: np.ndarray,
    *,
    root_u: int,
    root_v: int,
    weight_by_id: np.ndarray,
    eps: float = 1e-6,
) -> List[Tuple[int, int]]:
    """Reconstruct one optimal matching attaining A[root_u, root_v]."""

    child_offG = encG.child_offsets
    child_flatG = encG.child_flat
    child_offH = encH.child_offsets
    child_flatH = encH.child_flat

    if isinstance(encG, EncodedSubtreeEquality):
        idsG = encG.label_ids
        idsH = encH.label_ids  # type: ignore[assignment]

        def score_uv(u: int, v: int) -> float:
            return _score_uv_equality_py(idsG, u, idsH, v, weight_by_id)

    else:
        offG = encG.offsets
        flatG = encG.flat_token_ids
        offH = encH.offsets  # type: ignore[assignment]
        flatH = encH.flat_token_ids  # type: ignore[assignment]

        def score_uv(u: int, v: int) -> float:
            return _score_uv_overlap_py(offG, flatG, u, offH, flatH, v, weight_by_id)

    # Tie-break order: rooted > skip-u > skip-v
    def reconstruct_free(u: int, v: int) -> List[Tuple[int, int]]:
        best = float(A[u, v])

        # children
        cu = _child_list(child_offG, child_flatG, u)
        cv = _child_list(child_offH, child_flatH, v)

        # skip-u
        best1 = 0.0
        arg1 = -1
        for x in cu:
            val = float(A[int(x), v])
            if val > best1 + eps:
                best1 = val
                arg1 = int(x)

        # skip-v
        best2 = 0.0
        arg2 = -1
        for y in cv:
            val = float(A[u, int(y)])
            if val > best2 + eps:
                best2 = val
                arg2 = int(y)

        # rooted
        lcs_val, child_pairs = _lcs_children_pairs(cu, cv, A)
        M = score_uv(u, v) + lcs_val

        # choose with deterministic tie-break
        best_val = M
        choice = 3  # rooted
        if best1 > best_val + eps or (abs(best1 - best_val) <= eps and choice != 3):
            best_val = best1
            choice = 1
        if best2 > best_val + eps:
            best_val = best2
            choice = 2

        # If numerical noise makes A[u,v] slightly larger, trust best.
        # Otherwise, ensure we follow a branch that matches A[u,v] within eps.
        if abs(best - best_val) > 1e-4 and best > best_val + 1e-4:
            # Fall back: prefer a branch that matches A[u,v] closely.
            # (This should be rare and only due to float32 accumulation.)
            if abs(best - M) <= 1e-4:
                choice = 3
            elif abs(best - best1) <= 1e-4:
                choice = 1
            elif abs(best - best2) <= 1e-4:
                choice = 2

        if choice == 3 and M + 1e-4 >= best:
            out: List[Tuple[int, int]] = [(u, v)]
            for (a, b) in child_pairs:
                out.extend(reconstruct_free(a, b))
            return out
        if choice == 1 and arg1 >= 0 and best1 + 1e-4 >= best:
            return reconstruct_free(arg1, v)
        if choice == 2 and arg2 >= 0 and best2 + 1e-4 >= best:
            return reconstruct_free(u, arg2)

        # Last-resort deterministic fallback (should be unreachable).
        if M >= best1 and M >= best2:
            out = [(u, v)]
            for (a, b) in child_pairs:
                out.extend(reconstruct_free(a, b))
            return out
        if best1 >= best2 and arg1 >= 0:
            return reconstruct_free(arg1, v)
        if arg2 >= 0:
            return reconstruct_free(u, arg2)
        return []

    return reconstruct_free(int(root_u), int(root_v))


# -----------------------------------------------------------------------------
# Low-level alignment function
# -----------------------------------------------------------------------------


def align_subtrees_fast_encoded(
    G: EncodedSubtree,
    H: EncodedSubtree,
    *,
    weight_by_id: np.ndarray,
    root_u: int = 0,
    root_v: int = 0,
    return_matrix: bool = False,
) -> FastSubtreeAlignmentResult:
    """Compute optimal ordered-subtree matching between G[root_u] and H[root_v].

    Parameters
    ----------
    G, H:
        EncodedSubtree* objects produced by FastSubtreeMatcher.encode_tree(...).
    weight_by_id:
        Array mapping integerized token ids -> match weight.
    root_u, root_v:
        Internal node indices (0..n-1) for the subtree roots.
    return_matrix:
        If True, include the full DP table A in the result.

    Returns
    -------
    FastSubtreeAlignmentResult with internal-index pairs.
    """

    if isinstance(G, EncodedSubtreeEquality) and isinstance(H, EncodedSubtreeEquality):
        degH = (H.child_offsets[1:] - H.child_offsets[:-1]).astype(np.int32)
        max_deg_H = int(degH.max()) if degH.size else 0
        A = _dp_subtree_equality_numba(
            np.asarray(G.child_offsets, dtype=np.int32),
            np.asarray(G.child_flat, dtype=np.int32),
            np.asarray(G.label_ids, dtype=np.int32),
            np.asarray(H.child_offsets, dtype=np.int32),
            np.asarray(H.child_flat, dtype=np.int32),
            np.asarray(H.label_ids, dtype=np.int32),
            np.asarray(weight_by_id, dtype=np.float32),
            max_deg_H,
        )
        score = float(A[int(root_u), int(root_v)])
        pairs = _reconstruct_matching(G, H, A, root_u=int(root_u), root_v=int(root_v), weight_by_id=np.asarray(weight_by_id, dtype=np.float32))
        if return_matrix:
            return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)), A=A)
        return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)))

    if isinstance(G, EncodedSubtreeOverlap) and isinstance(H, EncodedSubtreeOverlap):
        degH = (H.child_offsets[1:] - H.child_offsets[:-1]).astype(np.int32)
        max_deg_H = int(degH.max()) if degH.size else 0
        A = _dp_subtree_overlap_numba(
            np.asarray(G.child_offsets, dtype=np.int32),
            np.asarray(G.child_flat, dtype=np.int32),
            np.asarray(G.offsets, dtype=np.int32),
            np.asarray(G.flat_token_ids, dtype=np.int32),
            np.asarray(H.child_offsets, dtype=np.int32),
            np.asarray(H.child_flat, dtype=np.int32),
            np.asarray(H.offsets, dtype=np.int32),
            np.asarray(H.flat_token_ids, dtype=np.int32),
            np.asarray(weight_by_id, dtype=np.float32),
            max_deg_H,
        )
        score = float(A[int(root_u), int(root_v)])
        pairs = _reconstruct_matching(G, H, A, root_u=int(root_u), root_v=int(root_v), weight_by_id=np.asarray(weight_by_id, dtype=np.float32))
        if return_matrix:
            return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)), A=A)
        return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)))

    raise TypeError("Encoded tree types do not match. Use equality+equality or overlap+overlap.")


# -----------------------------------------------------------------------------
# High-level matcher class (mirrors FastTreePathMatcher)
# -----------------------------------------------------------------------------


class FastSubtreeMatcher:
    """Fast exact ordered-subtree matcher for equality/overlap weights.

    Parameters
    ----------
    mode:
        "equality" or "overlap".
    label_getter, token_weights, default_weight:
        Passed to FastLabelEncoder (see path_matcher.fast_match).
    phi_name, order, ts_field, strict_tree:
        Passed through when converting igraph -> TreeData.
        To use timestamps stored on vertices, set `ts_field=...` and either
        `order='timestamp'` or `order='auto'` (auto uses timestamp when ts_field is given).
    encoder:
        Optional pre-built FastLabelEncoder.

    Notes
    -----
    - The DP assumes an *ordered* child list at each node. With timestamp ordering,
      this is induced by the global TreeData ordering.
    - For many pairwise comparisons, fit one encoder on your dataset and
      pre-encode each tree once (see `fit_encoder` / `encode_tree`).
    """

    def __init__(
        self,
        *,
        mode: str = "equality",
        label_getter: LabelGetter = None,
        token_weights: Optional[Mapping[Any, float]] = None,
        default_weight: float = 1.0,
        phi_name: str = "label",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        encoder: Optional[FastLabelEncoder] = None,
    ) -> None:
        self.mode = mode.lower().strip()
        self.label_getter = label_getter
        self.token_weights = dict(token_weights or {})
        self.default_weight = float(default_weight)
        self.phi_name = phi_name
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = strict_tree

        self.encoder = encoder or FastLabelEncoder(
            mode=self.mode,
            label_getter=self.label_getter,
            token_weights=self.token_weights,
            default_weight=self.default_weight,
        )

        self.treeG_: Optional[TreeData] = None
        self.treeH_: Optional[TreeData] = None
        self.encG_: Optional[EncodedSubtree] = None
        self.encH_: Optional[EncodedSubtree] = None

    def _to_tree(self, G: Any, *, timestamps: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None) -> TreeData:
        return _as_treedata(
            G,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
            timestamps=timestamps,
        )

    def fit_encoder(self, trees: Sequence[Any], *, timestamps_list: Optional[Sequence[Optional[Union[Sequence[Any], Mapping[Any, Any]]]]] = None) -> "FastSubtreeMatcher":
        if timestamps_list is None:
            td_trees = [self._to_tree(T) for T in trees]
        else:
            if len(timestamps_list) != len(trees):
                raise ValueError("timestamps_list must have the same length as trees")
            td_trees = [self._to_tree(T, timestamps=ts) for T, ts in zip(trees, timestamps_list)]
        self.encoder.fit_from_trees(td_trees)
        return self

    def encode_tree(self, G: Any, *, timestamps: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None) -> EncodedSubtree:
        tree = self._to_tree(G, timestamps=timestamps)
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([tree])

        base = self.encoder.transform_tree(tree)
        child_off, child_flat = _build_children_arrays(tree.parent)

        if isinstance(base, EncodedTreeEquality):
            return EncodedSubtreeEquality(
                tree=tree,
                label_ids=np.asarray(base.label_ids, dtype=np.int32),
                child_offsets=np.asarray(child_off, dtype=np.int32),
                child_flat=np.asarray(child_flat, dtype=np.int32),
            )
        elif isinstance(base, EncodedTreeOverlap):
            return EncodedSubtreeOverlap(
                tree=tree,
                offsets=np.asarray(base.offsets, dtype=np.int32),
                flat_token_ids=np.asarray(base.flat_token_ids, dtype=np.int32),
                child_offsets=np.asarray(child_off, dtype=np.int32),
                child_flat=np.asarray(child_flat, dtype=np.int32),
            )
        else:  # pragma: no cover
            raise TypeError("Unexpected encoded tree type")

    def fit(
        self,
        G: Any,
        H: Any,
        *,
        timestampsG: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
        timestampsH: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    ) -> "FastSubtreeMatcher":
        treeG = self._to_tree(G, timestamps=timestampsG)
        treeH = self._to_tree(H, timestamps=timestampsH)
        self.treeG_ = treeG
        self.treeH_ = treeH
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([treeG, treeH])
        self.encG_ = self.encode_tree(treeG)
        self.encH_ = self.encode_tree(treeH)
        return self

    def predict(
        self,
        G: Any = None,
        H: Any = None,
        *,
        rootG: Optional[int] = None,
        rootH: Optional[int] = None,
        timestampsG: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
        timestampsH: Optional[Union[Sequence[Any], Mapping[Any, Any]]] = None,
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Return (matched_pairs, score) using *original* vertex ids.

        rootG/rootH are interpreted as original vertex ids. If omitted, we use
        the (original) root of each tree.
        """
        if G is not None or H is not None:
            if G is None or H is None:
                raise ValueError("Either provide both G and H, or provide neither.")
            treeG = self._to_tree(G, timestamps=timestampsG)
            treeH = self._to_tree(H, timestamps=timestampsH)
            if not self.encoder.is_fitted:
                self.encoder.fit_from_trees([treeG, treeH])
            encG = self.encode_tree(treeG)
            encH = self.encode_tree(treeH)
        else:
            if self.encG_ is None or self.encH_ is None or self.treeG_ is None or self.treeH_ is None:
                raise RuntimeError("Must call fit(G,H) before predict() if no inputs are provided.")
            treeG, treeH = self.treeG_, self.treeH_
            encG, encH = self.encG_, self.encH_

        # Map roots (original ids) -> internal indices.
        mapG = _orig_to_internal(treeG)
        mapH = _orig_to_internal(treeH)
        if rootG is None:
            root_u = 0
        else:
            if int(rootG) not in mapG:
                raise KeyError(f"rootG={rootG} not found among original vertex ids")
            root_u = mapG[int(rootG)]
        if rootH is None:
            root_v = 0
        else:
            if int(rootH) not in mapH:
                raise KeyError(f"rootH={rootH} not found among original vertex ids")
            root_v = mapH[int(rootH)]

        res = align_subtrees_fast_encoded(
            encG,
            encH,
            weight_by_id=self.encoder.weight_by_id,  # type: ignore[arg-type]
            root_u=root_u,
            root_v=root_v,
            return_matrix=False,
        )

        pairs_orig = [(int(treeG.orig_index[u]), int(treeH.orig_index[v])) for (u, v) in res.pairs_internal]
        return pairs_orig, float(res.score)

    def predict_encoded(
        self,
        G: EncodedSubtree,
        H: EncodedSubtree,
        *,
        rootG_internal: int = 0,
        rootH_internal: int = 0,
    ) -> Tuple[List[Tuple[int, int]], float]:
        """Predict using pre-encoded trees.

        rootG_internal/rootH_internal are **internal** node indices.
        Pairs are returned in original ids via the embedded TreeData.orig_index.
        """
        res = align_subtrees_fast_encoded(
            G,
            H,
            weight_by_id=self.encoder.weight_by_id,  # type: ignore[arg-type]
            root_u=int(rootG_internal),
            root_v=int(rootH_internal),
            return_matrix=False,
        )
        pairs_orig = [(int(G.tree.orig_index[u]), int(H.tree.orig_index[v])) for (u, v) in res.pairs_internal]
        return pairs_orig, float(res.score)


__all__ = [
    "HAVE_NUMBA",
    "EncodedSubtreeEquality",
    "EncodedSubtreeOverlap",
    "EncodedSubtree",
    "FastSubtreeAlignmentResult",
    "align_subtrees_fast_encoded",
    "FastSubtreeMatcher",
    "reorder_treedata_by_timestamp",
]
