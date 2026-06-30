from __future__ import annotations

"""fast_subgraph_match_unordered.py

Exact DP for **unordered-children** subtree matching.

This is the sibling of `subgraph_match/fast_subgraph_match.py` (which assumes
an ordered left-to-right child order induced by timestamps).

When children are *unordered*, matching the children of a matched root-pair
(u,v) requires solving a maximum-weight bipartite matching / assignment
problem between the child sets of u and v, with edge weights A[child_u, child_v].

Complexity
----------
Let n=|G|, m=|H| and let d(u), d(v) be out-degrees.
The exact recurrence is:

  M[u,v] = w(u,v) + MWBM( {A[c,d] : c in ch(u), d in ch(v)} )
  A[u,v] = max( M[u,v], max_{c in ch(u)} A[c,v], max_{d in ch(v)} A[u,d] )

Where MWBM is the maximum-weight bipartite matching value.

We provide two solvers for MWBM:

  - unordered_solver='bitmask' (default): exact DP in O(L*s*2^s) where
    s=min(d(u),d(v)) and L=max(d(u),d(v)). This is fast when branching factors
    are small (e.g., s<=12..16).

  - unordered_solver='hungarian': exact Hungarian algorithm via
    scipy.optimize.linear_sum_assignment, which is O(D^3) per (u,v) where
    D=max(d(u),d(v)). This is intended for small trees / demos.

The unordered matcher is inherently much slower than the timestamp-ordered
matcher, so it is best used when degrees (and/or subtree sizes) are small.

API
---
`FastUnorderedSubtreeMatcher` mirrors `FastSubtreeMatcher`.

"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

# Reuse encoder + encoded-tree containers from the ordered matcher.
from .fast_subgraph_match import (
    HAVE_NUMBA,
    njit,  # type: ignore
    EncodedSubtree,
    EncodedSubtreeEquality,
    EncodedSubtreeOverlap,
    FastSubtreeAlignmentResult,
    FastSubtreeMatcher,
)


# -----------------------------------------------------------------------------
# Numba helpers: overlap scoring + bitmask matching
# -----------------------------------------------------------------------------


if HAVE_NUMBA:

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
    def _max_children_match_bitmask_numba(
        A: np.ndarray,
        child_offG: np.ndarray,
        child_flatG: np.ndarray,
        u: int,
        child_offH: np.ndarray,
        child_flatH: np.ndarray,
        v: int,
        bitmask_max: int,
        dp: np.ndarray,
        newdp: np.ndarray,
    ) -> float:
        """Exact max-weight matching between child sets using a bitmask DP.

        Runs in O(L*s*2^s) where s=min(deg(u),deg(v)).

        Preconditions
        -------------
        s <= bitmask_max.
        """
        off_u = int(child_offG[u])
        off_u_end = int(child_offG[u + 1])
        k = off_u_end - off_u

        off_v = int(child_offH[v])
        off_v_end = int(child_offH[v + 1])
        l = off_v_end - off_v

        if k == 0 or l == 0:
            return 0.0

        # Choose the smaller side for the mask.
        small_from_G = True
        s = k
        L = l
        if l < k:
            small_from_G = False
            s = l
            L = k

        if s > bitmask_max:
            # Sentinel (caller should avoid this path).
            return -1.0

        maskN = 1 << s
        neg_inf = -1.0e30

        for mask in range(maskN):
            dp[mask] = neg_inf
        dp[0] = 0.0

        # Iterate over the larger side rows.
        for row in range(L):
            for mask in range(maskN):
                newdp[mask] = dp[mask]  # skip this row

            for mask in range(maskN):
                base = dp[mask]
                if base <= neg_inf * 0.5:
                    continue

                for col in range(s):
                    if (mask >> col) & 1:
                        continue

                    if small_from_G:
                        # columns are G children; rows are H children
                        cg = int(child_flatG[off_u + col])
                        ch = int(child_flatH[off_v + row])
                        w = float(A[cg, ch])
                    else:
                        # columns are H children; rows are G children
                        cg = int(child_flatG[off_u + row])
                        ch = int(child_flatH[off_v + col])
                        w = float(A[cg, ch])

                    m2 = mask | (1 << col)
                    val = base + w
                    if val > newdp[m2]:
                        newdp[m2] = val

            tmp = dp
            dp = newdp
            newdp = tmp

        best = 0.0
        for mask in range(maskN):
            val = dp[mask]
            if val > best:
                best = val
        return best


    @njit(cache=True)
    def _dp_subtree_equality_unordered_bitmask_numba(
        child_offG: np.ndarray,
        child_flatG: np.ndarray,
        idsG: np.ndarray,
        child_offH: np.ndarray,
        child_flatH: np.ndarray,
        idsH: np.ndarray,
        weight_by_id: np.ndarray,
        bitmask_max: int,
    ) -> np.ndarray:
        n = idsG.shape[0]
        m = idsH.shape[0]
        A = np.zeros((n, m), dtype=np.float32)

        # scratch for bitmask DP
        if bitmask_max < 0 or bitmask_max > 20:
            bitmask_max = 20
        dp = np.empty(1 << bitmask_max, dtype=np.float32)
        newdp = np.empty(1 << bitmask_max, dtype=np.float32)

        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])
            idu = int(idsG[u])

            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])

                w_uv = 0.0
                if idu >= 0 and idu == int(idsH[v]):
                    w_uv = float(weight_by_id[idu])

                match_children = _max_children_match_bitmask_numba(
                    A,
                    child_offG,
                    child_flatG,
                    u,
                    child_offH,
                    child_flatH,
                    v,
                    bitmask_max,
                    dp,
                    newdp,
                )

                M = w_uv + match_children

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
    def _dp_subtree_overlap_unordered_bitmask_numba(
        child_offG: np.ndarray,
        child_flatG: np.ndarray,
        offG: np.ndarray,
        flatG: np.ndarray,
        child_offH: np.ndarray,
        child_flatH: np.ndarray,
        offH: np.ndarray,
        flatH: np.ndarray,
        weight_by_id: np.ndarray,
        bitmask_max: int,
    ) -> np.ndarray:
        n = child_offG.shape[0] - 1
        m = child_offH.shape[0] - 1
        A = np.zeros((n, m), dtype=np.float32)

        if bitmask_max < 0 or bitmask_max > 20:
            bitmask_max = 20
        dp = np.empty(1 << bitmask_max, dtype=np.float32)
        newdp = np.empty(1 << bitmask_max, dtype=np.float32)

        for u in range(n - 1, -1, -1):
            off_u = int(child_offG[u])
            off_u_end = int(child_offG[u + 1])

            for v in range(m - 1, -1, -1):
                off_v = int(child_offH[v])
                off_v_end = int(child_offH[v + 1])

                w_uv = _score_overlap_vertex_pair_numba(offG, flatG, u, offH, flatH, v, weight_by_id)

                match_children = _max_children_match_bitmask_numba(
                    A,
                    child_offG,
                    child_flatG,
                    u,
                    child_offH,
                    child_flatH,
                    v,
                    bitmask_max,
                    dp,
                    newdp,
                )

                M = w_uv + match_children

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
    # We provide only the Hungarian Python fallback below.
    pass


# -----------------------------------------------------------------------------
# Python helpers: overlap scoring + Hungarian matching (also used for traceback)
# -----------------------------------------------------------------------------


def _score_overlap_vertex_pair_py(
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


def _children(child_off: np.ndarray, child_flat: np.ndarray, u: int) -> np.ndarray:
    a = int(child_off[u])
    b = int(child_off[u + 1])
    return np.asarray(child_flat[a:b], dtype=np.int32)


def _max_weight_matching_value_and_pairs_bitmask(
    A: np.ndarray,
    cu: np.ndarray,
    cv: np.ndarray,
    *,
    eps: float = 1e-9,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Return (value, chosen_pairs) using a bitmask DP (no SciPy dependency).

    This is intended for **small degrees** (e.g. min(|cu|,|cv|) <= 12..16).

    The algorithm masks the smaller side to achieve O(L*s*2^s).
    """
    k = int(cu.shape[0])
    l = int(cv.shape[0])
    if k == 0 or l == 0:
        return 0.0, []

    # Choose the smaller side for the mask; always return pairs as (childG, childH).
    if k <= l:
        small = cu
        big = cv
        small_is_G = True
    else:
        small = cv
        big = cu
        small_is_G = False

    s = int(small.shape[0])
    L = int(big.shape[0])
    maskN = 1 << s

    neg_inf = -1.0e30
    dp0 = np.full(maskN, neg_inf, dtype=np.float32)
    dp0[0] = 0.0

    hist: List[np.ndarray] = []
    dp = dp0

    # Forward DP
    for row in range(L):
        newdp = dp.copy()
        for mask in range(maskN):
            base = float(dp[mask])
            if base <= neg_inf * 0.5:
                continue
            for col in range(s):
                if (mask >> col) & 1:
                    continue
                if small_is_G:
                    cg = int(small[col])
                    ch = int(big[row])
                else:
                    cg = int(big[row])
                    ch = int(small[col])
                w = float(A[cg, ch])
                m2 = mask | (1 << col)
                val = base + w
                if val > float(newdp[m2]):
                    newdp[m2] = val
        hist.append(newdp)
        dp = newdp

    # Best terminal state
    best_mask = int(np.argmax(dp))
    best_val = float(dp[best_mask])

    # Backtrack
    pairs_rev: List[Tuple[int, int]] = []
    mask = best_mask
    for row in range(L - 1, -1, -1):
        dp_prev = dp0 if row == 0 else hist[row - 1]
        dp_curr = hist[row]

        # If skipping this row attains dp_curr[mask], skip.
        if abs(float(dp_curr[mask]) - float(dp_prev[mask])) <= 1e-6:
            continue

        # Otherwise find a used column that explains the transition.
        found = False
        for col in range(s):
            if ((mask >> col) & 1) == 0:
                continue
            prev_mask = mask ^ (1 << col)
            if small_is_G:
                cg = int(small[col])
                ch = int(big[row])
            else:
                cg = int(big[row])
                ch = int(small[col])
            w = float(A[cg, ch])
            if abs(float(dp_curr[mask]) - (float(dp_prev[prev_mask]) + w)) <= 1e-4:
                if w > eps:
                    pairs_rev.append((cg, ch))
                mask = prev_mask
                found = True
                break

        if not found:
            # Numerical tie / noise: fall back to skipping.
            continue

    pairs_rev.reverse()
    return best_val, pairs_rev


def _max_weight_matching_value_and_pairs(
    A: np.ndarray,
    cu: np.ndarray,
    cv: np.ndarray,
    *,
    eps: float = 1e-9,
    bitmask_threshold: int = 16,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Choose a reconstruction method for child matching.

    - Uses the bitmask DP when min(|cu|,|cv|) <= bitmask_threshold.
    - Otherwise uses Hungarian (requires SciPy).
    """
    if min(int(cu.shape[0]), int(cv.shape[0])) <= int(bitmask_threshold):
        return _max_weight_matching_value_and_pairs_bitmask(A, cu, cv, eps=eps)
    return _max_weight_matching_value_and_pairs_hungarian(A, cu, cv, eps=eps)


def _max_weight_matching_value_and_pairs_hungarian(
    A: np.ndarray,
    cu: np.ndarray,
    cv: np.ndarray,
    *,
    eps: float = 1e-9,
) -> Tuple[float, List[Tuple[int, int]]]:
    """Return (value, chosen_pairs) for max-weight matching between cu and cv.

    Uses Hungarian on a padded square matrix with dummy nodes of 0 weight,
    which allows leaving vertices unmatched.
    """
    k = int(cu.shape[0])
    l = int(cv.shape[0])
    if k == 0 or l == 0:
        return 0.0, []

    # Build weight matrix.
    W = A[np.ix_(cu.astype(np.int64), cv.astype(np.int64))].astype(np.float32, copy=False)
    d = max(k, l)
    if W.shape[0] != d or W.shape[1] != d:
        Wpad = np.zeros((d, d), dtype=np.float32)
        Wpad[:k, :l] = W
        W = Wpad

    try:
        from scipy.optimize import linear_sum_assignment
    except Exception as e:  # pragma: no cover
        raise ImportError("scipy is required for unordered_solver='hungarian'") from e

    r, c = linear_sum_assignment(-W)  # maximize W
    total = float(W[r, c].sum())

    pairs: List[Tuple[int, int]] = []
    for rr, cc in zip(r.tolist(), c.tolist()):
        if rr < k and cc < l:
            w = float(W[rr, cc])
            if w > eps:
                pairs.append((int(cu[rr]), int(cv[cc])))

    return total, pairs


def _dp_subtree_equality_unordered_hungarian_py(
    encG: EncodedSubtreeEquality,
    encH: EncodedSubtreeEquality,
    weight_by_id: np.ndarray,
) -> np.ndarray:
    idsG = np.asarray(encG.label_ids, dtype=np.int32)
    idsH = np.asarray(encH.label_ids, dtype=np.int32)
    child_offG = np.asarray(encG.child_offsets, dtype=np.int32)
    child_flatG = np.asarray(encG.child_flat, dtype=np.int32)
    child_offH = np.asarray(encH.child_offsets, dtype=np.int32)
    child_flatH = np.asarray(encH.child_flat, dtype=np.int32)

    n = int(idsG.shape[0])
    m = int(idsH.shape[0])
    A = np.zeros((n, m), dtype=np.float32)

    for u in range(n - 1, -1, -1):
        cu = _children(child_offG, child_flatG, u)
        idu = int(idsG[u])
        for v in range(m - 1, -1, -1):
            cv = _children(child_offH, child_flatH, v)

            w_uv = float(weight_by_id[idu]) if (idu >= 0 and idu == int(idsH[v])) else 0.0

            match_children, _ = _max_weight_matching_value_and_pairs_hungarian(A, cu, cv)
            M = w_uv + float(match_children)

            best1 = float(A[cu, v].max()) if cu.size else 0.0
            best2 = float(A[u, cv].max()) if cv.size else 0.0

            A[u, v] = max(M, best1, best2)

    return A


def _dp_subtree_overlap_unordered_hungarian_py(
    encG: EncodedSubtreeOverlap,
    encH: EncodedSubtreeOverlap,
    weight_by_id: np.ndarray,
) -> np.ndarray:
    offG = np.asarray(encG.offsets, dtype=np.int32)
    flatG = np.asarray(encG.flat_token_ids, dtype=np.int32)
    offH = np.asarray(encH.offsets, dtype=np.int32)
    flatH = np.asarray(encH.flat_token_ids, dtype=np.int32)

    child_offG = np.asarray(encG.child_offsets, dtype=np.int32)
    child_flatG = np.asarray(encG.child_flat, dtype=np.int32)
    child_offH = np.asarray(encH.child_offsets, dtype=np.int32)
    child_flatH = np.asarray(encH.child_flat, dtype=np.int32)

    n = int(child_offG.shape[0] - 1)
    m = int(child_offH.shape[0] - 1)
    A = np.zeros((n, m), dtype=np.float32)

    for u in range(n - 1, -1, -1):
        cu = _children(child_offG, child_flatG, u)
        for v in range(m - 1, -1, -1):
            cv = _children(child_offH, child_flatH, v)

            w_uv = _score_overlap_vertex_pair_py(offG, flatG, u, offH, flatH, v, weight_by_id)

            match_children, _ = _max_weight_matching_value_and_pairs_hungarian(A, cu, cv)
            M = float(w_uv) + float(match_children)

            best1 = float(A[cu, v].max()) if cu.size else 0.0
            best2 = float(A[u, cv].max()) if cv.size else 0.0

            A[u, v] = max(M, best1, best2)

    return A


# -----------------------------------------------------------------------------
# Traceback (Python): reconstruct one optimal matching
# -----------------------------------------------------------------------------


def _reconstruct_matching_unordered(
    encG: EncodedSubtree,
    encH: EncodedSubtree,
    A: np.ndarray,
    *,
    root_u: int,
    root_v: int,
    weight_by_id: np.ndarray,
    eps: float = 1e-6,
) -> List[Tuple[int, int]]:
    child_offG = encG.child_offsets
    child_flatG = encG.child_flat
    child_offH = encH.child_offsets
    child_flatH = encH.child_flat

    if isinstance(encG, EncodedSubtreeEquality):
        idsG = encG.label_ids
        idsH = encH.label_ids  # type: ignore[assignment]

        def score_uv(u: int, v: int) -> float:
            idu = int(idsG[u])
            if idu >= 0 and idu == int(idsH[v]):
                return float(weight_by_id[idu])
            return 0.0

    else:
        offG = encG.offsets
        flatG = encG.flat_token_ids
        offH = encH.offsets  # type: ignore[assignment]
        flatH = encH.flat_token_ids  # type: ignore[assignment]

        def score_uv(u: int, v: int) -> float:
            return _score_overlap_vertex_pair_py(offG, flatG, u, offH, flatH, v, weight_by_id)

    def reconstruct_free(u: int, v: int) -> List[Tuple[int, int]]:
        best = float(A[u, v])

        cu = _children(child_offG, child_flatG, u)
        cv = _children(child_offH, child_flatH, v)

        # skip-u
        best1 = 0.0
        arg1 = -1
        if cu.size:
            vals = A[cu, v]
            idx = int(np.argmax(vals))
            best1 = float(vals[idx])
            arg1 = int(cu[idx])

        # skip-v
        best2 = 0.0
        arg2 = -1
        if cv.size:
            vals = A[u, cv]
            idx = int(np.argmax(vals))
            best2 = float(vals[idx])
            arg2 = int(cv[idx])

        match_children_val, child_pairs = _max_weight_matching_value_and_pairs(A, cu, cv)
        M = score_uv(u, v) + float(match_children_val)

        # deterministic tie-break: rooted > skip-u > skip-v
        choice = 3
        best_val = M
        if best1 > best_val + eps:
            best_val = best1
            choice = 1
        if best2 > best_val + eps:
            best_val = best2
            choice = 2

        # Align with stored A[u,v] when float noise exists
        if abs(best - best_val) > 1e-4 and best > best_val + 1e-4:
            if abs(best - M) <= 1e-4:
                choice = 3
            elif abs(best - best1) <= 1e-4:
                choice = 1
            elif abs(best - best2) <= 1e-4:
                choice = 2

        if choice == 3 and M + 1e-4 >= best:
            out: List[Tuple[int, int]] = [(u, v)]
            for a, b in child_pairs:
                out.extend(reconstruct_free(int(a), int(b)))
            return out
        if choice == 1 and arg1 >= 0 and best1 + 1e-4 >= best:
            return reconstruct_free(arg1, v)
        if choice == 2 and arg2 >= 0 and best2 + 1e-4 >= best:
            return reconstruct_free(u, arg2)

        # fallback
        if M >= best1 and M >= best2:
            out = [(u, v)]
            for a, b in child_pairs:
                out.extend(reconstruct_free(int(a), int(b)))
            return out
        if best1 >= best2 and arg1 >= 0:
            return reconstruct_free(arg1, v)
        if arg2 >= 0:
            return reconstruct_free(u, arg2)
        return []

    return reconstruct_free(int(root_u), int(root_v))


# -----------------------------------------------------------------------------
# Public alignment function
# -----------------------------------------------------------------------------


def align_subtrees_unordered_fast_encoded(
    G: EncodedSubtree,
    H: EncodedSubtree,
    *,
    weight_by_id: np.ndarray,
    root_u: int = 0,
    root_v: int = 0,
    unordered_solver: str = "bitmask",
    bitmask_max: int = 16,
    return_matrix: bool = False,
) -> FastSubtreeAlignmentResult:
    """Compute optimal unordered-children subtree matching.

    Parameters
    ----------
    unordered_solver:
        "bitmask" or "hungarian".
    bitmask_max:
        Maximum allowed s=min(deg(u),deg(v)) for the bitmask solver.
        If exceeded, raise ValueError.
    """

    solver = str(unordered_solver).lower().strip()
    bitmask_max = int(bitmask_max)

    if solver not in {"bitmask", "hungarian"}:
        raise ValueError("unordered_solver must be 'bitmask' or 'hungarian'")

    if isinstance(G, EncodedSubtreeEquality) and isinstance(H, EncodedSubtreeEquality):
        degG = (G.child_offsets[1:] - G.child_offsets[:-1]).astype(np.int32)
        degH = (H.child_offsets[1:] - H.child_offsets[:-1]).astype(np.int32)
        min_max_deg = int(min(int(degG.max()) if degG.size else 0, int(degH.max()) if degH.size else 0))

        if solver == "bitmask":
            if not HAVE_NUMBA:
                raise RuntimeError("unordered_solver='bitmask' requires numba")
            if min_max_deg > bitmask_max:
                raise ValueError(
                    f"bitmask_max={bitmask_max} is too small for these trees (min(max_deg_G,max_deg_H)={min_max_deg}). "
                    "Use unordered_solver='hungarian' or increase bitmask_max (<=20 recommended)."
                )

            A = _dp_subtree_equality_unordered_bitmask_numba(
                np.asarray(G.child_offsets, dtype=np.int32),
                np.asarray(G.child_flat, dtype=np.int32),
                np.asarray(G.label_ids, dtype=np.int32),
                np.asarray(H.child_offsets, dtype=np.int32),
                np.asarray(H.child_flat, dtype=np.int32),
                np.asarray(H.label_ids, dtype=np.int32),
                np.asarray(weight_by_id, dtype=np.float32),
                bitmask_max,
            )
        else:
            A = _dp_subtree_equality_unordered_hungarian_py(
                G,
                H,
                np.asarray(weight_by_id, dtype=np.float32),
            )

        score = float(A[int(root_u), int(root_v)])
        pairs = _reconstruct_matching_unordered(
            G,
            H,
            A,
            root_u=int(root_u),
            root_v=int(root_v),
            weight_by_id=np.asarray(weight_by_id, dtype=np.float32),
        )
        if return_matrix:
            return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)), A=A)
        return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)))

    if isinstance(G, EncodedSubtreeOverlap) and isinstance(H, EncodedSubtreeOverlap):
        degG = (G.child_offsets[1:] - G.child_offsets[:-1]).astype(np.int32)
        degH = (H.child_offsets[1:] - H.child_offsets[:-1]).astype(np.int32)
        min_max_deg = int(min(int(degG.max()) if degG.size else 0, int(degH.max()) if degH.size else 0))

        if solver == "bitmask":
            if not HAVE_NUMBA:
                raise RuntimeError("unordered_solver='bitmask' requires numba")
            if min_max_deg > bitmask_max:
                raise ValueError(
                    f"bitmask_max={bitmask_max} is too small for these trees (min(max_deg_G,max_deg_H)={min_max_deg}). "
                    "Use unordered_solver='hungarian' or increase bitmask_max (<=20 recommended)."
                )

            A = _dp_subtree_overlap_unordered_bitmask_numba(
                np.asarray(G.child_offsets, dtype=np.int32),
                np.asarray(G.child_flat, dtype=np.int32),
                np.asarray(G.offsets, dtype=np.int32),
                np.asarray(G.flat_token_ids, dtype=np.int32),
                np.asarray(H.child_offsets, dtype=np.int32),
                np.asarray(H.child_flat, dtype=np.int32),
                np.asarray(H.offsets, dtype=np.int32),
                np.asarray(H.flat_token_ids, dtype=np.int32),
                np.asarray(weight_by_id, dtype=np.float32),
                bitmask_max,
            )
        else:
            A = _dp_subtree_overlap_unordered_hungarian_py(
                G,
                H,
                np.asarray(weight_by_id, dtype=np.float32),
            )

        score = float(A[int(root_u), int(root_v)])
        pairs = _reconstruct_matching_unordered(
            G,
            H,
            A,
            root_u=int(root_u),
            root_v=int(root_v),
            weight_by_id=np.asarray(weight_by_id, dtype=np.float32),
        )
        if return_matrix:
            return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)), A=A)
        return FastSubtreeAlignmentResult(pairs_internal=pairs, score=score, roots_internal=(int(root_u), int(root_v)))

    raise TypeError("Encoded tree types do not match. Use equality+equality or overlap+overlap.")


# -----------------------------------------------------------------------------
# High-level matcher class
# -----------------------------------------------------------------------------


class FastUnorderedSubtreeMatcher(FastSubtreeMatcher):
    """Unordered-children subtree matcher.

    This subclasses `FastSubtreeMatcher` to reuse:
      - Tree conversion / timestamp handling
      - FastLabelEncoder
      - Tree encoding

    The only difference is that alignment is performed with unordered-child
    constraints via `align_subtrees_unordered_fast_encoded`.
    """

    def __init__(
        self,
        *,
        unordered_solver: str = "bitmask",
        bitmask_max: int = 16,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.unordered_solver = str(unordered_solver)
        self.bitmask_max = int(bitmask_max)

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
        """Return (matched_pairs, score) using *original* vertex ids."""

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
        mapG: Dict[int, int] = {int(o): int(i) for i, o in enumerate(np.asarray(treeG.orig_index))}
        mapH: Dict[int, int] = {int(o): int(i) for i, o in enumerate(np.asarray(treeH.orig_index))}

        root_u = 0 if rootG is None else mapG[int(rootG)]
        root_v = 0 if rootH is None else mapH[int(rootH)]

        res = align_subtrees_unordered_fast_encoded(
            encG,
            encH,
            weight_by_id=self.encoder.weight_by_id,  # type: ignore[arg-type]
            root_u=int(root_u),
            root_v=int(root_v),
            unordered_solver=self.unordered_solver,
            bitmask_max=self.bitmask_max,
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
        res = align_subtrees_unordered_fast_encoded(
            G,
            H,
            weight_by_id=self.encoder.weight_by_id,  # type: ignore[arg-type]
            root_u=int(rootG_internal),
            root_v=int(rootH_internal),
            unordered_solver=self.unordered_solver,
            bitmask_max=self.bitmask_max,
            return_matrix=False,
        )
        pairs_orig = [(int(G.tree.orig_index[u]), int(H.tree.orig_index[v])) for (u, v) in res.pairs_internal]
        return pairs_orig, float(res.score)


__all__ = [
    "FastUnorderedSubtreeMatcher",
    "align_subtrees_unordered_fast_encoded",
]
