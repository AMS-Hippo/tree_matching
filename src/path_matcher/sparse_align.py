
"""
Sparse-candidate alignment (exact DP relative to a restricted match set).

This implements the idea:
- allow "skip" moves everywhere (options 1 and 2 from Algorithm 1),
- but allow "match" move (option 3) only on a precomputed sparse set of candidate pairs.

This is useful when both trees are large but true matches are sparse.

Workflow
--------
1) Preprocess each tree once with `sparse_preprocess.preprocess_treedata` / `preprocess_igraph`.
2) For a specific pair (G,H), build candidate pairs using inverted indices (bucket join)
   + rarity-aware pruning + optional subtree-sketch keys.
3) Run the sparse DP to obtain best score + traceback path.

All per-tree heavy work is in preprocessing; per-pair work is dominated by the number
of candidate edges you generate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import heapq
import math
import numpy as np

from .needleman_wunsch_tree import AlignmentResult
from .sparse_preprocess import PreprocessedTree, _UINT64_MAX
from .candidates import select_subset


@dataclass(frozen=True)
class SparseCandidateConfig:
    """
    Controls candidate generation for sparse alignment.
    """
    # Key-based candidates:
    stop_key_threshold: int = 10_000   # ignore keys with freq > threshold in either tree
    max_keys_per_node: Optional[int] = 4  # per-node: only use this many (rarest) blocking keys
    max_candidates_per_key: Optional[int] = 500  # cap |bucket_H(key)| used for any one key
    max_candidates_per_u: Optional[int] = 2000   # final per-u cap after union across keys
    candidate_select_mode: str = "first"
    seed: int = 0

    # Sketch-based candidates (optional):
    use_subtree_sketch_keys: bool = False
    sketch_keys_per_node: int = 4              # use up to this many sketch hashes per node
    max_candidates_per_sketch_key: Optional[int] = 500  # cap per sketch hash bucket


def _select_rarest_keys(
    keys: Sequence[Any],
    *,
    counts_G: Dict[Any, int],
    counts_H: Dict[Any, int],
    max_keys: int,
) -> List[Any]:
    """
    Pick up to max_keys keys, preferring those that are rare across both trees.
    """
    if len(keys) <= max_keys:
        return list(keys)

    scored = []
    for k in keys:
        c = counts_G.get(k, 0) + counts_H.get(k, 0)
        scored.append((c, k))
    scored.sort(key=lambda x: x[0])
    return [k for (_, k) in scored[:max_keys]]


def generate_sparse_candidates(
    G: PreprocessedTree,
    H: PreprocessedTree,
    *,
    cfg: SparseCandidateConfig = SparseCandidateConfig(),
) -> List[List[int]]:
    """
    Build candidate v-lists for each u in G.

    Candidates come from:
      - shared blocking keys (bucket join), with rarity-aware pruning/caps,
      - optionally, shared subtree-sketch hash keys.

    Returns
    -------
    candidates[u] = list of v indices (0..m-1) in H.
    """
    n = G.tree.n
    m = H.tree.n
    rng = np.random.default_rng(cfg.seed)

    candidates: List[List[int]] = []

    for u in range(n):
        cand_set: Set[int] = set()

        # 1) Key-based candidates.
        keys_u = list(G.node_keys[u])
        if cfg.max_keys_per_node is not None and cfg.max_keys_per_node > 0:
            keys_u = _select_rarest_keys(
                keys_u,
                counts_G=G.key_counts,
                counts_H=H.key_counts,
                max_keys=cfg.max_keys_per_node,
            )

        for k in keys_u:
            cG = G.key_counts.get(k, 0)
            cH = H.key_counts.get(k, 0)
            if cH == 0:
                continue
            if max(cG, cH) > cfg.stop_key_threshold:
                continue

            vs = H.key_to_nodes.get(k, [])
            if not vs:
                continue

            if cfg.max_candidates_per_key is not None:
                vs_use = vs[: cfg.max_candidates_per_key]
            else:
                vs_use = vs

            for v in vs_use:
                cand_set.add(int(v))

        # 2) Sketch-based candidates (optional).
        if cfg.use_subtree_sketch_keys:
            if G.subtree_sketch is None or H.sketch_to_nodes is None or H.subtree_sketch is None:
                raise ValueError(
                    "use_subtree_sketch_keys=True requires preprocessing with build_subtree_sketch=True "
                    "for both trees."
                )
            hv_row = G.subtree_sketch[u]
            taken = 0
            for hv in hv_row:
                if hv == _UINT64_MAX:
                    break
                bucket = H.sketch_to_nodes.get(int(hv), [])
                if not bucket:
                    continue

                if cfg.max_candidates_per_sketch_key is not None:
                    bucket_use = bucket[: cfg.max_candidates_per_sketch_key]
                else:
                    bucket_use = bucket

                for v in bucket_use:
                    cand_set.add(int(v))

                taken += 1
                if taken >= cfg.sketch_keys_per_node:
                    break

        cand_list = sorted(cand_set)
        if cfg.max_candidates_per_u is not None:
            cand_list = select_subset(cand_list, cfg.max_candidates_per_u, mode=cfg.candidate_select_mode, rng=rng)

        candidates.append(cand_list)

    return candidates


def align_trees_sparse_candidates(
    G: PreprocessedTree,
    H: PreprocessedTree,
    *,
    candidates: Optional[List[List[int]]] = None,
    cfg: SparseCandidateConfig = SparseCandidateConfig(),
    w: Optional[Any] = None,
    prefer_match_on_tie: bool = True,
) -> AlignmentResult:
    """
    Sparse-candidate DP alignment.

    Parameters
    ----------
    candidates:
        Optional precomputed candidates[u] list of v indices.
        If None, we build candidates using generate_sparse_candidates(G,H,cfg).
    w:
        Weight function to score matches. If None, uses G.w (the weight used to build keys).
        Must be callable w(label_u, label_v)->float.
    prefer_match_on_tie:
        If True, tie-break like the paper: prefer option 3 over 2 over 1.
        If False, prefer skipping on ties (often avoids long 0-weight match paths).

    Returns
    -------
    AlignmentResult (path_internal, score, end_internal).
    """
    if candidates is None:
        candidates = generate_sparse_candidates(G, H, cfg=cfg)

    w_fn = G.w if w is None else w
    if not callable(w_fn):
        raise TypeError("w must be callable")

    n, m = G.tree.n, H.tree.n
    labelsG = G.tree.label
    labelsH = H.tree.label

    ancG = G.tree.ancestors_shifted()
    ancH = H.tree.ancestors_shifted()

    # Candidate membership sets in shifted coordinates.
    cand_shift: List[Set[int]] = []
    for u in range(n):
        s = set((int(v) + 1) for v in candidates[u])
        cand_shift.append(s)

    A_rows: List[Dict[int, float]] = [dict() for _ in range(n + 1)]
    C_rows: List[Dict[int, int]] = [dict() for _ in range(n + 1)]
    A_rows[0][0] = 0.0
    C_rows[0][0] = 0

    best_score: float = 0.0
    best_U: int = 0
    best_V: int = 0

    def _tie_break(opt1: float, opt2: float, opt3: float) -> Tuple[float, int]:
        if prefer_match_on_tie:
            if opt3 >= opt2 and opt3 >= opt1:
                return opt3, 3
            if opt2 >= opt1:
                return opt2, 2
            return opt1, 1
        # Prefer skipping on ties.
        if opt1 >= opt2 and opt1 >= opt3:
            return opt1, 1
        if opt2 >= opt3:
            return opt2, 2
        return opt3, 3

    def ensure_cell(U: int, V: int) -> float:
        nonlocal best_score, best_U, best_V

        if V in A_rows[U]:
            return A_rows[U][V]

        stack: List[Tuple[int, int, int]] = [(U, V, 0)]
        while stack:
            uU, vV, state = stack.pop()

            if vV in A_rows[uU]:
                continue

            if uU == 0 or vV == 0:
                A_rows[uU][vV] = 0.0
                C_rows[uU][vV] = 0
                continue

            u = uU - 1
            v = vV - 1
            aU = int(ancG[u])
            aV = int(ancH[v])

            if state == 0:
                stack.append((uU, vV, 1))
                stack.append((aU, vV, 0))
                stack.append((uU, aV, 0))
                stack.append((aU, aV, 0))
                continue

            opt1 = A_rows[aU][vV]
            opt2 = A_rows[uU][aV]
            base = A_rows[aU][aV]

            if vV in cand_shift[u]:
                w_uv = float(w_fn(labelsG[u], labelsH[v]))
                opt3 = w_uv + base
            else:
                opt3 = -math.inf

            val, choice = _tie_break(opt1, opt2, opt3)
            A_rows[uU][vV] = float(val)
            C_rows[uU][vV] = int(choice)

            if uU != 0 and vV != 0 and val > best_score:
                best_score = float(val)
                best_U = int(uU)
                best_V = int(vV)

        return A_rows[U][V]

    # Drive computation by ensuring all candidate cells (and their closures) are computed.
    for u in range(n):
        U = u + 1
        for v in candidates[u]:
            V = int(v) + 1
            ensure_cell(U, V)

    if best_U == 0 or best_V == 0:
        return AlignmentResult(path_internal=[], score=0.0, end_internal=(0, 0), A=None, C=None)

    # Traceback from best cell.
    path_rev: List[Tuple[int, int]] = []
    U, V = best_U, best_V
    while U != 0 and V != 0:
        choice = int(C_rows[U].get(V, 0))
        if choice == 0:
            break

        if choice == 3:
            path_rev.append((U - 1, V - 1))

        if choice == 1:
            U = int(ancG[U - 1])
        elif choice == 2:
            V = int(ancH[V - 1])
        elif choice == 3:
            U = int(ancG[U - 1])
            V = int(ancH[V - 1])
        else:
            raise RuntimeError(f"Invalid traceback choice {choice} at (U,V)=({U},{V})")

    path_rev.reverse()
    return AlignmentResult(path_internal=path_rev, score=float(best_score), end_internal=(best_U - 1, best_V - 1), A=None, C=None)
