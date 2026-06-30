
"""
Specialized fast matcher for the old "tree vs weighted reference sequence" objective.

This is a direct adaptation of the older `top_path_match` / `dev_top_path_match`
logic to the current internal `TreeData` representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from numbers import Number
from typing import Any, Callable, List, Optional, Sequence, Tuple

from .igraph_io import igraph_to_treedata
from .tree_data import TreeData

ReferenceMatchRule = Callable[[Any, Any], bool]
CombineScores = Callable[[float, float], float]

@dataclass(frozen=True)
class ReferenceAlignmentResult:
    score: float
    path_internal: List[int]
    path_orig: List[int]
    matched_labels: List[Optional[Any]]


def _default_match_rule(label_sym: Any, ref_sym: Any) -> bool:
    return (label_sym == ref_sym) or (isinstance(label_sym, str) and label_sym[:1] == ref_sym)


def _make_combine_fn(combine_scores: str | CombineScores) -> CombineScores:
    if callable(combine_scores):
        return combine_scores
    mode = str(combine_scores).lower().strip()
    if mode == "add":
        return lambda a, b: a + b
    if mode == "min":
        return lambda a, b: min(a, b)
    if mode == "multiply":
        return lambda a, b: a * b
    raise ValueError("combine_scores must be 'add', 'min', 'multiply', or a callable")


def _extract_labels_and_weights(payload: Any) -> Tuple[List[Any], List[Optional[float]]]:
    if payload is None:
        return [], []
    if isinstance(payload, list):
        if all(isinstance(t, tuple) and len(t) == 2 for t in payload):
            return [a for a, _ in payload], [float(b) if isinstance(b, Number) else None for _, b in payload]
        return list(payload), [None] * len(payload)
    if isinstance(payload, tuple):
        if len(payload) == 2 and isinstance(payload[1], Number):
            return [payload[0]], [float(payload[1])]
        return [payload], [None]
    if isinstance(payload, dict):
        labels = payload.get("Labels") or payload.get("labels") or []
        weights = payload.get("Weights") or payload.get("weights")
        if weights is None:
            weights_out: List[Optional[float]] = [None] * len(labels)
        else:
            weights_out = [float(w) if isinstance(w, Number) else None for w in weights]
            if len(weights_out) < len(labels):
                weights_out.extend([None] * (len(labels) - len(weights_out)))
            elif len(weights_out) > len(labels):
                weights_out = weights_out[: len(labels)]
        return list(labels), weights_out
    return [payload], [None]


def _children_from_parent(tree: TreeData) -> List[List[int]]:
    children: List[List[int]] = [[] for _ in range(tree.n)]
    for u in range(1, tree.n):
        p = int(tree.parent[u])
        children[p].append(u)
    return children


def align_tree_to_weighted_reference(
    tree: TreeData,
    ref: Sequence[Any],
    ref_weights: Sequence[float],
    *,
    match_rule: Optional[ReferenceMatchRule] = None,
    combine_scores: str | CombineScores = "add",
) -> ReferenceAlignmentResult:
    ref_list = list(ref)
    wt_list = [float(x) for x in ref_weights]
    if len(ref_list) != len(wt_list):
        raise ValueError("ref and ref_weights must have the same length")
    m = len(ref_list)
    if m == 0:
        return ReferenceAlignmentResult(score=0.0, path_internal=[], path_orig=[], matched_labels=[])

    rule = _default_match_rule if match_rule is None else match_rule
    combine_fn = _make_combine_fn(combine_scores)
    children = _children_from_parent(tree)
    labels = list(tree.label)

    @lru_cache(maxsize=None)
    def get_lw(v: int) -> Tuple[Tuple[Any, ...], Tuple[Optional[float], ...]]:
        L, W = _extract_labels_and_weights(labels[v])
        return tuple(L), tuple(W)

    @lru_cache(maxsize=None)
    def dp(v: int, j: int) -> Tuple[float, Tuple[str, Optional[int], int, Optional[Any]]]:
        best_score = 0.0
        best_choice: Tuple[str, Optional[int], int, Optional[Any]] = ("skip", None, j, None)
        for c in children[v]:
            s_child, _ = dp(c, j)
            if s_child > best_score:
                best_score = s_child
                best_choice = ("skip", c, j, None)
        L, W = get_lw(v)
        for k, label_sym in enumerate(L):
            w_local = W[k]
            label_bonus = float(w_local) if isinstance(w_local, Number) else 0.0
            for i_ref in range(j, m):
                if not rule(label_sym, ref_list[i_ref]):
                    continue
                gain = float(combine_fn(float(wt_list[i_ref]), label_bonus))
                if children[v]:
                    for c in children[v]:
                        s_child, _ = dp(c, i_ref + 1)
                        total = gain + s_child
                        if total > best_score:
                            best_score = total
                            best_choice = ("match", c, i_ref + 1, label_sym)
                else:
                    if gain > best_score:
                        best_score = gain
                        best_choice = ("match", None, i_ref + 1, label_sym)
                break
        return best_score, best_choice

    best_root_score, _ = dp(0, 0)
    path_internal: List[int] = []
    matched_labels: List[Optional[Any]] = []
    v: Optional[int] = 0
    j = 0
    while v is not None:
        _score, choice = dp(v, j)
        _action, child, next_j, label = choice
        path_internal.append(int(v))
        matched_labels.append(label)
        v = child
        j = next_j
    path_orig = [int(tree.orig_index[u]) for u in path_internal]
    return ReferenceAlignmentResult(
        score=float(best_root_score),
        path_internal=path_internal,
        path_orig=path_orig,
        matched_labels=matched_labels,
    )


def align_tree_to_weighted_reference_dev(
    tree: TreeData,
    ref: Sequence[Any],
    ref_weights: Sequence[float],
    *,
    match_rule: Optional[ReferenceMatchRule] = None,
    combine_scores: str | CombineScores = "add",
) -> ReferenceAlignmentResult:
    ref_list = list(ref)
    wt_list = [float(x) for x in ref_weights]
    if len(ref_list) != len(wt_list):
        raise ValueError("ref and ref_weights must have the same length")
    m = len(ref_list)

    rule = _default_match_rule if match_rule is None else match_rule
    combine_fn = _make_combine_fn(combine_scores)
    children = _children_from_parent(tree)
    labels = list(tree.label)

    @lru_cache(maxsize=None)
    def get_lw(v: int) -> Tuple[Tuple[Any, ...], Tuple[Optional[float], ...]]:
        L, W = _extract_labels_and_weights(labels[v])
        return tuple(L), tuple(W)

    @lru_cache(maxsize=None)
    def dp(v: int, j: int) -> Tuple[float, Tuple[str, Optional[int], int, Optional[Any]]]:
        child_list = children[v]
        if j >= m and not child_list:
            return 0.0, ("end", None, j, None)
        best_score = 0.0
        best_choice: Tuple[str, Optional[int], int, Optional[Any]] = ("end", None, j, None)
        for c in child_list:
            s_child, _ = dp(c, j)
            if s_child > best_score:
                best_score = s_child
                best_choice = ("skip_vertex", c, j, None)
        if j < m:
            s_skip_ref, _ = dp(v, j + 1)
            if s_skip_ref > best_score:
                best_score = s_skip_ref
                best_choice = ("skip_ref", v, j + 1, None)
        L, W = get_lw(v)
        for k, label_sym in enumerate(L):
            w_local = W[k]
            label_bonus = float(w_local) if isinstance(w_local, Number) else 0.0
            for i_ref in range(j, m):
                if not rule(label_sym, ref_list[i_ref]):
                    continue
                gain = float(combine_fn(float(wt_list[i_ref]), label_bonus))
                if child_list:
                    best_child_score = 0.0
                    best_child: Optional[int] = None
                    for c in child_list:
                        s_child, _ = dp(c, i_ref + 1)
                        if s_child > best_child_score:
                            best_child_score = s_child
                            best_child = c
                    total = gain + best_child_score
                    if total > best_score:
                        best_score = total
                        best_choice = ("match", best_child, i_ref + 1, label_sym)
                else:
                    if gain > best_score:
                        best_score = gain
                        best_choice = ("match", None, i_ref + 1, label_sym)
        return best_score, best_choice

    best_root_score, _ = dp(0, 0)
    path_internal: List[int] = []
    matched_labels: List[Optional[Any]] = []
    v: Optional[int] = 0
    j = 0
    while v is not None:
        _score, choice = dp(v, j)
        action, next_v, next_j, label = choice
        if action == "end":
            break
        if action in {"match", "skip_vertex"}:
            path_internal.append(int(v))
            matched_labels.append(label if action == "match" else None)
            if next_v is None:
                break
            v, j = next_v, next_j
        elif action == "skip_ref":
            j = next_j
        else:
            break
    path_orig = [int(tree.orig_index[u]) for u in path_internal]
    return ReferenceAlignmentResult(
        score=float(best_root_score),
        path_internal=path_internal,
        path_orig=path_orig,
        matched_labels=matched_labels,
    )


def match_igraph_to_weighted_reference(
    G: Any,
    ref: Sequence[Any],
    ref_weights: Sequence[float],
    *,
    phi_name: str = "label",
    order: str = "auto",
    ts_field: Optional[str] = None,
    strict_tree: bool = True,
    match_rule: Optional[ReferenceMatchRule] = None,
    combine_scores: str | CombineScores = "add",
    dev: bool = False,
) -> ReferenceAlignmentResult:
    tree = igraph_to_treedata(G, phi_name=phi_name, order=order, ts_field=ts_field, strict_tree=strict_tree)
    if dev:
        return align_tree_to_weighted_reference_dev(tree, ref, ref_weights, match_rule=match_rule, combine_scores=combine_scores)
    return align_tree_to_weighted_reference(tree, ref, ref_weights, match_rule=match_rule, combine_scores=combine_scores)
