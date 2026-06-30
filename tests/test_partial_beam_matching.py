"""Tests for the partial-matching beam search implementation."""

from __future__ import annotations

import numpy as np
import pytest

from path_matcher import TreePathMatcher
from path_matcher.beam_align import align_trees_beam
from path_matcher.needleman_wunsch_tree import align_trees_algorithm1
from path_matcher.tree_data import TreeData


def _chain(labels):
    n = len(labels)
    return TreeData(
        parent=np.asarray([-1] + list(range(n - 1)), dtype=np.int32),
        label=list(labels),
        orig_index=np.arange(n, dtype=np.int32),
    )


def _is_strictly_ordered_path(pairs, G: TreeData, H: TreeData) -> bool:
    def is_ancestor(parent, a, b):
        cur = int(parent[b])
        while cur >= 0:
            if cur == a:
                return True
            cur = int(parent[cur])
        return False

    for (u0, v0), (u1, v1) in zip(pairs, pairs[1:]):
        if not is_ancestor(G.parent, u0, u1):
            return False
        if not is_ancestor(H.parent, v0, v1):
            return False
    return True


def test_partial_beam_matches_simple_chain():
    G = _chain("xABCDy")
    H = _chain("zABCDw")

    res = align_trees_beam(G, H, beam_width=2, expansion_width=4, random_fraction=0.0)

    assert res.path_internal == [(1, 1), (2, 2), (3, 3), (4, 4)]
    assert res.score == 4.0
    assert _is_strictly_ordered_path(res.path_internal, G, H)


def test_tree_path_matcher_beam_uses_partial_matching_search():
    # There are two A-B-C-like regions, but only one legal path has length 3 in both trees.
    G = TreeData(
        parent=np.asarray([-1, 0, 1, 2, 0, 4, 5], dtype=np.int32),
        label=list("XABCABC"),
        orig_index=np.arange(7, dtype=np.int32),
    )
    H = TreeData(
        parent=np.asarray([-1, 0, 1, 2, 0, 4, 5], dtype=np.int32),
        label=list("YABZABC"),
        orig_index=np.arange(7, dtype=np.int32),
    )

    exact = align_trees_algorithm1(G, H, prefer_match_on_tie=False)
    matcher = TreePathMatcher(
        method="beam",
        beam_width=2,
        beam_expansion_width=8,
        beam_random_fraction=0.0,
        seed=0,
    )
    pairs, score = matcher.predict(G, H)

    assert exact.score == 3.0
    assert score == exact.score
    assert pairs == exact.path_internal


def test_candidate_heuristic_callable_is_honored():
    G = TreeData(
        parent=np.asarray([-1, 0, 0, 1], dtype=np.int32),
        label=["root", "A", "C", "B"],
        orig_index=np.arange(4, dtype=np.int32),
    )
    H = TreeData(
        parent=np.asarray([-1, 0, 0, 1], dtype=np.int32),
        label=["root", "A", "C", "B"],
        orig_index=np.arange(4, dtype=np.int32),
    )

    baseline = align_trees_beam(G, H, beam_width=1, expansion_width=1, random_fraction=0.0, seed=0)
    assert baseline.score == 3.0

    calls = []

    def favor_c(ctx):
        calls.append((ctx.u, ctx.v))
        if ctx.label_u == "C" and ctx.label_v == "C":
            return 1_000.0
        return -1_000.0

    biased = align_trees_beam(
        G,
        H,
        beam_width=1,
        expansion_width=1,
        random_fraction=0.0,
        seed=0,
        candidate_heuristic=favor_c,
    )

    assert calls
    assert biased.path_internal == [(2, 2)]
    assert biased.score == 1.0


def test_custom_expansion_fn_is_validated():
    G = _chain("ABC")
    H = _chain("ABC")

    def invalid_expansion(ctx):
        if ctx.last_u == -1:
            return [(1, 1)]
        # Returning the same terminal again is not a strict-descendant extension.
        return [(ctx.last_u, ctx.last_v)]

    with pytest.raises(ValueError, match="strict descendant"):
        align_trees_beam(
            G,
            H,
            beam_width=1,
            expansion_width=2,
            expansion_fn=invalid_expansion,
            random_fraction=0.0,
        )
