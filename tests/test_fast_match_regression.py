"""Regression tests for FastTreePathMatcher traceback semantics.

These tests intentionally use TreeData directly, so they do not depend on igraph.
"""

from __future__ import annotations

import numpy as np

from path_matcher.fast_match import FastTreePathMatcher
from path_matcher.matcher import TreePathMatcher
from path_matcher.tree_data import TreeData


def _chain(labels):
    n = len(labels)
    parent = np.asarray([-1] + list(range(n - 1)), dtype=np.int32)
    return TreeData(parent=parent, label=list(labels), orig_index=np.arange(n, dtype=np.int32))


def test_fast_equality_returns_only_positive_matches_by_default():
    G = _chain(list("xAxxxBxxCx"))
    H = _chain(list("yAyyyByyCy"))

    fast = FastTreePathMatcher(mode="equality")
    fast.fit(G, H)
    pairs, score = fast.predict()

    assert pairs == [(1, 1), (5, 5), (8, 8)]
    assert score == 3.0

    # This is what the generic matcher returns when non-matches are penalized:
    # only actual matched vertices, not zero-score diagonal tie transitions.
    exact = TreePathMatcher(method="exact", w=lambda a, b: 1.0 if a == b else -0.25)
    exact.fit(G, H)
    exact_pairs, exact_score = exact.predict()
    assert pairs == exact_pairs
    assert score == exact_score


def test_fast_equality_can_return_raw_traceback_for_debugging():
    G = _chain(list("xAxxxBxxCx"))
    H = _chain(list("yAyyyByyCy"))

    fast = FastTreePathMatcher(mode="equality", keep_zero_weight_matches=True)
    fast.fit(G, H)
    pairs, score = fast.predict()

    assert score == 3.0
    assert pairs == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)]


def test_fast_equality_no_matches_returns_empty_path():
    G = _chain(list("abcdef"))
    H = _chain(list("uvwxyz"))

    fast = FastTreePathMatcher(mode="equality")
    fast.fit(G, H)
    pairs, score = fast.predict()

    assert pairs == []
    assert score == 0.0


def test_fast_overlap_returns_only_positive_matches_by_default():
    G = _chain([("x",), ("A",), ("q",), ("B",), ("r",)])
    H = _chain([("y",), ("A",), ("z",), ("B",), ("s",)])

    fast = FastTreePathMatcher(mode="overlap")
    fast.fit(G, H)
    pairs, score = fast.predict()

    assert pairs == [(1, 1), (3, 3)]
    assert score == 2.0


def test_default_extractor_understands_sym_dict_labels():
    G = _chain([{"sym": c, "weight": 1.0} for c in "xAxxxBxxCx"])
    H = _chain([{"sym": c, "weight": 1.0} for c in "yAyyyByyCy"])

    fast = FastTreePathMatcher(mode="equality")
    fast.fit(G, H)
    pairs, score = fast.predict()

    assert pairs == [(1, 1), (5, 5), (8, 8)]
    assert score == 3.0


def test_exact_matcher_honors_prefer_match_on_tie_flag():
    G = _chain(list("xAxxxBxxCx"))
    H = _chain(list("yAyyyByyCy"))
    w0 = lambda a, b: 1.0 if a == b else 0.0

    exact = TreePathMatcher(method="exact", w=w0, prefer_match_on_tie=False)
    exact.fit(G, H)
    pairs, score = exact.predict()

    assert pairs == [(1, 1), (5, 5), (8, 8)]
    assert score == 3.0
