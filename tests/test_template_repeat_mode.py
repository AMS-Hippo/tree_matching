"""Tests for TreePathMatcher(mode="template_repeat")."""

from __future__ import annotations

import numpy as np
import pytest

from path_matcher.matcher import TreePathMatcher
from path_matcher.tree_data import TreeData


def _chain(labels):
    n = len(labels)
    parent = np.asarray([-1] + list(range(n - 1)), dtype=np.int32)
    return TreeData(parent=parent, label=list(labels), orig_index=np.arange(n, dtype=np.int32))


def _eq_or_large_penalty(a, b):
    return 1.0 if a == b else -100.0


def test_template_repeat_reuses_template_vertices_in_order():
    G = _chain([1, 1, 2, 3, 3, 3])
    H = _chain([1, 2, 3])

    matcher = TreePathMatcher(method="exact", mode="template_repeat", w=_eq_or_large_penalty)
    pairs, score = matcher.predict(G, H)

    assert score == 6.0
    assert pairs == [(0, 0), (1, 0), (2, 1), (3, 2), (4, 2), (5, 2)]
    assert [j for _i, j in pairs] == [0, 0, 1, 2, 2, 2]


def test_unique_mode_keeps_existing_one_to_one_template_behavior():
    G = _chain([1, 1, 2, 3, 3, 3])
    H = _chain([1, 2, 3])

    matcher = TreePathMatcher(method="exact", mode="unique", w=_eq_or_large_penalty)
    pairs, score = matcher.predict(G, H)

    assert score == 3.0
    assert len(pairs) == 3
    assert [j for _i, j in pairs] == [0, 1, 2]


def test_template_repeat_never_decreases_template_index():
    G = _chain([1, 2, 1, 2, 3])
    H = _chain([1, 2, 3])

    matcher = TreePathMatcher(method="exact", mode="template_repeat", w=_eq_or_large_penalty)
    pairs, score = matcher.predict(G, H)

    template_indices = [j for _i, j in pairs]
    assert score == 4.0
    assert len(pairs) == 4
    assert template_indices == sorted(template_indices)
    assert len(pairs) < G.n  # one tree vertex must be skipped to preserve template order


def test_template_repeat_penalty_charges_only_extra_uses():
    G = _chain(["A", "A", "A"])
    H = _chain(["A"])

    unpenalized = TreePathMatcher(method="exact", mode="template_repeat", w=_eq_or_large_penalty)
    pairs0, score0 = unpenalized.predict(G, H)
    assert pairs0 == [(0, 0), (1, 0), (2, 0)]
    assert score0 == 3.0

    penalized = TreePathMatcher(
        method="exact",
        mode="template_repeat",
        w=_eq_or_large_penalty,
        template_repeat_penalty=0.25,
    )
    pairs1, score1 = penalized.predict(G, H)
    assert pairs1 == [(0, 0), (1, 0), (2, 0)]
    assert score1 == 2.5  # 1 + (1 - 0.25) + (1 - 0.25)

    high_penalty = TreePathMatcher(
        method="exact",
        mode="template_repeat",
        w=_eq_or_large_penalty,
        template_repeat_penalty=1.25,
    )
    pairs2, score2 = high_penalty.predict(G, H)
    assert pairs2 == [(0, 0)]
    assert score2 == 1.0


def test_template_repeat_requires_second_input_to_be_path():
    G = _chain(["A", "B", "C"])
    H = TreeData(
        parent=np.asarray([-1, 0, 0], dtype=np.int32),
        label=["A", "B", "C"],
        orig_index=np.arange(3, dtype=np.int32),
    )

    matcher = TreePathMatcher(method="exact", mode="template_repeat", w=_eq_or_large_penalty)
    with pytest.raises(ValueError, match="single root-to-leaf path"):
        matcher.predict(G, H)


def test_template_repeat_is_exact_only_for_now():
    with pytest.raises(ValueError, match="method='exact'"):
        TreePathMatcher(method="beam", mode="template_repeat")
