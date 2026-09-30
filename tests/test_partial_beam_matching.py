"""Correctness tests for the two beam-search state spaces."""

from __future__ import annotations

import numpy as np
import pytest

from path_matcher.beam_align import align_trees_beam
from path_matcher.local_beam_align import align_trees_local_beam
from path_matcher.matcher import TreePathMatcher
from path_matcher.needleman_wunsch_tree import align_trees_algorithm1
from path_matcher.tree_data import TreeData


def _chain(labels):
    n = len(labels)
    parent = np.asarray([-1] + list(range(n - 1)), dtype=np.int32)
    return TreeData(parent=parent, label=list(labels), orig_index=np.arange(n, dtype=np.int32))


def _tree(parent, labels):
    parent_arr = np.asarray(parent, dtype=np.int32)
    return TreeData(parent=parent_arr, label=list(labels), orig_index=np.arange(len(labels), dtype=np.int32))


def _is_strict_descendant(parent, ancestor, node):
    cur = int(parent[node])
    while cur >= 0:
        if cur == ancestor:
            return True
        cur = int(parent[cur])
    return False


def _assert_valid_matching(G, H, pairs):
    for (u0, v0), (u1, v1) in zip(pairs, pairs[1:]):
        assert _is_strict_descendant(G.parent, u0, u1)
        assert _is_strict_descendant(H.parent, v0, v1)


def _matching_score(G, H, pairs, w):
    return sum(float(w(G.label[u], H.label[v])) for u, v in pairs)


def _random_tree(rng, n, n_labels=4):
    parent = [-1]
    parent.extend(int(rng.integers(0, node)) for node in range(1, n))
    labels = [int(x) for x in rng.integers(0, n_labels, size=n)]
    return _tree(parent, labels)


def test_partial_beam_default_finds_ordered_chain_matches():
    G = _chain("xAxxBxxC")
    H = _chain("yAyyBzzC")

    matcher = TreePathMatcher(method="beam", beam_symmetric=False, seed=123)
    pairs, score = matcher.predict(G, H)

    assert pairs == [(1, 1), (4, 4), (7, 7)]
    assert score == 3.0


def test_partial_beam_exhaustive_settings_match_exact_score_on_small_tree():
    G = _tree(
        [-1, 0, 0, 1, 1, 2, 5],
        ["r", "A", "x", "B", "C", "A", "D"],
    )
    H = _tree(
        [-1, 0, 0, 1, 2, 2],
        ["z", "A", "A", "B", "C", "D"],
    )

    exact = align_trees_algorithm1(G, H, prefer_match_on_tie=False)
    beam = align_trees_beam(
        G,
        H,
        beam_width=200,
        expansion_width=None,
        max_label_pair_scan=10_000,
        max_label_pairs_per_expansion=None,
        max_nodes_per_label_side=100,
        candidate_select_mode="first",
        seed=0,
    )

    assert beam.score == exact.score
    _assert_valid_matching(G, H, beam.path_internal)


def test_partial_beam_exhaustive_randomized_matches_exact_dp():
    """Wide, uncapped Algorithm 7 agrees with exact DP on random small inputs."""
    rng = np.random.default_rng(20260924)

    for _ in range(100):
        G = _random_tree(rng, int(rng.integers(1, 8)))
        H = _random_tree(rng, int(rng.integers(1, 8)))
        weights = rng.integers(0, 4, size=(4, 4))

        def w(a, b, weights=weights):
            return float(weights[int(a), int(b)])

        exact = align_trees_algorithm1(G, H, w=w, prefer_match_on_tie=False)
        beam = align_trees_beam(
            G,
            H,
            w=w,
            beam_width=G.n * H.n,
            expansion_width=None,
            max_label_pair_scan=10_000,
            max_label_pairs_per_expansion=None,
            max_nodes_per_label_side=max(G.n, H.n),
            candidate_select_mode="first",
            random_fraction=0.0,
            rarity_weight=0.0,
            gap_penalty=0.0,
            balance_penalty=0.0,
            candidate_future_weight=0.0,
            priority_future_weight=0.0,
            seed=0,
        )

        assert beam.score == pytest.approx(exact.score)
        _assert_valid_matching(G, H, beam.path_internal)
        assert _matching_score(G, H, beam.path_internal, w) == pytest.approx(beam.score)


def test_local_beam_exhaustive_randomized_matches_exact_dp():
    """Algorithm 6 with all children and no effective beam pruning is exact."""
    rng = np.random.default_rng(20260925)

    for _ in range(100):
        G = _random_tree(rng, int(rng.integers(1, 8)))
        H = _random_tree(rng, int(rng.integers(1, 8)))
        weights = rng.integers(0, 4, size=(4, 4))

        def w(a, b, weights=weights):
            return float(weights[int(a), int(b)])

        exact = align_trees_algorithm1(G, H, w=w, prefer_match_on_tie=False)
        beam = align_trees_local_beam(
            G,
            H,
            w=w,
            beam_width=(G.n + 1) * (H.n + 1),
            child_cap=None,
        )

        assert beam.score == pytest.approx(exact.score)
        _assert_valid_matching(G, H, beam.path_internal)
        assert _matching_score(G, H, beam.path_internal, w) == pytest.approx(beam.score)


def test_local_beam_capped_expansion_and_high_level_api():
    G = _tree([-1, 0, 0], ["root-G", "first-G", "B"])
    H = _tree([-1, 0, 0], ["root-H", "first-H", "B"])

    capped = align_trees_local_beam(G, H, beam_width=100, child_cap=1)
    full = align_trees_local_beam(G, H, beam_width=100, child_cap=None)

    assert capped.path_internal == []
    assert capped.score == 0.0
    assert full.path_internal == [(2, 2)]
    assert full.score == 1.0

    matcher = TreePathMatcher(
        method="beam_local",
        beam_width=100,
        beam_local_child_cap=None,
    )
    pairs, score = matcher.predict(G, H)
    assert pairs == full.path_internal
    assert score == full.score


def test_partial_beam_priority_callable_is_used():
    G = _chain("xAxxBxxC")
    H = _chain("yAyyBzzC")
    calls = {"n": 0}

    def priority(ctx):
        calls["n"] += 1
        return ctx.default_priority

    matcher = TreePathMatcher(
        method="beam",
        beam_symmetric=False,
        beam_priority_fn=priority,
        seed=0,
    )
    pairs, score = matcher.predict(G, H)

    assert score == 3.0
    assert pairs == [(1, 1), (4, 4), (7, 7)]
    assert calls["n"] > 0


def test_partial_beam_candidate_heuristic_callable_is_used():
    G = _chain("AxxB")
    H = _chain("AyyB")
    calls = {"n": 0}

    def candidate_priority(ctx):
        calls["n"] += 1
        return ctx.default_priority

    matcher = TreePathMatcher(
        method="beam",
        beam_symmetric=False,
        beam_candidate_heuristic=candidate_priority,
        seed=0,
    )
    pairs, score = matcher.predict(G, H)

    assert score == 2.0
    assert pairs == [(0, 0), (3, 3)]
    assert calls["n"] > 0


def test_partial_beam_custom_expansion_fn_is_validated():
    G = _chain("AB")
    H = _chain("AB")

    def bad_expansion(_ctx):
        return [(999, 0)]

    matcher = TreePathMatcher(
        method="beam",
        beam_symmetric=False,
        beam_expansion_fn=bad_expansion,
    )

    with pytest.raises(ValueError, match="outside tree bounds"):
        matcher.predict(G, H)


def test_beam_lookahead_can_rescue_low_width_prefix_choice():
    # With a one-state, one-candidate beam and no ordinary future bonus, the
    # deterministic local heuristic can choose an isolated terminal match.  The
    # optional lookahead sketch sees that the second A nodes have a B-C
    # continuation below them and keeps that prefix instead.
    G = TreeData(
        parent=np.asarray([-1, 0, 0, 2, 3], dtype=np.int32),
        label=["Groot", "A", "A", "B", "C"],
        orig_index=np.arange(5, dtype=np.int32),
    )
    H = TreeData(
        parent=np.asarray([-1, 0, 0, 2, 3], dtype=np.int32),
        label=["Hroot", "A", "A", "B", "C"],
        orig_index=np.arange(5, dtype=np.int32),
    )
    common = dict(
        beam_width=1,
        expansion_width=1,
        random_fraction=0.0,
        seed=0,
        candidate_select_mode="first",
        max_nodes_per_label_side=8,
        rarity_weight=0.0,
        gap_penalty=0.0,
        balance_penalty=0.0,
        candidate_future_weight=0.0,
        priority_future_weight=0.0,
    )

    baseline = align_trees_beam(G, H, **common)
    assert baseline.score == 1.0

    lookahead = align_trees_beam(
        G,
        H,
        **common,
        lookahead=True,
        lookahead_weight=2.0,
        lookahead_sketch_size=8,
        lookahead_depth_discount=1.0,
    )
    assert lookahead.path_internal == [(2, 2), (3, 3), (4, 4)]
    assert lookahead.score == 3.0

    matcher = TreePathMatcher(
        method="beam",
        beam_symmetric=False,
        beam_width=1,
        beam_expansion_width=1,
        beam_random_fraction=0.0,
        seed=0,
        candidate_select_mode="first",
        beam_max_nodes_per_label_side=8,
        beam_rarity_weight=0.0,
        beam_gap_penalty=0.0,
        beam_balance_penalty=0.0,
        beam_candidate_future_weight=0.0,
        beam_priority_future_weight=0.0,
        beam_lookahead=True,
        beam_lookahead_weight=2.0,
        beam_lookahead_sketch_size=8,
        beam_lookahead_depth_discount=1.0,
    )
    pairs, score = matcher.predict(G, H)
    assert pairs == lookahead.path_internal
    assert score == lookahead.score


def test_lookahead_score_is_exposed_to_custom_candidate_heuristic():
    parent = np.asarray([-1, 0, 0, 2, 3], dtype=np.int32)
    G = TreeData(parent=parent, label=["rootG", "A", "A", "B", "C"], orig_index=np.arange(5, dtype=np.int32))
    H = TreeData(parent=parent, label=["rootH", "A", "A", "B", "C"], orig_index=np.arange(5, dtype=np.int32))

    seen = []

    def collect(ctx):
        seen.append(float(ctx.lookahead_score))
        return ctx.default_priority

    res = align_trees_beam(
        G,
        H,
        beam_width=2,
        expansion_width=4,
        random_fraction=0.0,
        lookahead=True,
        lookahead_weight=0.5,
        candidate_heuristic=collect,
    )

    assert res.score >= 3.0
    assert seen
    assert max(seen) > 0.0


def test_chunk_only_lookahead_option_is_used():
    # This isolates the path-chunk part of lookahead.  The isolated A match is
    # locally tied with the A that leads to B-C; the fixed-length chunk sketch
    # makes the latter candidate more attractive even with label overlap disabled.
    G = TreeData(
        parent=np.asarray([-1, 0, 0, 2, 3], dtype=np.int32),
        label=["Groot", "A", "A", "B", "C"],
        orig_index=np.arange(5, dtype=np.int32),
    )
    H = TreeData(
        parent=np.asarray([-1, 0, 0, 2, 3], dtype=np.int32),
        label=["Hroot", "A", "A", "B", "C"],
        orig_index=np.arange(5, dtype=np.int32),
    )
    common = dict(
        beam_width=1,
        expansion_width=1,
        random_fraction=0.0,
        seed=0,
        candidate_select_mode="first",
        max_nodes_per_label_side=8,
        rarity_weight=0.0,
        gap_penalty=0.0,
        balance_penalty=0.0,
        candidate_future_weight=0.0,
        priority_future_weight=0.0,
    )

    baseline = align_trees_beam(G, H, **common)
    chunk_lookahead = align_trees_beam(
        G,
        H,
        **common,
        lookahead=True,
        lookahead_weight=2.0,
        lookahead_sketch_size=8,
        lookahead_depth_discount=1.0,
        lookahead_chunk_size=2,
        lookahead_label_weight=0.0,
        lookahead_chunk_weight=1.0,
    )

    assert baseline.score == 1.0
    assert chunk_lookahead.score > baseline.score
    assert chunk_lookahead.path_internal[0] == (2, 2)
