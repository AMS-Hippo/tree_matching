"""Correctness tests for the sparse product-poset chain algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from path_matcher.bucketable_weight import BucketableWeight
from path_matcher.fast_sparse_match import FastSparseTreePathMatcher
from path_matcher.matcher import TreePathMatcher
from path_matcher.needleman_wunsch_tree import align_trees_algorithm1
from path_matcher.sparse_align import (
    SparseCandidateConfig,
    align_trees_sparse_candidates,
    generate_sparse_candidates,
)
from path_matcher.sparse_chain import align_scored_sparse_chain, align_trees_sparse_chain
from path_matcher.sparse_preprocess import preprocess_treedata
from path_matcher.tree_data import TreeData


def _tree(parent, labels):
    parent_arr = np.asarray(parent, dtype=np.int32)
    return TreeData(parent=parent_arr, label=list(labels), orig_index=np.arange(len(labels), dtype=np.int32))


def _random_tree(rng, n, n_labels=4):
    parent = [-1]
    parent.extend(int(rng.integers(0, node)) for node in range(1, n))
    labels = [int(x) for x in rng.integers(0, n_labels, size=n)]
    return _tree(parent, labels)


def _is_strict_ancestor(parent, ancestor, node):
    cur = int(parent[node])
    while cur >= 0:
        if cur == ancestor:
            return True
        cur = int(parent[cur])
    return False


def _assert_valid_matching(G, H, pairs):
    for (u0, v0), (u1, v1) in zip(pairs, pairs[1:]):
        assert _is_strict_ancestor(G.parent, u0, u1)
        assert _is_strict_ancestor(H.parent, v0, v1)


def _path_score(G, H, pairs, w):
    return sum(float(w(G.label[u], H.label[v])) for u, v in pairs)


def _restricted_chain_score(G, H, scored_candidates):
    """Quadratic oracle for the maximum chain on a small candidate set."""
    pairs = []
    for u, row in enumerate(scored_candidates):
        best_by_v = {}
        for v, weight in row:
            weight = float(weight)
            if weight <= 0.0:
                continue
            best_by_v[int(v)] = max(weight, best_by_v.get(int(v), -np.inf))
        for v, weight in best_by_v.items():
            pairs.append((u, v, weight))

    pairs.sort(key=lambda x: (x[0], x[1]))
    dp = [0.0] * len(pairs)
    best = 0.0
    for i, (u, v, weight) in enumerate(pairs):
        predecessor = 0.0
        for j in range(i):
            u0, v0, _ = pairs[j]
            if _is_strict_ancestor(G.parent, u0, u) and _is_strict_ancestor(H.parent, v0, v):
                predecessor = max(predecessor, dp[j])
        dp[i] = predecessor + weight
        best = max(best, dp[i])
    return best


def test_sparse_chain_all_pairs_randomized_matches_exact_dp():
    """The chain recurrence agrees with dense DP when every pair is supplied."""
    rng = np.random.default_rng(20260926)

    for _ in range(150):
        G = _random_tree(rng, int(rng.integers(1, 9)))
        H = _random_tree(rng, int(rng.integers(1, 9)))
        weights = rng.integers(-1, 4, size=(4, 4))

        def w(a, b, weights=weights):
            return float(weights[int(a), int(b)])

        # One universal blocking key is enough because this test supplies every
        # pair explicitly.  It also exercises nonpositive weights, which can be
        # discarded because arbitrary ancestor jumps are allowed.
        bucket_w = BucketableWeight(w=w, keyer=lambda _label: ("all",))
        preG = preprocess_treedata(G, w=bucket_w)
        preH = preprocess_treedata(H, w=bucket_w)
        candidates = [list(range(H.n)) for _ in range(G.n)]

        exact = align_trees_algorithm1(G, H, w=w, prefer_match_on_tie=False)
        sparse = align_trees_sparse_chain(preG, preH, candidates=candidates, w=w)

        assert sparse.score == pytest.approx(exact.score)
        _assert_valid_matching(G, H, sparse.path_internal)
        assert _path_score(G, H, sparse.path_internal, w) == pytest.approx(sparse.score)


def test_sparse_chain_random_restricted_candidates_match_quadratic_oracle():
    """The segment-tree implementation is exact relative to an arbitrary candidate set."""
    rng = np.random.default_rng(20260927)

    for _ in range(150):
        G = _random_tree(rng, int(rng.integers(1, 10)))
        H = _random_tree(rng, int(rng.integers(1, 10)))
        weights = rng.integers(0, 5, size=(4, 4))

        scored_candidates = []
        allowed = set()
        for u in range(G.n):
            row = []
            for v in range(H.n):
                if rng.random() < 0.35:
                    weight = float(weights[int(G.label[u]), int(H.label[v])])
                    row.append((v, weight))
                    allowed.add((u, v))
                    if rng.random() < 0.10:  # duplicate pair exercises canonicalization
                        row.append((v, max(0.0, weight - 1.0)))
            scored_candidates.append(row)

        sparse = align_scored_sparse_chain(G, H, scored_candidates)
        oracle = _restricted_chain_score(G, H, scored_candidates)

        assert sparse.score == pytest.approx(oracle)
        _assert_valid_matching(G, H, sparse.path_internal)
        assert all(pair in allowed for pair in sparse.path_internal)
        score = sum(float(weights[int(G.label[u]), int(H.label[v])]) for u, v in sparse.path_internal)
        assert score == pytest.approx(sparse.score)


def test_sparse_chain_exhaustive_equality_candidates_and_high_level_api():
    G = _tree([-1, 0, 0, 1, 1, 2, 5], ["r", "A", "x", "B", "C", "A", "D"])
    H = _tree([-1, 0, 0, 1, 2, 2], ["z", "A", "A", "B", "C", "D"])

    exact = align_trees_algorithm1(G, H, prefer_match_on_tie=False)

    matcher = TreePathMatcher(method="sparse_chain")
    assert matcher.sparse_cfg == SparseCandidateConfig.exhaustive()

    # TreeData inputs now use the same reusable preprocessing path as igraph inputs.
    preG = matcher.preprocess(G)
    preH = matcher.preprocess(H)
    pairs, score = matcher.predict(preG, preH)

    assert score == pytest.approx(exact.score)
    _assert_valid_matching(G, H, pairs)
    assert _path_score(G, H, pairs, lambda a, b: 1.0 if a == b else 0.0) == pytest.approx(score)

    generated = generate_sparse_candidates(preG, preH, cfg=SparseCandidateConfig.exhaustive())
    direct = align_trees_sparse_chain(preG, preH, candidates=generated)
    assert direct.score == pytest.approx(exact.score)


def test_sparse_chain_exhaustive_overlap_blocking_matches_exact_dp():
    """Multiple shared keys may generate duplicates; the exact score is unchanged."""
    rng = np.random.default_rng(20260928)
    token_weights = np.asarray([0.5, 1.0, 1.5, 2.0, 3.0], dtype=float)

    def random_labels(n):
        labels = []
        for _ in range(n):
            size = int(rng.integers(1, 4))
            labels.append(tuple(sorted(set(int(x) for x in rng.integers(0, 5, size=size)))))
        return labels

    def w(a, b):
        common = set(a).intersection(b)
        return max((float(token_weights[t]) for t in common), default=0.0)

    bucket_w = BucketableWeight(w=w, keyer=lambda label: tuple(label))

    for _ in range(75):
        n = int(rng.integers(1, 9))
        m = int(rng.integers(1, 9))
        parentG = [-1] + [int(rng.integers(0, u)) for u in range(1, n)]
        parentH = [-1] + [int(rng.integers(0, v)) for v in range(1, m)]
        G = _tree(parentG, random_labels(n))
        H = _tree(parentH, random_labels(m))

        preG = preprocess_treedata(G, w=bucket_w)
        preH = preprocess_treedata(H, w=bucket_w)
        exact = align_trees_algorithm1(G, H, w=w, prefer_match_on_tie=False)
        sparse = align_trees_sparse_chain(preG, preH, w=bucket_w)

        assert sparse.score == pytest.approx(exact.score)
        _assert_valid_matching(G, H, sparse.path_internal)
        assert _path_score(G, H, sparse.path_internal, w) == pytest.approx(sparse.score)


def test_sparse_chain_cannot_chain_sibling_endpoints():
    G = _tree([-1, 0, 0], ["root", "A", "B"])
    H = _tree([-1, 0, 0], ["root", "A", "B"])
    scored = [[], [(1, 5.0)], [(2, 7.0)]]

    result = align_scored_sparse_chain(G, H, scored)

    assert result.score == 7.0
    assert result.path_internal == [(2, 2)]


def test_fast_sparse_equality_randomized_matches_dense_fast_matcher():
    from path_matcher.fast_match import FastTreePathMatcher
    from path_matcher.fast_sparse_match import FastSparseTreePathMatcher

    rng = np.random.default_rng(20260929)
    token_weights = {0: 0.5, 1: 1.0, 2: 2.0, 3: 3.5}

    for _ in range(100):
        G = _random_tree(rng, int(rng.integers(1, 12)))
        H = _random_tree(rng, int(rng.integers(1, 12)))

        dense = FastTreePathMatcher(mode="equality", token_weights=token_weights)
        sparse = FastSparseTreePathMatcher(mode="equality", token_weights=token_weights)

        dense.fit(G, H)
        sparse.fit(G, H)
        dense_pairs, dense_score = dense.predict()
        sparse_pairs, sparse_score = sparse.predict()

        assert sparse_score == pytest.approx(dense_score)
        _assert_valid_matching(G, H, sparse_pairs)
        assert sum(token_weights[G.label[u]] for u, _ in sparse_pairs) == pytest.approx(sparse_score)


def test_fast_sparse_overlap_randomized_matches_dense_fast_matcher():
    from path_matcher.fast_match import FastTreePathMatcher
    from path_matcher.fast_sparse_match import FastSparseTreePathMatcher

    rng = np.random.default_rng(20260930)
    token_weights = {0: 0.25, 1: 0.75, 2: 1.0, 3: 2.0, 4: 4.0}

    def random_labels(n):
        labels = []
        for _ in range(n):
            size = int(rng.integers(1, 4))
            labels.append(tuple(sorted(set(int(x) for x in rng.integers(0, 5, size=size)))))
        return labels

    for _ in range(75):
        n = int(rng.integers(1, 11))
        m = int(rng.integers(1, 11))
        G = _tree([-1] + [int(rng.integers(0, u)) for u in range(1, n)], random_labels(n))
        H = _tree([-1] + [int(rng.integers(0, v)) for v in range(1, m)], random_labels(m))

        dense = FastTreePathMatcher(mode="overlap", token_weights=token_weights)
        sparse = FastSparseTreePathMatcher(mode="overlap", token_weights=token_weights)

        dense.fit(G, H)
        sparse.fit(G, H)
        _dense_pairs, dense_score = dense.predict()
        sparse_pairs, sparse_score = sparse.predict()

        assert sparse_score == pytest.approx(dense_score)
        _assert_valid_matching(G, H, sparse_pairs)

        score = 0.0
        for u, v in sparse_pairs:
            common = set(G.label[u]).intersection(H.label[v])
            score += max(token_weights[t] for t in common)
        assert score == pytest.approx(sparse_score)


def test_fast_sparse_prepared_batch_workflow_reuses_tree_indices():
    trees = [
        _tree([-1, 0, 1, 1], ["root", "A", "B", "C"]),
        _tree([-1, 0, 0, 2], ["root", "A", "x", "B"]),
        _tree([-1, 0, 1], ["root", "A", "B"]),
    ]
    matcher = FastSparseTreePathMatcher(mode="equality")
    matcher.fit_encoder(trees)
    prepared = [matcher.prepare_tree(tree) for tree in trees]

    pairs, score = matcher.predict_prepared(prepared[0], prepared[2])

    assert pairs == [(0, 0), (1, 1), (2, 2)]
    assert score == 3.0
    assert prepared[0].children[0] == (1,)
    assert prepared[0].nodes_by_token


def test_fast_encoder_warns_and_drops_unseen_labels() -> None:
    training = _tree([-1, 0], ["root", "A"])
    unseen = _tree([-1, 0], ["root", "B"])
    matcher = FastSparseTreePathMatcher(mode="equality")
    matcher.fit_encoder([training])

    with pytest.warns(UserWarning, match="fit_encoder"):
        prepared = matcher.prepare_tree(unseen)

    assert int(prepared.encoded.label_ids[0]) >= 0
    assert int(prepared.encoded.label_ids[1]) == -1


def test_fast_sparse_prepared_trees_reject_incompatible_encoder_mappings() -> None:
    tree_a = _tree([-1, 0], ["root", "A"])
    tree_b = _tree([-1, 0], ["root", "B"])

    first = FastSparseTreePathMatcher(mode="equality")
    first.fit_encoder([tree_a, tree_b])
    prepared_a = first.prepare_tree(tree_a)

    second = FastSparseTreePathMatcher(mode="equality")
    second.fit_encoder([tree_a, tree_b])
    prepared_b = second.prepare_tree(tree_b)

    with pytest.raises(ValueError, match="different FastLabelEncoder|incompatible"):
        first.predict_prepared(prepared_a, prepared_b)


@pytest.mark.parametrize("method", ["closure", "chain"])
def test_sparse_weight_override_warns_but_runs(method: str) -> None:
    G = _tree([-1, 0, 1], ["root", "A", "B"])
    H = _tree([-1, 0, 1], ["root", "A", "B"])
    keyer = lambda label: (label,)
    preprocess_weight = BucketableWeight(
        w=lambda a, b: 1.0 if a == b else 0.0,
        keyer=keyer,
    )
    scoring_weight = BucketableWeight(
        w=lambda a, b: 2.0 if a == b else 0.0,
        keyer=keyer,
    )
    pre_g = preprocess_treedata(G, w=preprocess_weight)
    pre_h = preprocess_treedata(H, w=preprocess_weight)

    with pytest.warns(UserWarning, match="different weight object"):
        if method == "chain":
            result = align_trees_sparse_chain(
                pre_g,
                pre_h,
                cfg=SparseCandidateConfig.exhaustive(),
                w=scoring_weight,
            )
        else:
            result = align_trees_sparse_candidates(
                pre_g,
                pre_h,
                cfg=SparseCandidateConfig.exhaustive(),
                w=scoring_weight,
            )

    assert result.score == pytest.approx(6.0)
