from __future__ import annotations

import numpy as np
import pytest

from path_matcher.beam_align import align_trees_beam
from path_matcher.fast_beam_match import FastBeamTreePathMatcher
from path_matcher.fast_match import FastTreePathMatcher
from path_matcher.tree_data import TreeData


def _random_tree(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.asarray([-1] + [int(rng.integers(0, u)) for u in range(1, n)], dtype=np.int64)


def _strict_ancestor(parent: np.ndarray, u: int, v: int) -> bool:
    node = int(v)
    while node >= 0:
        node = int(parent[node])
        if node == int(u):
            return True
    return False


def _assert_valid_score(
    G: TreeData,
    H: TreeData,
    path: list[tuple[int, int]],
    score: float,
    weight,
) -> None:
    for (u1, v1), (u2, v2) in zip(path, path[1:]):
        assert _strict_ancestor(G.parent, u1, u2)
        assert _strict_ancestor(H.parent, v1, v2)
    explicit = sum(float(weight(G.label[u], H.label[v])) for u, v in path)
    assert np.isclose(explicit, score)


def test_fast_equality_beam_matches_generic_score_only_beam_randomized() -> None:
    rng = np.random.default_rng(4071)
    for _ in range(120):
        n = int(rng.integers(4, 28))
        m = int(rng.integers(4, 28))
        alphabet = 8
        labels_g = [int(rng.integers(alphabet)) for _ in range(n)]
        labels_h = [int(rng.integers(alphabet)) for _ in range(m)]
        G = TreeData(_random_tree(n, rng), labels_g, np.arange(n))
        H = TreeData(_random_tree(m, rng), labels_h, np.arange(m))
        token_weights = {token: float(rng.integers(1, 6)) for token in range(alphabet)}

        def weight(a: int, b: int) -> float:
            return token_weights[a] if a == b else 0.0

        generic = align_trees_beam(
            G,
            H,
            w=weight,
            beam_width=30,
            expansion_width=20,
            max_nodes_per_label_side=8,
            candidate_select_mode="first",
            random_fraction=0.0,
            rarity_weight=0.0,
            gap_penalty=0.0,
            balance_penalty=0.0,
            candidate_future_weight=0.0,
            priority_future_weight=0.0,
            priority_length_weight=0.0,
            lookahead=False,
            n_restarts=1,
            seed=0,
        )
        fast = FastBeamTreePathMatcher(
            mode="equality",
            token_weights=token_weights,
            beam_width=30,
            expansion_width=20,
            max_nodes_per_token_side=8,
        )
        path, score = fast.predict(G, H)

        assert np.isclose(score, generic.score)
        _assert_valid_score(G, H, path, score, weight)


def test_fast_beam_exhaustive_budgets_recover_exact_equality_and_overlap() -> None:
    rng = np.random.default_rng(4072)
    for mode in ("equality", "overlap"):
        for _ in range(80):
            n = int(rng.integers(2, 11))
            m = int(rng.integers(2, 11))
            alphabet = 6
            if mode == "equality":
                labels_g = [int(rng.integers(alphabet)) for _ in range(n)]
                labels_h = [int(rng.integers(alphabet)) for _ in range(m)]
            else:
                labels_g = [
                    tuple(sorted(set(int(x) for x in rng.integers(alphabet, size=int(rng.integers(1, 4))))))
                    for _ in range(n)
                ]
                labels_h = [
                    tuple(sorted(set(int(x) for x in rng.integers(alphabet, size=int(rng.integers(1, 4))))))
                    for _ in range(m)
                ]
            G = TreeData(_random_tree(n, rng), labels_g, np.arange(n))
            H = TreeData(_random_tree(m, rng), labels_h, np.arange(m))
            token_weights = {token: float(rng.integers(1, 5)) for token in range(alphabet)}

            exact = FastTreePathMatcher(mode=mode, token_weights=token_weights)
            _, exact_score = exact.predict(G, H)

            exhaustive = FastBeamTreePathMatcher(
                mode=mode,
                token_weights=token_weights,
                beam_width=n * m + 1,
                expansion_width=n * m + 1,
                max_nodes_per_token_side=max(n, m),
            )
            path, score = exhaustive.predict(G, H)
            assert np.isclose(score, exact_score)

            if mode == "equality":
                def weight(a, b):
                    return token_weights[a] if a == b else 0.0
            else:
                def weight(a, b):
                    common = set(a).intersection(b)
                    return max((token_weights[token] for token in common), default=0.0)
            _assert_valid_score(G, H, path, score, weight)


def test_fast_beam_prepared_api_diagnostics_and_encoder_guard() -> None:
    G = TreeData(
        np.asarray([-1, 0, 0, 1, 1, 2], dtype=np.int64),
        ["a", "b", "c", "d", "a", "b"],
        np.arange(6),
    )
    H = TreeData(
        np.asarray([-1, 0, 0, 1, 2], dtype=np.int64),
        ["a", "b", "c", "d", "b"],
        np.arange(5),
    )
    matcher = FastBeamTreePathMatcher(
        mode="equality",
        beam_width=20,
        expansion_width=20,
        max_nodes_per_token_side=20,
        collect_diagnostics=True,
    )
    matcher.fit_encoder([G, H])
    prepared_g = matcher.prepare_tree(G)
    prepared_h = matcher.prepare_tree(H)

    direct_path, direct_score = matcher.predict(G, H)
    prepared_path, prepared_score = matcher.predict_prepared(prepared_g, prepared_h)
    assert prepared_path == direct_path
    assert prepared_score == direct_score

    diag = matcher.last_diagnostics_
    assert diag is not None
    assert diag.algorithm == "fast_beam_partial_equality"
    assert diag.states_generated is not None and diag.states_generated > 0
    assert diag.states_retained is not None and diag.states_retained > 0
    assert diag.extra["positive_token_types"] > 0
    assert diag.extra["candidate_semantics"] == "descending_token_weight_first_descendants"

    other = FastBeamTreePathMatcher(mode="equality")
    other.fit_encoder([G, H])
    with pytest.raises(ValueError, match="different FastLabelEncoder"):
        other.predict_prepared(prepared_g, prepared_h)



def test_fast_overlap_beam_finite_budget_reports_explicit_score() -> None:
    rng = np.random.default_rng(4073)
    for _ in range(100):
        n = int(rng.integers(5, 24))
        m = int(rng.integers(5, 24))
        alphabet = 9
        labels_g = [
            tuple(sorted(set(int(x) for x in rng.integers(alphabet, size=int(rng.integers(1, 5))))))
            for _ in range(n)
        ]
        labels_h = [
            tuple(sorted(set(int(x) for x in rng.integers(alphabet, size=int(rng.integers(1, 5))))))
            for _ in range(m)
        ]
        G = TreeData(_random_tree(n, rng), labels_g, np.arange(n))
        H = TreeData(_random_tree(m, rng), labels_h, np.arange(m))
        token_weights = {token: float(rng.integers(1, 8)) for token in range(alphabet)}

        matcher = FastBeamTreePathMatcher(
            mode="overlap",
            token_weights=token_weights,
            beam_width=17,
            expansion_width=13,
            max_nodes_per_token_side=4,
        )
        path, score = matcher.predict(G, H)

        def weight(a, b):
            common = set(a).intersection(b)
            return max((token_weights[token] for token in common), default=0.0)

        _assert_valid_score(G, H, path, score, weight)

def test_fast_beam_rejects_unsupported_search_and_bad_budgets() -> None:
    with pytest.raises(ValueError, match="partial"):
        FastBeamTreePathMatcher(search="local")
    with pytest.raises(ValueError, match="beam_width"):
        FastBeamTreePathMatcher(beam_width=0)
    with pytest.raises(ValueError, match="expansion_width"):
        FastBeamTreePathMatcher(expansion_width=0)
    with pytest.raises(ValueError, match="max_nodes_per_token_side"):
        FastBeamTreePathMatcher(max_nodes_per_token_side=0)
    with pytest.raises(ValueError, match="max_token_types_per_expansion"):
        FastBeamTreePathMatcher(max_token_types_per_expansion=0)
