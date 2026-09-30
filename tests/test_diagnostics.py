"""Regression tests for opt-in matcher diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from path_matcher import MatchDiagnostics
from path_matcher.beam_align import align_trees_beam, align_trees_beam_symmetric
from path_matcher.fast_match import FastTreePathMatcher
from path_matcher.fast_sparse_match import FastSparseTreePathMatcher
from path_matcher.local_beam_align import align_trees_local_beam
from path_matcher.matcher import TreePathMatcher
from path_matcher.needleman_wunsch_tree import align_trees_algorithm1
from path_matcher.sparse_align import SparseCandidateConfig, align_trees_sparse_candidates
from path_matcher.sparse_chain import align_trees_sparse_chain
from path_matcher.sparse_preprocess import preprocess_treedata
from path_matcher.tree_data import TreeData


def _tree(parent, labels):
    return TreeData(
        parent=np.asarray(parent, dtype=np.int32),
        label=list(labels),
        orig_index=np.arange(len(labels), dtype=np.int32),
    )


def _example_trees():
    G = _tree([-1, 0, 0, 1, 1], ["root-G", "A", "x", "B", "C"])
    H = _tree([-1, 0, 0, 1, 2], ["root-H", "A", "A", "B", "C"])
    return G, H


def _assert_common(diag: MatchDiagnostics, *, algorithm: str, n_g: int, n_h: int, score: float, length: int):
    assert diag.algorithm == algorithm
    assert diag.n_g == n_g
    assert diag.n_h == n_h
    assert diag.result_score == pytest.approx(score)
    assert diag.result_length == length
    assert diag.total_seconds >= 0.0
    assert diag.preprocessing_seconds >= 0.0
    assert diag.candidate_generation_seconds >= 0.0
    assert diag.search_seconds >= 0.0
    assert diag.traceback_seconds >= 0.0
    assert diag.as_dict()["algorithm"] == algorithm


def test_dense_exact_diagnostics_count_cells_and_scores():
    G, H = _example_trees()
    diag = MatchDiagnostics()

    result = align_trees_algorithm1(G, H, diagnostics=diag, prefer_match_on_tie=False)

    _assert_common(
        diag,
        algorithm="exact_dense",
        n_g=G.n,
        n_h=H.n,
        score=result.score,
        length=len(result.path_internal),
    )
    assert diag.dp_cells_computed == G.n * H.n
    assert diag.node_pair_score_evaluations == G.n * H.n
    assert diag.positive_candidate_pairs == 4
    assert diag.candidate_pairs_generated is None


def test_local_beam_diagnostics_report_expansion_and_frontier_work():
    G, H = _example_trees()
    diag = MatchDiagnostics()

    result = align_trees_local_beam(G, H, beam_width=3, diagnostics=diag)

    _assert_common(
        diag,
        algorithm="beam_local",
        n_g=G.n,
        n_h=H.n,
        score=result.score,
        length=len(result.path_internal),
    )
    assert diag.candidate_pairs_generated == diag.node_pair_score_evaluations
    assert diag.candidate_pairs_generated is not None and diag.candidate_pairs_generated > 0
    assert diag.states_generated is not None and diag.states_generated >= diag.candidate_pairs_generated
    assert diag.states_after_endpoint_pruning is not None
    assert diag.states_retained is not None
    assert diag.states_retained <= diag.states_after_endpoint_pruning
    assert diag.peak_frontier_size is not None and diag.peak_frontier_size <= 3
    assert diag.layers_processed is not None and diag.layers_processed > 0
    assert diag.extra["beam_width"] == 3


def test_partial_beam_diagnostics_aggregate_restarts_and_directions():
    G, H = _example_trees()
    common = dict(
        beam_width=20,
        expansion_width=None,
        max_label_pair_scan=10_000,
        max_label_pairs_per_expansion=None,
        max_nodes_per_label_side=20,
        candidate_select_mode="first",
        random_fraction=0.0,
        rarity_weight=0.0,
        gap_penalty=0.0,
        balance_penalty=0.0,
        candidate_future_weight=0.0,
        priority_future_weight=0.0,
    )

    restart_diag = MatchDiagnostics()
    restart_result = align_trees_beam(G, H, n_restarts=2, diagnostics=restart_diag, **common)
    _assert_common(
        restart_diag,
        algorithm="beam_partial",
        n_g=G.n,
        n_h=H.n,
        score=restart_result.score,
        length=len(restart_result.path_internal),
    )
    assert restart_diag.restarts == 2
    assert restart_diag.directions == 1
    assert restart_diag.states_generated == restart_diag.candidate_pairs_generated
    assert restart_diag.states_retained is not None and restart_diag.states_retained > 0
    assert restart_diag.extra["selected_restart"] in {0, 1}
    assert len(restart_diag.extra["restart_scores"]) == 2
    assert "selected_run" in restart_diag.extra

    symmetric_diag = MatchDiagnostics()
    symmetric_result = align_trees_beam_symmetric(G, H, diagnostics=symmetric_diag, **common)
    _assert_common(
        symmetric_diag,
        algorithm="beam_partial_symmetric",
        n_g=G.n,
        n_h=H.n,
        score=symmetric_result.score,
        length=len(symmetric_result.path_internal),
    )
    assert symmetric_diag.directions == 2
    assert symmetric_diag.extra["selected_direction"] in {"forward", "reverse"}
    assert "forward_run" in symmetric_diag.extra
    assert "reverse_run" in symmetric_diag.extra


def test_sparse_diagnostics_distinguish_candidates_from_dp_closure():
    G, H = _example_trees()
    matcher = TreePathMatcher(method="sparse_chain")
    preG = matcher.preprocess(G)
    preH = matcher.preprocess(H)
    cfg = SparseCandidateConfig.exhaustive()

    closure_diag = MatchDiagnostics()
    closure = align_trees_sparse_candidates(
        preG,
        preH,
        cfg=cfg,
        prefer_match_on_tie=False,
        diagnostics=closure_diag,
    )
    _assert_common(
        closure_diag,
        algorithm="sparse_closure",
        n_g=G.n,
        n_h=H.n,
        score=closure.score,
        length=len(closure.path_internal),
    )
    assert closure_diag.candidate_pairs_generated == 4
    assert closure_diag.node_pair_score_evaluations == 4
    assert closure_diag.dp_cells_computed is not None
    assert closure_diag.dp_cells_computed >= closure_diag.candidate_pairs_generated

    chain_diag = MatchDiagnostics()
    chain = align_trees_sparse_chain(preG, preH, cfg=cfg, diagnostics=chain_diag)
    _assert_common(
        chain_diag,
        algorithm="sparse_chain",
        n_g=G.n,
        n_h=H.n,
        score=chain.score,
        length=len(chain.path_internal),
    )
    assert chain_diag.candidate_pairs_generated == 4
    assert chain_diag.node_pair_score_evaluations == 4
    assert chain_diag.positive_candidate_pairs == 4
    assert chain_diag.dp_cells_computed is None
    assert chain_diag.extra["candidate_states_created"] == 4
    assert chain_diag.extra["segment_tree_point_queries"] == 4


def test_high_level_diagnostics_are_opt_in_and_record_fit_separately():
    G, H = _example_trees()

    plain = TreePathMatcher(method="exact")
    plain.predict(G, H)
    assert plain.last_diagnostics_ is None
    assert plain.last_fit_diagnostics_ is None

    matcher = TreePathMatcher(method="exact", collect_diagnostics=True)
    matcher.fit(G, H)
    assert matcher.last_fit_diagnostics_ is not None
    assert matcher.last_fit_diagnostics_.algorithm == "exact_fit"

    pairs, score = matcher.predict()
    assert pairs
    assert score > 0.0
    assert matcher.last_diagnostics_ is not None
    assert matcher.last_diagnostics_.algorithm == "exact_dense"
    assert matcher.last_diagnostics_.input_preprocessing_seconds == 0.0
    assert matcher.last_diagnostics_.extra["used_fitted_inputs"] is True


def test_fast_matcher_diagnostics_cover_dense_and_sparse_paths():
    G, H = _example_trees()

    dense = FastTreePathMatcher(mode="equality", collect_diagnostics=True)
    dense.fit(G, H)
    dense_pairs, dense_score = dense.predict()
    assert dense.last_fit_diagnostics_ is not None
    assert dense.last_diagnostics_ is not None
    _assert_common(
        dense.last_diagnostics_,
        algorithm="fast_dense_equality",
        n_g=G.n,
        n_h=H.n,
        score=dense_score,
        length=len(dense_pairs),
    )
    assert dense.last_diagnostics_.dp_cells_computed == G.n * H.n
    assert dense.last_diagnostics_.node_pair_score_evaluations == G.n * H.n

    sparse = FastSparseTreePathMatcher(mode="equality", collect_diagnostics=True)
    sparse.fit(G, H)
    sparse_pairs, sparse_score = sparse.predict()
    assert sparse.last_fit_diagnostics_ is not None
    assert sparse.last_diagnostics_ is not None
    _assert_common(
        sparse.last_diagnostics_,
        algorithm="fast_sparse_equality",
        n_g=G.n,
        n_h=H.n,
        score=sparse_score,
        length=len(sparse_pairs),
    )
    assert sparse_score == pytest.approx(dense_score)
    assert sparse.last_diagnostics_.candidate_pairs_generated == 4
    assert sparse.last_diagnostics_.extra["posting_hits"] == 4


def test_diagnostics_do_not_change_search_results_randomized():
    """Instrumentation must remain observational for every principal Python path."""
    rng = np.random.default_rng(20260924)
    beam_kwargs = dict(
        beam_width=8,
        expansion_width=10,
        max_label_pair_scan=10_000,
        max_label_pairs_per_expansion=None,
        max_nodes_per_label_side=20,
        candidate_select_mode="first",
        random_fraction=0.0,
        rarity_weight=0.0,
        gap_penalty=0.0,
        balance_penalty=0.0,
        candidate_future_weight=0.0,
        priority_future_weight=0.0,
    )

    for _ in range(25):
        n = int(rng.integers(1, 10))
        m = int(rng.integers(1, 10))
        parent_g = np.empty(n, dtype=np.int32)
        parent_h = np.empty(m, dtype=np.int32)
        parent_g[0] = -1
        parent_h[0] = -1
        for u in range(1, n):
            parent_g[u] = int(rng.integers(0, u))
        for v in range(1, m):
            parent_h[v] = int(rng.integers(0, v))
        G = _tree(parent_g, rng.integers(0, 4, size=n).tolist())
        H = _tree(parent_h, rng.integers(0, 4, size=m).tolist())

        low_level_calls = [
            lambda d: align_trees_algorithm1(G, H, prefer_match_on_tie=False, diagnostics=d),
            lambda d: align_trees_local_beam(G, H, beam_width=8, diagnostics=d),
            lambda d: align_trees_beam(G, H, diagnostics=d, **beam_kwargs),
        ]
        for call in low_level_calls:
            plain = call(None)
            record = MatchDiagnostics()
            measured = call(record)
            assert measured.path_internal == plain.path_internal
            assert measured.score == pytest.approx(plain.score)

        sparse_matcher = TreePathMatcher(method="sparse_chain")
        pre_g = sparse_matcher.preprocess(G)
        pre_h = sparse_matcher.preprocess(H)
        cfg = SparseCandidateConfig.exhaustive()
        sparse_calls = [
            lambda d: align_trees_sparse_candidates(
                pre_g,
                pre_h,
                cfg=cfg,
                prefer_match_on_tie=False,
                diagnostics=d,
            ),
            lambda d: align_trees_sparse_chain(pre_g, pre_h, cfg=cfg, diagnostics=d),
        ]
        for call in sparse_calls:
            plain = call(None)
            record = MatchDiagnostics()
            measured = call(record)
            assert measured.path_internal == plain.path_internal
            assert measured.score == pytest.approx(plain.score)
