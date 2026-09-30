from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarking.path_match_throughput import (
    ThroughputRegime,
    generate_throughput_corpus,
    run_throughput_config,
)
from benchmarking.shared import load_config


ROOT = Path(__file__).resolve().parents[1]


def _tiny_equality_regime() -> dict:
    return {
        "name": "tiny_throughput_equality",
        "n_queries": 3,
        "n_templates": 2,
        "n_g": 24,
        "n_h": 9,
        "shape_g": "medium",
        "shape_h": "path",
        "comparison": "tree_path",
        "alphabet_size": 7,
        "symbol_distribution": "uniform",
        "symbols_per_node": 1,
        "score_mode": "equality",
        "weight_mode": "uniform",
        "planted_length": 5,
        "p_obs_g": 0.9,
        "p_obs_h": 1.0,
        "planted_path_selection": "random_leaf",
        "seed": 8701,
    }


def test_throughput_corpus_is_deterministic_and_uses_path_templates() -> None:
    regime = ThroughputRegime.from_mapping(_tiny_equality_regime())
    first = generate_throughput_corpus(regime)
    second = generate_throughput_corpus(regime)

    assert len(first.queries) == 3
    assert len(first.templates) == 2
    assert first.n_pairs == 6
    assert first.score_model == second.score_model

    for tree_a, tree_b in zip(first.queries, second.queries):
        assert np.array_equal(tree_a.parent, tree_b.parent)
        assert tree_a.label == tree_b.label
    for tree_a, tree_b in zip(first.templates, second.templates):
        assert np.array_equal(tree_a.parent, tree_b.parent)
        assert tree_a.label == tree_b.label
        assert np.array_equal(tree_a.parent, np.asarray([-1] + list(range(tree_a.n - 1))))

    assert first.metadata["dense_cells_total"] == (
        sum(tree.n for tree in first.queries)
        * sum(tree.n for tree in first.templates)
    )
    assert first.metadata["posting_hits_total"] >= 0


def test_small_throughput_run_validates_prepared_apis_and_exact_matrices(tmp_path: Path) -> None:
    config = {
        "suite_name": "unit_throughput",
        "repeats": 1,
        "warmup_pairs": 1,
        "validation_pairs": 3,
        "diagnostic_pairs": 1,
        "quality_floor": 0.90,
        "exact_score_abs_tol": 1e-5,
        "exact_score_rel_tol": 1e-6,
        "algorithm_order_seed": 12,
        "max_total_generic_dense_cells": 100_000,
        "max_total_fast_dense_cells": 100_000,
        "max_total_sparse_posting_hits": 100_000,
        "algorithms": [
            "exact_dense",
            "fast_dense",
            "sparse_chain",
            "fast_sparse",
            "fast_beam_partial",
            "beam_local",
            "beam_partial_score",
        ],
        "regimes": [_tiny_equality_regime()],
    }

    rows, summary, metadata = run_throughput_config(config, output_dir=tmp_path)

    assert len(rows) == len(config["algorithms"])
    assert set(rows["status"]) == {"ok"}
    assert set(rows["oracle_status"]) == {"exact_agreement"}
    assert rows["oracle_valid"].all()
    assert rows["validation_paths_valid"].all()
    assert rows["validation_reported_scores_valid"].all()
    assert (rows["repeat_matrix_max_abs_difference"] == 0.0).all()

    exact = rows[rows["algorithm_exact"] == True]  # noqa: E712
    assert len(exact) == 4
    assert float(exact["exact_matrix_max_abs_difference"].max()) <= 1e-5
    assert np.allclose(exact["matrix_total_accuracy"].to_numpy(dtype=float), 1.0)

    prepared = rows[rows["prepared_validation_applicable"] == True]  # noqa: E712
    assert set(prepared["algorithm"]) == {
        "fast_dense",
        "sparse_chain",
        "fast_sparse",
        "fast_beam_partial",
    }
    assert prepared["prepared_validation_passed"].all()
    assert np.allclose(
        prepared["prepared_validation_max_score_difference"].to_numpy(dtype=float),
        0.0,
    )

    assert rows["pairs_per_second"].notna().all()
    assert rows["one_time_preparation_seconds"].notna().all()
    assert not summary.empty
    assert "throughput_frontier" in summary
    assert metadata["config_hash"].startswith("throughput_")

    for filename in (
        "throughput_rows.csv",
        "throughput_summary.csv",
        "throughput_metadata.json",
        "throughput_score_matrices.npz",
    ):
        assert (tmp_path / filename).exists()

    metadata_payload = json.loads((tmp_path / "throughput_metadata.json").read_text())
    assert metadata_payload["suite_name"] == "unit_throughput"
    assert metadata_payload["matrix_archive"]["keys"]

    archive = np.load(tmp_path / "throughput_score_matrices.npz")
    assert len(archive.files) == len(config["algorithms"])
    assert all(archive[key].shape == (3, 2) for key in archive.files)


def test_throughput_jaccard_skips_specialized_matchers(tmp_path: Path) -> None:
    regime = _tiny_equality_regime()
    regime.update({
        "name": "tiny_throughput_jaccard",
        "score_mode": "jaccard",
        "alphabet_size": 20,
        "symbols_per_node": 2,
        "seed": 8702,
    })
    config = {
        "suite_name": "unit_throughput_jaccard",
        "repeats": 1,
        "warmup_pairs": 0,
        "validation_pairs": 2,
        "diagnostic_pairs": 0,
        "algorithms": ["exact_dense", "fast_dense", "sparse_chain", "fast_sparse"],
        "regimes": [regime],
    }

    rows, _, _ = run_throughput_config(config, output_dir=tmp_path)
    by_name = rows.set_index("algorithm")

    assert by_name.loc["exact_dense", "status"] == "ok"
    assert by_name.loc["sparse_chain", "status"] == "ok"
    assert by_name.loc["fast_dense", "status"] == "skipped"
    assert by_name.loc["fast_sparse", "status"] == "skipped"
    assert str(by_name.loc["fast_dense", "skip_reason"]).startswith(
        "incompatible_score_mode"
    )
    assert by_name.loc["exact_dense", "oracle_status"] == "exact_agreement"
    assert by_name.loc["sparse_chain", "matrix_total_accuracy"] == 1.0


def test_standard_throughput_config_has_the_three_planned_regimes() -> None:
    config = load_config(ROOT / "experiments" / "path_match_throughput_full.json")
    regimes = config["regimes"]

    assert len(regimes) == 3
    assert {item["score_mode"] for item in regimes} == {"equality", "overlap", "jaccard"}
    assert {item["comparison"] for item in regimes} == {"tree_path", "tree_tree"}
    by_mode = {item["score_mode"]: item for item in regimes}
    assert by_mode["equality"]["shape_h"] == "path"
    assert by_mode["jaccard"]["shape_h"] == "path"
    assert by_mode["overlap"]["comparison"] == "tree_tree"
    assert by_mode["overlap"]["n_h"] >= 1000
    assert all(item["n_queries"] > item["n_templates"] for item in regimes)
    assert all(item.get("expected_algorithm") for item in regimes)
    assert config["max_total_generic_dense_cells"] < config["max_total_fast_dense_cells"]
