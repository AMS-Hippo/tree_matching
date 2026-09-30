from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarking.path_match_benchmark import (
    add_oracle_columns,
    default_algorithm_specs,
    resolve_algorithm_specs,
    run_benchmark_config,
    summarize_benchmark_rows,
)
from benchmarking.path_match_cases import PairRegime, generate_synthetic_pair
from benchmarking.shared import load_config


ROOT = Path(__file__).resolve().parents[1]


def _is_strict_ancestor(parent: np.ndarray, u: int, v: int) -> bool:
    x = int(v)
    while x >= 0:
        x = int(parent[x])
        if x == int(u):
            return True
    return False


def test_generator_is_deterministic_and_supports_tree_to_path_multisymbol() -> None:
    regime = PairRegime(
        name="tree_path_test",
        n_g=80,
        n_h=20,
        shape_g="narrow",
        shape_h="medium",  # comparison must force this to a path
        comparison="tree_path",
        alphabet_size=15,
        symbol_distribution="zipf",
        zipf_exponent=1.4,
        symbols_per_node={
            "kind": "categorical",
            "values": [1, 2, 3],
            "probs": [0.5, 0.35, 0.15],
        },
        score_mode="jaccard",
        planted_length=10,
        planted_path_selection="random_leaf",
        seed=456,
    )
    first = generate_synthetic_pair(regime, 2)
    second = generate_synthetic_pair(regime, 2)

    assert regime.shape_h == "path"
    assert np.array_equal(first.G.parent, second.G.parent)
    assert np.array_equal(first.H.parent, second.H.parent)
    assert first.G.label == second.G.label
    assert first.H.label == second.H.label
    assert first.truth_pairs == second.truth_pairs

    assert np.array_equal(first.H.parent, np.asarray([-1] + list(range(regime.n_h - 1))))
    assert all(isinstance(label, tuple) for label in first.G.label)
    assert all(1 <= len(label) <= 4 for label in first.G.label)
    assert first.metadata["dense_cells"] == regime.n_g * regime.n_h
    assert first.metadata["posting_hits"] >= len(first.truth_pairs)

    for (u1, v1), (u2, v2) in zip(first.truth_pairs, first.truth_pairs[1:]):
        assert _is_strict_ancestor(first.G.parent, u1, u2)
        assert _is_strict_ancestor(first.H.parent, v1, v2)


def test_tail_planted_tokens_do_not_mutate_background_probabilities() -> None:
    regime = PairRegime(
        name="tail_tokens",
        n_g=50,
        n_h=45,
        alphabet_size=200,
        symbols_per_node={"kind": "categorical", "values": [1, 2], "probs": [0.8, 0.2]},
        score_mode="jaccard",
        planted_length=8,
        planted_token_policy="tail",
        seed=99,
    )
    pair = generate_synthetic_pair(regime)
    probabilities = np.asarray(pair.metadata["background_probabilities"], dtype=float)
    assert np.isclose(float(probabilities.sum()), 1.0)
    assert np.all(probabilities >= 0.0)


def test_wide_generator_randomizes_structurally_special_branch_order() -> None:
    regime = PairRegime(
        name="wide_order_randomization",
        n_g=450,
        n_h=30,
        shape_g="wide",
        shape_h="path",
        comparison="tree_path",
        alphabet_size=10,
        symbols_per_node=1,
        score_mode="equality",
        planted_length=7,
        planted_path_selection="random_deep_leaf",
        seed=123,
    )

    root_child_ranks = []
    for instance in range(40):
        pair = generate_synthetic_pair(regime, instance)
        root_children = [
            u for u in range(1, pair.G.n) if int(pair.G.parent[u]) == 0
        ]
        first_path_child = int(pair.metadata["candidate_planted_path_g"][1])
        root_child_ranks.append(root_children.index(first_path_child))
        assert pair.metadata["tree_g"]["node_order_randomized_within_levels"] is True

    # The old deterministic construction always placed the useful branch in
    # the first 16 children.  Fixed-seed randomization now exercises both sides
    # of that cap and a broad range of sibling positions.
    assert min(root_child_ranks) < 16
    assert max(root_child_ranks) >= 16
    assert len(set(root_child_ranks)) >= 20


def test_named_algorithm_presets_pin_score_only_beam_parameters() -> None:
    specs = default_algorithm_specs()
    assert {
        "exact_dense",
        "fast_dense",
        "sparse_closure",
        "sparse_chain",
        "fast_sparse",
        "beam_local",
        "beam_local_capped",
        "beam_partial_score",
        "beam_partial_heuristic",
    }.issubset(specs)

    score_only = specs["beam_partial_score"].kwargs
    assert score_only["beam_width"] == 200
    assert score_only["beam_expansion_width"] == 64
    assert score_only["beam_random_fraction"] == 0.0
    assert score_only["beam_rarity_weight"] == 0.0
    assert score_only["beam_priority_future_weight"] == 0.0
    assert score_only["beam_lookahead"] is False
    assert score_only["candidate_select_mode"] == "first"


def test_algorithm_overrides_can_be_given_unique_aliases() -> None:
    specs = resolve_algorithm_specs([
        {"name": "beam_partial_score", "alias": "beam_B50", "kwargs": {"beam_width": 50}},
        {"name": "beam_partial_score", "alias": "beam_B500", "kwargs": {"beam_width": 500}},
    ])
    assert [spec.name for spec in specs] == ["beam_B50", "beam_B500"]
    assert [spec.kwargs["beam_width"] for spec in specs] == [50, 500]

    with pytest.raises(ValueError, match="unique"):
        resolve_algorithm_specs(["exact_dense", "exact_dense"])


def test_small_benchmark_run_writes_outputs_and_exact_methods_agree(tmp_path: Path) -> None:
    config = {
        "suite_name": "unit_path_match",
        "repeats": 1,
        "warmup": 0,
        "quality_floor": 0.99,
        "max_dense_cells": 100_000,
        "max_sparse_posting_hits": 100_000,
        "algorithms": [
            "exact_dense",
            "fast_dense",
            "sparse_closure",
            "sparse_chain",
            "fast_sparse",
            "beam_local",
            "beam_partial_score",
        ],
        "regimes": [
            {
                "name": "tiny_equality",
                "n_g": 18,
                "n_h": 16,
                "shape_g": "medium",
                "shape_h": "medium",
                "alphabet_size": 6,
                "symbols_per_node": 1,
                "score_mode": "equality",
                "planted_length": 6,
                "p_obs_g": 1.0,
                "p_obs_h": 1.0,
                "n_instances": 1,
                "seed": 11,
            }
        ],
    }

    rows, summary, metadata = run_benchmark_config(config, output_dir=tmp_path)

    assert len(rows) == len(config["algorithms"])
    assert set(rows["status"]) == {"ok"}
    exact = rows[rows["algorithm_exact"] == True]  # noqa: E712
    assert len(exact) == 5
    assert float(exact["score"].max() - exact["score"].min()) <= 1e-5
    assert float(rows["exact_score_spread"].dropna().max()) <= 1e-5
    assert rows["score_ratio"].notna().all()
    assert rows["accuracy_score"].notna().all()
    assert rows["warm_timing_score"].notna().all()
    assert rows["setup_timing_score"].notna().all()
    assert "setup_plus_warm_search_seconds" in rows
    assert "cold_seconds_estimate" not in rows
    assert metadata["config_hash"].startswith("pathmatch_")
    assert not summary.empty
    assert summary["complete_coverage"].all()

    assert (tmp_path / "benchmark_rows.csv").exists()
    assert (tmp_path / "benchmark_summary.csv").exists()
    metadata_path = tmp_path / "benchmark_metadata.json"
    assert metadata_path.exists()
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert payload["suite_name"] == "unit_path_match"


def test_generic_jaccard_skips_specialized_matchers_but_exact_sparse_agrees(tmp_path: Path) -> None:
    config = {
        "suite_name": "unit_jaccard",
        "repeats": 1,
        "warmup": 0,
        "algorithms": ["exact_dense", "fast_dense", "sparse_chain", "fast_sparse"],
        "regimes": [
            {
                "name": "tiny_jaccard",
                "n_g": 14,
                "n_h": 13,
                "shape_g": "medium",
                "shape_h": "medium",
                "alphabet_size": 12,
                "symbols_per_node": 2,
                "score_mode": "jaccard",
                "planted_length": 5,
                "n_instances": 1,
                "seed": 21,
            }
        ],
    }
    rows, _, _ = run_benchmark_config(config, output_dir=tmp_path)
    by_name = rows.set_index("algorithm")

    assert by_name.loc["fast_dense", "status"] == "skipped"
    assert by_name.loc["fast_sparse", "status"] == "skipped"
    assert str(by_name.loc["fast_dense", "skip_reason"]).startswith("incompatible_score_mode")
    assert by_name.loc["exact_dense", "status"] == "ok"
    assert by_name.loc["sparse_chain", "status"] == "ok"
    assert abs(float(by_name.loc["exact_dense", "score"]) - float(by_name.loc["sparse_chain", "score"])) <= 1e-5


def test_full_config_covers_requested_benchmark_axes() -> None:
    config = load_config(ROOT / "experiments" / "path_match_benchmark_full.json")
    regimes = config["regimes"]

    shapes = {item["shape_g"] for item in regimes}.union({item["shape_h"] for item in regimes})
    comparisons = {item["comparison"] for item in regimes}
    distributions = {item["symbol_distribution"] for item in regimes}
    score_modes = {item["score_mode"] for item in regimes}

    assert {"narrow", "medium", "wide", "path"}.issubset(shapes)
    assert comparisons == {"tree_tree", "tree_path"}
    assert distributions == {"uniform", "zipf"}
    assert {"equality", "overlap", "jaccard"}.issubset(score_modes)
    assert any(isinstance(item["symbols_per_node"], dict) for item in regimes)
    assert all(item.get("expected_algorithm") for item in regimes)

    by_name = {item["name"]: item for item in regimes}
    assert {"rare_anchor_oracle", "deep_rare_anchor_stress"}.issubset(by_name)

    oracle = by_name["rare_anchor_oracle"]
    stress = by_name["deep_rare_anchor_stress"]
    assert oracle["n_g"] * oracle["n_h"] <= int(config["max_dense_cells"])
    assert stress["n_g"] * stress["n_h"] > int(config["max_dense_cells"])
    assert oracle["planted_token_policy"] == "reserved"
    assert oracle["planted_distinct_tokens"] == oracle["planted_length"]

    # The companion regime must also remain within the configured exhaustive
    # sparse-candidate budget.  This is a deterministic generation check, not a
    # timing or accuracy assertion.
    pair = generate_synthetic_pair(PairRegime.from_mapping(oracle), 0)
    assert int(pair.metadata["posting_hits"]) <= int(config["max_sparse_posting_hits"])


def test_exact_disagreement_invalidates_accuracy_oracle() -> None:
    rows = pd.DataFrame([
        {
            "regime": "r",
            "instance": 0,
            "algorithm": "exact_a",
            "algorithm_exact": True,
            "status": "ok",
            "score": 10.0,
            "predict_seconds_median": 1.0,
            "setup_plus_warm_search_seconds": 1.2,
        },
        {
            "regime": "r",
            "instance": 0,
            "algorithm": "exact_b",
            "algorithm_exact": True,
            "status": "ok",
            "score": 9.0,
            "predict_seconds_median": 0.8,
            "setup_plus_warm_search_seconds": 1.0,
        },
        {
            "regime": "r",
            "instance": 0,
            "algorithm": "beam",
            "algorithm_exact": False,
            "status": "ok",
            "score": 8.0,
            "predict_seconds_median": 0.1,
            "setup_plus_warm_search_seconds": 0.2,
        },
    ])

    scored = add_oracle_columns(rows, exact_abs_tol=1e-8, exact_rel_tol=0.0)
    assert set(scored["oracle_status"]) == {"exact_disagreement"}
    assert not scored["oracle_valid"].any()
    assert scored["accuracy_score"].isna().all()
    assert scored["score_ratio"].isna().all()


def test_summary_requires_completion_and_never_uses_planted_recall_as_accuracy() -> None:
    regime = PairRegime(
        name="coverage",
        n_g=4,
        n_h=4,
        alphabet_size=2,
        symbols_per_node=1,
        score_mode="equality",
        planted_length=2,
        n_instances=2,
    )
    rows = pd.DataFrame([
        {
            "regime": "coverage",
            "instance": instance,
            "algorithm": "exact",
            "algorithm_config_index": 0,
            "algorithm_exact": True,
            "status": "ok",
            "score": 10.0,
            "predict_seconds_median": 1.0,
            "setup_plus_warm_search_seconds": 1.1,
            "truth_pair_recall": 1.0,
        }
        for instance in range(2)
    ] + [
        {
            "regime": "coverage",
            "instance": 0,
            "algorithm": "beam",
            "algorithm_config_index": 1,
            "algorithm_exact": False,
            "status": "ok",
            "score": 10.0,
            "predict_seconds_median": 0.1,
            "setup_plus_warm_search_seconds": 0.2,
            "truth_pair_recall": 1.0,
        },
        {
            "regime": "coverage",
            "instance": 1,
            "algorithm": "beam",
            "algorithm_config_index": 1,
            "algorithm_exact": False,
            "status": "error",
            "score": None,
            "predict_seconds_median": None,
            "setup_plus_warm_search_seconds": None,
            "truth_pair_recall": 1.0,
        },
    ])
    scored = add_oracle_columns(rows)
    summary = summarize_benchmark_rows(scored, [regime], quality_floor=0.99)
    by_algorithm = summary.set_index("algorithm")
    assert bool(by_algorithm.loc["exact", "quality_eligible"])
    assert not bool(by_algorithm.loc["beam", "complete_coverage"])
    assert not bool(by_algorithm.loc["beam", "quality_eligible"])

    no_oracle = rows[rows["algorithm"] == "beam"].copy()
    no_oracle.loc[:, "status"] = "ok"
    no_oracle.loc[:, "score"] = 10.0
    no_oracle.loc[:, "predict_seconds_median"] = 0.1
    no_oracle.loc[:, "setup_plus_warm_search_seconds"] = 0.2
    scored_no_oracle = add_oracle_columns(no_oracle)
    summary_no_oracle = summarize_benchmark_rows(
        scored_no_oracle,
        [regime],
        quality_floor=0.99,
    )
    assert summary_no_oracle.loc[0, "quality_basis"] is None
    assert not bool(summary_no_oracle.loc[0, "quality_eligible"])


def test_algorithm_execution_order_is_cyclically_balanced(tmp_path: Path) -> None:
    config = {
        "suite_name": "balanced_order",
        "repeats": 1,
        "warmup": 0,
        "algorithm_order_seed": 73,
        "algorithms": ["exact_dense", "sparse_chain", "beam_local"],
        "regimes": [{
            "name": "tiny",
            "n_g": 8,
            "n_h": 8,
            "shape_g": "medium",
            "shape_h": "medium",
            "alphabet_size": 4,
            "symbols_per_node": 1,
            "score_mode": "equality",
            "planted_length": 4,
            "n_instances": 3,
            "seed": 19,
        }],
    }
    rows, _, metadata = run_benchmark_config(config, output_dir=tmp_path)
    for _, group in rows.groupby("algorithm"):
        assert set(group["execution_order_position"]) == {0, 1, 2}
    assert metadata["algorithm_order_strategy"].startswith("seeded_base_permutation")
