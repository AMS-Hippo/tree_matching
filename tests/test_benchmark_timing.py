from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarking.path_match_benchmark import (
    default_algorithm_specs,
    run_algorithm_on_pair,
)
from benchmarking.path_match_cases import PairRegime, generate_synthetic_pair
from benchmarking.path_match_throughput import (
    ThroughputRegime,
    generate_throughput_corpus,
    run_algorithm_on_corpus,
)
from benchmarking.shared import load_config
from benchmarking.timing import (
    THREAD_ENVIRONMENT_VARIABLES,
    run_isolated_task,
    timing_sample_summary,
)


ROOT = Path(__file__).resolve().parents[1]


def _tiny_pair_regime() -> PairRegime:
    return PairRegime(
        name="isolated_pair",
        n_g=14,
        n_h=12,
        shape_g="medium",
        shape_h="medium",
        comparison="tree_tree",
        alphabet_size=6,
        symbols_per_node=1,
        score_mode="equality",
        planted_length=5,
        p_obs_g=1.0,
        p_obs_h=1.0,
        seed=911,
    )


def _tiny_throughput_regime() -> ThroughputRegime:
    return ThroughputRegime.from_mapping({
        "name": "isolated_throughput",
        "n_queries": 2,
        "n_templates": 2,
        "n_g": 14,
        "n_h": 7,
        "shape_g": "medium",
        "shape_h": "path",
        "comparison": "tree_path",
        "alphabet_size": 6,
        "symbols_per_node": 1,
        "score_mode": "equality",
        "weight_mode": "uniform",
        "planted_length": 5,
        "p_obs_g": 1.0,
        "p_obs_h": 1.0,
        "seed": 912,
    })


def test_timing_sample_summary_records_distribution_and_samples() -> None:
    summary = timing_sample_summary([1.0, 2.0, 3.0, 10.0], prefix="elapsed")

    assert summary["elapsed_median"] == 2.5
    assert summary["elapsed_min"] == 1.0
    assert summary["elapsed_max"] == 10.0
    assert summary["elapsed_q25"] == 1.75
    assert summary["elapsed_q75"] == 4.75
    assert summary["elapsed_sample_count"] == 4
    assert json.loads(summary["elapsed_samples"]) == [1.0, 2.0, 3.0, 10.0]


def test_isolated_task_enforces_timeout() -> None:
    result = run_isolated_task(
        "sleep",
        {"seconds": 5.0},
        timeout_seconds=0.2,
        thread_count=1,
    )

    assert result.status == "timeout"
    assert result.payload is None
    assert result.wall_seconds < 3.0


def test_pair_benchmark_isolated_worker_enforces_threads_and_reports_memory() -> None:
    pair = generate_synthetic_pair(_tiny_pair_regime(), 0)
    spec = default_algorithm_specs()["exact_dense"]

    row = run_algorithm_on_pair(
        pair,
        spec,
        repeats=2,
        warmup=0,
        execution_mode="isolated",
        timeout_seconds=30.0,
        thread_count=1,
    )

    assert row["status"] == "ok"
    assert row["execution_mode"] == "isolated"
    assert row["thread_settings_enforced"] is True
    assert row["fresh_process_total_seconds"] >= row["fresh_process_task_seconds"]
    assert row["cold_fit_seconds"] >= 0.0
    assert row["cold_predict_seconds"] >= 0.0
    assert row["cold_setup_plus_predict_seconds"] == (
        row["cold_fit_seconds"] + row["cold_predict_seconds"]
    )
    assert abs(row["cold_score_difference_from_warm"]) <= 1e-6
    assert row["predict_seconds_sample_count"] == 2
    assert row["predict_seconds_q25"] <= row["predict_seconds_median"]
    assert row["predict_seconds_median"] <= row["predict_seconds_q75"]
    assert row["peak_rss_bytes"] is None or int(row["peak_rss_bytes"]) > 0

    worker_env = json.loads(row["worker_thread_environment"])
    for name in THREAD_ENVIRONMENT_VARIABLES:
        assert worker_env[name] == "1"


def test_throughput_benchmark_isolated_worker_returns_expected_matrix() -> None:
    corpus = generate_throughput_corpus(_tiny_throughput_regime())
    spec = default_algorithm_specs()["fast_dense"]

    row, matrix = run_algorithm_on_corpus(
        corpus,
        spec,
        repeats=2,
        warmup_pairs=0,
        validation_pairs=2,
        diagnostic_pairs=1,
        execution_mode="isolated",
        timeout_seconds=30.0,
        thread_count=1,
    )

    assert row["status"] == "ok"
    assert matrix is not None
    assert matrix.shape == (2, 2)
    assert np.all(np.isfinite(matrix))
    assert row["score_matrix_seconds_sample_count"] == 2
    assert row["cold_one_time_preparation_seconds"] >= 0.0
    assert row["cold_score_matrix_seconds"] >= 0.0
    assert row["cold_setup_plus_one_matrix_seconds"] == (
        row["cold_one_time_preparation_seconds"] + row["cold_score_matrix_seconds"]
    )
    assert row["cold_matrix_max_abs_difference_from_warm"] <= 1e-6
    assert row["fresh_process_total_seconds"] >= row["fresh_process_task_seconds"]
    assert row["thread_settings_enforced"] is True


def test_standard_presets_use_isolated_single_thread_workers() -> None:
    pair_config = load_config(ROOT / "experiments" / "path_match_benchmark_full.json")
    throughput_config = load_config(
        ROOT / "experiments" / "path_match_throughput_full.json"
    )

    assert pair_config["execution_mode"] == "isolated"
    assert pair_config["thread_count"] == 1
    assert pair_config["timeout_seconds"] > 0
    assert throughput_config["execution_mode"] == "isolated"
    assert throughput_config["thread_count"] == 1
    assert throughput_config["timeout_seconds"] > 0
