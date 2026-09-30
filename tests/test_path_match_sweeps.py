from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from benchmarking import PairRegime, ThroughputRegime, load_config, resolve_algorithm_specs, run_sweep_config


ROOT = Path(__file__).resolve().parents[1]


def test_large_configs_are_valid_and_larger_than_standard() -> None:
    standard = load_config(ROOT / "experiments" / "path_match_benchmark_full.json")
    large = load_config(ROOT / "experiments" / "path_match_benchmark_large.json")
    resolve_algorithm_specs(large["algorithms"])
    standard_by_name = {item["name"]: PairRegime.from_mapping(item) for item in standard["regimes"]}
    large_by_name = {item["name"]: PairRegime.from_mapping(item) for item in large["regimes"]}
    assert set(standard_by_name) == set(large_by_name)
    assert int(large["repeats"]) > int(standard["repeats"])
    assert all(large_by_name[name].n_instances >= standard_by_name[name].n_instances for name in standard_by_name)
    assert any(large_by_name[name].n_g > standard_by_name[name].n_g for name in standard_by_name)

    standard_t = load_config(ROOT / "experiments" / "path_match_throughput_full.json")
    large_t = load_config(ROOT / "experiments" / "path_match_throughput_large.json")
    resolve_algorithm_specs(large_t["algorithms"])
    standard_t_by_name = {item["name"]: ThroughputRegime.from_mapping(item) for item in standard_t["regimes"]}
    large_t_by_name = {item["name"]: ThroughputRegime.from_mapping(item) for item in large_t["regimes"]}
    assert set(standard_t_by_name) == set(large_t_by_name)
    assert int(large_t["repeats"]) > int(standard_t["repeats"])
    assert all(large_t_by_name[name].n_queries >= standard_t_by_name[name].n_queries for name in standard_t_by_name)
    assert all(large_t_by_name[name].n_templates >= standard_t_by_name[name].n_templates for name in standard_t_by_name)


def test_checked_in_sweep_config_resolves_all_algorithms() -> None:
    config = json.loads((ROOT / "experiments" / "path_match_sweeps_large.json").read_text(encoding="utf-8"))
    names = [item["name"] for item in config["sweeps"]]
    assert len(names) == len(set(names))
    assert any(item["kind"] == "size" for item in config["sweeps"])
    assert any(item["kind"] == "algorithm" for item in config["sweeps"])
    implementation_sweep = next(
        item for item in config["sweeps"]
        if item["name"] == "partial_beam_implementation_rare_anchor"
    )
    assert implementation_sweep["algorithms"] == [
        "beam_partial_score",
        "fast_beam_partial",
    ]
    for sweep in config["sweeps"]:
        resolve_algorithm_specs(sweep["algorithms"])
        if sweep["kind"] == "size":
            assert len(sweep["points"]) >= 4
        else:
            aliases = [
                item.get("alias", item["name"]) if isinstance(item, dict) else item
                for item in sweep["algorithms"]
            ]
            assert set(sweep.get("parameter_values", {})).issubset(aliases)


def test_tiny_sweep_run_writes_combined_tables(tmp_path: Path) -> None:
    config = {
        "suite_name": "tiny_sweep_test",
        "base_config": str(ROOT / "experiments" / "path_match_benchmark_quick.json"),
        "execution_overrides": {
            "execution_mode": "in_process",
            "repeats": 1,
            "warmup": 0,
            "max_dense_cells": 100000,
            "max_sparse_posting_hits": 100000,
        },
        "sweeps": [
            {
                "name": "tiny_size",
                "kind": "size",
                "regime": "quick_generic_jaccard",
                "x_name": "nodes_per_tree",
                "algorithms": ["exact_dense", "sparse_chain"],
                "n_instances": 1,
                "points": [
                    {"label": "n20", "x_value": 20, "overrides": {"n_g": 20, "n_h": 20}}
                ],
            },
            {
                "name": "tiny_beam_width",
                "kind": "algorithm",
                "regime": "quick_tree_to_path",
                "n_instances": 1,
                "parameter_name": "beam_width",
                "algorithms": [
                    "exact_dense",
                    {"name": "beam_local", "alias": "local_B10", "kwargs": {"beam_width": 10}},
                    {"name": "beam_local", "alias": "local_B20", "kwargs": {"beam_width": 20}},
                ],
                "parameter_values": {"local_B10": 10, "local_B20": 20},
            },
        ],
    }
    rows, summary, metadata = run_sweep_config(config, output_dir=tmp_path)
    assert set(rows["sweep_name"]) == {"tiny_size", "tiny_beam_width"}
    assert set(summary["sweep_kind"]) == {"size", "algorithm"}
    assert (tmp_path / "sweep_rows.csv").exists()
    assert (tmp_path / "sweep_summary.csv").exists()
    assert (tmp_path / "sweep_metadata.json").exists()
    loaded = pd.read_csv(tmp_path / "sweep_summary.csv")
    assert set(loaded["sweep_name"]) == {"tiny_size", "tiny_beam_width"}
    assert metadata["suite_name"] == "tiny_sweep_test"


def test_benchmark_guide_pdf_is_present() -> None:
    path = ROOT / "benchmarking" / "BENCHMARK_RUN_AND_FIGURE_GUIDE.pdf"
    payload = path.read_bytes()
    assert payload.startswith(b"%PDF")
    assert len(payload) > 5000


def test_talk_figure_script_writes_png_and_pdf(tmp_path: Path) -> None:
    import subprocess
    import sys

    import pytest

    pytest.importorskip("matplotlib")

    pairwise = pd.DataFrame([
        {
            "regime": "dense_common_overlap",
            "regime_order": 0,
            "algorithm": "exact_dense",
            "successful_instances": 2,
            "median_accuracy_percent": 100.0,
            "median_predict_seconds": 0.2,
            "warm_speed_accuracy_frontier": False,
        },
        {
            "regime": "dense_common_overlap",
            "regime_order": 0,
            "algorithm": "fast_dense",
            "successful_instances": 2,
            "median_accuracy_percent": 100.0,
            "median_predict_seconds": 0.01,
            "warm_speed_accuracy_frontier": True,
        },
        {
            "regime": "dense_common_overlap",
            "regime_order": 0,
            "algorithm": "fast_beam_partial",
            "successful_instances": 2,
            "median_accuracy_percent": 80.0,
            "median_predict_seconds": 0.005,
            "warm_speed_accuracy_frontier": True,
        },
    ])
    throughput = pd.DataFrame([
        {
            "regime": "equality_tree_to_path_common_labels",
            "regime_order": 0,
            "algorithm": "fast_dense",
            "completed": True,
            "matrix_total_accuracy": 1.0,
            "score_matrix_seconds_median": 0.02,
            "throughput_frontier": True,
        },
        {
            "regime": "equality_tree_to_path_common_labels",
            "regime_order": 0,
            "algorithm": "fast_sparse",
            "completed": True,
            "matrix_total_accuracy": 1.0,
            "score_matrix_seconds_median": 0.10,
            "throughput_frontier": False,
        },
    ])
    sweeps = pd.DataFrame([
        {
            "sweep_name": "dense_overlap_size",
            "sweep_kind": "size",
            "base_regime": "dense_common_overlap",
            "sweep_point": "n100",
            "sweep_x_name": "nodes_per_tree",
            "sweep_x_value": 100,
            "algorithm": "exact_dense",
            "successful_instances": 2,
            "median_predict_seconds": 0.03,
            "median_accuracy_percent": 100.0,
        },
        {
            "sweep_name": "dense_overlap_size",
            "sweep_kind": "size",
            "base_regime": "dense_common_overlap",
            "sweep_point": "n200",
            "sweep_x_name": "nodes_per_tree",
            "sweep_x_value": 200,
            "algorithm": "exact_dense",
            "successful_instances": 2,
            "median_predict_seconds": 0.12,
            "median_accuracy_percent": 100.0,
        },
        {
            "sweep_name": "partial_beam_width_rare_anchor",
            "sweep_kind": "algorithm",
            "base_regime": "rare_anchor_oracle",
            "sweep_point": "algorithm_grid",
            "sweep_x_name": "beam_width",
            "sweep_x_value": 50,
            "algorithm": "partial_B50",
            "successful_instances": 2,
            "median_predict_seconds": 0.04,
            "median_accuracy_percent": 98.0,
        },
        {
            "sweep_name": "partial_beam_width_rare_anchor",
            "sweep_kind": "algorithm",
            "base_regime": "rare_anchor_oracle",
            "sweep_point": "algorithm_grid",
            "sweep_x_name": "beam_width",
            "sweep_x_value": 100,
            "algorithm": "partial_B100",
            "successful_instances": 2,
            "median_predict_seconds": 0.07,
            "median_accuracy_percent": 99.5,
        },
    ])
    pair_path = tmp_path / "benchmark_summary.csv"
    through_path = tmp_path / "throughput_summary.csv"
    sweep_path = tmp_path / "sweep_summary.csv"
    pairwise.to_csv(pair_path, index=False)
    throughput.to_csv(through_path, index=False)
    sweeps.to_csv(sweep_path, index=False)
    outdir = tmp_path / "figures"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "benchmarking" / "make_benchmark_figures.py"),
            "--pairwise", str(pair_path),
            "--throughput", str(through_path),
            "--sweeps", str(sweep_path),
            "--outdir", str(outdir),
        ],
        check=True,
    )
    assert (outdir / "01_pairwise_speed_accuracy.png").exists()
    assert (outdir / "01_pairwise_speed_accuracy.pdf").exists()
    assert (outdir / "04_size_dense_overlap_size.png").exists()
    assert (outdir / "05_beam_partial_beam_width_rare_anchor.pdf").exists()
    assert (outdir / "figure_manifest.csv").exists()
