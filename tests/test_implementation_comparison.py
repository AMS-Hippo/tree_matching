from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarking.implementation_comparison import compare_implementations, summarize_implementation_comparisons

ROOT = Path(__file__).resolve().parents[1]


def pair_rows(mode="equality"):
    return pd.DataFrame([
        {"regime": "r", "instance": i, "algorithm": algo, "status": "ok", "score_mode": mode,
         "n_g": 20, "n_h": 20, "score": 10.0, "predict_seconds_median": t,
         "setup_plus_warm_search_seconds": t+1, "cold_setup_plus_predict_seconds": t+2}
        for i, (g, e) in enumerate([(1.0, 0.01), (100.0, 10.0)])
        for algo, t in [("beam_partial_score", g), ("fast_beam_partial", e)]
    ])


def test_paired_speedups_and_score_checks_without_oracle():
    pairs = compare_implementations(pair_rows())
    assert pairs["score_agreement"].all()
    assert pairs["warm_speedup"].tolist() == [100.0, 10.0]
    summary = summarize_implementation_comparisons(pairs).iloc[0]
    assert summary["median_warm_speedup"] == 55.0
    assert summary["scores_checked"] == 2
    assert bool(summary["all_observed_scores_agree"])


def test_incomplete_runs_and_overlap_differences_are_not_hidden():
    rows = pair_rows("overlap")
    rows.loc[1, "score"] = 8
    rows.loc[3, "status"] = "timeout"
    pairs = compare_implementations(rows)
    assert not bool(pairs.iloc[0]["score_agreement"])
    assert pairs.iloc[0]["max_abs_score_difference"] == 2
    assert "different finite-budget" in pairs.iloc[0]["comparison_note"]
    assert np.isnan(pairs.iloc[1]["warm_speedup"])
    summary = summarize_implementation_comparisons(pairs).iloc[0]
    assert not bool(summary["complete_coverage"])
    assert summary["paired_complete_count"] == 1
    assert not bool(summary["all_observed_scores_agree"])


def test_throughput_compares_entries_not_only_sums():
    rows = pair_rows().query("instance == 0").drop(columns="instance").rename(columns={
        "score": "matrix_score_sum", "predict_seconds_median": "score_matrix_seconds_median",
        "setup_plus_warm_search_seconds": "setup_plus_one_matrix_seconds",
    })
    matrices = {("r", "beam_partial_score"): np.array([[1., 2.]]),
                ("r", "fast_beam_partial"): np.array([[2., 1.]])}
    without = compare_implementations(rows, kind="throughput")
    assert without.iloc[0]["score_agreement"] is None
    check = compare_implementations(rows, kind="throughput", matrices=matrices).iloc[0]
    assert not bool(check["score_agreement"])
    assert check["score_mismatch_count"] == 2
    assert check["max_abs_score_difference"] == 1


def test_invalid_or_unselected_inputs():
    rows = pair_rows()
    with pytest.raises(ValueError, match="Duplicate"):
        compare_implementations(pd.concat([rows, rows]))
    rows.loc[1, "score"] = np.inf
    assert not bool(compare_implementations(rows).iloc[0]["score_agreement"])
    local = pair_rows().query("instance == 0").iloc[:1].copy()
    local["algorithm"] = "beam_local"
    assert compare_implementations(local).empty
    assert "family" in compare_implementations(local).columns


def test_presets_keep_both_beams_and_four_way_recipe():
    for prefix in ["path_match_benchmark", "path_match_throughput"]:
        for size in ["quick", "full", "large"]:
            cfg = json.loads((ROOT / "experiments" / f"{prefix}_{size}.json").read_text())
            assert {"beam_partial_score", "fast_beam_partial"}.issubset(cfg["algorithms"])
    cfg = json.loads((ROOT / "experiments/path_match_implementation_comparison.json").read_text())
    comparison = next(s for s in cfg["sweeps"] if s["name"] == "partial_beam_implementation_rare_anchor")
    assert comparison["algorithms"] == ["exact_dense", "fast_dense", "beam_partial_score", "fast_beam_partial"]
    assert all(point["overrides"]["n_g"] * point["overrides"]["n_h"] <= cfg["execution_overrides"]["max_dense_cells"] for point in comparison["points"])
    for name in ["01_algorithm_ranking", "02_throughput"]:
        nb = json.loads((ROOT / f"benchmarking/{name}.ipynb").read_text())
        text = "\n".join("".join(c["source"]) for c in nb["cells"])
        assert "beam_partial_score" in text and "fast_beam_partial" in text
        assert "implementation_comparison_summary.csv" in text
        assert "existing results will not be overwritten" in text


def test_partial_figure_is_available_without_pairwise_input(tmp_path):
    pytest.importorskip("matplotlib")
    frame = pd.DataFrame([
        {"regime": "r", "regime_order": 0, "algorithm": a, "status": "ok", "completed": True,
         "matrix_total_accuracy": .99, "score_matrix_seconds_median": t}
        for a, t in [("beam_partial_score", 1.0), ("fast_beam_partial", .02)]
    ])
    path = tmp_path / "throughput_summary.csv"
    frame.to_csv(path, index=False)
    out = tmp_path / "figures"
    subprocess.run([sys.executable, str(ROOT / "benchmarking/make_benchmark_figures.py"),
                    "--throughput", str(path), "--outdir", str(out)], check=True, capture_output=True)
    assert (out / "06_partial_generic_vs_encoded_throughput.png").is_file()
    assert (out / "06_partial_generic_vs_encoded_throughput.pdf").is_file()
    data = pd.read_csv(out / "partial_implementation_throughput_plot_data.csv")
    assert data.iloc[0]["speedup_ratio_of_medians"] == 50


def load_archiver():
    spec = importlib.util.spec_from_file_location("archive_benchmark_data", ROOT / "benchmarking/archive_benchmark_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_run(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    run = tmp_path / "benchmarking/results/run1"
    run.mkdir(parents=True)
    (run / "throughput_rows.csv").write_text("algorithm,score\nx,10\n")
    (run / "throughput_summary.csv").write_text("summary\n1\n")
    (run / "throughput_metadata.json").write_text('{"fixture":true}')
    np.savez(run / "throughput_score_matrices.npz", scores=np.array([[10.]]))
    (run / "figure.png").write_bytes(b"not a real PNG")
    (run / "nested").mkdir()
    (run / "nested/raw.csv").write_text("x\n2\n")
    return run


def test_archive_preserves_numerical_files_and_never_overwrites(tmp_path):
    module = load_archiver()
    run = fake_run(tmp_path)
    snapshot = module.archive_runs(tmp_path, [run], "kept_v1")
    assert not (snapshot / "run1/figure.png").exists()
    assert (snapshot / "run1/nested/raw.csv").read_text() == "x\n2\n"
    manifest = json.loads((snapshot / "archive_manifest.json").read_text())
    for entry in manifest["files"]:
        assert module.sha256(snapshot / entry["path"]) == entry["sha256"]
    assert (snapshot / "run1/throughput_score_matrices.npz").read_bytes() == (run / "throughput_score_matrices.npz").read_bytes()
    with pytest.raises(FileExistsError):
        module.archive_runs(tmp_path, [run], "kept_v1")


def test_archive_rejects_incomplete_and_symlink_inputs(tmp_path):
    module = load_archiver()
    run = fake_run(tmp_path)
    (run / "throughput_summary.csv").unlink()
    with pytest.raises(FileNotFoundError):
        module.archive_runs(tmp_path, [run], "incomplete")
    assert not (tmp_path / "benchmarking/recorded_runs/incomplete").exists()
    (run / "throughput_summary.csv").write_text("summary\n1\n")
    (run / "other.json").symlink_to(run / "throughput_metadata.json")
    with pytest.raises(ValueError, match="symlink"):
        module.archive_runs(tmp_path, [run], "symlink")


def test_sweep_backend_saves_paired_comparison(tmp_path):
    from benchmarking import run_sweep_config
    cfg = {
        "base_config": str(ROOT / "experiments/path_match_benchmark_quick.json"),
        "execution_overrides": {"execution_mode": "in_process", "warmup": 0, "repeats": 1},
        "sweeps": [{"name": "tiny_impl", "kind": "size", "regime": "quick_tree_to_path", "n_instances": 1,
                    "algorithms": ["exact_dense", "fast_dense", "beam_partial_score", "fast_beam_partial"],
                    "points": [{"label": "n20", "x_value": 20, "overrides": {"n_g": 20, "n_h": 10, "depth_g": 10, "planted_length": 8, "score_mode": "equality", "symbols_per_node": 1}}]}],
    }
    rows, _, _ = run_sweep_config(cfg, output_dir=tmp_path)
    assert set(rows["status"]) == {"ok"}
    pairs = pd.read_csv(tmp_path / "implementation_comparison_rows.csv")
    assert set(pairs["family"]) == {"exact_dp", "partial_beam"}
    assert pairs["score_agreement"].all()
