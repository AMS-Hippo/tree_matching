#!/usr/bin/env python3
"""Run the reproducible path-matching benchmark matrix.

Examples
--------
Quick smoke test::

    python experiments/run_path_match_benchmarks.py \
        --config experiments/path_match_benchmark_quick.json

Larger starter matrix::

    python experiments/run_path_match_benchmarks.py \
        --config experiments/path_match_benchmark_full.json \
        --output-dir .cache/benchmarks/path_match/full_v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow execution from a source checkout without requiring ``pip install -e .``.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarking.path_match_benchmark import run_benchmark_config  # noqa: E402
from benchmarking.shared import config_hash, load_config  # noqa: E402


def _default_output_dir(config_path: Path, config: dict) -> Path:
    suite = str(config.get("suite_name", config_path.stem))
    digest = config_hash(config, prefix="")
    return ROOT / ".cache" / "benchmarks" / "path_match" / f"{suite}__{digest}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run exact, sparse, and beam path-matching algorithms on named "
            "synthetic tree-pair regimes."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "path_match_benchmark_quick.json",
        help="JSON or YAML benchmark configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a config-hashed directory under .cache/.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = load_config(config_path)
    output_dir = args.output_dir.resolve() if args.output_dir is not None else _default_output_dir(config_path, config)

    rows, summary, metadata = run_benchmark_config(config, output_dir=output_dir)

    print(f"Suite: {metadata['suite_name']}")
    print(f"Config hash: {metadata['config_hash']}")
    print(f"Rows: {len(rows)}")
    print(f"Output: {output_dir}")

    statuses = rows.groupby(["status"], dropna=False).size().to_dict()
    print("Statuses: " + json.dumps({str(k): int(v) for k, v in statuses.items()}, sort_keys=True))

    display_columns = [
        "regime",
        "expected_algorithm",
        "observed_fastest_warm_eligible",
        "observed_fastest_setup_plus_warm_eligible",
        "observed_best_warm_accuracy_harmonic",
        "algorithm",
        "complete_coverage",
        "successful_instances",
        "median_predict_seconds",
        "median_accuracy_percent",
        "median_warm_timing_score",
        "median_warm_accuracy_harmonic_score",
        "quality_basis",
        "min_quality",
        "quality_eligible",
    ]
    available = [name for name in display_columns if name in summary.columns]
    if available:
        print("\nSummary:")
        print(summary[available].to_string(index=False))

    errors = rows[rows["status"] == "error"]
    if not errors.empty:
        print("\nErrors:", file=sys.stderr)
        for item in errors[["regime", "instance", "algorithm", "error_type", "error_message"]].to_dict(orient="records"):
            print(json.dumps(item, sort_keys=True, default=str), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
