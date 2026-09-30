from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarking import run_sweep_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run path-matching size and beam sweeps")
    parser.add_argument(
        "--config",
        default=str(ROOT / "experiments" / "path_match_sweeps_large.json"),
        help="Sweep JSON/YAML configuration",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "benchmarking" / "results" / "path_match_sweeps_large_v1"),
        help="Directory for sweep artifacts",
    )
    args = parser.parse_args()

    rows, summary, _ = run_sweep_config(args.config, output_dir=args.output_dir)
    print(f"Completed {len(rows):,} algorithm-instance rows")
    print(f"Summary rows: {len(summary):,}")
    print(f"Rows:    {Path(args.output_dir) / 'sweep_rows.csv'}")
    print(f"Summary: {Path(args.output_dir) / 'sweep_summary.csv'}")


if __name__ == "__main__":
    main()
