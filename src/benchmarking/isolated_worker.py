from __future__ import annotations

"""Subprocess worker used by :mod:`benchmarking.timing`.

This module intentionally imports only the Python standard library before the
worker environment has been established by the parent process.
"""

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import os
import pickle
import sys
import time
import traceback


_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "PYTHONHASHSEED",
)


def _peak_rss_bytes() -> tuple[Optional[int], Optional[str]]:
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if sys.platform == "darwin":  # macOS reports bytes.
            return value, "resource.ru_maxrss_bytes"
        # Linux and the BSDs commonly report KiB.  The benchmark target is
        # Ubuntu; keep the platform qualification explicit in the semantics.
        return value * 1024, "resource.ru_maxrss_kib_converted_to_bytes"
    except Exception:
        return None, None


def _thread_environment() -> Dict[str, Optional[str]]:
    return {name: os.environ.get(name) for name in _THREAD_VARS}


def _dispatch(task: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    if task == "pair":
        from .path_match_benchmark import (
            _measure_pair_cold_start,
            _run_algorithm_on_pair_in_process,
        )

        args = dict(payload)
        cold_metrics = _measure_pair_cold_start(args["pair"], args["spec"])
        row = _run_algorithm_on_pair_in_process(**args)
        return {"row": row, "cold_metrics": cold_metrics}
    if task == "throughput":
        import numpy as np

        from .path_match_throughput import (
            _measure_throughput_cold_start,
            _run_algorithm_on_corpus_in_process,
        )

        args = dict(payload)
        cold_metrics, cold_matrix = _measure_throughput_cold_start(
            args["corpus"], args["spec"]
        )
        row, matrix = _run_algorithm_on_corpus_in_process(**args)
        max_difference = None
        if matrix is not None and np.asarray(matrix).shape == np.asarray(cold_matrix).shape:
            max_difference = float(
                np.max(
                    np.abs(np.asarray(matrix, dtype=float) - np.asarray(cold_matrix, dtype=float)),
                    initial=0.0,
                )
            )
        return {
            "row": row,
            "matrix": matrix,
            "cold_metrics": cold_metrics,
            "cold_matrix_max_abs_difference": max_difference,
        }
    if task == "sleep":  # Small deterministic hook used by timeout tests.
        time.sleep(float(payload.get("seconds", 0.0)))
        return {"slept": float(payload.get("seconds", 0.0))}
    raise ValueError(f"unknown isolated benchmark task {task!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    try:
        with input_path.open("rb") as handle:
            payload = pickle.load(handle)
        start = time.perf_counter()
        result = _dispatch(str(args.task), payload)
        task_seconds = float(time.perf_counter() - start)
        peak_rss, semantics = _peak_rss_bytes()
        result.update({
            "worker_task_seconds": task_seconds,
            "peak_rss_bytes": peak_rss,
            "peak_rss_semantics": semantics,
            "thread_environment": _thread_environment(),
        })
        with output_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return 0
    except BaseException as exc:
        peak_rss, semantics = _peak_rss_bytes()
        failure = {
            "worker_exception_type": type(exc).__name__,
            "worker_exception_message": str(exc),
            "worker_traceback": traceback.format_exc(),
            "peak_rss_bytes": peak_rss,
            "peak_rss_semantics": semantics,
            "thread_environment": _thread_environment(),
        }
        try:
            with output_path.open("wb") as handle:
                pickle.dump(failure, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
        print(failure["worker_traceback"], file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
