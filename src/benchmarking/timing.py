from __future__ import annotations

"""Timing helpers for reproducible path-matching benchmarks.

The main benchmark runners can execute an algorithm either in the current
Python process (convenient for smoke tests) or in a fresh subprocess.  The
subprocess route provides three protections that an in-process timer cannot:

* thread-count environment variables are set before NumPy/Numba are imported;
* a wall-clock timeout can terminate a stalled algorithm;
* peak resident memory is measured over the complete worker lifetime.

The warm algorithm timings are still measured inside the worker after any
configured warm-up.  Fresh-process wall time is reported separately and
includes Python startup, imports, payload loading, algorithm setup, and result
serialization.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import json
import math
import os
import pickle
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


@dataclass(frozen=True)
class IsolatedTaskResult:
    """Result returned by :func:`run_isolated_task`."""

    status: str
    payload: Optional[Mapping[str, Any]]
    wall_seconds: float
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timeout_seconds: Optional[float]
    thread_count: Optional[int]


def timing_sample_summary(samples: Sequence[float], *, prefix: str) -> Dict[str, Any]:
    """Return stable descriptive statistics for one collection of timings."""

    values = np.asarray([float(x) for x in samples], dtype=float)
    if values.size == 0:
        raise ValueError("timing_sample_summary requires at least one sample")
    if np.any(~np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("timing samples must be finite and nonnegative")

    median_value = float(np.median(values))
    deviations = np.abs(values - median_value)
    return {
        f"{prefix}_median": median_value,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_mean": float(values.mean()),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_std": float(values.std(ddof=1)) if values.size >= 2 else 0.0,
        f"{prefix}_mad": float(np.median(deviations)),
        f"{prefix}_sample_count": int(values.size),
        f"{prefix}_samples": json.dumps([float(x) for x in values.tolist()]),
    }


def thread_environment(thread_count: Optional[int]) -> Dict[str, str]:
    """Return environment overrides for a fixed worker thread count."""

    if thread_count is None:
        return {}
    count = int(thread_count)
    if count < 1:
        raise ValueError("thread_count must be positive or None")
    out = {name: str(count) for name in THREAD_ENVIRONMENT_VARIABLES}
    out.update({
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    })
    return out


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover - exercised on Windows only
            proc.kill()
    except ProcessLookupError:
        return
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def run_isolated_task(
    task: str,
    payload: Mapping[str, Any],
    *,
    timeout_seconds: Optional[float],
    thread_count: Optional[int],
) -> IsolatedTaskResult:
    """Execute one benchmark task in a fresh Python subprocess.

    Parameters
    ----------
    task:
        Worker dispatch key.  Production tasks are ``"pair"`` and
        ``"throughput"``.
    payload:
        Pickleable task arguments.
    timeout_seconds:
        Wall-clock limit for the complete subprocess.  ``None`` disables the
        timeout.
    thread_count:
        If provided, common BLAS/OpenMP/Numba thread variables are set before
        the worker imports numerical packages.
    """

    if timeout_seconds is not None:
        timeout_seconds = float(timeout_seconds)
        if (not math.isfinite(timeout_seconds)) or timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite or None")
    if thread_count is not None and int(thread_count) < 1:
        raise ValueError("thread_count must be positive or None")

    repo_root = Path(__file__).resolve().parents[2]
    src_root = repo_root / "src"

    with tempfile.TemporaryDirectory(prefix="tree_match_benchmark_") as tmp:
        tmp_dir = Path(tmp)
        input_path = tmp_dir / "input.pkl"
        output_path = tmp_dir / "output.pkl"
        with input_path.open("wb") as handle:
            pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)

        env = os.environ.copy()
        env.update(thread_environment(thread_count))
        env["PYTHONHASHSEED"] = "0"
        old_pythonpath = env.get("PYTHONPATH", "")
        entries = [str(src_root)]
        if old_pythonpath:
            entries.append(old_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(entries)

        command = [
            sys.executable,
            "-m",
            "benchmarking.isolated_worker",
            "--task",
            str(task),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ]
        start = time.perf_counter()
        proc = subprocess.Popen(
            command,
            cwd=str(repo_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name != "nt"),
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout_seconds)
            status = "completed"
        except subprocess.TimeoutExpired:
            _terminate_process(proc)
            stdout, stderr = proc.communicate()
            return IsolatedTaskResult(
                status="timeout",
                payload=None,
                wall_seconds=float(time.perf_counter() - start),
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=stderr,
                timeout_seconds=timeout_seconds,
                thread_count=(None if thread_count is None else int(thread_count)),
            )

        wall_seconds = float(time.perf_counter() - start)
        result_payload: Optional[Mapping[str, Any]] = None
        payload_error: Optional[str] = None
        if output_path.exists():
            try:
                with output_path.open("rb") as handle:
                    loaded = pickle.load(handle)
                if isinstance(loaded, Mapping):
                    result_payload = loaded
                else:
                    payload_error = "worker payload is not a mapping"
            except Exception as exc:
                payload_error = f"failed to read worker payload: {exc}"
        else:
            payload_error = "worker did not create an output payload"

        if proc.returncode != 0 or payload_error is not None:
            detail = stderr
            if payload_error is not None:
                detail = (detail + f"\n{payload_error}").strip()
            return IsolatedTaskResult(
                status="worker_error",
                payload=result_payload,
                wall_seconds=wall_seconds,
                exit_code=proc.returncode,
                stdout=stdout,
                stderr=detail,
                timeout_seconds=timeout_seconds,
                thread_count=(None if thread_count is None else int(thread_count)),
            )

        return IsolatedTaskResult(
            status=status,
            payload=result_payload,
            wall_seconds=wall_seconds,
            exit_code=proc.returncode,
            stdout=stdout,
            stderr=stderr,
            timeout_seconds=timeout_seconds,
            thread_count=(None if thread_count is None else int(thread_count)),
        )


def isolation_metadata(
    result: IsolatedTaskResult,
    worker_payload: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Flatten common subprocess measurements into benchmark row columns."""

    worker_payload = dict(worker_payload or {})
    task_seconds = worker_payload.get("worker_task_seconds")
    peak_rss_bytes = worker_payload.get("peak_rss_bytes")
    overhead = None
    if task_seconds is not None:
        overhead = max(0.0, float(result.wall_seconds) - float(task_seconds))
    return {
        "execution_mode": "isolated",
        "thread_count_requested": result.thread_count,
        "thread_settings_enforced": result.thread_count is not None,
        "timeout_seconds": result.timeout_seconds,
        "fresh_process_total_seconds": float(result.wall_seconds),
        "fresh_process_task_seconds": (
            None if task_seconds is None else float(task_seconds)
        ),
        "fresh_process_overhead_seconds": overhead,
        "peak_rss_bytes": None if peak_rss_bytes is None else int(peak_rss_bytes),
        "peak_rss_mib": (
            None if peak_rss_bytes is None else float(peak_rss_bytes) / (1024.0 ** 2)
        ),
        "peak_rss_semantics": worker_payload.get("peak_rss_semantics"),
        "isolated_process_exit_code": result.exit_code,
        "worker_thread_environment": json.dumps(
            worker_payload.get("thread_environment", {}), sort_keys=True
        ),
    }


__all__ = [
    "THREAD_ENVIRONMENT_VARIABLES",
    "IsolatedTaskResult",
    "isolation_metadata",
    "run_isolated_task",
    "thread_environment",
    "timing_sample_summary",
]
