from __future__ import annotations

"""Reproducible benchmark configurations and runner for tree-path matchers."""

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import math
import os
import platform
import sys
import traceback

import numpy as np
import pandas as pd

from path_matcher import (
    FastSparseTreePathMatcher,
    FastTreePathMatcher,
    SparseCandidateConfig,
    TreePathMatcher,
)

from .path_match_cases import PairRegime, ScoreModel, SyntheticPair, generate_synthetic_pair
from .shared import config_hash, ensure_dir, load_config
from .timing import isolation_metadata, run_isolated_task, timing_sample_summary


DEFAULT_EXACT_SCORE_ABS_TOL = 1e-5
DEFAULT_EXACT_SCORE_REL_TOL = 1e-6


@dataclass(frozen=True)
class AlgorithmSpec:
    """A named, fully specified matcher configuration.

    These are the "named reproducible algorithm configurations" used by the
    benchmark runner.  They are algorithm-side presets, separate from the
    synthetic data regimes.  Per-suite ``kwargs`` may override the defaults.
    """

    name: str
    family: str
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    exact: bool = False
    compatible_score_modes: Tuple[str, ...] = ("equality", "overlap", "jaccard")
    description: str = ""

    def with_overrides(self, overrides: Optional[Mapping[str, Any]]) -> "AlgorithmSpec":
        merged = dict(self.kwargs)
        merged.update(dict(overrides or {}))
        return replace(self, kwargs=merged)

    def as_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["kwargs"] = dict(self.kwargs)
        return out


def default_algorithm_specs() -> Dict[str, AlgorithmSpec]:
    """Return the standard benchmark presets.

    The score-only partial beam deliberately retains a finite candidate budget;
    it is a simple deterministic approximation, not the exhaustive limiting
    version of Algorithm 7.  Its exact parameters are recorded in every result.
    """

    return {
        "exact_dense": AlgorithmSpec(
            name="exact_dense",
            family="generic",
            kwargs={"method": "exact"},
            exact=True,
            description="Generic O(nm) exact dynamic program.",
        ),
        "fast_dense": AlgorithmSpec(
            name="fast_dense",
            family="fast_dense",
            exact=True,
            compatible_score_modes=("equality", "overlap"),
            description="Specialized integerized/Numba exact matcher.",
        ),
        "sparse_closure": AlgorithmSpec(
            name="sparse_closure",
            family="generic_sparse",
            kwargs={"method": "sparse"},
            exact=True,
            description="Existing selected-cell skip-closure dynamic program with exhaustive blocking candidates.",
        ),
        "sparse_chain": AlgorithmSpec(
            name="sparse_chain",
            family="generic_sparse",
            kwargs={"method": "sparse_chain"},
            exact=True,
            description="Exact product-poset maximum-chain solver over exhaustive positive blocking candidates.",
        ),
        "fast_sparse": AlgorithmSpec(
            name="fast_sparse",
            family="fast_sparse",
            exact=True,
            compatible_score_modes=("equality", "overlap"),
            description="Specialized equality/overlap sparse-chain matcher.",
        ),
        "beam_local": AlgorithmSpec(
            name="beam_local",
            family="generic",
            kwargs={
                "method": "beam_local",
                "beam_width": 200,
                "beam_local_child_cap": None,
            },
            exact=False,
            description="Algorithm 6: score-ranked local-transition beam with all children by default.",
        ),
        "beam_local_capped": AlgorithmSpec(
            name="beam_local_capped",
            family="generic",
            kwargs={
                "method": "beam_local",
                "beam_width": 200,
                "beam_local_child_cap": 16,
            },
            exact=False,
            description="Algorithm 6 with a deterministic per-node child cap.",
        ),
        "beam_partial_score": AlgorithmSpec(
            name="beam_partial_score",
            family="generic",
            kwargs={
                "method": "beam",
                "beam_width": 200,
                "beam_expansion_width": 64,
                "beam_random_fraction": 0.0,
                "beam_n_restarts": 1,
                "beam_symmetric": False,
                "beam_rarity_weight": 0.0,
                "beam_gap_penalty": 0.0,
                "beam_balance_penalty": 0.0,
                "beam_candidate_future_weight": 0.0,
                "beam_priority_future_weight": 0.0,
                "beam_priority_length_weight": 0.0,
                "beam_lookahead": False,
                "candidate_select_mode": "first",
                "seed": 0,
            },
            exact=False,
            description="Deterministic score-only Algorithm 7 state space with explicit B=200 and expansion budget 64.",
        ),
        "beam_partial_heuristic": AlgorithmSpec(
            name="beam_partial_heuristic",
            family="generic",
            kwargs={
                "method": "beam",
                "beam_width": 200,
                "beam_expansion_width": 64,
                "beam_random_fraction": 0.10,
                "beam_n_restarts": 1,
                "beam_symmetric": False,
                "beam_rarity_weight": 0.25,
                "beam_gap_penalty": 0.03,
                "beam_balance_penalty": 0.01,
                "beam_candidate_future_weight": 0.03,
                "beam_priority_future_weight": 0.20,
                "beam_priority_length_weight": 0.0,
                "beam_lookahead": False,
                "candidate_select_mode": "mixed",
                "seed": 0,
            },
            exact=False,
            description="Current production-style partial-matching beam, without lookahead.",
        ),
    }


def resolve_algorithm_specs(items: Sequence[Any]) -> List[AlgorithmSpec]:
    registry = default_algorithm_specs()
    out: List[AlgorithmSpec] = []
    for item in items:
        if isinstance(item, str):
            name = item
            overrides: Mapping[str, Any] = {}
        elif isinstance(item, Mapping):
            name = str(item.get("name", ""))
            overrides = dict(item.get("kwargs", {}))
        else:
            raise TypeError("algorithm entries must be names or mappings")
        if name not in registry:
            raise ValueError(f"Unknown algorithm preset {name!r}; choices are {sorted(registry)}")
        base_method = registry[name].kwargs.get("method")
        if "method" in overrides and base_method is not None and overrides["method"] != base_method:
            raise ValueError(
                f"Preset {name!r} fixes method={base_method!r}; choose another preset "
                "instead of changing its algorithm family through kwargs"
            )
        resolved = registry[name].with_overrides(overrides)
        if isinstance(item, Mapping):
            alias = str(item.get("alias", item.get("as", name))).strip()
            if not alias:
                raise ValueError("algorithm alias must be non-empty")
            if alias != name:
                resolved = replace(resolved, name=alias)
        out.append(resolved)
    if not out:
        raise ValueError("benchmark suite must contain at least one algorithm")
    names = [spec.name for spec in out]
    if len(set(names)) != len(names):
        raise ValueError(
            "resolved algorithm names must be unique; use an 'alias' when benchmarking "
            "multiple overrides of the same preset"
        )
    return out


def _matcher_for(
    spec: AlgorithmSpec,
    score_model: ScoreModel,
    *,
    collect_diagnostics: bool,
):
    kwargs = dict(spec.kwargs)
    if spec.family == "generic":
        return TreePathMatcher(
            w=score_model.score,
            collect_diagnostics=collect_diagnostics,
            **kwargs,
        )
    if spec.family == "generic_sparse":
        kwargs.setdefault("sparse_cfg", SparseCandidateConfig.exhaustive())
        return TreePathMatcher(
            w=score_model.bucketable_weight(),
            collect_diagnostics=collect_diagnostics,
            **kwargs,
        )
    if spec.family == "fast_dense":
        mode = score_model.fast_mode
        if mode is None:
            raise ValueError(f"{spec.name} is incompatible with score mode {score_model.mode!r}")
        return FastTreePathMatcher(
            mode=mode,
            token_weights=score_model.token_weights,
            default_weight=score_model.default_weight,
            collect_diagnostics=collect_diagnostics,
            **kwargs,
        )
    if spec.family == "fast_sparse":
        mode = score_model.fast_mode
        if mode is None:
            raise ValueError(f"{spec.name} is incompatible with score mode {score_model.mode!r}")
        return FastSparseTreePathMatcher(
            mode=mode,
            token_weights=score_model.token_weights,
            default_weight=score_model.default_weight,
            collect_diagnostics=collect_diagnostics,
            **kwargs,
        )
    raise ValueError(f"Unknown algorithm family {spec.family!r}")


def _environment_metadata() -> Dict[str, Any]:
    try:
        import numba  # type: ignore

        numba_version: Optional[str] = str(numba.__version__)
    except Exception:
        numba_version = None
    thread_vars = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMBA_NUM_THREADS",
        )
    }
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "numba_version": numba_version,
        "thread_environment": thread_vars,
    }


def _flatten_diagnostics(diag: Any) -> Dict[str, Any]:
    if diag is None:
        return {}
    raw = diag.as_dict()
    extra = dict(raw.pop("extra", {}))
    out = {f"diag_{k}": v for k, v in raw.items()}
    for key, value in sorted(extra.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[f"diag_extra_{key}"] = value
        else:
            out[f"diag_extra_{key}"] = json.dumps(value, sort_keys=True, default=str)
    return out


def _score_path(pair: SyntheticPair, path: Sequence[Tuple[int, int]]) -> float:
    return float(sum(pair.score_model.score(pair.G.label[int(u)], pair.H.label[int(v)]) for u, v in path))


def _quality_metrics(pair: SyntheticPair, path: Sequence[Tuple[int, int]]) -> Dict[str, Any]:
    found = {(int(u), int(v)) for u, v in path}
    truth = set(pair.truth_pairs)
    found_g = {u for u, _ in found}
    found_h = {v for _, v in found}
    truth_g = set(pair.planted_nodes_g)
    truth_h = set(pair.planted_nodes_h)

    return {
        "matched_length": len(path),
        "truth_pair_recall": float(len(found.intersection(truth)) / len(truth)) if truth else None,
        "truth_pair_precision": float(len(found.intersection(truth)) / len(found)) if found else None,
        "planted_g_node_recall": float(len(found_g.intersection(truth_g)) / len(truth_g)) if truth_g else None,
        "planted_h_node_recall": float(len(found_h.intersection(truth_h)) / len(truth_h)) if truth_h else None,
    }


def _skip_reason(
    pair: SyntheticPair,
    spec: AlgorithmSpec,
    *,
    max_dense_cells: Optional[int],
    max_sparse_posting_hits: Optional[int],
) -> Optional[str]:
    if pair.score_model.mode not in spec.compatible_score_modes:
        return f"incompatible_score_mode:{pair.score_model.mode}"
    dense_cells = int(pair.metadata.get("dense_cells", pair.G.n * pair.H.n))
    posting_hits = int(pair.metadata.get("posting_hits", dense_cells))
    is_dense = spec.family == "fast_dense" or (
        spec.family == "generic" and str(spec.kwargs.get("method", "")) == "exact"
    )
    is_sparse = spec.family in {"generic_sparse", "fast_sparse"}
    if is_dense and max_dense_cells is not None and dense_cells > max_dense_cells:
        return f"dense_cell_limit:{dense_cells}>{max_dense_cells}"
    if is_sparse and max_sparse_posting_hits is not None and posting_hits > max_sparse_posting_hits:
        return f"sparse_posting_limit:{posting_hits}>{max_sparse_posting_hits}"
    return None


def _run_algorithm_on_pair_in_process(
    pair: SyntheticPair,
    spec: AlgorithmSpec,
    *,
    repeats: int = 3,
    warmup: int = 1,
    max_dense_cells: Optional[int] = None,
    max_sparse_posting_hits: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one named algorithm configuration on one synthetic pair."""

    base: Dict[str, Any] = {
        "regime": pair.regime_name,
        "instance": pair.instance_index,
        "algorithm": spec.name,
        "algorithm_family": spec.family,
        "algorithm_exact": bool(spec.exact),
        "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
        "score_mode": pair.score_model.mode,
        "n_g": pair.G.n,
        "n_h": pair.H.n,
        "dense_cells": int(pair.metadata.get("dense_cells", pair.G.n * pair.H.n)),
        "posting_hits": int(pair.metadata.get("posting_hits", 0)),
        "truth_score": float(pair.truth_score),
        "score": None,
    }
    skip = _skip_reason(
        pair,
        spec,
        max_dense_cells=max_dense_cells,
        max_sparse_posting_hits=max_sparse_posting_hits,
    )
    if skip is not None:
        return {**base, "status": "skipped", "skip_reason": skip}

    repeats = max(1, int(repeats))
    warmup = max(0, int(warmup))
    try:
        # Diagnostic pass: this records algorithmic work.  It is intentionally
        # separate from the clean external timing pass.
        diag_matcher = _matcher_for(spec, pair.score_model, collect_diagnostics=True)
        t0 = perf_counter()
        diag_matcher.fit(pair.G, pair.H)
        diagnostic_fit_seconds = perf_counter() - t0
        path_diag, score_diag = diag_matcher.predict()
        diag = getattr(diag_matcher, "last_diagnostics_", None)
        fit_diag = getattr(diag_matcher, "last_fit_diagnostics_", None)

        timed_matcher = _matcher_for(spec, pair.score_model, collect_diagnostics=False)
        t0 = perf_counter()
        timed_matcher.fit(pair.G, pair.H)
        fit_seconds = perf_counter() - t0
        for _ in range(warmup):
            timed_matcher.predict()

        elapsed: List[float] = []
        timed_path: Sequence[Tuple[int, int]] = path_diag
        timed_score = float(score_diag)
        for _ in range(repeats):
            t0 = perf_counter()
            timed_path, timed_score = timed_matcher.predict()
            elapsed.append(perf_counter() - t0)

        if not math.isclose(float(timed_score), float(score_diag), rel_tol=1e-6, abs_tol=1e-6):
            raise RuntimeError(
                f"diagnostic and timed runs disagree: {score_diag} versus {timed_score}"
            )
        explicit_score = _score_path(pair, path_diag)
        if not math.isclose(explicit_score, float(score_diag), rel_tol=1e-5, abs_tol=1e-5):
            raise RuntimeError(
                f"reported score {score_diag} disagrees with explicit path score {explicit_score}"
            )

        result: Dict[str, Any] = {
            **base,
            "status": "ok",
            "skip_reason": None,
            "score": float(score_diag),
            "explicit_path_score": float(explicit_score),
            "fit_seconds": float(fit_seconds),
            "diagnostic_fit_seconds": float(diagnostic_fit_seconds),
            **timing_sample_summary(elapsed, prefix="predict_seconds"),
            "setup_plus_warm_search_seconds": float(fit_seconds + median(elapsed)),
            "timing_repeats": repeats,
            "warmup_runs": warmup,
            **_quality_metrics(pair, path_diag),
            **_flatten_diagnostics(diag),
        }
        if fit_diag is not None:
            fit_raw = fit_diag.as_dict()
            fit_extra = fit_raw.pop("extra", {})
            result.update({f"fit_diag_{k}": v for k, v in fit_raw.items()})
            for key, value in fit_extra.items():
                result[f"fit_diag_extra_{key}"] = value
        return result
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "skip_reason": None,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error_traceback": traceback.format_exc(limit=12),
        }




def _measure_pair_cold_start(
    pair: SyntheticPair,
    spec: AlgorithmSpec,
) -> Dict[str, Any]:
    """Measure the first fit and prediction in a fresh benchmark worker.

    This helper is called only by the isolated worker, before diagnostics or
    warm-up runs can populate Numba or allocator caches.  It deliberately uses
    a diagnostics-disabled matcher.
    """

    matcher = _matcher_for(spec, pair.score_model, collect_diagnostics=False)
    started = perf_counter()
    matcher.fit(pair.G, pair.H)
    fit_seconds = float(perf_counter() - started)
    started = perf_counter()
    path, score = matcher.predict()
    predict_seconds = float(perf_counter() - started)
    explicit_score = float(_score_path(pair, path))
    if not math.isclose(explicit_score, float(score), rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError(
            "cold prediction score disagrees with the explicit returned-path score: "
            f"{score} versus {explicit_score}"
        )
    return {
        "cold_fit_seconds": fit_seconds,
        "cold_predict_seconds": predict_seconds,
        "cold_setup_plus_predict_seconds": float(fit_seconds + predict_seconds),
        "cold_score": float(score),
        "cold_explicit_path_score": explicit_score,
    }

def run_algorithm_on_pair(
    pair: SyntheticPair,
    spec: AlgorithmSpec,
    *,
    repeats: int = 3,
    warmup: int = 1,
    max_dense_cells: Optional[int] = None,
    max_sparse_posting_hits: Optional[int] = None,
    execution_mode: str = "in_process",
    timeout_seconds: Optional[float] = None,
    thread_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Run one named algorithm configuration on one synthetic pair.

    ``execution_mode="isolated"`` starts a fresh Python subprocess, applies the
    requested thread-count environment before numerical libraries are imported,
    enforces a wall-clock timeout, and records lifetime peak resident memory.
    The primary warm-search timing is still measured inside that worker after
    the configured warm-up.
    """

    mode = str(execution_mode).strip().lower()
    if mode not in {"in_process", "isolated"}:
        raise ValueError("execution_mode must be 'in_process' or 'isolated'")
    if mode == "in_process":
        row = _run_algorithm_on_pair_in_process(
            pair,
            spec,
            repeats=repeats,
            warmup=warmup,
            max_dense_cells=max_dense_cells,
            max_sparse_posting_hits=max_sparse_posting_hits,
        )
        row.setdefault("execution_mode", "in_process")
        row.setdefault("thread_count_requested", thread_count)
        row.setdefault("thread_settings_enforced", False)
        row.setdefault("timeout_seconds", timeout_seconds)
        return row

    # Avoid paying subprocess startup for configurations that are known to be
    # incompatible or outside the declared work budget.
    skip = _skip_reason(
        pair,
        spec,
        max_dense_cells=max_dense_cells,
        max_sparse_posting_hits=max_sparse_posting_hits,
    )
    if skip is not None:
        row = _run_algorithm_on_pair_in_process(
            pair,
            spec,
            repeats=repeats,
            warmup=warmup,
            max_dense_cells=max_dense_cells,
            max_sparse_posting_hits=max_sparse_posting_hits,
        )
        row.update({
            "execution_mode": "isolated",
            "thread_count_requested": thread_count,
            "thread_settings_enforced": False,
            "execution_skipped_before_worker": True,
            "timeout_seconds": timeout_seconds,
        })
        return row

    isolated = run_isolated_task(
        "pair",
        {
            "pair": pair,
            "spec": spec,
            "repeats": repeats,
            "warmup": warmup,
            "max_dense_cells": max_dense_cells,
            "max_sparse_posting_hits": max_sparse_posting_hits,
        },
        timeout_seconds=timeout_seconds,
        thread_count=thread_count,
    )
    if isolated.status == "timeout":
        return {
            "regime": pair.regime_name,
            "instance": pair.instance_index,
            "algorithm": spec.name,
            "algorithm_family": spec.family,
            "algorithm_exact": bool(spec.exact),
            "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
            "score_mode": pair.score_model.mode,
            "n_g": pair.G.n,
            "n_h": pair.H.n,
            "dense_cells": int(pair.metadata.get("dense_cells", pair.G.n * pair.H.n)),
            "posting_hits": int(pair.metadata.get("posting_hits", 0)),
            "truth_score": float(pair.truth_score),
            "score": None,
            "status": "timeout",
            "skip_reason": None,
            "error_type": "TimeoutError",
            "error_message": f"isolated worker exceeded {timeout_seconds} seconds",
            **isolation_metadata(isolated, None),
        }
    if isolated.status != "completed" or isolated.payload is None:
        return {
            "regime": pair.regime_name,
            "instance": pair.instance_index,
            "algorithm": spec.name,
            "algorithm_family": spec.family,
            "algorithm_exact": bool(spec.exact),
            "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
            "score_mode": pair.score_model.mode,
            "n_g": pair.G.n,
            "n_h": pair.H.n,
            "dense_cells": int(pair.metadata.get("dense_cells", pair.G.n * pair.H.n)),
            "posting_hits": int(pair.metadata.get("posting_hits", 0)),
            "truth_score": float(pair.truth_score),
            "score": None,
            "status": "error",
            "skip_reason": None,
            "error_type": "IsolatedWorkerError",
            "error_message": isolated.stderr.strip() or "isolated worker failed",
            "worker_stdout": isolated.stdout[-20000:],
            "worker_stderr": isolated.stderr[-20000:],
            **isolation_metadata(isolated, isolated.payload),
        }

    payload = dict(isolated.payload)
    row = dict(payload.get("row", {}))
    if not row:
        return {
            "regime": pair.regime_name,
            "instance": pair.instance_index,
            "algorithm": spec.name,
            "algorithm_family": spec.family,
            "algorithm_exact": bool(spec.exact),
            "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
            "score_mode": pair.score_model.mode,
            "n_g": pair.G.n,
            "n_h": pair.H.n,
            "dense_cells": int(pair.metadata.get("dense_cells", pair.G.n * pair.H.n)),
            "posting_hits": int(pair.metadata.get("posting_hits", 0)),
            "truth_score": float(pair.truth_score),
            "score": None,
            "status": "error",
            "skip_reason": None,
            "error_type": "IsolatedWorkerProtocolError",
            "error_message": "isolated worker returned no benchmark row",
            **isolation_metadata(isolated, payload),
        }
    cold_metrics = payload.get("cold_metrics", {})
    if isinstance(cold_metrics, Mapping):
        row.update(dict(cold_metrics))
        if row.get("status") == "ok" and row.get("score") is not None:
            cold_score = row.get("cold_score")
            if cold_score is not None:
                row["cold_score_difference_from_warm"] = float(cold_score) - float(row["score"])
    row.update(isolation_metadata(isolated, payload))
    return row

def _harmonic_mean(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    a_f = float(a)
    b_f = float(b)
    if not math.isfinite(a_f) or not math.isfinite(b_f):
        return None
    if a_f < 0.0 or b_f < 0.0:
        return None
    if a_f + b_f <= 0.0:
        return 0.0
    return float(2.0 * a_f * b_f / (a_f + b_f))


def add_oracle_columns(
    rows: pd.DataFrame,
    *,
    exact_abs_tol: float = DEFAULT_EXACT_SCORE_ABS_TOL,
    exact_rel_tol: float = DEFAULT_EXACT_SCORE_REL_TOL,
) -> pd.DataFrame:
    """Attach exact-oracle, accuracy, timing, and secondary combined scores.

    Approximate methods are allowed to return suboptimal scores.  By contrast,
    disagreement among methods declared exact invalidates the oracle for that
    regime-instance: no accuracy ratio or quality ranking is then reported.

    The normalized timing score is the fastest successful time on the same
    regime-instance divided by the method's time.  It lies in [0,1], with one
    denoting the fastest successful method.  The harmonic timing/accuracy score
    is reported only as a secondary summary; it is not used as the headline
    ranking criterion.
    """

    if exact_abs_tol < 0.0 or exact_rel_tol < 0.0:
        raise ValueError("exact score tolerances must be nonnegative")

    out = rows.copy()
    instance_info: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for key, group in out.groupby(["regime", "instance"], dropna=False):
        exact = group[
            (group["status"] == "ok")
            & (group["algorithm_exact"] == True)  # noqa: E712
        ]
        vals = [float(x) for x in exact["score"].dropna().tolist()]
        instance_key = (str(key[0]), int(key[1]))
        if not vals:
            instance_info[instance_key] = {
                "oracle_score": None,
                "oracle_valid": False,
                "oracle_status": "no_exact_result",
                "exact_method_count": 0,
                "exact_score_spread": None,
                "exact_agreement_tolerance": None,
            }
            continue

        lo = min(vals)
        hi = max(vals)
        spread = hi - lo
        scale = max(abs(lo), abs(hi), 1.0)
        tolerance = max(float(exact_abs_tol), float(exact_rel_tol) * scale)
        agreed = spread <= tolerance
        if not agreed:
            status = "exact_disagreement"
            oracle_score: Optional[float] = None
        elif len(vals) == 1:
            status = "single_exact_result"
            oracle_score = hi
        else:
            status = "exact_agreement"
            oracle_score = hi
        instance_info[instance_key] = {
            "oracle_score": oracle_score,
            "oracle_valid": bool(agreed),
            "oracle_status": status,
            "exact_method_count": len(vals),
            "exact_score_spread": float(spread),
            "exact_agreement_tolerance": float(tolerance),
        }

    oracle_values: List[Optional[float]] = []
    oracle_valid: List[bool] = []
    oracle_status: List[str] = []
    exact_counts: List[int] = []
    exact_spreads: List[Optional[float]] = []
    exact_tolerances: List[Optional[float]] = []
    ratios: List[Optional[float]] = []
    gaps: List[Optional[float]] = []
    accuracy_scores: List[Optional[float]] = []
    accuracy_percents: List[Optional[float]] = []
    score_consistent: List[Optional[bool]] = []

    for row in out.to_dict(orient="records"):
        key = (str(row["regime"]), int(row["instance"]))
        info = instance_info[key]
        opt = info["oracle_score"]
        score = row.get("score")
        oracle_values.append(opt)
        oracle_valid.append(bool(info["oracle_valid"]))
        oracle_status.append(str(info["oracle_status"]))
        exact_counts.append(int(info["exact_method_count"]))
        exact_spreads.append(info["exact_score_spread"])
        exact_tolerances.append(info["exact_agreement_tolerance"])

        if (
            not bool(info["oracle_valid"])
            or opt is None
            or score is None
            or bool(pd.isna(score))
        ):
            ratios.append(None)
            gaps.append(None)
            accuracy_scores.append(None)
            accuracy_percents.append(None)
            score_consistent.append(None)
            continue

        score_f = float(score)
        opt_f = float(opt)
        tolerance = float(info["exact_agreement_tolerance"] or exact_abs_tol)
        consistent = score_f <= opt_f + tolerance
        score_consistent.append(bool(consistent))
        if not consistent:
            ratios.append(None)
            gaps.append(float(opt_f - score_f))
            accuracy_scores.append(None)
            accuracy_percents.append(None)
            continue

        if abs(score_f - opt_f) <= tolerance:
            ratio = 1.0
        elif opt_f <= 0.0:
            ratio = 1.0 if abs(score_f) <= tolerance else None
        else:
            ratio = score_f / opt_f
        ratios.append(ratio)
        gaps.append(float(opt_f - score_f))
        if ratio is None:
            accuracy_scores.append(None)
            accuracy_percents.append(None)
        else:
            normalized = min(1.0, max(0.0, float(ratio)))
            accuracy_scores.append(normalized)
            accuracy_percents.append(100.0 * normalized)

    out["oracle_score"] = oracle_values
    out["oracle_valid"] = oracle_valid
    out["oracle_status"] = oracle_status
    out["exact_method_count"] = exact_counts
    out["exact_score_spread"] = exact_spreads
    out["exact_agreement_tolerance"] = exact_tolerances
    out["score_ratio"] = ratios
    out["score_gap"] = gaps
    out["accuracy_score"] = accuracy_scores
    out["accuracy_percent"] = accuracy_percents
    out["score_consistent_with_oracle"] = score_consistent

    # Normalize timing within each regime-instance.  This preserves the raw
    # timing columns while giving a dimensionless score for secondary combined
    # summaries.
    out["warm_timing_score"] = np.nan
    out["setup_timing_score"] = np.nan
    out["warm_accuracy_harmonic_score"] = np.nan
    out["setup_accuracy_harmonic_score"] = np.nan

    for _, group in out.groupby(["regime", "instance"], dropna=False):
        ok = group[group["status"] == "ok"]
        warm = (
            pd.to_numeric(ok["predict_seconds_median"], errors="coerce")
            if "predict_seconds_median" in ok
            else pd.Series(dtype=float)
        )
        setup = (
            pd.to_numeric(ok["setup_plus_warm_search_seconds"], errors="coerce")
            if "setup_plus_warm_search_seconds" in ok
            else pd.Series(dtype=float)
        )
        warm_positive = warm[(warm >= 0.0) & np.isfinite(warm)]
        setup_positive = setup[(setup >= 0.0) & np.isfinite(setup)]
        best_warm = float(warm_positive.min()) if not warm_positive.empty else None
        best_setup = float(setup_positive.min()) if not setup_positive.empty else None

        for idx in group.index:
            if out.at[idx, "status"] != "ok":
                continue
            warm_value = out.at[idx, "predict_seconds_median"]
            setup_value = out.at[idx, "setup_plus_warm_search_seconds"]
            accuracy = out.at[idx, "accuracy_score"]

            warm_score: Optional[float] = None
            if best_warm is not None and warm_value is not None and not pd.isna(warm_value):
                warm_f = float(warm_value)
                if best_warm == 0.0:
                    warm_score = 1.0 if warm_f == 0.0 else 0.0
                elif warm_f > 0.0:
                    warm_score = min(1.0, best_warm / warm_f)
            setup_score: Optional[float] = None
            if best_setup is not None and setup_value is not None and not pd.isna(setup_value):
                setup_f = float(setup_value)
                if best_setup == 0.0:
                    setup_score = 1.0 if setup_f == 0.0 else 0.0
                elif setup_f > 0.0:
                    setup_score = min(1.0, best_setup / setup_f)

            if warm_score is not None:
                out.at[idx, "warm_timing_score"] = warm_score
            if setup_score is not None:
                out.at[idx, "setup_timing_score"] = setup_score
            if accuracy is not None and not pd.isna(accuracy):
                warm_h = _harmonic_mean(float(accuracy), warm_score)
                setup_h = _harmonic_mean(float(accuracy), setup_score)
                if warm_h is not None:
                    out.at[idx, "warm_accuracy_harmonic_score"] = warm_h
                if setup_h is not None:
                    out.at[idx, "setup_accuracy_harmonic_score"] = setup_h

    return out


def summarize_benchmark_rows(
    rows: pd.DataFrame,
    regimes: Sequence[PairRegime],
    *,
    quality_floor: float = 0.99,
) -> pd.DataFrame:
    """Aggregate by regime and identify complete, quality-eligible methods.

    A method is eligible for the headline timing comparison only when it:

    1. finishes every intended instance in the regime;
    2. has a valid exact-score oracle for every instance; and
    3. meets the requested minimum accuracy on every instance.

    Timing and accuracy remain separate primary outputs.  Their harmonic mean is
    included only as a clearly labelled secondary score.
    """

    expected = {r.name: r.expected_algorithm for r in regimes}
    regime_order = {r.name: i for i, r in enumerate(regimes)}
    summaries: List[Dict[str, Any]] = []

    for regime_name, regime_rows in rows.groupby("regime", sort=False):
        total_instances = int(regime_rows["instance"].nunique())
        instance_oracles = regime_rows.drop_duplicates(subset=["instance"])
        oracle_valid_instances = int(instance_oracles["oracle_valid"].fillna(False).sum())
        exact_disagreement_instances = int(
            (instance_oracles["oracle_status"] == "exact_disagreement").sum()
        )
        alg_rows: List[Dict[str, Any]] = []

        for algorithm, group in regime_rows.groupby("algorithm", sort=False):
            ok = group[group["status"] == "ok"]
            successful_instances = int(len(ok))
            skipped_instances = int((group["status"] == "skipped").sum())
            timeout_instances = int((group["status"] == "timeout").sum())
            error_instances = int((group["status"] == "error").sum())
            complete_coverage = successful_instances == total_instances

            accuracy_values = [
                float(x)
                for x in ok.get("accuracy_score", pd.Series(dtype=float)).dropna().tolist()
            ]
            complete_accuracy_coverage = (
                complete_coverage and len(accuracy_values) == total_instances
            )
            accuracy_median = float(median(accuracy_values)) if accuracy_values else None
            accuracy_min = float(min(accuracy_values)) if accuracy_values else None

            runtimes = [
                float(x)
                for x in ok.get("predict_seconds_median", pd.Series(dtype=float)).dropna().tolist()
            ]
            setup_times = [
                float(x)
                for x in ok.get(
                    "setup_plus_warm_search_seconds", pd.Series(dtype=float)
                ).dropna().tolist()
            ]
            runtime_q25 = float(np.quantile(runtimes, 0.25)) if runtimes else None
            runtime_q75 = float(np.quantile(runtimes, 0.75)) if runtimes else None
            setup_q25 = float(np.quantile(setup_times, 0.25)) if setup_times else None
            setup_q75 = float(np.quantile(setup_times, 0.75)) if setup_times else None
            cold_predict_times = [
                float(x)
                for x in ok.get("cold_predict_seconds", pd.Series(dtype=float)).dropna().tolist()
            ]
            cold_setup_times = [
                float(x)
                for x in ok.get(
                    "cold_setup_plus_predict_seconds", pd.Series(dtype=float)
                ).dropna().tolist()
            ]
            fresh_process_times = [
                float(x)
                for x in ok.get("fresh_process_total_seconds", pd.Series(dtype=float)).dropna().tolist()
            ]
            peak_memory = [
                float(x)
                for x in ok.get("peak_rss_mib", pd.Series(dtype=float)).dropna().tolist()
            ]
            warm_timing_scores = [
                float(x)
                for x in ok.get("warm_timing_score", pd.Series(dtype=float)).dropna().tolist()
            ]
            setup_timing_scores = [
                float(x)
                for x in ok.get("setup_timing_score", pd.Series(dtype=float)).dropna().tolist()
            ]
            warm_harmonic = [
                float(x)
                for x in ok.get(
                    "warm_accuracy_harmonic_score", pd.Series(dtype=float)
                ).dropna().tolist()
            ]
            setup_harmonic = [
                float(x)
                for x in ok.get(
                    "setup_accuracy_harmonic_score", pd.Series(dtype=float)
                ).dropna().tolist()
            ]

            quality_basis = "exact_optimal_score" if complete_accuracy_coverage else None
            quality_eligible = bool(
                complete_accuracy_coverage
                and accuracy_min is not None
                and accuracy_min >= float(quality_floor)
            )
            algorithm_config_index = int(
                group.get("algorithm_config_index", pd.Series([0])).iloc[0]
            )

            alg_rows.append({
                "algorithm": algorithm,
                "algorithm_config_index": algorithm_config_index,
                "total_instances": total_instances,
                "successful_instances": successful_instances,
                "skipped_instances": skipped_instances,
                "timeout_instances": timeout_instances,
                "error_instances": error_instances,
                "complete_coverage": bool(complete_coverage),
                "oracle_scored_instances": len(accuracy_values),
                "complete_accuracy_coverage": bool(complete_accuracy_coverage),
                "median_predict_seconds": float(median(runtimes)) if runtimes else None,
                "q25_predict_seconds": runtime_q25,
                "q75_predict_seconds": runtime_q75,
                "median_setup_plus_warm_search_seconds": (
                    float(median(setup_times)) if setup_times else None
                ),
                "q25_setup_plus_warm_search_seconds": setup_q25,
                "q75_setup_plus_warm_search_seconds": setup_q75,
                "median_cold_predict_seconds": (
                    float(median(cold_predict_times)) if cold_predict_times else None
                ),
                "median_cold_setup_plus_predict_seconds": (
                    float(median(cold_setup_times)) if cold_setup_times else None
                ),
                "median_fresh_process_total_seconds": (
                    float(median(fresh_process_times)) if fresh_process_times else None
                ),
                "median_peak_rss_mib": float(median(peak_memory)) if peak_memory else None,
                "median_accuracy_score": accuracy_median,
                "min_accuracy_score": accuracy_min,
                "median_accuracy_percent": (
                    100.0 * accuracy_median if accuracy_median is not None else None
                ),
                # Backwards-compatible score-ratio names.  Accuracy is the same
                # optimal-score ratio, clipped to [0,1] only for normalization.
                "median_score_ratio": accuracy_median,
                "min_score_ratio": accuracy_min,
                "median_warm_timing_score": (
                    float(median(warm_timing_scores)) if warm_timing_scores else None
                ),
                "median_setup_timing_score": (
                    float(median(setup_timing_scores)) if setup_timing_scores else None
                ),
                "median_warm_accuracy_harmonic_score": (
                    float(median(warm_harmonic)) if warm_harmonic else None
                ),
                "median_setup_accuracy_harmonic_score": (
                    float(median(setup_harmonic)) if setup_harmonic else None
                ),
                "quality_basis": quality_basis,
                "median_quality": accuracy_median,
                "min_quality": accuracy_min,
                "quality_eligible": quality_eligible,
            })

        warm_eligible = [
            x for x in alg_rows
            if x["quality_eligible"] and x["median_predict_seconds"] is not None
        ]
        setup_eligible = [
            x for x in alg_rows
            if x["quality_eligible"]
            and x["median_setup_plus_warm_search_seconds"] is not None
        ]
        harmonic_candidates = [
            x for x in alg_rows
            if x["complete_accuracy_coverage"]
            and x["median_warm_accuracy_harmonic_score"] is not None
        ]
        observed_warm = (
            min(warm_eligible, key=lambda x: x["median_predict_seconds"])["algorithm"]
            if warm_eligible else None
        )
        observed_setup = (
            min(
                setup_eligible,
                key=lambda x: x["median_setup_plus_warm_search_seconds"],
            )["algorithm"]
            if setup_eligible else None
        )
        observed_harmonic = (
            max(
                harmonic_candidates,
                key=lambda x: x["median_warm_accuracy_harmonic_score"],
            )["algorithm"]
            if harmonic_candidates else None
        )

        for entry in alg_rows:
            summaries.append({
                "regime": regime_name,
                "regime_order": regime_order.get(str(regime_name), 0),
                "expected_algorithm": expected.get(str(regime_name)),
                "oracle_valid_instances": oracle_valid_instances,
                "exact_disagreement_instances": exact_disagreement_instances,
                "observed_fastest_eligible": observed_warm,
                "observed_fastest_warm_eligible": observed_warm,
                "observed_fastest_setup_plus_warm_eligible": observed_setup,
                "observed_best_warm_accuracy_harmonic": observed_harmonic,
                "quality_floor": float(quality_floor),
                **entry,
            })

    result = pd.DataFrame(summaries)
    if not result.empty:
        result = result.sort_values(
            ["regime_order", "algorithm_config_index"],
            kind="stable",
        ).reset_index(drop=True)
    return result


def _balanced_algorithm_order(
    algorithms: Sequence[AlgorithmSpec],
    *,
    regime_index: int,
    instance_index: int,
    seed: int,
) -> List[Tuple[int, AlgorithmSpec]]:
    """Return a deterministic, cyclically balanced execution order.

    Each regime receives one seeded base permutation.  Successive instances
    rotate that permutation, so algorithms move through timing positions rather
    than always running first or last.
    """

    n = len(algorithms)
    if n == 0:
        return []
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(regime_index)])
    )
    base = [int(x) for x in rng.permutation(n)]
    shift = int(instance_index) % n
    order = base[shift:] + base[:shift]
    return [(idx, algorithms[idx]) for idx in order]


def run_benchmark_config(
    config_or_path: Any,
    *,
    output_dir: Optional[Path | str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run a complete benchmark config and optionally write CSV/JSON outputs."""

    cfg = load_config(config_or_path)
    suite_name = str(cfg.get("suite_name", "path_match_benchmark"))
    algorithms = resolve_algorithm_specs(cfg.get("algorithms", list(default_algorithm_specs())))
    regimes = [PairRegime.from_mapping(x) for x in cfg.get("regimes", [])]
    if not regimes:
        raise ValueError("benchmark config must contain at least one regime")

    repeats = int(cfg.get("repeats", 3))
    warmup = int(cfg.get("warmup", 1))
    quality_floor = float(cfg.get("quality_floor", 0.99))
    exact_score_abs_tol = float(
        cfg.get("exact_score_abs_tol", DEFAULT_EXACT_SCORE_ABS_TOL)
    )
    exact_score_rel_tol = float(
        cfg.get("exact_score_rel_tol", DEFAULT_EXACT_SCORE_REL_TOL)
    )
    algorithm_order_seed = int(cfg.get("algorithm_order_seed", 0))
    execution_mode = str(cfg.get("execution_mode", "in_process")).strip().lower()
    timeout_seconds_raw = cfg.get("timeout_seconds")
    timeout_seconds = None if timeout_seconds_raw is None else float(timeout_seconds_raw)
    thread_count_raw = cfg.get("thread_count")
    thread_count = None if thread_count_raw is None else int(thread_count_raw)
    max_dense_cells = cfg.get("max_dense_cells")
    max_dense_cells = None if max_dense_cells is None else int(max_dense_cells)
    max_sparse_posting_hits = cfg.get("max_sparse_posting_hits")
    max_sparse_posting_hits = None if max_sparse_posting_hits is None else int(max_sparse_posting_hits)
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if not (0.0 <= quality_floor <= 1.0):
        raise ValueError("quality_floor must lie in [0,1]")
    if exact_score_abs_tol < 0.0 or exact_score_rel_tol < 0.0:
        raise ValueError("exact score tolerances must be nonnegative")
    if execution_mode not in {"in_process", "isolated"}:
        raise ValueError("execution_mode must be 'in_process' or 'isolated'")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be positive and finite or null")
    if thread_count is not None and thread_count < 1:
        raise ValueError("thread_count must be positive or null")
    if max_dense_cells is not None and max_dense_cells < 0:
        raise ValueError("max_dense_cells must be nonnegative or null")
    if max_sparse_posting_hits is not None and max_sparse_posting_hits < 0:
        raise ValueError("max_sparse_posting_hits must be nonnegative or null")

    rows: List[Dict[str, Any]] = []
    generation_seconds = 0.0
    for regime_index, regime in enumerate(regimes):
        for instance in range(int(regime.n_instances)):
            t0 = perf_counter()
            pair = generate_synthetic_pair(regime, instance)
            pair_generation_seconds = perf_counter() - t0
            generation_seconds += pair_generation_seconds
            execution_order = _balanced_algorithm_order(
                algorithms,
                regime_index=regime_index,
                instance_index=instance,
                seed=algorithm_order_seed,
            )
            for execution_position, (algorithm_config_index, spec) in enumerate(
                execution_order
            ):
                row = run_algorithm_on_pair(
                    pair,
                    spec,
                    repeats=repeats,
                    warmup=warmup,
                    max_dense_cells=max_dense_cells,
                    max_sparse_posting_hits=max_sparse_posting_hits,
                    execution_mode=execution_mode,
                    timeout_seconds=timeout_seconds,
                    thread_count=thread_count,
                )
                row.update({
                    "comparison": regime.comparison,
                    "algorithm_config_index": int(algorithm_config_index),
                    "execution_order_position": int(execution_position),
                    "execution_order_cycle": int(instance // max(1, len(algorithms))),
                    "pair_generation_seconds": pair_generation_seconds,
                    "shared_keys": int(pair.metadata.get("shared_keys", 0)),
                    "truth_pair_count": len(pair.truth_pairs),
                    "observed_planted_nodes_g": len(pair.planted_nodes_g),
                    "observed_planted_nodes_h": len(pair.planted_nodes_h),
                    "candidate_path_length_g": len(pair.metadata.get("candidate_planted_path_g", ())),
                    "candidate_path_length_h": len(pair.metadata.get("candidate_planted_path_h", ())),
                    "shape_g": regime.shape_g,
                    "shape_h": regime.shape_h,
                    "alphabet_size": regime.alphabet_size,
                    "symbol_distribution": regime.symbol_distribution,
                    "symbols_per_node": json.dumps(regime.symbols_per_node, sort_keys=True),
                    "weight_mode": regime.weight_mode,
                    "planted_length": regime.planted_length,
                    "planted_token_policy": regime.planted_token_policy,
                    "expected_algorithm": regime.expected_algorithm,
                    "regime_notes": regime.notes,
                    "tree_g_depth": pair.metadata["tree_g"]["depth"],
                    "tree_h_depth": pair.metadata["tree_h"]["depth"],
                    "tree_g_max_width": pair.metadata["tree_g"]["max_width"],
                    "tree_h_max_width": pair.metadata["tree_h"]["max_width"],
                    "tree_g_max_out_degree": pair.metadata["tree_g"]["max_out_degree"],
                    "tree_h_max_out_degree": pair.metadata["tree_h"]["max_out_degree"],
                })
                rows.append(row)

    row_frame = add_oracle_columns(
        pd.DataFrame(rows),
        exact_abs_tol=exact_score_abs_tol,
        exact_rel_tol=exact_score_rel_tol,
    )
    summary_frame = summarize_benchmark_rows(row_frame, regimes, quality_floor=quality_floor)
    metadata = {
        "suite_name": suite_name,
        "config_hash": config_hash(cfg, prefix="pathmatch_"),
        "config": cfg,
        "repeats": repeats,
        "warmup": warmup,
        "quality_floor": quality_floor,
        "exact_score_abs_tol": exact_score_abs_tol,
        "exact_score_rel_tol": exact_score_rel_tol,
        "algorithm_order_seed": algorithm_order_seed,
        "algorithm_order_strategy": (
            "isolated_subprocesses_with_seeded_scheduling_order"
            if execution_mode == "isolated"
            else "seeded_base_permutation_with_cyclic_rotation"
        ),
        "execution_mode": execution_mode,
        "timeout_seconds": timeout_seconds,
        "thread_count": thread_count,
        "thread_settings_enforced": execution_mode == "isolated" and thread_count is not None,
        "max_dense_cells": max_dense_cells,
        "max_sparse_posting_hits": max_sparse_posting_hits,
        "generation_seconds": generation_seconds,
        "timing_semantics": {
            "cold_predict_seconds": "first diagnostics-disabled prediction in a fresh worker before diagnostic or warm-up calls",
            "predict_seconds_median": "median diagnostics-disabled prediction after configured warm-up",
            "setup_plus_warm_search_seconds": "pair fit/setup plus median warm prediction",
            "fresh_process_total_seconds": "complete isolated worker wall time including startup, imports, cold pass, diagnostics, setup, warm-up, repeats, and serialization",
            "peak_rss_mib": "lifetime worker peak RSS, including runtime and imported libraries",
        },
        "environment": _environment_metadata(),
        "algorithms": [x.as_dict() for x in algorithms],
        "regimes": [x.as_dict() for x in regimes],
    }

    if output_dir is not None:
        out = ensure_dir(output_dir)
        row_frame.to_csv(out / "benchmark_rows.csv", index=False)
        summary_frame.to_csv(out / "benchmark_summary.csv", index=False)
        (out / "benchmark_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
    return row_frame, summary_frame, metadata


__all__ = [
    "AlgorithmSpec",
    "add_oracle_columns",
    "default_algorithm_specs",
    "resolve_algorithm_specs",
    "run_algorithm_on_pair",
    "run_benchmark_config",
    "summarize_benchmark_rows",
]
