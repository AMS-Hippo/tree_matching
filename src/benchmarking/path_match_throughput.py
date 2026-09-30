from __future__ import annotations

"""Many-query / template-bank throughput benchmarks for tree-path matching.

The pairwise benchmark in :mod:`benchmarking.path_match_benchmark` asks how one
algorithm behaves on one tree pair.  This module measures a different, common
workflow: prepare a fixed bank of templates, then match many query trees against
all templates to produce a score matrix.

The timing split is deliberate:

* encoder fitting (specialized equality/overlap matchers);
* reusable query-tree preparation;
* reusable template-tree preparation;
* warm score-matrix search.

Prepared and direct APIs are compared on a small deterministic sample before
any timing result is accepted.  Exact score matrices are also compared
entry-by-entry before they are used as an accuracy oracle for beam methods.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import json
import math
import os
import platform
import re
import sys
import traceback

import numpy as np
import pandas as pd

from .implementation_comparison import write_implementation_comparisons

from path_matcher import SparseCandidateConfig
from path_matcher.tree_data import TreeData

from .path_match_benchmark import (
    AlgorithmSpec,
    _environment_metadata,
    _flatten_diagnostics,
    _matcher_for,
    resolve_algorithm_specs,
)
from .path_match_cases import PairRegime, ScoreModel, generate_synthetic_pair
from .shared import config_hash, ensure_dir, load_config
from .timing import isolation_metadata, run_isolated_task, timing_sample_summary


DEFAULT_EXACT_SCORE_ABS_TOL = 1e-5
DEFAULT_EXACT_SCORE_REL_TOL = 1e-6


@dataclass(frozen=True)
class ThroughputRegime:
    """A synthetic query-corpus and template-bank workload."""

    pair_regime: PairRegime
    n_queries: int
    n_templates: int
    expected_algorithm: Optional[str] = None
    notes: str = ""

    @property
    def name(self) -> str:
        return self.pair_regime.name

    @classmethod
    def from_mapping(cls, spec: Mapping[str, Any]) -> "ThroughputRegime":
        cfg = dict(spec)
        n_queries = int(cfg.pop("n_queries"))
        n_templates = int(cfg.pop("n_templates"))
        expected_algorithm = cfg.pop("expected_algorithm", None)
        notes = str(cfg.pop("notes", ""))
        # PairRegime's instance count is not used here.  Each query/template is
        # generated from a distinct deterministic instance stream.
        cfg["n_instances"] = 1
        cfg["expected_algorithm"] = expected_algorithm
        cfg["notes"] = notes
        pair_regime = PairRegime.from_mapping(cfg)
        return cls(
            pair_regime=pair_regime,
            n_queries=n_queries,
            n_templates=n_templates,
            expected_algorithm=expected_algorithm,
            notes=notes,
        )

    def __post_init__(self) -> None:
        if int(self.n_queries) < 1:
            raise ValueError("n_queries must be positive")
        if int(self.n_templates) < 1:
            raise ValueError("n_templates must be positive")

    def as_dict(self) -> Dict[str, Any]:
        out = self.pair_regime.as_dict()
        out["n_queries"] = int(self.n_queries)
        out["n_templates"] = int(self.n_templates)
        out["expected_algorithm"] = self.expected_algorithm
        out["notes"] = self.notes
        return out


@dataclass(frozen=True)
class ThroughputCorpus:
    regime_name: str
    queries: Tuple[TreeData, ...]
    templates: Tuple[TreeData, ...]
    score_model: ScoreModel
    metadata: Mapping[str, Any]

    @property
    def n_pairs(self) -> int:
        return len(self.queries) * len(self.templates)


def _label_keys(label: Any, score_mode: str) -> Tuple[str, ...]:
    if score_mode == "equality":
        return (str(label),)
    if label is None:
        return ()
    if isinstance(label, str):
        return (label,)
    if isinstance(label, (tuple, list, set, frozenset, np.ndarray)):
        return tuple(sorted({str(x) for x in label}))
    return (str(label),)


def _aggregate_postings(trees: Sequence[TreeData], score_mode: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tree in trees:
        for label in tree.label:
            for key in set(_label_keys(label, score_mode)):
                out[key] = out.get(key, 0) + 1
    return out


def _corpus_work_estimates(
    queries: Sequence[TreeData],
    templates: Sequence[TreeData],
    score_mode: str,
) -> Dict[str, int]:
    query_nodes = sum(int(tree.n) for tree in queries)
    template_nodes = sum(int(tree.n) for tree in templates)
    q_post = _aggregate_postings(queries, score_mode)
    t_post = _aggregate_postings(templates, score_mode)
    posting_hits = sum(int(count) * int(t_post.get(key, 0)) for key, count in q_post.items())
    return {
        "query_nodes": int(query_nodes),
        "template_nodes": int(template_nodes),
        "dense_cells_total": int(query_nodes * template_nodes),
        "posting_hits_total": int(posting_hits),
        "shared_keys": int(sum(1 for key in q_post if key in t_post)),
    }


def generate_throughput_corpus(regime: ThroughputRegime) -> ThroughputCorpus:
    """Generate deterministic query and template collections.

    This is a workload generator rather than a classification benchmark.  Each
    tree is generated independently from the same regime and score model.  The
    score matrix therefore measures matching throughput without conflating it
    with a particular downstream prediction rule.
    """

    queries: List[TreeData] = []
    templates: List[TreeData] = []
    score_model: Optional[ScoreModel] = None

    for i in range(int(regime.n_queries)):
        pair = generate_synthetic_pair(regime.pair_regime, instance_index=i)
        if score_model is None:
            score_model = pair.score_model
        elif pair.score_model != score_model:
            raise AssertionError("throughput query instances produced inconsistent score models")
        queries.append(pair.G)

    # Use a disjoint deterministic stream so changing n_queries does not alter
    # the templates generated for a fixed regime.
    template_offset = 1_000_000
    for j in range(int(regime.n_templates)):
        pair = generate_synthetic_pair(regime.pair_regime, instance_index=template_offset + j)
        if score_model is None:
            score_model = pair.score_model
        elif pair.score_model != score_model:
            raise AssertionError("throughput template instances produced inconsistent score models")
        templates.append(pair.H)

    assert score_model is not None
    estimates = _corpus_work_estimates(queries, templates, score_model.mode)
    metadata: Dict[str, Any] = {
        "regime": regime.as_dict(),
        "query_instance_indices": list(range(int(regime.n_queries))),
        "template_instance_indices": [template_offset + j for j in range(int(regime.n_templates))],
        "n_queries": int(regime.n_queries),
        "n_templates": int(regime.n_templates),
        "n_pairs": int(regime.n_queries * regime.n_templates),
        "query_sizes": [int(tree.n) for tree in queries],
        "template_sizes": [int(tree.n) for tree in templates],
        **estimates,
    }
    return ThroughputCorpus(
        regime_name=regime.name,
        queries=tuple(queries),
        templates=tuple(templates),
        score_model=score_model,
        metadata=metadata,
    )


@dataclass
class _PreparedAlgorithm:
    spec: AlgorithmSpec
    matcher: Any
    queries: Sequence[Any]
    templates: Sequence[Any]
    predict_pair: Callable[[int, int], Tuple[List[Tuple[int, int]], float]]
    predict_direct: Optional[Callable[[int, int], Tuple[List[Tuple[int, int]], float]]]
    encoder_fit_seconds: float = 0.0
    query_preparation_seconds: float = 0.0
    template_preparation_seconds: float = 0.0
    matcher_construction_seconds: float = 0.0
    preparation_kind: str = "none"

    @property
    def one_time_preparation_seconds(self) -> float:
        return float(
            self.matcher_construction_seconds
            + self.encoder_fit_seconds
            + self.query_preparation_seconds
            + self.template_preparation_seconds
        )


def _prepare_algorithm(corpus: ThroughputCorpus, spec: AlgorithmSpec) -> _PreparedAlgorithm:
    t0 = perf_counter()
    matcher = _matcher_for(spec, corpus.score_model, collect_diagnostics=False)
    construction_seconds = perf_counter() - t0

    if spec.family == "fast_dense":
        fit_start = perf_counter()
        matcher.fit_encoder([*corpus.queries, *corpus.templates])
        encoder_fit_seconds = perf_counter() - fit_start

        q_start = perf_counter()
        prepared_queries = [matcher.encode_tree(tree) for tree in corpus.queries]
        query_seconds = perf_counter() - q_start
        t_start = perf_counter()
        prepared_templates = [matcher.encode_tree(tree) for tree in corpus.templates]
        template_seconds = perf_counter() - t_start

        return _PreparedAlgorithm(
            spec=spec,
            matcher=matcher,
            queries=prepared_queries,
            templates=prepared_templates,
            predict_pair=lambda i, j: matcher.predict_encoded(prepared_queries[i], prepared_templates[j]),
            predict_direct=lambda i, j: matcher.predict(corpus.queries[i], corpus.templates[j]),
            encoder_fit_seconds=encoder_fit_seconds,
            query_preparation_seconds=query_seconds,
            template_preparation_seconds=template_seconds,
            matcher_construction_seconds=construction_seconds,
            preparation_kind="shared_encoder_plus_encoded_trees",
        )

    if spec.family == "fast_sparse":
        fit_start = perf_counter()
        matcher.fit_encoder([*corpus.queries, *corpus.templates])
        encoder_fit_seconds = perf_counter() - fit_start

        q_start = perf_counter()
        prepared_queries = [matcher.prepare_tree(tree) for tree in corpus.queries]
        query_seconds = perf_counter() - q_start
        t_start = perf_counter()
        prepared_templates = [matcher.prepare_tree(tree) for tree in corpus.templates]
        template_seconds = perf_counter() - t_start

        return _PreparedAlgorithm(
            spec=spec,
            matcher=matcher,
            queries=prepared_queries,
            templates=prepared_templates,
            predict_pair=lambda i, j: matcher.predict_prepared(prepared_queries[i], prepared_templates[j]),
            predict_direct=lambda i, j: matcher.predict(corpus.queries[i], corpus.templates[j]),
            encoder_fit_seconds=encoder_fit_seconds,
            query_preparation_seconds=query_seconds,
            template_preparation_seconds=template_seconds,
            matcher_construction_seconds=construction_seconds,
            preparation_kind="shared_encoder_plus_sparse_indices",
        )

    if spec.family == "fast_beam":
        fit_start = perf_counter()
        matcher.fit_encoder([*corpus.queries, *corpus.templates])
        encoder_fit_seconds = perf_counter() - fit_start

        q_start = perf_counter()
        prepared_queries = [matcher.prepare_tree(tree) for tree in corpus.queries]
        query_seconds = perf_counter() - q_start
        t_start = perf_counter()
        prepared_templates = [matcher.prepare_tree(tree) for tree in corpus.templates]
        template_seconds = perf_counter() - t_start

        return _PreparedAlgorithm(
            spec=spec,
            matcher=matcher,
            queries=prepared_queries,
            templates=prepared_templates,
            predict_pair=lambda i, j: matcher.predict_prepared(prepared_queries[i], prepared_templates[j]),
            predict_direct=lambda i, j: matcher.predict(corpus.queries[i], corpus.templates[j]),
            encoder_fit_seconds=encoder_fit_seconds,
            query_preparation_seconds=query_seconds,
            template_preparation_seconds=template_seconds,
            matcher_construction_seconds=construction_seconds,
            preparation_kind="shared_encoder_plus_fast_beam_indices",
        )

    if spec.family == "generic_sparse":
        q_start = perf_counter()
        prepared_queries = [matcher.preprocess(tree) for tree in corpus.queries]
        query_seconds = perf_counter() - q_start
        t_start = perf_counter()
        prepared_templates = [matcher.preprocess(tree) for tree in corpus.templates]
        template_seconds = perf_counter() - t_start

        return _PreparedAlgorithm(
            spec=spec,
            matcher=matcher,
            queries=prepared_queries,
            templates=prepared_templates,
            predict_pair=lambda i, j: matcher.predict(prepared_queries[i], prepared_templates[j]),
            predict_direct=lambda i, j: matcher.predict(corpus.queries[i], corpus.templates[j]),
            query_preparation_seconds=query_seconds,
            template_preparation_seconds=template_seconds,
            matcher_construction_seconds=construction_seconds,
            preparation_kind="bucket_indices_and_tree_intervals",
        )

    # Dense generic and beam methods consume TreeData directly.  Tree creation
    # belongs to corpus generation, not to an algorithm's reusable setup.
    return _PreparedAlgorithm(
        spec=spec,
        matcher=matcher,
        queries=corpus.queries,
        templates=corpus.templates,
        predict_pair=lambda i, j: matcher.predict(corpus.queries[i], corpus.templates[j]),
        predict_direct=None,
        matcher_construction_seconds=construction_seconds,
        preparation_kind="none",
    )


def _strict_ancestor(parent: np.ndarray, u: int, v: int) -> bool:
    node = int(v)
    while node >= 0:
        node = int(parent[node])
        if node == int(u):
            return True
    return False


def _validate_returned_path(
    G: TreeData,
    H: TreeData,
    score_model: ScoreModel,
    path: Sequence[Tuple[int, int]],
    score: float,
    *,
    atol: float,
    rtol: float,
) -> Tuple[bool, bool, float]:
    valid = True
    for (u1, v1), (u2, v2) in zip(path, path[1:]):
        if not _strict_ancestor(np.asarray(G.parent), int(u1), int(u2)):
            valid = False
            break
        if not _strict_ancestor(np.asarray(H.parent), int(v1), int(v2)):
            valid = False
            break
    explicit = float(sum(score_model.score(G.label[int(u)], H.label[int(v)]) for u, v in path))
    score_ok = bool(np.isclose(explicit, float(score), atol=atol, rtol=rtol))
    return valid, score_ok, explicit


def _validation_pair_indices(n_queries: int, n_templates: int, count: int) -> List[Tuple[int, int]]:
    total = int(n_queries) * int(n_templates)
    if total <= 0 or count <= 0:
        return []
    use = min(int(count), total)
    flat = np.unique(np.rint(np.linspace(0, total - 1, num=use)).astype(int))
    return [(int(k // n_templates), int(k % n_templates)) for k in flat]


def _validate_context(
    context: _PreparedAlgorithm,
    corpus: ThroughputCorpus,
    *,
    pair_count: int,
    atol: float,
    rtol: float,
) -> Dict[str, Any]:
    indices = _validation_pair_indices(len(corpus.queries), len(corpus.templates), pair_count)
    paths_valid = True
    reported_scores_valid = True
    prepared_passed = True
    max_prepared_score_diff = 0.0
    prepared_path_mismatches = 0

    for i, j in indices:
        path, score = context.predict_pair(i, j)
        valid, score_ok, _ = _validate_returned_path(
            corpus.queries[i], corpus.templates[j], corpus.score_model, path, score,
            atol=atol, rtol=rtol,
        )
        paths_valid = paths_valid and valid
        reported_scores_valid = reported_scores_valid and score_ok

        if context.predict_direct is not None:
            direct_path, direct_score = context.predict_direct(i, j)
            diff = abs(float(score) - float(direct_score))
            max_prepared_score_diff = max(max_prepared_score_diff, diff)
            if not np.isclose(float(score), float(direct_score), atol=atol, rtol=rtol):
                prepared_passed = False
            if list(path) != list(direct_path):
                prepared_path_mismatches += 1
                prepared_passed = False

    if not paths_valid:
        raise AssertionError(f"{context.spec.name} returned an ancestor-inconsistent path")
    if not reported_scores_valid:
        raise AssertionError(f"{context.spec.name} reported a score inconsistent with its returned path")
    if context.predict_direct is not None and not prepared_passed:
        raise AssertionError(
            f"{context.spec.name} prepared and direct APIs disagreed "
            f"(max score difference {max_prepared_score_diff:g}, "
            f"path mismatches {prepared_path_mismatches})"
        )

    return {
        "validation_pairs_checked": len(indices),
        "validation_paths_valid": bool(paths_valid),
        "validation_reported_scores_valid": bool(reported_scores_valid),
        "prepared_validation_applicable": context.predict_direct is not None,
        "prepared_validation_passed": bool(prepared_passed) if context.predict_direct is not None else None,
        "prepared_validation_max_score_difference": (
            float(max_prepared_score_diff) if context.predict_direct is not None else None
        ),
        "prepared_validation_path_mismatches": (
            int(prepared_path_mismatches) if context.predict_direct is not None else None
        ),
    }


def _run_score_matrix(context: _PreparedAlgorithm) -> np.ndarray:
    n_queries = len(context.queries)
    n_templates = len(context.templates)
    scores = np.empty((n_queries, n_templates), dtype=np.float64)
    for i in range(n_queries):
        for j in range(n_templates):
            _, score = context.predict_pair(i, j)
            scores[i, j] = float(score)
    return scores


def _collect_sample_diagnostics(
    context: _PreparedAlgorithm,
    *,
    pair_count: int,
) -> Dict[str, Any]:
    indices = _validation_pair_indices(len(context.queries), len(context.templates), pair_count)
    if not indices or not hasattr(context.matcher, "collect_diagnostics"):
        return {"diagnostic_pairs_checked": 0}

    original = bool(context.matcher.collect_diagnostics)
    context.matcher.collect_diagnostics = True
    records: List[Dict[str, Any]] = []
    try:
        for i, j in indices:
            context.predict_pair(i, j)
            records.append(_flatten_diagnostics(getattr(context.matcher, "last_diagnostics_", None)))
    finally:
        context.matcher.collect_diagnostics = original

    numeric_keys = sorted({
        key
        for record in records
        for key, value in record.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
    })
    out: Dict[str, Any] = {"diagnostic_pairs_checked": len(records)}
    for key in numeric_keys:
        values = [float(record[key]) for record in records if record.get(key) is not None]
        finite = [value for value in values if math.isfinite(value)]
        if finite:
            out[f"sample_median_{key}"] = float(median(finite))
            out[f"sample_mean_{key}"] = float(np.mean(finite))
    return out


def _skip_reason(
    corpus: ThroughputCorpus,
    spec: AlgorithmSpec,
    *,
    max_total_dense_cells: Optional[int],
    max_total_generic_dense_cells: Optional[int],
    max_total_fast_dense_cells: Optional[int],
    max_total_sparse_posting_hits: Optional[int],
) -> Optional[str]:
    if corpus.score_model.mode not in spec.compatible_score_modes:
        return f"incompatible_score_mode:{corpus.score_model.mode}"
    dense_cells = int(corpus.metadata["dense_cells_total"])
    posting_hits = int(corpus.metadata["posting_hits_total"])
    is_generic_dense = spec.family == "generic" and str(spec.kwargs.get("method", "")) == "exact"
    is_fast_dense = spec.family == "fast_dense"
    is_sparse = spec.family in {"generic_sparse", "fast_sparse"}
    generic_dense_limit = (
        max_total_generic_dense_cells
        if max_total_generic_dense_cells is not None
        else max_total_dense_cells
    )
    fast_dense_limit = (
        max_total_fast_dense_cells
        if max_total_fast_dense_cells is not None
        else max_total_dense_cells
    )
    if is_generic_dense and generic_dense_limit is not None and dense_cells > int(generic_dense_limit):
        return f"total_generic_dense_cell_limit:{dense_cells}>{int(generic_dense_limit)}"
    if is_fast_dense and fast_dense_limit is not None and dense_cells > int(fast_dense_limit):
        return f"total_fast_dense_cell_limit:{dense_cells}>{int(fast_dense_limit)}"
    if is_sparse and max_total_sparse_posting_hits is not None and posting_hits > int(max_total_sparse_posting_hits):
        return f"total_sparse_posting_limit:{posting_hits}>{int(max_total_sparse_posting_hits)}"
    return None


def _run_algorithm_on_corpus_in_process(
    corpus: ThroughputCorpus,
    spec: AlgorithmSpec,
    *,
    repeats: int = 3,
    warmup_pairs: int = 1,
    validation_pairs: int = 3,
    diagnostic_pairs: int = 3,
    max_total_dense_cells: Optional[int] = None,
    max_total_generic_dense_cells: Optional[int] = None,
    max_total_fast_dense_cells: Optional[int] = None,
    max_total_sparse_posting_hits: Optional[int] = None,
    exact_score_abs_tol: float = DEFAULT_EXACT_SCORE_ABS_TOL,
    exact_score_rel_tol: float = DEFAULT_EXACT_SCORE_REL_TOL,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Run one algorithm over the complete query-template score matrix."""

    base: Dict[str, Any] = {
        "regime": corpus.regime_name,
        "algorithm": spec.name,
        "algorithm_family": spec.family,
        "algorithm_exact": bool(spec.exact),
        "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
        "score_mode": corpus.score_model.mode,
        "n_queries": len(corpus.queries),
        "n_templates": len(corpus.templates),
        "n_pairs": corpus.n_pairs,
        "query_nodes": int(corpus.metadata["query_nodes"]),
        "template_nodes": int(corpus.metadata["template_nodes"]),
        "dense_cells_total": int(corpus.metadata["dense_cells_total"]),
        "posting_hits_total": int(corpus.metadata["posting_hits_total"]),
        "score_matrix_seconds_median": None,
        "pairs_per_second": None,
        "matrix_score_sum": None,
    }
    skip = _skip_reason(
        corpus,
        spec,
        max_total_dense_cells=max_total_dense_cells,
        max_total_generic_dense_cells=max_total_generic_dense_cells,
        max_total_fast_dense_cells=max_total_fast_dense_cells,
        max_total_sparse_posting_hits=max_total_sparse_posting_hits,
    )
    if skip is not None:
        return {**base, "status": "skipped", "skip_reason": skip}, None

    repeats = max(1, int(repeats))
    warmup_pairs = max(0, int(warmup_pairs))
    try:
        context = _prepare_algorithm(corpus, spec)
        validation = _validate_context(
            context,
            corpus,
            pair_count=int(validation_pairs),
            atol=float(exact_score_abs_tol),
            rtol=float(exact_score_rel_tol),
        )

        warm_indices = _validation_pair_indices(
            len(corpus.queries), len(corpus.templates), warmup_pairs
        )
        for i, j in warm_indices:
            context.predict_pair(i, j)

        matrices: List[np.ndarray] = []
        times: List[float] = []
        for _ in range(repeats):
            start = perf_counter()
            matrix = _run_score_matrix(context)
            times.append(perf_counter() - start)
            matrices.append(matrix)

        reference_matrix = matrices[0]
        repeat_max_abs_diff = 0.0
        for matrix in matrices[1:]:
            repeat_max_abs_diff = max(
                repeat_max_abs_diff,
                float(np.max(np.abs(matrix - reference_matrix), initial=0.0)),
            )

        diagnostics = _collect_sample_diagnostics(context, pair_count=int(diagnostic_pairs))
        matrix_seconds = float(median(times))
        n_pairs = int(corpus.n_pairs)
        row: Dict[str, Any] = {
            **base,
            "status": "ok",
            "preparation_kind": context.preparation_kind,
            "matcher_construction_seconds": float(context.matcher_construction_seconds),
            "encoder_fit_seconds": float(context.encoder_fit_seconds),
            "query_preparation_seconds": float(context.query_preparation_seconds),
            "template_preparation_seconds": float(context.template_preparation_seconds),
            "one_time_preparation_seconds": float(context.one_time_preparation_seconds),
            **timing_sample_summary(times, prefix="score_matrix_seconds"),
            "setup_plus_one_matrix_seconds": float(context.one_time_preparation_seconds + matrix_seconds),
            "pairs_per_second": float(n_pairs / matrix_seconds) if matrix_seconds > 0.0 else math.inf,
            "queries_per_second": float(len(corpus.queries) / matrix_seconds) if matrix_seconds > 0.0 else math.inf,
            "seconds_per_pair": float(matrix_seconds / n_pairs),
            "seconds_per_query": float(matrix_seconds / len(corpus.queries)),
            "matrix_score_sum": float(reference_matrix.sum()),
            "matrix_score_mean": float(reference_matrix.mean()),
            "matrix_score_max": float(reference_matrix.max(initial=0.0)),
            "matrix_nonzero_fraction": float(np.mean(reference_matrix > 0.0)),
            "repeat_matrix_max_abs_difference": float(repeat_max_abs_diff),
            "score_matrix_shape": json.dumps(list(reference_matrix.shape)),
            **validation,
            **diagnostics,
        }
        return row, reference_matrix
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }, None




def _measure_throughput_cold_start(
    corpus: ThroughputCorpus,
    spec: AlgorithmSpec,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Measure the first prepared score-matrix pass in a fresh worker."""

    context = _prepare_algorithm(corpus, spec)
    started = perf_counter()
    matrix = _run_score_matrix(context)
    score_matrix_seconds = float(perf_counter() - started)
    metrics = {
        "cold_matcher_construction_seconds": float(context.matcher_construction_seconds),
        "cold_encoder_fit_seconds": float(context.encoder_fit_seconds),
        "cold_query_preparation_seconds": float(context.query_preparation_seconds),
        "cold_template_preparation_seconds": float(context.template_preparation_seconds),
        "cold_one_time_preparation_seconds": float(context.one_time_preparation_seconds),
        "cold_score_matrix_seconds": score_matrix_seconds,
        "cold_setup_plus_one_matrix_seconds": float(
            context.one_time_preparation_seconds + score_matrix_seconds
        ),
        "cold_matrix_score_sum": float(matrix.sum()),
    }
    return metrics, matrix

def run_algorithm_on_corpus(
    corpus: ThroughputCorpus,
    spec: AlgorithmSpec,
    *,
    repeats: int = 3,
    warmup_pairs: int = 1,
    validation_pairs: int = 3,
    diagnostic_pairs: int = 3,
    max_total_dense_cells: Optional[int] = None,
    max_total_generic_dense_cells: Optional[int] = None,
    max_total_fast_dense_cells: Optional[int] = None,
    max_total_sparse_posting_hits: Optional[int] = None,
    exact_score_abs_tol: float = DEFAULT_EXACT_SCORE_ABS_TOL,
    exact_score_rel_tol: float = DEFAULT_EXACT_SCORE_REL_TOL,
    execution_mode: str = "in_process",
    timeout_seconds: Optional[float] = None,
    thread_count: Optional[int] = None,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    """Run one algorithm over a complete query-template score matrix.

    In isolated mode the complete algorithm/regime job runs in a fresh worker,
    giving enforceable timeouts, pre-import thread settings, and lifetime peak
    resident memory.  Warm score-matrix timings remain internal to the worker.
    """

    mode = str(execution_mode).strip().lower()
    if mode not in {"in_process", "isolated"}:
        raise ValueError("execution_mode must be 'in_process' or 'isolated'")
    if mode == "in_process":
        row, matrix = _run_algorithm_on_corpus_in_process(
            corpus,
            spec,
            repeats=repeats,
            warmup_pairs=warmup_pairs,
            validation_pairs=validation_pairs,
            diagnostic_pairs=diagnostic_pairs,
            max_total_dense_cells=max_total_dense_cells,
            max_total_generic_dense_cells=max_total_generic_dense_cells,
            max_total_fast_dense_cells=max_total_fast_dense_cells,
            max_total_sparse_posting_hits=max_total_sparse_posting_hits,
            exact_score_abs_tol=exact_score_abs_tol,
            exact_score_rel_tol=exact_score_rel_tol,
        )
        row.setdefault("execution_mode", "in_process")
        row.setdefault("thread_count_requested", thread_count)
        row.setdefault("thread_settings_enforced", False)
        row.setdefault("timeout_seconds", timeout_seconds)
        return row, matrix

    skip = _skip_reason(
        corpus,
        spec,
        max_total_dense_cells=max_total_dense_cells,
        max_total_generic_dense_cells=max_total_generic_dense_cells,
        max_total_fast_dense_cells=max_total_fast_dense_cells,
        max_total_sparse_posting_hits=max_total_sparse_posting_hits,
    )
    if skip is not None:
        row, matrix = _run_algorithm_on_corpus_in_process(
            corpus,
            spec,
            repeats=repeats,
            warmup_pairs=warmup_pairs,
            validation_pairs=validation_pairs,
            diagnostic_pairs=diagnostic_pairs,
            max_total_dense_cells=max_total_dense_cells,
            max_total_generic_dense_cells=max_total_generic_dense_cells,
            max_total_fast_dense_cells=max_total_fast_dense_cells,
            max_total_sparse_posting_hits=max_total_sparse_posting_hits,
            exact_score_abs_tol=exact_score_abs_tol,
            exact_score_rel_tol=exact_score_rel_tol,
        )
        row.update({
            "execution_mode": "isolated",
            "thread_count_requested": thread_count,
            "thread_settings_enforced": False,
            "execution_skipped_before_worker": True,
            "timeout_seconds": timeout_seconds,
        })
        return row, matrix

    isolated = run_isolated_task(
        "throughput",
        {
            "corpus": corpus,
            "spec": spec,
            "repeats": repeats,
            "warmup_pairs": warmup_pairs,
            "validation_pairs": validation_pairs,
            "diagnostic_pairs": diagnostic_pairs,
            "max_total_dense_cells": max_total_dense_cells,
            "max_total_generic_dense_cells": max_total_generic_dense_cells,
            "max_total_fast_dense_cells": max_total_fast_dense_cells,
            "max_total_sparse_posting_hits": max_total_sparse_posting_hits,
            "exact_score_abs_tol": exact_score_abs_tol,
            "exact_score_rel_tol": exact_score_rel_tol,
        },
        timeout_seconds=timeout_seconds,
        thread_count=thread_count,
    )
    base = {
        "regime": corpus.regime_name,
        "algorithm": spec.name,
        "algorithm_family": spec.family,
        "algorithm_exact": bool(spec.exact),
        "algorithm_kwargs": json.dumps(dict(spec.kwargs), sort_keys=True, default=str),
        "score_mode": corpus.score_model.mode,
        "n_queries": len(corpus.queries),
        "n_templates": len(corpus.templates),
        "n_pairs": corpus.n_pairs,
        "query_nodes": int(corpus.metadata["query_nodes"]),
        "template_nodes": int(corpus.metadata["template_nodes"]),
        "dense_cells_total": int(corpus.metadata["dense_cells_total"]),
        "posting_hits_total": int(corpus.metadata["posting_hits_total"]),
        "score_matrix_seconds_median": None,
        "pairs_per_second": None,
        "matrix_score_sum": None,
    }
    if isolated.status == "timeout":
        return {
            **base,
            "status": "timeout",
            "error_type": "TimeoutError",
            "error_message": f"isolated worker exceeded {timeout_seconds} seconds",
            **isolation_metadata(isolated, None),
        }, None
    if isolated.status != "completed" or isolated.payload is None:
        return {
            **base,
            "status": "error",
            "error_type": "IsolatedWorkerError",
            "error_message": isolated.stderr.strip() or "isolated worker failed",
            "worker_stdout": isolated.stdout[-20000:],
            "worker_stderr": isolated.stderr[-20000:],
            **isolation_metadata(isolated, isolated.payload),
        }, None

    payload = dict(isolated.payload)
    row = dict(payload.get("row", {}))
    matrix = payload.get("matrix")
    if not row:
        return {
            **base,
            "status": "error",
            "error_type": "IsolatedWorkerProtocolError",
            "error_message": "isolated worker returned no benchmark row",
            **isolation_metadata(isolated, payload),
        }, None
    cold_metrics = payload.get("cold_metrics", {})
    if isinstance(cold_metrics, Mapping):
        row.update(dict(cold_metrics))
    cold_matrix_max_abs_difference = payload.get("cold_matrix_max_abs_difference")
    if cold_matrix_max_abs_difference is not None:
        row["cold_matrix_max_abs_difference_from_warm"] = float(
            cold_matrix_max_abs_difference
        )
    row.update(isolation_metadata(isolated, payload))
    if matrix is not None:
        matrix = np.asarray(matrix, dtype=np.float64)
    return row, matrix

def _agreement_tolerance(values: np.ndarray, abs_tol: float, rel_tol: float) -> np.ndarray:
    scale = np.maximum(1.0, np.abs(values))
    return float(abs_tol) + float(rel_tol) * scale


def add_matrix_oracle_columns(
    rows: pd.DataFrame,
    matrices: Mapping[Tuple[str, str], np.ndarray],
    *,
    exact_score_abs_tol: float = DEFAULT_EXACT_SCORE_ABS_TOL,
    exact_score_rel_tol: float = DEFAULT_EXACT_SCORE_REL_TOL,
) -> pd.DataFrame:
    """Validate exact score matrices and attach approximate accuracy columns."""

    out = rows.copy()
    out["oracle_status"] = pd.Series([None] * len(out), index=out.index, dtype="object")
    out["oracle_valid"] = pd.Series([pd.NA] * len(out), index=out.index, dtype="boolean")
    out["score_consistent_with_oracle"] = pd.Series(
        [pd.NA] * len(out), index=out.index, dtype="boolean"
    )
    for column in (
        "exact_method_count",
        "exact_matrix_max_abs_difference",
        "matrix_total_accuracy",
        "mean_pair_accuracy",
        "median_pair_accuracy",
        "min_pair_accuracy",
        "exact_pair_fraction",
        "throughput_timing_score",
        "throughput_accuracy_harmonic_score",
    ):
        out[column] = np.nan

    for regime, group in out.groupby("regime", sort=False):
        exact_names = [
            str(row.algorithm)
            for row in group.itertuples()
            if bool(row.algorithm_exact)
            and row.status == "ok"
            and (str(regime), str(row.algorithm)) in matrices
        ]
        oracle: Optional[np.ndarray] = None
        max_diff: Optional[float] = None
        if not exact_names:
            oracle_status = "no_exact_result"
            oracle_valid = False
        else:
            exact_mats = [np.asarray(matrices[(str(regime), name)], dtype=float) for name in exact_names]
            oracle = np.maximum.reduce(exact_mats)
            max_diff = 0.0
            agreed = True
            for matrix in exact_mats:
                diff = np.abs(matrix - oracle)
                max_diff = max(max_diff, float(diff.max(initial=0.0)))
                tolerance = _agreement_tolerance(oracle, exact_score_abs_tol, exact_score_rel_tol)
                if np.any(diff > tolerance):
                    agreed = False
            oracle_valid = bool(agreed)
            if not agreed:
                oracle_status = "exact_disagreement"
                oracle = None
            elif len(exact_names) == 1:
                oracle_status = "single_exact_result"
            else:
                oracle_status = "exact_agreement"

        regime_indices = list(group.index)
        out.loc[regime_indices, "oracle_status"] = oracle_status
        out.loc[regime_indices, "oracle_valid"] = bool(oracle_valid)
        out.loc[regime_indices, "exact_method_count"] = int(len(exact_names))
        out.loc[regime_indices, "exact_matrix_max_abs_difference"] = max_diff

        if oracle is not None:
            tolerance = _agreement_tolerance(oracle, exact_score_abs_tol, exact_score_rel_tol)
            oracle_sum = float(oracle.sum())
            for idx in regime_indices:
                row = out.loc[idx]
                key = (str(regime), str(row["algorithm"]))
                if row["status"] != "ok" or key not in matrices:
                    continue
                matrix = np.asarray(matrices[key], dtype=float)
                if matrix.shape != oracle.shape:
                    out.at[idx, "score_consistent_with_oracle"] = False
                    continue
                consistent = bool(np.all(matrix <= oracle + tolerance))
                out.at[idx, "score_consistent_with_oracle"] = consistent
                if not consistent:
                    continue

                near_equal = np.abs(matrix - oracle) <= tolerance
                ratios = np.ones_like(oracle, dtype=float)
                positive = oracle > tolerance
                ratios[positive] = matrix[positive] / oracle[positive]
                ratios[near_equal] = 1.0
                ratios = np.clip(ratios, 0.0, 1.0)
                if bool(np.all(near_equal)):
                    total_accuracy = 1.0
                elif oracle_sum <= float(exact_score_abs_tol):
                    total_accuracy = 1.0 if abs(float(matrix.sum())) <= float(exact_score_abs_tol) else 0.0
                else:
                    total_accuracy = min(1.0, max(0.0, float(matrix.sum()) / oracle_sum))
                out.at[idx, "matrix_total_accuracy"] = total_accuracy
                out.at[idx, "mean_pair_accuracy"] = float(ratios.mean())
                out.at[idx, "median_pair_accuracy"] = float(np.median(ratios))
                out.at[idx, "min_pair_accuracy"] = float(ratios.min(initial=1.0))
                out.at[idx, "exact_pair_fraction"] = float(np.mean(near_equal))

        ok_times = pd.to_numeric(
            group.loc[group["status"] == "ok", "score_matrix_seconds_median"],
            errors="coerce",
        )
        finite_times = ok_times[np.isfinite(ok_times) & (ok_times >= 0.0)]
        best_time = float(finite_times.min()) if not finite_times.empty else None
        for idx in regime_indices:
            if out.at[idx, "status"] != "ok" or best_time is None:
                continue
            value = out.at[idx, "score_matrix_seconds_median"]
            if value is None or pd.isna(value):
                continue
            value_f = float(value)
            timing_score = 1.0 if best_time == 0.0 and value_f == 0.0 else (
                0.0 if best_time == 0.0 else min(1.0, best_time / value_f)
            )
            out.at[idx, "throughput_timing_score"] = timing_score
            accuracy = out.at[idx, "matrix_total_accuracy"]
            if accuracy is not None and not pd.isna(accuracy) and timing_score + float(accuracy) > 0.0:
                out.at[idx, "throughput_accuracy_harmonic_score"] = (
                    2.0 * timing_score * float(accuracy) / (timing_score + float(accuracy))
                )

    return out


def summarize_throughput_rows(
    rows: pd.DataFrame,
    regimes: Sequence[ThroughputRegime],
    *,
    quality_floor: float = 0.99,
) -> pd.DataFrame:
    out = rows.copy()
    out["quality_floor"] = float(quality_floor)
    out["completed"] = out["status"].eq("ok")
    out["timed_out"] = out["status"].eq("timeout")
    out["quality_eligible"] = (
        (out["status"] == "ok")
        & out["oracle_valid"].eq(True)
        & out["score_consistent_with_oracle"].eq(True)
        & (pd.to_numeric(out["min_pair_accuracy"], errors="coerce") >= float(quality_floor))
    )
    out["throughput_frontier"] = False
    if "score_matrix_seconds_q25" in out and "score_matrix_seconds_q75" in out:
        med = pd.to_numeric(out.get("score_matrix_seconds_median"), errors="coerce")
        q25 = pd.to_numeric(out.get("score_matrix_seconds_q25"), errors="coerce")
        q75 = pd.to_numeric(out.get("score_matrix_seconds_q75"), errors="coerce")
        out["score_matrix_relative_iqr"] = np.where(
            med > 0.0, (q75 - q25) / med, np.nan
        )

    expected = {regime.name: regime.expected_algorithm for regime in regimes}
    regime_order = {regime.name: i for i, regime in enumerate(regimes)}
    out["regime_order"] = out["regime"].map(regime_order)
    out["expected_algorithm"] = out["regime"].map(expected)
    out["observed_fastest_quality_eligible"] = None
    out["observed_best_harmonic_score"] = None

    for regime, group in out.groupby("regime", sort=False):
        eligible = group[group["quality_eligible"] == True]  # noqa: E712
        fastest: Optional[str] = None
        if not eligible.empty:
            fastest = str(
                eligible.sort_values("score_matrix_seconds_median", kind="mergesort").iloc[0]["algorithm"]
            )
        harmonic_rows = group[
            (group["status"] == "ok")
            & pd.to_numeric(group["throughput_accuracy_harmonic_score"], errors="coerce").notna()
        ]
        best_harmonic: Optional[str] = None
        if not harmonic_rows.empty:
            best_harmonic = str(
                harmonic_rows.sort_values(
                    "throughput_accuracy_harmonic_score", ascending=False, kind="mergesort"
                ).iloc[0]["algorithm"]
            )
        out.loc[group.index, "observed_fastest_quality_eligible"] = fastest
        out.loc[group.index, "observed_best_harmonic_score"] = best_harmonic

        # A point is on the empirical speed/accuracy frontier when no other
        # successful method is both at least as accurate and no slower, with one
        # strict inequality.  Use aggregate matrix accuracy for this display.
        successful = group[
            (group["status"] == "ok")
            & pd.to_numeric(group["matrix_total_accuracy"], errors="coerce").notna()
            & pd.to_numeric(group["score_matrix_seconds_median"], errors="coerce").notna()
        ]
        for idx, row in successful.iterrows():
            dominated = False
            for jdx, other in successful.iterrows():
                if idx == jdx:
                    continue
                no_slower = float(other["score_matrix_seconds_median"]) <= float(row["score_matrix_seconds_median"])
                no_less_accurate = float(other["matrix_total_accuracy"]) >= float(row["matrix_total_accuracy"])
                strictly_better = (
                    float(other["score_matrix_seconds_median"]) < float(row["score_matrix_seconds_median"])
                    or float(other["matrix_total_accuracy"]) > float(row["matrix_total_accuracy"])
                )
                if no_slower and no_less_accurate and strictly_better:
                    dominated = True
                    break
            out.at[idx, "throughput_frontier"] = not dominated

    return out.sort_values(["regime_order", "algorithm_config_index"], kind="mergesort").reset_index(drop=True)


def _safe_npz_key(regime: str, algorithm: str) -> str:
    text = f"{regime}__{algorithm}"
    return re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")


def run_throughput_config(
    config_or_path: str | Path | Mapping[str, Any],
    *,
    output_dir: Optional[str | Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run a complete many-query/template-bank benchmark suite."""

    config = load_config(config_or_path)
    specs = resolve_algorithm_specs(config.get("algorithms", []))
    regimes = [ThroughputRegime.from_mapping(item) for item in config.get("regimes", [])]
    if not regimes:
        raise ValueError("throughput suite must contain at least one regime")

    repeats = max(1, int(config.get("repeats", 3)))
    warmup_pairs = max(0, int(config.get("warmup_pairs", 1)))
    validation_pairs = max(0, int(config.get("validation_pairs", 3)))
    diagnostic_pairs = max(0, int(config.get("diagnostic_pairs", 3)))
    quality_floor = float(config.get("quality_floor", 0.99))
    exact_abs_tol = float(config.get("exact_score_abs_tol", DEFAULT_EXACT_SCORE_ABS_TOL))
    exact_rel_tol = float(config.get("exact_score_rel_tol", DEFAULT_EXACT_SCORE_REL_TOL))
    max_dense = config.get("max_total_dense_cells")
    max_generic_dense = config.get("max_total_generic_dense_cells")
    max_fast_dense = config.get("max_total_fast_dense_cells")
    max_sparse = config.get("max_total_sparse_posting_hits")
    order_seed = int(config.get("algorithm_order_seed", 20260928))
    execution_mode = str(config.get("execution_mode", "in_process")).strip().lower()
    timeout_raw = config.get("timeout_seconds")
    timeout_seconds = None if timeout_raw is None else float(timeout_raw)
    thread_raw = config.get("thread_count")
    thread_count = None if thread_raw is None else int(thread_raw)
    if execution_mode not in {"in_process", "isolated"}:
        raise ValueError("execution_mode must be 'in_process' or 'isolated'")
    if timeout_seconds is not None and (
        not math.isfinite(timeout_seconds) or timeout_seconds <= 0.0
    ):
        raise ValueError("timeout_seconds must be positive and finite or null")
    if thread_count is not None and thread_count < 1:
        raise ValueError("thread_count must be positive or null")

    rows: List[Dict[str, Any]] = []
    matrices: Dict[Tuple[str, str], np.ndarray] = {}
    corpus_metadata: Dict[str, Any] = {}
    generation_seconds: Dict[str, float] = {}

    for regime_index, regime in enumerate(regimes):
        start = perf_counter()
        corpus = generate_throughput_corpus(regime)
        generation_seconds[regime.name] = perf_counter() - start
        corpus_metadata[regime.name] = dict(corpus.metadata)

        rng = np.random.default_rng(np.random.SeedSequence([order_seed, regime_index]))
        order = list(rng.permutation(len(specs)).astype(int))
        for execution_position, spec_index in enumerate(order):
            spec = specs[spec_index]
            row, matrix = run_algorithm_on_corpus(
                corpus,
                spec,
                repeats=repeats,
                warmup_pairs=warmup_pairs,
                validation_pairs=validation_pairs,
                diagnostic_pairs=diagnostic_pairs,
                max_total_dense_cells=(None if max_dense is None else int(max_dense)),
                max_total_generic_dense_cells=(
                    None if max_generic_dense is None else int(max_generic_dense)
                ),
                max_total_fast_dense_cells=(
                    None if max_fast_dense is None else int(max_fast_dense)
                ),
                max_total_sparse_posting_hits=(None if max_sparse is None else int(max_sparse)),
                exact_score_abs_tol=exact_abs_tol,
                exact_score_rel_tol=exact_rel_tol,
                execution_mode=execution_mode,
                timeout_seconds=timeout_seconds,
                thread_count=thread_count,
            )
            row["regime_order"] = regime_index
            row["algorithm_config_index"] = spec_index
            row["execution_order_position"] = execution_position
            row["corpus_generation_seconds"] = generation_seconds[regime.name]
            rows.append(row)
            if matrix is not None:
                matrices[(regime.name, spec.name)] = matrix

    rows_df = pd.DataFrame(rows)
    rows_df = add_matrix_oracle_columns(
        rows_df,
        matrices,
        exact_score_abs_tol=exact_abs_tol,
        exact_score_rel_tol=exact_rel_tol,
    )
    summary_df = summarize_throughput_rows(rows_df, regimes, quality_floor=quality_floor)

    suite_name = str(config.get("suite_name", "path_match_throughput"))
    metadata: Dict[str, Any] = {
        "suite_name": suite_name,
        "config": config,
        "config_hash": config_hash(config, prefix="throughput_"),
        "algorithms": [spec.as_dict() for spec in specs],
        "regimes": [regime.as_dict() for regime in regimes],
        "corpora": corpus_metadata,
        "corpus_generation_seconds": generation_seconds,
        "environment": _environment_metadata(),
        "timing_semantics": {
            "one_time_preparation_seconds": "matcher construction + encoder fit + reusable query/template preparation",
            "cold_score_matrix_seconds": "first complete score-matrix pass in a fresh worker before validation, diagnostics, or warm-up",
            "score_matrix_seconds_median": "median warm time to compute every query-template score, including traceback",
            "setup_plus_one_matrix_seconds": "one-time preparation plus one median warm score-matrix pass",
            "fresh_process_total_seconds": "complete isolated worker wall time including startup, imports, cold pass, validation, diagnostics, setup, warm-up, repeats, and serialization",
            "peak_rss_mib": "lifetime worker peak RSS, including runtime and imported libraries",
        },
        "algorithm_order_seed": order_seed,
        "algorithm_order_strategy": (
            "isolated_subprocesses_with_seeded_scheduling_order"
            if execution_mode == "isolated"
            else "one_seeded_permutation_per_regime"
        ),
        "execution_mode": execution_mode,
        "timeout_seconds": timeout_seconds,
        "thread_count": thread_count,
        "thread_settings_enforced": execution_mode == "isolated" and thread_count is not None,
        "matrix_archive": {},
    }

    if output_dir is not None:
        out_dir = ensure_dir(output_dir)
        rows_path = out_dir / "throughput_rows.csv"
        summary_path = out_dir / "throughput_summary.csv"
        metadata_path = out_dir / "throughput_metadata.json"
        matrix_path = out_dir / "throughput_score_matrices.npz"

        rows_df.to_csv(rows_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        archive_payload: Dict[str, np.ndarray] = {}
        matrix_map: Dict[str, Dict[str, str]] = {}
        used_keys: set[str] = set()
        for (regime, algorithm), matrix in matrices.items():
            base = _safe_npz_key(regime, algorithm) or "matrix"
            key = base
            suffix = 2
            while key in used_keys:
                key = f"{base}_{suffix}"
                suffix += 1
            used_keys.add(key)
            archive_payload[key] = np.asarray(matrix, dtype=np.float64)
            matrix_map.setdefault(regime, {})[algorithm] = key
        np.savez_compressed(matrix_path, **archive_payload)
        metadata["matrix_archive"] = {
            "path": matrix_path.name,
            "keys": matrix_map,
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

        write_implementation_comparisons(
            rows_df, out_dir, kind="throughput", matrices=matrices,
            abs_tol=float(config.get("exact_score_abs_tol", 1e-5)),
            rel_tol=float(config.get("exact_score_rel_tol", 1e-6)),
        )
    return rows_df, summary_df, metadata


__all__ = [
    "ThroughputRegime",
    "ThroughputCorpus",
    "generate_throughput_corpus",
    "run_algorithm_on_corpus",
    "add_matrix_oracle_columns",
    "summarize_throughput_rows",
    "run_throughput_config",
]
