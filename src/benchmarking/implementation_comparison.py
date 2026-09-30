"""Paired generic/encoded comparisons, without running a matcher.

Equality-beam score agreement is checked even without an exact oracle. It is
not an optimality certificate. Finite-budget overlap beams deliberately have
different proposal rules. Throughput checks compare every matrix entry, not
just the sum (opposite errors could cancel in that sum).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Tuple

import numpy as np
import pandas as pd

IMPLEMENTATION_PAIRS = (
    ("exact_dp", "exact_dense", "fast_dense"),
    ("partial_beam", "beam_partial_score", "fast_beam_partial"),
)
CONTEXT = ("sweep_name", "sweep_kind", "sweep_point", "sweep_x_name", "sweep_x_value", "regime")


def _float(value) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def _median(values) -> float:
    data = pd.to_numeric(values, errors="coerce").dropna()
    return float(data.median()) if len(data) else float("nan")


def _ratio(numerator, denominator) -> float:
    a, b = _float(numerator), _float(denominator)
    return a / b if np.isfinite(a) and np.isfinite(b) and a >= 0 and b > 0 else float("nan")


def compare_implementations(
    rows: pd.DataFrame,
    *,
    kind: str = "pairwise",
    matrices: Optional[Mapping[Tuple[str, str], np.ndarray]] = None,
    abs_tol: float = 1e-5,
    rel_tol: float = 1e-6,
) -> pd.DataFrame:
    """One comparison per instance (or throughput matrix) and method family.

    Incomplete pairs remain visible; their speedup is NaN. Unselected families
    are omitted. Unknown/absent matrices give an *unknown* agreement, never an
    agreement inferred from totals. Tolerances use max(abs_tol, rel_tol*scale),
    matching the benchmark's oracle convention.
    """
    if kind not in {"pairwise", "throughput"}:
        raise ValueError("kind must be pairwise or throughput")
    if min(abs_tol, rel_tol) < 0 or not np.isfinite([abs_tol, rel_tol]).all():
        raise ValueError("score tolerances must be finite and nonnegative")
    keys = [c for c in CONTEXT if c in rows]
    if kind == "pairwise":
        keys.append("instance")
    required = {"regime", "algorithm", "status", *keys}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")
    if rows.empty:
        return pd.DataFrame(columns=[*keys, "family", "paired_complete", "warm_speedup", "score_agreement"])
    if rows.duplicated([*keys, "algorithm"]).any():
        raise ValueError("Duplicate algorithm/instance rows; do not concatenate distinct runs without a run key")

    warm = "predict_seconds_median" if kind == "pairwise" else "score_matrix_seconds_median"
    setup = "setup_plus_warm_search_seconds" if kind == "pairwise" else "setup_plus_one_matrix_seconds"
    cold = "cold_setup_plus_predict_seconds" if kind == "pairwise" else "cold_setup_plus_one_matrix_seconds"
    score = "score" if kind == "pairwise" else "matrix_score_sum"
    records = []
    for values, group in rows.groupby(keys, sort=False, dropna=False):
        if not isinstance(values, tuple):
            values = (values,)
        context = dict(zip(keys, values))
        indexed = group.set_index("algorithm")
        for family, generic, encoded in IMPLEMENTATION_PAIRS:
            if generic not in indexed.index and encoded not in indexed.index:
                continue
            g = indexed.loc[generic] if generic in indexed.index else pd.Series(dtype=object)
            e = indexed.loc[encoded] if encoded in indexed.index else pd.Series(dtype=object)
            gs, es = str(g.get("status", "not_selected")), str(e.get("status", "not_selected"))
            complete = gs == es == "ok"
            mode = str(g.get("score_mode", e.get("score_mode", "unknown")))
            if complete and str(e.get("score_mode", mode)) != mode:
                raise ValueError(f"Score modes differ for {context}")
            for field in ("n_g", "n_h", "n_queries", "n_templates"):
                if complete and field in g and field in e and g[field] != e[field]:
                    raise ValueError(f"{field} differs between implementations for {context}")
            rec = {
                **context, "family": family, "score_mode": mode,
                "generic_algorithm": generic, "encoded_algorithm": encoded,
                "generic_status": gs, "encoded_status": es, "paired_complete": complete,
                "generic_warm_seconds": _float(g.get(warm)),
                "encoded_warm_seconds": _float(e.get(warm)),
                "warm_speedup": _ratio(g.get(warm), e.get(warm)) if complete else np.nan,
                "setup_speedup": _ratio(g.get(setup), e.get(setup)) if complete else np.nan,
                "cold_setup_speedup": _ratio(g.get(cold), e.get(cold)) if complete else np.nan,
                "generic_score": _float(g.get(score)), "encoded_score": _float(e.get(score)),
                "max_abs_score_difference": np.nan, "score_mismatch_count": np.nan,
                "score_agreement": None,
                "comparison_note": (
                    "different finite-budget overlap proposal rules" if family == "partial_beam" and mode == "overlap"
                    else "observed score agreement is not beam optimality; compare identical budgets" if family == "partial_beam"
                    else "same exact objective"
                ),
            }
            if complete:
                if kind == "pairwise":
                    a = np.asarray([rec["generic_score"]], dtype=float)
                    b = np.asarray([rec["encoded_score"]], dtype=float)
                elif matrices is not None and (str(context["regime"]), generic) in matrices and (str(context["regime"]), encoded) in matrices:
                    a = np.asarray(matrices[(str(context["regime"]), generic)], dtype=float)
                    b = np.asarray(matrices[(str(context["regime"]), encoded)], dtype=float)
                else:
                    a = b = None
                if a is not None:
                    if a.shape != b.shape:
                        rec.update(score_agreement=False, comparison_note="matrix shapes differ")
                    elif not (np.isfinite(a).all() and np.isfinite(b).all()):
                        rec.update(score_agreement=False, comparison_note="nonfinite score")
                    else:
                        diff = np.abs(a - b)
                        tolerance = np.maximum(abs_tol, rel_tol * np.maximum(np.abs(a), np.abs(b)))
                        mismatches = int(np.count_nonzero(diff > tolerance))
                        rec.update(max_abs_score_difference=float(diff.max()) if diff.size else 0.0,
                                   score_mismatch_count=mismatches, score_agreement=mismatches == 0)
            records.append(rec)
    return pd.DataFrame.from_records(records) if records else pd.DataFrame(
        columns=[*keys, "family", "score_mode", "paired_complete", "warm_speedup", "score_agreement"]
    )


def summarize_implementation_comparisons(comparisons: pd.DataFrame) -> pd.DataFrame:
    """Median of matched-instance speedups, not ratio of unmatched medians."""
    keys = [c for c in CONTEXT if c in comparisons] + ["family", "score_mode"]
    records = []
    if comparisons.empty:
        return pd.DataFrame(columns=[*keys, "paired_complete_count", "median_warm_speedup", "all_observed_scores_agree"])
    for values, group in comparisons.groupby(keys, sort=False, dropna=False):
        ok = group[group["paired_complete"]]
        known = ok[ok["score_agreement"].notna()]
        speed = pd.to_numeric(ok["warm_speedup"], errors="coerce").dropna()
        records.append({
            **dict(zip(keys, values)),
            "generic_algorithm": group["generic_algorithm"].iloc[0],
            "encoded_algorithm": group["encoded_algorithm"].iloc[0],
            "intended_pair_count": len(group), "paired_complete_count": len(ok),
            "complete_coverage": len(ok) == len(group), "scores_checked": len(known),
            "all_observed_scores_agree": bool(known["score_agreement"].all()) if len(known) else None,
            "score_mismatch_count": int(pd.to_numeric(known["score_mismatch_count"], errors="coerce").fillna(0).sum()),
            "max_abs_score_difference": pd.to_numeric(known["max_abs_score_difference"], errors="coerce").max(),
            "median_warm_speedup": speed.median() if len(speed) else np.nan,
            "q25_warm_speedup": speed.quantile(0.25) if len(speed) else np.nan,
            "q75_warm_speedup": speed.quantile(0.75) if len(speed) else np.nan,
            "median_setup_speedup": _median(ok["setup_speedup"]),
            "median_cold_setup_speedup": _median(ok["cold_setup_speedup"]),
            "comparison_note": group["comparison_note"].iloc[0],
        })
    return pd.DataFrame.from_records(records)


def write_implementation_comparisons(rows: pd.DataFrame, output_dir: Path, **kwargs):
    """Write data even when a counterpart was skipped or unselected."""
    comparisons = compare_implementations(rows, **kwargs)
    summary = summarize_implementation_comparisons(comparisons)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    comparisons.to_csv(destination / "implementation_comparison_rows.csv", index=False)
    summary.to_csv(destination / "implementation_comparison_summary.csv", index=False)
    return comparisons, summary
