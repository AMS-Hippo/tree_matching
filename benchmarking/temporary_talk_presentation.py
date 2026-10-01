#!/usr/bin/env python3
"""Read-only talk figures for the path-matching benchmarks.

Requires numpy, pandas and matplotlib; does NOT import the matcher or run a
benchmark. Accepts a result directory, its summary CSV, or an archive containing
ONE matching summary. Run only after the relevant benchmark has finished.

Typical use (from the repository root)::

    python benchmarking/temporary_talk_presentation.py \
      --pairwise benchmarking/results/algorithm_ranking_large_fast_beam_v2 \
      --throughput benchmarking/results/path_match_throughput_large_fast_beam_v2 \
      --sweeps benchmarking/results/implementation_comparison_v2 \
      --outdir benchmarking/results/talk_v1

One figure per file, 16:9 canvas. No multipanel overview, text on scatter points,
logarithmic bars with arbitrary baselines, or invented/jittered measurements.

Outputs: PNG/PDF (or --formats png svg), an HTML gallery, manifest, exact input
CSV copies and hashes, and per-figure numerical tables. The existing repository
archiver can preserve these CSV/JSON tables but intentionally excludes figures.
A code commit alone does not archive ignored benchmark results.

--quality minimum (default): worst observed instance/pair objective-score ratio.
--quality typical: median instance ratio (pairwise), ratio of score-matrix totals
(throughput). These are empirical summaries, never future-performance guarantees.

The main routines are importable: pareto_mask, fastest_at_requirements,
normalise_summary, and build_gallery. Run --self-test for small offline checks.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import html
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import textwrap
from typing import Any
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator, MaxNLocator
import numpy as np
import pandas as pd

VERSION = "1.1"
# Keep curation here, not in the benchmark/notebook. At most four curves per view.
METHOD_LABELS = {
    "exact_dense": "Exact DP · generic",
    "fast_dense": "Exact DP · encoded",
    "sparse_closure": "Sparse closure",
    "sparse_chain": "Sparse chain · generic",
    "fast_sparse": "Sparse chain · encoded",
    "beam_local": "Local beam",
    "beam_local_capped": "Local beam · capped",
    "beam_partial_score": "Partial beam · generic",
    "fast_beam_partial": "Partial beam · encoded",
    "beam_partial_heuristic": "Partial beam · heuristic",
}
REGIME_TITLES = {
    "tiny_generic_dense": "Small trees, generic scoring",
    "dense_common_overlap": "Dense overlap",
    "generic_sparse_jaccard": "Sparse Jaccard",
    "large_sparse_overlap": "Large trees, sparse overlap",
    "narrow_tree_to_path": "Narrow tree to path",
    "wide_shallow_tree_pair": "Wide, shallow trees",
    "deep_rare_anchor_stress": "Rare anchors · stress test",
    "rare_anchor_oracle": "Rare anchors · exact reference",
    "rare_anchor_oracle_fast_check": "Rare anchors · implementation check",
    "equality_tree_to_path_common_labels": "Common labels · tree to path",
    "sparse_overlap_tree_template_bank": "Sparse overlap · template bank",
    "generic_jaccard_tree_to_path": "Generic Jaccard · tree to path",
    "dense_overlap_size": "Dense overlap: implementation scaling",
    "sparse_jaccard_size": "Sparse Jaccard: exact versus approximate",
    "sparse_overlap_size": "Sparse overlap: exploiting candidate sparsity",
    "narrow_tree_path_size": "Narrow tree to path: scaling",
    "rare_anchor_size": "Rare anchors: scaling",
    "partial_beam_implementation_rare_anchor": "Rare anchors: implementation scaling",
    "partial_beam_width_rare_anchor": "Rare anchors: beam width",
    "partial_expansion_rare_anchor": "Rare anchors: candidate budget",
    "local_beam_width_narrow": "Narrow tree to path: beam width",
    "local_child_cap_wide": "Wide trees: child budget",
}
CURATED_VIEWS = {
    "generic_exact_and_sparse": ["exact_dense", "sparse_closure", "sparse_chain", "beam_local"],
    "encoded_candidates": ["fast_dense", "fast_sparse", "beam_local", "fast_beam_partial"],
    "exact_implementations": ["exact_dense", "fast_dense"],
    "partial_implementations": ["beam_partial_score", "fast_beam_partial"],
}
PAIR_FAMILIES = {
    "exact_dp": ("exact_dense", "fast_dense"),
    "partial_beam": ("beam_partial_score", "fast_beam_partial"),
}
REQUIREMENTS = (0.90, 0.95, 0.99, 1.0)
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "h", "*")
FIGSIZE = (14.4, 8.1)
ABS_TOL, REL_TOL = 1e-5, 1e-6
CONTEXT = ["sweep_name", "sweep_kind", "sweep_point", "sweep_x_name", "sweep_x_value", "regime"]


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-") or "figure"


def method_label(name: str) -> str:
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    match = re.fullmatch(r"(fast_partial|partial|local)_([BER])(\d+)", name)
    if match:
        family, symbol, value = match.groups()
        family = {"fast_partial": "Encoded partial", "partial": "Partial", "local": "Local"}[family]
        param = {"B": "width", "E": "expansion", "R": "child cap"}[symbol]
        return f"{family} · {param} {value}"
    return name.replace("_", " ")


def title_of(name: str) -> str:
    return REGIME_TITLES.get(name, name.replace("_", " ").capitalize())


def family_of(name: str) -> str:
    if name.startswith("fast_partial_"):
        return "fast_beam_partial"
    if name.startswith("partial_"):
        return "beam_partial_score"
    if name.startswith("local_R"):
        return "beam_local_capped"
    if name.startswith("local_"):
        return "beam_local"
    return name


def num(frame: pd.DataFrame, *names: str, default: float = np.nan) -> pd.Series:
    for name in names:
        if name in frame:
            return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(default, index=frame.index, dtype=float)


def bools(series: pd.Series) -> pd.Series:
    # bool('False') is True; CSV strings must be parsed explicitly.
    return series.map(lambda x: x is True or str(x).strip().lower() in {"true", "1", "1.0"})


def known_false(series: pd.Series) -> pd.Series:
    return series.map(lambda x: x is False or str(x).strip().lower() in {"false", "0", "0.0"})


def normalise_summary(frame: pd.DataFrame, kind: str, timing: str = "warm") -> pd.DataFrame:
    """Keep source columns; add explicit comparable fields without pooling runs."""
    if not {"algorithm", "regime"}.issubset(frame):
        raise ValueError("Summary needs algorithm and regime columns.")
    d = frame.copy().reset_index(drop=True)
    identity = [k for k in CONTEXT if k in d] + ["algorithm"]
    if d.duplicated(identity).any():
        raise ValueError("Duplicate method/context rows in summary; select one completed run.")
    if kind == "throughput":
        if "status" not in d:
            raise ValueError("Throughput summary lacks status; completion cannot be inferred.")
        d["completed"] = d["status"].eq("ok")
        d["expected"] = 1
        d["finished"] = d["completed"].astype(int)
        d["time_s"] = num(d, "score_matrix_seconds_median" if timing == "warm" else "setup_plus_one_matrix_seconds")
        d["lo_s"] = num(d, "score_matrix_seconds_q25") if timing == "warm" else np.nan
        d["hi_s"] = num(d, "score_matrix_seconds_q75") if timing == "warm" else np.nan
        d["typical_q"] = num(d, "matrix_total_accuracy")
        d["minimum_q"] = num(d, "min_pair_accuracy")
        valid = bools(d["oracle_valid"]) if "oracle_valid" in d else pd.Series(False, index=d.index)
        if "score_consistent_with_oracle" in d:
            valid = valid & ~known_false(d["score_consistent_with_oracle"])
    else:
        if not {"total_instances", "successful_instances"}.issubset(d):
            raise ValueError("Pairwise/sweep summary lacks total/successful instance counts.")
        d["expected"] = num(d, "total_instances")
        d["finished"] = num(d, "successful_instances")
        d["completed"] = (d["expected"] > 0) & (d["finished"] == d["expected"])
        if "complete_coverage" in d:
            d["completed"] = d["completed"] & bools(d["complete_coverage"])
        d["time_s"] = num(d, "median_predict_seconds" if timing == "warm" else "median_setup_plus_warm_search_seconds")
        d["lo_s"] = num(d, "q25_predict_seconds" if timing == "warm" else "q25_setup_plus_warm_search_seconds")
        d["hi_s"] = num(d, "q75_predict_seconds" if timing == "warm" else "q75_setup_plus_warm_search_seconds")
        d["typical_q"] = num(d, "median_accuracy_score", "median_score_ratio")
        if "median_accuracy_score" not in d and "median_score_ratio" not in d:
            d["typical_q"] = num(d, "median_accuracy_percent") / 100.0
        d["minimum_q"] = num(d, "min_accuracy_score", "min_score_ratio")
        if "complete_accuracy_coverage" in d:
            valid = bools(d["complete_accuracy_coverage"])
        elif "oracle_scored_instances" in d:
            valid = num(d, "oracle_scored_instances").eq(d["expected"])
        elif "oracle_valid_instances" in d:
            valid = num(d, "oracle_valid_instances").eq(d["expected"])
        else:
            valid = pd.Series(False, index=d.index)
        if "exact_disagreement_instances" in d:
            valid = valid & num(d, "exact_disagreement_instances").fillna(0).eq(0)
    for col in ("typical_q", "minimum_q"):
        out_of_range = valid & ((d[col] < -1e-10) | (d[col] > 1 + 1e-6))
        if out_of_range.any():
            raise ValueError(f"Invalid objective-score ratio in {col}; fix benchmark validation before plotting.")
        d.loc[~valid, col] = np.nan
    if (valid & d["minimum_q"].gt(d["typical_q"] + 1e-7)).any():
        raise ValueError("Minimum reported accuracy exceeds typical accuracy; check summary semantics.")
    d["score_reference_valid"] = valid
    d["timed_complete"] = d["completed"] & np.isfinite(d["time_s"]) & d["time_s"].gt(0)
    d["method_label"] = d["algorithm"].map(method_label)
    return d


def pareto_mask(times, qualities, eligible=None) -> np.ndarray:
    """Empirical nondominance: no slower and no worse, with one strict inequality.

    Coincident methods are both retained. Missing, failed or oracle-free results
    cannot be efficient. Timing noise is not silently rounded into dominance.
    """
    t, q = np.asarray(times, float), np.asarray(qualities, float)
    if t.shape != q.shape or t.ndim != 1:
        raise ValueError("Times and qualities must be one-dimensional and equally long.")
    valid = np.isfinite(t) & (t > 0) & np.isfinite(q)
    if eligible is not None:
        valid = valid & np.asarray(eligible, bool)
    answer = valid.copy()
    for i in np.flatnonzero(valid):
        better = valid & (t <= t[i]) & (q >= q[i]) & ((t < t[i]) | (q > q[i]))
        answer[i] = not better.any()
    return answer


def fastest_at_requirements(d: pd.DataFrame, quality_col: str, requirements=REQUIREMENTS) -> pd.DataFrame:
    records = []
    for threshold in requirements:
        ok = d[d["timed_complete"] & d[quality_col].ge(threshold - 1e-12)]
        best = ok["time_s"].min() if len(ok) else np.nan
        tied = ok[ok["time_s"] == best]
        records.append({"required_q": threshold, "time_s": best,
                        "algorithm": " | ".join(tied["algorithm"].astype(str)),
                        "method_label": " / ".join(tied["method_label"].astype(str)),
                        "achieved_q": tied[quality_col].min() if len(tied) else np.nan,
                        "tie_count": len(tied)})
    return pd.DataFrame(records)


@dataclass
class Source:
    path: Path
    kind: str
    prefix: str
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    contents: dict[str, bytes] = field(default_factory=dict)
    locations: dict[str, str] = field(default_factory=dict)

    def read_all(self) -> "Source":
        summary = {"pairwise": "benchmark_summary.csv", "throughput": "throughput_summary.csv", "sweeps": "sweep_summary.csv"}[self.kind]
        rows = {"pairwise": "benchmark_rows.csv", "throughput": "throughput_rows.csv", "sweeps": "sweep_rows.csv"}[self.kind]
        wanted = [summary, rows, "implementation_comparison_summary.csv", "implementation_comparison_rows.csv"]
        if self.path.is_dir():
            found = [self.path / summary] if (self.path / summary).is_file() else list(self.path.rglob(summary))
            if len(found) != 1:
                raise ValueError(f"Expected ONE {summary} under {self.path}; found {len(found)}. Select a single run directory.")
            parent = found[0].parent
            for name in wanted:
                f = parent / name
                if f.is_file():
                    self.contents[name] = f.read_bytes(); self.locations[name] = str(f.resolve())
        elif self.path.suffix.lower() == ".zip":
            with zipfile.ZipFile(self.path) as z:
                found = [n for n in z.namelist() if Path(n).name == summary]
                if len(found) != 1:
                    raise ValueError(f"Expected ONE {summary} in {self.path}; found {len(found)}. Extract and select one run.")
                parent = PurePosixPath(found[0]).parent
                for name in wanted:
                    member = str(parent / name)
                    if member in z.namelist():
                        self.contents[name] = z.read(member); self.locations[name] = f"{self.path.resolve()}::{member}"
        elif self.path.is_file() and self.path.suffix.lower() == ".csv":
            self.contents[summary] = self.path.read_bytes(); self.locations[summary] = str(self.path.resolve())
            for name in wanted[1:]:
                f = self.path.parent / name
                if f.is_file():
                    self.contents[name] = f.read_bytes(); self.locations[name] = str(f.resolve())
        else:
            raise FileNotFoundError(f"Not a result directory, CSV or ZIP: {self.path}")
        self.tables = {n: pd.read_csv(io.BytesIO(b)) for n, b in self.contents.items()}
        self.tables["summary"] = self.tables[summary]
        self.tables["rows"] = self.tables.get(rows, pd.DataFrame())
        return self


def paired_comparisons(source: Source, timing: str) -> pd.DataFrame:
    """Use saved paired checks when available; otherwise pair raw observations.

    Throughput score totals are NEVER used to infer entry-by-entry agreement.
    The older summaries without paired raw information do not yield speedups.
    """
    saved = source.tables.get("implementation_comparison_summary.csv")
    ratio = "median_warm_speedup" if timing == "warm" else "median_setup_speedup"
    if saved is not None and ratio in saved:
        p = saved.copy()
        p["speedup"] = num(p, ratio)
        p["lo"] = num(p, "q25_warm_speedup") if timing == "warm" else np.nan
        p["hi"] = num(p, "q75_warm_speedup") if timing == "warm" else np.nan
        # Setup quartiles, when possible, are taken from the paired rows.
        raw = source.tables.get("implementation_comparison_rows.csv")
        keys = [k for k in CONTEXT if k in p] + ["family", "score_mode"]
        if timing == "setup" and raw is not None and "setup_speedup" in raw:
            q = raw[bools(raw["paired_complete"])].groupby(keys, dropna=False)["setup_speedup"].quantile([.25, .75]).unstack()
            q = q.rename(columns={.25: "_lo", .75: "_hi"}).reset_index()
            p = p.merge(q, on=keys, how="left", validate="one_to_one")
            p["lo"], p["hi"] = p.pop("_lo"), p.pop("_hi")
        p["complete"] = bools(p["complete_coverage"])
        p["n"] = num(p, "paired_complete_count")
        p["agreement"] = p.get("all_observed_scores_agree", pd.Series(np.nan, index=p.index))
        p["checks"] = num(p, "scores_checked", default=0)
        return p
    raw = source.tables["rows"]
    if raw.empty:
        return pd.DataFrame()
    keys = [k for k in CONTEXT if k in raw]
    if source.kind != "throughput":
        if "instance" not in raw:
            return pd.DataFrame()
        join_keys = keys + ["instance"]
    else:
        join_keys = keys
    if raw.duplicated(join_keys + ["algorithm"]).any():
        raise ValueError("Duplicate paired observations: select a single run, not concatenated runs.")
    time_col = ("score_matrix_seconds_median" if timing == "warm" else "setup_plus_one_matrix_seconds") if source.kind == "throughput" else ("predict_seconds_median" if timing == "warm" else "setup_plus_warm_search_seconds")
    if time_col not in raw:
        return pd.DataFrame()
    records = []
    for vals, group in raw.groupby(keys, sort=False, dropna=False):
        vals = vals if isinstance(vals, tuple) else (vals,)
        ctx = dict(zip(keys, vals))
        for family, (generic, encoded) in PAIR_FAMILIES.items():
            g, e = group[group.algorithm == generic], group[group.algorithm == encoded]
            if g.empty or e.empty:
                continue
            merged = g.merge(e, on=join_keys, suffixes=("_g", "_e"), how="outer", validate="one_to_one")
            valid = merged.status_g.eq("ok") & merged.status_e.eq("ok")
            a = num(merged, time_col + "_g"); b = num(merged, time_col + "_e")
            valid = valid & np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
            ratios = (a[valid] / b[valid])
            agreement, checks = None, 0
            if source.kind != "throughput" and {"score_g", "score_e"}.issubset(merged):
                sa, sb = num(merged, "score_g")[valid], num(merged, "score_e")[valid]
                finite = np.isfinite(sa) & np.isfinite(sb)
                checks = int(finite.sum())
                tolerance = np.maximum(ABS_TOL, REL_TOL * np.maximum(np.abs(sa[finite]), np.abs(sb[finite])))
                agreement = bool((np.abs(sa[finite] - sb[finite]) <= tolerance).all()) if checks else None
            modes = set(group.get("score_mode", pd.Series(["unknown"])).dropna().astype(str))
            if len(modes) != 1:
                raise ValueError("Inconsistent score modes inside a paired comparison.")
            mode = next(iter(modes))
            records.append({**ctx, "family": family, "score_mode": mode,
                            "speedup": ratios.median(), "lo": ratios.quantile(.25), "hi": ratios.quantile(.75),
                            "complete": bool(valid.all()), "n": int(valid.sum()),
                            "checks": checks, "agreement": agreement})
    return pd.DataFrame(records)


def display_number(x: float) -> str:
    if not np.isfinite(x): return "—"
    if abs(x) >= 1000: return f"{x:,.0f}"
    if abs(x) >= 10: return f"{x:.1f}".rstrip("0").rstrip(".")
    if abs(x) >= 1: return f"{x:.2f}".rstrip("0").rstrip(".")
    return f"{x:.3g}"


def percentage(x: float) -> str:
    return f"{100*x:.2f}" if np.isfinite(x) else "—"


def log_axis(ax, axis: str, values, observed_ticks: bool = False) -> None:
    v = np.asarray(values, float); v = v[np.isfinite(v) & (v > 0)]
    if not len(v): return
    low, high = min(v), max(v)
    if low == high: low, high = low / 1.8, high * 1.8
    lo, hi = low / 1.22, high * 1.22
    setter = ax.set_xscale if axis == "x" else ax.set_yscale
    setter("log")
    (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
    target = ax.xaxis if axis == "x" else ax.yaxis
    if observed_ticks and len(np.unique(v)) <= 8:
        ticks = np.unique(v)
    else:
        candidates = np.array([m * 10.0 ** p for p in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1) for m in (1, 2, 5)])
        candidates = candidates[(candidates >= lo) & (candidates <= hi)]
        if len(candidates) < 2: candidates = np.array([low, high])
        # Greedy separation in log coordinates: at most five comfortable labels.
        ticks = []
        gap = (math.log10(hi) - math.log10(lo)) / 5.2
        for value in candidates:
            if not ticks or math.log10(value / ticks[-1]) >= gap:
                ticks.append(value)
        ticks = np.asarray(ticks)
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda x, pos: display_number(x)))
    target.set_minor_locator(NullLocator())
    target.set_minor_formatter(NullFormatter())
    ax.grid(True, axis=axis, which="major", alpha=.16)


def canvas(title: str, subtitle: str, label: str, *, table=False):
    fig = plt.figure(figsize=FIGSIZE, dpi=120)
    fig.text(.045, .944, title, fontsize=24, fontweight="bold", va="top")
    fig.text(.045, .883, subtitle, fontsize=14, va="top")
    if label:
        fig.text(.045, .014, label, fontsize=10, alpha=.8)
    ax = fig.add_axes([.10, .31, .855, .48] if not table else [.337, .205, .318, .585])
    ax.tick_params(labelsize=13)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def table_y(fig, ax, i: int) -> float:
    return fig.transFigure.inverted().transform(ax.transData.transform((1, i)))[1]


def footer(fig, text: str) -> None:
    for i, line in enumerate(textwrap.wrap(text, width=145)[:3]):
        fig.text(.045, .10 - .024*i, line, fontsize=11, va="top")


def timing_label(kind: str, timing: str) -> str:
    if kind == "throughput":
        return "Warm matrix time (ms)" if timing == "warm" else "Preparation + one warm matrix (ms)"
    return "Warm time per pair (ms)" if timing == "warm" else "Setup + one warm search (ms)"


def quality_label(kind: str, quality: str) -> str:
    if quality == "minimum":
        return "Worst observed pair score / optimum (%)" if kind == "throughput" else "Worst observed instance score / optimum (%)"
    return "Aggregate score / aggregate optimum (%)" if kind == "throughput" else "Median instance score / optimum (%)"


@dataclass
class Gallery:
    out: Path
    label: str
    formats: list[str]
    records: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def save(self, fig, stem: str, title: str, caption: str, data: pd.DataFrame, role="alternative") -> None:
        stem = slug(stem)
        if any(x["stem"] == stem for x in self.records):
            raise ValueError(f"Duplicate output stem {stem}")
        paths = {}
        try:
            # Save a fixed 16:9 canvas. All text must fit; do not shrink via bbox tight.
            fig.canvas.draw()
            # A single wide axis normally suffices; thin pathological tick labels
            # rather than rotate/shrink them or let them overlap.
            for ax in fig.axes:
                for _ in range(5):
                    texts = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text()]
                    renderer = fig.canvas.get_renderer()
                    boxes = sorted((t.get_window_extent(renderer) for t in texts), key=lambda b: b.x0)
                    if not any(a.x1 + 12 > b.x0 for a, b in zip(boxes, boxes[1:])):
                        break
                    ticks = ax.get_xticks()
                    if len(ticks) <= 2:
                        break
                    keep = list(range(0, len(ticks), 2))
                    if keep[-1] != len(ticks)-1:
                        keep.append(len(ticks)-1)
                    # Do not leave the final two ticks almost adjacent.
                    if len(keep) > 2 and keep[-1]-keep[-2] == 1:
                        keep.pop(-2)
                    ax.xaxis.set_major_locator(FixedLocator(ticks[keep]))
                    fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            for text in fig.texts:
                box = text.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
                    raise ValueError(f"Figure text is clipped in {stem}: {text.get_text()!r}")
            for fmt in self.formats:
                name = f"{stem}.{fmt}"
                fig.savefig(self.out / name, dpi=190)
                paths[fmt] = name
        finally:
            plt.close(fig)
        data_name = f"{stem}.csv"
        data.to_csv(self.out / data_name, index=False)
        self.records.append({"stem": stem, "title": title, "caption": caption, "role": role, "data": data_name, **paths})

    def finish(self, sources: list[Source], args: dict) -> None:
        meta = {"script_version": VERSION, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "created_utc": datetime.now(timezone.utc).isoformat(),
                "arguments": args, "python": sys.version, "numpy": np.__version__,
                "pandas": pd.__version__, "matplotlib": matplotlib.__version__,
                "sources": [], "notes": self.notes, "figures": self.records}
        (self.out / "input_tables").mkdir()
        for source in sources:
            for name, content in source.contents.items():
                stored = f"input_tables/{source.prefix}__{name}"
                (self.out / stored).write_bytes(content)
                meta["sources"].append({"location": source.locations[name], "sha256": hashlib.sha256(content).hexdigest(), "copy": stored})
        (self.out / "talk_manifest.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        pd.DataFrame(self.records).to_csv(self.out / "talk_manifest.csv", index=False)
        cards = []
        for r in self.records:
            media = f'<img loading="lazy" src="{html.escape(r["png"])}" alt="{html.escape(r["title"])}">' if "png" in r else ""
            links = " · ".join(f'<a href="{html.escape(r[f])}">{f.upper()}</a>' for f in self.formats if f in r)
            cards.append(f'<section><h2>{html.escape(r["title"])}</h2><p>{html.escape(r["caption"])}</p>{media}<p>{links} · <a href="{r["data"]}">DATA</a></p></section>')
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in self.notes)
        document = '<!doctype html><meta charset="utf-8"><title>Talk figure choices</title><style>body{font:17px sans-serif;max-width:1100px;margin:35px auto;line-height:1.5;padding:0 20px}img{width:100%;height:auto}section{border-top:1px solid;padding:24px 0}h1,h2{line-height:1.2}</style>'
        document += '<h1>Talk figure choices</h1><p>Saved measurements only. Frontier membership is empirical; no confidence or optimality guarantee is implied. Each figure has its own numerical table.</p>'
        document += f'<ul>{notes}</ul>' + ''.join(cards)
        (self.out / "index.html").write_text(document, encoding="utf-8")


def plot_ledger(gallery: Gallery, d: pd.DataFrame, name: str, stem: str, kind: str, timing: str, quality: str) -> None:
    qcol = "minimum_q" if quality == "minimum" else "typical_q"
    d = d.copy()
    d["frontier"] = pareto_mask(d.time_s, d[qcol], d.timed_complete)
    d = d.sort_values("time_s", na_position="last", kind="stable").reset_index(drop=True)
    # Preserve all methods, but split long tables instead of shrinking the type.
    for start in range(0, len(d), 9):
        sub = d.iloc[start:start+9].reset_index(drop=True)
        page = f" · {start//9+1}" if len(d) > 9 else ""
        title = title_of(name) + page
        fig, ax = canvas(title, "Speed and score retained (% of the exact optimum)", gallery.label, table=True)
        ax.set_ylim(len(sub)-.5, -.6); ax.set_yticks([]); ax.spines["left"].set_visible(False)
        vals = sub.time_s.dropna().to_numpy() * 1000
        if len(vals): log_axis(ax, "x", vals)
        else: ax.set_xlim(1, 10); ax.set_xticks([])
        ax.set_xlabel(timing_label(kind, timing), fontsize=14, labelpad=12)
        # Header positions deliberately separated; row labels never sit over points.
        for x, txt in ((.045, "METHOD"), (.675, "TIME (ms)"), (.785, "TOTAL %" if kind == "throughput" else "MEDIAN %"), (.902, "WORST %")):
            fig.text(x, .818, txt, fontsize=12, fontweight="bold")
        for i, row in sub.iterrows():
            y = table_y(fig, ax, i)
            marker = "◆" if row.frontier else ""
            fig.text(.024, y, marker, fontsize=14, va="center")
            label = row.method_label
            if not row.completed:
                label += f" [{int(row.finished)}/{int(row.expected)}]"
            fig.text(.045, y, label, fontsize=14, va="center")
            t = float(row.time_s) * 1000
            if np.isfinite(t) and t > 0:
                lo, hi = row.lo_s * 1000, row.hi_s * 1000
                err = np.array([[max(0., t-lo)], [max(0., hi-t)]]) if np.isfinite([lo, hi]).all() else None
                ax.errorbar(t, i, xerr=err, marker="D" if row.frontier else "o", fillstyle="full" if row.frontier else "none", markersize=7, capsize=3, linestyle="none", alpha=1.0 if row.completed else .4)
            fig.text(.675, y, display_number(t), fontsize=14, va="center")
            fig.text(.80, y, percentage(row.typical_q), fontsize=14, va="center")
            fig.text(.916, y, percentage(row.minimum_q), fontsize=14, va="center")
        interval = "Whiskers: middle 50% of repeated matrix timings (not corpus uncertainty)." if kind == "throughput" else "Whiskers: middle 50% of instance-median times (not a confidence interval)."
        if timing == "setup" and kind == "throughput": interval = "No uncertainty bars are inferred for summed setup and matrix time."
        criterion = "minimum observed score ratio" if quality == "minimum" else ("aggregate score ratio" if kind == "throughput" else "median score ratio")
        footer(fig, f"◆ Empirical frontier using {criterion}; complete runs only. Brackets show incomplete coverage. — means unavailable, not zero. {interval}")
        gallery.save(fig, f"{stem}_ledger_{start//9+1}", title, f"Readable alternative to the scatter plot. Frontier compares {criterion} and {timing} time; all methods stay visible.", sub, "main")


def plot_requirements(gallery: Gallery, d: pd.DataFrame, name: str, stem: str, kind: str, timing: str, quality: str) -> None:
    qcol = "minimum_q" if quality == "minimum" else "typical_q"
    if not (d.timed_complete & d[qcol].notna()).any():
        gallery.notes.append(f"{stem}: no valid accuracy reference; no requirement/frontier claims.")
        return
    choices = fastest_at_requirements(d, qcol)
    fig, ax = canvas("How much speed can we keep?", title_of(name) + " · fastest tested method meeting each score requirement", gallery.label, table=True)
    ax.set_ylim(3.7, -.7); ax.set_yticks([]); ax.spines["left"].set_visible(False)
    ax.set_xlabel(timing_label(kind, timing), fontsize=14, labelpad=12)
    ts = choices.time_s.dropna().to_numpy()*1000
    log_axis(ax, "x", ts) if len(ts) else ax.set_xticks([])
    fig.text(.045, .815, "REQUIRED SCORE", fontsize=13, fontweight="bold")
    fig.text(.68, .815, "FASTEST ELIGIBLE METHOD", fontsize=13, fontweight="bold")
    for i, row in choices.iterrows():
        y = table_y(fig, ax, i)
        label = f"≥ {row.required_q*100:.0f}%" if row.required_q < 1 else "100% observed"
        fig.text(.07, y, label, fontsize=20, va="center")
        if np.isfinite(row.time_s):
            ax.plot(row.time_s*1000, i, marker="o", markersize=10, linestyle="none")
            name_lines = ([f"{int(row.tie_count)} methods tied (see data)"] if row.tie_count > 1
                          else textwrap.wrap(row.method_label, 29))
            fig.text(.68, y+.017, name_lines[0], fontsize=15, va="center")
            detail = f"{display_number(row.time_s*1000)} ms · score {100*row.achieved_q:.2f}%"
            if len(name_lines)>1: detail = name_lines[1] + "\n" + detail
            fig.text(.68, y-.025, detail, fontsize=12, va="center")
        else:
            fig.text(.68, y, "No eligible completed method", fontsize=14, va="center")
    footer(fig, quality_label(kind, quality) + ". This is a choice among the tested configurations, not an interpolated solver or a guarantee on future data. All intended jobs must finish.")
    gallery.save(fig, stem+"_requirements", title_of(name)+" · accuracy requirements", "The lower envelope of the tested speed–quality choices, without overlapping points. Requirement levels are explicit.", choices, "main")


def agree_text(row) -> str:
    if int(row.get("checks", 0) or 0) == 0 or pd.isna(row.get("agreement")):
        return "score agreement unavailable"
    if bools(pd.Series([row.agreement])).iloc[0]:
        return "observed scores agree"
    return "observed scores DIFFER"


def plot_implementation_ratios(gallery: Gallery, p: pd.DataFrame, prefix: str, timing: str, kind: str) -> None:
    if p.empty:
        gallery.notes.append(f"{prefix}: paired raw/comparison data unavailable; no speedup ratios invented from marginal medians.")
        return
    if "sweep_name" in p: return  # handled with x-varying head-to-head plots
    for family, sub in p.groupby("family", sort=False):
        sub = sub[sub.complete & num(sub, "speedup").gt(0)].copy().reset_index(drop=True)
        if sub.empty: continue
        for start in range(0, len(sub), 5):
            page = sub.iloc[start:start+5].reset_index(drop=True)
            family_title = "Exact DP" if family == "exact_dp" else "Partial beam"
            fig, ax = canvas(f"{family_title}: generic versus encoded", "Paired speedup · generic time / encoded time", gallery.label, table=True)
            ax.set_position([.325, .22, .36, .56]); ax.set_ylim(len(page)-.5, -.65); ax.set_yticks([]); ax.spines["left"].set_visible(False)
            all_x = np.r_[page.speedup, page.lo.dropna(), page.hi.dropna(), 1.0]
            log_axis(ax, "x", all_x)
            ax.axvline(1, linestyle="--", linewidth=1, alpha=.45)
            ax.set_xlabel("Speedup (×), higher is faster", fontsize=15, labelpad=12)
            for i, row in page.iterrows():
                y = table_y(fig, ax, i)
                fig.text(.045, y, "\n".join(textwrap.wrap(title_of(row.regime), 25)), fontsize=15, va="center")
                err = np.array([[max(0.,row.speedup-row.lo)], [max(0.,row.hi-row.speedup)]]) if np.isfinite([row.lo,row.hi]).all() else None
                ax.errorbar(row.speedup, i, xerr=err, marker="o", markersize=9, capsize=4, linestyle="none")
                fig.text(.715, y+.026, f"{row.speedup:.1f}×", fontsize=23, fontweight="bold", va="center")
                fig.text(.715, y-.011, agree_text(row), fontsize=12, va="center")
                extra = "overlap: different proposals" if family == "partial_beam" and row.get("score_mode") == "overlap" else f"{int(row.n)} paired observation(s)"
                fig.text(.715, y-.043, extra, fontsize=11, va="center")
            footer(fig, f"{timing.capitalize()} timing. Dots are medians of paired ratios; whiskers are their middle 50%, not confidence intervals. Beam agreement does not establish optimality. Missing/incomplete pairs are omitted from this speedup chart, not from the ledger.")
            gallery.save(fig, f"{prefix}_{family}_paired_speedup_{start//5+1}", family_title+" · paired implementation speedup", "No ratio of independently aggregated medians. Overlap beam results are marked as different proposal rules.", page, "main")


def group_views(methods: list[str]) -> list[tuple[str, list[str]]]:
    methods = list(dict.fromkeys(methods))
    views = []
    if len(methods) <= 4:
        views.append(("all", methods))
    else:
        for name in ("generic_exact_and_sparse", "encoded_candidates"):
            chosen = [m for m in CURATED_VIEWS[name] if m in methods]
            if len(chosen) >= 2: views.append((name, chosen))
    for name in ("exact_implementations", "partial_implementations"):
        chosen = CURATED_VIEWS[name]
        if all(m in methods for m in chosen) and not any(set(v) == set(chosen) for _,v in views):
            views.append((name, chosen))
    covered = {m for _, v in views for m in v}
    extra = [m for m in methods if m not in covered]
    for i in range(0, len(extra), 4): views.append((f"additional_{i//4+1}", extra[i:i+4]))
    return views


def sweep_context(source: Source, name: str) -> str:
    raw = source.tables["rows"]
    if raw.empty or "sweep_name" not in raw: return ""
    d = raw[raw.sweep_name == name]
    changes = []
    for alternatives, label in ((("n_h",), "second-tree size"),
                                (("tree_g_depth", "depth_g"), "first-tree depth"),
                                (("tree_h_depth", "depth_h"), "second-tree depth"),
                                (("alphabet_size",), "alphabet size")):
        c = next((c for c in alternatives if c in d), None)
        if c is not None and d[c].nunique() > 1: changes.append(label)
    return "Also varies: " + ", ".join(changes) + "." if changes else ""


def curve_figure(gallery: Gallery, groups: list[tuple[str,pd.DataFrame]], title: str, subtitle: str, stem: str, x_label: str, y_col: str, y_label: str, *, log_y=True, bands=True, footer_text="", quality=False, reference=None, loss=False) -> None:
    fig, ax = canvas(title, subtitle, gallery.label)
    seen_x, seen_y, data_parts = [], [], []
    labels, handles = [], []
    union_x = sorted({float(v) for _, frame in groups for v in num(frame, "sweep_x_value").dropna() if v > 0})
    original_groups = list(groups)
    memberships = {name: [(name, frame)] for name, frame in groups}
    if quality:
        # Merge exactly coincident score curves instead of hiding one behind
        # another. Their original rows and membership remain in the figure CSV.
        collapsed, signatures = [], []
        for name, frame in groups:
            temp = frame.set_index("sweep_x_value").reindex(union_x)
            sig = num(temp, y_col).where(bools(temp.timed_complete) & bools(temp.score_reference_valid)).to_numpy()
            duplicate = next((i for i, old in enumerate(signatures) if np.array_equal(sig, old, equal_nan=True)), None)
            if duplicate is None:
                collapsed.append((name, frame)); signatures.append(sig)
            else:
                representative = collapsed[duplicate][0]
                memberships[representative].append((name, frame))
        groups = collapsed
        if len(groups) < len(original_groups):
            footer_text += " Exactly coincident score curves are combined; method membership is saved in the CSV."
    for j, (name, df) in enumerate(groups):
        df = df.sort_values("sweep_x_value").copy()
        df["sweep_x_value"] = num(df, "sweep_x_value")
        if df.sweep_x_value.duplicated().any(): raise ValueError(f"Duplicated x points for {name}; do not combine independent runs without grouping.")
        # Reindex onto all observed x locations to avoid connecting across an
        # absent method/size cell. These inserted cells never become data points.
        df = df.set_index("sweep_x_value").reindex(union_x).reset_index()
        x, y = num(df, "sweep_x_value").to_numpy(), num(df,y_col).to_numpy()
        good = bools(df.timed_complete).to_numpy(dtype=bool, copy=True) if "timed_complete" in df else np.ones(len(df), bool)
        if quality:
            good = good & bools(df.score_reference_valid).to_numpy(dtype=bool, copy=False)
        y = np.where(good & np.isfinite(y) & ((y>0) if log_y else True), y, np.nan)
        y = y * (100 if quality else 1000 if y_col=="time_s" else 1)
        if loss: y = 100.0 - y
        if not np.isfinite(y).any(): continue
        err = None
        if bands and y_col in {"time_s", "speedup"}:
            lo = num(df,"lo_s" if y_col=="time_s" else "lo").to_numpy() * (1000 if y_col=="time_s" else 1)
            hi = num(df,"hi_s" if y_col=="time_s" else "hi").to_numpy() * (1000 if y_col=="time_s" else 1)
            lo = np.where(good & np.isfinite(y),lo,np.nan); hi=np.where(good & np.isfinite(y),hi,np.nan)
            err = np.vstack([np.maximum(0,y-lo),np.maximum(0,hi-y)])
        artist = ax.errorbar(x, y, yerr=err, marker=MARKERS[j%len(MARKERS)], markersize=8,
                            linewidth=2.5, capsize=4, label=method_label(name))
        line = artist.lines[0]
        members = memberships.get(name, [(name, df)])
        names = {n for n, _ in members}
        display_label = method_label(name)
        if len(members)>1:
            if names == set(PAIR_FAMILIES["exact_dp"]): display_label = "Both exact-DP implementations"
            elif names == set(PAIR_FAMILIES["partial_beam"]): display_label = "Both partial-beam implementations"
            else: display_label = f"Identical scores: {len(members)} methods (see data)"
        labels.append(display_label); handles.append(line)
        seen_x.extend(x.tolist()); seen_y.extend(y[np.isfinite(y)].tolist())
        for original_name, original_frame in memberships.get(name, [(name, df)]):
            original_frame = original_frame.set_index("sweep_x_value").reindex(union_x).reset_index()
            original_frame["plotted_value"] = y
            original_frame["view_series"] = original_name
            original_frame["displayed_series"] = display_label
            data_parts.append(original_frame)
    if not handles:
        plt.close(fig); gallery.notes.append(stem+": no complete finite curve data."); return
    log_axis(ax,"x",seen_x,observed_ticks=True)
    if log_y: log_axis(ax,"y",seen_y)
    else:
        # Accuracy is never artificially jittered. A separate CSV records ties.
        upper = max(0.05, max(seen_y)*1.18) if loss else 102
        ax.set_ylim(0,upper); ax.yaxis.set_major_locator(MaxNLocator(6)); ax.grid(True,axis="y",alpha=.16)
    if reference is not None: ax.axhline(reference,linewidth=1,linestyle="--",alpha=.4)
    ax.set_xlabel(x_label,fontsize=16,labelpad=10); ax.set_ylabel(y_label,fontsize=16,labelpad=10)
    fig.legend(handles,labels,loc="lower center",bbox_to_anchor=(.53,.135),ncol=2 if len(handles)>1 else 1,frameon=False,fontsize=13)
    footer(fig,footer_text)
    gallery.save(fig,stem,title+" · "+subtitle,footer_text,pd.concat(data_parts,ignore_index=True),"main" if len(groups)<=3 else "alternative")


def plot_sweeps(gallery: Gallery, source: Source, d: pd.DataFrame, timing: str, quality: str, selected: set[str] | None) -> None:
    required={"sweep_name","sweep_kind","sweep_x_value"}
    if not required.issubset(d): raise ValueError(f"Sweep summary needs {sorted(required)}")
    pairs = paired_comparisons(source,timing)
    for name, part in d.groupby("sweep_name",sort=False):
        if selected is not None and name not in selected: continue
        stem = source.prefix+"_"+slug(name)
        kind = str(part.sweep_kind.iloc[0])
        x_label = str(part.sweep_x_name.dropna().iloc[0]).replace("_"," ") if "sweep_x_name" in part and len(part.sweep_x_name.dropna()) else "parameter value"
        note = sweep_context(source,name)
        qcol = "minimum_q" if quality=="minimum" else "typical_q"
        if kind == "algorithm":
            plot_ledger(gallery,part,name,stem,"pairwise",timing,quality)
            plot_requirements(gallery,part,name,stem,"pairwise",timing,quality)
            varying = part[num(part,"sweep_x_value").notna()].copy()
            groups = [(fam, sub) for fam, sub in varying.groupby(varying.algorithm.map(family_of),sort=False)]
            # An exact reference without a swept x value is not linked as a curve.
            curve_figure(gallery,groups,title_of(name),"Budget versus runtime",stem+"_runtime",x_label,"time_s",timing_label("pairwise",timing),footer_text="Only the labelled parameter varies within this sweep. Whiskers: interquartile spread across instances. Missing or incomplete points break the line. "+note)
            curve_figure(gallery,groups,title_of(name),"Budget versus objective score",stem+"_accuracy",x_label,qcol,quality_label("pairwise",quality),log_y=False,bands=False,quality=True,footer_text="Accuracy uses an exact reference, not agreement between approximate solvers. Read this with the separate runtime figure. No interpolated budget has been measured. "+note)
        else:
            methods=part.algorithm.astype(str).drop_duplicates().tolist()
            for view, names in group_views(methods):
                groups=[(m,part[part.algorithm==m]) for m in names]
                footer_text="Whiskers: middle 50% of instance-median times, not confidence intervals. Missing or incomplete points break lines. "+note
                curve_figure(gallery,groups,title_of(name),"Runtime · "+view.replace("_"," "),stem+"_"+view+"_runtime",x_label,"time_s",timing_label("pairwise",timing),footer_text=footer_text)
                finite_q=part[part.algorithm.isin(names)&part.completed][qcol].dropna()
                if len(finite_q) and (np.abs(finite_q-1)>1e-9).any():
                    curve_figure(gallery,groups,title_of(name),"Objective score · "+view.replace("_"," "),stem+"_"+view+"_accuracy",x_label,qcol,quality_label("pairwise",quality),log_y=False,bands=False,quality=True,footer_text="Independent accuracy view; points are not jittered to separate identical scores. No exact oracle means no accuracy point. "+note)
                    if finite_q.min() >= .90:
                        curve_figure(gallery,groups,title_of(name),"Near-optimum detail · "+view.replace("_"," "),stem+"_"+view+"_score_loss",x_label,qcol,"Loss below optimum (percentage points)",log_y=False,bands=False,quality=True,loss=True,footer_text="An alternative to the full 0–100% score view: zero loss means the exact reference was matched. Lower is better. "+note)
                elif not len(finite_q): gallery.notes.append(stem+"_"+view+": runtime only; no exact-oracle accuracy.")
                else: gallery.notes.append(stem+"_"+view+": all available accuracy values are 100%; flat duplicate accuracy figure omitted.")
            if not pairs.empty and "sweep_name" in pairs:
                for fam,p in pairs[pairs.sweep_name==name].groupby("family",sort=False):
                    p=p.copy(); p["timed_complete"]=p.complete; p["time_s"]=np.nan
                    # speedup is unitless; CI not invented when per-point paired data absent
                    curve_figure(gallery,[("Paired implementation speedup",p)],title_of(name),"Generic / encoded · "+fam.replace("_"," "),stem+"_"+fam+"_speedup",x_label,"speedup","Paired speedup (×), higher is faster",bands=True,reference=1.,footer_text="Median of paired per-instance ratios; whiskers show their middle 50%. This is implementation speedup, not an optimality claim. Source table includes score-agreement checks. "+note)


def plot_amortization(gallery: Gallery, d: pd.DataFrame, prefix: str, selected: set[str] | None) -> None:
    """Accounting projection for repeating THE SAME PREPARED score-matrix workload."""
    if not {"one_time_preparation_seconds","score_matrix_seconds_median"}.issubset(d): return
    for regime, part in d.groupby("regime",sort=False):
        if selected is not None and regime not in selected: continue
        ok=part[part.timed_complete & num(part,"one_time_preparation_seconds").ge(0)]
        # Do not promote an inaccurate method merely because its setup is cheap.
        ok=ok[ok.minimum_q.ge(.99)]
        prefs=["fast_dense","sparse_chain","fast_sparse","fast_beam_partial"]
        chosen=[m for m in prefs if m in set(ok.algorithm)][:4]
        if len(chosen)<2: continue
        rows=[]
        for m in chosen:
            row=ok[ok.algorithm==m].iloc[0]
            for repeats in (1,2,5,10,20,50,100):
                rows.append({"algorithm":m,"sweep_x_value":repeats,"projected_seconds":row.one_time_preparation_seconds + repeats*row.score_matrix_seconds_median,"setup_seconds":row.one_time_preparation_seconds,"matrix_seconds":row.score_matrix_seconds_median,"minimum_q":row.minimum_q,"timed_complete":True})
        p=pd.DataFrame(rows)
        curve_figure(gallery,[(m,p[p.algorithm==m]) for m in chosen],title_of(regime),"When does preparation pay off? · accounting projection",prefix+"_"+slug(regime)+"_amortization","Repetitions of the SAME prepared workload","projected_seconds","Projected total time (s)",bands=False,footer_text="Projection: one measured setup + R × measured warm matrix time. Not measured batch scaling or new-query arrival cost; excludes JIT/process startup. Shown methods met 99% worst-pair accuracy.")


def build_gallery(sources: list[Source], out: Path, *, quality="minimum", timing="warm", formats=None, label="", regimes=None, sweep_names=None, amortization=False, args=None) -> Gallery:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {out}. Choose a new name; existing figures are not overwritten.")
    # No writes under source data, even accidentally.
    resolved=out.resolve()
    for source in sources:
        if source.path.is_dir() and (resolved==source.path.resolve() or source.path.resolve() in resolved.parents):
            raise ValueError("Use a separate figure directory, not an input run or its child.")
    out.mkdir(parents=True,exist_ok=True)
    gallery=Gallery(out,label,list(formats or ["png","pdf"]))
    for source in sources:
        raw=source.tables["summary"]
        d=normalise_summary(raw,source.kind,timing)
        if source.kind=="sweeps":
            plot_sweeps(gallery,source,d,timing,quality,sweep_names)
        else:
            for name,part in d.groupby("regime",sort=False):
                if regimes is not None and name not in regimes: continue
                stem=source.prefix+"_"+slug(name)
                plot_ledger(gallery,part,name,stem,source.kind,timing,quality)
                plot_requirements(gallery,part,name,stem,source.kind,timing,quality)
            paired=paired_comparisons(source,timing)
            if regimes is not None and not paired.empty: paired=paired[paired.regime.isin(regimes)]
            plot_implementation_ratios(gallery,paired,source.prefix,timing,source.kind)
            if amortization and source.kind=="throughput": plot_amortization(gallery,d,source.prefix,regimes)
    if not gallery.records: raise ValueError("No figures selected; check names and completed input tables.")
    gallery.finish(sources,args or {})
    return gallery


def self_test():
    import tempfile
    # Correct dominance, coincident points, missing qualities and completion.
    assert pareto_mask([1,2,3],[.8,.9,.85]).tolist()==[True,True,False]
    assert pareto_mask([1,1,2],[1,1,1]).tolist()==[True,True,False]
    assert pareto_mask([.1,1,2],[1,1,np.nan],[False,True,True]).tolist()==[False,True,False]
    assert bools(pd.Series(["False","True",np.nan,False,True])).tolist()==[False,True,False,False,True]
    sample=pd.DataFrame({"regime":["case"]*4,"algorithm":["a","b","c","d"],"total_instances":[2]*4,"successful_instances":[2,2,1,2],"complete_accuracy_coverage":[True,True,False,False],"median_predict_seconds":[1,2,.01,.001],"median_accuracy_score":[.95,1,1,np.nan],"min_accuracy_score":[.9,.99,1,np.nan]})
    d=normalise_summary(sample,"pairwise")
    choices=fastest_at_requirements(d,"minimum_q")
    assert choices.algorithm.tolist()==["a","b","b",""]
    assert pd.isna(d.iloc[2].minimum_q)
    invalid=sample.copy(); invalid["exact_disagreement_instances"]=1
    assert normalise_summary(invalid,"pairwise").minimum_q.isna().all()
    # Saved False strings must not be treated as success in throughput.
    th=pd.DataFrame({"regime":["r"],"algorithm":["a"],"status":["ok"],"oracle_valid":["False"],"score_matrix_seconds_median":[1],"matrix_total_accuracy":[1],"min_pair_accuracy":[1]})
    assert normalise_summary(th,"throughput").minimum_q.isna().all()
    # Matched median differs from ratio of marginal medians: [2,2,100] -> 2.
    raw=pd.DataFrame([dict(regime="r",instance=i,algorithm=a,status="ok",predict_seconds_median=t,score=5,score_mode="equality") for i,(g,e) in enumerate([(2,1),(4,2),(100,1)]) for a,t in [("exact_dense",g),("fast_dense",e)]])
    source=Source(Path("unused"),"pairwise","check",tables={"rows":raw})
    p=paired_comparisons(source,"warm"); assert p.speedup.iloc[0]==2 and p.agreement.iloc[0]
    # A missing counterpart is incomplete, not silently a smaller successful sample.
    source.tables["rows"]=raw.drop(raw.index[-1]); assert not paired_comparisons(source,"warm").complete.iloc[0]
    # Unsupported/ambiguous sources and duplicate identities are explicit errors.
    source.tables["rows"]=pd.concat([raw,raw.iloc[:1]])
    try: paired_comparisons(source,"warm")
    except ValueError: pass
    else: raise AssertionError("duplicate raw rows were accepted")
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); sample.to_csv(root/"benchmark_summary.csv",index=False)
        src=Source(root,"pairwise","test").read_all()
        before=(root/"benchmark_summary.csv").read_bytes()
        out=root.parent/(root.name+"_figures")
        try:
            g=build_gallery([src],out,formats=["png"])
            assert (out/"index.html").exists() and len(g.records)==2
            assert (root/"benchmark_summary.csv").read_bytes()==before
            try: build_gallery([src],out,formats=["png"])
            except FileExistsError: pass
            else: raise AssertionError("nonempty output overwritten")
        finally:
            import shutil
            if out.exists(): shutil.rmtree(out)
    # Regression: pandas Copy-on-Write may expose read-only arrays from
    # Series.to_numpy(); sweep plotting must never modify those arrays in place.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "sweep"
        root.mkdir()
        sweep = pd.DataFrame({
            "sweep_name": ["demo", "demo", "demo", "demo"],
            "sweep_kind": ["size"] * 4,
            "sweep_point": [0, 0, 1, 1],
            "sweep_x_name": ["n_g"] * 4,
            "sweep_x_value": [100, 100, 200, 200],
            "regime": ["demo"] * 4,
            "algorithm": ["exact_dense", "beam_local", "exact_dense", "beam_local"],
            "total_instances": [2] * 4,
            "successful_instances": [2] * 4,
            "complete_coverage": [True] * 4,
            "complete_accuracy_coverage": [True] * 4,
            "median_predict_seconds": [0.01, 0.005, 0.04, 0.008],
            "q25_predict_seconds": [0.009, 0.004, 0.038, 0.007],
            "q75_predict_seconds": [0.011, 0.006, 0.042, 0.009],
            "median_accuracy_score": [1.0, 0.95, 1.0, 0.90],
            "min_accuracy_score": [1.0, 0.90, 1.0, 0.85],
            "exact_disagreement_instances": [0] * 4,
        })
        sweep.to_csv(root / "sweep_summary.csv", index=False)
        pd.DataFrame({"sweep_name": ["demo"], "n_h": [100]}).to_csv(root / "sweep_rows.csv", index=False)
        src = Source(root, "sweeps", "sweeps1").read_all()
        out = Path(tmp) / "figures"
        g = build_gallery([src], out, formats=["png"])
        assert g.records and (out / "index.html").exists()

    print("Self-test passed: dominance/ties, missing oracles, failures, CSV booleans, paired ratios, duplicates, read-only inputs, sweep rendering and overwrite protection.")


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairwise",type=Path)
    p.add_argument("--throughput",type=Path)
    p.add_argument("--sweeps",type=Path,action="append",default=[],help="May be repeated; separate runs are never pooled")
    p.add_argument("--outdir",type=Path)
    p.add_argument("--regimes",nargs="+",help="Optional subset of pairwise/throughput regime names")
    p.add_argument("--sweep-names",nargs="+",help="Optional subset of saved sweep names")
    p.add_argument("--quality",choices=["minimum","typical"],default="minimum")
    p.add_argument("--time",choices=["warm","setup"],default="warm",dest="timing")
    p.add_argument("--formats",nargs="+",choices=["png","pdf","svg"],default=["png","pdf"])
    p.add_argument("--label",default="",help="Optional run/hardware label printed on every figure")
    p.add_argument("--amortization",action="store_true",help="Extra explicitly projected setup-amortization figures")
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test: self_test(); return
    if a.outdir is None or not (a.pairwise or a.throughput or a.sweeps):
        p.error("Provide --outdir and at least one of --pairwise/--throughput/--sweeps")
    try:
        sources=[]
        for kind,path in (("pairwise",a.pairwise),("throughput",a.throughput)):
            if path: sources.append(Source(path,kind,kind).read_all())
        for i,path in enumerate(a.sweeps): sources.append(Source(path,"sweeps",f"sweeps{i+1}").read_all())
        args={k:(str(v) if isinstance(v,Path) else [str(x) for x in v] if isinstance(v,list) else v) for k,v in vars(a).items()}
        g=build_gallery(sources,a.outdir,quality=a.quality,timing=a.timing,formats=a.formats,label=a.label,regimes=set(a.regimes) if a.regimes else None,sweep_names=set(a.sweep_names) if a.sweep_names else None,amortization=a.amortization,args=args)
        print(f"Wrote {len(g.records)} figure choices to {a.outdir}")
        print(f"Open {a.outdir/'index.html'} to choose figures. No benchmarks were run.")
        for note in g.notes: print("NOTE:",note)
    except (OSError,ValueError,KeyError) as exc:
        p.exit(2,f"Input/plot error: {exc}\n")


if __name__=="__main__": main()
