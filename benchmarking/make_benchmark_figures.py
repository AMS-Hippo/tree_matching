#!/usr/bin/env python3
from __future__ import annotations

"""Create talk-ready figures from saved benchmark CSV/ZIP/directory outputs.

The benchmark notebooks intentionally favor complete tables and diagnostic
plots.  This script produces a smaller, more legible set of figures for talks:

* speed-accuracy frontiers with a shared legend rather than point labels;
* the generic versus encoded exact and partial-beam implementations;
* sparse exact-method comparisons;
* size-scaling curves;
* beam-width / candidate-budget tradeoff curves.

All figures are written as both PNG and PDF.  The script never reruns a matcher.
"""

import argparse
import math
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ALGORITHM_LABELS: Dict[str, str] = {
    "exact_dense": "Exact DP (generic)",
    "fast_dense": "Exact DP (encoded)",
    "sparse_closure": "Sparse closure",
    "sparse_chain": "Sparse chain (generic)",
    "fast_sparse": "Sparse chain (encoded)",
    "beam_local": "Local beam",
    "beam_local_capped": "Local beam (capped)",
    "beam_partial_score": "Partial-matching beam",
    "fast_beam_partial": "Partial beam (encoded)",
    "beam_partial_heuristic": "Partial beam (heuristic)",
}

REGIME_LABELS: Dict[str, str] = {
    "tiny_generic_dense": "Tiny generic dense",
    "dense_common_overlap": "Dense common overlap",
    "generic_sparse_jaccard": "Sparse Jaccard",
    "large_sparse_overlap": "Large sparse overlap",
    "narrow_tree_to_path": "Narrow tree to path",
    "wide_shallow_tree_pair": "Wide shallow tree pair",
    "deep_rare_anchor_stress": "Rare-anchor stress",
    "rare_anchor_oracle": "Rare-anchor oracle",
    "equality_tree_to_path_common_labels": "Equality tree to path",
    "sparse_overlap_tree_template_bank": "Sparse-overlap template bank",
    "generic_jaccard_tree_to_path": "Generic Jaccard tree to path",
}

# Fixed colors make figures comparable from one run to the next.
ALGORITHM_COLORS: Dict[str, str] = {
    "exact_dense": "#1f77b4",
    "fast_dense": "#ff7f0e",
    "sparse_closure": "#2ca02c",
    "sparse_chain": "#d62728",
    "fast_sparse": "#9467bd",
    "beam_local": "#8c564b",
    "beam_local_capped": "#e377c2",
    "beam_partial_score": "#7f7f7f",
    "fast_beam_partial": "#17becf",
    "beam_partial_heuristic": "#bcbd22",
}

ALGORITHM_MARKERS: Dict[str, str] = {
    "exact_dense": "o",
    "fast_dense": "s",
    "sparse_closure": "^",
    "sparse_chain": "D",
    "fast_sparse": "P",
    "beam_local": "X",
    "beam_local_capped": "v",
    "beam_partial_score": "*",
    "fast_beam_partial": "H",
    "beam_partial_heuristic": "h",
}


def _base_algorithm(name: str) -> str:
    """Map sweep aliases such as ``partial_B200`` to a stable visual family."""
    if name in ALGORITHM_LABELS:
        return name
    lower = str(name).lower()
    if lower.startswith("fast_partial_") or lower.startswith("fast_beam_partial"):
        return "fast_beam_partial"
    if lower.startswith("partial_") or lower.startswith("beam_partial"):
        return "beam_partial_score"
    if lower.startswith("local_r") or "cap" in lower:
        return "beam_local_capped"
    if lower.startswith("local_b") or lower.startswith("beam_local"):
        return "beam_local"
    return str(name)


def _algorithm_label(name: str) -> str:
    base = _base_algorithm(name)
    if name in ALGORITHM_LABELS:
        return ALGORITHM_LABELS[name]
    if base in {"beam_partial_score", "fast_beam_partial", "beam_local", "beam_local_capped"}:
        return str(name).replace("_", " ")
    return ALGORITHM_LABELS.get(base, str(name).replace("_", " "))


def _style(name: str) -> Tuple[str, str]:
    base = _base_algorithm(name)
    return (
        ALGORITHM_COLORS.get(base, "#333333"),
        ALGORITHM_MARKERS.get(base, "o"),
    )


def _label_regime(name: str) -> str:
    return REGIME_LABELS.get(str(name), str(name).replace("_", " "))


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return text.strip("._-") or "figure"


def _read_table(source: Optional[str], expected_name: str) -> Optional[pd.DataFrame]:
    if source is None:
        return None
    path = Path(source)
    if path.is_dir():
        candidate = path / expected_name
        if not candidate.exists():
            matches = list(path.rglob(expected_name))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {expected_name} under {path}, found {len(matches)}"
                )
            candidate = matches[0]
        return pd.read_csv(candidate)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            matches = [name for name in archive.namelist() if name.endswith(expected_name)]
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected exactly one {expected_name} inside {path}, found {len(matches)}"
                )
            with archive.open(matches[0]) as handle:
                return pd.read_csv(handle)
    raise ValueError(f"Unsupported result source: {path}")


def _save(fig: plt.Figure, outdir: Path, stem: str, manifest: List[Dict[str, str]], description: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{stem}.png"
    pdf = outdir / f"{stem}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    manifest.append({"figure": stem, "png": str(png), "pdf": str(pdf), "description": description})
    plt.close(fig)


def _legend_handles(algorithms: Iterable[str]) -> List[Line2D]:
    handles: List[Line2D] = []
    seen = set()
    for name in algorithms:
        base = _base_algorithm(str(name))
        if base in seen:
            continue
        seen.add(base)
        color, marker = _style(base)
        handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="none",
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=0.5,
                markersize=8,
                label=ALGORITHM_LABELS.get(base, base),
            )
        )
    return handles


def _frontier_scatter(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    frontier_col: Optional[str],
    x_label: str,
    y_label: str,
    title: str,
) -> None:
    for _, row in data.iterrows():
        algorithm = str(row["algorithm"])
        color, marker = _style(algorithm)
        frontier = bool(row.get(frontier_col, False)) if frontier_col else False
        ax.scatter(
            float(row[x_col]),
            float(row[y_col]),
            s=120 if frontier else 70,
            marker=marker,
            c=[color],
            edgecolors="black",
            linewidths=1.5 if frontier else 0.5,
            zorder=4 if frontier else 3,
        )
    ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.22)
    if not data.empty:
        y_min = float(pd.to_numeric(data[y_col], errors="coerce").min())
        y_max = float(pd.to_numeric(data[y_col], errors="coerce").max())
        if np.isfinite(y_min) and np.isfinite(y_max):
            if y_max - y_min < 5.0:
                y_min = max(0.0, y_max - 5.0)
            else:
                y_min = max(0.0, y_min - 3.0)
            ax.set_ylim(y_min, min(101.0, y_max + 1.0))


def plot_pairwise_frontiers(summary: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    data = summary[
        (pd.to_numeric(summary["successful_instances"], errors="coerce") > 0)
        & pd.to_numeric(summary["median_accuracy_percent"], errors="coerce").notna()
        & pd.to_numeric(summary["median_predict_seconds"], errors="coerce").gt(0)
    ].copy()
    regimes = list(summary.sort_values("regime_order")["regime"].drop_duplicates())
    if not regimes:
        return
    ncols = min(4, len(regimes))
    nrows = int(math.ceil(len(regimes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.8 * nrows), squeeze=False)
    axes_flat = list(axes.flat)
    for ax, regime in zip(axes_flat, regimes):
        sub = data[data["regime"] == regime].copy()
        if sub.empty:
            ax.axis("off")
            ax.set_title(_label_regime(regime))
            ax.text(
                0.5,
                0.5,
                "Scalability-only regime\n(no exact oracle)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#5b6573",
            )
            continue
        sub["warm_ms"] = 1000.0 * pd.to_numeric(sub["median_predict_seconds"], errors="coerce")
        _frontier_scatter(
            ax,
            sub,
            x_col="warm_ms",
            y_col="median_accuracy_percent",
            frontier_col="warm_speed_accuracy_frontier" if "warm_speed_accuracy_frontier" in sub else None,
            x_label="Warm time per pair (ms)",
            y_label="Score / optimum (%)",
            title=_label_regime(regime),
        )
    for ax in axes_flat[len(regimes):]:
        ax.axis("off")
    handles = _legend_handles(data["algorithm"].astype(str))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Pairwise speed-accuracy tradeoffs", fontsize=16)
    fig.subplots_adjust(bottom=0.14, top=0.90, wspace=0.34, hspace=0.38)
    _save(
        fig,
        outdir,
        "01_pairwise_speed_accuracy",
        manifest,
        "Warm pairwise runtime versus percent of the exact optimum; black outlines mark frontier points.",
    )


def plot_throughput_frontiers(summary: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    data = summary[
        summary["completed"].fillna(False).astype(bool)
        & pd.to_numeric(summary["score_matrix_seconds_median"], errors="coerce").gt(0)
        & pd.to_numeric(summary["matrix_total_accuracy"], errors="coerce").notna()
    ].copy()
    data["matrix_accuracy_percent"] = 100.0 * pd.to_numeric(data["matrix_total_accuracy"], errors="coerce")
    regimes = list(summary.sort_values("regime_order")["regime"].drop_duplicates())
    if not regimes:
        return
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.0 * len(regimes), 4.3), squeeze=False)
    axes_flat = list(axes.flat)
    for ax, regime in zip(axes_flat, regimes):
        sub = data[data["regime"] == regime].copy()
        sub["matrix_ms"] = 1000.0 * pd.to_numeric(sub["score_matrix_seconds_median"], errors="coerce")
        _frontier_scatter(
            ax,
            sub,
            x_col="matrix_ms",
            y_col="matrix_accuracy_percent",
            frontier_col="throughput_frontier" if "throughput_frontier" in sub else None,
            x_label="Warm score-matrix time (ms)",
            y_label="Aggregate score / optimum (%)",
            title=_label_regime(regime),
        )
    handles = _legend_handles(data["algorithm"].astype(str))
    if handles:
        fig.legend(handles=handles, loc="lower center", ncol=min(4, len(handles)), frameon=False)
    fig.suptitle("Template-bank throughput tradeoffs", fontsize=16)
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.33)
    _save(
        fig,
        outdir,
        "02_throughput_speed_accuracy",
        manifest,
        "Warm score-matrix time versus aggregate score accuracy for repeated query-template matching.",
    )


def plot_exact_generic_vs_encoded(
    pairwise: Optional[pd.DataFrame],
    throughput: Optional[pd.DataFrame],
    outdir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    panels: List[Tuple[str, pd.DataFrame, str, Mapping[str, str]]] = []
    p = (pairwise[pairwise["algorithm"].isin(["exact_dense", "fast_dense"])].copy()
         if pairwise is not None else pd.DataFrame())
    if not p.empty:
        pivot = p.pivot_table(index="regime", columns="algorithm", values="median_predict_seconds", aggfunc="first")
        if {"exact_dense", "fast_dense"}.issubset(pivot.columns):
            valid = pivot.dropna(subset=["exact_dense", "fast_dense"]).index
            p = p[p["regime"].isin(valid)]
            if not p.empty:
                panels.append(("Pairwise", p, "median_predict_seconds", REGIME_LABELS))
    if throughput is not None:
        t = throughput[throughput["algorithm"].isin(["exact_dense", "fast_dense"])].copy()
        if not t.empty:
            pivot = t.pivot_table(index="regime", columns="algorithm", values="score_matrix_seconds_median", aggfunc="first")
            if {"exact_dense", "fast_dense"}.issubset(pivot.columns):
                valid = pivot.dropna(subset=["exact_dense", "fast_dense"]).index
                t = t[t["regime"].isin(valid)]
                if not t.empty:
                    panels.append(("Throughput", t, "score_matrix_seconds_median", REGIME_LABELS))
    if not panels:
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(7.2 * len(panels), 4.6), squeeze=False)
    for ax, (panel_name, data, time_col, _) in zip(axes.flat, panels):
        regimes = list(data["regime"].drop_duplicates())
        y = np.arange(len(regimes), dtype=float)
        height = 0.34
        generic = data[data["algorithm"] == "exact_dense"].set_index("regime")
        encoded = data[data["algorithm"] == "fast_dense"].set_index("regime")
        generic_ms = np.array([1000.0 * float(generic.loc[r, time_col]) for r in regimes])
        encoded_ms = np.array([1000.0 * float(encoded.loc[r, time_col]) for r in regimes])
        ax.barh(y + height / 2, generic_ms, height=height, color=ALGORITHM_COLORS["exact_dense"], label=ALGORITHM_LABELS["exact_dense"])
        ax.barh(y - height / 2, encoded_ms, height=height, color=ALGORITHM_COLORS["fast_dense"], label=ALGORITHM_LABELS["fast_dense"])
        ax.set_xscale("log")
        ax.set_yticks(y)
        ax.set_yticklabels([_label_regime(r) for r in regimes])
        ax.invert_yaxis()
        ax.set_xlabel("Warm time (ms, log scale)")
        ax.set_title(panel_name)
        ax.grid(True, axis="x", which="both", alpha=0.22)
        for i, (g, e) in enumerate(zip(generic_ms, encoded_ms)):
            ax.text(max(g, e) * 1.05, y[i], f"{g / e:.1f}x", va="center", fontsize=10)
        ax.set_xlim(right=float(max(np.max(generic_ms), np.max(encoded_ms))) * 1.32)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle("The same exact DP: generic versus encoded implementation", fontsize=16)
    fig.subplots_adjust(top=0.84, wspace=0.38)
    _save(
        fig,
        outdir,
        "03_exact_generic_vs_encoded",
        manifest,
        "Warm runtime comparison for the generic and encoded implementations of the original exact dynamic program.",
    )


def plot_partial_generic_vs_encoded(
    pairwise: Optional[pd.DataFrame],
    throughput: Optional[pd.DataFrame],
    outdir: Path,
    manifest: List[Dict[str, str]],
) -> None:
    """Add the beam counterpart to the existing exact-DP comparison.

    These figures use ratios of summary medians. Per-instance paired speedups
    and score differences are saved separately by the benchmark backend.
    """
    for kind, frame, time_col, accuracy_col in (
        ("pairwise", pairwise, "median_predict_seconds", "median_accuracy_percent"),
        ("throughput", throughput, "score_matrix_seconds_median", "matrix_total_accuracy"),
    ):
        if frame is None or frame.empty:
            continue
        selected = frame[frame["algorithm"].isin(["beam_partial_score", "fast_beam_partial"])].copy()
        if "complete_coverage" in selected:
            selected = selected[selected["complete_coverage"] == True]
        elif "status" in selected:
            selected = selected[selected["status"] == "ok"]
        if selected.empty:
            continue
        if selected.duplicated(["regime", "algorithm"]).any():
            raise ValueError("The beam comparison requires one summary row per regime and algorithm")
        rows = []
        for regime, group in selected.groupby("regime", sort=False):
            indexed = group.set_index("algorithm")
            if not {"beam_partial_score", "fast_beam_partial"}.issubset(indexed.index):
                continue
            old, new = indexed.loc["beam_partial_score"], indexed.loc["fast_beam_partial"]
            a, b = float(old[time_col]), float(new[time_col])
            if not (np.isfinite(a) and np.isfinite(b) and min(a, b) > 0):
                continue
            scale = 100.0 if kind == "throughput" else 1.0
            rows.append({
                "regime": regime, "generic_warm_ms": 1000*a, "encoded_warm_ms": 1000*b,
                "speedup_ratio_of_medians": a/b,
                "generic_accuracy_percent": scale*float(old.get(accuracy_col, np.nan)),
                "encoded_accuracy_percent": scale*float(new.get(accuracy_col, np.nan)),
            })
        if not rows:
            continue
        table = pd.DataFrame(rows)
        outdir.mkdir(parents=True, exist_ok=True)
        table.to_csv(outdir / f"partial_implementation_{kind}_plot_data.csv", index=False)
        fig = plt.figure(figsize=(10, max(4.8, len(table)*0.65+2.2)))
        ax = fig.add_subplot(111)
        y = np.arange(len(table))
        # Slightly separated rows, not jittered data: avoid coincident markers.
        ax.plot(table["generic_warm_ms"], y+0.13, linestyle="none", marker="*", markersize=11,
                label="Partial beam (generic)")
        ax.plot(table["encoded_warm_ms"], y-0.13, linestyle="none", marker="H", markersize=9,
                label="Partial beam (encoded)")
        ax.set_yticks(y)
        ax.set_yticklabels([_label_regime(r) for r in table["regime"]])
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("Warm time per pair (ms)" if kind == "pairwise" else "Warm complete-matrix time (ms)")
        ax.set_title("Partial beam: generic versus encoded — " + kind)
        ax.grid(True, axis="x", which="major", alpha=0.25)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
        fig.subplots_adjust(left=0.28, bottom=0.25, top=0.88)
        fig.text(0.28, 0.025, "Compare accuracy as well as time; overlap uses different finite-budget proposals.", fontsize=9)
        _save(fig, outdir, f"06_partial_generic_vs_encoded_{kind}", manifest,
              "Old and fast partial beam; see implementation_comparison_rows.csv for paired score checks.")


def _plot_size_sweep(sweep: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    name = str(sweep["sweep_name"].iloc[0])
    x_name = str(sweep["sweep_x_name"].dropna().iloc[0]) if sweep["sweep_x_name"].notna().any() else "size"
    data = sweep[
        (pd.to_numeric(sweep["successful_instances"], errors="coerce") > 0)
        & pd.to_numeric(sweep["sweep_x_value"], errors="coerce").notna()
        & pd.to_numeric(sweep["median_predict_seconds"], errors="coerce").gt(0)
    ].copy()
    if data.empty:
        return
    data["x"] = pd.to_numeric(data["sweep_x_value"], errors="coerce")
    data["warm_ms"] = 1000.0 * pd.to_numeric(data["median_predict_seconds"], errors="coerce")
    data["accuracy"] = pd.to_numeric(data["median_accuracy_percent"], errors="coerce")
    show_accuracy = data["accuracy"].notna().any() and (
        data["accuracy"].min() < 99.999 or any("beam" in str(a) for a in data["algorithm"])
    )
    ncols = 2 if show_accuracy else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7.0 * ncols, 4.8), squeeze=False)
    runtime_ax = axes.flat[0]
    algorithms = list(data["algorithm"].drop_duplicates())
    for algorithm in algorithms:
        sub = data[data["algorithm"] == algorithm].sort_values("x")
        color, marker = _style(algorithm)
        runtime_ax.plot(sub["x"], sub["warm_ms"], marker=marker, color=color, linewidth=2, markersize=7, label=_algorithm_label(algorithm))
    runtime_ax.set_xscale("log")
    runtime_ax.set_yscale("log")
    runtime_ax.set_xlabel(x_name.replace("_", " "))
    runtime_ax.set_ylabel("Warm time per pair (ms)")
    runtime_ax.set_title("Runtime scaling")
    runtime_ax.grid(True, which="both", alpha=0.22)
    runtime_ax.legend(frameon=False, fontsize=9)

    if show_accuracy:
        accuracy_ax = axes.flat[1]
        for algorithm in algorithms:
            sub = data[(data["algorithm"] == algorithm) & data["accuracy"].notna()].sort_values("x")
            if sub.empty:
                continue
            color, marker = _style(algorithm)
            accuracy_ax.plot(sub["x"], sub["accuracy"], marker=marker, color=color, linewidth=2, markersize=7, label=_algorithm_label(algorithm))
        accuracy_ax.set_xscale("log")
        accuracy_ax.set_xlabel(x_name.replace("_", " "))
        accuracy_ax.set_ylabel("Score / optimum (%)")
        accuracy_ax.set_title("Accuracy scaling")
        accuracy_ax.grid(True, which="both", alpha=0.22)
        y_min = float(data["accuracy"].min()) if data["accuracy"].notna().any() else 95.0
        accuracy_ax.set_ylim(max(0.0, y_min - 3.0), 101.0)

    fig.suptitle(name.replace("_", " "), fontsize=16)
    fig.subplots_adjust(top=0.84, wspace=0.28)
    _save(
        fig,
        outdir,
        f"04_size_{_slug(name)}",
        manifest,
        f"Size sweep {name}: warm runtime and, where informative, objective-score accuracy.",
    )


def _plot_algorithm_sweep(sweep: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    name = str(sweep["sweep_name"].iloc[0])
    parameter_name = str(sweep["sweep_x_name"].dropna().iloc[0]) if sweep["sweep_x_name"].notna().any() else "parameter"
    data = sweep[
        (pd.to_numeric(sweep["successful_instances"], errors="coerce") > 0)
        & pd.to_numeric(sweep["sweep_x_value"], errors="coerce").notna()
        & pd.to_numeric(sweep["median_predict_seconds"], errors="coerce").gt(0)
    ].copy()
    if data.empty:
        return
    data["parameter"] = pd.to_numeric(data["sweep_x_value"], errors="coerce")
    data["warm_ms"] = 1000.0 * pd.to_numeric(data["median_predict_seconds"], errors="coerce")
    data["accuracy"] = pd.to_numeric(data["median_accuracy_percent"], errors="coerce")
    data = data.sort_values("parameter")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    # One sweep normally contains one algorithm family, but grouping keeps this robust.
    for family, sub in data.groupby(data["algorithm"].map(_base_algorithm), sort=False):
        color, marker = _style(family)
        label = ALGORITHM_LABELS.get(family, family)
        axes[0].plot(sub["parameter"], sub["accuracy"], marker=marker, color=color, linewidth=2, markersize=7, label=label)
        axes[1].plot(sub["parameter"], sub["warm_ms"], marker=marker, color=color, linewidth=2, markersize=7, label=label)
        axes[2].plot(sub["warm_ms"], sub["accuracy"], marker=marker, color=color, linewidth=2, markersize=7, label=label)
        for _, row in sub.iterrows():
            axes[2].annotate(
                f"{int(row['parameter'])}",
                (float(row["warm_ms"]), float(row["accuracy"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel(parameter_name.replace("_", " "))
    axes[0].set_ylabel("Score / optimum (%)")
    axes[0].set_title("Accuracy")
    axes[0].grid(True, which="both", alpha=0.22)
    axes[1].set_xscale("log", base=2)
    axes[1].set_yscale("log")
    axes[1].set_xlabel(parameter_name.replace("_", " "))
    axes[1].set_ylabel("Warm time per pair (ms)")
    axes[1].set_title("Runtime")
    axes[1].grid(True, which="both", alpha=0.22)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Warm time per pair (ms)")
    axes[2].set_ylabel("Score / optimum (%)")
    axes[2].set_title("Speed-accuracy path")
    axes[2].grid(True, which="both", alpha=0.22)
    y_min = float(data["accuracy"].min())
    for ax in (axes[0], axes[2]):
        ax.set_ylim(max(0.0, y_min - 3.0), 101.0)
    axes[0].legend(frameon=False)
    fig.suptitle(name.replace("_", " "), fontsize=16)
    fig.subplots_adjust(top=0.82, wspace=0.30)
    _save(
        fig,
        outdir,
        f"05_beam_{_slug(name)}",
        manifest,
        f"Beam-parameter sweep {name}: objective accuracy, runtime, and the resulting speed-accuracy path.",
    )


def plot_sweeps(sweeps: pd.DataFrame, outdir: Path, manifest: List[Dict[str, str]]) -> None:
    if sweeps.empty:
        return
    for name, group in sweeps.groupby("sweep_name", sort=False):
        kind = str(group["sweep_kind"].iloc[0])
        if kind == "size":
            _plot_size_sweep(group, outdir, manifest)
        elif kind == "algorithm":
            _plot_algorithm_sweep(group, outdir, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create talk-ready path-matching benchmark figures")
    parser.add_argument("--pairwise", help="Pairwise result directory, ZIP, or benchmark_summary.csv")
    parser.add_argument("--throughput", help="Throughput result directory, ZIP, or throughput_summary.csv")
    parser.add_argument("--sweeps", help="Sweep result directory, ZIP, or sweep_summary.csv")
    parser.add_argument("--outdir", required=True, help="Output directory for PNG/PDF figures")
    args = parser.parse_args()

    if not any((args.pairwise, args.throughput, args.sweeps)):
        parser.error("Provide at least one of --pairwise, --throughput, or --sweeps")

    pairwise = _read_table(args.pairwise, "benchmark_summary.csv")
    throughput = _read_table(args.throughput, "throughput_summary.csv")
    sweeps = _read_table(args.sweeps, "sweep_summary.csv")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []

    if pairwise is not None:
        plot_pairwise_frontiers(pairwise, outdir, manifest)
    if throughput is not None:
        plot_throughput_frontiers(throughput, outdir, manifest)
    plot_exact_generic_vs_encoded(pairwise, throughput, outdir, manifest)
    plot_partial_generic_vs_encoded(pairwise, throughput, outdir, manifest)
    if sweeps is not None:
        plot_sweeps(sweeps, outdir, manifest)

    pd.DataFrame(manifest).to_csv(outdir / "figure_manifest.csv", index=False)
    print(f"Wrote {len(manifest)} figures (PNG and PDF) to {outdir}")


if __name__ == "__main__":
    main()
