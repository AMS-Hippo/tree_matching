from __future__ import annotations

"""Orchestration for size and beam-parameter benchmark sweeps.

The ordinary pairwise benchmark runner deliberately accepts one fixed matrix of
regimes and algorithms.  This module adds a thin, reproducible layer for the two
sweep types used in talks and development:

* size sweeps: clone one named regime at several tree sizes;
* algorithm-parameter sweeps: run aliased algorithm presets at one fixed regime.

Every point is still evaluated by :func:`run_benchmark_config`, so correctness
checks, process isolation, timeouts, diagnostics, and exact-oracle handling are
identical to the main pairwise notebook.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import json
import re
import warnings

import pandas as pd

from .implementation_comparison import write_implementation_comparisons

from .path_match_benchmark import run_benchmark_config
from .shared import ensure_dir, load_config


ConfigInput = Union[str, Path, Mapping[str, Any]]


def _safe_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    text = text.strip("._-")
    if not text:
        raise ValueError(f"Could not form a safe file name from {value!r}")
    return text


def _resolve_relative_path(path: Union[str, Path], *, config_path: Optional[Path]) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    if config_path is not None:
        relative_to_config = config_path.parent / candidate
        if relative_to_config.exists():
            return relative_to_config
        relative_to_repo = config_path.parent.parent / candidate
        if relative_to_repo.exists():
            return relative_to_repo
    return candidate


def _load_sweep_spec(config_or_path: ConfigInput) -> Tuple[Dict[str, Any], Optional[Path]]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path), None
    path = Path(config_or_path).resolve()
    return load_config(path), path


def _select_base_regime(base_config: Mapping[str, Any], name: str) -> Dict[str, Any]:
    matches = [deepcopy(item) for item in base_config.get("regimes", []) if item.get("name") == name]
    if len(matches) != 1:
        available = [item.get("name") for item in base_config.get("regimes", [])]
        raise ValueError(
            f"Expected one base regime named {name!r}, found {len(matches)}; "
            f"available regimes are {available}"
        )
    return matches[0]


def _apply_config_overrides(config: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> None:
    for key, value in overrides.items():
        config[str(key)] = deepcopy(value)


def _annotate(
    frame: pd.DataFrame,
    *,
    sweep_name: str,
    sweep_kind: str,
    base_regime: str,
    point_label: str,
    x_name: Optional[str],
    x_value: Any,
    parameter_values: Optional[Mapping[str, Any]] = None,
) -> pd.DataFrame:
    out = frame.copy()
    out.insert(0, "sweep_name", sweep_name)
    out.insert(1, "sweep_kind", sweep_kind)
    out.insert(2, "base_regime", base_regime)
    out.insert(3, "sweep_point", point_label)
    out.insert(4, "sweep_x_name", x_name)
    if parameter_values is None:
        out.insert(5, "sweep_x_value", x_value)
    else:
        values = out["algorithm"].map(dict(parameter_values))
        out.insert(5, "sweep_x_value", values)
    return out


def _base_run_config(
    base_config: Mapping[str, Any],
    *,
    default_overrides: Mapping[str, Any],
    sweep_overrides: Mapping[str, Any],
    algorithms: Sequence[Any],
    regime: Mapping[str, Any],
    suite_name: str,
) -> Dict[str, Any]:
    config = deepcopy(dict(base_config))
    _apply_config_overrides(config, default_overrides)
    _apply_config_overrides(config, sweep_overrides)
    config["suite_name"] = suite_name
    config["algorithms"] = deepcopy(list(algorithms))
    config["regimes"] = [deepcopy(dict(regime))]
    return config


def run_sweep_config(
    config_or_path: ConfigInput,
    *,
    output_dir: Union[str, Path],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Run all sweeps and save combined row/summary tables.

    Parameters
    ----------
    config_or_path:
        Sweep configuration mapping or JSON/YAML file.
    output_dir:
        Root directory.  Each sweep point is also written to a subdirectory so
        the ordinary benchmark artifacts remain available for inspection.
    """

    sweep_config, config_path = _load_sweep_spec(config_or_path)
    base_path_raw = sweep_config.get("base_config")
    if base_path_raw is None:
        raise ValueError("sweep config must define base_config")
    base_path = _resolve_relative_path(base_path_raw, config_path=config_path)
    base_config = load_config(base_path)

    suite_name = str(sweep_config.get("suite_name", "path_match_sweeps"))
    default_overrides = dict(sweep_config.get("execution_overrides", {}))
    sweeps = list(sweep_config.get("sweeps", []))
    if not sweeps:
        raise ValueError("sweep config must contain at least one sweep")

    destination = ensure_dir(output_dir)
    row_frames = []
    summary_frames = []
    run_records = []
    seen_names = set()

    for sweep_index, raw_sweep in enumerate(sweeps):
        sweep = dict(raw_sweep)
        name = str(sweep.get("name", "")).strip()
        if not name:
            raise ValueError(f"Sweep {sweep_index} has no name")
        if name in seen_names:
            raise ValueError(f"Duplicate sweep name {name!r}")
        seen_names.add(name)

        kind = str(sweep.get("kind", "size")).strip().lower()
        if kind not in {"size", "algorithm"}:
            raise ValueError(f"Sweep {name!r} has unsupported kind {kind!r}")
        base_regime_name = str(sweep.get("regime", "")).strip()
        if not base_regime_name:
            raise ValueError(f"Sweep {name!r} must define regime")
        base_regime = _select_base_regime(base_config, base_regime_name)
        base_regime.update(deepcopy(dict(sweep.get("regime_overrides", {}))))
        if "n_instances" in sweep:
            base_regime["n_instances"] = int(sweep["n_instances"])

        algorithms = list(sweep.get("algorithms", []))
        if not algorithms:
            raise ValueError(f"Sweep {name!r} must select at least one algorithm")
        sweep_overrides = dict(sweep.get("execution_overrides", {}))
        sweep_dir = ensure_dir(destination / _safe_name(name))

        if kind == "size":
            x_name = str(sweep.get("x_name", "size"))
            points = list(sweep.get("points", []))
            if not points:
                raise ValueError(f"Size sweep {name!r} has no points")
            for point_index, raw_point in enumerate(points):
                point = dict(raw_point)
                label = str(point.get("label", f"point_{point_index}"))
                x_value = point.get("x_value")
                if x_value is None:
                    raise ValueError(f"Size sweep {name!r} point {label!r} has no x_value")
                regime = deepcopy(base_regime)
                regime.update(deepcopy(dict(point.get("overrides", {}))))
                regime["name"] = f"{name}__{_safe_name(label)}"
                point_config = _base_run_config(
                    base_config,
                    default_overrides=default_overrides,
                    sweep_overrides=sweep_overrides,
                    algorithms=algorithms,
                    regime=regime,
                    suite_name=f"{suite_name}__{name}__{label}",
                )
                point_dir = sweep_dir / _safe_name(label)
                rows, summary, metadata = run_benchmark_config(point_config, output_dir=point_dir)
                rows = _annotate(
                    rows,
                    sweep_name=name,
                    sweep_kind=kind,
                    base_regime=base_regime_name,
                    point_label=label,
                    x_name=x_name,
                    x_value=x_value,
                )
                summary = _annotate(
                    summary,
                    sweep_name=name,
                    sweep_kind=kind,
                    base_regime=base_regime_name,
                    point_label=label,
                    x_name=x_name,
                    x_value=x_value,
                )
                row_frames.append(rows)
                summary_frames.append(summary)
                run_records.append({
                    "sweep_name": name,
                    "sweep_kind": kind,
                    "point_label": label,
                    "x_name": x_name,
                    "x_value": x_value,
                    "output_dir": str(point_dir),
                    "metadata": metadata,
                })
        else:
            parameter_name = str(sweep.get("parameter_name", "parameter"))
            parameter_values = dict(sweep.get("parameter_values", {}))
            regime = deepcopy(base_regime)
            regime["name"] = name
            run_config = _base_run_config(
                base_config,
                default_overrides=default_overrides,
                sweep_overrides=sweep_overrides,
                algorithms=algorithms,
                regime=regime,
                suite_name=f"{suite_name}__{name}",
            )
            rows, summary, metadata = run_benchmark_config(run_config, output_dir=sweep_dir)
            resolved_names = set(summary["algorithm"].astype(str))
            missing_values = sorted(
                alias for alias in parameter_values if str(alias) not in resolved_names
            )
            if missing_values:
                raise ValueError(
                    f"Sweep {name!r} parameter_values contains unknown aliases {missing_values}; "
                    f"resolved algorithms are {sorted(resolved_names)}"
                )
            rows = _annotate(
                rows,
                sweep_name=name,
                sweep_kind=kind,
                base_regime=base_regime_name,
                point_label="algorithm_grid",
                x_name=parameter_name,
                x_value=None,
                parameter_values=parameter_values,
            )
            summary = _annotate(
                summary,
                sweep_name=name,
                sweep_kind=kind,
                base_regime=base_regime_name,
                point_label="algorithm_grid",
                x_name=parameter_name,
                x_value=None,
                parameter_values=parameter_values,
            )
            row_frames.append(rows)
            summary_frames.append(summary)
            run_records.append({
                "sweep_name": name,
                "sweep_kind": kind,
                "parameter_name": parameter_name,
                "parameter_values": parameter_values,
                "output_dir": str(sweep_dir),
                "metadata": metadata,
            })

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.*",
            category=FutureWarning,
        )
        combined_rows = pd.concat(row_frames, ignore_index=True, sort=False)
        combined_summary = pd.concat(summary_frames, ignore_index=True, sort=False)
    combined_rows.to_csv(destination / "sweep_rows.csv", index=False)
    combined_summary.to_csv(destination / "sweep_summary.csv", index=False)

    metadata = {
        "suite_name": suite_name,
        "base_config": str(base_path),
        "execution_overrides": default_overrides,
        "sweeps": sweeps,
        "runs": run_records,
    }
    (destination / "sweep_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    write_implementation_comparisons(combined_rows, destination)
    return combined_rows, combined_summary, metadata


__all__ = ["run_sweep_config"]
