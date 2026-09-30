from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "benchmarking" / "01_algorithm_ranking.ipynb"


def test_algorithm_ranking_notebook_is_parseable_and_has_small_parameter_surface() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4

    parameter_cells = [
        cell
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
        and "parameters" in cell.get("metadata", {}).get("tags", [])
    ]
    assert len(parameter_cells) == 1

    source = "".join(parameter_cells[0]["source"])
    module = ast.parse(source)
    assigned = {
        target.id
        for node in module.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert assigned == {
        "SCALE",
        "RUN_LABEL",
        "REGIMES",
        "ALGORITHMS",
        "SIZE_OVERRIDES",
        "N_INSTANCES",
        "REPEATS",
        "SEED",
    }


def test_algorithm_ranking_notebook_uses_tested_backend_and_has_no_saved_outputs() -> None:
    payload = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in payload["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "run_benchmark_config" in code
    assert "exact_disagreement" in code
    assert "accuracy_percent" in code
    assert "truth_pair_recall" not in code
    assert "SIZE_OVERRIDES" in code
    assert "RUN_LABEL" in code
    assert "ALGORITHMS" in code
    assert "allowed_size_fields" in code
    assert "path_match_benchmark_large.json" in code
    assert "column in work_summary.columns" in code

    for cell in payload["cells"]:
        if cell.get("cell_type") == "code":
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None
