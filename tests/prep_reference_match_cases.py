"""tests/prep_reference_match_cases.py

Prep script for *frozen* reference cases used by run_reference_match_tests.py.

Why this exists
---------------
We don't want the test runner to *generate* graphs using code that might change.
Instead, we generate a small suite of graphs once (using the dev notebook logic),
serialize them to a pickle, and then future test runs only load the pickle.

Typical usage
-------------
From repo root:

    python tests/prep_reference_match_cases.py

You can also run it as a module:

    python -m tests.prep_reference_match_cases

This creates:

    tests/fixtures/reference_match_cases.pkl

Edit DEFAULT_CONFIG below if you want to change which cases get frozen.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import pickle
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _repo_root_from_here() -> Path:
    # This file lives in tests/, so repo root is the parent directory.
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> Path:
    root = _repo_root_from_here()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


def _git_rev(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output([
            "git",
            "-C",
            str(repo_root),
            "rev-parse",
            "HEAD",
        ], stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _to_builtin(x: Any) -> Any:
    """Convert numpy scalars etc. into plain Python types for stable pickling."""
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return x.item()
    except Exception:
        pass

    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, (tuple, list)):
        return [_to_builtin(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_builtin(v) for k, v in x.items()}
    return x


def serialize_igraph_graph(g: Any) -> Dict[str, Any]:
    """Serialize an igraph.Graph into a pure-Python dict."""
    # Import lazily so this script can at least show errors if igraph isn't installed.
    import igraph as ig  # type: ignore

    if not isinstance(g, ig.Graph):
        raise TypeError(f"Expected igraph.Graph, got {type(g)}")

    edges = [(int(e.source), int(e.target)) for e in g.es]

    vs_attrs: Dict[str, List[Any]] = {}
    for attr in g.vs.attributes():
        vs_attrs[str(attr)] = [_to_builtin(v) for v in list(g.vs[attr])]

    graph_attrs: Dict[str, Any] = {}
    for attr in g.attributes():
        graph_attrs[str(attr)] = _to_builtin(g[attr])

    return {
        "n": int(g.vcount()),
        "edges": edges,
        "directed": bool(g.is_directed()),
        "vs_attrs": vs_attrs,
        "graph_attrs": graph_attrs,
    }


DEFAULT_CONFIG = {
    "rootleaf": {
        "seed": 0,
        "seg_len": 6,
        # sampler knobs (match notebook)
        "stub_prob": 0.75,
        "stubs_per_spine_vertex": 1,
        "stub_chain_length": 1,
        "noise_k": 2,
        "weight_mode": "increasing",
    },
    "no_matches": {
        "seed": 123,
        "spine_length_G": 10,
        "spine_length_H": 10,
        "stub_prob": 0.7,
        "stubs_per_spine_vertex": 1,
        "stub_chain_length": 1,
    },
    "blockswap": {
        "seed": 7,
    },
}


def _case_id_rootleaf(include_root_G: bool, include_leaf_G: bool, include_root_H: bool, include_leaf_H: bool) -> str:
    def tf(x: bool) -> str:
        return "T" if x else "F"

    return f"rootleaf_rG{tf(include_root_G)}_lG{tf(include_leaf_G)}_rH{tf(include_root_H)}_lH{tf(include_leaf_H)}"


def build_reference_cases() -> Dict[str, Any]:
    """Generate the exact set of cases used by the dev notebook and freeze expected outputs."""
    _ensure_repo_on_path()

    # Import dev utilities *after* path setup. Support both script and module execution.
    try:
        from .dev_utils import (  # type: ignore
            case_root_leaf_inclusion,
            case_no_matches,
            case_weighted_vs_unweighted_blockswap,
        )
    except ImportError:
        from dev_utils import (  # type: ignore
            case_root_leaf_inclusion,
            case_no_matches,
            case_weighted_vs_unweighted_blockswap,
        )

    cases: List[Dict[str, Any]] = []

    # --- Root/leaf sweep (16 cases) ------------------------------------------
    cfg = DEFAULT_CONFIG["rootleaf"]
    for include_root_G in [True, False]:
        for include_leaf_G in [True, False]:
            for include_root_H in [True, False]:
                for include_leaf_H in [True, False]:
                    res = case_root_leaf_inclusion(
                        seed=cfg["seed"],
                        seg_len=cfg["seg_len"],
                        include_root_G=include_root_G,
                        include_leaf_G=include_leaf_G,
                        include_root_H=include_root_H,
                        include_leaf_H=include_leaf_H,
                        stub_prob=cfg["stub_prob"],
                        stubs_per_spine_vertex=cfg["stubs_per_spine_vertex"],
                        stub_chain_length=cfg["stub_chain_length"],
                        noise_k=cfg["noise_k"],
                        weight_mode=cfg["weight_mode"],
                    )
                    case_id = _case_id_rootleaf(include_root_G, include_leaf_G, include_root_H, include_leaf_H)
                    cases.append(
                        {
                            "case_id": case_id,
                            "group": "rootleaf",
                            "name": res.name,
                            "seed": int(res.seed),
                            "w_kind": "weighted_identity",
                            "weight_map": _to_builtin(res.weight_map),
                            "G": serialize_igraph_graph(res.G),
                            "H": serialize_igraph_graph(res.H),
                            # Expected matcher output (frozen)
                            "expected_pairs": _to_builtin(res.found_pairs),
                            "expected_score": float(res.found_score),
                            "expected_pos_pairs": _to_builtin(res.found_pos_pairs),
                            "expected_pos_score": float(res.found_pos_score),
                            # Ground truth (useful sanity check)
                            "truth_pairs": _to_builtin(res.truth_pairs),
                            "truth_score": float(res.truth_score),
                            "truth_path_G": _to_builtin(res.truth_path_G),
                            "truth_path_H": _to_builtin(res.truth_path_H),
                            "truth_includes_root_G": bool(res.truth_includes_root_G),
                            "truth_includes_leaf_G": bool(res.truth_includes_leaf_G),
                            "truth_includes_root_H": bool(res.truth_includes_root_H),
                            "truth_includes_leaf_H": bool(res.truth_includes_leaf_H),
                            "notes": str(res.notes),
                            "config": {
                                "seg_len": cfg["seg_len"],
                                "stub_prob": cfg["stub_prob"],
                                "stubs_per_spine_vertex": cfg["stubs_per_spine_vertex"],
                                "stub_chain_length": cfg["stub_chain_length"],
                                "noise_k": cfg["noise_k"],
                                "weight_mode": cfg["weight_mode"],
                                "include_root_G": include_root_G,
                                "include_leaf_G": include_leaf_G,
                                "include_root_H": include_root_H,
                                "include_leaf_H": include_leaf_H,
                            },
                        }
                    )

    # --- No-matches case ------------------------------------------------------
    cfg = DEFAULT_CONFIG["no_matches"]
    res0 = case_no_matches(
        seed=cfg["seed"],
        spine_length_G=cfg["spine_length_G"],
        spine_length_H=cfg["spine_length_H"],
        stub_prob=cfg["stub_prob"],
        stubs_per_spine_vertex=cfg["stubs_per_spine_vertex"],
        stub_chain_length=cfg["stub_chain_length"],
    )
    cases.append(
        {
            "case_id": "no_matches_disjoint_alphabets",
            "group": "no_matches",
            "name": res0.name,
            "seed": int(res0.seed),
            "w_kind": "weighted_identity",
            "weight_map": _to_builtin(res0.weight_map),
            "G": serialize_igraph_graph(res0.G),
            "H": serialize_igraph_graph(res0.H),
            "expected_pairs": _to_builtin(res0.found_pairs),
            "expected_score": float(res0.found_score),
            "expected_pos_pairs": _to_builtin(res0.found_pos_pairs),
            "expected_pos_score": float(res0.found_pos_score),
            "truth_pairs": _to_builtin(res0.truth_pairs),
            "truth_score": float(res0.truth_score),
            "truth_path_G": _to_builtin(res0.truth_path_G),
            "truth_path_H": _to_builtin(res0.truth_path_H),
            "truth_includes_root_G": bool(res0.truth_includes_root_G),
            "truth_includes_leaf_G": bool(res0.truth_includes_leaf_G),
            "truth_includes_root_H": bool(res0.truth_includes_root_H),
            "truth_includes_leaf_H": bool(res0.truth_includes_leaf_H),
            "notes": str(res0.notes),
            "config": _to_builtin(cfg),
        }
    )

    # --- Blockswap compare (weighted vs unweighted) ---------------------------
    cfg = DEFAULT_CONFIG["blockswap"]
    cmp = case_weighted_vs_unweighted_blockswap(seed=cfg["seed"])

    cases.append(
        {
            "case_id": "blockswap_weighted_identity",
            "group": "blockswap",
            "name": cmp.weighted.name,
            "seed": int(cmp.weighted.seed),
            "w_kind": "weighted_identity",
            "weight_map": _to_builtin(cmp.weighted.weight_map),
            "G": serialize_igraph_graph(cmp.weighted.G),
            "H": serialize_igraph_graph(cmp.weighted.H),
            "expected_pairs": _to_builtin(cmp.weighted.found_pairs),
            "expected_score": float(cmp.weighted.found_score),
            "expected_pos_pairs": _to_builtin(cmp.weighted.found_pos_pairs),
            "expected_pos_score": float(cmp.weighted.found_pos_score),
            "truth_pairs": _to_builtin(cmp.weighted.truth_pairs),
            "truth_score": float(cmp.weighted.truth_score),
            "truth_path_G": _to_builtin(cmp.weighted.truth_path_G),
            "truth_path_H": _to_builtin(cmp.weighted.truth_path_H),
            "truth_includes_root_G": bool(cmp.weighted.truth_includes_root_G),
            "truth_includes_leaf_G": bool(cmp.weighted.truth_includes_leaf_G),
            "truth_includes_root_H": bool(cmp.weighted.truth_includes_root_H),
            "truth_includes_leaf_H": bool(cmp.weighted.truth_includes_leaf_H),
            "notes": str(cmp.weighted.notes),
            "config": _to_builtin(cfg),
        }
    )

    cases.append(
        {
            "case_id": "blockswap_unweighted_identity",
            "group": "blockswap",
            "name": cmp.unweighted.name,
            "seed": int(cmp.unweighted.seed),
            "w_kind": "unweighted_identity",
            "weight_map": _to_builtin(cmp.unweighted.weight_map),
            "G": serialize_igraph_graph(cmp.unweighted.G),
            "H": serialize_igraph_graph(cmp.unweighted.H),
            "expected_pairs": _to_builtin(cmp.unweighted.found_pairs),
            "expected_score": float(cmp.unweighted.found_score),
            "expected_pos_pairs": _to_builtin(cmp.unweighted.found_pos_pairs),
            "expected_pos_score": float(cmp.unweighted.found_pos_score),
            "truth_pairs": _to_builtin(cmp.unweighted.truth_pairs),
            "truth_score": float(cmp.unweighted.truth_score),
            "truth_path_G": _to_builtin(cmp.unweighted.truth_path_G),
            "truth_path_H": _to_builtin(cmp.unweighted.truth_path_H),
            "truth_includes_root_G": bool(cmp.unweighted.truth_includes_root_G),
            "truth_includes_leaf_G": bool(cmp.unweighted.truth_includes_leaf_G),
            "truth_includes_root_H": bool(cmp.unweighted.truth_includes_root_H),
            "truth_includes_leaf_H": bool(cmp.unweighted.truth_includes_leaf_H),
            "notes": str(cmp.unweighted.notes),
            "config": _to_builtin(cfg),
        }
    )

    repo_root = _repo_root_from_here()

    try:
        import numpy as np

        numpy_ver = np.__version__
    except Exception:
        numpy_ver = None

    try:
        import igraph as ig

        igraph_ver = getattr(ig, "__version__", None)
    except Exception:
        igraph_ver = None

    data: Dict[str, Any] = {
        "metadata": {
            "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": numpy_ver,
            "igraph": igraph_ver,
            "git_rev": _git_rev(repo_root),
            "default_config": _to_builtin(DEFAULT_CONFIG),
        },
        "cases": cases,
    }

    return data


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate frozen reference cases for the exact tree-path matcher.")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output pickle path (default: tests/fixtures/reference_match_cases.pkl)",
    )

    args = parser.parse_args(argv)

    repo_root = _ensure_repo_on_path()
    default_out = Path(__file__).parent / "fixtures" / "reference_match_cases.pkl"
    default_out.parent.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out).expanduser() if args.out else default_out

    data = build_reference_cases()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {len(data['cases'])} cases -> {out_path}")
    print(f"Metadata: git_rev={data['metadata'].get('git_rev')}  generated_at={data['metadata'].get('generated_at')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
