"""tests/run_reference_match_tests.py

Run snapshot-style unit tests for the exact matcher, and (when applicable)
for the fast special-case matchers.

This is intended to be stable and lightweight:
- it does NOT call any samplers
- it does NOT generate new graphs
- it only loads a pickle produced by prep_reference_match_cases.py

Usage
-----
From repo root:

    python tests/run_reference_match_tests.py

You can also run it as a module:

    python -m tests.run_reference_match_tests

Outputs
-------
- Exit code 0 on success, 1 if any failures.
- Appends a human-readable summary to a log file on every run.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EPS = 1e-9


def _repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_repo_on_path() -> Path:
    root = _repo_root_from_here()
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return root


# -----------------------------------------------------------------------------
# Minimal scoring utilities (kept local so the runner doesn't depend on dev utils)
# -----------------------------------------------------------------------------

WeightMap = Dict[Any, float]
WeightFn = Callable[[Any, Any], float]
LabelGetter = Optional[Callable[[Any], Any]]


def _default_extract(label: Any) -> Any:
    if isinstance(label, dict):
        for key in ("Labels", "labels", "label", "tokens", "token_set"):
            if key in label:
                return label[key]
    return label


def _make_label_getter(case: Dict[str, Any]) -> LabelGetter:
    field = case.get("label_getter_field", None)
    if field is None:
        return None

    field = str(field)

    def getter(label: Any) -> Any:
        if isinstance(label, dict):
            return label.get(field)
        return getattr(label, field)

    return getter


def _extract_raw(label: Any, label_getter: LabelGetter) -> Any:
    return label_getter(label) if label_getter is not None else _default_extract(label)


def _is_scalar_token(x: Any) -> bool:
    return isinstance(x, (str, bytes, int, float))


def _iter_overlap_tokens(label: Any, label_getter: LabelGetter) -> Iterable[Any]:
    raw = _extract_raw(label, label_getter)
    if raw is None:
        return ()
    if _is_scalar_token(raw):
        return (raw,)
    if isinstance(raw, dict):
        raise TypeError("Overlap scoring got dict after extraction; pass label_getter_field in the case metadata.")
    try:
        vals = list(raw)
    except TypeError:
        vals = [raw]
    # Unique, deterministic enough for scoring.
    seen: Dict[Any, None] = {}
    for v in vals:
        try:
            seen[v] = None
        except TypeError:
            seen[str(v)] = None
    return tuple(seen.keys())


def _lookup_weight(weight_map: Mapping[Any, float], key: Any, default: float = 0.0) -> float:
    try:
        if key in weight_map:
            return float(weight_map[key])
    except TypeError:
        pass
    skey = str(key)
    if skey in weight_map:
        return float(weight_map[skey])
    return float(default)


def weighted_identity_w(weight_map: WeightMap, *, label_getter: LabelGetter = None) -> WeightFn:
    def w(a: Any, b: Any) -> float:
        aa = _extract_raw(a, label_getter)
        bb = _extract_raw(b, label_getter)
        if aa != bb:
            return 0.0
        return _lookup_weight(weight_map, aa, 0.0)

    return w


def unweighted_identity_w(a: Any, b: Any) -> float:
    return 1.0 if a == b else 0.0


def weighted_overlap_w(weight_map: WeightMap, *, label_getter: LabelGetter = None) -> WeightFn:
    def w(a: Any, b: Any) -> float:
        toks_a = set(_iter_overlap_tokens(a, label_getter))
        toks_b = set(_iter_overlap_tokens(b, label_getter))
        if not toks_a or not toks_b:
            return 0.0
        inter = toks_a.intersection(toks_b)
        if not inter:
            return 0.0
        return max(_lookup_weight(weight_map, tok, 0.0) for tok in inter)

    return w


def unweighted_overlap_w(a: Any, b: Any) -> float:
    try:
        sa = set(a) if not _is_scalar_token(a) else {a}
    except TypeError:
        sa = {a}
    try:
        sb = set(b) if not _is_scalar_token(b) else {b}
    except TypeError:
        sb = {b}
    return 1.0 if sa.intersection(sb) else 0.0


def score_pairs(G: Any, H: Any, pairs: Sequence[Tuple[int, int]], w: WeightFn) -> float:
    labG = G.vs["label"]
    labH = H.vs["label"]
    s = 0.0
    for u, v in pairs:
        s += float(w(labG[int(u)], labH[int(v)]))
    return float(s)


def filter_positive_pairs(
    G: Any,
    H: Any,
    pairs: Sequence[Tuple[int, int]],
    w: WeightFn,
    eps: float = EPS,
) -> List[Tuple[int, int]]:
    labG = G.vs["label"]
    labH = H.vs["label"]
    out: List[Tuple[int, int]] = []
    for u, v in pairs:
        val = float(w(labG[int(u)], labH[int(v)]))
        if val > eps:
            out.append((int(u), int(v)))
    return out


# -----------------------------------------------------------------------------
# Reference-case IO
# -----------------------------------------------------------------------------


def load_reference_pickle(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "cases" not in data:
        raise ValueError("Reference pickle has unexpected format (expected dict with key 'cases').")
    if not isinstance(data["cases"], list):
        raise ValueError("Reference pickle has unexpected format (cases must be a list).")
    return data


def reconstruct_igraph_graph(d: Dict[str, Any]) -> Any:
    import igraph as ig  # type: ignore

    n = int(d["n"])
    edges = [(int(u), int(v)) for (u, v) in d["edges"]]
    directed = bool(d.get("directed", True))

    g = ig.Graph(n=n, edges=edges, directed=directed)

    vs_attrs: Dict[str, List[Any]] = d.get("vs_attrs", {})
    for attr, vals in vs_attrs.items():
        g.vs[str(attr)] = list(vals)

    graph_attrs: Dict[str, Any] = d.get("graph_attrs", {})
    for attr, val in graph_attrs.items():
        try:
            g[str(attr)] = val
        except Exception:
            pass

    return g


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------


def configure_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("reference_match_tests")
    logger.setLevel(logging.INFO)

    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path for h in logger.handlers):
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fmt = logging.Formatter("%(message)s")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(sh)

    return logger


# -----------------------------------------------------------------------------
# Test execution helpers
# -----------------------------------------------------------------------------


def _basic_graph_sanity(g: Any) -> List[str]:
    """Return list of error strings; empty means OK."""
    errs: List[str] = []

    if not getattr(g, "is_directed", lambda: False)():
        errs.append("graph is not directed")

    vs_attrs = set(getattr(g.vs, "attributes", lambda: [])())
    for req in ["label", "is_planted"]:
        if req not in vs_attrs:
            errs.append(f"missing vertex attribute {req!r}")

    try:
        roots = g.vs.select(_indegree_eq=0)
        if len(roots) != 1:
            errs.append(f"expected exactly one root; found {len(roots)}")
    except Exception:
        errs.append("could not compute root via indegree")

    return errs


def _make_w(case: Dict[str, Any], *, label_getter: LabelGetter = None) -> Optional[WeightFn]:
    kind = str(case.get("w_kind", "weighted_identity")).lower().strip()

    if kind == "weighted_identity":
        wm = case.get("weight_map", {})
        if not isinstance(wm, dict):
            raise TypeError("weight_map must be a dict for weighted_identity")
        return weighted_identity_w(dict(wm), label_getter=label_getter)

    if kind == "unweighted_identity":
        return lambda a, b: unweighted_identity_w(_extract_raw(a, label_getter), _extract_raw(b, label_getter))

    if kind in {"weighted_overlap", "weighted_any_overlap", "max_overlap", "weighted_max_overlap"}:
        wm = case.get("weight_map", {})
        if not isinstance(wm, dict):
            raise TypeError("weight_map must be a dict for weighted_overlap")
        return weighted_overlap_w(dict(wm), label_getter=label_getter)

    if kind in {"unweighted_overlap", "any_overlap", "overlap"}:
        def _uw(a: Any, b: Any) -> float:
            return 1.0 if set(_iter_overlap_tokens(a, label_getter)).intersection(_iter_overlap_tokens(b, label_getter)) else 0.0
        return _uw

    if kind in {"none", "null"}:
        return None

    raise ValueError(f"Unknown w_kind={kind!r}")


def _compare_pairs(expected: Sequence[Tuple[int, int]], got: Sequence[Tuple[int, int]]) -> Optional[str]:
    if list(expected) == list(got):
        return None
    exp = list(expected)
    gg = list(got)
    m = min(len(exp), len(gg))
    first_mismatch = None
    for i in range(m):
        if exp[i] != gg[i]:
            first_mismatch = i
            break
    if first_mismatch is None:
        return f"pairs differ (prefix). expected_len={len(exp)} got_len={len(gg)}"
    return f"pairs differ at i={first_mismatch}: expected={exp[first_mismatch]} got={gg[first_mismatch]} (expected_len={len(exp)} got_len={len(gg)})"


def _import_fast_matcher() -> Any:
    try:
        from path_matcher.fast_match import FastTreePathMatcher  # type: ignore
        return FastTreePathMatcher
    except Exception:
        try:
            from fast_match import FastTreePathMatcher  # type: ignore
            return FastTreePathMatcher
        except Exception as e:
            raise ImportError("Could not import FastTreePathMatcher from package or loose file.") from e


def _fast_modes_for_case(case: Dict[str, Any]) -> List[str]:
    raw = case.get("fast_modes", None)
    if raw is not None:
        if isinstance(raw, str):
            vals = [raw]
        else:
            vals = list(raw)
        return [str(v).lower().strip() for v in vals]

    raw = case.get("fast_mode", None)
    if raw is not None:
        return [str(raw).lower().strip()]

    kind = str(case.get("w_kind", "weighted_identity")).lower().strip()
    if kind in {"weighted_identity", "unweighted_identity"}:
        return ["equality"]
    if kind in {"weighted_overlap", "weighted_any_overlap", "max_overlap", "weighted_max_overlap", "unweighted_overlap", "any_overlap", "overlap"}:
        return ["overlap"]
    return []


def _fast_kwargs_for_case(case: Dict[str, Any]) -> Dict[str, Any]:
    label_getter = _make_label_getter(case)
    kind = str(case.get("w_kind", "weighted_identity")).lower().strip()
    weight_map = case.get("weight_map", {})

    kwargs: Dict[str, Any] = {
        "label_getter": label_getter,
    }
    if kind in {"weighted_identity", "weighted_overlap", "weighted_any_overlap", "max_overlap", "weighted_max_overlap"}:
        kwargs["token_weights"] = dict(weight_map)
        kwargs["default_weight"] = 0.0
    else:
        kwargs["token_weights"] = None
        kwargs["default_weight"] = 1.0
    return kwargs


def _run_backend(
    *,
    backend_name: str,
    G: Any,
    H: Any,
    case: Dict[str, Any],
    logger: logging.Logger,
    score_tol: float,
) -> bool:
    label_getter = _make_label_getter(case)
    w = _make_w(case, label_getter=label_getter)

    expected_pairs = [(int(u), int(v)) for (u, v) in case.get("expected_pos_pairs", case.get("expected_pairs", []))]
    expected_score = float(case.get("expected_pos_score", case.get("expected_score", 0.0)))
    truth_pairs = [(int(u), int(v)) for (u, v) in case.get("truth_pairs", [])]
    truth_score = float(case.get("truth_score", 0.0))

    if backend_name == "exact":
        try:
            from path_matcher.matcher import TreePathMatcher  # type: ignore
        except Exception:
            from matcher import TreePathMatcher  # type: ignore

        matcher = TreePathMatcher(method="exact", w=w)
        matcher.fit(G, H)
        pairs, score = matcher.predict()
    elif backend_name.startswith("fast_"):
        FastTreePathMatcher = _import_fast_matcher()
        mode = "equality" if "equality" in backend_name else "overlap"
        kwargs = _fast_kwargs_for_case(case)
        matcher = FastTreePathMatcher(mode=mode, **kwargs)
        if backend_name.endswith("preencoded"):
            matcher.fit_encoder([G, H])
            encG = matcher.encode_tree(G)
            encH = matcher.encode_tree(H)
            pairs, score = matcher.predict_encoded(encG, encH)
        else:
            matcher.fit(G, H)
            pairs, score = matcher.predict()
    else:
        raise ValueError(f"Unknown backend_name={backend_name!r}")

    pairs2 = [(int(u), int(v)) for (u, v) in pairs]
    score2 = float(score)

    if w is None:
        pos_pairs = pairs2
        pos_score = score2
    else:
        pos_pairs = filter_positive_pairs(G, H, pairs2, w)
        pos_score = score_pairs(G, H, pos_pairs, w)

    pair_err = _compare_pairs(expected_pairs, pos_pairs)
    score_ok = abs(pos_score - expected_score) <= score_tol
    truth_pair_ok = (pos_pairs == truth_pairs) if truth_pairs else True
    truth_score_ok = abs(pos_score - truth_score) <= score_tol if truth_pairs else True

    case_id = str(case.get("case_id", case.get("name", "<unnamed>")))
    if pair_err is None and score_ok:
        logger.info(f"PASS {case_id} [{backend_name}]: score={pos_score:.6g} len={len(pos_pairs)}")
        return True

    logger.info(f"FAIL {case_id} [{backend_name}]:")
    if pair_err is not None:
        logger.info(f"  {pair_err}")
        logger.info(f"  expected_pairs={expected_pairs}")
        logger.info(f"  got_pairs     ={pos_pairs}")
    if not score_ok:
        logger.info(f"  score mismatch: expected={expected_score:.12g} got={pos_score:.12g} tol={score_tol}")
    if truth_pairs:
        logger.info(f"  truth_pairs_ok={truth_pair_ok} truth_score_ok={truth_score_ok}")
        if not truth_pair_ok:
            logger.info(f"  truth_pairs={truth_pairs}")
    return False


def run_one_case(
    case: Dict[str, Any],
    *,
    logger: logging.Logger,
    score_tol: float = 1e-6,
    include_fast: bool = True,
) -> Dict[str, bool]:
    """Run all applicable backends for one case. Never raises."""
    case_id = str(case.get("case_id", case.get("name", "<unnamed>")))
    results: Dict[str, bool] = {}

    try:
        G = reconstruct_igraph_graph(case["G"])
        H = reconstruct_igraph_graph(case["H"])

        errs = _basic_graph_sanity(G) + [f"H: {e}" for e in _basic_graph_sanity(H)]
        if errs:
            logger.info(f"FAIL {case_id}: graph sanity errors: {errs}")
            return {"exact": False}

        backend_names: List[str] = ["exact"]
        if include_fast:
            for mode in _fast_modes_for_case(case):
                if mode == "equality":
                    backend_names.extend(["fast_equality", "fast_equality_preencoded"])
                elif mode == "overlap":
                    backend_names.extend(["fast_overlap", "fast_overlap_preencoded"])
                else:
                    logger.info(f"WARN {case_id}: ignoring unknown fast mode {mode!r}")

        for backend_name in backend_names:
            try:
                results[backend_name] = _run_backend(
                    backend_name=backend_name,
                    G=G,
                    H=H,
                    case=case,
                    logger=logger,
                    score_tol=score_tol,
                )
            except Exception as e:
                logger.info(f"ERROR {case_id} [{backend_name}]: exception {type(e).__name__}: {e}")
                logger.info(traceback.format_exc())
                results[backend_name] = False

        return results

    except Exception as e:
        logger.info(f"ERROR {case_id}: exception {type(e).__name__}: {e}")
        logger.info(traceback.format_exc())
        return {"exact": False}


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run snapshot tests for the exact and fast tree-path matchers.")
    parser.add_argument(
        "--ref",
        type=str,
        default=None,
        help="Reference pickle path (default: tests/fixtures/reference_match_cases.pkl)",
    )
    parser.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log file path (default: .cache/test_logs/reference_match_tests.log)",
    )
    parser.add_argument(
        "--score-tol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for score comparisons.",
    )
    parser.add_argument(
        "--skip-fast",
        action="store_true",
        help="Only run the generic exact matcher; skip fast special-case matchers.",
    )

    args = parser.parse_args(argv)

    _ensure_repo_on_path()

    ref_path = Path(args.ref).expanduser() if args.ref else Path(__file__).parent / "fixtures" / "reference_match_cases.pkl"
    log_path = Path(args.log).expanduser() if args.log else Path(__file__).parents[1] / ".cache" / "test_logs" / "reference_match_tests.log"

    logger = configure_logger(log_path)

    ts = _dt.datetime.now().isoformat(timespec="seconds")
    logger.info("\n" + "=" * 80)
    logger.info(f"Reference matcher test run @ {ts}")
    logger.info(f"ref_pickle={ref_path}")
    logger.info(f"include_fast={not bool(args.skip_fast)}")

    if not ref_path.exists():
        logger.info("FAIL: reference pickle not found. Run prep_reference_match_cases.py first.")
        return 1

    data = load_reference_pickle(ref_path)
    cases = data.get("cases", [])

    n_case_pass = 0
    n_case_fail = 0
    backend_pass: Dict[str, int] = {}
    backend_fail: Dict[str, int] = {}

    for case in cases:
        case_results = run_one_case(
            case,
            logger=logger,
            score_tol=float(args.score_tol),
            include_fast=not bool(args.skip_fast),
        )
        case_ok = all(bool(v) for v in case_results.values())
        if case_ok:
            n_case_pass += 1
        else:
            n_case_fail += 1

        for name, ok in case_results.items():
            if ok:
                backend_pass[name] = backend_pass.get(name, 0) + 1
            else:
                backend_fail[name] = backend_fail.get(name, 0) + 1

    logger.info("-" * 80)
    logger.info(f"Case summary: pass={n_case_pass} fail={n_case_fail} total={n_case_pass + n_case_fail}")
    all_backend_names = sorted(set(backend_pass) | set(backend_fail))
    if all_backend_names:
        logger.info("Backend summary:")
        for name in all_backend_names:
            logger.info(
                f"  {name}: pass={backend_pass.get(name, 0)} fail={backend_fail.get(name, 0)} total={backend_pass.get(name, 0) + backend_fail.get(name, 0)}"
            )

    return 0 if n_case_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
