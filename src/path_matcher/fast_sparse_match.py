"""
Specialized exact sparse matching for equality and token-overlap scores.

This is the sparse counterpart of ``fast_match.py``.  It reuses
``FastLabelEncoder`` to integerize labels, builds one inverted index per tree,
and then solves the maximum-weight candidate chain with
``align_scored_sparse_chain``.

The method is exact for the two supported score families:

1. equality: a pair is positive exactly when the integer label ids agree;
2. overlap: a pair is positive exactly when the two token sets share a token
   having positive weight, and its score is the maximum shared token weight.

For repeated comparisons, fit one encoder on the whole dataset and prepare each
tree once.  Pairwise work is then candidate generation plus O(K log m) chain
optimization, where K is the number of positive node pairs.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .diagnostics import MatchDiagnostics
from .fast_match import (
    EncodedTree,
    EncodedTreeEquality,
    EncodedTreeOverlap,
    FastLabelEncoder,
    LabelGetter,
)
from .igraph_io import igraph_to_treedata
from .needleman_wunsch_tree import AlignmentResult
from .sparse_chain import ScoredCandidate, align_scored_sparse_chain
from .sparse_preprocess import _build_tree_structure
from .tree_data import TreeData


@dataclass(frozen=True)
class FastSparsePreparedTree:
    """Integerized tree plus reusable structural and token-posting indices."""

    encoded: EncodedTree
    children: Tuple[Tuple[int, ...], ...]
    tin: np.ndarray
    tout: np.ndarray
    nodes_by_token: Dict[int, Tuple[int, ...]]
    encoder_fingerprint: Optional[Tuple[str, int]] = None

    @property
    def tree(self) -> TreeData:
        return self.encoded.tree

    @property
    def mode(self) -> str:
        if isinstance(self.encoded, EncodedTreeEquality):
            return "equality"
        if isinstance(self.encoded, EncodedTreeOverlap):
            return "overlap"
        raise TypeError(f"Unsupported encoded tree type {type(self.encoded)!r}")


def _looks_like_treedata(x: Any) -> bool:
    return hasattr(x, "parent") and hasattr(x, "label") and hasattr(x, "orig_index")


def _as_treedata(
    G: Any,
    *,
    phi_name: str,
    order: str,
    ts_field: Optional[str],
    strict_tree: bool,
) -> TreeData:
    if isinstance(G, TreeData) or _looks_like_treedata(G):
        return G  # type: ignore[return-value]
    return igraph_to_treedata(
        G,
        phi_name=phi_name,
        order=order,
        ts_field=ts_field,
        strict_tree=strict_tree,
    )


def prepare_fast_sparse_encoded(
    encoded: EncodedTree,
    *,
    encoder_fingerprint: Optional[Tuple[str, int]] = None,
) -> FastSparsePreparedTree:
    """Build reusable DFS intervals and token-to-node postings for one encoded tree."""
    children, tin, tout = _build_tree_structure(encoded.tree)
    postings_mut: Dict[int, List[int]] = {}

    if isinstance(encoded, EncodedTreeEquality):
        for u, token_id_raw in enumerate(encoded.label_ids):
            token_id = int(token_id_raw)
            if token_id >= 0:
                postings_mut.setdefault(token_id, []).append(u)
    elif isinstance(encoded, EncodedTreeOverlap):
        for u in range(encoded.tree.n):
            start = int(encoded.offsets[u])
            stop = int(encoded.offsets[u + 1])
            for token_id_raw in encoded.flat_token_ids[start:stop]:
                token_id = int(token_id_raw)
                postings_mut.setdefault(token_id, []).append(u)
    else:
        raise TypeError(f"Unsupported encoded tree type {type(encoded)!r}")

    postings = {token_id: tuple(nodes) for token_id, nodes in postings_mut.items()}
    return FastSparsePreparedTree(
        encoded=encoded,
        children=children,
        tin=tin,
        tout=tout,
        nodes_by_token=postings,
        encoder_fingerprint=encoder_fingerprint,
    )


def _validate_weight_array(weight_by_id: np.ndarray) -> np.ndarray:
    weights = np.asarray(weight_by_id, dtype=float)
    if weights.ndim != 1:
        raise ValueError("weight_by_id must be a one-dimensional array")
    if not np.all(np.isfinite(weights)):
        raise ValueError("weight_by_id must contain only finite values")
    return weights


def generate_fast_sparse_scored_candidates(
    G: FastSparsePreparedTree,
    H: FastSparsePreparedTree,
    *,
    weight_by_id: np.ndarray,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> List[List[ScoredCandidate]]:
    """Enumerate every positive pair for the supported integerized score family."""
    if G.mode != H.mode:
        raise TypeError(f"Encoded modes disagree: G is {G.mode!r}, H is {H.mode!r}")
    weights = _validate_weight_array(weight_by_id)

    candidates: List[List[ScoredCandidate]] = [[] for _ in range(G.tree.n)]
    posting_hits = 0

    if G.mode == "equality":
        encodedG = G.encoded
        assert isinstance(encodedG, EncodedTreeEquality)
        for u, token_id_raw in enumerate(encodedG.label_ids):
            token_id = int(token_id_raw)
            if token_id < 0:
                continue
            if token_id >= len(weights):
                raise ValueError(f"G contains token id {token_id} outside weight_by_id")
            weight = float(weights[token_id])
            if weight <= 0.0:
                continue
            bucket = H.nodes_by_token.get(token_id, ())
            if diagnostics is not None:
                posting_hits += len(bucket)
            candidates[u] = [(int(v), weight) for v in bucket]
        if diagnostics is not None:
            diagnostics.extra["posting_hits"] = posting_hits
        return candidates

    encodedG = G.encoded
    assert isinstance(encodedG, EncodedTreeOverlap)
    for u in range(G.tree.n):
        start = int(encodedG.offsets[u])
        stop = int(encodedG.offsets[u + 1])
        best_by_v: Dict[int, float] = {}
        for token_id_raw in encodedG.flat_token_ids[start:stop]:
            token_id = int(token_id_raw)
            if token_id < 0 or token_id >= len(weights):
                raise ValueError(f"G contains token id {token_id} outside weight_by_id")
            weight = float(weights[token_id])
            if weight <= 0.0:
                continue
            bucket = H.nodes_by_token.get(token_id, ())
            if diagnostics is not None:
                posting_hits += len(bucket)
            for v in bucket:
                old = best_by_v.get(v)
                if old is None or weight > old:
                    best_by_v[v] = weight
        candidates[u] = [(int(v), float(weight)) for v, weight in best_by_v.items()]
    if diagnostics is not None:
        diagnostics.extra["posting_hits"] = posting_hits
    return candidates


def align_trees_fast_sparse_prepared(
    G: FastSparsePreparedTree,
    H: FastSparsePreparedTree,
    *,
    weight_by_id: np.ndarray,
    diagnostics: Optional[MatchDiagnostics] = None,
) -> AlignmentResult:
    """Exact sparse-chain alignment of two prepared equality/overlap trees."""
    total_start = perf_counter()
    if G.mode != H.mode:
        raise TypeError(f"Encoded modes disagree: G is {G.mode!r}, H is {H.mode!r}")
    if (
        G.encoder_fingerprint is not None
        and H.encoder_fingerprint is not None
        and G.encoder_fingerprint != H.encoder_fingerprint
    ):
        raise ValueError(
            "Prepared trees were produced by incompatible FastLabelEncoder mappings. "
            "Fit one encoder on the full corpus and prepare both trees with that encoder."
        )
    if diagnostics is not None:
        diagnostics.reset(
            algorithm=f"fast_sparse_{G.mode}",
            n_g=G.tree.n,
            n_h=H.tree.n,
        )

    candidate_start = perf_counter()
    candidates = generate_fast_sparse_scored_candidates(
        G,
        H,
        weight_by_id=weight_by_id,
        diagnostics=diagnostics,
    )
    candidate_seconds = perf_counter() - candidate_start
    candidate_count = sum(len(row) for row in candidates)

    core_diagnostics = MatchDiagnostics() if diagnostics is not None else None
    result = align_scored_sparse_chain(
        G.tree,
        H.tree,
        candidates,
        childrenG=G.children,
        tinH=H.tin,
        toutH=H.tout,
        diagnostics=core_diagnostics,
    )
    if diagnostics is not None:
        assert core_diagnostics is not None
        posting_hits = int(diagnostics.extra.get("posting_hits", candidate_count))
        copied = core_diagnostics.copy()
        diagnostics.__dict__.clear()
        diagnostics.__dict__.update(copied.__dict__)
        diagnostics.algorithm = f"fast_sparse_{G.mode}"
        diagnostics.candidate_generation_seconds += candidate_seconds
        diagnostics.candidate_pairs_generated = candidate_count
        diagnostics.node_pair_score_evaluations = candidate_count
        diagnostics.total_seconds = perf_counter() - total_start
        diagnostics.extra["posting_hits"] = posting_hits
        diagnostics.extra["candidate_pairs_after_token_deduplication"] = candidate_count
    return result


class FastSparseTreePathMatcher:
    """
    Exact sparse matcher for equality or any-overlap token scores.

    The label model and encoder semantics match ``FastTreePathMatcher``.  The
    difference is computational: this class enumerates only positive node pairs
    and runs the sparse product-poset algorithm instead of filling the dense
    n-by-m dynamic-programming table.

    Typical repeated-comparison workflow
    ------------------------------------
    >>> matcher = FastSparseTreePathMatcher(mode="equality")
    >>> matcher.fit_encoder(trees)
    >>> prepared = [matcher.prepare_tree(T) for T in trees]
    >>> path, score = matcher.predict_prepared(prepared[i], prepared[j])

    Labels not present when the encoder was fitted are dropped with a warning.
    Prepared trees also record the encoder fit that produced their integer ids,
    preventing accidental cross-encoder comparisons.
    """

    def __init__(
        self,
        *,
        mode: str = "equality",
        label_getter: LabelGetter = None,
        token_weights: Optional[Mapping[Any, float]] = None,
        default_weight: float = 1.0,
        phi_name: str = "label",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        encoder: Optional[FastLabelEncoder] = None,
        collect_diagnostics: bool = False,
    ) -> None:
        mode = mode.lower().strip()
        if mode not in {"equality", "overlap"}:
            raise ValueError("mode must be 'equality' or 'overlap'")
        if encoder is not None and encoder.mode != mode:
            raise ValueError(f"encoder.mode={encoder.mode!r} does not match matcher mode={mode!r}")

        self.mode = mode
        self.label_getter = label_getter
        self.token_weights = dict(token_weights or {})
        self.default_weight = float(default_weight)
        self.phi_name = phi_name
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = strict_tree
        self.collect_diagnostics = bool(collect_diagnostics)
        self.encoder = encoder or FastLabelEncoder(
            mode=mode,
            label_getter=label_getter,
            token_weights=self.token_weights,
            default_weight=self.default_weight,
        )

        self.preparedG_: Optional[FastSparsePreparedTree] = None
        self.preparedH_: Optional[FastSparsePreparedTree] = None
        self.last_diagnostics_: Optional[MatchDiagnostics] = None
        self.last_fit_diagnostics_: Optional[MatchDiagnostics] = None

    def _to_tree(self, G: Any) -> TreeData:
        return _as_treedata(
            G,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
        )

    def fit_encoder(self, trees: Sequence[Any]) -> "FastSparseTreePathMatcher":
        td_trees = [self._to_tree(T) for T in trees]
        self.encoder.fit_from_trees(td_trees)
        return self

    def prepare_tree(self, G: Any) -> FastSparsePreparedTree:
        tree = self._to_tree(G)
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([tree])
        return prepare_fast_sparse_encoded(
            self.encoder.transform_tree(tree),
            encoder_fingerprint=self.encoder.fingerprint,
        )

    def fit(self, G: Any, H: Any) -> "FastSparseTreePathMatcher":
        start = perf_counter()
        self.last_fit_diagnostics_ = None
        treeG = self._to_tree(G)
        treeH = self._to_tree(H)
        if not self.encoder.is_fitted:
            self.encoder.fit_from_trees([treeG, treeH])
        fingerprint = self.encoder.fingerprint
        self.preparedG_ = prepare_fast_sparse_encoded(
            self.encoder.transform_tree(treeG),
            encoder_fingerprint=fingerprint,
        )
        self.preparedH_ = prepare_fast_sparse_encoded(
            self.encoder.transform_tree(treeH),
            encoder_fingerprint=fingerprint,
        )
        if self.collect_diagnostics:
            diag = MatchDiagnostics(algorithm="fast_sparse_fit", n_g=treeG.n, n_h=treeH.n)
            diag.input_preprocessing_seconds = perf_counter() - start
            diag.total_seconds = diag.input_preprocessing_seconds
            self.last_fit_diagnostics_ = diag
        else:
            self.last_fit_diagnostics_ = None
        return self

    def predict(
        self,
        G: Optional[Any] = None,
        H: Optional[Any] = None,
    ) -> Tuple[List[Tuple[int, int]], float]:
        total_start = perf_counter()
        self.last_diagnostics_ = None
        input_start = total_start
        if G is not None or H is not None:
            if G is None or H is None:
                raise ValueError("Either provide both G and H, or provide neither.")
            treeG = self._to_tree(G)
            treeH = self._to_tree(H)
            if not self.encoder.is_fitted:
                self.encoder.fit_from_trees([treeG, treeH])
            fingerprint = self.encoder.fingerprint
            preparedG = prepare_fast_sparse_encoded(
                self.encoder.transform_tree(treeG),
                encoder_fingerprint=fingerprint,
            )
            preparedH = prepare_fast_sparse_encoded(
                self.encoder.transform_tree(treeH),
                encoder_fingerprint=fingerprint,
            )
        else:
            if self.preparedG_ is None or self.preparedH_ is None:
                raise RuntimeError("Must call fit(G,H) before predict() if no inputs are provided.")
            preparedG = self.preparedG_
            preparedH = self.preparedH_

        input_seconds = perf_counter() - input_start if (G is not None or H is not None) else 0.0
        return self._predict_prepared(
            preparedG,
            preparedH,
            input_preprocessing_seconds=input_seconds,
            total_start=total_start,
        )

    def predict_prepared(
        self,
        G: FastSparsePreparedTree,
        H: FastSparsePreparedTree,
    ) -> Tuple[List[Tuple[int, int]], float]:
        self.last_diagnostics_ = None
        return self._predict_prepared(G, H, input_preprocessing_seconds=0.0, total_start=perf_counter())

    def _predict_prepared(
        self,
        G: FastSparsePreparedTree,
        H: FastSparsePreparedTree,
        *,
        input_preprocessing_seconds: float,
        total_start: float,
    ) -> Tuple[List[Tuple[int, int]], float]:
        if not self.encoder.is_fitted or self.encoder.weight_by_id is None:
            raise RuntimeError("FastLabelEncoder is not fitted")
        current_fingerprint = self.encoder.fingerprint
        for name, prepared in (("G", G), ("H", H)):
            if (
                prepared.encoder_fingerprint is not None
                and prepared.encoder_fingerprint != current_fingerprint
            ):
                raise ValueError(
                    f"Prepared tree {name} was encoded with a different FastLabelEncoder "
                    "mapping (or before the current encoder was refitted). Refit the encoder "
                    "on the full corpus and prepare every tree again."
                )
        diagnostics = MatchDiagnostics() if self.collect_diagnostics else None
        result = align_trees_fast_sparse_prepared(
            G,
            H,
            weight_by_id=self.encoder.weight_by_id,
            diagnostics=diagnostics,
        )
        if diagnostics is not None:
            diagnostics.input_preprocessing_seconds += input_preprocessing_seconds
            diagnostics.total_seconds = perf_counter() - total_start
            self.last_diagnostics_ = diagnostics.copy()
        else:
            self.last_diagnostics_ = None
        path_orig = [
            (int(G.tree.orig_index[u]), int(H.tree.orig_index[v]))
            for u, v in result.path_internal
        ]
        return path_orig, result.score


__all__ = [
    "FastSparsePreparedTree",
    "FastSparseTreePathMatcher",
    "prepare_fast_sparse_encoded",
    "generate_fast_sparse_scored_candidates",
    "align_trees_fast_sparse_prepared",
]
