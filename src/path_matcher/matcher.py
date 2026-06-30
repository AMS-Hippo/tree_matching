
"""
Second performance patch:
- uses duck-typing for TreeData / PreprocessedTree detection, so benchmarks do not
  silently miss the fast path because of module-identity mismatches;
- keeps the specialized predict_weighted_reference(...) entry point for the old objective.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .igraph_io import igraph_to_treedata
from .tree_data import TreeData
from .needleman_wunsch_tree import AlignmentResult, WeightFn, align_trees_algorithm1
from .beam_align import CandidateFn, MatchPredicate, align_trees_beam, align_trees_beam_symmetric
from .bucketable_weight import EqualityBucketWeight, assert_bucketable_weight
from .sparse_preprocess import PreprocessedTree, preprocess_igraph
from .sparse_align import SparseCandidateConfig, align_trees_sparse_candidates
from .normalizer import exponential_count
from .reference_match import (
    ReferenceAlignmentResult,
    align_tree_to_weighted_reference,
    align_tree_to_weighted_reference_dev,
)

GraphLike = Any
ExactBeamInput = Union[GraphLike, TreeData]
SparseInput = Union[GraphLike, PreprocessedTree]


def _looks_like_treedata(x: Any) -> bool:
    return (
        hasattr(x, "parent")
        and hasattr(x, "label")
        and hasattr(x, "orig_index")
    )


def _looks_like_preprocessed_tree(x: Any) -> bool:
    return hasattr(x, "tree") and hasattr(x, "nodes_by_key")


class TreePathMatcher:
    def __init__(
        self,
        *,
        phi_name: str = "label",
        w: Optional[Any] = None,
        method: str = "exact",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        dtype: Any = np.float32,
        beam_width: int = 200,
        beam_symmetric: bool = True,
        candidate_fn: Optional[CandidateFn] = None,
        max_candidates_per_label: Optional[int] = 200,
        max_candidates_per_u: Optional[int] = None,
        candidate_select_mode: str = "first",
        seed: int = 0,
        match_predicate: Optional[MatchPredicate] = None,
        prefer_match_on_tie: bool = True,
        max_nodes_per_key: Optional[int] = None,
        key_select_mode: str = "first",
        build_subtree_sketch: bool = False,
        sketch_k: int = 8,
        sketch_max_key_freq: int = 50,
        sketch_hash_salt: int = 0,
        sparse_cfg: SparseCandidateConfig = SparseCandidateConfig(),
    ) -> None:
        method = method.lower()
        if method not in {"exact", "beam", "sparse"}:
            raise ValueError("method must be one of: 'exact', 'beam', 'sparse'")
        self.phi_name = phi_name
        self.method = method
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = strict_tree
        self.dtype = dtype
        self.beam_width = beam_width
        self.beam_symmetric = beam_symmetric
        self.candidate_fn = candidate_fn
        self.max_candidates_per_label = max_candidates_per_label
        self.max_candidates_per_u = max_candidates_per_u
        self.candidate_select_mode = candidate_select_mode
        self.seed = seed
        self.match_predicate = match_predicate
        self.prefer_match_on_tie = prefer_match_on_tie
        self.max_nodes_per_key = max_nodes_per_key
        self.key_select_mode = key_select_mode
        self.build_subtree_sketch = build_subtree_sketch
        self.sketch_k = sketch_k
        self.sketch_max_key_freq = sketch_max_key_freq
        self.sketch_hash_salt = sketch_hash_salt
        self.sparse_cfg = sparse_cfg

        self.w = EqualityBucketWeight() if (self.method == "sparse" and w is None) else w

        self.treeG_: Optional[TreeData] = None
        self.treeH_: Optional[TreeData] = None
        self.preG_: Optional[PreprocessedTree] = None
        self.preH_: Optional[PreprocessedTree] = None
        self.auxG: List[TreeData] = []

        self._last_raw_G: Any = None
        self._last_tree_G: Optional[TreeData] = None
        self._last_raw_H: Any = None
        self._last_tree_H: Optional[TreeData] = None

    def _convert_raw_graph(self, G: GraphLike) -> TreeData:
        return igraph_to_treedata(
            G,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
        )

    def _coerce_exact_beam_input(self, X: ExactBeamInput, *, slot: str) -> TreeData:
        if isinstance(X, TreeData) or _looks_like_treedata(X):
            return X  # type: ignore[return-value]

        if isinstance(X, PreprocessedTree) or _looks_like_preprocessed_tree(X):
            raise TypeError("PreprocessedTree inputs are only supported for method='sparse'.")

        if slot == "G" and X is self._last_raw_G and self._last_tree_G is not None:
            return self._last_tree_G
        if slot == "H" and X is self._last_raw_H and self._last_tree_H is not None:
            return self._last_tree_H

        tree = self._convert_raw_graph(X)
        if slot == "G":
            self._last_raw_G = X
            self._last_tree_G = tree
        else:
            self._last_raw_H = X
            self._last_tree_H = tree
        return tree

    def preprocess(self, G: GraphLike) -> PreprocessedTree:
        if self.method != "sparse":
            raise RuntimeError("preprocess() is only meaningful for method='sparse'")
        assert_bucketable_weight(self.w, mode_name="sparse")
        return preprocess_igraph(
            G,
            w=self.w,
            phi_name=self.phi_name,
            order=self.order,
            ts_field=self.ts_field,
            strict_tree=self.strict_tree,
            max_nodes_per_key=self.max_nodes_per_key,
            key_select_mode=self.key_select_mode,
            seed=self.seed,
            build_subtree_sketch=self.build_subtree_sketch,
            sketch_k=self.sketch_k,
            sketch_max_key_freq=self.sketch_max_key_freq,
            sketch_hash_salt=self.sketch_hash_salt,
        )

    def fit(self, G: Union[ExactBeamInput, SparseInput], H: Union[ExactBeamInput, SparseInput]) -> "TreePathMatcher":
        if self.method == "sparse":
            assert_bucketable_weight(self.w, mode_name="sparse")
            self.preG_ = G if (isinstance(G, PreprocessedTree) or _looks_like_preprocessed_tree(G)) else self.preprocess(G)
            self.preH_ = H if (isinstance(H, PreprocessedTree) or _looks_like_preprocessed_tree(H)) else self.preprocess(H)
            self.treeG_ = self.preG_.tree
            self.treeH_ = self.preH_.tree
            return self

        self.treeG_ = self._coerce_exact_beam_input(G, slot="G")
        self.treeH_ = self._coerce_exact_beam_input(H, slot="H")
        return self

    def normalize(self, mode: str = "exponential_count", replace: bool = False, hp: float = 1.0):
        if self.treeG_ is None:
            raise RuntimeError("Call fit() before normalize().")
        if mode == "exponential_count":
            newG = exponential_count(self.treeG_, hp)
            if replace:
                self.treeG_ = newG
            else:
                self.auxG.append(newG)
            return newG
        raise ValueError(f"Unknown normalization mode: {mode!r}")

    def predict(self, G: Optional[ExactBeamInput | SparseInput] = None, H: Optional[ExactBeamInput | SparseInput] = None) -> Tuple[List[Tuple[int, int]], float]:
        if self.method == "sparse":
            assert_bucketable_weight(self.w, mode_name="sparse")
            if G is not None or H is not None:
                if G is None or H is None:
                    raise ValueError("Either provide both G and H, or provide neither.")
                preG = G if (isinstance(G, PreprocessedTree) or _looks_like_preprocessed_tree(G)) else self.preprocess(G)
                preH = H if (isinstance(H, PreprocessedTree) or _looks_like_preprocessed_tree(H)) else self.preprocess(H)
            else:
                if self.preG_ is None or self.preH_ is None:
                    raise RuntimeError("Must call fit(G,H) before predict() if no inputs are provided.")
                preG, preH = self.preG_, self.preH_

            res: AlignmentResult = align_trees_sparse_candidates(
                preG,
                preH,
                candidates=None,
                cfg=self.sparse_cfg,
                w=self.w,
                prefer_match_on_tie=self.prefer_match_on_tie,
            )
            path_orig = [(int(preG.tree.orig_index[u]), int(preH.tree.orig_index[v])) for (u, v) in res.path_internal]
            return path_orig, res.score

        if G is not None or H is not None:
            if G is None or H is None:
                raise ValueError("Either provide both G and H, or provide neither.")
            treeG = self._coerce_exact_beam_input(G, slot="G")
            treeH = self._coerce_exact_beam_input(H, slot="H")
        else:
            if self.treeG_ is None or self.treeH_ is None:
                raise RuntimeError("Must call fit(G,H) before predict() if no inputs are provided.")
            treeG, treeH = self.treeG_, self.treeH_

        if self.w is None:
            w_fn: Optional[WeightFn] = None
        else:
            if not callable(self.w):
                raise TypeError("w must be callable for method='exact'/'beam'")
            w_fn = self.w

        if self.method == "exact":
            res = align_trees_algorithm1(
                treeG,
                treeH,
                w=w_fn,
                dtype=self.dtype,
                prefer_match_on_tie=self.prefer_match_on_tie,
            )
        else:
            if self.beam_symmetric:
                res = align_trees_beam_symmetric(
                    treeG,
                    treeH,
                    w=w_fn,
                    beam_width=self.beam_width,
                    candidate_fn=self.candidate_fn,
                    max_candidates_per_label=self.max_candidates_per_label,
                    max_candidates_per_u=self.max_candidates_per_u,
                    candidate_select_mode=self.candidate_select_mode,
                    seed=self.seed,
                    match_predicate=self.match_predicate,
                    prefer_match_on_tie=self.prefer_match_on_tie,
                )
            else:
                res = align_trees_beam(
                    treeG,
                    treeH,
                    w=w_fn,
                    beam_width=self.beam_width,
                    candidate_fn=self.candidate_fn,
                    max_candidates_per_label=self.max_candidates_per_label,
                    max_candidates_per_u=self.max_candidates_per_u,
                    candidate_select_mode=self.candidate_select_mode,
                    seed=self.seed,
                    match_predicate=self.match_predicate,
                    prefer_match_on_tie=self.prefer_match_on_tie,
                )

        path_orig = [(int(treeG.orig_index[u]), int(treeH.orig_index[v])) for (u, v) in res.path_internal]
        return path_orig, res.score

    def predict_weighted_reference(
        self,
        G: ExactBeamInput,
        ref: List[Any],
        ref_weights: List[float],
        *,
        match_rule: Optional[Any] = None,
        combine_scores: Any = "add",
        dev: bool = False,
    ) -> Tuple[List[int], float, List[Optional[Any]]]:
        treeG = self._coerce_exact_beam_input(G, slot="G")
        if dev:
            res: ReferenceAlignmentResult = align_tree_to_weighted_reference_dev(
                treeG,
                ref,
                ref_weights,
                match_rule=match_rule,
                combine_scores=combine_scores,
            )
        else:
            res = align_tree_to_weighted_reference(
                treeG,
                ref,
                ref_weights,
                match_rule=match_rule,
                combine_scores=combine_scores,
            )
        return res.path_orig, res.score, res.matched_labels
