"""
High-level matcher interface for exact, beam, and sparse tree-path matching.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from .igraph_io import igraph_to_treedata
from .tree_data import TreeData
from .needleman_wunsch_tree import AlignmentResult, WeightFn, align_trees_algorithm1, align_tree_to_repeating_template
from .beam_align import (
    BeamCandidateHeuristicFn,
    BeamExpansionFn,
    BeamPriorityFn,
    CandidateFn,
    MatchPredicate,
    align_trees_beam,
    align_trees_beam_symmetric,
)
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
        mode: str = "unique",
        order: str = "auto",
        ts_field: Optional[str] = None,
        strict_tree: bool = True,
        dtype: Any = np.float32,
        beam_width: int = 200,
        beam_symmetric: bool = False,
        beam_expansion_width: Optional[int] = 64,
        beam_expansion_fn: Optional[BeamExpansionFn] = None,
        beam_candidate_heuristic_fn: Optional[BeamCandidateHeuristicFn] = None,
        beam_priority_fn: Optional[BeamPriorityFn] = None,
        beam_min_match_score: float = 0.0,
        beam_random_fraction: float = 0.10,
        beam_n_restarts: int = 1,
        beam_max_length: Optional[int] = None,
        beam_max_label_pair_scan: int = 100_000,
        beam_max_label_pairs_per_expansion: Optional[int] = 2_048,
        beam_max_nodes_per_label_side: int = 8,
        beam_rarity_weight: float = 0.25,
        beam_gap_penalty: float = 0.03,
        beam_balance_penalty: float = 0.01,
        beam_candidate_future_weight: float = 0.03,
        beam_priority_future_weight: float = 0.20,
        beam_priority_length_weight: float = 0.0,
        candidate_fn: Optional[CandidateFn] = None,
        max_candidates_per_label: Optional[int] = None,
        max_candidates_per_u: Optional[int] = None,
        candidate_select_mode: str = "mixed",
        seed: int = 0,
        match_predicate: Optional[MatchPredicate] = None,
        prefer_match_on_tie: bool = True,
        template_repeat_penalty: float = 0.0,
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
        mode = mode.lower().strip()
        if mode not in {"unique", "template_repeat"}:
            raise ValueError("mode must be one of: 'unique', 'template_repeat'")
        if mode == "template_repeat" and method != "exact":
            raise ValueError("mode='template_repeat' is currently supported only with method='exact'")
        template_repeat_penalty = float(template_repeat_penalty)
        if template_repeat_penalty < 0.0:
            raise ValueError("template_repeat_penalty must be nonnegative")
        if beam_width < 1:
            raise ValueError("beam_width must be >= 1")
        if beam_expansion_width is not None and beam_expansion_width < 1:
            raise ValueError("beam_expansion_width must be >= 1, or None")
        if beam_n_restarts < 1:
            raise ValueError("beam_n_restarts must be >= 1")
        if not (0.0 <= float(beam_random_fraction) <= 1.0):
            raise ValueError("beam_random_fraction must be between 0 and 1")
        if beam_max_label_pair_scan < 1:
            raise ValueError("beam_max_label_pair_scan must be >= 1")
        if beam_max_label_pairs_per_expansion is not None and beam_max_label_pairs_per_expansion < 1:
            raise ValueError("beam_max_label_pairs_per_expansion must be >= 1, or None")
        if beam_max_nodes_per_label_side < 1:
            raise ValueError("beam_max_nodes_per_label_side must be >= 1")
        if beam_expansion_fn is not None and candidate_fn is not None:
            raise ValueError("Provide only one of beam_expansion_fn or candidate_fn")

        self.phi_name = phi_name
        self.method = method
        self.mode = mode
        self.order = order
        self.ts_field = ts_field
        self.strict_tree = strict_tree
        self.dtype = dtype

        self.beam_width = int(beam_width)
        self.beam_symmetric = bool(beam_symmetric)
        self.beam_expansion_width = beam_expansion_width
        self.beam_expansion_fn = beam_expansion_fn
        self.beam_candidate_heuristic_fn = beam_candidate_heuristic_fn
        self.beam_priority_fn = beam_priority_fn
        self.beam_min_match_score = float(beam_min_match_score)
        self.beam_random_fraction = float(beam_random_fraction)
        self.beam_n_restarts = int(beam_n_restarts)
        self.beam_max_length = beam_max_length
        self.beam_max_label_pair_scan = int(beam_max_label_pair_scan)
        self.beam_max_label_pairs_per_expansion = beam_max_label_pairs_per_expansion
        self.beam_max_nodes_per_label_side = int(beam_max_nodes_per_label_side)
        self.beam_rarity_weight = float(beam_rarity_weight)
        self.beam_gap_penalty = float(beam_gap_penalty)
        self.beam_balance_penalty = float(beam_balance_penalty)
        self.beam_candidate_future_weight = float(beam_candidate_future_weight)
        self.beam_priority_future_weight = float(beam_priority_future_weight)
        self.beam_priority_length_weight = float(beam_priority_length_weight)

        self.candidate_fn = candidate_fn
        self.max_candidates_per_label = max_candidates_per_label
        self.max_candidates_per_u = max_candidates_per_u
        self.candidate_select_mode = candidate_select_mode
        self.seed = int(seed)
        self.match_predicate = match_predicate
        self.prefer_match_on_tie = prefer_match_on_tie
        self.template_repeat_penalty = template_repeat_penalty
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

    def predict(
        self,
        G: Optional[ExactBeamInput | SparseInput] = None,
        H: Optional[ExactBeamInput | SparseInput] = None,
    ) -> Tuple[List[Tuple[int, int]], float]:
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
            if self.mode == "template_repeat":
                res = align_tree_to_repeating_template(
                    treeG,
                    treeH,
                    w=w_fn,
                    dtype=self.dtype,
                    prefer_match_on_tie=self.prefer_match_on_tie,
                    repeat_penalty=self.template_repeat_penalty,
                )
            else:
                res = align_trees_algorithm1(
                    treeG,
                    treeH,
                    w=w_fn,
                    dtype=self.dtype,
                    prefer_match_on_tie=self.prefer_match_on_tie,
                )
        else:
            beam_fn = align_trees_beam_symmetric if self.beam_symmetric else align_trees_beam
            res = beam_fn(
                treeG,
                treeH,
                w=w_fn,
                beam_width=self.beam_width,
                expansion_width=self.beam_expansion_width,
                expansion_fn=self.beam_expansion_fn,
                candidate_heuristic=self.beam_candidate_heuristic_fn,
                priority_fn=self.beam_priority_fn,
                n_restarts=self.beam_n_restarts,
                min_match_score=self.beam_min_match_score,
                max_label_pair_scan=self.beam_max_label_pair_scan,
                max_label_pairs_per_expansion=self.beam_max_label_pairs_per_expansion,
                max_nodes_per_label_side=self.beam_max_nodes_per_label_side,
                random_fraction=self.beam_random_fraction,
                rarity_weight=self.beam_rarity_weight,
                gap_penalty=self.beam_gap_penalty,
                balance_penalty=self.beam_balance_penalty,
                candidate_future_weight=self.beam_candidate_future_weight,
                priority_future_weight=self.beam_priority_future_weight,
                priority_length_weight=self.beam_priority_length_weight,
                candidate_fn=self.candidate_fn,
                max_candidates_per_label=self.max_candidates_per_label,
                max_candidates_per_u=self.max_candidates_per_u,
                candidate_select_mode=self.candidate_select_mode,
                seed=self.seed,
                match_predicate=self.match_predicate,
                prefer_match_on_tie=self.prefer_match_on_tie,
                max_length=self.beam_max_length,
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
