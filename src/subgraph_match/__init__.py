"""subgraph_match package

This package contains utilities for matching *subtrees/subgraphs* of rooted trees.

It provides:

1) **Timestamp-ordered subtree matching** (ordered children)
   - fast exact DP in O(|G||H|)
   - implemented in `fast_subgraph_match.py`

2) **Unordered-children subtree matching** (no timestamps / no left-to-right order)
   - exact DP, but inherently slower (requires solving child assignment)
   - implemented in `fast_subgraph_match_unordered.py`

See `SYNTAX_subgraph_matching.ipynb` for usage examples.
"""

from .fast_subgraph_match import (
    HAVE_NUMBA,
    FastSubtreeMatcher,
    FastSubtreeAlignmentResult,
    align_subtrees_fast_encoded,
)

from .fast_subgraph_match_unordered import (
    FastUnorderedSubtreeMatcher,
    align_subtrees_unordered_fast_encoded,
)

__all__ = [
    "HAVE_NUMBA",
    "FastSubtreeMatcher",
    "FastUnorderedSubtreeMatcher",
    "FastSubtreeAlignmentResult",
    "align_subtrees_fast_encoded",
    "align_subtrees_unordered_fast_encoded",
]
