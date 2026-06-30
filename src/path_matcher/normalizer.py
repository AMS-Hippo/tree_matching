from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

try:
    # If you're using a package layout
    from .tree_data import TreeData
except Exception:
    # If this file sits next to tree_data.py
    from tree_data import TreeData


def exponential_count(tree: TreeData, c: float) -> TreeData:
    """
    Return a new TreeData with exponentially-weighted subtree label counts.

    Parameters
    ----------
    tree:
        Input rooted tree in TreeData format.
    c:
        Real number > 1.0. Larger c puts more weight on deeper descendants.

    Returns
    -------
    TreeData
        Same structure as `tree` (same parent/orig_index), but `label[v]` is a dict
        mapping each original label L to its weighted count in the subtree of v.

    Raises
    ------
    ValueError
        If c <= 1.
    """
    c = float(c)
    if not (c > 1.0):
        raise ValueError(f"Expected c > 1, got c={c!r}.")

    parent = np.asarray(tree.parent, dtype=np.int64)
    n = int(parent.shape[0])

    # Build children lists (O(n)).
    children: List[List[int]] = [[] for _ in range(n)]
    for u in range(1, n):
        p = int(parent[u])
        children[p].append(u)

    # Preprocess: list of all labels (mainly for the user's stated workflow).
    # (We don't strictly need this list for the dict-based DP, but it can be handy.)
    _all_labels = list(tree.label)  # noqa: F841

    # Bottom-up DP: new_label[u] is a dict label -> weight for subtree(u).
    new_label: List[Dict[Any, float]] = [dict() for _ in range(n)]

    for v in range(n - 1, -1, -1):
        d: Dict[Any, float] = {tree.label[v]: 1.0}

        # Merge children's dicts, scaling by c (dist(v, ·) = 1 + dist(child, ·)).
        for ch in children[v]:
            cd = new_label[ch]
            if cd:
                for lab, w in cd.items():
                    d[lab] = d.get(lab, 0.0) + c * float(w)

        new_label[v] = d

    return TreeData(
        parent=np.asarray(tree.parent).copy(),
        label=new_label,
        orig_index=np.asarray(tree.orig_index).copy(),
    )
