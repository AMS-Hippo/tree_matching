
"""
Preprocessing for sparse-candidate matching.

The goal: do per-tree work once (close to linear in tree size), then reuse it across
many pairwise match computations.

This preprocessing is built around a *bucketable weight* w, meaning:
  - w is callable (score),
  - w.blocking_keys(label) returns an iterable of hashable keys,
and plausible matches are expected to share at least one blocking key.

We compute:
- TreeData (possibly reordered to satisfy parent<child),
- node_keys[u]: tuple of blocking keys for each node u,
- key_to_nodes: inverted index key -> list of node ids,
- key_counts: full counts per key (before truncation),
- optional subtree sketches (k-minhash on rare keys) + index for sketch keys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple, Union

import hashlib
import numpy as np

from .tree_data import TreeData
from .igraph_io import igraph_to_treedata
from .bucketable_weight import BucketableWeight, EqualityBucketWeight, assert_bucketable_weight
from .candidates import select_subset


_UINT64_MAX = np.uint64(2**64 - 1)


def stable_hash64(x: Any, *, salt: int = 0) -> int:
    """
    Stable 64-bit hash for arbitrary Python objects using repr(x).

    Notes
    -----
    - This is not cryptographic; it's just stable across runs.
    - For speed, keep x small / simple (strings/tuples are ideal).
    """
    b = repr(x).encode("utf-8", errors="backslashreplace")
    salt_b = int(salt).to_bytes(8, "little", signed=False)
    h = hashlib.blake2b(salt_b + b, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


@dataclass(frozen=True)
class PreprocessedTree:
    """
    Cached per-tree data used by sparse-candidate matching.

    Attributes
    ----------
    tree:
        Internal TreeData (ordered, with orig_index mapping).
    w:
        The bucketable weight used to generate node_keys (and typically also to score).
    node_keys:
        node_keys[u] is a tuple of blocking keys for node u.
    key_to_nodes:
        Inverted index: key -> list of node ids (possibly truncated for storage).
    key_counts:
        Full counts per key (before truncation).
    sketch_k:
        Number of minhash values per node in subtree sketch.
    subtree_sketch:
        shape (n, sketch_k) uint64 array; padding with UINT64_MAX.
    sketch_to_nodes:
        Inverted index: sketch_hash -> list of node ids that contain that hash in their sketch.
    """
    tree: TreeData
    w: BucketableWeight
    node_keys: List[Tuple[Hashable, ...]]
    key_to_nodes: Dict[Hashable, List[int]]
    key_counts: Dict[Hashable, int]
    sketch_k: int = 0
    subtree_sketch: Optional[np.ndarray] = None
    sketch_to_nodes: Optional[Dict[int, List[int]]] = None


def preprocess_treedata(
    tree: TreeData,
    *,
    w: Optional[BucketableWeight] = None,
    max_nodes_per_key: Optional[int] = None,
    key_select_mode: str = "first",
    seed: int = 0,
    build_subtree_sketch: bool = False,
    sketch_k: int = 8,
    sketch_max_key_freq: int = 50,
    sketch_hash_salt: int = 0,
) -> PreprocessedTree:
    """
    Preprocess a TreeData object.

    Parameters
    ----------
    w:
        Bucketable weight. If None, uses EqualityBucketWeight (requires hashable labels).
    max_nodes_per_key:
        If set, truncate the stored node list for each key to at most this many nodes.
        (Full counts are still stored in key_counts.)
    build_subtree_sketch:
        If True, compute a subtree sketch per node (k-minhash over *rare* keys).
    sketch_k:
        Number of minhash values per node.
    sketch_max_key_freq:
        Only keys with key_counts[key] <= sketch_max_key_freq contribute to the sketch.
    sketch_hash_salt:
        Salt for stable_hash64 used in sketch.

    Returns
    -------
    PreprocessedTree
    """
    if w is None:
        w_use: BucketableWeight = EqualityBucketWeight()
    else:
        w_use = w

    # Validate bucketable interface using the root label as a sample.
    assert_bucketable_weight(w_use, sample_label=tree.label[0], mode_name="preprocess")

    rng = np.random.default_rng(seed)

    n = tree.n
    node_keys: List[Tuple[Hashable, ...]] = []
    key_counts: Dict[Hashable, int] = {}
    key_to_nodes_full: Dict[Hashable, List[int]] = {}

    for u in range(n):
        lab = tree.label[u]
        keys = list(w_use.blocking_keys(lab))
        # Ensure hashable.
        for k in keys:
            hash(k)  # may raise
        keys_t = tuple(keys)
        node_keys.append(keys_t)
        for k in keys_t:
            key_counts[k] = key_counts.get(k, 0) + 1
            key_to_nodes_full.setdefault(k, []).append(u)

    # Truncate stored lists if requested.
    if max_nodes_per_key is not None:
        key_to_nodes = {
            k: select_subset(vs, max_nodes_per_key, mode=key_select_mode, rng=rng)
            for k, vs in key_to_nodes_full.items()
        }
    else:
        key_to_nodes = key_to_nodes_full

    subtree_sketch: Optional[np.ndarray] = None
    sketch_to_nodes: Optional[Dict[int, List[int]]] = None

    if build_subtree_sketch:
        if sketch_k <= 0:
            raise ValueError("sketch_k must be positive when build_subtree_sketch=True")

        parent = np.asarray(tree.parent, dtype=np.int64)
        children: List[List[int]] = [[] for _ in range(n)]
        for u in range(1, n):
            p = int(parent[u])
            children[p].append(u)

        # Precompute eligible hashed keys for each node (only rare keys contribute).
        own_hashes: List[List[int]] = [[] for _ in range(n)]
        for u in range(n):
            vals = []
            for k in node_keys[u]:
                if key_counts.get(k, 0) <= sketch_max_key_freq:
                    vals.append(stable_hash64(k, salt=sketch_hash_salt))
            own_hashes[u] = vals

        sketch = np.full((n, sketch_k), _UINT64_MAX, dtype=np.uint64)

        # Postorder (children before parent) because parent<child => reverse order is postorder.
        for u in range(n - 1, -1, -1):
            merged: List[int] = []
            merged.extend(own_hashes[u])
            for ch in children[u]:
                ch_vals = sketch[ch]
                # filter padding
                merged.extend([int(x) for x in ch_vals if x != _UINT64_MAX])
            if merged:
                # k smallest hashes
                ksmall = sorted(merged)[:sketch_k]
                # pad
                if len(ksmall) < sketch_k:
                    ksmall.extend([int(_UINT64_MAX)] * (sketch_k - len(ksmall)))
                sketch[u, :] = np.asarray(ksmall, dtype=np.uint64)
            else:
                # no rare keys in subtree
                pass

        subtree_sketch = sketch

        idx: Dict[int, List[int]] = {}
        for u in range(n):
            for hv in subtree_sketch[u]:
                if hv == _UINT64_MAX:
                    break
                idx.setdefault(int(hv), []).append(u)
        sketch_to_nodes = idx

    return PreprocessedTree(
        tree=tree,
        w=w_use,
        node_keys=node_keys,
        key_to_nodes=key_to_nodes,
        key_counts=key_counts,
        sketch_k=(sketch_k if build_subtree_sketch else 0),
        subtree_sketch=subtree_sketch,
        sketch_to_nodes=sketch_to_nodes,
    )


def preprocess_igraph(
    G: Any,
    *,
    w: Optional[BucketableWeight] = None,
    phi_name: str = "label",
    order: str = "auto",
    ts_field: Optional[str] = None,
    strict_tree: bool = True,
    max_nodes_per_key: Optional[int] = None,
    key_select_mode: str = "first",
    seed: int = 0,
    build_subtree_sketch: bool = False,
    sketch_k: int = 8,
    sketch_max_key_freq: int = 50,
    sketch_hash_salt: int = 0,
) -> PreprocessedTree:
    """
    Convenience wrapper: convert igraph -> TreeData and preprocess.
    """
    tree = igraph_to_treedata(G, phi_name=phi_name, order=order, ts_field=ts_field, strict_tree=strict_tree)
    return preprocess_treedata(
        tree,
        w=w,
        max_nodes_per_key=max_nodes_per_key,
        key_select_mode=key_select_mode,
        seed=seed,
        build_subtree_sketch=build_subtree_sketch,
        sketch_k=sketch_k,
        sketch_max_key_freq=sketch_max_key_freq,
        sketch_hash_salt=sketch_hash_salt,
    )
