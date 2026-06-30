
"""
planted_path_sampler.py

A small, modular implementation of the synthetic "planted labelled path" generator
from Section 4 of *The Needle Is a Thread: Finding Planted Paths in Noisy Process Trees*.

This module implements:
- Galton–Watson (Poisson offspring) tree samplers with optional truncation by max_nodes,
  and two expansion orders ("bfs" and "dfs").
- Random root-to-leaf planted paths (leaf chosen uniformly at random).
- Algorithm 3 (label planting) and Algorithm 4 (toy model with M classes produced by
  permuting a base planted sequence).

Primary outputs match the user request:
(A) list[igraph.Graph] of N labelled graphs (vertex attributes: "label", "is_planted"),
(B) list[int] of class labels for each graph,
(C) list[list[str]] of planted sequences, one per class.

Notes
-----
- The paper's Algorithm 3 samples a subset S2 of indices with probability proportional to
  a product of weights. This module implements *exactly* the distribution
      P(S2) ∝ ∏_{i∈S2} w_i
  via a simple dynamic-programming sampler.

- Truncating a Galton–Watson process to a fixed max_nodes necessarily changes the exact
  GW distribution; the sampler here follows the requested behavior ("prune to max size"),
  and uses rejection sampling if the tree is too small.

Dependencies: numpy, python-igraph
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import igraph as ig


# -----------------------------
# Utilities: alphabet + π
# -----------------------------

def make_nonsense_dictionary(k: int, prefix: str = "sym") -> List[str]:
    """Create a simple 'nonsense' dictionary: [f'{prefix}{i}', ...]."""
    if k <= 0:
        raise ValueError("alphabet size k must be positive")
    return [f"{prefix}{i}" for i in range(k)]


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("probabilities must be 1D")
    if np.any(p < 0):
        raise ValueError("probabilities must be nonnegative")
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0:
        raise ValueError("probabilities must sum to a positive finite number")
    return p / s


def build_pi(
    alphabet: Sequence[str],
    *,
    pi: Optional[Union[Sequence[float], Dict[str, float]]] = None,
    dirichlet_alpha: Optional[Union[float, Sequence[float]]] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Build a distribution π over a given alphabet.

    Options:
      - pi: explicit probabilities (list/np.array) aligned with alphabet,
            OR dict {symbol: prob} (missing symbols get prob 0).
      - dirichlet_alpha: if provided (and pi is None), sample π ~ Dirichlet(alpha).
      - default: uniform.

    Returns: np.ndarray shape (K,) summing to 1.
    """
    K = len(alphabet)
    if K == 0:
        raise ValueError("alphabet must be non-empty")

    if pi is not None and dirichlet_alpha is not None:
        raise ValueError("Specify at most one of pi or dirichlet_alpha.")

    if pi is None and dirichlet_alpha is None:
        return np.full(K, 1.0 / K, dtype=float)

    if pi is not None:
        if isinstance(pi, dict):
            arr = np.array([float(pi.get(sym, 0.0)) for sym in alphabet], dtype=float)
            return normalize_probs(arr)
        else:
            arr = np.array(list(pi), dtype=float)
            if arr.shape != (K,):
                raise ValueError(f"pi must have length {K}, got {arr.shape}")
            return normalize_probs(arr)

    # Dirichlet
    if rng is None:
        rng = np.random.default_rng()
    if isinstance(dirichlet_alpha, (int, float)):
        alpha = np.full(K, float(dirichlet_alpha), dtype=float)
    else:
        alpha = np.array(list(dirichlet_alpha), dtype=float)
        if alpha.shape != (K,):
            raise ValueError(f"dirichlet_alpha must have length {K}, got {alpha.shape}")
    if np.any(alpha <= 0):
        raise ValueError("Dirichlet concentration parameters must be positive.")
    return rng.dirichlet(alpha)


def sample_symbols(
    alphabet: Sequence[str], pi: np.ndarray, size: int, rng: np.random.Generator
) -> List[str]:
    """Sample 'size' iid symbols from π over alphabet."""
    idx = rng.choice(len(alphabet), size=size, replace=True, p=pi)
    return [alphabet[int(i)] for i in idx]


# -----------------------------------------
# Exact weighted subset sampler (product)
# -----------------------------------------

def sample_subset_product_weights(
    weights: Sequence[float],
    subset_size: int,
    rng: np.random.Generator,
) -> List[int]:
    """
    Sample a subset S of {0,...,K-1} with |S| = subset_size and probability
        P(S) ∝ ∏_{i∈S} weights[i].

    This is an *exact* sampler using DP:
      DP[i][j] = sum_{S ⊆ {0,...,i-1}, |S|=j} ∏_{t∈S} w_t
    and a backward sampling pass.

    Returns: sorted list of indices.
    """
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1:
        raise ValueError("weights must be 1D")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    K = w.size
    r = int(subset_size)
    if r < 0 or r > K:
        raise ValueError(f"subset_size must be between 0 and {K}")
    if r == 0:
        return []

    # DP table (K+1) x (r+1)
    DP = np.zeros((K + 1, r + 1), dtype=float)
    DP[0, 0] = 1.0
    for i in range(1, K + 1):
        DP[i, 0] = 1.0
        wi = float(w[i - 1])
        maxj = min(i, r)
        for j in range(1, maxj + 1):
            DP[i, j] = DP[i - 1, j] + wi * DP[i - 1, j - 1]

    if DP[K, r] <= 0 or not np.isfinite(DP[K, r]):
        raise ValueError("Degenerate weights: cannot sample a subset of requested size.")

    # Backward sample
    chosen: List[int] = []
    j = r
    for i in range(K, 0, -1):
        if j == 0:
            break
        wi = float(w[i - 1])
        denom = DP[i, j]
        # Probability that item i-1 is included, conditioned on needing j items from first i:
        numer = wi * DP[i - 1, j - 1]
        p_include = 0.0 if denom <= 0 else numer / denom
        if p_include < 0:
            p_include = 0.0
        if p_include > 1:
            p_include = 1.0
        if rng.random() < p_include:
            chosen.append(i - 1)
            j -= 1
        # else: excluded
    chosen.sort()
    if len(chosen) != r:
        # This should not happen; guard for floating error.
        raise RuntimeError("Internal error: sampled subset has wrong size.")
    return chosen


# -----------------------------
# Tree + path samplers
# -----------------------------

class TreePathSampler:
    """Abstract interface for sampling a (directed) tree and a planted path Γ inside it."""
    def sample(self, rng: np.random.Generator) -> Tuple[ig.Graph, List[int]]:
        raise NotImplementedError


@dataclass(frozen=True)
class GaltonWatsonTreeSampler(TreePathSampler):
    """
    Galton–Watson tree sampler with Poisson(λ) offspring, truncated to max_nodes.

    Parameters
    ----------
    max_depth:
        Maximum depth (root has depth 0). Nodes at depth >= max_depth are not expanded.
    lam:
        Poisson mean λ > 0.
    max_nodes:
        Hard cap on number of nodes. If the GW expansion would exceed this, it is pruned.
    traversal:
        'bfs' expands nodes in FIFO order; 'dfs' expands in LIFO order.
        (These differ only because we prune at max_nodes.)
    min_nodes:
        Rejection threshold. If generated tree has fewer than min_nodes nodes, resample.
        If None, defaults to max_nodes (i.e., "fill to max size" if possible).
    require_full_depth:
        If True, reject unless the tree reaches depth max_depth (i.e., some node has depth == max_depth).
        This mirrors the paper's rejection scheme for GW(N, λ). 
    max_tries:
        Max number of attempts before raising an error (to avoid infinite loops).
    """
    max_depth: int
    lam: float
    max_nodes: int
    traversal: str = "bfs"
    min_nodes: Optional[int] = None
    require_full_depth: bool = True
    max_tries: int = 2000

    def sample(self, rng: np.random.Generator) -> Tuple[ig.Graph, List[int]]:
        if self.max_depth < 0:
            raise ValueError("max_depth must be >= 0")
        if self.lam <= 0:
            raise ValueError("lam must be > 0")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be > 0")
        if self.traversal not in {"bfs", "dfs"}:
            raise ValueError("traversal must be 'bfs' or 'dfs'")
        min_nodes = self.max_nodes if self.min_nodes is None else int(self.min_nodes)
        if min_nodes <= 0 or min_nodes > self.max_nodes:
            raise ValueError("min_nodes must be in [1, max_nodes]")

        for _ in range(int(self.max_tries)):
            parents = [-1]  # root
            depths = [0]
            # frontier as list to support bfs/dfs cheaply
            frontier: List[int] = [0]
            head = 0  # for bfs

            while True:
                if self.traversal == "bfs":
                    if head >= len(frontier):
                        break
                    v = frontier[head]
                    head += 1
                else:
                    if not frontier:
                        break
                    v = frontier.pop()

                dv = depths[v]
                if dv >= self.max_depth:
                    continue

                # Sample offspring and prune to capacity.
                k = int(rng.poisson(self.lam))
                remaining = self.max_nodes - len(parents)
                if remaining <= 0:
                    break
                if k > remaining:
                    k = remaining

                # Add children
                for _c in range(k):
                    child = len(parents)
                    parents.append(v)
                    depths.append(dv + 1)
                    frontier.append(child)

                if len(parents) >= self.max_nodes:
                    break

            n = len(parents)
            if n < min_nodes:
                continue
            if self.require_full_depth and (max(depths) < self.max_depth):
                continue

            edges = [(parents[v], v) for v in range(1, n)]
            g = ig.Graph(n=n, edges=edges, directed=True)

            # Sample planted path Γ as a deep root-to-leaf path:
            # choose uniformly among the deepest leaves (depth = max(depths)).
            leaves = [v.index for v in g.vs.select(_outdegree_eq=0)]
            if not leaves:
                continue
            
            max_d = max(depths)
            deep_leaves = [v for v in leaves if depths[v] == max_d]
            leaf = int(rng.choice(deep_leaves if deep_leaves else leaves))
            
            path = _path_to_root_from_parents(leaf, parents)
            path.reverse()
            return g, path

        raise RuntimeError(
            "Failed to sample a Galton–Watson tree meeting constraints after max_tries. "
            "Try increasing lam, relaxing require_full_depth/min_nodes, or increasing max_tries."
        )



def make_default_tree_samplers(
    *,
    max_depth: int = 12,
    lam: float = 1.8,
    max_nodes: int = 300,
    min_nodes: Optional[int] = None,
    require_full_depth: bool = True,
    max_tries: int = 2000,
) -> Dict[str, GaltonWatsonTreeSampler]:
    """Convenience constructor returning two default GW samplers: BFS- and DFS-expanded."""
    return {
        "gw_bfs": GaltonWatsonTreeSampler(
            max_depth=max_depth,
            lam=lam,
            max_nodes=max_nodes,
            traversal="bfs",
            min_nodes=min_nodes if min_nodes is not None else max_nodes,
            require_full_depth=require_full_depth,
            max_tries=max_tries,
        ),
        "gw_dfs": GaltonWatsonTreeSampler(
            max_depth=max_depth,
            lam=lam,
            max_nodes=max_nodes,
            traversal="dfs",
            min_nodes=min_nodes if min_nodes is not None else max_nodes,
            require_full_depth=require_full_depth,
            max_tries=max_tries,
        ),
    }

def _path_to_root_from_parents(v: int, parents: Sequence[int]) -> List[int]:
    """Return list [v, parent(v), parent(parent(v)), ..., root] (reverse direction)."""
    out = []
    cur = int(v)
    while cur != -1:
        out.append(cur)
        cur = int(parents[cur])
    return out


# -----------------------------
# Algorithm 3: plant labels
# -----------------------------

def label_tree_with_planted_path(
    g: ig.Graph,
    path: Sequence[int],
    *,
    alphabet: Sequence[str],
    pi: np.ndarray,
    planted_sequence: Sequence[str],
    p_obs: float,
    rate_fn: Optional[Union[Callable[[str], float], Dict[str, float]]] = None,
    rng: Optional[np.random.Generator] = None,
    label_attr: str = "label",
    planted_attr: str = "is_planted",
) -> ig.Graph:
    """
    Label a tree with a planted path according to Algorithm 3 (Randomly Labeling Trees from Template).

    Inputs correspond to the paper's Algorithm 3 and surrounding text: path Γ is a connected
    root-to-leaf path; labels off the selected template positions are iid π; the number of
    observed template positions is Bin(|Γ|, p). (Algorithm 3, Section 4.1) 

    Returns the same igraph.Graph object (mutated) for convenience.
    """
    if rng is None:
        rng = np.random.default_rng()

    if not (0.0 <= p_obs <= 1.0):
        raise ValueError("p_obs must be in [0,1]")

    n = g.vcount()
    if n == 0:
        raise ValueError("graph must have at least one vertex")
    if len(path) == 0:
        raise ValueError("path must be non-empty")
    if any((v < 0 or v >= n) for v in path):
        raise ValueError("path contains invalid vertex indices")

    K = len(planted_sequence)
    ell = len(path)

    # Step 1: N' ~ Bin(ell, p), N = min(N', K)
    N_prime = int(rng.binomial(ell, p_obs))
    N = min(N_prime, K)

    # Step 2: sample S1 ⊆ {0,...,ell-1}, |S1|=N uniformly.
    if N > 0:
        S1 = sorted(rng.choice(ell, size=N, replace=False).tolist())

        # Sample S2 ⊆ {0,...,K-1}, |S2|=N with prob ∝ ∏ r(a_i) over chosen indices.
        if rate_fn is None:
            weights = [1.0] * K
        elif callable(rate_fn):
            weights = [float(rate_fn(str(planted_sequence[i]))) for i in range(K)]
        else:
            # dict mapping symbol -> rate
            d = dict(rate_fn)
            weights = [float(d.get(str(planted_sequence[i]), 1.0)) for i in range(K)]
        if any(w < 0 for w in weights):
            raise ValueError("rate_fn produced negative weights")

        S2 = sample_subset_product_weights(weights, N, rng)
    else:
        S1, S2 = [], []

    # Step 6-7: label everything iid π, then overwrite planted template positions.
    labels = sample_symbols(alphabet, pi, size=n, rng=rng)

    # Step 3-5: overwrite N planted template positions
    for i in range(N):
        v = int(path[S1[i]])
        labels[v] = str(planted_sequence[S2[i]])

    is_planted = [0] * n
    for v in path:
        is_planted[int(v)] = 1

    g.vs[label_attr] = labels
    g.vs[planted_attr] = is_planted
    return g


# -----------------------------
# Algorithm 4: toy model
# -----------------------------

@dataclass(frozen=True)
class ToyModelConfig:
    """
    Configuration for Algorithm 4 ("Sampling from Toy Model"). 

    Only a subset of paper parameters are included here; the rest (e.g. µ) are
    supplied via the tree_path_sampler argument to sample_toy_model().
    """
    n_classes: int
    n_per_class: Union[int, Sequence[int]]
    planted_seq_len: int
    alphabet_size: int = 10
    p_obs: float = 0.9
    rate_fn: Optional[Union[Callable[[str], float], Dict[str, float]]] = None
    # π options:
    pi: Optional[Union[Sequence[float], Dict[str, float]]] = None
    dirichlet_alpha: Optional[Union[float, Sequence[float]]] = None
    # optional override:
    base_sequence: Optional[Sequence[str]] = None
    alphabet: Optional[Sequence[str]] = None
    seed: Optional[int] = None


def sample_toy_model(
    cfg: ToyModelConfig,
    *,
    tree_path_sampler: Optional[TreePathSampler] = None,
) -> Tuple[List[ig.Graph], List[int], List[List[str]]]:
    """
    Sample data from Algorithm 4 ("Sampling from Toy Model"), returning:
      (A) graphs,
      (B) class labels,
      (C) planted sequences per class.

    In the paper's basic model, each class corresponds to a permutation of a single base
    sequence.  
    """
    rng = np.random.default_rng(cfg.seed)

    if cfg.n_classes <= 0:
        raise ValueError("n_classes must be positive")
    if cfg.planted_seq_len <= 0:
        raise ValueError("planted_seq_len must be positive")
    if cfg.alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive")
    if not (0.0 <= cfg.p_obs <= 1.0):
        raise ValueError("p_obs must be in [0,1]")

    if tree_path_sampler is None:
        # Reasonable defaults consistent with the paper's "bushy GW" examples.
        tree_path_sampler = GaltonWatsonTreeSampler(
            max_depth=12,
            lam=1.8,
            max_nodes=300,
            traversal="bfs",
            min_nodes=300,
            require_full_depth=True,
            max_tries=50000, # Could make this smaller...
        )

    if cfg.alphabet is None:
        alphabet = make_nonsense_dictionary(cfg.alphabet_size)
    else:
        alphabet = list(cfg.alphabet)
        if len(alphabet) != cfg.alphabet_size:
            raise ValueError("If cfg.alphabet is provided, it must have length alphabet_size.")

    pi = build_pi(
        alphabet,
        pi=cfg.pi,
        dirichlet_alpha=cfg.dirichlet_alpha,
        rng=rng,
    )

    if cfg.base_sequence is None:
        base_seq = sample_symbols(alphabet, pi, size=cfg.planted_seq_len, rng=rng)
    else:
        base_seq = [str(x) for x in cfg.base_sequence]
        if len(base_seq) != cfg.planted_seq_len:
            raise ValueError("base_sequence must have length planted_seq_len")

    # Number of observations per class
    if isinstance(cfg.n_per_class, int):
        n_per = [int(cfg.n_per_class)] * cfg.n_classes
    else:
        n_per = [int(x) for x in cfg.n_per_class]
        if len(n_per) != cfg.n_classes:
            raise ValueError("n_per_class must be an int or a length-n_classes sequence")
    if any(x < 0 for x in n_per):
        raise ValueError("n_per_class values must be nonnegative")

    # Sample permutations and build per-class planted sequences
    class_sequences: List[List[str]] = []
    permutations: List[np.ndarray] = []
    for _i in range(cfg.n_classes):
        perm = rng.permutation(cfg.planted_seq_len)
        permutations.append(perm)
        class_sequences.append([base_seq[int(t)] for t in perm])

    graphs: List[ig.Graph] = []
    labels: List[int] = []

    for class_id in range(cfg.n_classes):
        planted_seq = class_sequences[class_id]
        for _ in range(n_per[class_id]):
            g, path = tree_path_sampler.sample(rng)
            label_tree_with_planted_path(
                g,
                path,
                alphabet=alphabet,
                pi=pi,
                planted_sequence=planted_seq,
                p_obs=cfg.p_obs,
                rate_fn=cfg.rate_fn,
                rng=rng,
                label_attr="label",
                planted_attr="is_planted",
            )
            graphs.append(g)
            labels.append(class_id)

    return graphs, labels, class_sequences


# -----------------------------
# Quick smoke-test / demo
# -----------------------------

def _demo() -> None:
    cfg = ToyModelConfig(
        n_classes=3,
        n_per_class=[5, 5, 5],
        planted_seq_len=10,
        alphabet_size=8,
        p_obs=0.85,
        dirichlet_alpha=1.0,
        seed=123,
    )
    sampler = GaltonWatsonTreeSampler(
        max_depth=10, lam=1.7, max_nodes=120, traversal="bfs", min_nodes=120, require_full_depth=True
    )
    Gs, y, class_seqs = sample_toy_model(cfg, tree_path_sampler=sampler)
    print(f"N graphs = {len(Gs)}")
    print(f"Class labels counts = {np.bincount(np.array(y, dtype=int))}")
    print(f"Example class planted seq[0] = {class_seqs[0]}")
    # Show some sanity checks on attributes
    g0 = Gs[0]
    assert "label" in g0.vs.attributes()
    assert "is_planted" in g0.vs.attributes()
    print(f"Example graph 0: v={g0.vcount()}, e={g0.ecount()}, planted={sum(g0.vs['is_planted'])}")


if __name__ == "__main__":
    _demo()
