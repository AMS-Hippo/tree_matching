from __future__ import annotations

"""Notebook-oriented helpers for the tree embedding / clustering workflow."""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from .tree_cluster import TreeEmbeddingResult, expand_tree_list, _normalize_tokens


def pack_graphs_by_class(
    graphs: Sequence[Any],
    labels: Sequence[int],
    *,
    class_sequences: Optional[Sequence[Sequence[Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Pack flat ``(graphs, labels)`` data into the older nested ``some_trees`` structure.
    """
    lab = np.asarray(labels, dtype=int)
    out: List[Dict[str, Any]] = []
    for c in sorted(np.unique(lab).tolist()):
        idx = np.where(lab == c)[0].tolist()
        block: Dict[str, Any] = {"trees": [[graphs[i], None] for i in idx]}
        if class_sequences is not None and int(c) < len(class_sequences):
            block["a"] = list(class_sequences[int(c)])
        out.append(block)
    return out


def make_inverse_frequency_token_weights(
    trees: Sequence[Any],
    *,
    phi_name: str = "label",
    label_getter=None,
    power: float = 1.0,
    eps: float = 1e-8,
) -> Dict[Any, float]:
    """
    Empirical inverse-frequency token weights useful for rare-label emphasis.
    """
    flat, _ = expand_tree_list(list(trees))
    counts: Counter = Counter()

    for tree in flat:
        if hasattr(tree, "vs"):
            labels = tree.vs[phi_name]
        elif hasattr(tree, "label"):
            labels = tree.label
        else:
            raise TypeError("Unsupported tree type for token-weight estimation")
        for lab in labels:
            for tok in _normalize_tokens(lab, label_getter=label_getter):
                counts[tok] += 1

    total = float(sum(counts.values()))
    return {tok: 1.0 / max((cnt / total) ** power, eps) for tok, cnt in counts.items()}


def clustering_summary(gt_labels: Optional[Sequence[int]], pred_labels: Sequence[int]) -> Dict[str, float]:
    pred = np.asarray(pred_labels, dtype=int)
    summary = {
        "n_points": int(pred.shape[0]),
        "n_pred_clusters": int(len(set(pred.tolist())) - (1 if -1 in pred else 0)),
        "noise_fraction": float(np.mean(pred == -1)) if pred.size else 0.0,
    }
    if gt_labels is None:
        summary.update({"ari": np.nan, "nmi": np.nan})
    else:
        gt = np.asarray(gt_labels, dtype=int)
        summary.update(
            {
                "ari": float(adjusted_rand_score(gt, pred)),
                "nmi": float(normalized_mutual_info_score(gt, pred)),
            }
        )
    return summary


def summarize_clustering_result(
    result: TreeEmbeddingResult,
    *,
    gt_labels: Optional[Sequence[int]] = None,
) -> Dict[str, float]:
    gt = result.gt_labels if gt_labels is None else np.asarray(gt_labels, dtype=int)
    out = clustering_summary(gt, result.pred_labels)
    out.update(
        {
            "similarity_mode": result.similarity_mode,
            "n_landmarks": int(len(result.landmark_indices)) if result.landmark_indices is not None else 0,
            "n_candidate_pairs": int(len(result.candidate_pairs)) if result.candidate_pairs is not None else 0,
            "n_scored_pairs": int(len(result.matches)),
        }
    )
    return out


def result_overview_frame(
    results: Mapping[str, TreeEmbeddingResult],
    *,
    gt_labels: Optional[Sequence[int]] = None,
    runtime_sec: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """
    Assemble a compact comparison table for several clustering results.

    Parameters
    ----------
    results:
        Mapping from display name to ``TreeEmbeddingResult``.
    gt_labels:
        Optional ground-truth labels to use for *all* rows. This is handy when
        the results were fit on a plain tree list and therefore do not carry
        labels internally.
    runtime_sec:
        Optional mapping ``name -> runtime in seconds``.
    """
    rows = []
    for name, res in results.items():
        row = {"name": name}
        row.update(summarize_clustering_result(res, gt_labels=gt_labels))
        if runtime_sec is not None:
            row["runtime_sec"] = float(runtime_sec.get(name, np.nan))
        rows.append(row)

    df = pd.DataFrame(rows)
    preferred = [
        "name",
        "runtime_sec",
        "n_points",
        "n_pred_clusters",
        "noise_fraction",
        "ari",
        "nmi",
        "similarity_mode",
        "n_landmarks",
        "n_candidate_pairs",
        "n_scored_pairs",
    ]
    present = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in present]
    return df[present + rest]


def candidate_pair_diagnostics(
    candidate_pairs: Sequence[Tuple[int, int]],
    *,
    n_points: Optional[int] = None,
    gt_labels: Optional[Sequence[int]] = None,
    full_result: Optional[TreeEmbeddingResult] = None,
    exact_top_k: int = 10,
) -> pd.DataFrame:
    """
    Summarize how a candidate graph behaves.

    This is useful for understanding why ``similarity_mode='approx_knn'`` works
    well or poorly on a given dataset.
    """
    if full_result is None and n_points is None:
        raise ValueError("Provide either n_points=... or full_result=...")

    if full_result is not None:
        n = len(full_result.tree_list)
    else:
        n = int(n_points)

    pairs = sorted(
        {
            (int(i), int(j)) if int(i) < int(j) else (int(j), int(i))
            for (i, j) in candidate_pairs
            if int(i) != int(j)
        }
    )

    nbrs = [set() for _ in range(n)]
    for i, j in pairs:
        if 0 <= i < n and 0 <= j < n:
            nbrs[i].add(j)
            nbrs[j].add(i)

    deg = np.asarray([len(x) for x in nbrs], dtype=int)
    row: Dict[str, float] = {
        "n_points": int(n),
        "n_candidate_pairs": int(len(pairs)),
        "avg_degree": float(deg.mean()) if deg.size else 0.0,
        "min_degree": int(deg.min()) if deg.size else 0,
        "max_degree": int(deg.max()) if deg.size else 0,
        "isolated_fraction": float(np.mean(deg == 0)) if deg.size else 0.0,
    }

    if gt_labels is not None:
        gt = np.asarray(gt_labels, dtype=int)
        if gt.shape[0] != n:
            raise ValueError("gt_labels length does not match n_points.")
        if pairs:
            same = np.asarray([gt[i] == gt[j] for i, j in pairs], dtype=float)
            row["same_class_fraction"] = float(same.mean())
        else:
            row["same_class_fraction"] = np.nan

        total_same = 0
        for c in np.unique(gt):
            m = int(np.sum(gt == c))
            total_same += m * (m - 1) // 2
        same_captured = int(sum(gt[i] == gt[j] for i, j in pairs))
        row["same_class_pair_recall"] = float(same_captured / max(total_same, 1))

    if full_result is not None:
        S = full_result.normalized_scores
        if S is None:
            S = full_result.raw_scores
        if S is not None:
            S = np.asarray(S, dtype=float)
            tri = np.triu_indices(n, 1)
            all_scores = S[tri]
            cand_scores = np.asarray([S[i, j] for i, j in pairs], dtype=float) if pairs else np.zeros(0, dtype=float)
            row["mean_exact_sim_all_pairs"] = float(np.mean(all_scores)) if all_scores.size else np.nan
            row["mean_exact_sim_candidates"] = float(np.mean(cand_scores)) if cand_scores.size else np.nan

            k = max(1, min(int(exact_top_k), max(n - 1, 1)))
            recalls = []
            for i in range(n):
                order = np.argsort(-S[i])
                order = [int(j) for j in order if int(j) != i][:k]
                if not order:
                    continue
                hits = sum(int(j in nbrs[i]) for j in order)
                recalls.append(hits / len(order))
            row["topk_neighbor_recall"] = float(np.mean(recalls)) if recalls else np.nan

    return pd.DataFrame([row])


def plot_embedding_result(
    result: TreeEmbeddingResult,
    *,
    title: Optional[str] = None,
    use_pred_labels: bool = False,
    annotate: bool = False,
    palette: str = "deep",
    ax=None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns

    if ax is None:
        _fig, ax = plt.subplots(figsize=(6, 5))

    hue = result.pred_labels if use_pred_labels or result.gt_labels is None else result.gt_labels
    if hue is None:
        sns.scatterplot(x=result.embedding[:, 0], y=result.embedding[:, 1], ax=ax)
    else:
        sns.scatterplot(x=result.embedding[:, 0], y=result.embedding[:, 1], hue=hue, palette=palette, ax=ax)
        ax.legend([], [], frameon=False)

    if annotate:
        for i, (x, y) in enumerate(result.embedding[:, :2]):
            ax.text(float(x), float(y), str(i), fontsize=8)

    if title is None:
        title = f"{result.similarity_mode} embedding"
    ax.set_title(title)
    return ax


def top_matches_table(
    result: TreeEmbeddingResult,
    *,
    top_k: int = 10,
) -> pd.DataFrame:
    if result.raw_scores is None or result.raw_scores.shape[0] != result.raw_scores.shape[1]:
        raise ValueError("top_matches_table currently expects a square raw score matrix.")
    S = np.asarray(result.raw_scores, dtype=float).copy()
    np.fill_diagonal(S, -np.inf)
    rows = []
    n = S.shape[0]
    for i in range(n):
        order = np.argsort(-S[i])[:top_k]
        for rank, j in enumerate(order, start=1):
            if not np.isfinite(S[i, j]):
                continue
            rows.append({"tree": i, "rank": rank, "nbr": int(j), "score": float(S[i, j])})
    return pd.DataFrame(rows)


__all__ = [
    "pack_graphs_by_class",
    "make_inverse_frequency_token_weights",
    "clustering_summary",
    "summarize_clustering_result",
    "result_overview_frame",
    "candidate_pair_diagnostics",
    "plot_embedding_result",
    "top_matches_table",
]
