from __future__ import annotations

"""Metrics for exemplar-sequence inference."""

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from .bags import SequenceBag
from .medoid import edit_distance



def normalized_edit_distance(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> float:
    denom = max(len(seq_a), len(seq_b), 1)
    return float(edit_distance(seq_a, seq_b)) / float(denom)



def weighted_mean_distance(exemplar: Sequence[Any], bag: SequenceBag) -> float:
    if bag.n_unique == 0:
        return 0.0
    dists = np.asarray([edit_distance(exemplar, seq) for seq in bag.sequences], dtype=float)
    weights = np.asarray(bag.weights, dtype=float)
    return float(np.average(dists, weights=np.maximum(weights, 1e-12)))



def weighted_mean_normalized_distance(exemplar: Sequence[Any], bag: SequenceBag) -> float:
    if bag.n_unique == 0:
        return 0.0
    dists = np.asarray([normalized_edit_distance(exemplar, seq) for seq in bag.sequences], dtype=float)
    weights = np.asarray(bag.weights, dtype=float)
    return float(np.average(dists, weights=np.maximum(weights, 1e-12)))



def separation_gap(
    exemplar: Sequence[Any],
    own_bag: SequenceBag,
    other_bags: Mapping[int, SequenceBag],
) -> float:
    own = weighted_mean_distance(exemplar, own_bag)
    other_vals = [weighted_mean_distance(exemplar, bag) for bag in other_bags.values() if bag.n_unique > 0]
    if not other_vals:
        return float("nan")
    return float(np.mean(other_vals) - own)



def evaluate_exemplar(
    exemplar: Sequence[Any],
    *,
    truth: Optional[Sequence[Any]] = None,
    bag: Optional[SequenceBag] = None,
    other_bags: Optional[Mapping[int, SequenceBag]] = None,
) -> Dict[str, float]:
    row: Dict[str, float] = {
        "length": float(len(exemplar)),
    }
    if truth is not None:
        row.update(
            {
                "edit_to_truth": float(edit_distance(exemplar, truth)),
                "norm_edit_to_truth": float(normalized_edit_distance(exemplar, truth)),
                "length_ratio_to_truth": float(len(exemplar)) / float(max(len(truth), 1)),
            }
        )
    else:
        row.update(
            {
                "edit_to_truth": float("nan"),
                "norm_edit_to_truth": float("nan"),
                "length_ratio_to_truth": float("nan"),
            }
        )
    if bag is not None:
        row.update(
            {
                "within_mean_edit": weighted_mean_distance(exemplar, bag),
                "within_mean_norm_edit": weighted_mean_normalized_distance(exemplar, bag),
            }
        )
    else:
        row.update({"within_mean_edit": float("nan"), "within_mean_norm_edit": float("nan")})
    if bag is not None and other_bags is not None:
        row["separation_gap"] = separation_gap(exemplar, bag, other_bags)
    else:
        row["separation_gap"] = float("nan")
    return row



def bag_truth_diagnostics(
    bag: SequenceBag,
    truth: Optional[Sequence[Any]],
) -> Dict[str, float]:
    row: Dict[str, float] = {
        'n_unique_obs': float(bag.n_unique),
        'n_obs': float(bag.n_observations),
        'duplicate_fraction': float(1.0 - (bag.n_unique / max(bag.n_observations, 1))),
        'bag_mean_length': float(bag.mean_length()),
        'bag_median_length': float(bag.median_length()),
        'bag_min_length': float(min((len(seq) for seq in bag.sequences), default=0)),
        'bag_max_length': float(max((len(seq) for seq in bag.sequences), default=0)),
    }
    if truth is None:
        row.update({
            'truth_present_in_bag': float('nan'),
            'best_observed_edit_to_truth': float('nan'),
            'best_observed_norm_edit': float('nan'),
        })
        return row
    best_edit = min((edit_distance(seq, truth) for seq in bag.sequences), default=float('inf'))
    best_norm = min((normalized_edit_distance(seq, truth) for seq in bag.sequences), default=float('inf'))
    truth_present = any(list(seq) == list(truth) for seq in bag.sequences)
    row.update({
        'truth_present_in_bag': float(bool(truth_present)),
        'best_observed_edit_to_truth': float(best_edit),
        'best_observed_norm_edit': float(best_norm),
    })
    return row
