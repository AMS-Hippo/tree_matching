from __future__ import annotations

"""Notebook helpers for cluster exemplar experiments."""

from typing import Any, Dict, Mapping, Sequence

import pandas as pd

from .bags import SequenceBag



def bags_overview_frame(bags: Mapping[int, SequenceBag]) -> pd.DataFrame:
    rows = []
    for c, bag in bags.items():
        row = {"cluster": int(c)}
        row.update(bag.summary())
        rows.append(row)
    return pd.DataFrame(rows)



def exemplars_frame(exemplars: Mapping[int, Sequence[Any]]) -> pd.DataFrame:
    rows = []
    for c, seq in exemplars.items():
        rows.append({"cluster": int(c), "exemplar": " ".join(map(str, seq)), "length": len(seq)})
    return pd.DataFrame(rows)
