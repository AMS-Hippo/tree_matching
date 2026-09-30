"""Common diagnostics for tree-path matching algorithms.

Diagnostics are opt-in.  Low-level alignment functions accept an optional
``MatchDiagnostics`` instance, and high-level matcher classes expose the most
recent record as ``last_diagnostics_`` when constructed with
``collect_diagnostics=True``.

The counters intentionally describe algorithmic work rather than Python
implementation details.  In particular, ``node_pair_score_evaluations`` counts
logical evaluations of a node-pair score during candidate discovery or the main
search; it does not include a score re-read used only to filter a traceback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Dict, Iterable, Optional


@dataclass
class MatchDiagnostics:
    """Standardized timing and work counters for one matching call.

    Counts that do not apply to an algorithm are left as ``None``.  Beam-state
    counts are cumulative over processed layers, except
    ``peak_frontier_size``, which is a maximum.  For restart or symmetric
    wrappers, cumulative counters measure the total work over all executed
    runs, while ``result_score`` and ``result_length`` describe the selected
    output.
    """

    algorithm: str = ""
    n_g: int = 0
    n_h: int = 0

    # Wall-clock phases, in seconds.
    input_preprocessing_seconds: float = 0.0
    preprocessing_seconds: float = 0.0
    candidate_generation_seconds: float = 0.0
    search_seconds: float = 0.0
    traceback_seconds: float = 0.0
    total_seconds: float = 0.0

    # Pair/cell work.
    node_pair_score_evaluations: int = 0
    candidate_pairs_generated: Optional[int] = None
    positive_candidate_pairs: Optional[int] = None
    dp_cells_computed: Optional[int] = None

    # Search-state work.  These are principally meaningful for beam methods.
    states_generated: Optional[int] = None
    states_after_endpoint_pruning: Optional[int] = None
    states_retained: Optional[int] = None
    peak_frontier_size: Optional[int] = None
    layers_processed: Optional[int] = None

    restarts: int = 1
    directions: int = 1

    result_score: Optional[float] = None
    result_length: Optional[int] = None

    # Algorithm-specific counters that are useful but not universal.
    extra: Dict[str, Any] = field(default_factory=dict)

    def reset(self, *, algorithm: str, n_g: int, n_h: int) -> None:
        """Reset this instance for a fresh alignment call."""
        fresh = MatchDiagnostics(algorithm=str(algorithm), n_g=int(n_g), n_h=int(n_h))
        self.__dict__.clear()
        self.__dict__.update(fresh.__dict__)

    def copy(self) -> "MatchDiagnostics":
        """Return an independent shallow copy, including a copied ``extra`` dict."""
        return replace(self, extra=dict(self.extra))

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly dictionary."""
        out = asdict(self)
        out["extra"] = dict(self.extra)
        return out

    @property
    def accounted_seconds(self) -> float:
        """Sum of the named phase timings, excluding ``total_seconds``."""
        return float(
            self.input_preprocessing_seconds
            + self.preprocessing_seconds
            + self.candidate_generation_seconds
            + self.search_seconds
            + self.traceback_seconds
        )

    def accumulate(self, other: "MatchDiagnostics") -> None:
        """Add total work from another run into this record.

        This is used by restart and symmetric wrappers.  Optional cumulative
        counters are summed whenever they are present in either record; maxima
        use the largest available value.  Result fields are deliberately not
        changed because the caller must set them from the selected run.
        """

        for name in (
            "input_preprocessing_seconds",
            "preprocessing_seconds",
            "candidate_generation_seconds",
            "search_seconds",
            "traceback_seconds",
            "total_seconds",
            "node_pair_score_evaluations",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))

        for name in (
            "candidate_pairs_generated",
            "positive_candidate_pairs",
            "dp_cells_computed",
            "states_generated",
            "states_after_endpoint_pruning",
            "states_retained",
            "layers_processed",
        ):
            left = getattr(self, name)
            right = getattr(other, name)
            if left is None and right is None:
                continue
            setattr(self, name, int(left or 0) + int(right or 0))

        left_peak = self.peak_frontier_size
        right_peak = other.peak_frontier_size
        if left_peak is not None or right_peak is not None:
            self.peak_frontier_size = max(int(left_peak or 0), int(right_peak or 0))


def aggregate_diagnostics(
    children: Iterable[MatchDiagnostics],
    *,
    algorithm: str,
    n_g: int,
    n_h: int,
) -> MatchDiagnostics:
    """Aggregate work counters from several complete alignment runs."""
    out = MatchDiagnostics(algorithm=algorithm, n_g=n_g, n_h=n_h)
    for child in children:
        out.accumulate(child)
    return out


__all__ = ["MatchDiagnostics", "aggregate_diagnostics"]
