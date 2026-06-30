"""
bucketable_weight.py

Maintainable "bucketable / indexable weight" utilities.

Why this exists
---------------
Some matchers (e.g. sparse-candidate DP / bucketed beam search) want to avoid
evaluating w(a,b) for *all* label pairs. Instead we generate a candidate set by
"blocking" on cheap keys:

    blocking_keys(label) -> iterable[hashable]

Two labels are plausible matches if they share at least one key.

This module provides:
- A generic wrapper `BucketableWeight` that adds `blocking_keys` to any score function.
- A small set of built-in keyers (exact equality, token overlap).
- Runtime checks used by matchers.

Recommended usage
-----------------
Token-overlap blocking (default):

    w_bucket = make_bucketable_weight(w_plain, scheme="token_overlap", min_token_overlap=1)
    # or simply:
    w_bucket = make_bucketable_weight(w_plain)

Exact-equality blocking:

    w_bucket = make_bucketable_weight(w_plain, scheme="exact")

Custom blocking:

    w_bucket = make_bucketable_weight(w_plain, keyer=my_keyer_callable)

Notes
-----
- For min_token_overlap > 1, the token-overlap keyer emits k-token-combination keys.
  This is exact for the overlap threshold, but can grow combinatorially; `max_keys`
  caps worst cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)
import itertools
import re


# -----------------------------------------------------------------------------
# Interfaces (for type-checkers + runtime validation)
# -----------------------------------------------------------------------------

@runtime_checkable
class BucketableWeightLike(Protocol):
    """Anything callable with a blocking_keys(label) method."""
    def __call__(self, a: Any, b: Any) -> float: ...
    def blocking_keys(self, label: Any) -> Iterable[Hashable]: ...


BlockingKeyer = Callable[[Any], Iterable[Hashable]]


# -----------------------------------------------------------------------------
# Tokenization + built-in keyers
# -----------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def default_tokenize(x: str) -> List[str]:
    """Default tokenizer: lowercase alphanumeric runs."""
    return _TOKEN_RE.findall(x.lower())


@dataclass(frozen=True)
class ExactKeyer:
    """blocking_keys(label) = (label,) with a hashability check."""
    def __call__(self, label: Any) -> Iterable[Hashable]:
        try:
            hash(label)
        except Exception as e:
            raise TypeError(
                "ExactKeyer requires labels to be hashable. "
                "If your labels are unhashable (e.g. dict/list), use a different keyer "
                "(e.g. TokenOverlapKeyer) or define your own."
            ) from e
        return (label,)


@dataclass(frozen=True)
class TokenOverlapKeyer:
    """
    Token-overlap blocking.

    min_token_overlap == 1:
        keys(label) = ["tok:<token>", ...]
    min_token_overlap == k > 1:
        keys(label) includes k-token-combination keys "tok{k}:t1|...|tk"
        Two labels share >=k tokens  <=>  they share at least one such key.

    Labels are coerced to str().
    """
    min_token_overlap: int = 1
    tokenizer: Callable[[str], Sequence[str]] = default_tokenize
    max_keys: int = 2048
    prefix: str = "tok"

    def __call__(self, label: Any) -> Iterable[Hashable]:
        k = int(self.min_token_overlap)
        if k <= 0:
            raise ValueError("min_token_overlap must be >= 1")

        toks = sorted(set(self.tokenizer(str(label))))

        if k == 1:
            return [f"{self.prefix}:{t}" for t in toks]

        if len(toks) < k:
            return []

        out: List[str] = []
        for i, comb in enumerate(itertools.combinations(toks, k)):
            if i >= self.max_keys:
                break
            out.append(f"{self.prefix}{k}:" + "|".join(comb))
        return out


# -----------------------------------------------------------------------------
# The actual "bucketable weight" wrapper
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BucketableWeight:
    """
    Generic wrapper that turns any score function into a bucketable weight:

      score(a,b) = w(a,b)
      blocking_keys(label) = keyer(label)

    This is the only "weight object" class you should need for most use-cases.
    """
    w: Callable[[Any, Any], float]
    keyer: BlockingKeyer

    def __call__(self, a: Any, b: Any) -> float:
        return float(self.w(a, b))

    def blocking_keys(self, label: Any) -> Iterable[Hashable]:
        return self.keyer(label)


# -----------------------------------------------------------------------------
# Factory
# -----------------------------------------------------------------------------

def make_bucketable_weight(
    w: Callable[[Any, Any], float],
    scheme: Union[str, int] = "token_overlap",
    *,
    keyer: Optional[BlockingKeyer] = None,
    # token_overlap options
    min_token_overlap: int = 1,
    tokenizer: Callable[[str], Sequence[str]] = default_tokenize,
    max_keys: int = 2048,
) -> BucketableWeight:
    """
    Create a BucketableWeight from a plain scoring function.

    Parameters
    ----------
    w:
        scoring function w(a,b) -> float
    scheme:
        "token_overlap" (default) or "exact".
        For backwards compatibility with earlier notebooks:
          scheme==3 is treated as "token_overlap".
    keyer:
        If provided, overrides scheme and is used as blocking_keys directly.
    """
    if keyer is not None:
        return BucketableWeight(w=w, keyer=keyer)

    # Back-compat: allow scheme=3 to mean token overlap.
    if isinstance(scheme, int):
        if scheme == 3:
            scheme = "token_overlap"
        else:
            raise ValueError(f"Unknown integer scheme={scheme}. Supported: 3 (token_overlap).")

    scheme_norm = str(scheme).lower().strip()
    if scheme_norm in {"token_overlap", "tokens", "token"}:
        return BucketableWeight(
            w=w,
            keyer=TokenOverlapKeyer(
                min_token_overlap=min_token_overlap,
                tokenizer=tokenizer,
                max_keys=max_keys,
            ),
        )
    if scheme_norm in {"exact", "equality", "eq"}:
        return BucketableWeight(w=w, keyer=ExactKeyer())

    raise ValueError(f"Unknown scheme={scheme!r}. Supported: 'token_overlap', 'exact'.")


# Convenience default
@dataclass(frozen=True)
class EqualityBucketWeight:
    """
    Default bucketable weight:
      - blocking_keys(label) = (label,)
      - score = 1 if labels equal else 0

    Requires labels to be hashable.
    """
    def blocking_keys(self, a: Any) -> Iterable[Hashable]:
        return ExactKeyer()(a)

    def __call__(self, a: Any, b: Any) -> float:
        return 1.0 if a == b else 0.0


# -----------------------------------------------------------------------------
# Runtime checks used by matchers
# -----------------------------------------------------------------------------

def is_bucketable_weight(w: Any) -> bool:
    """Return True if w looks like a bucketable weight at runtime."""
    return callable(w) and hasattr(w, "blocking_keys") and callable(getattr(w, "blocking_keys", None))


def _check_hashable_keys(keys: Iterable[Any]) -> None:
    for k in keys:
        try:
            hash(k)
        except Exception as e:
            raise TypeError(
                "blocking_keys() must return hashable keys (usable as dict keys). "
                f"Found unhashable key of type {type(k)}."
            ) from e


def assert_bucketable_weight(
    w: Any,
    *,
    sample_label: Optional[Any] = None,
    mode_name: str = "sparse",
) -> None:
    """
    Raise a TypeError / ValueError if `w` does not have the bucketable interface.

    If sample_label is provided, we also call blocking_keys(sample_label) to validate:
      - it returns an iterable,
      - whose elements are hashable.
    """
    if not is_bucketable_weight(w):
        raise TypeError(f"{mode_name} mode requires a bucketable weight with blocking_keys(label).")

    if sample_label is not None:
        try:
            keys = list(w.blocking_keys(sample_label))
        except Exception as e:
            raise TypeError(
                "w.blocking_keys(label) must return an iterable of hashable keys."
            ) from e
        _check_hashable_keys(keys)
