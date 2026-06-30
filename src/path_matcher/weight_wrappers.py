
"""
Adapters that add a `blocking_keys(label)` method to a plain weight function w(label_u, label_v).

Motivation
----------
Sparse candidate-based methods need a way to generate plausible matches without
evaluating w on all |G|*|H| label pairs. A common approach is *blocking* (a.k.a.
bucketing / indexing): compute one or more hashable keys from each label, build an
inverted index key -> nodes, and only compare labels that share a key.

This file provides wrappers that:

- preserve the original scoring behavior:
      wrapped(a,b) == w(a,b)
- and add a blocking_keys(label) method suitable for candidate generation.

Important correctness note
--------------------------
For a sparse candidate method to be faithful to your scoring function, your blocking rule
should be *recall-safe* for the pairs you care about:

  If w(label_u, label_v) can be large, then blocking_keys(label_u) and blocking_keys(label_v)
  should share at least one key.

Otherwise the candidate generator may never consider that pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Union
import hashlib


WeightFn = Callable[[Any, Any], float]
Extractor = Union[str, int, Callable[[Any], Any]]


def _stable_hash64(x: Any, *, salt: int = 0) -> int:
    """
    Stable 64-bit hash for arbitrary Python objects using repr(x).

    This is used only for deterministic ordering / selection. It does not need to be fast.
    """
    b = repr(x).encode("utf-8", errors="backslashreplace")
    salt_b = int(salt).to_bytes(8, "little", signed=False)
    h = hashlib.blake2b(salt_b + b, digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _get_field(label: Any, field: Extractor) -> Any:
    """
    Extract a field from label.

    - If field is callable: return field(label)
    - If field is str: try label[field] if mapping, else getattr(label, field)
    - If field is int: try label[field] if label is a list/tuple (not a str/bytes)
    """
    if callable(field):
        return field(label)

    if isinstance(field, str):
        if isinstance(label, Mapping) and field in label:
            return label[field]
        if hasattr(label, field):
            return getattr(label, field)
        raise KeyError(f"Could not extract field '{field}' from label of type {type(label)}")

    if isinstance(field, int):
        if isinstance(label, (list, tuple)):
            return label[field]
        raise TypeError(f"Integer field access requires label to be list/tuple, got {type(label)}")

    raise TypeError("field must be a str, int, or callable")


@dataclass(frozen=True)
class FieldAnyOverlapWeight:
    """
    Blocking rule (1):
      label has a field that is an iterable/set of tags from a small dictionary.
      Two labels are *plausible matches* if they share at least one tag.

    blocking_keys(label) returns those tags (optionally with namespacing).

    Parameters
    ----------
    w:
        Original scoring function.
    field:
        How to extract the tag-set field from label (str attr/key, int index, or callable).
    namespace:
        If provided, keys are returned as (namespace, tag) to avoid collisions with other schemes.
    max_keys:
        Optional cap on the number of tags emitted per label. If used, tags are selected
        deterministically using stable hashing.
    """
    w: WeightFn
    field: Extractor
    namespace: Optional[Hashable] = ("field_any",)
    max_keys: Optional[int] = None
    hash_salt: int = 0

    def __call__(self, a: Any, b: Any) -> float:
        return float(self.w(a, b))

    def blocking_keys(self, a: Any) -> Iterable[Hashable]:
        tags = _get_field(a, self.field)
        if tags is None:
            return ()

        try:
            tag_list = list(tags)
        except Exception as e:
            raise TypeError(
                "FieldAnyOverlapWeight expected the extracted field to be iterable (e.g. set/list)."
            ) from e

        # Deterministic truncation if needed.
        if self.max_keys is not None and len(tag_list) > self.max_keys:
            tag_list.sort(key=lambda t: _stable_hash64(t, salt=self.hash_salt))
            tag_list = tag_list[: self.max_keys]

        # Enforce hashable tags.
        out: List[Hashable] = []
        for t in tag_list:
            hash(t)  # may raise
            out.append((self.namespace, t) if self.namespace is not None else t)
        return out


def _default_tokenizer(label: Any) -> List[Any]:
    """
    Reasonable default tokenizer:
      - list/tuple -> elements
      - str -> split on '/' if present else whitespace
      - otherwise -> [repr(label)]
    """
    if isinstance(label, (list, tuple)):
        return list(label)
    if isinstance(label, str):
        if "/" in label:
            toks = [t for t in label.split("/") if t]
        else:
            toks = [t for t in label.split() if t]
        return toks if toks else [label]
    return [repr(label)]


@dataclass(frozen=True)
class PrefixBlockingWeight:
    """
    Blocking rule (2):
      Match based on shared prefixes of a token sequence representation of the label.

    blocking_keys(label) returns prefix keys up to max_depth:
        prefix(1), prefix(2), ..., prefix(max_depth)
    where prefix(d) is a tuple of the first d tokens.

    This pairs naturally with SparseCandidateConfig.max_keys_per_node, which can select
    the *rarest* prefix keys across a dataset, giving the "data-dependent depth" effect.

    Parameters
    ----------
    w:
        Original scoring function.
    tokenizer:
        Function mapping label -> sequence of tokens. If None, uses _default_tokenizer.
    max_depth:
        Maximum prefix depth to emit.
    namespace:
        If provided, keys are returned as (namespace, prefix_tuple).
    """
    w: WeightFn
    tokenizer: Optional[Callable[[Any], Sequence[Any]]] = None
    max_depth: int = 5
    namespace: Optional[Hashable] = ("prefix",)

    def __call__(self, a: Any, b: Any) -> float:
        return float(self.w(a, b))

    def blocking_keys(self, a: Any) -> Iterable[Hashable]:
        tok_fn = self.tokenizer or _default_tokenizer
        toks = list(tok_fn(a))
        if not toks:
            return ()

        dmax = min(int(self.max_depth), len(toks))
        out: List[Hashable] = []
        for d in range(1, dmax + 1):
            prefix = tuple(toks[:d])
            # ensure hashable
            hash(prefix)
            out.append((self.namespace, prefix) if self.namespace is not None else prefix)
        return out


@dataclass(frozen=True)
class TokenOverlapBlockingWeight:
    """
    Blocking rule (3):
      Minimum token overlap. The blocking stage can only enforce a *necessary* condition:
      if two labels share >=k tokens then they must share at least one token.

    blocking_keys(label) returns (a subset of) the tokens.

    Parameters
    ----------
    w:
        Original scoring function.
    tokenizer:
        Function mapping label -> iterable of tokens. If None, uses _default_tokenizer.
    max_tokens:
        Optional cap on number of tokens emitted per label (deterministically by stable hash).
    namespace:
        If provided, keys are returned as (namespace, token).
    """
    w: WeightFn
    tokenizer: Optional[Callable[[Any], Iterable[Any]]] = None
    max_tokens: Optional[int] = 50
    namespace: Optional[Hashable] = ("tok",)
    hash_salt: int = 0

    def __call__(self, a: Any, b: Any) -> float:
        return float(self.w(a, b))

    def blocking_keys(self, a: Any) -> Iterable[Hashable]:
        tok_fn = self.tokenizer or _default_tokenizer
        toks = list(tok_fn(a))
        if not toks:
            return ()

        # Deterministic order by stable hash (important if toks came from a set)
        toks.sort(key=lambda t: _stable_hash64(t, salt=self.hash_salt))

        if self.max_tokens is not None and len(toks) > self.max_tokens:
            toks = toks[: self.max_tokens]

        out: List[Hashable] = []
        for t in toks:
            hash(t)
            out.append((self.namespace, t) if self.namespace is not None else t)
        return out


def make_bucketable_weight(
    w: WeightFn,
    scheme: str,
    **kwargs: Any,
):
    """
    Wrap a plain weight function with an appropriate blocking_keys() method.

    Parameters
    ----------
    w:
        Callable weight function w(a,b)->float.
    scheme:
        One of:
          - "field_any" / "1": blocking based on overlap of a set-valued field (rule 1)
          - "prefix" / "2": prefix blocking on tokenized labels (rule 2)
          - "token_overlap" / "3": token blocking for overlap-based matching (rule 3)
    kwargs:
        Passed to the corresponding wrapper dataclass.

    Returns
    -------
    A callable object with a blocking_keys(label) method, suitable for method="sparse".
    """
    if not callable(w):
        raise TypeError("w must be callable")

    s = scheme.strip().lower()
    if s in {"1", "field_any", "field-any", "fieldany", "any_field_overlap"}:
        if "field" not in kwargs:
            raise ValueError("scheme='field_any' requires kwarg field=... to extract the tag set from each label")
        return FieldAnyOverlapWeight(w=w, **kwargs)

    if s in {"2", "prefix", "prefixes"}:
        return PrefixBlockingWeight(w=w, **kwargs)

    if s in {"3", "token_overlap", "token-overlap", "overlap", "tokens"}:
        return TokenOverlapBlockingWeight(w=w, **kwargs)

    raise ValueError("Unknown scheme. Use one of: 'field_any'/'1', 'prefix'/'2', 'token_overlap'/'3'.")
