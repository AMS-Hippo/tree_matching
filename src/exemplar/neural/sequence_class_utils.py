
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, Hashable, Iterable, List, Sequence, Tuple

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def normalize_label(label: Any) -> Hashable:
    """
    Normalize labels so common user formats all work.

    Accepted examples:
      1
      "class_a"
      [1]
      ("biology",)
      numpy/int-like scalars with .item()
    """
    if isinstance(label, (list, tuple)) and len(label) == 1:
        label = label[0]

    if hasattr(label, "item") and not isinstance(label, (str, bytes, list, tuple, dict, set)):
        try:
            label = label.item()
        except Exception:
            pass

    return label


def _validate_sequence(seq: Sequence[Any], index: int) -> List[str]:
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            f"Sequence at position {index} must be a list/tuple of tokens, got {type(seq)!r}."
        )
    out = [str(tok) for tok in seq]
    return out


def standardize_raw_pairs(raw_pairs: Sequence[Tuple[Sequence[Any], Any]]) -> List[Tuple[List[str], Hashable]]:
    """
    Convert user data into a standard internal format:
        [(["A", "dog"], 1), (["Q"], "class_b"), ...]

    Parameters
    ----------
    raw_pairs:
        Sequence of (sequence, label) pairs.
    """
    standardized: List[Tuple[List[str], Hashable]] = []
    for i, pair in enumerate(raw_pairs):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(
                f"Entry {i} must be a pair (sequence, label). Got: {pair!r}"
            )
        seq, label = pair
        standardized.append((_validate_sequence(seq, i), normalize_label(label)))
    return standardized


def build_token_vocab(
    raw_pairs: Sequence[Tuple[Sequence[Any], Any]],
    min_freq: int = 1,
    specials: Sequence[str] = SPECIAL_TOKENS,
) -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build a token vocabulary from raw string-token data.
    """
    standardized = standardize_raw_pairs(raw_pairs)
    counter: Counter[str] = Counter()
    for seq, _ in standardized:
        counter.update(seq)

    token_to_id: Dict[str, int] = {}
    for tok in specials:
        if tok in token_to_id:
            raise ValueError(f"Duplicate special token: {tok}")
        token_to_id[tok] = len(token_to_id)

    for tok, freq in sorted(counter.items(), key=lambda x: (-x[1], x[0])):
        if freq >= min_freq and tok not in token_to_id:
            token_to_id[tok] = len(token_to_id)

    id_to_token = {idx: tok for tok, idx in token_to_id.items()}
    return token_to_id, id_to_token


def encode_sequence(
    seq: Sequence[Any],
    token_to_id: Dict[str, int],
    unk_token: str = UNK_TOKEN,
) -> List[int]:
    unk_id = token_to_id[unk_token]
    return [token_to_id.get(str(tok), unk_id) for tok in seq]


def decode_token_ids(
    token_ids: Sequence[int],
    id_to_token: Dict[int, str],
    stop_at_eos: bool = True,
    skip_special_tokens: bool = True,
) -> List[str]:
    """
    Decode a list of token ids back into string tokens.
    """
    out: List[str] = []
    specials = set(SPECIAL_TOKENS)
    for idx in token_ids:
        tok = id_to_token[int(idx)]
        if stop_at_eos and tok == EOS_TOKEN:
            break
        if skip_special_tokens and tok in specials:
            continue
        out.append(tok)
    return out


def encode_raw_pairs(
    raw_pairs: Sequence[Tuple[Sequence[Any], Any]],
    token_to_id: Dict[str, int] | None = None,
    min_freq: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, int], Dict[int, str]]:
    """
    Encode raw string-token data into integer-token records.

    Returns
    -------
    encoded_pairs:
        List of dicts with keys:
            - "sequence_ids": list[int]
            - "label": normalized label
    token_to_id, id_to_token:
        The learned vocabulary mappings.
    """
    standardized = standardize_raw_pairs(raw_pairs)
    if token_to_id is None:
        token_to_id, id_to_token = build_token_vocab(standardized, min_freq=min_freq)
    else:
        id_to_token = {idx: tok for tok, idx in token_to_id.items()}

    encoded_pairs: List[Dict[str, Any]] = []
    for seq, label in standardized:
        encoded_pairs.append(
            {
                "sequence_ids": encode_sequence(seq, token_to_id=token_to_id),
                "label": label,
            }
        )
    return encoded_pairs, token_to_id, id_to_token


def group_encoded_pairs_by_label(
    encoded_pairs: Sequence[Dict[str, Any]]
) -> Dict[Hashable, List[List[int]]]:
    grouped: Dict[Hashable, List[List[int]]] = defaultdict(list)
    for item in encoded_pairs:
        grouped[item["label"]].append(list(item["sequence_ids"]))
    return dict(grouped)


def infer_max_raw_length(encoded_pairs: Sequence[Dict[str, Any]]) -> int:
    if len(encoded_pairs) == 0:
        raise ValueError("encoded_pairs is empty.")
    return max(len(item["sequence_ids"]) for item in encoded_pairs)


def summarize_encoded_dataset(encoded_pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = group_encoded_pairs_by_label(encoded_pairs)
    lengths = [len(item["sequence_ids"]) for item in encoded_pairs]
    return {
        "num_examples": len(encoded_pairs),
        "num_labels": len(grouped),
        "label_sizes": {label: len(v) for label, v in grouped.items()},
        "min_length": min(lengths) if lengths else None,
        "max_length": max(lengths) if lengths else None,
        "avg_length": (sum(lengths) / len(lengths)) if lengths else None,
    }


def pretty_decode_sequences(
    sequences: Sequence[Sequence[int]],
    id_to_token: Dict[int, str],
) -> List[List[str]]:
    return [decode_token_ids(seq, id_to_token=id_to_token) for seq in sequences]



def average_sequence_by_length(
    sequences: Sequence[Sequence[Any]],
    weights: Sequence[float] | None = None,
    ignore_tokens: Iterable[Any] | None = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Group sequences by length and compute a coordinate-wise average within each
    length bucket.

    For discrete tokens, an ordinary arithmetic mean is not defined. Here,
    "average" means the empirical average of one-hot vectors at each position.
    The returned ``average_sequence`` is the coordinate-wise argmax token
    (equivalently, the per-position mode), and the full per-position token
    probabilities are returned as well.

    Parameters
    ----------
    sequences:
        A list of token sequences. Tokens can be strings, integers, or any
        hashable values.
    weights:
        Optional nonnegative weights for the sequences.
    ignore_tokens:
        Optional iterable of tokens to drop before processing each sequence.
        This is useful if you want to ignore markers such as ``<EOS>``.

    Returns
    -------
    summary_by_length:
        Dictionary keyed by sequence length. Each value is a dict with keys:
            - ``count``: number of sequences of that length
            - ``total_weight``: sum of weights in that bucket
            - ``average_sequence``: coordinate-wise consensus sequence
            - ``position_token_probs``: list of dicts giving token probabilities
              at each position
            - ``position_token_counts``: list of dicts giving weighted counts
              at each position

    Example
    -------
    >>> average_sequence_by_length([
    ...     ["A", "B", "C"],
    ...     ["A", "X", "C"],
    ...     ["A", "B", "D"],
    ... ])
    {3: {
        'count': 3,
        'total_weight': 3.0,
        'average_sequence': ['A', 'B', 'C'],
        ...
    }}
    """
    if len(sequences) == 0:
        return {}

    ignore_set = set(ignore_tokens) if ignore_tokens is not None else None

    cleaned_sequences: List[List[Any]] = []
    for i, seq in enumerate(sequences):
        if not isinstance(seq, (list, tuple)):
            raise TypeError(
                f"Sequence at position {i} must be a list/tuple of tokens, got {type(seq)!r}."
            )
        seq_list = list(seq)
        if ignore_set is not None:
            seq_list = [tok for tok in seq_list if tok not in ignore_set]
        cleaned_sequences.append(seq_list)

    if weights is None:
        weights = [1.0] * len(cleaned_sequences)
    elif len(weights) != len(cleaned_sequences):
        raise ValueError(
            f"weights has length {len(weights)}, but sequences has length {len(cleaned_sequences)}."
        )

    buckets: Dict[int, List[Tuple[List[Any], float]]] = defaultdict(list)
    for seq, w in zip(cleaned_sequences, weights):
        w = float(w)
        if w < 0:
            raise ValueError("weights must be nonnegative.")
        buckets[len(seq)].append((seq, w))

    summary_by_length: Dict[int, Dict[str, Any]] = {}
    for length, bucket in sorted(buckets.items()):
        position_counters = [Counter() for _ in range(length)]
        total_weight = 0.0
        for seq, w in bucket:
            total_weight += w
            for pos, tok in enumerate(seq):
                position_counters[pos][tok] += w

        if total_weight <= 0:
            raise ValueError(
                f"Total weight for length bucket {length} is nonpositive."
            )

        average_sequence: List[Any] = []
        position_token_probs: List[Dict[Any, float]] = []
        position_token_counts: List[Dict[Any, float]] = []

        for counter in position_counters:
            items = sorted(counter.items(), key=lambda kv: (-kv[1], str(kv[0])))
            average_sequence.append(items[0][0] if items else None)
            position_token_counts.append(dict(items))
            position_token_probs.append({tok: cnt / total_weight for tok, cnt in items})

        summary_by_length[length] = {
            'count': len(bucket),
            'total_weight': total_weight,
            'average_sequence': average_sequence,
            'position_token_probs': position_token_probs,
            'position_token_counts': position_token_counts,
        }

    return summary_by_length
