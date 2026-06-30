from __future__ import annotations

"""Shared utilities for cached synthetic datasets and benchmark plumbing.

This module is intentionally lightweight. Pass 1 focused on a shared simulation /
cache layer so that embedding and exemplar experiments reuse the same sampled data
and, when desired, the same expensive exact pairwise matcher outputs.
Pass 2 adds a cached sequence-bag layer for exemplar benchmarking.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import gzip
import hashlib
import json
import pickle
import time


DEFAULT_CACHE_ROOT = Path(".cache/benchmarks")
DEFAULT_DATASET_DIR = DEFAULT_CACHE_ROOT / "sampled_datasets"
DEFAULT_PAIR_DIR = DEFAULT_CACHE_ROOT / "exact_similarity_cache"
DEFAULT_BAG_DIR = DEFAULT_CACHE_ROOT / "extracted_sequence_bags"


@dataclass
class DatasetBundle:
    name: str
    graphs: List[Any]
    labels: List[int]
    class_sequences: List[List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_graphs(self) -> int:
        return len(self.graphs)

    @property
    def n_classes(self) -> int:
        return len(set(int(x) for x in self.labels))


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def canonicalize_for_hash(x: Any) -> Any:
    """Convert a nested config object into a JSON-stable form."""
    if isinstance(x, Mapping):
        return {str(k): canonicalize_for_hash(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [canonicalize_for_hash(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def config_hash(spec: Mapping[str, Any], *, prefix: str = "") -> str:
    payload = json.dumps(canonicalize_for_hash(dict(spec)), sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha1(payload).hexdigest()[:12]
    return f"{prefix}{h}" if prefix else h


def load_config(config_or_path: Union[str, Path, Mapping[str, Any]]) -> Dict[str, Any]:
    if isinstance(config_or_path, Mapping):
        return dict(config_or_path)
    path = Path(config_or_path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Reading YAML configs requires PyYAML. Either install it or pass a dict / JSON config."
            ) from e
        data = yaml.safe_load(text)
        return {} if data is None else dict(data)
    raise ValueError(f"Unsupported config format: {path.suffix}")


def build_tree_sampler(spec: Optional[Mapping[str, Any]] = None):
    from path_matcher.planted_path_sampler import GaltonWatsonTreeSampler

    cfg = dict(spec or {})
    return GaltonWatsonTreeSampler(
        max_depth=int(cfg.get("max_depth", 12)),
        lam=float(cfg.get("lam", 1.8)),
        max_nodes=int(cfg.get("max_nodes", 300)),
        traversal=str(cfg.get("traversal", "bfs")),
        min_nodes=cfg.get("min_nodes", 300),
        require_full_depth=bool(cfg.get("require_full_depth", True)),
        max_tries=int(cfg.get("max_tries", 5000)),
    )


def build_toy_model_cfg(spec: Mapping[str, Any]):
    from path_matcher.planted_path_sampler import ToyModelConfig

    cfg = dict(spec)
    return ToyModelConfig(
        n_classes=int(cfg.get("n_classes", 3)),
        n_per_class=cfg.get("n_per_class", 20),
        planted_seq_len=int(cfg.get("planted_seq_len", 10)),
        alphabet_size=int(cfg.get("alphabet_size", 10)),
        p_obs=float(cfg.get("p_obs", 0.9)),
        rate_fn=cfg.get("rate_fn"),
        pi=cfg.get("pi"),
        dirichlet_alpha=cfg.get("dirichlet_alpha"),
        base_sequence=cfg.get("base_sequence"),
        alphabet=cfg.get("alphabet"),
        seed=cfg.get("seed"),
    )


def _dataset_filename(name: str, spec_hash_value: str) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)
    return f"{safe_name}__{spec_hash_value}.pkl.gz"


def sample_or_load_dataset(
    dataset_spec: Mapping[str, Any],
    *,
    cache_dir: Union[str, Path] = DEFAULT_DATASET_DIR,
    force: bool = False,
    verbose: bool = False,
) -> DatasetBundle:
    """Load a cached toy dataset, or sample it using the current sampler code."""
    spec = dict(dataset_spec)
    name = str(spec.get("name", "toy_dataset"))
    toy_cfg_spec = dict(spec.get("toy_model", spec))
    tree_sampler_spec = dict(spec.get("tree_sampler", {}))
    hash_value = config_hash({"toy_model": toy_cfg_spec, "tree_sampler": tree_sampler_spec}, prefix="toy_")
    path = ensure_dir(cache_dir) / _dataset_filename(name, hash_value)

    if path.exists() and not force:
        with gzip.open(path, "rb") as f:
            payload = pickle.load(f)
        return DatasetBundle(**payload)

    from path_matcher.planted_path_sampler import sample_toy_model

    t0 = time.perf_counter()
    toy_cfg = build_toy_model_cfg(toy_cfg_spec)
    sampler = build_tree_sampler(tree_sampler_spec)
    graphs, labels, class_sequences = sample_toy_model(toy_cfg, tree_path_sampler=sampler)
    dt = time.perf_counter() - t0

    bundle = DatasetBundle(
        name=name,
        graphs=list(graphs),
        labels=[int(x) for x in labels],
        class_sequences=[list(seq) for seq in class_sequences],
        metadata={
            "dataset_hash": hash_value,
            "toy_model": toy_cfg_spec,
            "tree_sampler": tree_sampler_spec,
            "sample_runtime_sec": dt,
        },
    )

    with gzip.open(path, "wb") as f:
        pickle.dump(bundle.__dict__, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Saved dataset cache: {path}")
    return bundle


def dataset_to_nested_blocks(bundle: DatasetBundle) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_class: Dict[int, List[Any]] = {}
    for g, y in zip(bundle.graphs, bundle.labels):
        by_class.setdefault(int(y), []).append(g)
    for c in sorted(by_class):
        out.append({
            "trees": [[g, None] for g in by_class[c]],
            "a": list(bundle.class_sequences[c]) if c < len(bundle.class_sequences) else None,
        })
    return out


def _pair_cache_filename(dataset_name: str, dataset_hash: str, matcher_kind: str, phi_name: str, matcher_kwargs: Mapping[str, Any]) -> str:
    h = config_hash({
        "dataset_hash": dataset_hash,
        "matcher_kind": matcher_kind,
        "phi_name": phi_name,
        "matcher_kwargs": dict(matcher_kwargs),
    }, prefix="pairs_")
    safe_name = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_name)
    return f"{safe_name}__{h}.pkl.gz"


def compute_or_load_exact_pair_scores(
    graphs: Sequence[Any],
    *,
    dataset_name: str,
    dataset_hash: str,
    matcher_kind: str = "fast",
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    phi_name: str = "label",
    cache_dir: Union[str, Path] = DEFAULT_PAIR_DIR,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Compute or load the expensive exact matcher outputs for a graph collection."""
    from embedder.tree_cluster import compute_sims_and_matches

    mkwargs = dict(matcher_kwargs or {})
    path = ensure_dir(cache_dir) / _pair_cache_filename(dataset_name, dataset_hash, matcher_kind, phi_name, mkwargs)

    if path.exists() and not force:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)

    t0 = time.perf_counter()
    sims, matches = compute_sims_and_matches(
        graphs,
        phi_name=phi_name,
        matcher_kind=matcher_kind,
        matcher_kwargs=mkwargs,
    )
    dt = time.perf_counter() - t0
    payload = {
        "sims": sims,
        "matches": matches,
        "runtime_sec": dt,
        "dataset_name": dataset_name,
        "dataset_hash": dataset_hash,
        "matcher_kind": matcher_kind,
        "matcher_kwargs": mkwargs,
        "phi_name": phi_name,
    }
    with gzip.open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f"Saved pair-score cache: {path}")
    return payload


def _bag_cache_filename(
    dataset_name: str,
    dataset_hash: str,
    matcher_kind: str,
    matcher_kwargs: Mapping[str, Any],
    *,
    seq_mode: str,
    deduplicate: bool,
    rebalance_by_tree: bool,
    phi_name: str,
) -> str:
    h = config_hash({
        'dataset_hash': dataset_hash,
        'matcher_kind': matcher_kind,
        'matcher_kwargs': dict(matcher_kwargs),
        'seq_mode': seq_mode,
        'deduplicate': bool(deduplicate),
        'rebalance_by_tree': bool(rebalance_by_tree),
        'phi_name': phi_name,
    }, prefix='bags_')
    safe_name = ''.join(ch if ch.isalnum() or ch in {'-', '_'} else '_' for ch in dataset_name)
    return f'{safe_name}__{h}.pkl.gz'


def compute_or_load_sequence_bags(
    graphs: Sequence[Any],
    *,
    labels: Sequence[int],
    matches: Mapping[Any, Any],
    dataset_name: str,
    dataset_hash: str,
    matcher_kind: str = 'fast',
    matcher_kwargs: Optional[Mapping[str, Any]] = None,
    seq_mode: str = 'raw',
    deduplicate: bool = True,
    rebalance_by_tree: bool = True,
    phi_name: str = 'label',
    cache_dir: Union[str, Path] = DEFAULT_BAG_DIR,
    force: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    from exemplar.extract import extract_sequence_bags_by_cluster

    mkwargs = dict(matcher_kwargs or {})
    path = ensure_dir(cache_dir) / _bag_cache_filename(
        dataset_name,
        dataset_hash,
        matcher_kind,
        mkwargs,
        seq_mode=seq_mode,
        deduplicate=deduplicate,
        rebalance_by_tree=rebalance_by_tree,
        phi_name=phi_name,
    )
    if path.exists() and not force:
        with gzip.open(path, 'rb') as f:
            return pickle.load(f)

    t0 = time.perf_counter()
    bags = extract_sequence_bags_by_cluster(
        matches,
        labels,
        graphs,
        phi_name=phi_name,
        seq_mode=seq_mode,
        deduplicate=deduplicate,
        rebalance_by_tree=rebalance_by_tree,
    )
    dt = time.perf_counter() - t0
    payload = {
        'bags': bags,
        'runtime_sec': dt,
        'dataset_name': dataset_name,
        'dataset_hash': dataset_hash,
        'matcher_kind': matcher_kind,
        'matcher_kwargs': mkwargs,
        'phi_name': phi_name,
        'seq_mode': seq_mode,
        'deduplicate': bool(deduplicate),
        'rebalance_by_tree': bool(rebalance_by_tree),
    }
    with gzip.open(path, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    if verbose:
        print(f'Saved sequence-bag cache: {path}')
    return payload
