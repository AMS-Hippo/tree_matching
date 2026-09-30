#!/usr/bin/env python3
"""Copy completed numerical benchmark outputs into a versionable run snapshot.

No git index/remote changes, no result deletion, and no PNG/PDF/cache copies.
Use after a run has finished. The snapshot is opt-in; results/ stays ignored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

DATA_SUFFIXES = {".csv", ".json", ".npz", ".npy", ".log"}
REQUIRED = {
    "benchmark_rows.csv": ("benchmark_summary.csv", "benchmark_metadata.json"),
    "throughput_rows.csv": ("throughput_summary.csv", "throughput_metadata.json", "throughput_score_matrices.npz"),
    "sweep_rows.csv": ("sweep_summary.csv", "sweep_metadata.json"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_runs(repo: Path, sources: list[Path], label: str) -> Path:
    """Preflight all sources and copy data into a new directory atomically."""
    repo = repo.resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", label) is None:
        raise ValueError("label must use letters, digits, '.', '_' or '-' and begin with a letter/digit")
    if not sources:
        raise ValueError("Provide at least one completed run directory")
    if not (repo / "pyproject.toml").is_file():
        raise ValueError("Run from the tree_matching repository root")
    results_root = (repo / "benchmarking/results").resolve()
    destination = repo / "benchmarking/recorded_runs" / label
    if destination.exists():
        raise FileExistsError(f"Snapshot already exists: {destination}; use a new label")
    plan = []
    names = set()
    for original in sources:
        source = original if original.is_absolute() else repo / original
        if source.is_symlink():
            raise ValueError(f"Refusing a symlink: {source}")
        source = source.resolve()
        if not source.is_dir() or not source.is_relative_to(results_root) or source == results_root:
            raise ValueError(f"Expected an individual completed run under {results_root}: {source}")
        if source.name in names:
            raise ValueError(f"Duplicate run-directory name: {source.name}")
        names.add(source.name)
        kinds = [name for name in REQUIRED if (source / name).is_file()]
        if len(kinds) != 1:
            raise ValueError(f"Cannot identify exactly one completed benchmark in {source}")
        for name in REQUIRED[kinds[0]]:
            if not (source / name).is_file():
                raise FileNotFoundError(f"Required run artifact missing: {source / name}")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError(f"Refusing a symlink inside run data: {path}")
            if not path.is_file() or path.suffix.lower() not in DATA_SUFFIXES:
                continue
            if any(part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts):
                continue
            # This is a conservative local safeguard, not a claim about a host's file limit.
            if path.stat().st_size > 50 * 1024**2:
                raise ValueError(f"File exceeds the 50 MiB snapshot safeguard: {path}; arrange large-file storage first")
            target = Path(source.name) / path.relative_to(source)
            plan.append((path, target))
    if sum(path.stat().st_size for path, _ in plan) > 250 * 1024**2:
        raise ValueError("Numerical snapshot exceeds 250 MiB; select fewer runs or arrange large-file storage")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=".archive-", dir=destination.parent))
    try:
        records = []
        for source, relative in plan:
            target = temp / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            before = sha256(source)
            shutil.copy2(source, target)
            if sha256(target) != before or sha256(source) != before:
                raise RuntimeError(f"Run file changed while archiving: {source}. Wait until the run finishes.")
            records.append({"path": relative.as_posix(), "bytes": target.stat().st_size, "sha256": before})
        try:
            head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                  capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            head = None
        manifest = {
            "archived_at_utc": datetime.now(timezone.utc).isoformat(),
            "repository_head_when_archived": head,
            "provenance_note": "This HEAD is the collection-time revision, not necessarily the revision that ran older data. Original run metadata/configs are preserved unchanged.",
            "sources": [str(p if p.is_absolute() else repo / p) for p in sources],
            "excluded": "PNG/PDF figures, notebooks, interpreter/Numba caches, and other non-data files",
            "files": records,
        }
        (temp / "archive_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        (temp / "SHA256SUMS").write_text("".join(f"{r['sha256']}  {r['path']}\n" for r in records), encoding="utf-8")
        temp.rename(destination)
    except BaseException:
        shutil.rmtree(temp)
        raise
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="New snapshot label, e.g. fast_beam_v2")
    parser.add_argument("runs", nargs="+", type=Path, help="Completed result directories under benchmarking/results")
    args = parser.parse_args()
    try:
        destination = archive_runs(Path.cwd(), args.runs, args.label)
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"No snapshot created: {error}\n")
    print(f"Saved numerical data to {destination}")
    print(f"Review, then stage with: git add -f -- benchmarking/recorded_runs/{args.label}")
    print("Nothing has been committed or pushed. Original results are unchanged.")


if __name__ == "__main__":
    main()
