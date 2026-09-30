# Deliberately archived benchmark data

`benchmarking/results/` is scratch output and stays ignored by Git. This folder
is for selected completed **numerical** runs that should accompany the code.

From the repository root, run `python benchmarking/archive_benchmark_data.py
--label NAME RUN_DIRECTORY [RUN_DIRECTORY ...]`, inspect the copied data, then
`git add -f -- benchmarking/recorded_runs/NAME` and commit. Do not archive sensitive
real-data outputs into a public repository. The current synthetic runs contain
scores, timings, configurations, and environment metadata.

The helper copies CSV, JSON, NPZ/NPY and logs, including nested sweep-point
outputs. It excludes figures, executed notebooks and caches. It checks required
files and byte hashes, refuses to overwrite an existing snapshot, and leaves
the original data unchanged. `SHA256SUMS` can be checked using `sha256sum -c
SHA256SUMS` from a snapshot directory. It does **not** stage, commit, or push.

The archive manifest records the Git revision at archive time. For an older
run that is not necessarily its execution revision; the original metadata is
not silently rewritten. Per-run CSVs and JSON/NPZ contents are authoritative.
