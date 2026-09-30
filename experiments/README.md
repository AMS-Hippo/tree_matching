# Experiments and benchmarks

Benchmarking and exploratory notebooks live here. Generated artifacts should be
written under `.cache/`, not into version-controlled source folders.

Pairwise path-matching benchmarks are configured by:

- `path_match_benchmark_quick.json`: small smoke/correctness suite;
- `path_match_benchmark_full.json`: starter matrix across tree shapes, label
  distributions, score models, and tree--tree versus tree--path comparisons;
- `path_match_benchmark_large.json`: larger trees, more instances, and five timing repeats;
- `run_path_match_benchmarks.py`: command-line runner.

Repeated query-to-template throughput presets are:

- `path_match_throughput_quick.json`: tiny prepared-API and correctness check;
- `path_match_throughput_full.json`: equality, sparse overlap, and generic
  Jaccard tree-to-path workloads;
- `path_match_throughput_large.json`: larger trees and larger query-template banks.

The throughput presets are normally run through
[`../benchmarking/02_throughput.ipynb`](../benchmarking/02_throughput.ipynb),
which uses the tested backend in `src/benchmarking/path_match_throughput.py`.

Run the quick suite from the repository root:

```bash
python experiments/run_path_match_benchmarks.py \
    --config experiments/path_match_benchmark_quick.json
```

See [`../docs/PATH_MATCH_BENCHMARKS.md`](../docs/PATH_MATCH_BENCHMARKS.md) for
algorithm preset definitions, regime fields, output columns, and timing rules.

For notebook interfaces, use:

- [`../benchmarking/01_algorithm_ranking.ipynb`](../benchmarking/01_algorithm_ranking.ipynb)
  for controlled independent-pair comparisons;
- [`../benchmarking/02_throughput.ipynb`](../benchmarking/02_throughput.ipynb)
  for a reusable template bank and complete query-template score matrix.

The notebooks are the recommended interactive entry points; the JSON files and
backend modules remain the tested automation interface.

## Controlled timing mode

The `*_quick.json` presets run in-process and are intended only for smoke tests.
The `*_full.json` presets run each algorithm job in an isolated subprocess with
one numerical worker thread and an explicit wall-clock timeout.  Standard
outputs additionally record warm timing quartiles and raw repetitions,
whole-worker wall time, worker thread settings, and lifetime peak resident
memory.  A timeout remains in the result table and counts as incomplete
coverage; it is not silently omitted.


Size and beam-parameter sweeps are configured by:

- `path_match_sweeps_large.json`: five size sweeps and four beam calibration sweeps;
- `run_path_match_sweeps.py`: sweep runner that delegates every point to the ordinary pairwise backend.

Run them with:

```bash
python experiments/run_path_match_sweeps.py \
  --config experiments/path_match_sweeps_large.json \
  --output-dir benchmarking/results/path_match_sweeps_large_v1
```
