# Path-matching benchmarks

This folder is the notebook-facing entry point for the dense, sparse, and beam
path-matching benchmarks.  The tested implementation code remains under
`src/benchmarking/`; the notebooks select fixed presets, run that backend, and
present the results.

## Notebooks

### `01_algorithm_ranking.ipynb`

Runs the controlled pairwise benchmark.  It reports timing and matching
accuracy separately, plus a clearly secondary harmonic timing/accuracy score.
Approximate methods are allowed to be suboptimal.  Accuracy is reported only
when the exact methods that completed the instance agree within tolerance.
Planted-node recall is displayed only as a supplementary diagnostic and is not
used as a substitute for the optimal matching score.

The first editable cell keeps the main controls together:

```python
SCALE = "large"       # "smoke", "standard", or "large"
RUN_LABEL = "large_v1"  # change for each substantive run
REGIMES = "all"       # or one/a list of named regimes
ALGORITHMS = "all"    # or one/a list of named configurations
SIZE_OVERRIDES = {    # optional n/depth changes for named regimes
    # "rare_anchor_oracle": {"n_g": 2500, "n_h": 2500},
}
N_INSTANCES = None    # None keeps the preset value
REPEATS = None        # None keeps the preset value
SEED = 20260928
```

Only `n_g`, `n_h`, `depth_g`, and `depth_h` may be changed through
`SIZE_OVERRIDES`.  Algorithm hyperparameters remain in the named preset
configurations, so a result labelled by an algorithm name remains reproducible.

`smoke` checks the wiring and correctness gates quickly. It is not intended to
support performance conclusions. `standard` loads the first scientific matrix
from `experiments/path_match_benchmark_full.json`. `large` loads
`experiments/path_match_benchmark_large.json`, which increases tree sizes,
instance counts, timing repetitions, and worker limits.  The standard matrix now
contains both a 1,300-node exact-oracle rare-anchor companion and the larger
rare-anchor scalability stress case.

The two scales deliberately use different timing modes.  `smoke` runs in the
current notebook process for speed.  `standard` runs every algorithm-instance
job in a fresh worker process, fixes the common BLAS/OpenMP/Numba thread counts
to one before numerical packages are imported, enforces a wall-clock timeout,
and records lifetime peak resident memory.

The notebook writes raw rows, summary rows, metadata, and its compact display
tables under `benchmarking/results/algorithm_ranking_<scale>/`.  When
`RUN_LABEL` is set, the label is appended to that directory name.  Result
directories are ignored by Git.

### `02_throughput.ipynb`

Measures the repeated query-to-template workflow: fit any shared encoder,
prepare every query and template once, and then compute the complete score
matrix.  The notebook validates prepared calls against the ordinary API on a
small deterministic sample before accepting timing results.

Its first editable cell uses the same subset, size, and run-label controls, plus
query/template counts:

```python
SCALE = "large"        # "smoke", "standard", or "large"
RUN_LABEL = "large_v1"
REGIMES = "all"        # or one/a list of named regimes
ALGORITHMS = "all"     # or one/a list of named configurations
SIZE_OVERRIDES = {     # optional n/depth changes for named regimes
    # "sparse_overlap_tree_template_bank": {"n_g": 6000, "n_h": 3000},
}
N_QUERIES = None       # None keeps the preset value
N_TEMPLATES = None     # None keeps the preset value
REPEATS = None         # None keeps the preset value
SEED = 20260928
```

The standard and large throughput presets cover equality with common scalar
labels and path templates, sparse multi-token overlap with larger tree
templates, and generic Jaccard tree-to-path scores. The large preset increases
both the tree sizes and the number of query-template pairs.  The notebook reports
one-time preparation separately from warm score-matrix search.  It also saves
all score matrices, so exact implementations are compared entry by entry before
being used as an accuracy oracle.

As in the ranking notebook, the standard preset uses one isolated, single-thread
worker per algorithm-regime job.  The timeout applies to the complete worker
lifetime, while the warm score-matrix samples are measured internally after the
configured warm-up.

Outputs are written under `benchmarking/results/path_match_throughput_<scale>/`
(and have the run label appended when one is supplied).  They include row-level
and summary CSV files, metadata, the exact notebook configuration, and a
compressed NumPy archive of the score matrices.

## Setup

From the repository root, install the project and the small set of packages
needed by this notebook:

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[speedups,benchmark]"
python -m pip install notebook ipykernel matplotlib
python -m ipykernel install --user --name tree-matcher --display-name "Python (tree-matcher)"
jupyter notebook benchmarking/01_algorithm_ranking.ipynb
# or: jupyter notebook benchmarking/02_throughput.ipynb
```

The notebook also contains a source-tree bootstrap, so it can be run from a
checkout before an editable install.  Installing the package is still the
recommended workflow.

## Timing and resource columns

The standard presets report two distinct kinds of time:

- `cold_predict_seconds` or `cold_score_matrix_seconds` measure the first
  diagnostics-disabled search in a fresh worker, before any benchmark
  diagnostics or warm-up can populate JIT and allocator caches.  Their
  accompanying `cold_*setup*` fields include first-use preparation once.
- `predict_seconds_*` or `score_matrix_seconds_*` are warm internal samples
  measured after the configured warm-up.  Their median, quartiles, standard
  deviation, median absolute deviation, and raw sample list are retained.
- `fresh_process_total_seconds` is the wall time of the complete isolated
  benchmark worker.  It includes Python startup, imports, payload loading,
  diagnostics, matcher setup, warm-up, repeated timed calls, and result
  serialization.  It is a reproducible whole-job measurement, not a single
  cold prediction.

`peak_rss_mib` is the worker's lifetime peak resident set size.  It includes the
Python runtime and imported libraries as well as algorithm allocations, so it is
best interpreted comparatively on the same machine.  On platforms without the
standard `resource` interface it may be unavailable.

A timeout is recorded as `status="timeout"`; it counts as incomplete coverage
and is never dropped from a ranking silently.  Known incompatibilities or work
limits remain explicit `skipped` rows and do not start a worker.

## Interpretation rules

- A beam result below 100% accuracy is a valid approximate result, not a failed
  run.
- A method is eligible for the headline timing comparison only if it completes
  every intended instance, every instance has a valid exact oracle, and its
  minimum accuracy meets the configured quality floor.
- If exact methods disagree, the notebook stops before displaying an accuracy
  ranking for the disputed instance.
- Timing comparisons are machine- and environment-specific.  The smoke preset
  is for checking code paths, not for comparing algorithms.
- The harmonic timing/accuracy score is optional and deliberately secondary;
  the raw timing and percent-of-optimum columns are the primary outputs.
- In the throughput notebook, `pairs_per_second` excludes reusable setup;
  `setup_plus_one_matrix_seconds` includes encoder fitting and preparation once.
- A throughput method is quality-eligible only when every query-template score
  is at least the configured fraction of the exact score.  Aggregate matrix
  accuracy and per-pair accuracy summaries are still shown separately.
- The `smoke` presets in both notebooks are correctness checks, not evidence for
  performance conclusions.


## Scaling and beam-parameter sweeps

Run the checked-in size and beam sweeps from the repository root:

```bash
python experiments/run_path_match_sweeps.py \
  --config experiments/path_match_sweeps_large.json \
  --output-dir benchmarking/results/path_match_sweeps_large_v1
```

The sweep configuration contains five size sweeps and four controlled beam
parameter sweeps. It writes `sweep_rows.csv`, `sweep_summary.csv`, and the
ordinary benchmark artifacts for every individual point. The current sweep
file intentionally leaves lookahead disabled; its purpose is to calibrate the
base beams before adding further heuristics.

## Talk-ready figures

Generate uncluttered PNG and PDF figures from saved results without rerunning a
matcher:

```bash
python benchmarking/make_benchmark_figures.py \
  --pairwise benchmarking/results/algorithm_ranking_large_large_v1 \
  --throughput benchmarking/results/path_match_throughput_large_large_v1 \
  --sweeps benchmarking/results/path_match_sweeps_large_v1 \
  --outdir benchmarking/results/figures_large_v1
```

The script uses shared legends instead of overlapping point labels and creates
speed-accuracy panels, generic-versus-encoded exact comparisons, size-scaling
curves, and beam-parameter tradeoff figures. See
`BENCHMARK_RUN_AND_FIGURE_GUIDE.pdf` for the short operational guide.
