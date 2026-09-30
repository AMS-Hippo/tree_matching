# Path-matching benchmark harness

This benchmark layer compares the dense, sparse, and beam implementations on
controlled pairs of labelled trees. It is designed for algorithm development:
the synthetic generator records the planted correspondence, the runner records
both clean external timings and internal diagnostics, and every algorithm
configuration is written explicitly into the output.

## Two different kinds of names

A **named algorithm configuration** is a complete matcher preset. For example,
`beam_partial_score` means a deterministic partial-matching beam with
`beam_width=200`, expansion width 64, no random exploration, and all current
heuristic bonuses set to zero. The name prevents a result labelled merely
"beam" from silently changing when defaults change.

A **named data regime** specifies the input distribution: tree shape, sizes,
alphabet, label frequencies, symbols per node, score model, and planted path.
Examples are `tiny_generic_dense` and `narrow_tree_to_path`.

The benchmark is the cross-product of the selected algorithm configurations and
data regimes, subject to explicit safety limits.

## Existing sampler and the additional benchmark generator

`path_matcher.planted_path_sampler` remains the implementation of the paper's
Galton--Watson planted-path model. It is appropriate for statistical simulation
and returns `igraph` objects.

The algorithm benchmark additionally uses
`benchmarking.path_match_cases.generate_synthetic_pair`. This generator is not
a replacement for the paper model. It fills several benchmarking-specific gaps:

- exact control of the two tree sizes;
- narrow, medium-depth, wide, and path-shaped trees;
- tree--tree and tree--path comparisons;
- uniform and Zipf/heavy-tailed symbol frequencies;
- one or a variable number of symbols per node;
- equality, maximum-overlap, and weighted-Jaccard scores;
- a planted path selected independently of child index order;
- randomized node order within every depth level, so capped child expansion is
  not accidentally aligned with the generator's structurally special branch;
- direct `TreeData` output, so benchmark generation does not require `igraph`;
- cheap estimates of dense cells and posting-list candidate hits.

The generator is deterministic from `(regime.seed, instance_index)`.

## Standard algorithm configurations

The registry is returned by
`benchmarking.path_match_benchmark.default_algorithm_specs()`.

| Name | Meaning |
|---|---|
| `exact_dense` | Generic exact `O(nm)` dynamic program. |
| `fast_dense` | Specialized integerized/Numba exact equality or overlap matcher. |
| `sparse_closure` | Previous selected-cell skip-closure dynamic program, using exhaustive blocking candidates. |
| `sparse_chain` | Exact maximum-chain/product-poset solver over exhaustive positive blocking candidates. |
| `fast_sparse` | Specialized sparse equality or overlap matcher. |
| `beam_local` | Algorithm 6 local-transition beam, all children, score ranking. |
| `beam_local_capped` | Algorithm 6 with a deterministic child cap of 16. |
| `beam_partial_score` | Deterministic score-only Algorithm 7 state space, beam 200 and expansion budget 64. |
| `beam_partial_heuristic` | Current heuristic partial beam without the optional lookahead sketches. |

These are starting presets rather than claims that the parameter values are
optimal. A JSON configuration may override any preset keyword arguments; the
resolved arguments are copied into every result row. When the same preset is
used with several parameter values, give each entry a unique `alias`; otherwise
the runner rejects the ambiguous configuration instead of merging the results.

## Included suites

### Quick smoke suite

```bash
python experiments/run_path_match_benchmarks.py \
    --config experiments/path_match_benchmark_quick.json
```

The quick suite checks that all principal implementations run on generic,
dense-overlap, sparse-equality, and tree-to-path examples. Its instances are too
small to establish reliable runtime crossovers.

### Starter algorithm matrix

```bash
python experiments/run_path_match_benchmarks.py \
    --config experiments/path_match_benchmark_full.json \
    --output-dir .cache/benchmarks/path_match/full_v1
```

The starter matrix deliberately contains contrasting hypotheses:

- tiny generic scores should favor `exact_dense` because indexing overhead is
  not worthwhile;
- dense common-token overlap should favor `fast_dense`;
- a generic score with very sparse exact blocking should favor `sparse_chain`;
- a very large sparse multi-token overlap problem should make `fast_sparse`
  useful once a dense table exceeds the configured work limit;
- a narrow tree matched to a path should favor local transitions;
- shallow wide trees test whether a local child cap controls the child-pair
  product;
- a 1,300-node exact-oracle rare-anchor companion measures the partial beam's
  percent-of-optimum directly;
- a much larger deep rare-anchor stress case tests the same descendant-jump
  state space after dense and exhaustive sparse exact methods exceed the
  configured work limits.

`expected_algorithm` is therefore a recorded hypothesis, not a hard-coded
winner. The summary reports both the fastest warm/preprocessed method and the
fastest setup-plus-warm-search method satisfying the selected quality
criterion. A method is eligible only if it finishes every intended instance in
the regime.

Not every comparator has a deliberately favorable regime. In particular,
`sparse_closure` is retained to determine whether the newer direct chain solver
dominates it; the suite does not manufacture an artificial case merely to make
the older implementation win.

## Accuracy, timing, and combined scores

Approximate methods are allowed to return suboptimal answers; that is the
intended beam-search tradeoff. When one or more exact implementations complete
an instance, the runner reports

```text
score_ratio = returned_score / exact_optimal_score
accuracy_score = clip(score_ratio, 0, 1)
accuracy_percent = 100 * accuracy_score
```

Disagreement among methods marked exact is a different issue. If their scores
differ by more than the configured absolute/relative tolerance, the exact
oracle for that instance is declared invalid. No accuracy ratio or quality
ranking is then produced from it. The raw table records `oracle_status`,
`oracle_valid`, `exact_method_count`, `exact_score_spread`, and the applied
agreement tolerance.

There is deliberately **no planted-path-recall fallback**. Planted pair and node
recovery remain optional raw diagnostics, but they are not used as measures of
matching accuracy. A stress instance without an exact oracle is a scalability
experiment, not a quality-ranked experiment.

For each regime-instance, the runner also normalizes runtime as

```text
warm_timing_score = fastest_successful_warm_time / method_warm_time
setup_timing_score = fastest_successful_setup_time / method_setup_time
```

so the fastest successful method has timing score 1. The harmonic means of
accuracy and timing are reported as secondary, explicitly arbitrary combined
scores. The headline comparison remains: among methods that finish every
instance and attain the requested minimum accuracy on every instance, which is
fastest?

The raw table also records:

- the returned score and score gap from the exact oracle;
- matched-path length;
- planted pair precision/recall and planted-node recall as supplementary
  diagnostics only;
- all diagnostics exposed by the matcher.

## Timing convention

The quick presets run in-process because they are correctness and wiring checks.
The standard presets use `execution_mode="isolated"`: every algorithm-instance
(pairwise) or algorithm-regime (throughput) job runs in a fresh subprocess.
Before that process imports numerical libraries, the runner fixes the common
OpenMP, BLAS, NumExpr, and Numba thread-count environment variables to the
configured value (one in the supplied standard presets).  The subprocess also
provides an enforceable whole-job wall-clock timeout and a lifetime peak-RSS
measurement.

Inside each successful worker, the algorithm is run in three phases:

1. a diagnostics-disabled matcher performs one first-use search before any
   benchmark diagnostics or warm-up; this supplies `cold_predict_seconds` or
   `cold_score_matrix_seconds` and the corresponding cold setup fields;
2. a diagnostics-enabled pass records candidate counts, generated states,
   pruning, DP cells, and internal timing fields;
3. a separate diagnostics-disabled matcher is warmed and then timed repeatedly.

The first-use measurement is genuinely process-isolated, although it naturally
includes JIT compilation and first-use allocation when the implementation has
those costs.  The warm samples are summarized by median, minimum, maximum, mean, quartiles,
standard deviation, median absolute deviation, and the complete raw sample
list.  Pairwise columns use the prefix `predict_seconds_`; throughput columns
use `score_matrix_seconds_`.  `setup_plus_warm_search_seconds` and
`setup_plus_one_matrix_seconds` add reusable setup once to the corresponding
warm median.

The isolated-process fields have deliberately different semantics:

```text
cold_predict_seconds / cold_score_matrix_seconds
cold_setup_plus_predict_seconds / cold_setup_plus_one_matrix_seconds
fresh_process_total_seconds
fresh_process_task_seconds
fresh_process_overhead_seconds
peak_rss_bytes
peak_rss_mib
```

`fresh_process_total_seconds` covers the complete worker lifetime: interpreter
startup, imports, payload loading, diagnostics, matcher construction, warm-up,
all repeated timed calls, and result serialization.  It is therefore a robust
whole-job cost, not a single-call cold-start latency.  Peak RSS also includes the
runtime and imported libraries; compare it only within the same controlled
machine and environment.  The `resource.ru_maxrss` source and unit conversion
are recorded explicitly, and memory can be unavailable on unsupported
platforms.

A timed-out worker is retained as `status="timeout"` and makes coverage
incomplete.  Known score incompatibilities and declared work limits are checked
before process launch and remain explicit `skipped` rows.  They do not falsely
claim that worker thread settings were enforced.

The pairwise scheduler starts from a seeded permutation and rotates it across
instances.  Throughput uses a seeded permutation within each regime.  Isolation
removes Python-state and JIT carry-over between algorithms, while the saved
execution position still permits checking for residual machine-order effects.
Final timing claims should use several seeds or repeated complete runs on a
quiet machine; the supplied standard run is a baseline measurement, not a
hardware-independent fact.



## Large notebook presets and controlled sweeps

The notebook-facing presets now have three scales:

- `smoke`: wiring and correctness only;
- `standard`: the first controlled scientific matrix;
- `large`: larger trees, more independent instances or query-template pairs,
  five timing repeats, and longer isolated-worker limits.

The large pairwise and throughput configurations are
`experiments/path_match_benchmark_large.json` and
`experiments/path_match_throughput_large.json`.

The separate `experiments/path_match_sweeps_large.json` configuration expands
one base regime at a time. Size sweeps vary the tree size while fixing the score
model and algorithm presets. Algorithm sweeps use aliases such as
`partial_B200`, so every beam width or expansion budget remains explicit in the
saved rows. Run them with `experiments/run_path_match_sweeps.py`; all ordinary
correctness gates and isolated timing rules still come from
`run_benchmark_config`.

Talk-ready plots are produced by `benchmarking/make_benchmark_figures.py`. It
reads saved CSV files, directories, or ZIP archives and never reruns a matcher.


## Safety limits

Two suite-level limits currently prevent accidental large exact jobs:

```json
{
  "max_dense_cells": 5000000,
  "max_sparse_posting_hits": 5000000
}
```

- `max_dense_cells` skips `exact_dense` and `fast_dense` above the limit.
- `max_sparse_posting_hits` skips the exhaustive sparse methods when blocking
  buckets already imply too many candidate occurrences.

A skipped row remains in `benchmark_rows.csv` with an explicit reason. Beam
methods are not automatically skipped by these limits because their actual work
depends on the frontier and expansion budgets; large new regimes should be
introduced first in the quick suite or with conservative beam parameters.

## Outputs

Each run writes:

```text
benchmark_rows.csv
benchmark_summary.csv
benchmark_metadata.json
```

The row table is the primary record. It contains the complete algorithm keyword
arguments, regime dimensions, tree statistics, quality measurements, timings,
and matcher diagnostics. The metadata file contains the full input configuration, its stable hash, and
the Python/platform/NumPy/Pandas/Numba versions plus common thread-count
environment variables.

## Minimal custom configuration

```json
{
  "suite_name": "my_suite",
  "repeats": 3,
  "warmup": 1,
  "quality_floor": 0.99,
  "exact_score_abs_tol": 0.00001,
  "exact_score_rel_tol": 0.000001,
  "algorithm_order_seed": 20260928,
  "algorithms": [
    "exact_dense",
    "sparse_chain",
    {
      "name": "beam_partial_score",
      "alias": "beam_partial_B500_E128",
      "kwargs": {
        "beam_width": 500,
        "beam_expansion_width": 128
      }
    }
  ],
  "regimes": [
    {
      "name": "tree_to_path_example",
      "n_g": 2000,
      "n_h": 80,
      "shape_g": "narrow",
      "shape_h": "path",
      "comparison": "tree_path",
      "alphabet_size": 50,
      "symbol_distribution": "zipf",
      "symbols_per_node": {
        "kind": "poisson",
        "mean": 1.5,
        "min": 1,
        "max": 4
      },
      "score_mode": "jaccard",
      "planted_length": 20,
      "n_instances": 5,
      "seed": 12345
    }
  ]
}
```

Supported `symbols_per_node` forms are an integer or one of:

```json
{"kind": "fixed", "value": 2}
{"kind": "categorical", "values": [1, 2, 3], "probs": [0.6, 0.3, 0.1]}
{"kind": "poisson", "mean": 1.5, "min": 1, "max": 4}
```

## Repeated query-to-template throughput

The throughput backend in `src/benchmarking/path_match_throughput.py` models a
fixed template bank queried by many observed trees.  The corresponding presets
are:

```text
experiments/path_match_throughput_quick.json
experiments/path_match_throughput_full.json
```

For matchers with reusable representations, the benchmark performs the work in
four explicit stages:

1. fit one shared label encoder when required;
2. prepare every query tree once;
3. prepare every template once;
4. compute the complete query-template score matrix one or more times.

The ordinary and prepared APIs are compared on a deterministic sample before a
timing result is accepted.  The saved timing fields include:

```text
encoder_fit_seconds
query_preparation_seconds
template_preparation_seconds
one_time_preparation_seconds
score_matrix_seconds_median
setup_plus_one_matrix_seconds
pairs_per_second
queries_per_second
```

`score_matrix_seconds_median` is the warm operational measurement after reusable
setup.  `setup_plus_one_matrix_seconds` adds that setup once and is the more
relevant number when a corpus and template bank are used only once.

All completed exact score matrices are compared entry by entry.  When they
agree, each approximate method receives both aggregate and per-pair quality
summaries:

```text
matrix_total_accuracy
mean_pair_accuracy
median_pair_accuracy
min_pair_accuracy
exact_pair_fraction
```

The headline throughput comparison requires `min_pair_accuracy` to meet the
configured floor.  This is intentionally stricter than attaining a good total
score while failing badly on a few query-template pairs.  Aggregate matrix
accuracy, normalized timing, and their secondary harmonic mean are retained as
separate descriptive columns.

Each run writes:

```text
throughput_rows.csv
throughput_summary.csv
throughput_metadata.json
throughput_score_matrices.npz
```

The standard preset contains common scalar labels with path templates, sparse
multi-token overlap against a bank of larger tree templates, and generic
Jaccard tree-to-path matching.  Separate limits for
generic dense work, specialized dense work, and sparse posting-list hits prevent
an accidental very large run.

## Current measurement status and deliberate next steps

The supplied standard presets now include both an exact-oracle rare-anchor
companion and a larger oracle-free stress case.  They enforce thread counts,
isolate algorithm jobs, apply wall-clock timeouts, collect peak resident memory,
and retain uncertainty summaries for every repeated warm timing.  The next benchmark work is therefore
interpretive rather than infrastructural: inspect the first standard results,
calibrate any regime that is accidentally trivial or pathological, and then run
small one-factor-at-a-time beam-width, expansion-budget, and ranking-heuristic
ablations.

The optional lookahead implementation remains unchanged pending a separate
design discussion.  The thresholded sparse method and anchor/corridor
refinement also remain future algorithms; they should be added only after their
candidate-generation and stopping rules are specified.  Each should enter the
registry as a new named configuration rather than changing an existing preset
silently.

## Notebook interfaces

The interactive pairwise ranking entry point is
[`../benchmarking/01_algorithm_ranking.ipynb`](../benchmarking/01_algorithm_ranking.ipynb).
It exposes only the preset scale, selected regimes, instance count, timing
repetitions, and random seed.  Algorithm parameters remain fixed by the named
presets.  The notebook stops on implementation errors or exact-oracle
disagreement, reports approximate accuracy as percent of the agreed optimum,
and keeps timing, accuracy, and the secondary harmonic score separate.

The repeated-workflow entry point is
[`../benchmarking/02_throughput.ipynb`](../benchmarking/02_throughput.ipynb).
It exposes only preset scale, selected regimes, query count, template count,
timing repetitions, and random seed.  It separates one-time preparation from
warm matrix search and archives the complete score matrices used for exact
agreement and accuracy calculations.
