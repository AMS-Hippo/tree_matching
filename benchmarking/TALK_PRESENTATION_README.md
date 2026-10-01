# Temporary talk presentation figures

One standalone file: `temporary_talk_presentation.py`. This adds no matcher,
notebook, config, or benchmark changes. It uses NumPy, Pandas, and Matplotlib,
which are already in the benchmark environment. The running notebooks need not
be restarted. Plot a run only after it finishes.

## Install and check

From the existing repository with its virtual environment active:

```bash
cp ~/Downloads/temporary_talk_presentation.py benchmarking/
python benchmarking/temporary_talk_presentation.py --self-test
```

The script is also at the top of this ZIP. Extract it there instead of downloading
it separately. No patch needs to be applied.

## Make the main figures

After the two new notebooks complete, from the repository root:

```bash
python benchmarking/temporary_talk_presentation.py --pairwise benchmarking/results/algorithm_ranking_large_fast_beam_v2 --throughput benchmarking/results/path_match_throughput_large_fast_beam_v2 --outdir benchmarking/results/talk_main_v1
```

Open `benchmarking/results/talk_main_v1/index.html` in a browser to choose among
the figures. Each plot is a separate 16:9 PNG/PDF. No eight-panel overview is
created. Use a new output directory for another rendering; overwriting is refused.

## Size and parameter sweeps

For the older, already completed full sweep:

```bash
python benchmarking/temporary_talk_presentation.py --sweeps benchmarking/results/path_match_sweeps_large_v1 --outdir benchmarking/results/talk_sweeps_v1
```

For the new exact-and-beam comparison, after that sweep finishes:

```bash
python benchmarking/temporary_talk_presentation.py --sweeps benchmarking/results/implementation_comparison_v2 --outdir benchmarking/results/talk_implementations_v1
```

The three input flags can also be combined in a single call. `--sweeps` can be
repeated for different completed runs: they receive separate file prefixes and
are never statistically pooled. Inputs may be run directories, summary CSVs, or
ZIPs containing exactly one summary of the requested kind. With a CSV, put its
raw/paired comparison CSVs beside it to enable paired speedup figures. Archived
run subdirectories work exactly like their original result directories.

## Figure choices

* **Performance table:** one row per method, a logarithmic timing dot and (when
  available) interquartile whisker, plus typical and worst objective-score ratios.
  Diamonds mark the empirical efficient frontier. Failed/incomplete methods are
  shown but cannot be frontier members. The table is split after nine rows rather
  than shrinking fonts.
* **Accuracy-requirement table:** the fastest completed method attaining 90%,
  95%, 99%, or 100% of the reference score. This is a readable view of the
  lower envelope of the measured choices, not an interpolated algorithm.
* **Paired implementation speedups:** generic/encoded exact DP and generic/encoded
  partial beam stay separate. The statistic is a median of paired ratios, not a
  ratio of medians. Score-agreement checks are printed when available. Without
  raw/paired observations, this chart is omitted rather than inventing pairing.
* **Small head-to-head scaling views:** no more than four methods per figure,
  with additional two-method exact and partial-beam views when relevant. Runtime
  and score are separate files. Exactly coincident score curves are combined;
  all original method rows and group membership remain in the CSV. Near-optimum
  cases also get a percentage-point-loss alternative. Missing/incomplete cells
  break a curve, rather than being interpolated through.
* **Parameter studies:** runtime versus beam width/expansion/child budget, score
  versus that same parameter, and the readable frontier/requirement tables.
* **Optional preparation amortization:** add `--amortization`. This projects one
  measured setup plus R times measured warm matrix time for the SAME prepared
  workload. It is labelled as an accounting projection, not measured scaling in
  newly arriving queries. Only methods with >=99% worst-pair scores are shown.

For an abbreviated talk, choose one table, an exact-implementation comparison,
a sparse-chain scaling view, and a beam implementation/scaling view. The other
outputs are candidates or backup figures, not a recommendation to show all of them.

## A few display controls

`--quality minimum` (default) computes the frontier using the worst observed
instance ratio for pairwise runs, or worst observed pair ratio for throughput.
`--quality typical` instead uses median instance accuracy for pairwise runs and
sum(returned scores)/sum(exact optima) for throughput. The table always displays
both typical and minimum ratios. **Worst observed is not a guarantee on future
instances.** Neither criterion is planted-pair recall.

`--time warm` (default) versus `--time setup` selects warm time or the saved
setup-plus-warm-search measurement. Setup is not cold-process startup.

`--regimes NAME ...` and `--sweep-names NAME ...` limit the generated figure set.
`--formats png pdf svg` adds SVG. `--label 'Machine / run identifier'` prints a
short label on each figure. More detailed method curation is in `CURATED_VIEWS`
at the top of the script, not hidden inside benchmark code.

## Scientific and preservation safeguards

There is no accuracy point without a valid exact reference. Agreement between
beams does not supply one. Finite-budget overlap beams have different proposal
rules and are labelled accordingly. Frontier membership uses the observed timing
summaries; differences within timing noise are not declared statistically
significant. Interquartile ranges are NOT confidence intervals. Throughput timing
repeats concern one fixed corpus, not independent corpus replication.

A size sweep may change depth or both tree sizes as well. When raw columns are
available, the footer flags those changes. No asymptotic exponent is fitted.

The output includes `index.html`, `talk_manifest.csv`, `talk_manifest.json`, a
CSV of each figure's data, and byte-identical copies of the input CSVs in
`input_tables/`. The manifest records input hashes, script hash, library versions,
and plotting options. The source benchmark files are never altered. Original
NPZ score matrices remain in the run directories; this script does not duplicate
or remove them. Throughput agreement is never inferred just from matrix totals.

`benchmarking/results/` is still ignored by Git. Preserve the original numerical
runs using the existing `archive_benchmark_data.py` workflow in
`benchmarking/FAST_BEAM_BENCHMARKS.md`. Commit this script as an ordinary new source
file. A code-only push does not include ignored result/figure directories.

## Included previews and validation

The preview galleries use EARLIER SAVED DEVELOPMENT RESULTS, not measurements
from your currently running notebooks. They illustrate the layout only:

* `previews/earlier_standard/index.html`
* `previews/earlier_fast_partial/index.html`

No synthetic timing fixtures are included as empirical previews. Standalone
self-tests cover frontier dominance/ties, failed runs, missing oracles, Boolean
CSV parsing, paired speedups, duplicate detection, rendering, and overwrite/input
protection. Eight additional development tests exercised actual summary schemas,
raw and saved paired comparisons, ZIP reading, sweep curves, and line gaps.
Rendered PNGs and representative PDF pages were inspected. The script does not
run new scientific benchmarks or modify either matching implementation.
