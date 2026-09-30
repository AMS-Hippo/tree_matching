# Path-matching implementation and benchmark status

Status date: 2026-09-29.

This note records the cumulative engineering work and the first controlled
standard-preset measurements.  Timings are development-machine measurements,
not hardware-independent claims.  The row-level CSV files and complete
configuration remain the authoritative records.

## Implementation status

| Area | Status |
|---|---|
| Generic exact dynamic program | Preserved; diagnostics added; randomized regression against the original repository passed. |
| Specialized dense equality/overlap matcher | Preserved; diagnostics and prepared-workflow timing added. |
| Partial-matching beam (Algorithm 7 state space) | Original implementation audited; deterministic score-only benchmark preset added; randomized exhaustive checks agree with exact DP when candidate generation and retention are exhaustive. |
| Local-transition beam (Algorithm 6) | Implemented with uncapped and child-capped presets; randomized no-pruning checks agree with exact DP. |
| Original sparse skip-closure DP | Preserved as `sparse_closure` for comparison. |
| Direct sparse product-poset DP | Implemented as `sparse_chain`; exact for a complete positive candidate set. |
| Specialized sparse equality/overlap | Implemented as `FastSparseTreePathMatcher`, including reusable encoder and tree preparation. |
| API safeguards | Unseen-label warnings, prepared-encoder fingerprints, and changed-blocking-score warnings implemented. |
| Diagnostics | Common timing, candidate, DP, frontier, and sparse-chain counters implemented. |
| Synthetic benchmark generator | Narrow/medium/wide/path shapes, tree--tree/tree--path comparisons, uniform/Zipf labels, variable symbols per node, and randomized sibling order implemented. |
| Pairwise benchmark notebook | `benchmarking/01_algorithm_ranking.ipynb` implemented. |
| Throughput benchmark notebook | `benchmarking/02_throughput.ipynb` implemented. |
| Timing hardening | Standard presets use fresh single-thread workers, first-use and warm timings, wall-clock timeouts, peak RSS, balanced scheduling, and repeated-sample uncertainty summaries. |

The complete test suite currently passes.  Matching behavior on the original
exact, fast-dense, partial-beam, sparse-closure, and repeating-template routes
was also compared directly with the original repository on randomized inputs;
no regression was found.

## First standard pairwise run

All 152 requested algorithm-instance rows completed or were intentionally
skipped by a declared incompatibility/work limit.  There were no implementation
errors, timeouts, or disagreements among completed exact methods.

| Regime | Fastest method meeting the 99% per-instance quality floor | Main observation |
|---|---|---|
| Tiny generic dense score | `exact_dense` | Sparse indexing and beam overhead do not pay at this size. |
| Dense common-token overlap | `fast_dense` | Warm prediction is much faster than the generic or sparse exact routes; first use is materially slower because of compilation. |
| Generic sparse Jaccard | `sparse_chain` | Direct sparse-chain search is substantially faster than both generic dense DP and the older sparse closure. |
| Large sparse overlap | `fast_sparse` | It is faster than generic `sparse_chain`, although both are exact and the difference is moderate. |
| Narrow tree to path | `beam_local` | The uncapped local beam is fastest and has minimum accuracy about 99.6%; the arbitrary child cap falls below the 99% floor. |
| Wide shallow tree pair | `exact_dense` | After neutralizing child-order bias, neither local beam is reliably accurate.  The capped version is extremely fast but loses too much score. |
| Deep rare-anchor stress | No exact oracle at this scale | `beam_partial_score` recovers all planted rare anchors and scores roughly 14,000, versus roughly 1,000 for the local beams. The supplemental exact-oracle companion below confirms that the same state-space advantage persists on a smaller instance where percent-of-optimum is measurable. |

The wide-tree result is an important negative finding: a child cap based only on
arbitrary node order is not a good general search heuristic.  A useful capped
local method will need a meaningful child-ordering rule or a diversity rule;
the generator should not be tuned to hide this failure.

The first-use/warm split also matters.  In the dense common-overlap pair regime,
`fast_dense` takes about 0.21 seconds on first prediction and about 0.011 seconds
warm.  For most non-JIT methods, first-use and warm times are close.

### Rare-anchor exact-oracle companion

The new `rare_anchor_oracle` regime uses two 1,300-node, depth-160 trees,
14 reserved high-weight anchors, and three independent instances.  All 24
algorithm-instance jobs completed in isolated single-thread workers, with no
timeouts, errors, or exact-method disagreements.

The deterministic score-only partial beam obtained accuracies of 99.35%,
99.51%, and 99.70%, and recovered all 14 planted anchor pairs in every
instance.  The uncapped and capped local beams obtained 28.73%, 78.99%, and
100% on the same instances.  Thus the companion does what the larger stress
case could not: it verifies against an exact score oracle that direct
descendant jumps can preserve delayed rare evidence that local layer-by-layer
search sometimes loses.  At this scale `fast_dense` remains much faster than
the approximate methods, so this is a state-space/accuracy benchmark rather
than a claim that the partial beam is the fastest solver for 1,300-node trees.

## First standard throughput run

All 24 requested algorithm-regime rows completed or were intentionally skipped.
There were no implementation errors, timeouts, or exact-matrix disagreements.

| Regime | Fastest method meeting the 99% worst-pair quality floor | Warm complete-matrix time |
|---|---|---:|
| 40 query trees × 12 common-label path templates | `fast_dense` | about 0.032 s |
| 6 large query trees × 4 sparse-overlap tree templates | `fast_sparse` | about 0.340 s |
| 20 generic-Jaccard query trees × 8 path templates | `sparse_chain` | about 0.092 s |

In the sparse-overlap throughput regime, `fast_sparse` and `sparse_chain` are
close (about 0.340 versus 0.349 seconds).  The specialized implementation is not
yet a dramatic improvement there.  In contrast, the generic Jaccard throughput
regime strongly favors `sparse_chain` over both dense DP and the older sparse
closure.

The beams are not uniformly competitive in these throughput presets.  They are
exact on the short generic-Jaccard path-template workload but slower than
`sparse_chain`; on the larger sparse-overlap workload their score losses are
large.  This is useful baseline evidence before heuristic tuning.

## Work deliberately not done yet

1. **Fast specialized beam implementation.**  Both beam state spaces currently
   use the generic Python implementation.  A separate integerized fast path
   remains planned, with the partial-matching beam the higher priority.
2. **Lookahead redesign.**  The existing optional lookahead implementation was
   not changed.  Its representation, candidate-generation role, ranking role,
   preprocessing amortization, and cost budget should be agreed before editing
   it.
3. **Thresholded sparse matching.**  The proposed thresholded candidate pass
   with an additive score certificate has not been implemented.
4. **Anchor-chain/corridor refinement.**  The idea is promising, especially for
   rare important labels plus many common low-weight matches, but its top-K,
   diversity, suffix, and stopping rules still need a design pass.
5. **Beam-stack search.**  It was assessed as most natural over the local beam,
   but lower priority than sparse and anchor work.
6. **Real ACME benchmark.**  The repository does not contain the ACME4 data.
   The current notebooks are ready for a grouped-by-source-tree real-data
   benchmark once a reproducible data/conversion path is supplied.
7. **Heuristic ablations.**  Beam width, candidate budget, rarity, future-score,
   gap/balance, diversity, and eventual lookahead should be varied one at a time
   only after the baseline results are accepted.

## Near-term order

1. Review the two standard result sets and decide whether the current regimes
   are scientifically representative.
2. Make the small notebook-interface pass needed for user-chosen large runs
   (tree-size overrides and non-overwriting run labels), then archive the
   current baseline on GitHub.
3. Run one-factor beam-width and candidate-budget sweeps for the score-only
   baselines.
4. Design the thresholded sparse and anchor-chain refinements.
5. Add real grouped ACME workloads when the data pipeline is available.
