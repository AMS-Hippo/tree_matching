# Matcher diagnostics

Diagnostics are deliberately opt-in. They are intended for algorithmic audits,
heuristic ablations, and benchmark development; ordinary matching calls retain
their existing return type.

## High-level use

```python
from path_matcher import TreePathMatcher

matcher = TreePathMatcher(
    method="beam",
    collect_diagnostics=True,
    beam_width=200,
)

pairs, score = matcher.predict(G, H)
diag = matcher.last_diagnostics_
print(diag.as_dict())
```

The specialized fast matchers use the same switch:

```python
from path_matcher.fast_match import FastTreePathMatcher
from path_matcher.fast_sparse_match import FastSparseTreePathMatcher

fast = FastTreePathMatcher(mode="equality", collect_diagnostics=True)
sparse = FastSparseTreePathMatcher(mode="equality", collect_diagnostics=True)
```

`fit(...)` and `predict(...)` are recorded separately. After `fit(...)`,
`last_fit_diagnostics_` describes conversion, integerization, and reusable
per-tree preprocessing. After `predict(...)`, `last_diagnostics_` describes the
pairwise search. A fitted prediction records zero
`input_preprocessing_seconds`; a prediction supplied with raw inputs includes
conversion/preprocessing in that field.

## Low-level use

Every principal alignment function accepts an optional mutable record:

```python
from path_matcher import MatchDiagnostics
from path_matcher.needleman_wunsch_tree import align_trees_algorithm1

record = MatchDiagnostics()
result = align_trees_algorithm1(G, H, diagnostics=record)
```

The function resets and fills the supplied object. The matching result is
unchanged.

## Common fields

The named timing phases are wall-clock seconds:

- `input_preprocessing_seconds`: graph conversion, label encoding, or reusable
  tree preprocessing performed by a high-level matcher for this call;
- `preprocessing_seconds`: pair-local structural indices and tables;
- `candidate_generation_seconds`: generation, scoring, and deduplication of
  candidate pairs;
- `search_seconds`: the remaining dynamic-programming or search work;
- `traceback_seconds`: recovering the selected matching;
- `total_seconds`: the complete measured call.

The phase timings are diagnostic partitions, not a replacement for a dedicated
benchmark timer. In particular, the first call to a Numba-backed fast matcher
may include compilation in `search_seconds`. Warm the relevant kernel before
collecting runtime comparisons.

Fields that cannot be obtained without extra asymptotic work are left as `None`;
zero means that the field applies and no such work occurred.

The main work counters are:

- `node_pair_score_evaluations`: logical node-pair score evaluations in the
  search or candidate generator. It deliberately excludes score reads used
  only to filter a traceback;
- `candidate_pairs_generated`: candidate match pairs made available to the
  algorithm. For a partial-matching beam, this is after its per-state expansion
  cap. For local beam search, it counts match transitions and excludes skip
  transitions;
- `positive_candidate_pairs`: positive candidate pairs encountered or retained,
  depending on the algorithm;
- `dp_cells_computed`: dynamic-programming cells actually computed. It is `nm`
  for dense methods and may be much larger than the candidate count for the
  sparse skip-closure method.

Beam-specific counters are cumulative across processed layers and exclude the
initial sentinel state:

- `states_generated`: successor states offered before endpoint pruning;
- `states_after_endpoint_pruning`: terminal-pair winners;
- `states_retained`: states admitted to a post-pruning beam;
- `peak_frontier_size`: largest live frontier, including the initial frontier;
- `layers_processed`: layers whose frontier was processed.

For stochastic restarts or symmetric search, cumulative counters and phase
times include every executed run. `result_score` and `result_length` describe
the selected output. The record also states `restarts` and `directions`.

## Algorithm identifiers

The current `algorithm` values are:

| Matcher | Identifier |
|---|---|
| Dense generic DP | `exact_dense` |
| Local-transition beam (Algorithm 6) | `beam_local` |
| Partial-matching beam (Algorithm 7) | `beam_partial` |
| Symmetric partial-matching beam | `beam_partial_symmetric` |
| Sparse skip-closure DP | `sparse_closure` |
| Sparse product-poset chain | `sparse_chain` |
| Specialized dense equality/overlap | `fast_dense_equality`, `fast_dense_overlap` |
| Specialized sparse equality/overlap | `fast_sparse_equality`, `fast_sparse_overlap` |

The template-repeat exact modes use separate `exact_template_repeat...`
identifiers. Fit records append `_fit` at the high-level API.

## Algorithm-specific details

The `extra` dictionary contains useful non-universal quantities. Current
examples include:

- beam width, expansion width, state-pool size, selected restart, and lookahead
  cache size;
- whether sparse candidates were generated internally;
- sparse-chain point queries, interval insertions, and rollback-history size;
- raw posting-list hits before token-overlap candidate deduplication.

These keys are supplementary and may grow as benchmark requirements become
clear. Code should use the standardized top-level fields for cross-algorithm
comparisons.

## Timing discipline

Instrumentation adds counter updates and fine-grained calls to
`perf_counter()`. For final runtime tables, use two passes:

1. a diagnostic pass to verify that algorithms performed comparable work and
   that budget parameters were applied as intended;
2. a clean timing pass with `collect_diagnostics=False`, after all JIT warm-up.

This avoids mistaking instrumentation or compilation overhead for an
algorithmic difference.
