# Merge audit: beam, sparse, diagnostics, and benchmarking work

## Correct source archive

This merge is based on the user-supplied archive:

```text
tree_matching-main(3).zip
SHA-256: 915baf9aedfcc07b801d2a391ca4c3c557466621ebbfa7c6db18bdea35ddd29b
ZIP comment / source commit: 94e8ba769acf032cf9f06dc800da8539bc98d379
```

The archive is byte-for-byte identical to `tree_matching-main(2).zip`, the
matcher repository supplied at the beginning of this work.  The two archives
have the same SHA-256 digest, the same ZIP comment, the same file list, and the
same file contents.  Therefore there are **no intervening source changes or bug
fixes to reconcile** between the original starting point and the newly supplied
correct archive.

The previously supplied `tree_matching_expts-main.zip` is a different,
downstream workflow repository.  It is not used as the merge base here.

## Consequence for the earlier temporary rebase

A prior provisional merge was prepared against a different matcher snapshot
whose ZIP comment was `0f5110b85214604ed754b99b5c651753f60c684e`.
That snapshot had one unrelated change to zero-weight traceback behavior in the
generic exact matcher.  Because the correct archive supplied now is exactly the
original `94e8...` source, this final merge intentionally follows the supplied
source and does **not** import that unrelated traceback change or its regression
test.

## What was merged

The cumulative work from this chat was merged onto the exact `94e8...` source,
including:

- the local-transition beam implementation;
- the existing partial-matching beam plus stronger tests and corrected API
  documentation;
- the sparse product-poset chain algorithm;
- the specialized sparse equality/overlap matcher;
- diagnostics and resource counters;
- synthetic benchmark generators and reproducible algorithm configurations;
- pairwise-ranking and repeated-throughput benchmark backends;
- timing hardening, exact-oracle checks, completion requirements, timeouts, and
  memory reporting;
- the exact-oracle rare-anchor companion regime;
- the revised notebook controls for run labels, regime/algorithm subsets, and
  per-regime tree-size overrides.

No existing lookahead heuristic was modified.

## Notebook merge

The notebook revisions were applied directly to the cumulative `94e8...` tree.
The first settings cells now support:

- `RUN_LABEL`;
- `REGIMES`;
- `ALGORITHMS`;
- `SIZE_OVERRIDES` for `n_g`, `n_h`, `depth_g`, and `depth_h`;
- the existing instance/query/template count, repetition, and seed controls.

A previously discovered subset-display bug remains fixed: a diagnostics column
that is absent after aggregation is checked before the notebook attempts to
display it.

## Merge safety

The correct source archive is identical to the original base, so the merge does
not require conflict resolution against new upstream code.  The prepared Git
patch applies cleanly to an exact extraction of `tree_matching-main(3).zip`.
The matching changes are separated into new modules and localized integrations;
the old exact, fast-dense, partial-beam, sparse-closure, and template-repeat
routes were regression-tested during development.

## Verification

The final verification results are recorded below after testing the exact
packaged tree.

```text
pytest: 66 passed
pytest with NUMBA_DISABLE_JIT=1: 66 passed
pairwise smoke notebook: passed
throughput smoke notebook: passed
wheel build: passed offline with --no-build-isolation
patch apply check against 94e8 source: passed
```
