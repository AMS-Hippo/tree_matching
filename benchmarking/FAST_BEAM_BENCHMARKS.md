# Benchmarking the generic and encoded partial beam

This is a benchmark/reporting update, not a change to either matching algorithm.
The main presets already registered both implementations; this update adds
explicit paired score/speed comparisons to both notebooks and saved outputs.

## 1. Run the main notebooks

From the repository root, with the editable virtual environment active:

```bash
python -m pytest -q
python -m notebook benchmarking/01_algorithm_ranking.ipynb
```

Run `02_throughput.ipynb` in the same way. In each notebook's settings cell use:

```python
SCALE = "large"
RUN_LABEL = "fast_beam_v2"
REGIMES = "all"
ALGORITHMS = "all"
```

Leave other controls unchanged for a direct update of the existing benchmark.
For a quick installation check first use `SCALE="smoke"` and a unique label.
The notebooks now refuse a nonempty output directory: choose a new `RUN_LABEL`
for a repeat, rather than overwriting a completed or partly completed run.

Both `beam_partial_score` and `fast_beam_partial` are kept. The other algorithms
also remain in the preset. Fast equality/overlap methods are explicitly skipped
on Jaccard; a skipped method is not counted as successful coverage.

The final table compares generic and encoded **exact DP**, then generic and
encoded **partial beam**. Each run now writes:

- `implementation_comparison_rows.csv`: per-instance paired timings, scores,
  completion status, and score differences; throughput checks every matrix entry.
- `implementation_comparison_summary.csv`: coverage, paired median/quartile
  speedups, setup speedups, and score-agreement counts.

A speedup is generic time divided by encoded time. The table's median speedup
is a median of paired ratios, not a ratio of separately aggregated medians.
Agreement between two approximate beams does not establish optimality. The
ordinary exact-oracle accuracy columns remain the source for percent of optimum.
Equality should be compared with matched search budgets. Finite-budget overlap
uses different candidate rules; it is not a pure implementation-only ablation.

## 2. Focused exact-and-beam implementation sweep

A ready-to-run file is included; no file copying or manual JSON filtering is needed.
It retains the existing dense-overlap exact-DP sweep and adds all four methods to
the rare-anchor implementation sweep: `exact_dense`, `fast_dense`,
`beam_partial_score`, and `fast_beam_partial`. Sizes run through 5,000 per tree;
the dense-work limit accommodates an exact oracle there, subject to the worker
timeout. This is more work than the old two-beam-only sweep.

```bash
python experiments/run_path_match_sweeps.py --config experiments/path_match_implementation_comparison.json --output-dir benchmarking/results/implementation_comparison_v2
```

The existing `path_match_sweeps_large.json` is left unchanged. It still runs all
size and parameter sweeps. The new focused file avoids repeating unrelated work.
The backend saves paired comparisons both per point and for the combined sweep.
Use a new output directory for every sweep rerun; only the notebook entrypoints
have the nonempty-directory guard in this update.

## 3. Regenerate figures without rerunning matching

Run this command ONCE as one line:

```bash
python benchmarking/make_benchmark_figures.py --pairwise benchmarking/results/algorithm_ranking_large_fast_beam_v2 --throughput benchmarking/results/path_match_throughput_large_fast_beam_v2 --sweeps benchmarking/results/implementation_comparison_v2 --outdir benchmarking/results/figures_fast_beam_v2
```

The existing exact comparison `03_exact_generic_vs_encoded.*` is retained.
There are new beam comparisons:

```text
06_partial_generic_vs_encoded_pairwise.png/.pdf
06_partial_generic_vs_encoded_throughput.png/.pdf
```

The sweep figures show all selected methods, including both old and new beams.
The bar/dot comparison figures use ratios of summary medians; use the paired CSV
for median paired speedups and score-agreement checks. Every figure's source
numerical results are saved. Cosmetic revisions of the older multi-panel plots
are deliberately deferred.

## 4. Preserve results in Git (important)

`benchmarking/results/` is still ignored. A normal code commit does NOT include
its CSV/JSON/NPZ files. Local saving and pushing results are separate actions.

The explicit archive step below copies completed numerical results into
`benchmarking/recorded_runs/fast_beam_v2/`, leaving originals unchanged. It keeps
all CSV, JSON, NPZ/NPY and log files, including nested sweep-point results and
score matrices. It excludes PNG/PDF figures, notebook outputs, and caches.
File hashes and a manifest are added. It refuses missing required artifacts,
symlinks, overwriting an existing snapshot, files over 50 MiB, or a total over
250 MiB; these last two are conservative storage safeguards. It does not commit
or push anything by itself.

```bash
python benchmarking/archive_benchmark_data.py --label fast_beam_v2 benchmarking/results/algorithm_ranking_large_fast_beam_v2 benchmarking/results/path_match_throughput_large_fast_beam_v2 benchmarking/results/implementation_comparison_v2
```

To preserve the earlier successful two-beam run without rerunning it:

```bash
python benchmarking/archive_benchmark_data.py --label fast_partial_beam_v1 benchmarking/results/fast_partial_beam_v1
```

Archived directories have the same summary filenames as originals. They can be
passed directly to `make_benchmark_figures.py` after cloning on another machine.
Archive manifests identify Git HEAD at *archive* time, not necessarily the source
revision of an older run. Original run metadata is preserved byte-for-byte.

Only synthetic results are intended here. Review metadata and paths before
pushing; real or confidential data need a separate storage/privacy plan.

## 5. Commit and push

First save numerical results (step 4). Clean only the two tracked notebook outputs
before staging: this does not delete any saved CSV/JSON/NPZ files.

```bash
jupyter nbconvert --to notebook --ClearOutputPreprocessor.enabled=True --inplace benchmarking/01_algorithm_ranking.ipynb benchmarking/02_throughput.ipynb
python -m pytest -q
```

Your earlier status included *tracked* Python/Numba caches and egg-info. Adding
ignore rules alone does not untrack them. The following removes only generated
paths from the Git index, NOT from disk, and appends ignore rules without replacing
existing rules. It makes no commit and does not touch benchmark data.

```bash
python - <<'PY'
from pathlib import Path
import subprocess
tracked = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
generated = [p for p in tracked if p and (
    "__pycache__" in Path(p).parts
    or any(part.endswith(".egg-info") for part in Path(p).parts)
    or Path(p).suffix in {".pyc", ".pyo", ".nbc", ".nbi"}
)]
for start in range(0, len(generated), 100):
    subprocess.run(["git", "rm", "--cached", "-f", "--", *generated[start:start+100]], check=True)
p = Path(".gitignore")
old = p.read_text() if p.exists() else ""
rules = ["__pycache__/", "*.py[cod]", "*.nbc", "*.nbi", "*.egg-info/", ".pytest_cache/", ".venv/", "build/", "dist/"]
new = [rule for rule in rules if rule not in old.splitlines()]
if new:
    p.write_text(old.rstrip("\n") + "\n\n# Generated Python/build artifacts\n" + "\n".join(new) + "\n")
print("Generated files remain on disk; only their Git tracking was removed.")
PY
```

Stage source/configuration/documentation and the chosen numerical snapshot. These
commands also include the previous uncommitted fast-beam source changes:

```bash
git add -- .gitignore src/path_matcher/*.py src/benchmarking/*.py tests/*.py experiments/*.py experiments/*.json benchmarking/*.py benchmarking/*.md docs/*.md
git add -- benchmarking/01_algorithm_ranking.ipynb benchmarking/02_throughput.ipynb benchmarking/recorded_runs/README.md
git add -f -- benchmarking/recorded_runs/fast_beam_v2
# Also stage the older snapshot if created in step 4:
# git add -f -- benchmarking/recorded_runs/fast_partial_beam_v1

git diff --cached --stat
git diff --cached --check
git status --short
git commit -m "Compare generic and encoded beams and preserve benchmark data"
git push -u origin "$(git branch --show-current)"
```

Review staged changes before committing, especially any changes unrelated to this
work. The `-f` applies ONLY to the deliberately prepared numerical snapshot, never
to the scratch-results directory. The push uploads the snapshot after it is staged
and committed. No force-push or history rewrite is needed.
