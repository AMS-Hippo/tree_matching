# Local merge, testing, benchmark, and Git instructions

## 1. Source basis

This prepared tree and patch are based on:

```text
commit recorded in supplied ZIP: 94e8ba769acf032cf9f06dc800da8539bc98d379
ZIP SHA-256: 915baf9aedfcc07b801d2a391ca4c3c557466621ebbfa7c6db18bdea35ddd29b
```

The newly supplied correct ZIP is byte-identical to the matcher ZIP used at the
start of this work.  No intervening source change had to be merged.

## 2. Apply the patch to a Git checkout

Start from a clean checkout of the matcher repository at the source represented
by the supplied ZIP.  Create a branch before applying the patch:

```bash
git status --short
git rev-parse HEAD
git switch -c beam-sparse-benchmarks

# Use the path where you downloaded the prepared patch.
git apply --check /path/to/tree_matching_beam_sparse_benchmarks_from_94e8.patch
git apply /path/to/tree_matching_beam_sparse_benchmarks_from_94e8.patch
```

`git apply --check` should succeed without changing the checkout.  If it does
not, stop and inspect the branch or commit instead of forcing the patch.

An alternative is to inspect or copy the complete merge-ready source ZIP, but
the patch is preferable in an existing checkout because Git will expose any
unexpected conflict.

## 3. Create the Python environment

### Ubuntu or macOS

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,speedups,benchmark,viz]"
python -m pip install notebook ipykernel
python -m ipykernel install --user \
  --name tree-matcher \
  --display-name "Python (tree-matcher)"
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev,speedups,benchmark,viz]"
python -m pip install notebook ipykernel
python -m ipykernel install --user --name tree-matcher --display-name "Python (tree-matcher)"
```

If PowerShell blocks activation, run this once in the current shell and retry:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 4. Run the tests

```bash
pytest -q
```

The non-JIT path can also be checked:

```bash
NUMBA_DISABLE_JIT=1 pytest -q
```

On Windows PowerShell:

```powershell
$env:NUMBA_DISABLE_JIT = "1"
pytest -q
Remove-Item Env:NUMBA_DISABLE_JIT
```

A wheel build is an additional packaging check:

```bash
python -m pip install build
python -m build --wheel
```

## 5. Run the benchmark notebooks

Launch Jupyter from the repository root:

```bash
jupyter lab
```

or open a notebook directly:

```bash
jupyter notebook benchmarking/01_algorithm_ranking.ipynb
jupyter notebook benchmarking/02_throughput.ipynb
```

Select the `Python (tree-matcher)` kernel.

### Recommended first run: smoke checks

Leave the first settings cell at its defaults:

```python
SCALE = "smoke"
RUN_LABEL = None
REGIMES = "all"
ALGORITHMS = "all"
SIZE_OVERRIDES = {}
```

Then run all cells.  Smoke runs verify the wiring and correctness gates; they
are too small for performance conclusions.

### Standard pairwise algorithm ranking

In `01_algorithm_ranking.ipynb`, use for example:

```python
SCALE = "standard"
RUN_LABEL = "standard_v1"
REGIMES = "all"
ALGORITHMS = "all"
SIZE_OVERRIDES = {}
N_INSTANCES = None
REPEATS = None
SEED = 20260928
```

The standard preset includes:

```text
tiny_generic_dense
dense_common_overlap
generic_sparse_jaccard
large_sparse_overlap
narrow_tree_to_path
wide_shallow_tree_pair
deep_rare_anchor_stress
rare_anchor_oracle
```

For a targeted larger rare-anchor run:

```python
SCALE = "standard"
RUN_LABEL = "rare_anchor_2500_v1"
REGIMES = ["rare_anchor_oracle"]
ALGORITHMS = "all"
SIZE_OVERRIDES = {
    "rare_anchor_oracle": {
        "n_g": 2500,
        "n_h": 2500,
        "depth_g": 200,
        "depth_h": 200,
    },
}
N_INSTANCES = 3
REPEATS = 5
SEED = 20260928
```

Larger overrides can make exact or exhaustive methods expensive.  The backend
records explicit skips when configured work limits are exceeded, and standard
runs enforce worker timeouts, but it remains sensible to begin with one regime
and a small number of instances.

### Standard query-to-template throughput

In `02_throughput.ipynb`, use for example:

```python
SCALE = "standard"
RUN_LABEL = "standard_v1"
REGIMES = "all"
ALGORITHMS = "all"
SIZE_OVERRIDES = {}
N_QUERIES = None
N_TEMPLATES = None
REPEATS = None
SEED = 20260928
```

The standard throughput regimes are:

```text
equality_tree_to_path_common_labels
sparse_overlap_tree_template_bank
generic_jaccard_tree_to_path
```

For a larger sparse-overlap throughput run:

```python
SCALE = "standard"
RUN_LABEL = "sparse_overlap_large_v1"
REGIMES = ["sparse_overlap_tree_template_bank"]
ALGORITHMS = [
    "fast_dense",
    "sparse_chain",
    "fast_sparse",
    "beam_local",
    "beam_partial_score",
]
SIZE_OVERRIDES = {
    "sparse_overlap_tree_template_bank": {
        "n_g": 6000,
        "n_h": 3000,
    },
}
N_QUERIES = 8
N_TEMPLATES = 6
REPEATS = 5
SEED = 20260928
```

### Result locations

Pairwise notebook outputs are written under:

```text
benchmarking/results/algorithm_ranking_<scale>[_<RUN_LABEL>]/
```

Throughput notebook outputs are written under:

```text
benchmarking/results/path_match_throughput_<scale>[_<RUN_LABEL>]/
```

The folders contain row-level CSV files, summaries, metadata, the resolved
configuration, and—where applicable—compressed score matrices.  They are
ignored by Git by default.

### Headless notebook execution

After saving the desired settings in a notebook, it can be executed without the
browser interface:

```bash
jupyter nbconvert \
  --to notebook \
  --execute benchmarking/01_algorithm_ranking.ipynb \
  --output 01_algorithm_ranking.executed.ipynb \
  --ExecutePreprocessor.timeout=-1

jupyter nbconvert \
  --to notebook \
  --execute benchmarking/02_throughput.ipynb \
  --output 02_throughput.executed.ipynb \
  --ExecutePreprocessor.timeout=-1
```

The standard benchmark backend itself uses isolated worker processes and its
own configured per-job timeouts.  The `nbconvert` timeout is disabled above so
that the outer notebook process does not terminate a valid long benchmark.

### Pairwise command-line runner

The pairwise backend can also be run directly from its versioned JSON preset:

```bash
python experiments/run_path_match_benchmarks.py \
  --config experiments/path_match_benchmark_quick.json \
  --output-dir benchmarking/results/cli_smoke

python experiments/run_path_match_benchmarks.py \
  --config experiments/path_match_benchmark_full.json \
  --output-dir benchmarking/results/cli_standard_v1
```

Use the notebook when applying `SIZE_OVERRIDES`, selecting algorithm/regime
subsets, or changing counts without editing the checked-in JSON preset.

## 6. Review, commit, and push

After tests and a smoke benchmark pass:

```bash
git status --short
git diff --stat
git add -A
git commit -m "Add beam, sparse, diagnostics, and benchmark implementations"
git push -u origin beam-sparse-benchmarks
```

The generated benchmark-result directories are ignored.  Confirm with
`git status --short` that no large result files are staged before committing.
