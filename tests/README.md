# Tests

Run the pytest regression tests with:

```bash
python -m pytest tests
```

The project uses a `src/` layout. Pytest is configured in `pyproject.toml` to add `src/` to the test import path for local source-tree runs.

The snapshot-style reference runner is still available:

```bash
python tests/run_reference_match_tests.py
```

Regenerate the reference fixture only when intentionally updating expected behavior:

```bash
python tests/prep_reference_match_cases.py
```
