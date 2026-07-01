# Documentation notebooks

Syntax-oriented notebooks live here. They are meant to be read and run from the repository root, but each notebook also has a small bootstrap cell for running from this folder against the `src/` layout.

Current notebooks:
- `SYNTAX_basic_matching.ipynb`
- `SYNTAX_partial_beam_matching.ipynb`
- `SYNTAX_embedding_clustering.ipynb`
- `SYNTAX_subgraph_matching.ipynb`
- `SYNTAX_template_repeat_matching.ipynb`
- `SYNTAX_neural_exemplar_extraction.ipynb`
- `UNITTEST_exact_matcher_dev.ipynb`


`SYNTAX_basic_matching.ipynb` and `SYNTAX_partial_beam_matching.ipynb` document `method="beam"` as a partial-matching beam search over valid matched prefixes, not the older row-wise sparse-DP approximation.

`SYNTAX_partial_beam_matching.ipynb` also shows the optional `beam_lookahead=True` mode, which precomputes capped discounted descendant-label sketches plus fixed-length downward path-chunk sketches and uses their overlap as a non-admissible far-lookahead ranking feature.
