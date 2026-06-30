# Functions used primarily in the demo notebooks, not in the "main" method files.

import igraph as ig
from path_matcher.matcher import TreePathMatcher  # adjust if needed
from path_matcher.tree_viz import (
    PlotStyle,
    plot_tree_with_path,
    build_discrete_label_colormap,
    vertex_colors_by_label,
    vertex_labels_from_ordered_vertices,
)

def score_fixed_paths_via_matcher(G, pathG, H, pathH, w, *, method="exact", **matcher_kwargs):
    """
    Score the *given* paths (pathG in G, pathH in H) under the same objective as TreePathMatcher,
    by running the matcher on path-only chain graphs.
    """
    def chain_from_path(orig, path):
        k = len(path)
        C = ig.Graph(n=k, edges=[(i, i+1) for i in range(k-1)], directed=True)
        C.vs["label"] = [orig.vs["label"][v] for v in path]
        return C

    Gc = chain_from_path(G, pathG)
    Hc = chain_from_path(H, pathH)

    m = TreePathMatcher(method=method, w=w, **matcher_kwargs)
    m.fit(Gc, Hc)
    _, score = m.predict()
    return score

def annotate_depth_as_feature(G: ig.Graph, *, root: int = 0, attr: str = "dp_features") -> None:
    """
    Set a numeric attribute for coloring. Depth is a good default.
    Assumes the tree is connected and rooted at `root`.
    """
    # distances() returns a 2D list: [ [dist(root->v) for v in V] ]
    d = G.distances(source=root)[0]
    # convert None (unreachable) to nan
    vals = [float(x) if x is not None else float("nan") for x in d]
    G.vs[attr] = vals


def extract_planted_path_ordered(
    G: ig.Graph,
    *,
    planted_attr: str = "is_planted",
) -> list[int]:
    """
    Recover an ordered path from the 0/1 planted markers.
    For the planted-path sampler, the planted vertices form a simple root-to-leaf path.
    """
    if planted_attr not in G.vs.attributes():
        raise KeyError(f"Missing planted attribute '{planted_attr}'")

    planted = [i for i, x in enumerate(G.vs[planted_attr]) if int(x) == 1]
    if not planted:
        return []

    S = set(planted)

    # Find the start = planted vertex with no *planted* parent.
    # For directed parent->child trees, parent is in-neighbor.
    start = None
    for v in planted:
        in_planted = [u for u in G.neighbors(v, mode="IN") if u in S]
        if len(in_planted) == 0:
            start = v
            break
    if start is None:
        # Fall back: pick the minimum-depth planted vertex if the above fails.
        d = G.distances(source=0)[0]
        start = min(planted, key=lambda v: d[v] if d[v] is not None else 10**9)

    # Follow planted children until leaf (unique in a path).
    path = [start]
    cur = start
    while True:
        out_planted = [u for u in G.neighbors(cur, mode="OUT") if u in S]
        if len(out_planted) == 0:
            break
        if len(out_planted) > 1:
            raise ValueError("Planted vertices do not form a simple path (branching detected).")
        cur = out_planted[0]
        path.append(cur)

    return path


def pairs_to_vertex_paths(pairs) -> tuple[list[int], list[int]]:
    """
    Convert a predicted match path from the matcher into vertex-index paths for each tree.

    Accepts pairs like:
      [(u0,v0), (u1,v1), ...]  (typical)
    and is robust to gap markers:
      u or v may be None or -1  (these are dropped from that side).
    """
    path_u: list[int] = []
    path_v: list[int] = []

    def push_unique(lst: list[int], x: int):
        if not lst or lst[-1] != x:
            lst.append(x)

    for u, v in pairs:
        if u is not None and int(u) >= 0:
            push_unique(path_u, int(u))
        if v is not None and int(v) >= 0:
            push_unique(path_v, int(v))

    return path_u, path_v

def prep_match_plot_inputs(
    G,
    H,
    pairs,
    *,
    truth_G=None,
    truth_H=None,
    label_attr="label",
    colormap="viridis",
    background_color="black",
):
    """Prepare (match vertices, per-vertex colors, per-vertex numbering labels) for side-by-side plots."""
    matchG = [int(a) for (a, _b) in pairs]
    matchH = [int(b) for (_a, b) in pairs]

    truth_G = [] if truth_G is None else [int(v) for v in truth_G]
    truth_H = [] if truth_H is None else [int(v) for v in truth_H]

    # Highlight = matched OR truth
    hiG = sorted(set(matchG).union(truth_G))
    hiH = sorted(set(matchH).union(truth_H))

    # Build a *shared* label -> color mapping across both trees
    if label_attr not in G.vs.attributes():
        raise KeyError(f"Expected vertex attribute '{label_attr}' in G, but it's missing.")
    if label_attr not in H.vs.attributes():
        raise KeyError(f"Expected vertex attribute '{label_attr}' in H, but it's missing.")

    labels = [G.vs[label_attr][v] for v in hiG] + [H.vs[label_attr][v] for v in hiH]
    label_to_color = build_discrete_label_colormap(labels, colormap=colormap)

    colorsG, _ = vertex_colors_by_label(
        G,
        label_attr,
        highlight_vertices=hiG,
        label_to_color=label_to_color,
        colormap=colormap,
        background_color=background_color,
    )
    colorsH, _ = vertex_colors_by_label(
        H,
        label_attr,
        highlight_vertices=hiH,
        label_to_color=label_to_color,
        colormap=colormap,
        background_color=background_color,
    )

    # Number matched vertices by match order (kth pair -> label "k")
    labelsG = vertex_labels_from_ordered_vertices(G.vcount(), matchG, start=1, default="")
    labelsH = vertex_labels_from_ordered_vertices(H.vcount(), matchH, start=1, default="")

    return matchG, matchH, colorsG, colorsH, labelsG, labelsH