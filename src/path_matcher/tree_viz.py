"""
tree_viz.py

Lightweight plotting helpers for igraph trees + highlighting planted/extracted paths.

Designed to be dropped directly into your matcher repo (single file, no packaging).

Typical usage
-------------
    import matplotlib.pyplot as plt
    from tree_viz import plot_tree_with_path

    fig, ax = plot_tree_with_path(
        G,
        path_vertices=extracted_path,     # list[int] vertex indices
        color_by_attr="dp_features",      # numeric vertex attr -> colormap
        colormap="viridis",
    )
    plt.show()

If you don't pass `path_vertices`, the plotter will try to infer a path/marked set
from `infer_path_from_attr="is_planted"` (default) by highlighting those vertices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------
# Color helpers
# ---------------------------

def vertex_colors_from_numeric_attr(
    G: "igraph.Graph",
    attr: str,
    *,
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    nan_color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
) -> Tuple[List[Tuple[float, float, float, float]], Tuple[float, float]]:
    """
    Map a numeric vertex attribute to RGBA colors using a matplotlib colormap.

    Returns
    -------
    colors : list[RGBA]
        One RGBA tuple per vertex.
    (vmin, vmax) : tuple
        Normalization range actually used (handy if you want consistent scaling across plots).
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    if attr not in G.vs.attributes():
        raise KeyError(f"Vertex attribute '{attr}' not found.")

    values = np.asarray(G.vs[attr], dtype=float)
    n = G.vcount()
    if values.shape != (n,):
        raise ValueError(f"Attribute '{attr}' must have length {n}, got {values.shape}.")

    finite = np.isfinite(values)
    if not finite.any():
        return [nan_color] * n, (0.0, 1.0)

    if vmin is None:
        vmin = float(values[finite].min())
    if vmax is None:
        vmax = float(values[finite].max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmin == vmax:
        # avoid division by zero; still yield a well-defined color map
        vmin -= 1.0
        vmax += 1.0

    cmap = plt.cm.get_cmap(colormap)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    colors: List[Tuple[float, float, float, float]] = []
    for x in values:
        if np.isfinite(x):
            colors.append(tuple(float(c) for c in cmap(norm(float(x)))))
        else:
            colors.append(nan_color)
    return colors, (float(vmin), float(vmax))



# ---------------------------
# Discrete label -> color helpers
# ---------------------------

def build_discrete_label_colormap(
    labels: Sequence[Any],
    *,
    colormap: str = "viridis",
) -> Dict[Any, Tuple[float, float, float, float]]:
    """
    Assign a **discrete** RGBA color to each unique label using a matplotlib colormap.

    This is useful when vertex labels are categorical (strings / small dictionaries),
    and you want the *same* label to have the *same* color across multiple plots.

    Parameters
    ----------
    labels:
        Any hashable values (strings, ints, etc). Order matters: the first time a label
        appears determines its position in the colormap.
    colormap:
        A matplotlib colormap name (e.g., "viridis").

    Returns
    -------
    label_to_color:
        dict mapping label -> RGBA tuple.
    """
    import matplotlib.pyplot as plt

    # Preserve first-seen order
    uniq: List[Any] = list(dict.fromkeys(labels))
    m = len(uniq)
    if m == 0:
        return {}

    cmap = plt.cm.get_cmap(colormap)

    if m == 1:
        xs = [0.5]
    else:
        xs = [i / (m - 1) for i in range(m)]

    return {lab: tuple(float(c) for c in cmap(x)) for lab, x in zip(uniq, xs)}


def vertex_colors_by_label(
    G: "igraph.Graph",
    label_attr: str,
    *,
    highlight_vertices: Optional[Sequence[int]] = None,
    label_to_color: Optional[Dict[Any, Tuple[float, float, float, float]]] = None,
    colormap: str = "viridis",
    background_color: Union[str, Tuple[float, float, float, float]] = "black",
    missing_label_color: Tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0),
) -> Tuple[List[Union[str, Tuple[float, float, float, float]]], Dict[Any, Tuple[float, float, float, float]]]:
    """
    Color a tree by a *categorical* vertex attribute, with an optional highlighted subset.

    - Vertices not in `highlight_vertices` (if provided) get `background_color`.
    - Vertices in `highlight_vertices` are colored by their `label_attr` value using `label_to_color`.
      If `label_to_color` is not provided, it is built from the highlighted labels.

    Returns (colors, label_to_color_used).
    """
    if label_attr not in G.vs.attributes():
        raise KeyError(f"Vertex attribute '{label_attr}' not found.")

    n = G.vcount()

    if highlight_vertices is None:
        highlight_set = set(range(n))
    else:
        highlight_set = {int(v) for v in highlight_vertices}

    # Build (or reuse) label -> color mapping
    if label_to_color is None:
        labs: List[Any] = []
        for v in sorted(highlight_set):
            try:
                labs.append(G.vs[label_attr][v])
            except Exception:
                labs.append(None)
        label_to_color = build_discrete_label_colormap(labs, colormap=colormap)

    colors: List[Union[str, Tuple[float, float, float, float]]] = [background_color] * n
    for v in highlight_set:
        try:
            lab = G.vs[label_attr][v]
        except Exception:
            lab = None
        colors[v] = label_to_color.get(lab, missing_label_color)

    return colors, label_to_color


def vertex_labels_from_ordered_vertices(
    n_vertices: int,
    ordered_vertices: Sequence[int],
    *,
    start: int = 1,
    default: str = "",
) -> List[str]:
    """
    Create a per-vertex label list where `ordered_vertices[k]` gets label `start + k`.

    Useful for numbering matched vertices in the order they appear in a match list.
    """
    out = [default] * int(n_vertices)
    for k, v in enumerate(ordered_vertices, start=start):
        v = int(v)
        if 0 <= v < n_vertices:
            out[v] = str(k)
    return out
# ---------------------------
# Layout + path helpers
# ---------------------------

def get_rooted_tree_layout(
    G: "igraph.Graph",
    *,
    root: int = 0,
    layout: str = "rt",
) -> "igraph.Layout":
    """
    Get a rooted-tree layout.

    Default is Reingold–Tilford ("rt"). If `layout_reingold_tilford` is not available
    (depends on igraph version), falls back to `G.layout("rt")`.
    """
    import igraph as ig  # noqa: F401

    if layout in {"rt", "reingold_tilford"}:
        try:
            return G.layout_reingold_tilford(root=[root])
        except Exception:
            return G.layout("rt")
    return G.layout(layout)


def path_edge_ids(
    G: "igraph.Graph",
    path_vertices: Sequence[int],
    *,
    directed: Optional[bool] = None,
    on_missing: str = "skip",
) -> List[int]:
    """
    Convert an ordered vertex sequence [v0, v1, ..., vk] into edge IDs.

    Important note
    --------------
    In this project, `path_vertices` is often **not** a connected path: it may be a
    list of matched vertices (possibly disconnected). In that case we intentionally
    avoid "completing" the sequence by inserting intermediate vertices.

    Behavior
    --------
    - If an edge exists between consecutive vertices, we include it.
    - If no edge exists:
        * on_missing="skip" (default): skip that segment (still allows vertex highlighting).
        * on_missing="error": raise a ValueError with the offending segment.

    Notes
    -----
    - For directed graphs, we try the directed edge first, then the reverse direction.
    """
    if directed is None:
        directed = bool(G.is_directed())

    if on_missing not in {"skip", "error"}:
        raise ValueError("on_missing must be one of {'skip','error'}")

    eids: List[int] = []

    for a, b in zip(path_vertices, path_vertices[1:]):
        a = int(a)
        b = int(b)

        eid = G.get_eid(a, b, directed=directed, error=False)
        if eid != -1:
            eids.append(int(eid))
            continue

        eid = G.get_eid(b, a, directed=directed, error=False)
        if eid != -1:
            eids.append(int(eid))
            continue

        if on_missing == "error":
            raise ValueError(f"Consecutive vertices are not adjacent by an edge: ({a}, {b}).")

        # on_missing == "skip"
        continue

    return eids


# ---------------------------
# Plotting
# ---------------------------

@dataclass
class PlotStyle:
    # Base tree styling (close to your snippet)
    vertex_size: int = 25
    edge_width: float = 1.0
    edge_color: str = "gray"
    margin: int = 20

    # General vertex frames
    vertex_frame_color: str = "black"
    vertex_frame_width: float = 0.8

    # Highlight path (extracted / planted)
    path_edge_color: str = "crimson"
    path_edge_width: float = 3.0
    path_vertex_frame_color: str = "crimson"
    path_vertex_frame_width: float = 2.5

    # Optional second highlight (e.g., compare planted vs extracted)
    path2_edge_color: str = "dodgerblue"
    path2_edge_width: float = 3.0
    path2_vertex_frame_color: str = "dodgerblue"
    path2_vertex_frame_width: float = 2.5


def plot_tree(
    G: "igraph.Graph",
    *,
    ax: Optional["matplotlib.axes.Axes"] = None,
    layout: Union[str, "igraph.Layout"] = "rt",
    root: int = 0,
    vertex_color: Optional[Union[str, Sequence[Any]]] = None,
    vertex_label: Optional[Union[str, Sequence[Any]]] = None,
    style: Optional[PlotStyle] = None,
    invert_yaxis: bool = True,
    figsize: Tuple[float, float] = (7.0, 7.0),
    **igraph_plot_kwargs: Any,
):
    """
    Plot a tree with igraph + matplotlib, returning (fig, ax) so you can customize/save.

    Parameters
    ----------
    vertex_color:
        Either a single color or a list of per-vertex colors (length vcount()).
    vertex_label:
        Either None, a list (length vcount()), or a string attribute name.
    """
    import matplotlib.pyplot as plt
    import igraph as ig  # noqa: F401

    if style is None:
        style = PlotStyle()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    lay = get_rooted_tree_layout(G, root=root, layout=layout) if isinstance(layout, str) else layout

    if isinstance(vertex_label, str):
        if vertex_label in G.vs.attributes():
            vertex_label_val = G.vs[vertex_label]
        else:
            vertex_label_val = None
    else:
        vertex_label_val = vertex_label


    # igraph may default to showing the vertex attribute named "label" (or IDs)
    # when vertex_label is omitted/None. Force "no labels" unless explicitly provided.
    if vertex_label_val is None:
        vertex_label_val = [""] * G.vcount()

    ig.plot(
        G,
        target=ax,
        layout=lay,
        vertex_size=style.vertex_size,
        vertex_color=vertex_color,
        vertex_label=vertex_label_val,
        vertex_frame_color=style.vertex_frame_color,
        vertex_frame_width=style.vertex_frame_width,
        edge_width=style.edge_width,
        edge_color=style.edge_color,
        margin=style.margin,
        **igraph_plot_kwargs,
    )

    if invert_yaxis:
        ax.invert_yaxis()
    return fig, ax


def plot_tree_with_path(
    G: "igraph.Graph",
    path_vertices: Optional[Sequence[int]] = None,
    *,
    # A second set/sequence is handy for "truth vs estimate":
    path2_vertices: Optional[Sequence[int]] = None,
    # If consecutive vertices are not adjacent, how to handle edge-highlighting:
    # (We *do not* auto-complete disconnected sequences.)
    path_edge_on_missing: str = "skip",

    ax: Optional["matplotlib.axes.Axes"] = None,
    layout: Union[str, "igraph.Layout"] = "rt",
    root: int = 0,

    # Base vertex colors:
    # - If provided, this overrides `color_by_attr`.
    # - Can be a single color or a per-vertex list (length vcount()).
    vertex_color: Optional[Union[str, Sequence[Any]]] = None,

    # Legacy vertex coloring by numeric attribute:
    color_by_attr: Optional[str] = "dp_features",
    colormap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,

    # If no explicit path provided, infer highlighted vertices from a 0/1 attribute:
    infer_path_from_attr: Optional[str] = "is_planted",

    # Vertex labels:
    # - If provided, this overrides `show_vertex_labels` / `vertex_label_attr`.
    # - Can be a string attribute name or a per-vertex list (length vcount()).
    vertex_label: Optional[Union[str, Sequence[Any]]] = None,
    show_vertex_labels: bool = False,
    vertex_label_attr: str = "label",

    # Style:
    style: Optional[PlotStyle] = None,
    invert_yaxis: bool = True,
    figsize: Tuple[float, float] = (7.0, 7.0),
    **igraph_plot_kwargs: Any,
):
    """
    Plot a tree and highlight one (and optionally two) vertex sequences.

    Notes
    -----
    - `path_vertices` and `path2_vertices` may be true paths *or* just an ordered list
      of vertices (e.g., the matched vertices returned by the matcher).
    - Edge highlighting is **adjacency-only**: if consecutive vertices are not adjacent,
      we either skip that segment (default) or raise (path_edge_on_missing="error").
    """
    import matplotlib.pyplot as plt
    import igraph as ig  # noqa: F401

    if style is None:
        style = PlotStyle()

    lay = get_rooted_tree_layout(G, root=root, layout=layout) if isinstance(layout, str) else layout
    n = G.vcount()

    # -----------------------
    # Base vertex colors
    # -----------------------
    if vertex_color is not None:
        if isinstance(vertex_color, str):
            vertex_colors = vertex_color
        else:
            if len(vertex_color) != n:
                raise ValueError(f"vertex_color must have length {n}, got {len(vertex_color)}.")
            vertex_colors = list(vertex_color)
    elif color_by_attr is not None and color_by_attr in G.vs.attributes():
        vertex_colors, _used = vertex_colors_from_numeric_attr(
            G, color_by_attr, colormap=colormap, vmin=vmin, vmax=vmax
        )
    else:
        vertex_colors = "lightgray"

    # -----------------------
    # Determine highlighted vertices
    # -----------------------
    path_v: List[int] = []
    if path_vertices is not None:
        path_v = [int(v) for v in path_vertices]
    elif infer_path_from_attr is not None and infer_path_from_attr in G.vs.attributes():
        mask = [int(x) for x in G.vs[infer_path_from_attr]]
        path_v = [i for i, m in enumerate(mask) if m == 1]
    path_v_set = set(path_v)

    path2_v: List[int] = []
    if path2_vertices is not None:
        path2_v = [int(v) for v in path2_vertices]
    path2_v_set = set(path2_v)

    # -----------------------
    # Edge styles
    # -----------------------
    edge_colors = [style.edge_color] * G.ecount()
    edge_widths = [style.edge_width] * G.ecount()

    if path_vertices is not None and len(path_v) >= 2:
        for eid in path_edge_ids(G, path_v, on_missing=path_edge_on_missing):
            edge_colors[eid] = style.path_edge_color
            edge_widths[eid] = style.path_edge_width

    if path2_vertices is not None and len(path2_v) >= 2:
        for eid in path_edge_ids(G, path2_v, on_missing=path_edge_on_missing):
            edge_colors[eid] = style.path2_edge_color
            edge_widths[eid] = style.path2_edge_width

    # -----------------------
    # Vertex frame styles
    # -----------------------
    frame_colors = [style.vertex_frame_color] * n
    frame_widths = [style.vertex_frame_width] * n

    # Circle matched vertices (path_vertices) in red, and truth-only vertices (path2_vertices \ matched) in blue.
    for v in (path2_v_set - path_v_set):
        frame_colors[int(v)] = style.path2_vertex_frame_color
        frame_widths[int(v)] = style.path2_vertex_frame_width
    for v in path_v_set:
        frame_colors[int(v)] = style.path_vertex_frame_color
        frame_widths[int(v)] = style.path_vertex_frame_width

    # -----------------------
    # Vertex labels
    # -----------------------
    vertex_label_val = None
    if vertex_label is not None:
        if isinstance(vertex_label, str):
            if vertex_label not in G.vs.attributes():
                raise KeyError(f"Vertex attribute '{vertex_label}' not found.")
            vertex_label_val = G.vs[vertex_label]
        else:
            if len(vertex_label) != n:
                raise ValueError(f"vertex_label must have length {n}, got {len(vertex_label)}.")
            vertex_label_val = list(vertex_label)
    else:
        if show_vertex_labels and vertex_label_attr in G.vs.attributes():
            vertex_label_val = G.vs[vertex_label_attr]


    # igraph may default to showing the vertex attribute named "label" (or IDs)
    # when vertex_label is omitted/None. Force "no labels" unless explicitly provided.
    if vertex_label_val is None:
        vertex_label_val = [""] * n

    # -----------------------
    # Axes + plot
    # -----------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ig.plot(
        G,
        target=ax,
        layout=lay,
        vertex_size=style.vertex_size,
        vertex_color=vertex_colors,
        vertex_label=vertex_label_val,
        vertex_frame_color=frame_colors,
        vertex_frame_width=frame_widths,
        edge_color=edge_colors,
        edge_width=edge_widths,
        margin=style.margin,
        **igraph_plot_kwargs,
    )

    if invert_yaxis:
        ax.invert_yaxis()
    return fig, ax


def plot_trees_with_paths(
    graphs: Sequence["igraph.Graph"],
    paths: Optional[Sequence[Optional[Sequence[int]]]] = None,
    *,
    ncols: int = 3,
    figsize_per_plot: Tuple[float, float] = (4.0, 4.0),
    **kwargs: Any,
):
    """
    Grid plot helper for many trees.

    `paths` can be:
      - None (infer via infer_path_from_attr inside plot_tree_with_path), or
      - a list aligned with `graphs`, each entry either a vertex-index path or None.
    """
    import math
    import matplotlib.pyplot as plt

    m = len(graphs)
    if paths is None:
        paths = [None] * m
    if len(paths) != m:
        raise ValueError("paths must be None or the same length as graphs")

    ncols = max(1, int(ncols))
    nrows = int(math.ceil(m / ncols))
    fig_w = figsize_per_plot[0] * ncols
    fig_h = figsize_per_plot[1] * nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_w, fig_h))

    axes_list = np.array(axes).reshape(-1).tolist()
    for i, (G, p) in enumerate(zip(graphs, paths)):
        ax = axes_list[i]
        plot_tree_with_path(G, p, ax=ax, **kwargs)
        ax.set_axis_off()

    for j in range(m, len(axes_list)):
        axes_list[j].set_axis_off()

    fig.tight_layout()
    return fig, axes
