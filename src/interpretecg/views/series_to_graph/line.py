"""Line graph L(G) of a visibility graph, read in closed form.

Ported from the PRISM sibling project (`Paintings/src/extractors/
extract_line_graph_features.py`), where L(G) of the region adjacency graph
added over the graph itself.  A node of L(G) is an edge of G; two are adjacent
when the edges share a node, i.e. form a *wedge* (a, s, b).  Nothing is built
explicitly: the degree of edge (u, v) in L(G) is k_u + k_v - 2, the number of
L(G) edges is sum C(k, 2), and the wedges are enumerated per shared node.

Why it can add over G: the degree distribution of L(G) is a function of the
JOINT degree distribution of G's edges, so degree-degree correlations that
are second-order in G become first-order in L(G).  It is the same object made
more accessible -- the paper's thesis in miniature.

For a visibility graph an edge (i, j) is a pair of samples that see each
other.  Its attributes split by invariance, so the two halves are returned as
separate families:

- ordinal / time (invariant to any increasing transform, like the HVG): L(G)
  topology, the span j - i, whether the wedge's shared sample dominates both
  outer ones, whether the outer samples lie on opposite sides;
- millivolt (NOT invariant): the visibility level min(x_i, x_j), the step
  |x_i - x_j|, and their continuity across wedges.
"""

from __future__ import annotations

import numpy as np


def _gini(values: np.ndarray) -> float:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size == 0 or values.sum() <= 0:
        return 0.0
    cumulative = np.cumsum(values)
    return float((values.size + 1 - 2 * cumulative.sum() / cumulative[-1]) / values.size)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _stats(values: np.ndarray, prefix: str, quantiles=(10, 50, 90)) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{prefix}_mean": 0.0, f"{prefix}_std": 0.0,
                **{f"{prefix}_q{q}": 0.0 for q in quantiles}}
    return {f"{prefix}_mean": float(values.mean()), f"{prefix}_std": float(values.std()),
            **{f"{prefix}_q{q}": float(v) for q, v in zip(quantiles, np.percentile(values, quantiles))}}


def line_graph_features(x: np.ndarray, edges: np.ndarray) -> tuple[dict, dict]:
    """Return (ordinal, millivolt) feature dicts for L(G), G given as an (E, 2) edge array."""
    x = np.asarray(x, dtype=float)
    source, target = np.sort(np.asarray(edges, dtype=np.int64), axis=1).T
    n_nodes, n_edges = len(x), len(source)
    degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)
    line_degree = degree[source] + degree[target] - 2
    wedge_load = degree * np.maximum(degree - 1, 0) / 2
    n_line_edges = int(wedge_load.sum())

    # Wedges (edge_a, edge_b) sharing a node, enumerated per shared node.
    incidente = np.concatenate([source, target])
    aresta = np.concatenate([np.arange(n_edges), np.arange(n_edges)])
    ordem = np.argsort(incidente, kind="stable")
    incidente, aresta = incidente[ordem], aresta[ordem]
    inicios = np.searchsorted(incidente, np.arange(n_nodes))
    partes_a, partes_b, partes_s = [], [], []
    for s in np.flatnonzero(degree >= 2):
        ids = aresta[inicios[s]:inicios[s] + degree[s]]
        a, b = np.triu_indices(len(ids), 1)
        partes_a.append(ids[a]); partes_b.append(ids[b]); partes_s.append(np.full(len(a), s))
    edge_a, edge_b, shared = (np.concatenate(p) if p else np.zeros(0, int)
                              for p in (partes_a, partes_b, partes_s))
    if len(edge_a) != n_line_edges:
        raise RuntimeError(f"wedge count {len(edge_a)} != {n_line_edges}")
    outer_a = np.where(source[edge_a] == shared, target[edge_a], source[edge_a])
    outer_b = np.where(source[edge_b] == shared, target[edge_b], source[edge_b])

    span = (target - source).astype(float)
    possible = n_edges * (n_edges - 1) / 2
    ordinal = {
        "lg_n_edges": float(n_line_edges),
        "lg_density": n_line_edges / possible if possible else 0.0,
        **_stats(line_degree, "lg_degree"),
        "lg_degree_gini": _gini(line_degree),
        "lg_wedge_load_gini": _gini(wedge_load),
        "lg_wedge_load_max_frac": float(wedge_load.max() / n_line_edges) if n_line_edges else 0.0,
        "lg_degree_assortativity": _corr(line_degree[edge_a], line_degree[edge_b]),
        **_stats(span, "span"),
        "span_assortativity": _corr(span[edge_a], span[edge_b]),
        **_stats(np.abs(np.log(span[edge_a] / span[edge_b])), "wedge_span_log_ratio", (50, 90)),
        "wedge_opposite_sides_frac": float(np.mean((outer_a - shared) * (outer_b - shared) < 0))
        if len(shared) else 0.0,
        "wedge_shared_dominant_frac": float(np.mean((x[shared] > x[outer_a]) & (x[shared] > x[outer_b])))
        if len(shared) else 0.0,
        "wedge_shared_lowest_frac": float(np.mean((x[shared] < x[outer_a]) & (x[shared] < x[outer_b])))
        if len(shared) else 0.0,
        **_stats(np.abs(outer_a - outer_b).astype(float), "wedge_outer_time_distance", (50, 90)),
    }
    base = np.minimum(x[source], x[target])
    step = np.abs(x[source] - x[target])
    millivolt = {
        **_stats(base, "level"),
        **_stats(step, "step"),
        "level_assortativity": _corr(base[edge_a], base[edge_b]),
        "step_assortativity": _corr(step[edge_a], step[edge_b]),
        **_stats(np.abs(base[edge_a] - base[edge_b]), "wedge_level_delta", (50, 90)),
        **_stats(np.abs(x[outer_a] - x[outer_b]), "wedge_outer_amplitude_distance", (50, 90)),
        "span_step_corr": _corr(span, step),
    }
    return ordinal, millivolt
