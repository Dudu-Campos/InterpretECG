"""Transition graph (quantile partition).

Discretize the series into Q bins by quantiles, then build a directed weighted
graph where nodes are bins (states) and edge (a→b) weight = count of
transitions s_t=a → s_{t+1}=b. Self-loops kept (they carry persistence info).

References
----------
Campanharo et al. 2011 (transition networks); Silva et al. 2021 review
(arXiv 2110.09887, "transition" column of Table 1).

Contract
--------
- Input: 1D np.ndarray of length L.
- Output: nx.DiGraph with up to Q nodes (Q=10 default), node attribute
  `quantile_center` = midpoint of that bin, edge attribute `weight` = count.
- No hyperparameter besides Q; Q fixed at declaration time to keep it
  training-free and comparable across datasets (confound-controllable).
"""
from __future__ import annotations
import numpy as np
import networkx as nx


def transition_graph(
    x: np.ndarray,
    n_bins: int = 10,
    drop_empty: bool = False,
    bin_edges: np.ndarray | None = None,
) -> nx.DiGraph:
    """Quantile-partition transition graph.

    Parameters
    ----------
    x : np.ndarray shape (L,)
    n_bins : Q, number of quantile bins. Default 10. Fixed across the study
        for confound control (does NOT scale with L to avoid conflating
        graph size with series length).
    drop_empty : if True, drop bins with no observations. Default False so
        the node set is identical across series (aligns feature spaces).
    bin_edges : optional n_bins - 1 interior edges fixed from training data.
        Per-series quantiles make the graph invariant to amplitude; fixed
        edges keep it.  The outer edges become the series' min and max.

    Returns
    -------
    nx.DiGraph with n_bins nodes (or fewer if drop_empty=True), edges weighted
    by transition counts. Self-loops preserved.
    """
    x = np.asarray(x, dtype=float).ravel()
    L = len(x)
    if L < 2:
        raise ValueError(f"series too short: L={L}")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >=2, got {n_bins}")

    # Quantile edges: q_0=min, q_Q=max, q_i for i=1..Q-1 at i/Q quantile
    if bin_edges is None:
        q = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    else:
        inner = np.asarray(bin_edges, dtype=float).ravel()
        if inner.shape != (n_bins - 1,):
            raise ValueError(f"bin_edges must have {n_bins - 1} values")
        q = np.concatenate(([min(x.min(), inner[0])], inner, [max(x.max(), inner[-1])]))
    # Handle ties by making edges strictly increasing (fallback for constant series)
    for i in range(1, len(q)):
        if q[i] <= q[i - 1]:
            q[i] = np.nextafter(q[i - 1], np.inf)
    # np.digitize is right-open by default; use side='right' to include max
    states = np.clip(np.digitize(x, q[1:-1]), 0, n_bins - 1)

    G = nx.DiGraph()
    centers = 0.5 * (q[:-1] + q[1:])
    for s in range(n_bins):
        G.add_node(s, quantile_center=float(centers[s]))

    # Count transitions
    trans = {}
    for a, b in zip(states[:-1], states[1:]):
        key = (int(a), int(b))
        trans[key] = trans.get(key, 0) + 1
    for (a, b), w in trans.items():
        G.add_edge(a, b, weight=w)

    if drop_empty:
        observed = set(states.tolist())
        to_drop = [s for s in range(n_bins) if s not in observed]
        G.remove_nodes_from(to_drop)

    return G
