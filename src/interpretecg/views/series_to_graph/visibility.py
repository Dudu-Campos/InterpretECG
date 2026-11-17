"""Visibility graph (HVG default; NVG optional).

References
----------
Lacasa et al. 2008, PNAS — natural visibility graph.
Luque et al. 2009 — horizontal visibility graph.
Zhou et al. 2018, Scientific Reports 8:5493 — limited penetrable HVG;
    exact results for i.i.d. series, mean degree 4*(rho+1).

Contract
--------
- Input: 1D np.ndarray of length L.
- Output: nx.Graph (undirected, unweighted, L nodes indexed 0..L-1).
- Each node carries attribute `value` = x[i]; each node also carries `t` = i.
- Parameters: `kind` (HVG default) and `rho`, the penetrable
  distance for the limited penetrable HVG (`rho=0` is the plain HVG).
"""
from __future__ import annotations
import heapq

import numpy as np
import networkx as nx


def _hvg_edges(x: np.ndarray):
    """O(L^2) worst case, O(L log L) average on i.i.d. inputs.

    HVG rule (Luque 2009): edge (i,j) iff x[k] < min(x[i], x[j]) for all i<k<j.
    Equivalently: min(x[i], x[j]) > max(x[i+1], ..., x[j-1]).
    """
    L = len(x)
    for i in range(L - 1):
        yield (i, i + 1)  # immediate neighbor always visible
        running_max = x[i + 1]
        for j in range(i + 2, L):
            if min(x[i], x[j]) > running_max:
                yield (i, j)
            if x[j] >= x[i]:
                # x[i] can no longer see any j'>j: running_max would contain
                # x[j] >= x[i], so min(x[i], x[j']) <= x[i] <= running_max.
                break
            if x[j] > running_max:
                running_max = x[j]


def _lphvg_edges(x: np.ndarray, rho: int):
    """Limited penetrable HVG: edge (i,j) iff at most `rho` intermediate
    points obstruct horizontal visibility.

    LPHVG rule (Zhou et al. 2018): edge (i,j) iff
    #{k : i<k<j, x[k] >= min(x[i], x[j])} <= rho.

    Equivalent criterion used here: let M be the (rho+1)-th largest value
    among x[i+1..j-1] (-inf when there are fewer than rho+1 of them).  The
    edge exists iff min(x[i], x[j]) > M.  `rho=0` reduces to `_hvg_edges`.

    M is non-decreasing in j, which gives two valid early breaks for a fixed
    i: stop once M >= x[i] (no larger j can qualify, since the threshold
    never exceeds x[i]), and stop once M >= max(x[j+1..]) (no later value is
    high enough to clear M).  The suffix maxima are precomputed.
    """
    L = len(x)
    # suffix_max[t] = max(x[t:]); suffix_max[L] = -inf
    suffix_max = np.empty(L + 1)
    suffix_max[L] = -np.inf
    np.maximum.accumulate(x[::-1], out=suffix_max[:L][::-1])

    for i in range(L - 1):
        yield (i, i + 1)  # no intermediate points: always visible
        top = []  # min-heap with the largest rho+1 intermediates seen so far
        for j in range(i + 2, L):
            value = x[j - 1]  # x[j-1] becomes an intermediate point for this j
            if len(top) < rho + 1:
                heapq.heappush(top, value)
            elif value > top[0]:
                heapq.heapreplace(top, value)
            threshold = top[0] if len(top) == rho + 1 else -np.inf

            if min(x[i], x[j]) > threshold:
                yield (i, j)
            if threshold >= x[i] or threshold >= suffix_max[j + 1]:
                break


def _nvg_edges(x: np.ndarray):
    """Natural visibility graph — linear roof test.

    Edge (i,j) iff for all i<k<j: x[k] < x[j] + (x[i]-x[j]) * (j-k)/(j-i).
    """
    L = len(x)
    for i in range(L - 1):
        yield (i, i + 1)
        for j in range(i + 2, L):
            ok = True
            slope = (x[j] - x[i]) / (j - i)
            for k in range(i + 1, j):
                roof = x[i] + slope * (k - i)
                if x[k] >= roof:
                    ok = False
                    break
            if ok:
                yield (i, j)


def visibility_graph(x: np.ndarray, kind: str = "hvg", rho: int = 0) -> nx.Graph:
    """Build a visibility graph from a 1D series.

    Parameters
    ----------
    x : np.ndarray shape (L,)
    kind : {"hvg", "nvg"}. HVG is default: cheaper, canonical for
        degree-distribution-based structural signatures (Luque et al. 2009).
    rho : penetrable distance of the limited penetrable HVG (Zhou et al.
        2018): the number of intermediate points allowed to obstruct
        visibility.  `rho=0` is the plain HVG.  Only valid with
        ``kind="hvg"``; for i.i.d. series the mean degree is 4*(rho+1).

    Returns
    -------
    nx.Graph with L nodes, each with attributes `value` and `t`.
    """
    x = np.asarray(x, dtype=float).ravel()
    L = len(x)
    if L < 2:
        raise ValueError(f"series too short: L={L}")
    if not isinstance(rho, (int, np.integer)) or rho < 0:
        raise ValueError(f"rho must be a non-negative integer, got {rho}")
    if rho and kind != "hvg":
        raise ValueError("rho is only defined for kind='hvg'")

    G = nx.Graph()
    for i in range(L):
        G.add_node(i, value=float(x[i]), t=i)

    if kind == "hvg":
        edges = _hvg_edges(x) if rho == 0 else _lphvg_edges(x, rho)
    elif kind == "nvg":
        edges = _nvg_edges(x)
    else:
        raise ValueError(f"kind must be 'hvg' or 'nvg', got {kind!r}")
    G.add_edges_from(edges)
    return G
