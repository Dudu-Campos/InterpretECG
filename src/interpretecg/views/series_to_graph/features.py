"""phi_free feature extractor for series-to-graph conversions.

Contract:
    extract_features(G, kind: str) -> dict[str, float]

`kind` in {"hvg", "transition", "takens"} controls WHICH diagnostic C
gets appended. The common block is identical across conversions so the
downstream readout sees the same dimensionality and feature family — the
only free variable is the conversion itself. This preserves the §8
discipline of "one treatment degree of freedom".

Feature families (26 common + 1 C = 27 total):
- degree statistics: mean, std, max, skew of log(1+deg), q25/q50/q75
- spectral: top-10 eigenvalues of the normalized Laplacian (padded/truncated)
- clustering: transitivity, avg local clustering, assortativity
- connectivity: n_components, largest_cc_frac
- path: diameter of largest CC, avg shortest path (subsampled if N>500)
- weight (transition only, else NaN): mean and std of edge weights

Diagnostic scalars C (interpretive anchor, not feature-vector novelty):
- hvg:        C = degree exponent (fit ln P(k) ~ -lambda*k on tail; lambda is C)
              iid theory: lambda = ln(3/2) ≈ 0.405 (Luque 2010)
- transition: C = entropy rate = -sum_i pi_i sum_j P_ij log2 P_ij
              periodic ≈ 0 bits; uniform iid on Q bins → log2(Q) bits
- takens:     C = correlation dimension D2 (Grassberger-Procaccia local slope
              on log C(r) vs log r), estimated from the recurrence structure.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import networkx as nx
from scipy.stats import skew
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix
import warnings


# --------- helpers ---------

def _to_undirected(G: nx.Graph) -> nx.Graph:
    """Return an undirected view without loss of node/edge attributes."""
    if G.is_directed():
        H = nx.Graph()
        H.add_nodes_from(G.nodes(data=True))
        for u, v, d in G.edges(data=True):
            if H.has_edge(u, v):
                # Merge weights (sum) to keep directed transition info as weight mass
                H[u][v]["weight"] = H[u][v].get("weight", 1.0) + d.get("weight", 1.0)
            else:
                H.add_edge(u, v, **d)
        return H
    return G


def _degree_stats(G: nx.Graph) -> dict:
    if G.number_of_nodes() == 0:
        return {f"deg_{k}": 0.0 for k in ["mean", "std", "max", "logskew",
                                          "q25", "q50", "q75"]}
    degs = np.array([d for _, d in G.degree()], dtype=float)
    log_deg = np.log1p(degs)
    return {
        "deg_mean": float(degs.mean()),
        "deg_std": float(degs.std()),
        "deg_max": float(degs.max()),
        "deg_logskew": float(skew(log_deg)) if len(log_deg) > 2 else 0.0,
        "deg_q25": float(np.quantile(degs, 0.25)),
        "deg_q50": float(np.quantile(degs, 0.50)),
        "deg_q75": float(np.quantile(degs, 0.75)),
    }


def _spectral_top_k(G: nx.Graph, k: int = 10) -> dict:
    """Top-k eigenvalues of the normalized Laplacian, ascending. Padded with
    NaN when N < k+1. Uses sparse eigsh for large N."""
    N = G.number_of_nodes()
    out = {f"lap_lam_{i}": float("nan") for i in range(k)}
    if N == 0 or G.number_of_edges() == 0:
        return out
    try:
        L = nx.normalized_laplacian_matrix(G).astype(float)
        if N <= k + 2:
            # dense path for small graphs
            evs = np.sort(np.linalg.eigvalsh(L.toarray()))
        else:
            # smallest k eigenvalues via sigma=0 shift-invert
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                evs = eigsh(L, k=min(k, N - 1), which="SM",
                            return_eigenvectors=False)
            evs = np.sort(evs)
        for i in range(min(k, len(evs))):
            out[f"lap_lam_{i}"] = float(evs[i])
    except Exception:
        pass
    return out


def _clustering_stats(G: nx.Graph) -> dict:
    if G.number_of_edges() == 0:
        return {"transitivity": 0.0, "avg_clustering": 0.0, "assortativity": 0.0}
    trans = nx.transitivity(G)
    avg_c = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assort = nx.degree_assortativity_coefficient(G)
        if not np.isfinite(assort):
            assort = 0.0
    except Exception:
        assort = 0.0
    return {"transitivity": float(trans), "avg_clustering": float(avg_c),
            "assortativity": float(assort)}


def _connectivity_stats(G: nx.Graph) -> dict:
    N = G.number_of_nodes()
    if N == 0:
        return {"n_components": 0.0, "largest_cc_frac": 0.0}
    ccs = list(nx.connected_components(G))
    return {
        "n_components": float(len(ccs)),
        "largest_cc_frac": float(max(len(c) for c in ccs) / N),
    }


def _path_stats(G: nx.Graph, subsample_threshold: int = 500,
                rng: Optional[np.random.Generator] = None) -> dict:
    N = G.number_of_nodes()
    if N == 0 or G.number_of_edges() == 0:
        return {"diameter_lcc": 0.0, "avg_spl": 0.0}
    rng = rng or np.random.default_rng(0)
    ccs = sorted(nx.connected_components(G), key=len, reverse=True)
    # `.copy()` is not cosmetic: on a subgraph VIEW every adjacency lookup pays
    # a Python-level node filter, which dominates the BFS below.  Materializing
    # the component makes the identical computation ~9x faster.
    lcc = G.subgraph(ccs[0]).copy()
    n_lcc = lcc.number_of_nodes()
    if n_lcc <= subsample_threshold:
        try:
            diam = float(nx.diameter(lcc))
            avg_spl = float(nx.average_shortest_path_length(lcc))
        except Exception:
            diam, avg_spl = 0.0, 0.0
        return {"diameter_lcc": diam, "avg_spl": avg_spl}
    # Subsample: pick 100 source nodes, BFS distances
    nodes = list(lcc.nodes())
    sources = rng.choice(nodes, size=min(100, n_lcc), replace=False)
    dists_all = []
    diam_est = 0
    for s in sources:
        lengths = nx.single_source_shortest_path_length(lcc, s)
        vals = list(lengths.values())
        if vals:
            diam_est = max(diam_est, max(vals))
            dists_all.extend(vals)
    return {"diameter_lcc": float(diam_est),
            "avg_spl": float(np.mean(dists_all)) if dists_all else 0.0}


def _weight_stats(G: nx.Graph) -> dict:
    """Only meaningful for weighted graphs (transition). Else NaN."""
    weights = [d.get("weight", None) for _, _, d in G.edges(data=True)]
    weights = [w for w in weights if w is not None]
    if not weights or all(w == 1 for w in weights):
        return {"w_mean": float("nan"), "w_std": float("nan")}
    w = np.array(weights, dtype=float)
    return {"w_mean": float(w.mean()), "w_std": float(w.std())}


# --------- diagnostics C ---------

def C_hvg_degree_exponent(G: nx.Graph) -> float:
    """Fit ln P(k) ~ -lambda * k on the tail (k >= 3). Return lambda.
    iid theory: lambda = ln(3/2) ≈ 0.405 (Luque 2010, PRE)."""
    if G.number_of_nodes() < 20:
        return float("nan")
    degs = np.array([d for _, d in G.degree()], dtype=int)
    if len(degs) == 0:
        return float("nan")
    max_k = degs.max()
    counts = np.bincount(degs, minlength=max_k + 1)
    # Tail from k >= 3 (exclude boundary artifacts on very-low-k nodes)
    k_min = 3
    ks, cs = [], []
    for k in range(k_min, max_k + 1):
        if counts[k] > 0:
            ks.append(k)
            cs.append(counts[k])
    if len(ks) < 3:
        return float("nan")
    ks = np.array(ks, dtype=float)
    logP = np.log(np.array(cs, dtype=float) / counts.sum())
    # weighted LS: weights ∝ counts (Poisson variance)
    w = np.array(cs, dtype=float)
    A = np.vstack([ks, np.ones_like(ks)]).T
    W = np.diag(w)
    try:
        coef, *_ = np.linalg.lstsq(W @ A, W @ logP, rcond=None)
        slope, _ = coef
        return float(-slope)
    except Exception:
        return float("nan")


def C_transition_entropy_rate(G: nx.DiGraph) -> float:
    """Entropy rate of the Markov chain implied by the transition graph.
    H = -sum_i pi_i sum_j P_ij log2 P_ij  (bits/step).
    Uses row-normalized transition matrix; stationary pi from empirical
    node in-degree strength (equivalent to stationary distribution when
    the chain is irreducible)."""
    if G.number_of_edges() == 0:
        return 0.0
    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    Q = len(nodes)
    P = np.zeros((Q, Q))
    for u, v, d in G.edges(data=True):
        P[idx[u], idx[v]] = d.get("weight", 1.0)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums_safe = np.where(row_sums == 0, 1.0, row_sums)
    P = P / row_sums_safe
    # Stationary distribution: leading left eigenvector of P
    try:
        w, V = np.linalg.eig(P.T)
        i_stat = int(np.argmin(np.abs(w - 1.0)))
        pi = np.real(V[:, i_stat])
        pi = pi / pi.sum()
        pi = np.clip(pi, 0, None)
        pi = pi / max(pi.sum(), 1e-12)
    except Exception:
        # Fallback: empirical node visit frequency
        total = P.sum()
        pi = P.sum(axis=1) / max(total, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logP = np.where(P > 0, np.log2(P), 0.0)
        row_H = -(P * logP).sum(axis=1)
    return float((pi * row_H).sum())


def C_takens_correlation_dim(G: nx.Graph, n_r: int = 8) -> float:
    """Correlation dimension D2 from Grassberger-Procaccia on the SAME
    embedding used to build the recurrence graph. We reuse the node-attr
    'point' when available (added by takens_recurrence_graph); otherwise
    fall back to log(N)/log(<k>) as a coarse fractal proxy.
    """
    N = G.number_of_nodes()
    if N < 50:
        return float("nan")
    pts = []
    for _, d in G.nodes(data=True):
        p = d.get("point", None)
        if p is not None:
            pts.append(p)
    if len(pts) < 50:
        # Fallback: local slope of log C(r) via graph — use degree scaling
        degs = np.array([d for _, d in G.degree()], dtype=float)
        mean_k = degs.mean()
        return float(np.log(N) / max(np.log(max(mean_k, 2)), 1e-6))

    X = np.asarray(pts, dtype=float)
    from scipy.spatial import cKDTree
    tree = cKDTree(X)
    # radii between the graph's epsilon-neighborhood scale and 1/10 of it
    eps = G.graph.get("epsilon", None)
    if eps is None or not np.isfinite(eps) or eps <= 0.0:
        # `epsilon` is zero whenever the target recurrence rate is already
        # reached by exactly-identical embedding vectors, which happens on
        # leads with long flat stretches (PTB-XL 12722 V5 is identically
        # zero, V6 has 26 distinct values in 1000 samples).  The threshold
        # carries no scale then, so bracket the radii from the cloud itself.
        radii = np.linalg.norm(X - X.mean(axis=0), axis=1)
        eps = float(np.median(radii))
    if not np.isfinite(eps) or eps <= 0.0:
        # The cloud has zero extent: D2 is undefined, not merely unknown.
        return float("nan")
    rs = np.geomspace(eps / 10, eps * 1.5, n_r)
    counts = []
    for r in rs:
        c = tree.count_neighbors(tree, r)
        # exclude self-pairs
        c = (c - len(X)) / 2
        counts.append(max(c, 1))
    log_r = np.log(rs)
    log_c = np.log(np.array(counts) / (len(X) * (len(X) - 1) / 2))
    # local slope in the middle region (exclude first/last point)
    slope, _ = np.polyfit(log_r[1:-1], log_c[1:-1], 1)
    return float(slope)


# --------- public API ---------

_COMMON_KEYS = [
    "deg_mean", "deg_std", "deg_max", "deg_logskew",
    "deg_q25", "deg_q50", "deg_q75",
    "lap_lam_0", "lap_lam_1", "lap_lam_2", "lap_lam_3", "lap_lam_4",
    "lap_lam_5", "lap_lam_6", "lap_lam_7", "lap_lam_8", "lap_lam_9",
    "transitivity", "avg_clustering", "assortativity",
    "n_components", "largest_cc_frac",
    "diameter_lcc", "avg_spl",
    "w_mean", "w_std",
]  # 26 keys


def extract_features(G, kind: str,
                     rng: Optional[np.random.Generator] = None) -> dict:
    """Extract the common feature dict + diagnostic C for a graph.

    Parameters
    ----------
    G : nx.Graph or nx.DiGraph
    kind : {"hvg", "transition", "takens"}
        Controls which C_* is appended (as key "C").
    rng : optional np.random.Generator for deterministic subsampling.

    Returns
    -------
    dict with keys in _COMMON_KEYS + "C".
    """
    if kind not in {"hvg", "transition", "takens"}:
        raise ValueError(f"unknown kind: {kind!r}")

    # For structure metrics we want the undirected view; keep the directed
    # G for C_transition_entropy_rate specifically.
    G_und = _to_undirected(G)

    feats: dict[str, float] = {}
    feats.update(_degree_stats(G_und))
    feats.update(_spectral_top_k(G_und, k=10))
    feats.update(_clustering_stats(G_und))
    feats.update(_connectivity_stats(G_und))
    feats.update(_path_stats(G_und, rng=rng))
    feats.update(_weight_stats(G))  # weights from original (directed for trans.)

    # sanity: all expected keys present
    for k in _COMMON_KEYS:
        feats.setdefault(k, float("nan"))

    # Diagnostic C
    if kind == "hvg":
        feats["C"] = C_hvg_degree_exponent(G_und)
    elif kind == "transition":
        feats["C"] = C_transition_entropy_rate(G)  # directed
    elif kind == "takens":
        feats["C"] = C_takens_correlation_dim(G_und)

    return feats


def features_to_array(feats: dict) -> tuple[np.ndarray, list[str]]:
    """Flatten a feature dict to a numpy array with a stable column order:
    _COMMON_KEYS in order, then 'C'. NaN entries are preserved for the
    caller to impute at the readout level (documented, not silent)."""
    keys = _COMMON_KEYS + ["C"]
    arr = np.array([feats.get(k, float("nan")) for k in keys], dtype=float)
    return arr, keys
