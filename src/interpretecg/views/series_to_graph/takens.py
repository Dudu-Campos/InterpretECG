"""Takens recurrence network.

Embed the series in R^m via time-delay coordinates and connect vectors that
lie within an ε-neighborhood.

References
----------
Marwan et al. 2009 — recurrence networks.
Silva et al. 2021 (arXiv 2110.09887) — "proximity" column of the survey.

Contract
--------
- Input: 1D np.ndarray of length L.
- Output: nx.Graph on N = L - (m-1)*tau nodes; each node carries `t` (index of
  first coordinate of its delay vector) and `embed` (the vector, optional).
- Parameters: m, tau, and ε-strategy (recurrence rate `r` or absolute ε).
- Theiler window w=1: forbid edges between temporally adjacent vectors.
- Diagonal density is controlled by r (default r=0.05), NOT ε absolute —
  isolates structure from signal scale.

Auto-selection helpers (`auto_tau_mi`, `auto_m_fnn`) provided but the
pipeline caller is expected to fix τ, m at the dataset level (median of
per-series estimates) to keep the conversion training-free AND avoid
per-series hyperparameter jitter as a confound.
"""
from __future__ import annotations
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree


# ---------- parameter selection ----------

def auto_tau_mi(x: np.ndarray, max_lag: int = 50, n_bins: int = 16) -> int:
    """First local minimum of mutual information I(x_t; x_{t+τ}).

    Histogram-based MI (fast, adequate for τ selection). Falls back to
    ``max_lag`` if no minimum is found before the ceiling.
    """
    x = np.asarray(x, dtype=float).ravel()
    L = len(x)
    max_lag = min(max_lag, L // 3)
    if max_lag < 2:
        return 1

    # Fixed-support bins over the whole series
    edges = np.histogram_bin_edges(x, bins=n_bins)
    b = np.clip(np.digitize(x, edges[1:-1]), 0, n_bins - 1)

    def _mi(lag: int) -> float:
        a = b[:-lag]
        c = b[lag:]
        joint, _, _ = np.histogram2d(a, c, bins=n_bins)
        pxy = joint / joint.sum()
        px = pxy.sum(axis=1, keepdims=True)
        py = pxy.sum(axis=0, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_term = np.log(pxy / (px * py))
            log_term[~np.isfinite(log_term)] = 0.0
        return float((pxy * log_term).sum())

    prev = _mi(1)
    curr = _mi(2)
    for lag in range(2, max_lag):
        nxt = _mi(lag + 1)
        if curr < prev and curr < nxt:
            return lag
        prev, curr = curr, nxt
    return max_lag


def auto_m_fnn(
    x: np.ndarray,
    tau: int,
    m_max: int = 10,
    r_tol: float = 15.0,
    threshold: float = 0.01,
) -> int:
    """Cao-style false-nearest-neighbors (Kennel 1992 simplified).

    Increase m until FNN fraction drops below `threshold`.
    Returns m ≥ 2; capped at m_max.
    """
    x = np.asarray(x, dtype=float).ravel()
    L = len(x)

    def _embed(m: int):
        N = L - (m - 1) * tau
        if N < 2:
            return None
        return np.stack([x[i * tau : i * tau + N] for i in range(m)], axis=1)

    prev_emb = _embed(2)
    if prev_emb is None:
        return 2
    for m in range(2, m_max):
        emb = _embed(m + 1)
        if emb is None:
            return m
        # In dim m: find nearest neighbor for each point (exclude self)
        N_m = len(prev_emb)
        tree = cKDTree(prev_emb)
        d, idx = tree.query(prev_emb, k=2)
        d_nn = d[:, 1]
        j_nn = idx[:, 1]
        # In dim m+1: distance between the SAME pair of vectors (need both
        # embeddings to have valid indices; embedding m+1 has one fewer point).
        N_m1 = len(emb)
        valid = (np.arange(N_m) < N_m1) & (j_nn < N_m1)
        i_v = np.where(valid)[0]
        j_v = j_nn[valid]
        d_new_v = np.linalg.norm(emb[i_v] - emb[j_v], axis=1)
        d_nn_v = d_nn[valid]
        # FNN criterion: relative jump in distance from dim m to dim m+1
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                d_nn_v > 0,
                np.abs(d_new_v ** 2 - d_nn_v ** 2) ** 0.5 / d_nn_v,
                0.0,
            )
        fnn_frac = float(np.mean(ratio > r_tol)) if len(ratio) else 0.0
        if fnn_frac < threshold:
            return m
        prev_emb = emb
    return m_max


# ---------- graph construction ----------

def _embed(x: np.ndarray, m: int, tau: int) -> np.ndarray:
    L = len(x)
    N = L - (m - 1) * tau
    if N < 2:
        raise ValueError(
            f"embedding empty: L={L}, m={m}, tau={tau} → N={N}"
        )
    return np.stack([x[i * tau : i * tau + N] for i in range(m)], axis=1)


def takens_recurrence_graph(
    x: np.ndarray,
    m: int | None = None,
    tau: int | None = None,
    recurrence_rate: float | None = 0.05,
    epsilon: float | None = None,
    theiler_window: int = 1,
) -> nx.Graph:
    """Build a recurrence network from a Takens embedding.

    Parameters
    ----------
    x : np.ndarray shape (L,)
    m, tau : embedding dimension and delay. If None, estimated via
        `auto_m_fnn` / `auto_tau_mi` (per-series). Pipeline caller SHOULD
        pass explicit m, tau chosen at dataset level.
    recurrence_rate : target density of the recurrence matrix (fraction of
        off-diagonal pairs connected). Default 0.05. If provided, ε is
        chosen automatically as the corresponding distance quantile.
        Mutually exclusive with `epsilon`.
    epsilon : absolute distance threshold. Overrides recurrence_rate.
    theiler_window : forbid edges between vectors t_i, t_j with
        |i-j| ≤ theiler_window. Default 1 (blocks trivial temporal adjacency).

    Returns
    -------
    nx.Graph on N = L - (m-1)*tau nodes, each with `t` attribute.
    """
    x = np.asarray(x, dtype=float).ravel()
    if tau is None:
        tau = auto_tau_mi(x)
    if m is None:
        m = auto_m_fnn(x, tau)

    V = _embed(x, m=m, tau=tau)
    N = len(V)

    # Pairwise distances via KDTree (memory-friendly for large N)
    tree = cKDTree(V)

    if epsilon is None:
        if recurrence_rate is None:
            raise ValueError("provide either epsilon or recurrence_rate")
        # Sample pairs to estimate quantile without building full matrix
        # For moderate N we can afford full matrix; for large N use sample.
        if N <= 2000:
            # full pairwise
            d = np.linalg.norm(V[:, None, :] - V[None, :, :], axis=2)
            iu = np.triu_indices(N, k=theiler_window + 1)
            if len(iu[0]) == 0:
                raise ValueError("theiler_window too large; no valid pairs")
            epsilon = float(np.quantile(d[iu], recurrence_rate))
        else:
            # random sample of pairs (~1e5) to estimate the quantile
            rng = np.random.default_rng(0)
            n_sample = min(100_000, N * (N - 1) // 2)
            i_s = rng.integers(0, N, size=n_sample)
            j_s = rng.integers(0, N, size=n_sample)
            mask = np.abs(i_s - j_s) > theiler_window
            i_s, j_s = i_s[mask], j_s[mask]
            d_s = np.linalg.norm(V[i_s] - V[j_s], axis=1)
            epsilon = float(np.quantile(d_s, recurrence_rate))

    # Query ε-neighborhood; returns list of neighbor lists
    pairs = tree.query_pairs(r=epsilon, output_type="ndarray")
    if len(pairs) > 0:
        # Theiler filter
        keep = np.abs(pairs[:, 0] - pairs[:, 1]) > theiler_window
        pairs = pairs[keep]

    G = nx.Graph()
    for i in range(N):
        # store the embedding vector as a node attribute so downstream
        # feature extractors (e.g. correlation dimension) can reuse it
        # without recomputing the Takens embedding.
        G.add_node(i, t=i, point=V[i].tolist())
    G.add_edges_from(pairs.tolist())

    G.graph["m"] = m
    G.graph["tau"] = tau
    G.graph["epsilon"] = epsilon
    G.graph["recurrence_rate"] = recurrence_rate
    return G
