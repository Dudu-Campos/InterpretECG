"""Directed (time-respecting) reading of a visibility graph and its line graph.

The HVG is symmetric under time reversal: the HVG of x[::-1] is the mirror
image of the HVG of x, so every statistic of the undirected graph -- and of its
line graph (`line.py`) -- is identical for a signal played backwards.  This was
verified on PTB-XL: all 49 line-graph features unchanged under reversal.  The
ECG is strongly irreversible (fast QRS upstroke, slow T downstroke; P before
QRS, ST after), so that reading discards the arrow of time.

This module puts it back, on the SAME graph, so that "directed vs undirected"
is a controlled null: only the reading changes.

- Directed HVG (Lacasa et al. 2012, PRE 86, 'Time series irreversibility: a
  visibility graph approach'): each sample has an out-degree (visible samples
  in the future) and an in-degree (in the past); the irreversibility index is
  the KL divergence between the two degree distributions.
- Directed line graph: a wedge (a, s, b) is a CHAIN when the outer samples lie
  on opposite sides of s (a time-respecting path a -> s -> b), a FORWARD FORK
  when both lie in the future of s, a BACKWARD FORK when both lie in the past.
  The undirected family only had the opposite-sides fraction, which merges
  the two forks.
- Signed quantities: a chord (i, j), i < j, RISES when x_j > x_i.

Every feature here changes sign or swaps partner under reversal (forward <->
backward fork, in <-> out degree, rise <-> fall), which `tests` check.
Returned as two dicts: ordinal (invariant to increasing transforms of x) and
millivolt.
"""

from __future__ import annotations

import numpy as np


def _kl(p_counts: np.ndarray, q_counts: np.ndarray) -> float:
    """KL(p || q) of two count vectors on the same support, add-one smoothed."""
    p = (p_counts + 1.0) / (p_counts.sum() + len(p_counts))
    q = (q_counts + 1.0) / (q_counts.sum() + len(q_counts))
    return float(np.sum(p * np.log(p / q)))


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or left.std() <= 1e-12 or right.std() <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _q(values: np.ndarray, prefix: str, quantiles=(10, 50, 90)) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return {f"{prefix}_mean": 0.0, **{f"{prefix}_q{q}": 0.0 for q in quantiles}}
    return {f"{prefix}_mean": float(values.mean()),
            **{f"{prefix}_q{q}": float(v) for q, v in zip(quantiles, np.percentile(values, quantiles))}}


def _frac(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def wedges(n_nodes: int, source: np.ndarray, target: np.ndarray):
    """All wedges (edge_a, edge_b, shared node), enumerated per shared node."""
    degree = np.bincount(np.concatenate([source, target]), minlength=n_nodes)
    incidente = np.concatenate([source, target])
    aresta = np.concatenate([np.arange(len(source))] * 2)
    ordem = np.argsort(incidente, kind="stable")
    incidente, aresta = incidente[ordem], aresta[ordem]
    inicios = np.searchsorted(incidente, np.arange(n_nodes))
    partes_a, partes_b, partes_s = [], [], []
    for s in np.flatnonzero(degree >= 2):
        ids = aresta[inicios[s]:inicios[s] + degree[s]]
        a, b = np.triu_indices(len(ids), 1)
        partes_a.append(ids[a]); partes_b.append(ids[b]); partes_s.append(np.full(len(a), s))
    if not partes_a:
        vazio = np.zeros(0, dtype=np.int64)
        return vazio, vazio, vazio
    return np.concatenate(partes_a), np.concatenate(partes_b), np.concatenate(partes_s)


def directed_graph_features(x: np.ndarray, edges: np.ndarray) -> tuple[dict, dict]:
    """Return (ordinal, millivolt) time-directed features of a visibility graph and its L(G)."""
    x = np.asarray(x, dtype=float)
    source, target = np.sort(np.asarray(edges, dtype=np.int64), axis=1).T   # source < target
    n = len(x)
    k_out = np.bincount(source, minlength=n)     # edges to the future
    k_in = np.bincount(target, minlength=n)      # edges to the past
    support = int(max(k_out.max(), k_in.max())) + 1
    p_out = np.bincount(k_out, minlength=support).astype(float)
    p_in = np.bincount(k_in, minlength=support).astype(float)
    diff = (k_out - k_in).astype(float)
    top = x >= np.percentile(x, 95)              # the peaks (R waves, mostly)

    edge_a, edge_b, shared = wedges(n, source, target)
    outer_a = np.where(source[edge_a] == shared, target[edge_a], source[edge_a])
    outer_b = np.where(source[edge_b] == shared, target[edge_b], source[edge_b])
    future_a, future_b = outer_a > shared, outer_b > shared
    chain = future_a != future_b
    forward = future_a & future_b
    backward = ~future_a & ~future_b
    dominant = (x[shared] > x[outer_a]) & (x[shared] > x[outer_b])

    span = (target - source).astype(float)
    rises = x[target] > x[source]
    falls = x[target] < x[source]
    long = span >= np.percentile(span, 90)

    n_fwd, n_bwd = forward.sum(), backward.sum()
    ordinal = {
        "dir_irreversibility_kld": _kl(p_out, p_in),
        "dir_irreversibility_kld_rev": _kl(p_in, p_out),
        "dir_kout_minus_kin_peaks": float(diff[top].mean()),
        "dir_kout_minus_kin_rank_corr": _corr(diff, np.argsort(np.argsort(x)).astype(float)),
        "dir_kout_q90": float(np.percentile(k_out, 90)),
        "dir_kin_q90": float(np.percentile(k_in, 90)),
        "dir_wedge_forward_fork_frac": _frac(forward),
        "dir_wedge_backward_fork_frac": _frac(backward),
        "dir_fork_log_ratio": float(np.log((n_fwd + 1.0) / (n_bwd + 1.0))),
        "dir_forward_fork_dominant_frac": _frac(dominant[forward]),
        "dir_backward_fork_dominant_frac": _frac(dominant[backward]),
        "dir_chord_rise_frac": _frac(rises),
        "dir_long_chord_rise_frac": _frac(rises[long]),
        "dir_rise_minus_fall_span": float(span[rises].mean() - span[falls].mean())
        if rises.any() and falls.any() else 0.0,
    }

    signed_step = x[target] - x[source]
    level = np.minimum(x[source], x[target])
    # chains a -> s -> b, oriented in time: incoming chord ends at s, outgoing starts at s
    entrada = np.where(future_a[chain], edge_b[chain], edge_a[chain])
    saida = np.where(future_a[chain], edge_a[chain], edge_b[chain])
    antes = np.where(future_a[chain], outer_b[chain], outer_a[chain])
    depois = np.where(future_a[chain], outer_a[chain], outer_b[chain])
    millivolt = {
        **_q(signed_step, "dir_signed_step"),
        "dir_signed_step_long_mean": float(signed_step[long].mean()) if long.any() else 0.0,
        "dir_span_signed_step_corr": _corr(span, signed_step),
        **_q(level[saida] - level[entrada], "dir_chain_level_change"),
        **_q(x[depois] - x[antes], "dir_chain_outer_signed_distance"),
        "dir_rise_step_mean": float(signed_step[rises].mean()) if rises.any() else 0.0,
        "dir_fall_step_mean": float(signed_step[falls].mean()) if falls.any() else 0.0,
    }
    return ordinal, millivolt
