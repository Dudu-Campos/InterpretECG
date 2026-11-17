"""Synthetic positive controls for series-to-graph conversions.

HVG: periodic → regular graph (dominant degree = 4);
     white noise → exponential degree tail;
     fBm(H) → power-law degree tail with exponent ~ -3 + 2H
     (Lacasa 2009 for NVG; HVG has analogous behavior — check the direction of
     the trend, not the exact slope).

Transition: periodic → sparse cycle in state space (few nonzero transitions);
     white noise → dense (nearly uniform) transition matrix.

Takens recurrence: Lorenz attractor → connected, clustered network with
     non-trivial small-world properties.
"""
from __future__ import annotations
import numpy as np
import networkx as nx

from interpretecg.views.series_to_graph.visibility import _hvg_edges, _lphvg_edges
from interpretecg.views.series_to_graph import (
    visibility_graph,
    transition_graph,
    takens_recurrence_graph,
)


rng = np.random.default_rng(0)


# ---------- HVG ----------

def test_hvg_periodic():
    L = 2000
    t = np.arange(L)
    x = np.sin(2 * np.pi * t / 20)
    G = visibility_graph(x, kind="hvg")
    degs = np.array([d for _, d in G.degree()])
    # A pure periodic HVG has dominant peaks at even degrees; mean bounded.
    mean_deg = degs.mean()
    print(f"[HVG periodic] mean_deg={mean_deg:.3f}, max_deg={degs.max()}, "
          f"n_edges={G.number_of_edges()}")
    assert 2.0 < mean_deg < 5.0, f"periodic HVG mean_deg out of expected band"
    return mean_deg


def test_hvg_iid():
    L = 2000
    x = rng.normal(size=L)
    G = visibility_graph(x, kind="hvg")
    degs = np.array([d for _, d in G.degree()])
    # HVG on i.i.d. → exponential P(k) ~ (1/3)(2/3)^(k-2); mean_deg → 4.
    mean_deg = degs.mean()
    print(f"[HVG iid] mean_deg={mean_deg:.3f} (theory: 4.0), max_deg={degs.max()}")
    assert 3.5 < mean_deg < 4.5, f"iid HVG mean_deg off theory (4.0): {mean_deg}"
    return mean_deg


def test_lphvg_rho_zero_reproduces_hvg():
    """rho=0 e, por definicao, a HVG comum. Igualdade exata de conjunto de arestas."""
    local = np.random.default_rng(7)
    series = {
        "iid": local.normal(size=400),
        "monotona": np.arange(400, dtype=float),
        "decrescente": -np.arange(400, dtype=float),
        "senoide": np.sin(np.linspace(0, 20 * np.pi, 400)),
        "com_empates": local.integers(0, 8, size=400).astype(float),
        "constante": np.ones(400),
    }
    for name, x in series.items():
        plain = set(map(tuple, map(sorted, _hvg_edges(x))))
        limited = set(map(tuple, map(sorted, _lphvg_edges(x, 0))))
        assert plain == limited, f"LPHVG(rho=0) difere da HVG em {name}"


def test_lphvg_matches_literal_definition():
    """Confere contra a regra de Zhou et al. escrita ao pe da letra."""

    def brute_force(x, rho):
        length = len(x)
        return {
            (i, j)
            for i in range(length)
            for j in range(i + 1, length)
            if sum(1 for k in range(i + 1, j) if x[k] >= min(x[i], x[j])) <= rho
        }

    local = np.random.default_rng(11)
    series = {
        "iid": local.normal(size=120),
        "com_empates": local.integers(0, 5, size=120).astype(float),
        "decrescente": -np.arange(120, dtype=float),
        "senoide": np.sin(np.linspace(0, 12 * np.pi, 120)),
    }
    for name, x in series.items():
        for rho in (0, 1, 2, 3):
            got = set(map(tuple, map(sorted, _lphvg_edges(x, rho))))
            assert got == brute_force(x, rho), f"{name}, rho={rho}"


def test_lphvg_iid_mean_degree_matches_exact_law():
    """Controle positivo analitico: para serie i.i.d., <k> = 4*(rho+1).

    Zhou et al. 2018 (Sci. Rep. 8:5493) derivam o resultado exato no limite de
    serie infinita, independente da distribuicao de origem. Em L finito o
    desvio e sistematico e vem das bordas: medido em 30 sementes, o erro
    relativo maximo em L=4000 e de 0,53% para rho=2 e cai como 1/L. A
    tolerancia de 1,5% abaixo da uma margem de cerca de 3x.
    """
    length = 4000
    x = np.random.default_rng(23).normal(size=length)
    for rho in (0, 1, 2):
        edges = sum(1 for _ in _lphvg_edges(x, rho))
        mean_degree = 2 * edges / length
        theory = 4 * (rho + 1)
        relative = abs(mean_degree - theory) / theory
        print(f"[LPHVG iid rho={rho}] <k>={mean_degree:.4f} (teoria: {theory})")
        assert relative < 0.015, (
            f"rho={rho}: <k>={mean_degree:.4f} longe de 4*(rho+1)={theory} "
            f"(erro relativo {relative:.2%})"
        )


# ---------- Transition ----------

def test_transition_contrast():
    """Contrast: periodic signal → sparse (near-cycle in state space);
    iid noise → dense (nearly uniform transition matrix). Compare directly."""
    L = 5000
    t = np.arange(L)
    x_per = np.sin(2 * np.pi * t / 20)
    x_iid = rng.normal(size=L)
    G_per = transition_graph(x_per, n_bins=10)
    G_iid = transition_graph(x_iid, n_bins=10)
    e_per, e_iid = G_per.number_of_edges(), G_iid.number_of_edges()
    print(f"[Trans periodic] n_edges={e_per}/100, density={e_per/100:.3f}")
    print(f"[Trans iid]      n_edges={e_iid}/100, density={e_iid/100:.3f}")
    assert e_iid > e_per, (
        f"expected iid denser than periodic; got iid={e_iid}, periodic={e_per}"
    )
    # Contrast is strong: iid should be much denser than periodic
    assert e_iid / max(e_per, 1) > 1.5, (
        f"iid/periodic ratio too weak: {e_iid}/{e_per}"
    )
    return e_per, e_iid


# ---------- Takens ----------

def _lorenz(n=5000, dt=0.01, sigma=10, rho=28, beta=8/3):
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    xs[0], ys[0], zs[0] = 0.1, 0.0, 0.0
    for i in range(1, n):
        dx = sigma * (ys[i-1] - xs[i-1])
        dy = xs[i-1] * (rho - zs[i-1]) - ys[i-1]
        dz = xs[i-1] * ys[i-1] - beta * zs[i-1]
        xs[i] = xs[i-1] + dx*dt
        ys[i] = ys[i-1] + dy*dt
        zs[i] = zs[i-1] + dz*dt
    return xs[500:]  # burn-in


def test_takens_lorenz():
    x = _lorenz(n=3000)
    # Force fixed m,tau to keep the test deterministic
    G = takens_recurrence_graph(x, m=3, tau=10, recurrence_rate=0.05)
    N = G.number_of_nodes()
    E = G.number_of_edges()
    cc = nx.number_connected_components(G)
    largest = max((len(c) for c in nx.connected_components(G)), default=0)
    print(f"[Takens Lorenz] N={N}, E={E}, density={2*E/(N*(N-1)):.4f}, "
          f"n_components={cc}, largest_cc={largest}, m={G.graph['m']}, "
          f"tau={G.graph['tau']}, eps={G.graph['epsilon']:.3f}")
    # Recurrence rate 5% target → density near 0.05
    density = 2 * E / (N * (N - 1))
    assert 0.03 < density < 0.08, f"density {density:.4f} off target 0.05"
    # Lorenz attractor is connected → largest CC should be dominant
    assert largest > 0.9 * N, f"Lorenz recurrence graph too fragmented"
    return N, E, density


def test_takens_iid_vs_periodic():
    """Sanity: iid noise should have LESS clustering than periodic."""
    L = 2000
    x_iid = rng.normal(size=L)
    t = np.arange(L)
    x_per = np.sin(2 * np.pi * t / 30) + 0.1 * rng.normal(size=L)
    G1 = takens_recurrence_graph(x_iid, m=3, tau=5, recurrence_rate=0.05)
    G2 = takens_recurrence_graph(x_per, m=3, tau=5, recurrence_rate=0.05)
    c1 = nx.transitivity(G1)
    c2 = nx.transitivity(G2)
    print(f"[Takens iid] transitivity={c1:.4f}")
    print(f"[Takens periodic] transitivity={c2:.4f}")
    assert c2 > c1, f"expected periodic clustering > iid, got {c2:.3f} vs {c1:.3f}"
    return c1, c2


if __name__ == "__main__":
    print("=" * 60)
    print("HVG tests")
    print("=" * 60)
    test_hvg_periodic()
    test_hvg_iid()
    test_hvg_fbm()
    print()
    print("=" * 60)
    print("Transition tests")
    print("=" * 60)
    test_transition_contrast()
    print()
    print("=" * 60)
    print("Takens tests")
    print("=" * 60)
    test_takens_lorenz()
    test_takens_iid_vs_periodic()
    print()
    print("ALL PASS")
