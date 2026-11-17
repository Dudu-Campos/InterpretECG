"""Positive controls for the phi_free feature extractor.

Two things to check:
1. The extractor runs on all three conversion types and produces the
   expected 26-key dict without NaN in the load-bearing columns for
   reasonable-size graphs.
2. The diagnostic C behaves as theory predicts:
   - HVG on iid noise: lambda ≈ ln(3/2) ≈ 0.405 (Luque 2010)
   - Transition entropy rate: periodic ≪ iid
   - Takens D2: Lorenz ≈ 2.05 ± 0.5 (literature ≈ 2.05); iid noise fills the
     embedding space so D2 approaches m.
"""
from __future__ import annotations
import numpy as np
import networkx as nx

from interpretecg.views.series_to_graph import (
    visibility_graph,
    transition_graph,
    takens_recurrence_graph,
    extract_features,
    features_to_array,
)


rng = np.random.default_rng(0)


def _lorenz(n=3000, dt=0.01, sigma=10, rho=28, beta=8/3):
    xs = np.zeros(n); ys = np.zeros(n); zs = np.zeros(n)
    xs[0], ys[0], zs[0] = 0.1, 0.0, 0.0
    for i in range(1, n):
        dx = sigma * (ys[i-1] - xs[i-1])
        dy = xs[i-1] * (rho - zs[i-1]) - ys[i-1]
        dz = xs[i-1] * ys[i-1] - beta * zs[i-1]
        xs[i] = xs[i-1] + dx*dt
        ys[i] = ys[i-1] + dy*dt
        zs[i] = zs[i-1] + dz*dt
    return xs[500:]


def test_hvg_features_iid():
    x = rng.normal(size=3000)
    G = visibility_graph(x, kind="hvg")
    feats = extract_features(G, "hvg")
    print(f"[HVG iid features] n_keys={len(feats)}")
    for k, v in feats.items():
        print(f"    {k:20s} = {v:.4f}" if isinstance(v, float) and np.isfinite(v)
              else f"    {k:20s} = {v}")
    # Load-bearing sanity
    assert 3.5 < feats["deg_mean"] < 4.5, feats["deg_mean"]
    assert np.isfinite(feats["C"]), "C should be finite for HVG on iid"
    # Luque theory: lambda = ln(3/2) ≈ 0.405 (this is a REGRESSION fit on
    # noisy binned counts, so tolerance is loose but band should straddle it)
    print(f"    lambda_fit = {feats['C']:.4f} (theory: 0.405)")
    assert 0.25 < feats["C"] < 0.65, f"HVG iid lambda off theory: {feats['C']}"
    return feats["C"]


def test_transition_features_contrast():
    L = 5000
    t = np.arange(L)
    x_per = np.sin(2 * np.pi * t / 20)
    x_iid = rng.normal(size=L)
    f_per = extract_features(transition_graph(x_per, n_bins=10), "transition")
    f_iid = extract_features(transition_graph(x_iid, n_bins=10), "transition")
    print(f"[Trans features] entropy_rate periodic = {f_per['C']:.4f} bits/step")
    print(f"[Trans features] entropy_rate iid      = {f_iid['C']:.4f} bits/step "
          f"(theory upper bound = log2(10) = {np.log2(10):.4f})")
    assert f_iid["C"] > f_per["C"], (f_iid["C"], f_per["C"])
    # iid on Q=10 bins should approach log2(10) ≈ 3.32 bits
    assert 2.5 < f_iid["C"] < 3.5, f"iid entropy rate off theory: {f_iid['C']}"
    return f_per["C"], f_iid["C"]


def test_takens_features_lorenz_vs_iid():
    x_lor = _lorenz(n=3000)
    x_iid = rng.normal(size=2500)
    G_lor = takens_recurrence_graph(x_lor, m=3, tau=10, recurrence_rate=0.05)
    G_iid = takens_recurrence_graph(x_iid, m=3, tau=5, recurrence_rate=0.05)
    f_lor = extract_features(G_lor, "takens")
    f_iid = extract_features(G_iid, "takens")
    print(f"[Takens features] D2 Lorenz     = {f_lor['C']:.4f} "
          f"(literature ≈ 2.05)")
    print(f"[Takens features] D2 iid (m=3)  = {f_iid['C']:.4f} "
          f"(theory: should approach 3.0)")
    print(f"[Takens features] transitivity Lorenz = {f_lor['transitivity']:.4f}, "
          f"iid = {f_iid['transitivity']:.4f}")
    # Lorenz D2 should be well below iid D2 (attractor is low-dim)
    assert f_lor["C"] < f_iid["C"], (f_lor["C"], f_iid["C"])
    # Lorenz D2 in a reasonable band around 2.0-2.5
    assert 1.4 < f_lor["C"] < 2.8, f"Lorenz D2 off literature: {f_lor['C']}"
    return f_lor["C"], f_iid["C"]


def test_features_to_array_shape():
    x = rng.normal(size=1000)
    G = visibility_graph(x)
    feats = extract_features(G, "hvg")
    arr, cols = features_to_array(feats)
    print(f"[shape] arr.shape = {arr.shape}, n_cols = {len(cols)}")
    assert arr.shape == (27,), arr.shape  # 26 common + 1 diagnostic C
    assert cols[-1] == "C"
    return arr.shape


if __name__ == "__main__":
    print("=" * 60)
    print("HVG feature extractor")
    print("=" * 60)
    test_hvg_features_iid()
    print()
    print("=" * 60)
    print("Transition feature extractor")
    print("=" * 60)
    test_transition_features_contrast()
    print()
    print("=" * 60)
    print("Takens feature extractor")
    print("=" * 60)
    test_takens_features_lorenz_vs_iid()
    print()
    print("=" * 60)
    print("Array shape")
    print("=" * 60)
    test_features_to_array_shape()
    print()
    print("ALL PASS")


def test_takens_d2_survives_zero_recurrence_threshold():
    """A lead with long flat stretches drives `epsilon` to zero.

    When the target recurrence rate is already reached by exactly-identical
    embedding vectors, the quantile threshold is 0 and the radius bracket of
    the Grassberger-Procaccia fit has no scale.  The estimator must fall back
    to the point cloud instead of raising, as it did on PTB-XL record 12722
    (lead V6: 26 distinct values in 1000 samples).
    """
    x = np.zeros(1000)
    x[::97] = 1.0  # flat with sparse spikes: >5% of pairs are identical
    G = takens_recurrence_graph(x, m=11, tau=12, recurrence_rate=0.05)
    assert G.graph["epsilon"] == 0.0

    values, names = features_to_array(extract_features(G, kind="takens"))
    d2 = float(values[names.index("C")])
    assert np.isfinite(d2) and d2 > 0.0


def test_takens_d2_is_nan_for_a_zero_extent_cloud():
    """A constant lead gives a single embedding point: D2 is undefined."""
    G = takens_recurrence_graph(np.zeros(1000), m=12, tau=12, recurrence_rate=0.05)
    values, names = features_to_array(extract_features(G, kind="takens"))
    assert np.isnan(values[names.index("C")])
