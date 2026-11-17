"""Bootstraps pre-registrados da ablacao HVG x L(HVG) da GNN (results/ablation_gnn/REGISTRO.md).

Fold 9 (regime exploratorio). Para cada variante, AUROC macro por semente e
media +- dp; bootstrap pareado de 1.000 reamostragens de registros, em que a
estatistica de cada variante e a media das AUROCs macro das 3 sementes.
Saida: results/ablation_gnn/bootstrap.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
A, L = ROOT / "results/ablation_gnn", ROOT / "results/line_gnn_cv_exploratory"
ROT = ["NORM", "MI", "STTC", "CD", "HYP"]

ARQ = {
    "HVG": [A / f"hvgnode_d64_l4_e50_test9_val8_seed{s}_test_scores.npz" for s in (42, 43, 44)],
    "HVG dir": [A / f"hvgnode_dir_d64_l4_e50_test9_val8_seed{s}_test_scores.npz" for s in (42, 43, 44)],
    "L(HVG)": [L / "hvg_d64_l4_e50_test9_val8_test_scores.npz",
               *[L / f"hvg_d64_l4_e50_test9_val8_seed{s}_test_scores.npz" for s in (43, 44)]],
    "L(HVG) dir": [L / "hvg_dir_d64_l4_e50_test9_val8_test_scores.npz",
                   *[L / f"hvg_dir_d64_l4_e50_test9_val8_seed{s}_test_scores.npz" for s in (43, 44)]],
}
CONTRASTES = [("L(HVG)", "HVG"), ("L(HVG) dir", "HVG dir"), ("HVG dir", "HVG"), ("L(HVG) dir", "L(HVG)")]


def auc(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def main() -> None:
    lab = pd.read_csv(ROOT / "export/ecg_fm_bundle/ptbxl/labels.csv").set_index("ecg_id")
    ids = None
    esc = {}
    for v, arqs in ARQ.items():
        esc[v] = []
        for a in arqs:
            z = np.load(a)
            if ids is None:
                ids = z["ecg_id"]
            assert np.array_equal(ids, z["ecg_id"]), a
            esc[v].append(z["scores"])
    assert set(lab.loc[ids, "strat_fold"]) == {9}
    y = lab.loc[ids, ROT].to_numpy()

    out = {"variantes": {}, "contrastes": {}}
    for v, ss in esc.items():
        a = [auc(y, s) for s in ss]
        out["variantes"][v] = {"por_semente": a, "media": float(np.mean(a)), "dp": float(np.std(a, ddof=1))}
        print(f"{v:12s} {np.mean(a):.4f} ± {np.std(a, ddof=1):.4f}  {np.round(a, 4)}")

    rng = np.random.default_rng(42)
    boot = {v: [] for v in esc}
    n = len(ids)
    for _ in range(1000):
        i = rng.integers(0, n, n)
        if y[i].min(0).max() == 1 or y[i].max(0).min() == 0:
            continue
        for v, ss in esc.items():
            boot[v].append(np.mean([auc(y[i], s[i]) for s in ss]))
    boot = {v: np.array(b) for v, b in boot.items()}
    for a, b in CONTRASTES:
        d = boot[a] - boot[b]
        obs = out["variantes"][a]["media"] - out["variantes"][b]["media"]
        lo, hi = np.percentile(d, [2.5, 97.5])
        out["contrastes"][f"{a} - {b}"] = {"delta": obs, "ic95": [float(lo), float(hi)],
                                            "p_menor_que_0": float(np.mean(d <= 0))}
        print(f"{a} − {b}: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]")
    json.dump(out, open(A / "bootstrap.json", "w"), indent=2)


if __name__ == "__main__":
    main()
