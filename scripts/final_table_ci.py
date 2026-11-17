"""Tabela final (fold 10) nos tres datasets: AUROC macro, media das sementes, IC por bootstrap de registros.

Mesmo procedimento de ptbxl_table_ci.py, estendido ao chapman8 e ao code15:
para cada modelo, a estatistica e a media das AUROCs macro das sementes; o IC
vem de 2.000 reamostragens dos registros do fold 10, as MESMAS para todos os
modelos de um dataset (semente 42), o que tambem da as diferencas pareadas
contra o nosso boosting. IC de 95%.
So modelos pre-registrados; modelos ainda sem escores sao pulados.
Saida: results/final_table_ci.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ptbxl_table_ci import MODELOS as MODELOS_PTBXL, ORDEM_BOOSTING  # noqa: E402

R = ROOT / "results"
S = range(42, 47)
B = 2000


def externos(ds: str) -> dict:
    D = R / "external" / ds
    nosso = "todas_familias"
    m = {
        "catch22 + boosting": ([D / f"seeds/boosting_catch22_seed{s}_scores.npz" for s in S], "test",
                               D / "boosting_catch22_scores.npz"),
        "tsfresh + boosting": ([D / f"seeds/boosting_tsfresh_seed{s}_scores.npz" for s in S], "test",
                               D / "boosting_tsfresh_pruned_scores.npz"),
        "nosso boosting": ([D / f"seeds/boosting_{nosso}_seed{s}_scores.npz" for s in S], "test",
                           D / f"boosting_{nosso}_pruned_scores.npz"),
    }
    for nome, arch in (("xresnet1d101", "xresnet1d101"), ("inception1d", "inception1d"),
                       ("LSTM bidirecional", "lstm_bidir"), ("Transformer", "transformer")):
        m[nome] = ([D / f"deep/{arch}_confirmatory_seed{s}/test_scores.npz" for s in S], "escores", None)
    m["ECG-IMN"] = ([D / f"deep/imn/{ds}_multilabel5_confirmatory{'' if s == 42 else f'_seed{s}'}_test_scores.npz"
                     for s in S], "scores", None)
    melhor = {"chapman8": ("GNN na melhor visão (Takens)", "takens"),
              "code15": ("GNN na melhor visão (L(HVG))", "hvg")}[ds]
    m[melhor[0]] = ([D / f"deep/gnn/{melhor[1]}_dir_d64_l4_e50_conf_seed{s}_test_scores.npz" for s in S], "scores", None)
    if ds == "chapman8":
        m["GNN em L(HVG) (discussão)"] = ([D / f"deep/gnn/hvg_dir_d64_l4_e50_conf_seed{s}_test_scores.npz" for s in S],
                                          "scores", None)
    return m


def ptbxl() -> dict:
    m = {}
    for nome, (arqs, chave) in MODELOS_PTBXL.items():
        ordem = R / f"confirmatory/{ORDEM_BOOSTING[nome]}_scores.npz" if nome in ORDEM_BOOSTING else None
        m[nome if not nome.startswith("GNN") else "GNN na melhor visão (L(HVG))"] = (arqs, chave, ordem)
    return m


def auc(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def carregar(modelos: dict):
    ids_ref, escores, y_ref = None, {}, None
    for nome, (arqs, chave, ordem) in modelos.items():
        lista = []
        for a in arqs:
            if not Path(a).exists():
                continue
            z = np.load(a)
            ids = np.load(ordem)["ecg_id_test"] if ordem is not None else z["ecg_id"]
            ids_ref = np.sort(ids) if ids_ref is None else ids_ref
            assert np.array_equal(np.sort(ids), ids_ref), (nome, a)
            lista.append(pd.DataFrame(z[chave], index=ids).loc[ids_ref].to_numpy())
            if ordem is not None and "y_test" in z:
                y = pd.DataFrame(z["y_test"], index=ids).loc[ids_ref].to_numpy()
                assert y_ref is None or np.array_equal(y, y_ref)
                y_ref = y
        if lista:
            escores[nome] = lista
    return ids_ref, escores, y_ref


def reamostra(y, escores, idx):
    return {m: [np.mean([auc(y[i], s[i]) for s in ss]) for i in idx] for m, ss in escores.items()}


def main() -> None:
    linhas = []
    for ds in ("ptbxl", "chapman8", "code15"):
        modelos = ptbxl() if ds == "ptbxl" else externos(ds)
        ids, escores, y = carregar(modelos)
        if ds == "ptbxl":
            lab = pd.read_csv(ROOT / "export/ecg_fm_bundle/ptbxl/labels.csv").set_index("ecg_id")
            assert set(lab.loc[ids, "strat_fold"]) == {10}
            y2 = lab.loc[ids, ["NORM", "MI", "STTC", "CD", "HYP"]].to_numpy()
            assert y is None or np.array_equal(y, y2)
            y = y2
        else:
            lab = pd.read_csv(ROOT / "data/external" / ds / "labels.csv").set_index("ecg_id")
            assert set(lab.loc[ids, "strat_fold"]) == {10}
        rng = np.random.default_rng(42)
        n, idx = len(y), []
        while len(idx) < B:
            i = rng.integers(0, n, n)
            if (y[i].sum(0) == 0).any() or (y[i].sum(0) == n).any():
                continue
            idx.append(i)
        partes = Parallel(n_jobs=12)(delayed(reamostra)(y, escores, c) for c in np.array_split(np.arange(B), 12)
                                     for c in [[idx[k] for k in c]])
        boot = {m: np.concatenate([p[m] for p in partes]) for m in escores}
        ref = np.mean([auc(y, s) for s in escores["nosso boosting"]])
        for m, ss in escores.items():
            por = [auc(y, s) for s in ss]
            d = boot[m] - boot["nosso boosting"]
            linhas.append({"dataset": ds, "modelo": m, "sementes": len(ss), "auroc": np.mean(por),
                           "dp_sementes": np.std(por, ddof=1) if len(por) > 1 else np.nan,
                           "ic95_lo": np.percentile(boot[m], 2.5), "ic95_hi": np.percentile(boot[m], 97.5),
                           "dif_vs_nosso": np.mean(por) - ref,
                           "dif95_lo": np.percentile(d, 2.5), "dif95_hi": np.percentile(d, 97.5), "n": n})
        print(f"{ds}: n = {n}, {len(escores)} modelos", flush=True)
    q = pd.DataFrame(linhas)
    q.to_csv(R / "final_table_ci.csv", index=False)
    with pd.option_context("display.width", 220):
        print(q.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
