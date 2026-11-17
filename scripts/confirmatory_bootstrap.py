"""Bootstraps pareados (principal: AUROC media por semente; secundario: ensemble de sementes) pre-registrados (results/confirmatory/PREREGISTRATION.md) no fold 10.

Modelos com varias sementes: media dos escores das sementes antes da AUROC
(ensemble de sementes) e, separadamente, AUROC media por semente. As
diferencas usam os mesmos indices de reamostragem para os dois lados.
Comparacoes ausentes (GNN ainda nao rodada) sao puladas.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
C = ROOT / "results/confirmatory"


def carregar(padrao, chave_id, chave_esc):
    fs = sorted(glob.glob(str(ROOT / padrao)))
    out = []
    for f in fs:
        d = np.load(f)
        out.append(pd.DataFrame(d[chave_esc], index=d[chave_id]))
    return out


MODELOS = {
    "melhor + multiplex": lambda: carregar("results/confirmatory/boosting_melhor_multiplex_scores.npz", "ecg_id_test", "test"),
    "tsfresh": lambda: carregar("results/confirmatory/boosting_tsfresh_scores.npz", "ecg_id_test", "test"),
    "LSTM bidirecional": lambda: carregar("results/deep_lstm_bidir_confirmatory_seed*/test_scores.npz", "ecg_id", "escores"),
    "inception1d": lambda: carregar("results/deep_inception1d_confirmatory_seed*/test_scores.npz", "ecg_id", "escores"),
    "xresnet1d101": lambda: carregar("results/cnn_xresnet1d101*/test_scores.npz", "ecg_id", "escores"),
    "GNN direcionada": lambda: carregar("results/line_gnn_confirmatory/hvg_dir_*_test_scores.npz", "ecg_id", "scores"),
    "GNN nao direcionada": lambda: carregar("results/line_gnn_confirmatory/hvg_d64_*_test_scores.npz", "ecg_id", "scores"),
}
PARES = [("melhor + multiplex", "tsfresh"), ("GNN direcionada", "xresnet1d101"),
         ("GNN direcionada", "GNN nao direcionada"), ("melhor + multiplex", "LSTM bidirecional"),
         ("melhor + multiplex", "inception1d")]


def main() -> None:
    y = None
    esc, por_semente, sementes = {}, {}, {}
    ref = np.load(C / "boosting_tsfresh_scores.npz")
    ids, y = ref["ecg_id_test"], ref["y_test"]
    for nome, f in MODELOS.items():
        partes = [p.loc[ids].to_numpy() for p in f() if set(ids) <= set(p.index)]
        if not partes:
            continue
        if nome == "xresnet1d101":
            partes = [p for p in partes]  # todas as sementes confirmatorias do xresnet
        esc[nome] = np.mean(partes, 0)
        sementes[nome] = partes
        por_semente[nome] = [roc_auc_score(y, p, average="macro") for p in partes]
    rng = np.random.default_rng(0)
    amostras = [a for a in (rng.integers(0, len(y), len(y)) for _ in range(1000)) if (y[a].sum(0) > 0).all()]
    linhas = []
    for a, b in PARES:
        if a not in esc or b not in esc:
            print(f"pulado: {a} − {b}")
            continue
        d = np.array([roc_auc_score(y[s], esc[a][s], average="macro") - roc_auc_score(y[s], esc[b][s], average="macro")
                      for s in amostras])
        obs = roc_auc_score(y, esc[a], average="macro") - roc_auc_score(y, esc[b], average="macro")
        def media(nome, s):
            return np.mean([roc_auc_score(y[s], p[s], average="macro") for p in sementes[nome]])
        dm = np.array([media(a, s) - media(b, s) for s in amostras])
        linhas.append({"a": a, "b": b,
                       "dif_media_sementes": np.mean(por_semente[a]) - np.mean(por_semente[b]),
                       "ic_lo_media": np.quantile(dm, .025), "ic_hi_media": np.quantile(dm, .975),
                       "dif_ensemble": obs, "ic_lo_ensemble": np.quantile(d, .025), "ic_hi_ensemble": np.quantile(d, .975),
                       "sementes_a": len(por_semente[a]), "sementes_b": len(por_semente[b]),
                       "auroc_media_sementes_a": np.mean(por_semente[a]), "auroc_media_sementes_b": np.mean(por_semente[b])})
    df = pd.DataFrame(linhas)
    df.to_csv(C / "bootstrap_pareado.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
