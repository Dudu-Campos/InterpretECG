"""Tabela final do PTB-XL (fold 10): AUROC macro com IC 95% por bootstrap de registros.

Para cada modelo, a estatistica e a media das AUROCs macro das 5 sementes. O IC
vem de 2.000 reamostragens dos registros do fold 10, as MESMAS para todos os
modelos (semente 42), o que tambem da as diferencas pareadas contra o nosso.
Saida: results/confirmatory/tabela_ptbxl_ic.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
S = range(42, 47)


def sem(seed42, padrao):
    return [seed42] + [padrao.format(s=s) for s in S if s != 42]


MODELOS = {  # nome: (arquivos, chave dos escores)
    "catch22 + boosting": ([R / f"confirmatory/seeds/boosting_catch22_seed{s}_scores.npz" for s in S], "test"),
    "tsfresh + boosting": ([R / f"confirmatory/seeds/boosting_tsfresh_seed{s}_scores.npz" for s in S], "test"),
    "nosso boosting": ([R / f"confirmatory/seeds/boosting_melhor_multiplex_seed{s}_scores.npz" for s in S], "test"),
    "xresnet1d101": (sem(R / "cnn_xresnet1d101/test_scores.npz", str(R / "cnn_xresnet1d101_seed{s}/test_scores.npz")), "escores"),
    "inception1d": ([R / f"deep_inception1d_confirmatory_seed{s}/test_scores.npz" for s in S], "escores"),
    "LSTM bidirecional": ([R / f"deep_lstm_bidir_confirmatory_seed{s}/test_scores.npz" for s in S], "escores"),
    "Transformer": ([R / f"deep_transformer_confirmatory_seed{s}/test_scores.npz" for s in S], "escores"),
    "ECG-IMN": (sem(R / "imn_exploratory/multilabel5_confirmatory_test_scores.npz",
                    str(R / "imn_exploratory/multilabel5_confirmatory_seed{s}_test_scores.npz")), "scores"),
    "GNN L(HVG) direcionada": ([R / f"line_gnn_confirmatory/hvg_dir_d64_l4_e50_conf_seed{s}_test_scores.npz" for s in S], "scores"),
}
# os arquivos de semente do boosting nao guardam ecg_id: a ordem e a das execucoes originais
ORDEM_BOOSTING = {"catch22 + boosting": "boosting_catch22", "tsfresh + boosting": "boosting_tsfresh",
                  "nosso boosting": "boosting_melhor_multiplex"}


def auc(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def main() -> None:
    lab = pd.read_csv(ROOT / "export/ecg_fm_bundle/ptbxl/labels.csv").set_index("ecg_id")
    rot = ["NORM", "MI", "STTC", "CD", "HYP"]
    ids_ref = None
    escores = {}
    for nome, (arqs, chave) in MODELOS.items():
        lista = []
        for a in arqs:
            z = np.load(a)
            if nome in ORDEM_BOOSTING:
                ids = np.load(R / f"confirmatory/{ORDEM_BOOSTING[nome]}_scores.npz")["ecg_id_test"]
            else:
                ids = z["ecg_id"]
            ids_ref = np.sort(ids) if ids_ref is None else ids_ref
            assert np.array_equal(np.sort(ids), ids_ref), (nome, a)
            lista.append(pd.DataFrame(z[chave], index=ids).loc[ids_ref].to_numpy())
        escores[nome] = lista
    assert set(lab.loc[ids_ref, "strat_fold"]) == {10}
    y = lab.loc[ids_ref, rot].to_numpy()

    rng = np.random.default_rng(42)
    n, boot = len(y), {m: [] for m in escores}
    while len(boot["nosso boosting"]) < 2000:
        i = rng.integers(0, n, n)
        if (y[i].sum(0) == 0).any() or (y[i].sum(0) == n).any():
            continue
        for m, ss in escores.items():
            boot[m].append(np.mean([auc(y[i], s[i]) for s in ss]))
    boot = {m: np.array(v) for m, v in boot.items()}
    linhas = []
    for m, ss in escores.items():
        pontual = np.mean([auc(y, s) for s in ss])
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        d = boot[m] - boot["nosso boosting"]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        linhas.append({"modelo": m, "auroc": pontual, "ic_lo": lo, "ic_hi": hi, "sementes": len(ss),
                       "dif_vs_nosso": pontual - np.mean([auc(y, s) for s in escores["nosso boosting"]]),
                       "dif_ic_lo": dlo, "dif_ic_hi": dhi})
    q = pd.DataFrame(linhas)
    q.to_csv(R / "confirmatory/tabela_ptbxl_ic.csv", index=False)
    print(f"n = {n} registros do fold 10")
    for r in q.itertuples():
        print(f"{r.modelo:24s} {r.auroc:.4f} [{r.ic_lo:.4f}, {r.ic_hi:.4f}]   "
              f"vs nosso {r.dif_vs_nosso:+.4f} [{r.dif_ic_lo:+.4f}, {r.dif_ic_hi:+.4f}]")


if __name__ == "__main__":
    main()
