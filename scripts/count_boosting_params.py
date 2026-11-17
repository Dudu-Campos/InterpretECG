"""Reajusta cada passo das podas e conta os parametros aprendidos do boosting.

parametros = limiares (um por no de divisao) + valores de folha, somados sobre
todas as arvores dos 5 classificadores (um por rotulo) -- os numeros reais
aprendidos, analogos aos pesos de uma rede. `parametros_com_indice` soma
tambem o indice de feature de cada divisao. O reajuste usa o mesmo
BoostingPorRotulo, a mesma semente e as mesmas features de cada passo; a
AUROC de teste e conferida contra a registrada (deve bater).
Saida: results/pareto/boosting_params.csv (uma linha por passo, append).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import PTBXL, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SAIDA = ROOT / "results/pareto/boosting_params.csv"
CONJ = {"tsfresh (só série)": "serie", "melhor + multiplex": "multiview"}


def contar(modelo):
    folhas = divisoes = arvores = 0
    for m in modelo.modelos:
        for it in m._predictors:
            for t in it:
                f = int(t.nodes["is_leaf"].sum())
                folhas += f
                divisoes += len(t.nodes) - f
                arvores += 1
    return folhas, divisoes, arvores


def main() -> None:
    caminhos = json.load(open(ROOT / "results/heatmap_dimension_data/caminhos.json"))
    registrado = {}
    for nome, d in (("tsfresh (só série)", "results/pruning_tsfresh_exploratory"),
                    ("melhor + multiplex", "results/pruning_bestmx_exploratory")):
        for p in json.load(open(ROOT / d / "steps.json")):
            caminhos[CONJ[nome]].setdefault(str(p["n_features"]), p["features"])
            registrado[(nome, p["n_features"])] = p["test_macro_auroc"]
    grade = pd.read_csv(ROOT / "results/heatmap_dimension_data/celulas.csv")
    for nome, c in CONJ.items():
        for k in (8, 16):
            registrado[(nome, k)] = 1 - grade[(grade.conjunto == c) & (grade.k == k) & (grade.fracao == 1)].erro.iloc[0]

    quadros = carregar_quadros()
    quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    comuns = None
    fams = {f.split("::")[0] for c in caminhos.values() for fs in c.values() for f in fs} | {"catch22"}
    for fam in sorted(fams):
        comuns = quadros[fam].index if comuns is None else comuns.intersection(quadros[fam].index)
    reg = PTBXLDataLoader(PTBXL).records
    reg = reg.loc[reg.ecg_id.isin(comuns)].reset_index(drop=True)
    masc = PTBXLDataLoader.split_masks(reg, "exploratory")
    ids = {n: reg.loc[m, "ecg_id"].to_numpy() for n, m in masc.items()}
    rot = reg.set_index("ecg_id")[list(SUPERCLASSES)]
    y = {n: rot.loc[ids[n]].to_numpy(np.int8) for n in ids}

    feito = set()
    if SAIDA.exists():
        feito = {(r.modelo, r.k) for r in pd.read_csv(SAIDA).itertuples()}
    tarefas = sorted(((nome, int(k), fs) for nome, c in CONJ.items() for k, fs in caminhos[c].items()),
                     key=lambda t: t[1])
    for nome, k, fs in tarefas:
        if (nome, k) in feito:
            continue
        partes = {}
        for f in fs:
            fam, col = f.split("::", 1)
            partes.setdefault(fam, []).append(col)
        x = {n: np.hstack([quadros[fam].loc[ids[n], cs].to_numpy(np.float32) for fam, cs in partes.items()])
             for n in ids}
        t0 = time.perf_counter()
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        auc = roc_auc_score(y["test"], m.predict_proba(x["test"]), average="macro")
        folhas, divisoes, arvores = contar(m)
        linha = {"modelo": nome, "k": k, "auroc": auc, "auroc_registrada": registrado.get((nome, k)),
                 "folhas": folhas, "divisoes": divisoes, "arvores": arvores,
                 "parametros": folhas + divisoes, "parametros_com_indice": folhas + 2 * divisoes,
                 "minutos": (time.perf_counter() - t0) / 60}
        pd.DataFrame([linha]).to_csv(SAIDA, mode="a", header=not SAIDA.exists(), index=False)
        print(f"{nome:20s} k={k:5d} auc {auc:.4f} (reg {linha['auroc_registrada']:.4f})  "
              f"{arvores} arvores  {linha['parametros']:,} parametros  {linha['minutos']:.1f} min", flush=True)


if __name__ == "__main__":
    main()
