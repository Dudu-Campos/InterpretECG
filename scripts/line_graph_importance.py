"""Importancia por permutacao em grupo do line graph L(HVG), por classe.

Modelo: boosting de configuracao fixa (a do catch22) sobre `graph_hvg_line` +
`graph_hvg_line_mv` (49 features x 12 derivacoes), treino nos folds 1-7 com
early stopping no fold 8.

Importancia: queda de AUROC no fold 8 ao permutar um GRUPO de colunas, media
de R repeticoes. Permutar coluna a coluna dilui a importancia entre replicas
correlacionadas (a mesma feature em derivacoes vizinhas), por isso dois
agrupamentos:
    tipo        a mesma feature nas 12 derivacoes (49 grupos)
    derivacao   as 49 features de uma derivacao (12 grupos)
e o cruzamento tipo x derivacao so para as classes, quando pedido.

O fold 9 nao e usado: a analise e descritiva, sobre a validacao.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_views import PTBXL, ROOT, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import LEADS, SUPERCLASSES, PTBXLDataLoader  # noqa: E402

BOOSTING = {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 80,
            "l2_regularization": 0.0, "max_features": 0.5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="+", default=["graph_hvg_line", "graph_hvg_line_mv"])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--output", type=Path, default=ROOT / "results/line_graph_importance_exploratory")
    args = ap.parse_args()

    quadros = carregar_quadros()
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(quadros[args.families[0]].index)]
    mascaras = PTBXLDataLoader.split_masks(registros.reset_index(drop=True), "exploratory")
    registros = registros.reset_index(drop=True)
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items() if n != "test"}
    y = {n: registros.loc[mascaras[n], list(SUPERCLASSES)].to_numpy(np.int8) for n in ids}

    colunas, x = [], {n: [] for n in ids}
    for familia in args.families:
        quadro = quadros[familia]
        treino = quadro.loc[ids["train"]]
        uteis = treino.columns[treino.nunique(dropna=True) > 1]
        colunas += list(uteis)
        for n in ids:
            x[n].append(quadro.loc[ids[n], uteis].to_numpy(np.float32))
    x = {n: np.hstack(v) for n, v in x.items()}
    # coluna "V5__graph_hvg_line_mv__level_q10" -> derivacao V5, tipo "mv:level_q10"
    deriv = [c.split("__")[0] for c in colunas]
    tipo = [("mv:" if c.split("__")[1].endswith("_mv") else "ord:") + c.split("__")[2]
            for c in colunas]
    print(f"{len(colunas)} colunas, {len(set(tipo))} tipos; treino {len(x['train']):,}")

    modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
        x["train"], y["train"], x["validation"], y["validation"])
    xv, yv = x["validation"], y["validation"]
    imputado = modelo.imputador.transform(xv)

    def aurocs(matriz):
        p = np.column_stack([m.predict_proba(matriz)[:, 1] for m in modelo.modelos])
        return np.array([roc_auc_score(yv[:, i], p[:, i]) for i in range(len(SUPERCLASSES))])

    base = aurocs(imputado)
    print("AUROC val por classe:", dict(zip(SUPERCLASSES, base.round(4))),
          "macro", round(base.mean(), 4))
    rng = np.random.default_rng(0)

    def queda(indices):
        quedas = []
        for _ in range(args.repeats):
            permutado = imputado.copy()
            ordem = rng.permutation(len(permutado))
            permutado[:, indices] = imputado[ordem][:, indices]  # mesma permutacao no grupo
            quedas.append(base - aurocs(permutado))
        quedas = np.array(quedas)
        return quedas.mean(0), quedas.std(0)

    args.output.mkdir(parents=True, exist_ok=True)
    for nome, chave in (("tipo", tipo), ("derivacao", deriv)):
        linhas = []
        for g in sorted(set(chave)):
            idx = [k for k, c in enumerate(chave) if c == g]
            media, desvio = queda(idx)
            linhas.append({"grupo": g, "n_colunas": len(idx), "macro": media.mean(),
                           "macro_sd": desvio.mean(), **dict(zip(SUPERCLASSES, media))})
        tabela = pd.DataFrame(linhas).sort_values("macro", ascending=False)
        if nome == "derivacao":
            tabela["ordem"] = tabela["grupo"].map(LEADS.index)
        tabela.to_csv(args.output / f"importancia_{nome}.csv", index=False)
        print(f"\n== por {nome} (queda de AUROC no fold 8) ==")
        print(tabela.drop(columns=[c for c in ("ordem",) if c in tabela]).head(25).to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))
    pd.Series(dict(zip(SUPERCLASSES, base))).to_csv(args.output / "auroc_val.csv")


if __name__ == "__main__":
    main()
