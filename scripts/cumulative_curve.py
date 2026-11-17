"""Curva cumulativa: onde a adição de famílias para de ajudar.

Duas perguntas que a avaliação por conjunto não responde.

1. A curva satura? Famílias são ordenadas por AUROC de validação isolada e
   acrescentadas uma a uma. Se a redundância medida no grafo r² prediz o
   desfecho, a curva deve achatar antes das 11, porque três famílias são
   quase-duplicatas de outra (GASF↔GADF, HVG↔LPHVG, MTF↔transição).
2. A diferença é maior que o ruído? Comparações usam bootstrap PAREADO sobre
   os registros do teste — os mesmos registros reamostrados para os dois
   modelos, o que remove a variância comum e mede a diferença diretamente.
   Bootstrap não-pareado sobre AUROCs separados superestima o intervalo.

A ordenação usa apenas a validação; o teste é medido uma vez por ponto da curva.
Um único classificador em toda a curva, para que a forma não misture efeito de
representação com efeito de modelo.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import pathlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

# O mount NFS de /exp trava com leituras concorrentes destes parquets (estado
# `D` em nfs_set_open_stateid_locked, sem recuperacao quando o servidor volta).
# INTERPRETECG_FEATURES aponta para uma copia em tmpfs; os resultados continuam
# sendo escritos no projeto, que sao arquivos pequenos e nunca travaram.
FEATURES = pathlib.Path(os.environ.get("INTERPRETECG_FEATURES", str(ROOT / "data/features")))
PTBXL = pathlib.Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))


sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SEED = 42
N_BOOT = 2000


def modelo() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", OneVsRestClassifier(HistGradientBoostingClassifier(
            learning_rate=0.05, max_leaf_nodes=63, max_iter=400,
            early_stopping=True, validation_fraction=0.1,
            random_state=SEED), n_jobs=1))])


def macro_auroc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s, average="macro"))


def bootstrap_pareado(y, a, b, rng, n=N_BOOT):
    """IC 95% da diferença macro-AUROC(a) - macro-AUROC(b), mesmos reamostres."""
    diffs = np.empty(n)
    indices = np.arange(len(y))
    for i in range(n):
        amostra = rng.choice(indices, size=len(indices), replace=True)
        ys = y[amostra]
        if (ys.sum(axis=0) == 0).any() or (ys.sum(axis=0) == len(ys)).any():
            diffs[i] = np.nan
            continue
        diffs[i] = macro_auroc(ys, a[amostra]) - macro_auroc(ys, b[amostra])
    diffs = diffs[np.isfinite(diffs)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), diffs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--output", type=Path, default=ROOT / "results/cumulative_curve")
    args = ap.parse_args()

    loader = PTBXLDataLoader(PTBXL)
    quadros = {"catch22": pd.read_parquet(
        FEATURES / "catch22_12leads.parquet").set_index("ecg_id")}
    for caminho in sorted((FEATURES / "views").glob("*.parquet")):
        quadros[caminho.stem] = pd.read_parquet(caminho).set_index("ecg_id")

    comuns = None
    for q in quadros.values():
        comuns = q.index if comuns is None else comuns.intersection(q.index)
    registros = loader.records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, args.regime)
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    alvos = {n: registros.loc[m, list(SUPERCLASSES)].to_numpy(np.int8)
             for n, m in mascaras.items()}

    uteis = {}
    for nome, q in quadros.items():
        treino = q.loc[ids["train"]]
        uteis[nome] = treino.columns[treino.nunique(dropna=True) > 1]

    def blocos(familias):
        return {n: np.hstack([quadros[f].loc[ids[n], uteis[f]].to_numpy(np.float32)
                              for f in familias]) for n in ids}

    print("1) AUROC de validacao de cada familia isolada (define a ordem)\n")
    isoladas = {}
    escores_catch22 = None
    for nome in quadros:
        dados = blocos([nome])
        m = modelo()
        m.fit(dados["train"], alvos["train"])
        val = macro_auroc(alvos["validation"], m.predict_proba(dados["validation"]))
        isoladas[nome] = val
        if nome == "catch22":
            escores_catch22 = m.predict_proba(dados["test"])
        print(f"   {nome:20s} {len(uteis[nome]):5d}f  val {val:.4f}")

    ordem = sorted(isoladas, key=isoladas.get, reverse=True)
    print(f"\n2) curva cumulativa nessa ordem: {' > '.join(ordem)}\n")

    rng = np.random.default_rng(SEED)
    linhas, anterior = [], None
    for k in range(1, len(ordem) + 1):
        familias = ordem[:k]
        dados = blocos(familias)
        m = modelo()
        m.fit(dados["train"], alvos["train"])
        val = macro_auroc(alvos["validation"], m.predict_proba(dados["validation"]))
        escores = m.predict_proba(dados["test"])
        teste = macro_auroc(alvos["test"], escores)

        lo_c, hi_c, _ = bootstrap_pareado(alvos["test"], escores, escores_catch22, rng)
        ganho_passo = None
        if anterior is not None:
            lo_p, hi_p, _ = bootstrap_pareado(alvos["test"], escores, anterior, rng)
            ganho_passo = (teste - macro_auroc(alvos["test"], anterior), lo_p, hi_p)
        anterior = escores

        linhas.append(dict(k=k, adicionada=familias[-1],
                           n_features=sum(len(uteis[f]) for f in familias),
                           val_auroc=val, test_auroc=teste,
                           vs_catch22=teste - macro_auroc(alvos["test"], escores_catch22),
                           vs_catch22_lo=lo_c, vs_catch22_hi=hi_c,
                           passo=None if ganho_passo is None else ganho_passo[0],
                           passo_lo=None if ganho_passo is None else ganho_passo[1],
                           passo_hi=None if ganho_passo is None else ganho_passo[2]))
        marca = "" if ganho_passo is None else (
            "  <- passo significativo" if ganho_passo[1] > 0 else "  (passo nao significativo)")
        print(f"   k={k:2d} +{familias[-1]:20s} teste {teste:.4f}  "
              f"vs catch22 {linhas[-1]['vs_catch22']:+.4f} "
              f"[{lo_c:+.4f}, {hi_c:+.4f}]{marca}")

    tabela = pd.DataFrame(linhas)
    args.output.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(args.output / "curve.csv", index=False, float_format="%.5f")
    (args.output / "isoladas.json").write_text(json.dumps(
        {"validation_macro_auroc": isoladas, "ordem": ordem,
         "n_bootstrap": N_BOOT, "regime": args.regime}, indent=2) + "\n")

    significativos = tabela.loc[tabela["passo_lo"].notna() & (tabela["passo_lo"] > 0), "k"]
    print(f"\n{'='*86}")
    print(f"ultimo passo com ganho significativo: k={int(significativos.max())} "
          f"de {len(ordem)}" if len(significativos) else "nenhum passo significativo")
    print(f"-> {args.output}/")


if __name__ == "__main__":
    main()
