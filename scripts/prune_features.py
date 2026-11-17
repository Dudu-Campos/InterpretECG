"""Poda recursiva por metades de um conjunto de familias (boosting fixo).

Padrao: `base + direcionado`. Com --families escolhe-se outro ponto de partida
(ex. `tsfresh`, para a curva so-serie) e com --schedule fixam-se os k de cada
passo, para que curvas de conjuntos diferentes caiam na mesma grade de
dimensoes (1.075, 537, 268, 134, 67, 33 -- a grade de `base + direcionado`).

A cada passo: ajusta o boosting fixo (o mesmo de evaluate_augmented.py) no
treino (folds 1-7) com early stopping no fold 8, mede a importancia por
permutacao de cada coluna no fold 8 (queda do AUROC macro, media de R
repeticoes), descarta a metade menos importante e repete.

Por metades e nao por um ranking unico: com 12 derivacoes redundantes, a
permutacao de uma coluna isolada subestima features que tem copias; reajustar a
cada passo deixa a importancia migrar para a copia que sobra.

O ranking usa so o fold 8 (que tambem faz o early stopping), entao a validacao
dos conjuntos podados e otimista. A leitura honesta e o fold 9. O fold 10 nao e lido.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_augmented import BOOSTING, FAMILIAS  # noqa: E402
from evaluate_views import PTBXL, ROOT, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402


def queda(modelo, x, y, base, j, repeticoes, semente):
    rng = np.random.default_rng([semente, j])
    quedas = []
    for _ in range(repeticoes):
        xp = x.copy()
        xp[:, j] = rng.permutation(xp[:, j])
        quedas.append(base - roc_auc_score(y, modelo.predict_proba(xp), average="macro"))
    return float(np.mean(quedas))


def queda_bloco(modelo, x, y, base, js, repeticoes, semente):
    """O mesmo que `queda` para varias colunas, com uma copia so e 1 thread.

    Mesma semente por coluna ([semente, j]) e mesma permutacao: resultado
    identico a `queda`. Sem o limite de threads, cada worker abre um OpenMP
    com todos os nucleos e 12 workers sobrecarregam a maquina ~12x.
    """
    from threadpoolctl import threadpool_limits
    xp = np.array(x, copy=True)
    saida = []
    with threadpool_limits(1):
        for j in js:
            rng = np.random.default_rng([semente, j])
            original = xp[:, j].copy()
            quedas = []
            for _ in range(repeticoes):
                xp[:, j] = rng.permutation(original)
                quedas.append(base - roc_auc_score(y, modelo.predict_proba(xp), average="macro"))
            xp[:, j] = original
            saida.append(float(np.mean(quedas)))
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "results/pruning_exploratory")
    ap.add_argument("--min-features", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", nargs="+", default=FAMILIAS)
    ap.add_argument("--schedule", nargs="+", type=int, default=None,
                    help="k de cada passo depois do primeiro (decrescente); sem isto, metades")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    quadros = carregar_quadros()
    if "tsfresh" in args.families:
        import pandas as pd
        quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, "exploratory")
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    rotulos = registros.set_index("ecg_id")[list(SUPERCLASSES)]

    nomes, x = [], {n: [] for n in ids}
    for familia in args.families:
        treino = quadros[familia].loc[ids["train"]]
        cols = treino.columns[treino.nunique(dropna=True) > 1]
        nomes += [f"{familia}::{c}" for c in cols]
        for n in ids:
            x[n].append(quadros[familia].loc[ids[n], cols].to_numpy(np.float32))
    x = {n: np.hstack(v) for n, v in x.items()}
    y = {n: rotulos.loc[ids[n]].to_numpy(np.int8) for n in ids}
    nomes = np.array(nomes)

    ativas = np.arange(len(nomes))
    passos = []
    while True:
        inicio = time.perf_counter()
        modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
            x["train"][:, ativas], y["train"], x["validation"][:, ativas], y["validation"])
        escores = {n: modelo.predict_proba(x[n][:, ativas]) for n in ("validation", "test")}
        auc = {n: float(roc_auc_score(y[n], escores[n], average="macro")) for n in escores}
        passo = {"n_features": int(len(ativas)), "validation_macro_auroc": auc["validation"],
                 "test_macro_auroc": auc["test"],
                 "test_per_label": {c: float(roc_auc_score(y["test"][:, i], escores["test"][:, i]))
                                    for i, c in enumerate(SUPERCLASSES)},
                 "features": nomes[ativas].tolist()}
        np.savez(args.output / f"k{len(ativas)}_scores.npz", **escores,
                 ecg_id_validation=ids["validation"].to_numpy(), ecg_id_test=ids["test"].to_numpy(),
                 y_validation=y["validation"], y_test=y["test"])
        ajuste = time.perf_counter() - inicio
        print(f"{len(ativas):5d} features  val {auc['validation']:.4f}  teste {auc['test']:.4f}  "
              f"(ajuste {ajuste / 60:.1f} min)", flush=True)
        proximo = (len(ativas) // 2 if args.schedule is None
                   else next((k for k in args.schedule if k < len(ativas)), 0))
        if proximo < args.min_features:
            passo["importance"] = None
            passos.append(passo)
            break
        xv = x["validation"][:, ativas]
        blocos = np.array_split(np.arange(len(ativas)), max(1, min(len(ativas), 8 * args.n_jobs)))
        imp = np.concatenate(Parallel(n_jobs=args.n_jobs, verbose=10)(
            delayed(queda_bloco)(modelo, xv, y["validation"], auc["validation"], b, args.repeats, args.seed)
            for b in blocos))
        passo["importance"] = dict(zip(nomes[ativas].tolist(), imp.tolist()))
        passos.append(passo)
        print(f"      permutacao {(time.perf_counter() - inicio - ajuste) / 60:.1f} min", flush=True)
        ordem = np.argsort(-imp, kind="stable")
        ativas = ativas[np.sort(ordem[:proximo])]
        (args.output / "steps.json").write_text(json.dumps(passos, indent=1) + "\n")

    (args.output / "steps.json").write_text(json.dumps(passos, indent=1) + "\n")
    (args.output / "config.json").write_text(json.dumps({
        "families": args.families, "schedule": args.schedule, "repeats": args.repeats,
        "seed": args.seed, "boosting": BOOSTING, "regime": "exploratory (1-7 / 8 / 9)",
        "ranking": "importancia por permutacao no fold 8, reajuste a cada passo"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
