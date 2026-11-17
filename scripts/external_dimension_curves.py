"""Curvas erro x features nos datasets externos, no regime exploratorio (teste = fold 9).

Reproduz a figura do PTB-XL (plot_dimension_curves.py / plot_pareto_features.py)
num dataset externo: para cada passo da poda (prune_external.py) reajusta o
boosting fixo com treino nos folds 1-7 e parada no fold 8 -- exatamente como na
poda, entao a AUROC de validacao tem de bater com a gravada -- e preve o fold 9.
O fold 9 nao escolheu k nem features em nenhum lugar (no ajuste final ele so
faz early stopping). O fold 10 nao e lido. Ponto de referencia: catch22 sozinho.

Saida: results/external/<dataset>/curvas_fold9.csv e os escores por passo em
results/external/<dataset>/pruning_<conjunto>/k<K>_fold9_scores.npz.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import BoostingPorRotulo  # noqa: E402

CONJUNTOS = ("tsfresh", "melhor_multiplex", "todas_familias")


def auc_macro(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def ajusta(x, y, threads, destino, ids_teste):
    """Ajusta e grava na hora (uma queda no meio nao perde os passos prontos)."""
    with threadpool_limits(threads):
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        esc = {n: m.predict_proba(x[n]) for n in ("validation", "test")}
    np.savez(destino, validation=esc["validation"], test=esc["test"], ecg_id_test=ids_teste, y_test=y["test"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["chapman", "chapman8", "code15"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    out = ROOT / "results/external" / args.dataset
    prep = json.load(open(ROOT / "data/external" / args.dataset / "prep.json"))
    rotulos = prep["labels"]
    lab = pd.read_csv(ROOT / "data/external" / args.dataset / "labels.csv").set_index("ecg_id")
    ids = {"train": lab.index[lab.strat_fold.between(1, 7)].to_numpy(),
           "validation": lab.index[lab.strat_fold.eq(8)].to_numpy(),
           "test": lab.index[lab.strat_fold.eq(9)].to_numpy()}
    y = {n: lab.loc[v, rotulos].to_numpy(np.int8) for n, v in ids.items()}
    feats = ROOT / "data/features/external" / args.dataset
    cache = {}

    def matriz(nomes):
        partes = {}
        for f in nomes:
            fam, col = f.split("::", 1)
            partes.setdefault(fam, []).append(col)
        for fam in partes:
            if fam not in cache:
                cache[fam] = pd.read_parquet(feats / f"{fam}.parquet").set_index("ecg_id")
        x = {n: np.hstack([cache[fam].loc[ids[n], cs].to_numpy(np.float32) for fam, cs in partes.items()])
             for n in ids}
        for a in x.values():
            a[np.isinf(a)] = np.nan
        return x

    trabalhos = []  # (conjunto, k, val_gravada, arquivo, nomes)
    for c in CONJUNTOS:
        arq = out / f"pruning_{c}" / "steps.json"
        if not arq.exists():
            print(f"[sem poda {c}]")
            continue
        for p in json.load(open(arq)):
            destino = out / f"pruning_{c}" / f"k{p['n_features']}_fold9_scores.npz"
            trabalhos.append((c, p["n_features"], p["validation_macro_auroc"], destino, p["features"]))
    c22 = [f"catch22::{col}" for col in pd.read_parquet(feats / "catch22.parquet").columns if col != "ecg_id"]
    trabalhos.append(("catch22", len(c22), None, out / "catch22_fold9_scores.npz", c22))

    pendentes = [t for t in trabalhos if not t[3].exists()]
    print(f"{len(trabalhos)} passos, {len(pendentes)} a ajustar", flush=True)
    # matrizes montadas no processo principal (cache de familias); ajustes em paralelo
    lotes = [(t, matriz(t[4])) for t in pendentes]
    # os maiores primeiro, para o mais lento nao ficar sozinho no fim
    lotes.sort(key=lambda tx: -len(tx[0][4]))
    Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(ajusta)(x, y, args.threads, t[3], ids["test"]) for t, x in lotes)

    linhas = []
    for c, k, val_gravada, destino, _ in trabalhos:
        z = np.load(destino)
        val = auc_macro(y["validation"], z["validation"])
        if val_gravada is not None and abs(val - val_gravada) > 1e-6:
            print(f"AVISO {c} k={k}: validacao {val:.6f} != gravada {val_gravada:.6f}")
        linhas.append({"conjunto": c, "k": k, "val_fold8": val, "teste_fold9": auc_macro(y["test"], z["test"]),
                       "erro_fold9": 1 - auc_macro(y["test"], z["test"])})
    q = pd.DataFrame(linhas).sort_values(["conjunto", "k"])
    q.to_csv(out / "curvas_fold9.csv", index=False)
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
