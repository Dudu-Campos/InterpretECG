"""Poda dependente da tarefa num dataset externo (desvio de 11/nov, results/external/PREREGISTRATION.md).

Repete a poda do PTB-XL (scripts/prune_features.py) no proprio dataset:
treino folds 1-7, boosting fixo com early stopping no fold 8, importancia por
permutacao no fold 8, reajuste a cada passo, mesma grade de k. O fold 9 so faz
early stopping no ajuste final e o fold 10 nao e lido aqui.

Saida: results/external/<dataset>/pruning_<conjunto>/steps.json e a lista
escolhida (maior AUROC no fold 8) em selected.json, com SHA-256.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import BoostingPorRotulo  # noqa: E402
from prune_features import queda_bloco  # noqa: E402

CONJUNTOS = {
    "melhor_multiplex": (["catch22", "series_amplitude", "series_interlead", "graph_hvg", "graph_lphvg1",
                          "graph_hvg_line", "graph_lphvg1_line", "graph_hvg_line_mv", "graph_lphvg1_line_mv",
                          "graph_hvg_dir", "graph_lphvg1_dir", "graph_hvg_dir_mv", "graph_lphvg1_dir_mv",
                          "graph_hvg_multiplex", "graph_lphvg1_multiplex"],
                         [2150, 1075, 537, 268, 134, 67, 33]),
    "tsfresh": (["tsfresh"], [4449, 2150, 1075, 537, 268, 134, 67, 33]),
}
# todas as familias: as 15 acima + as 8 descartadas no PTB-XL (desvio de 11/nov)
CONJUNTOS["todas_familias"] = (CONJUNTOS["melhor_multiplex"][0]
                               + ["image_spectrogram", "image_scalogram", "image_gasf", "image_gadf",
                                  "image_mtf", "image_recurrence", "graph_takens", "graph_transition"],
                               [4449, 2150, 1075, 537, 268, 134, 67, 33])


def auc_macro(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["chapman", "chapman8", "code15"])
    ap.add_argument("conjunto", choices=list(CONJUNTOS))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    familias, grade = CONJUNTOS[args.conjunto]
    out = ROOT / "results/external" / args.dataset / f"pruning_{args.conjunto}"
    out.mkdir(parents=True, exist_ok=True)

    prep = json.load(open(ROOT / "data/external" / args.dataset / "prep.json"))
    rotulos = prep["labels"]
    lab = pd.read_csv(ROOT / "data/external" / args.dataset / "labels.csv").set_index("ecg_id")
    ids = {"train": lab.index[lab.strat_fold.between(1, 7)].to_numpy(),
           "validation": lab.index[lab.strat_fold.eq(8)].to_numpy()}
    y = {n: lab.loc[v, rotulos].to_numpy(np.int8) for n, v in ids.items()}
    feats = ROOT / "data/features/external" / args.dataset

    nomes, x = [], {n: [] for n in ids}
    for fam in familias:
        q = pd.read_parquet(feats / f"{fam}.parquet").set_index("ecg_id")
        treino = q.loc[ids["train"]]
        cols = treino.columns[treino.nunique(dropna=True) > 1]
        nomes += [f"{fam}::{c}" for c in cols]
        for n in ids:
            a = np.array(q.loc[ids[n], cols].to_numpy(np.float32), copy=True)
            a[np.isinf(a)] = np.nan
            x[n].append(a)
        del q, treino
    x = {n: np.hstack(v) for n, v in x.items()}
    nomes = np.array(nomes)
    print({n: a.shape for n, a in x.items()}, flush=True)

    ativas = np.arange(len(nomes))
    passos = []
    while True:
        inicio = time.perf_counter()
        modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
            x["train"][:, ativas], y["train"], x["validation"][:, ativas], y["validation"])
        ev = modelo.predict_proba(x["validation"][:, ativas])
        auc = auc_macro(y["validation"], ev)
        # roc_auc_score(average="macro") dentro de queda_bloco: igual a auc_macro sem rotulo degenerado
        passo = {"n_features": int(len(ativas)), "validation_macro_auroc": auc, "features": nomes[ativas].tolist()}
        print(f"{len(ativas):5d} features  val(fold 8) {auc:.4f}  ({(time.perf_counter() - inicio) / 60:.1f} min)",
              flush=True)
        proximo = next((k for k in grade if k < len(ativas)), 0)
        if proximo == 0:
            passos.append(passo)
            break
        xv = x["validation"][:, ativas]
        blocos = np.array_split(np.arange(len(ativas)), max(1, min(len(ativas), 8 * args.n_jobs)))
        imp = np.concatenate(Parallel(n_jobs=args.n_jobs, verbose=0)(
            delayed(queda_bloco)(modelo, xv, y["validation"], auc, b, args.repeats, args.seed) for b in blocos))
        passo["importance"] = dict(zip(nomes[ativas].tolist(), imp.tolist()))
        passos.append(passo)
        ordem = np.argsort(-imp, kind="stable")
        ativas = ativas[np.sort(ordem[:proximo])]
        (out / "steps.json").write_text(json.dumps(passos, indent=1) + "\n")
        print(f"      permutacao {(time.perf_counter() - inicio) / 60:.1f} min", flush=True)

    (out / "steps.json").write_text(json.dumps(passos, indent=1) + "\n")
    melhor = max(passos, key=lambda p: p["validation_macro_auroc"])
    sha = hashlib.sha256("\n".join(melhor["features"]).encode()).hexdigest()
    (out / "selected.json").write_text(json.dumps({
        "dataset": args.dataset, "conjunto": args.conjunto, "k": melhor["n_features"],
        "validation_macro_auroc_fold8": melhor["validation_macro_auroc"], "sha256_lista": sha,
        "criterio": "maior AUROC macro no fold 8 (treino 1-7)", "families": familias, "schedule": grade,
        "repeats": args.repeats, "seed": args.seed, "boosting": BOOSTING,
        "features": melhor["features"]}, indent=1) + "\n")
    print(f"escolhido k={melhor['n_features']} (val {melhor['validation_macro_auroc']:.4f}) sha {sha[:16]}", flush=True)


if __name__ == "__main__":
    main()
