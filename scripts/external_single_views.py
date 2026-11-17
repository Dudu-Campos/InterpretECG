"""Cada familia sozinha nos datasets externos: qual e a melhor visao por tarefa?

Mesmo criterio que escolheu L(HVG) mV no PTB-XL: boosting de configuracao
fixa (BOOSTING), treino folds 1-7, early stopping no fold 8, AUROC macro no
fold 8 para escolher; o fold 9 e reportado (teste exploratorio). O fold 10 nao
e lido. A melhor visao por dataset decide em que representacao roda o modelo
profundo "da melhor visao" na tabela final.

Saida: results/external/<dataset>/single_views.csv
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
from prune_external import CONJUNTOS  # noqa: E402


def auc(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def uma(ds, fam, threads):
    lab = pd.read_csv(ROOT / "data/external" / ds / "labels.csv").set_index("ecg_id")
    rot = json.load(open(ROOT / "data/external" / ds / "prep.json"))["labels"]
    f = lab["strat_fold"]
    ids = {"train": f.index[f.between(1, 7)], "validation": f.index[f.eq(8)], "test": f.index[f.eq(9)]}
    q = pd.read_parquet(ROOT / "data/features/external" / ds / f"{fam}.parquet").set_index("ecg_id")
    x = {n: np.array(q.loc[v].to_numpy(np.float32), copy=True) for n, v in ids.items()}
    for a in x.values():
        a[np.isinf(a)] = np.nan
    y = {n: lab.loc[v, rot].to_numpy(np.int8) for n, v in ids.items()}
    with threadpool_limits(threads):
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        return {"familia": fam, "n_features": x["train"].shape[1],
                "val_fold8": auc(y["validation"], m.predict_proba(x["validation"])),
                "teste_fold9": auc(y["test"], m.predict_proba(x["test"]))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["chapman8", "code15"])
    ap.add_argument("--n-jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    pasta = ROOT / "data/features/external" / args.dataset
    fams = [f for f in CONJUNTOS["todas_familias"][0] if (pasta / f"{f}.parquet").exists()]
    faltam = sorted(set(CONJUNTOS["todas_familias"][0]) - set(fams))
    if faltam:
        print("familias ainda nao extraidas:", faltam)
    linhas = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(uma)(args.dataset, f, args.threads) for f in fams)
    q = pd.DataFrame(linhas).sort_values("val_fold8", ascending=False)
    q.to_csv(ROOT / "results/external" / args.dataset / "single_views.csv", index=False)
    print(q.to_string(index=False))


if __name__ == "__main__":
    main()
