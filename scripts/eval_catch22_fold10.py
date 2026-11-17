"""catch22 (264 features, 12 derivacoes) + boosting de configuracao fixa, no fold 10.

Concorrente sem selecao: todas as features do catch22, configuracao `BOOSTING`
de evaluate_augmented.py, max_iter 2000, treino 1-8, early stopping no fold 9,
teste no fold 10 lido uma vez por dataset. Nenhuma feature, hiperparametro ou
k e escolhido. Decisao do Eduardo em 13/nov/2026 (tabela final). Cada execucao
e registrada em results/test_fold_usage.jsonl.

Saida: results/confirmatory/boosting_catch22{.json,_scores.npz} (PTB-XL) e
results/external/<dataset>/boosting_catch22{.json,_scores.npz}.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import BoostingPorRotulo  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

MOTIVO = ("catch22 + boosting de configuracao fixa, 264 features sem selecao, early stopping no fold 9, "
          "fold 10 nao usado para selecao; decisao do Eduardo 13/nov/2026 (tabela final)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["ptbxl", "chapman", "chapman8", "code15"])
    ap.add_argument("--threads", type=int, default=16)
    args = ap.parse_args()
    if args.dataset == "ptbxl":
        reg = PTBXLDataLoader(ROOT / "data/ptb-xl").records.set_index("ecg_id")
        rotulos, feats = list(SUPERCLASSES), ROOT / "data/features/catch22_12leads.parquet"
        out = ROOT / "results/confirmatory"
    else:
        reg = pd.read_csv(ROOT / "data/external" / args.dataset / "labels.csv").set_index("ecg_id")
        rotulos = json.load(open(ROOT / "data/external" / args.dataset / "prep.json"))["labels"]
        feats = ROOT / "data/features/external" / args.dataset / "catch22.parquet"
        out = ROOT / "results/external" / args.dataset
    destino = out / "boosting_catch22.json"
    if destino.exists():
        sys.exit(f"{destino} ja existe: o fold 10 deste dataset ja foi lido para o catch22")
    x_df = pd.read_parquet(feats).set_index("ecg_id")
    assert x_df.shape[1] == 264, x_df.shape
    fold = reg["strat_fold"]
    ids = {"train": fold.index[fold.between(1, 8)], "validation": fold.index[fold.eq(9)],
           "test": fold.index[fold.eq(10)]}
    x = {n: np.array(x_df.loc[v].to_numpy(np.float32), copy=True) for n, v in ids.items()}
    for a in x.values():
        a[np.isinf(a)] = np.nan
    y = {n: reg.loc[v, rotulos].to_numpy(np.int8) for n, v in ids.items()}
    with threadpool_limits(args.threads):
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        esc = {n: m.predict_proba(x[n]) for n in ("validation", "test")}
    auc = {n: float(np.mean([roc_auc_score(y[n][:, i], esc[n][:, i]) for i in range(len(rotulos))]))
           for n in esc}
    res = {"dataset": args.dataset, "conjunto": "catch22", "k": 264, "labels": rotulos,
           "split_sizes": {n: len(v) for n, v in ids.items()},
           "validation_macro_auroc": auc["validation"], "test_macro_auroc": auc["test"],
           "test_per_label": {r: float(roc_auc_score(y["test"][:, i], esc["test"][:, i]))
                              for i, r in enumerate(rotulos)},
           "iterations_per_label": [int(mm.n_iter_) for mm in m.modelos],
           "boosting": BOOSTING, "seed": 42, "finished": datetime.now(timezone.utc).isoformat()}
    out.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(res, indent=2) + "\n")
    np.savez(out / "boosting_catch22_scores.npz", test=esc["test"], ecg_id_test=ids["test"].to_numpy(),
             y_test=y["test"])
    with (ROOT / "results/test_fold_usage.jsonl").open("a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "reason": MOTIVO, "test_fold": 10, "model": f"boosting catch22 [{args.dataset}]",
                            "dataset": args.dataset, "feature_count": 264,
                            "test_macro_auroc": auc["test"], "output_dir": str(out)}) + "\n")
    print(f"{args.dataset}: val {auc['validation']:.4f}  teste {auc['test']:.4f}")


if __name__ == "__main__":
    main()
