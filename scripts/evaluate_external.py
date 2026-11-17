"""Generalizacao: listas congeladas do PTB-XL num dataset externo (results/external/PREREGISTRATION.md).

Treino folds 1-8, early stopping no fold 9, teste no fold 10 do dataset externo.
Dois conjuntos (tsfresh k=268, melhor + multiplex k=537), boosting de
configuracao fixa, e bootstrap pareado mv - tsfresh no teste.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from count_boosting_params import contar  # noqa: E402
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import BoostingPorRotulo  # noqa: E402

SHA = {"tsfresh": "3c7c0d8d63fbf093", "melhor_multiplex": "74b3396bd53d03dc"}


def auc_macro(y, s):
    return float(np.mean([roc_auc_score(y[:, i], s[:, i]) for i in range(y.shape[1])]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("dataset", choices=["chapman", "chapman8", "code15"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--lists", choices=["frozen", "pruned"], default="frozen",
                    help="frozen: listas do PTB-XL; pruned: poda no proprio dataset (prune_external.py)")
    ap.add_argument("--sets", nargs="+", default=["tsfresh", "melhor_multiplex"],
                    help="(pruned) conjuntos a ajustar; os ja ajustados sao lidos dos _scores.npz salvos")
    args = ap.parse_args()
    out = ROOT / "results/external" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if args.lists == "frozen":
        listas = json.load(open(ROOT / "results/confirmatory/frozen_feature_lists.json"))
        for nome, v in listas.items():
            h = hashlib.sha256("\n".join(v["features"]).encode()).hexdigest()
            assert h.startswith(SHA[nome]) and h == v["sha256_lista"], f"lista {nome} alterada"
        sufixo = ""
    else:
        listas = {n: json.load(open(out / f"pruning_{n}" / "selected.json")) for n in args.sets}
        for nome, v in listas.items():
            h = hashlib.sha256("\n".join(v["features"]).encode()).hexdigest()
            assert h == v["sha256_lista"], f"lista podada {nome} alterada"
        sufixo = "_pruned"

    prep = json.load(open(ROOT / "data/external" / args.dataset / "prep.json"))
    rotulos = prep["labels"]
    lab = pd.read_csv(ROOT / "data/external" / args.dataset / "labels.csv").set_index("ecg_id")
    feats = ROOT / "data/features/external" / args.dataset
    ids = {"train": lab.index[lab.strat_fold.between(1, 8)].to_numpy(),
           "validation": lab.index[lab.strat_fold.eq(9)].to_numpy(),
           "test": lab.index[lab.strat_fold.eq(10)].to_numpy()}
    y = {n: lab.loc[v, rotulos].to_numpy(np.int8) for n, v in ids.items()}
    print({n: len(v) for n, v in ids.items()}, {r: int(y["test"][:, i].sum()) for i, r in enumerate(rotulos)},
          flush=True)

    escores, resultados = {}, {}
    for nome, v in listas.items():
        partes = {}
        for f in v["features"]:
            fam, col = f.split("::", 1)
            partes.setdefault(fam, []).append(col)
        quadros = {fam: pd.read_parquet(feats / f"{fam}.parquet", columns=["ecg_id", *cs]).set_index("ecg_id")
                   for fam, cs in partes.items()}
        x = {n: np.hstack([quadros[fam].loc[ids[n], cs].to_numpy(np.float32) for fam, cs in partes.items()])
             for n in ids}
        n_inf = int(sum(np.isinf(a).sum() for a in x.values()))
        for a in x.values():
            a[np.isinf(a)] = np.nan
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        esc = {n: m.predict_proba(x[n]) for n in ("validation", "test")}
        folhas, divisoes, arvores = contar(m)
        res = {"dataset": args.dataset, "lists": args.lists, "conjunto": nome, "k": v["k"], "labels": rotulos,
               "split_sizes": {n: int(len(a)) for n, a in ids.items()},
               "validation_macro_auroc": auc_macro(y["validation"], esc["validation"]),
               "test_macro_auroc": auc_macro(y["test"], esc["test"]),
               "test_per_label": {r: float(roc_auc_score(y["test"][:, i], esc["test"][:, i]))
                                  for i, r in enumerate(rotulos)},
               "iterations_per_label": [int(mm.n_iter_) for mm in m.modelos],
               "parametros": folhas + divisoes, "inf_replaced_by_nan": n_inf,
               "sha256_lista": v["sha256_lista"], "boosting": BOOSTING, "seed": 42,
               "finished": datetime.now(timezone.utc).isoformat()}
        (out / f"boosting_{nome}{sufixo}.json").write_text(json.dumps(res, indent=2) + "\n")
        np.savez(out / f"boosting_{nome}{sufixo}_scores.npz", test=esc["test"], ecg_id_test=ids["test"], y_test=y["test"])
        escores[nome], resultados[nome] = esc["test"], res
        print(f"{nome:18s} k={v['k']}  val {res['validation_macro_auroc']:.4f}  teste {res['test_macro_auroc']:.4f}  "
              + " ".join(f"{r} {a:.3f}" for r, a in res["test_per_label"].items()), flush=True)

    # conjuntos ja ajustados antes (mesmo teste): le os escores salvos, nao reajusta
    for nome in ("tsfresh", "melhor_multiplex", "todas_familias"):
        arq = out / f"boosting_{nome}{sufixo}_scores.npz"
        if nome not in escores and arq.exists():
            z = np.load(arq)
            assert (z["ecg_id_test"] == ids["test"]).all()
            escores[nome] = z["test"]
            resultados[nome] = json.load(open(out / f"boosting_{nome}{sufixo}.json"))
    yt, b = y["test"], escores["tsfresh"]
    resumo = {"dataset": args.dataset, "boot": args.boot,
              "test_macro_auroc": {k: r["test_macro_auroc"] for k, r in resultados.items()}, "comparisons": {}}
    for nome in [k for k in ("melhor_multiplex", "todas_familias") if k in escores]:
        rng = np.random.default_rng(42)  # mesma sequencia de reamostragens para cada comparacao
        a = escores[nome]
        obs = auc_macro(yt, a) - auc_macro(yt, b)
        dif = []
        n = len(yt)
        while len(dif) < args.boot:
            i = rng.integers(0, n, n)
            if (yt[i].sum(0) == 0).any() or (yt[i].sum(0) == n).any():
                continue
            dif.append(auc_macro(yt[i], a[i]) - auc_macro(yt[i], b[i]))
        lo, hi = np.percentile(dif, [2.5, 97.5])
        resumo["comparisons"][f"{nome} - tsfresh"] = {
            "diff": obs, "ci95": [float(lo), float(hi)], "p_le_0": float(np.mean(np.array(dif) <= 0)),
            "per_label_diff": {r: resultados[nome]["test_per_label"][r] - resultados["tsfresh"]["test_per_label"][r]
                               for r in rotulos}}
        if nome == "melhor_multiplex":  # chaves antigas, para compatibilidade
            resumo |= {"diff_mv_minus_tsfresh": obs, "ci95": [float(lo), float(hi)],
                       "p_le_0": resumo["comparisons"][f"{nome} - tsfresh"]["p_le_0"]}
        print(f"{nome} - tsfresh: {obs:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    (out / f"bootstrap{sufixo}.json").write_text(json.dumps(resumo, indent=2) + "\n")


if __name__ == "__main__":
    main()
