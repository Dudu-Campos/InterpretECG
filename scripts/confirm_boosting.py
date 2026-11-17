"""Rodada confirmatoria do boosting (results/confirmatory/PREREGISTRATION.md, secao 1).

Listas de features congeladas (frozen_feature_lists.json, hash conferido),
treino folds 1-8, early stopping no fold 9, teste no fold 10. Uma execucao
por conjunto; cada uma registrada em results/test_fold_usage.jsonl.
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
from evaluate_views import PTBXL, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SHA = {"tsfresh": "3c7c0d8d63fbf093", "melhor_multiplex": "74b3396bd53d03dc"}
OUT = ROOT / "results/confirmatory"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--confirm-test-fold", required=True, metavar="JUSTIFICATIVA")
    args = ap.parse_args()
    listas = json.load(open(OUT / "frozen_feature_lists.json"))
    for nome, v in listas.items():
        h = hashlib.sha256("\n".join(v["features"]).encode()).hexdigest()
        assert h.startswith(SHA[nome]) and h == v["sha256_lista"], f"lista {nome} alterada"

    quadros = carregar_quadros()
    quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    # mesma intersecao de registros da exploracao (todas as familias da poda)
    fams = {f.split("::")[0] for v in listas.values() for f in v["features"]} | {"catch22"}
    comuns = None
    for fam in sorted(fams):
        comuns = quadros[fam].index if comuns is None else comuns.intersection(quadros[fam].index)
    reg = PTBXLDataLoader(PTBXL).records
    reg = reg.loc[reg.ecg_id.isin(comuns)].reset_index(drop=True)
    masc = PTBXLDataLoader.split_masks(reg, "confirmatory")
    ids = {n: reg.loc[m, "ecg_id"].to_numpy() for n, m in masc.items()}
    print({n: len(v) for n, v in ids.items()}, flush=True)
    rot = reg.set_index("ecg_id")[list(SUPERCLASSES)]
    y = {n: rot.loc[ids[n]].to_numpy(np.int8) for n in ids}

    for nome, v in listas.items():
        partes = {}
        for f in v["features"]:
            fam, col = f.split("::", 1)
            partes.setdefault(fam, []).append(col)
        x = {n: np.hstack([quadros[fam].loc[ids[n], cs].to_numpy(np.float32) for fam, cs in partes.items()])
             for n in ids}
        m = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(x["train"], y["train"], x["validation"], y["validation"])
        esc = {n: m.predict_proba(x[n]) for n in ("validation", "test")}
        auc = {n: float(roc_auc_score(y[n], esc[n], average="macro")) for n in esc}
        folhas, divisoes, arvores = contar(m)
        res = {"conjunto": nome, "k": v["k"], "regime": "confirmatory: treino 1-8, validacao 9, teste 10",
               "validation_macro_auroc": auc["validation"], "test_macro_auroc": auc["test"],
               "test_per_label": {c: float(roc_auc_score(y["test"][:, i], esc["test"][:, i]))
                                  for i, c in enumerate(SUPERCLASSES)},
               "parametros": folhas + divisoes, "parametros_com_indice": folhas + 2 * divisoes, "arvores": arvores,
               "sha256_lista": v["sha256_lista"], "boosting": BOOSTING,
               "finished": datetime.now(timezone.utc).isoformat()}
        (OUT / f"boosting_{nome}.json").write_text(json.dumps(res, indent=2) + "\n")
        np.savez(OUT / f"boosting_{nome}_scores.npz", test=esc["test"], ecg_id_test=ids["test"], y_test=y["test"])
        with (ROOT / "results/test_fold_usage.jsonl").open("a") as uso:
            uso.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "reason": f"{args.confirm_test_fold} [{nome}, k={v['k']}]", "test_fold": 10,
                                  "model": f"boosting {nome} k={v['k']} (lista congelada)",
                                  "feature_count": v["k"], "test_macro_auroc": auc["test"],
                                  "output_dir": str(OUT)}) + "\n")
        print(f"{nome:18s} k={v['k']}  val {auc['validation']:.4f}  teste(fold 10) {auc['test']:.4f}  "
              f"{res['parametros']:,} parametros", flush=True)


if __name__ == "__main__":
    main()
