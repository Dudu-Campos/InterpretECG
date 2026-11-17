"""Boosting fixo no conjunto podado `base + direcionado`, com e sem as copias aumentadas.

Treino = folds 1-7 originais (+ as copias de `extract_augmented_features.py`);
early stopping no fold 8 ORIGINAL; teste no fold 9 ORIGINAL. As copias de um
registro herdam o seu rotulo e o seu fold, entao nada vaza entre folds.

`--tag none` roda o controle sem aumento, que deve reproduzir o resultado de
`evaluate_views.py` para `base + direcionado` (val 0,9302, teste 0,9169).
Os escores de validacao e teste sao salvos para o bootstrap pareado.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_views import PTBXL, ROOT, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

BOOSTING = {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 80,
            "l2_regularization": 0.0, "max_features": 0.5}
FAMILIAS = ["catch22", "series_amplitude", "series_interlead", "graph_hvg_dir",
            "graph_lphvg1_dir", "graph_hvg_dir_mv", "graph_lphvg1_dir_mv"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="k2_gain10_wander10_noise1")
    ap.add_argument("--output", type=Path, default=ROOT / "results/augmentation_exploratory")
    ap.add_argument("--seed", type=int, default=42, help="random_state do boosting (42 = original)")
    args = ap.parse_args()
    nome = args.tag if args.seed == 42 else f"{args.tag}_seed{args.seed}"

    quadros = carregar_quadros()
    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, "exploratory")
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    rotulos = registros.set_index("ecg_id")[list(SUPERCLASSES)]

    colunas = {}
    x = {n: [] for n in ids}
    for familia in FAMILIAS:
        treino = quadros[familia].loc[ids["train"]]
        colunas[familia] = treino.columns[treino.nunique(dropna=True) > 1]
        for n in ids:
            x[n].append(quadros[familia].loc[ids[n], colunas[familia]].to_numpy(np.float32))
    x = {n: np.hstack(v) for n, v in x.items()}
    y = {n: rotulos.loc[ids[n]].to_numpy(np.int8) for n in ids}

    n_extra = 0
    if args.tag != "none":
        pasta = ROOT / "data/features/augmented" / args.tag
        partes, chave = [], None
        for familia in FAMILIAS:
            q = pd.read_parquet(pasta / f"{familia}.parquet")
            k = q[["ecg_id", "copia"]].to_numpy()
            assert chave is None or np.array_equal(chave, k), "familias fora de ordem"
            chave = k
            partes.append(q[colunas[familia]].to_numpy(np.float32))
        assert set(chave[:, 0]) <= set(ids["train"]), "copia fora do treino"
        x["train"] = np.vstack([x["train"], np.hstack(partes)])
        y["train"] = np.vstack([y["train"], rotulos.loc[chave[:, 0]].to_numpy(np.int8)])
        n_extra = len(chave)
    print(f"{args.tag}: treino {len(x['train']):,} linhas ({n_extra:,} aumentadas), "
          f"{x['train'].shape[1]} features", flush=True)

    inicio = time.perf_counter()
    modelo = BoostingPorRotulo(max_iter=2000, semente=args.seed, **BOOSTING).fit(
        x["train"], y["train"], x["validation"], y["validation"])
    escores = {n: modelo.predict_proba(x[n]) for n in ("validation", "test")}
    resultado = {
        "tag": args.tag, "seed": args.seed, "families": FAMILIAS, "augmented_rows": n_extra,
        "regime": "exploratory (folds 1-7 + copias / 8 / 9)", "boosting": BOOSTING,
        **{f"{n}_macro_auroc": float(roc_auc_score(y[n], escores[n], average="macro"))
           for n in escores},
        "test_per_label": {c: float(roc_auc_score(y["test"][:, i], escores["test"][:, i]))
                           for i, c in enumerate(SUPERCLASSES)},
        "minutes": round((time.perf_counter() - inicio) / 60, 1),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{nome}.json").write_text(json.dumps(resultado, indent=2) + "\n")
    np.savez(args.output / f"{nome}_scores.npz", **escores,
             ecg_id_validation=ids["validation"].to_numpy(), ecg_id_test=ids["test"].to_numpy(),
             y_validation=y["validation"], y_test=y["test"])
    print(f"val {resultado['validation_macro_auroc']:.4f}  teste {resultado['test_macro_auroc']:.4f}  "
          + " ".join(f"{c} {v:.4f}" for c, v in resultado["test_per_label"].items()))


if __name__ == "__main__":
    main()
