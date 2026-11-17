"""tsfresh como catch22 de alta dimensao: boosting fixo, regime exploratorio (1-7 / 8 / 9).

Mesmo boosting de evaluate_augmented.py. Colunas constantes no treino saem.
Conjuntos:

    tsfresh                             a familia sozinha, contra catch22 (0,8927 / 0,8768)
    tsfresh + amplitude + interderivacao  a `base` com tsfresh no lugar do catch22
    base(tsfresh) + direcionado         o candidato podado com tsfresh no lugar do catch22
    base + direcionado + tsfresh        o candidato podado com tsfresh somado

O fold 10 nao e lido.
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
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import PTBXL, ROOT, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

DIRECIONADO = ["graph_hvg_dir", "graph_lphvg1_dir", "graph_hvg_dir_mv", "graph_lphvg1_dir_mv"]
CONJUNTOS = {
    "tsfresh": ["tsfresh"],
    "tsfresh + amplitude + interderivacao": ["tsfresh", "series_amplitude", "series_interlead"],
    "base(tsfresh) + direcionado": ["tsfresh", "series_amplitude", "series_interlead"] + DIRECIONADO,
    "base + direcionado + tsfresh": ["catch22", "series_amplitude", "series_interlead"]
                                    + DIRECIONADO + ["tsfresh"],
}


def slug(nome: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in nome).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(CONJUNTOS))
    ap.add_argument("--output", type=Path, default=ROOT / "results/tsfresh_exploratory")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    quadros = carregar_quadros()
    quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, "exploratory")
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    rotulos = registros.set_index("ecg_id")[list(SUPERCLASSES)]
    y = {n: rotulos.loc[ids[n]].to_numpy(np.int8) for n in ids}
    print(f"registros: treino {len(ids['train']):,} / val {len(ids['validation']):,} / "
          f"teste {len(ids['test']):,}", flush=True)

    for nome in args.sets:
        x = {n: [] for n in ids}
        for familia in CONJUNTOS[nome]:
            treino = quadros[familia].loc[ids["train"]]
            cols = treino.columns[treino.nunique(dropna=True) > 1]
            for n in ids:
                x[n].append(quadros[familia].loc[ids[n], cols].to_numpy(np.float32))
        x = {n: np.hstack(v) for n, v in x.items()}
        x = {n: np.where(np.isfinite(v), v, np.nan) for n, v in x.items()}
        inicio = time.perf_counter()
        modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
            x["train"], y["train"], x["validation"], y["validation"])
        escores = {n: modelo.predict_proba(x[n]) for n in ("validation", "test")}
        resultado = {
            "set": nome, "families": CONJUNTOS[nome], "n_features": int(x["train"].shape[1]),
            "regime": "exploratory (folds 1-7 / 8 / 9)", "boosting": BOOSTING,
            **{f"{n}_macro_auroc": float(roc_auc_score(y[n], escores[n], average="macro"))
               for n in escores},
            "test_per_label": {c: float(roc_auc_score(y["test"][:, i], escores["test"][:, i]))
                               for i, c in enumerate(SUPERCLASSES)},
            "minutes": round((time.perf_counter() - inicio) / 60, 1),
        }
        (args.output / f"{slug(nome)}.json").write_text(json.dumps(resultado, indent=2) + "\n")
        np.savez(args.output / f"{slug(nome)}_scores.npz", **escores,
                 ecg_id_validation=ids["validation"].to_numpy(), ecg_id_test=ids["test"].to_numpy(),
                 y_validation=y["validation"], y_test=y["test"])
        print(f"{nome:40s} {resultado['n_features']:6d} feats  val {resultado['validation_macro_auroc']:.4f}  "
              f"teste {resultado['test_macro_auroc']:.4f}  ({resultado['minutes']} min)  "
              + " ".join(f"{c} {v:.3f}" for c, v in resultado["test_per_label"].items()), flush=True)


if __name__ == "__main__":
    main()
