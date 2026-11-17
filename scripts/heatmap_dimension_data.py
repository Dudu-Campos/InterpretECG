"""Grade k (features) x n (registros de treino): log(erro so-serie / erro multi-view).

erro = 1 - AUROC macro no fold 9 (regime exploratorio, treino 1-7, parada e
selecao no fold 8). Celula > 0: o multi-view erra menos.

Selecao de features: o caminho da poda recursiva ja calculado com o treino
COMPLETO (prune_features.py; results/pruning_tsfresh_exploratory e
results/pruning_bestmx_exploratory), estendido aqui a k = 16 e 8 com o mesmo
procedimento (permutacao no fold 8, 2 repeticoes, semente 42). Na grade, n
varia so no reajuste: a selecao das celulas com n pequeno usou o treino
inteiro, o que as favorece igualmente nos dois conjuntos. Isso fica declarado.

Subamostragem por paciente (strat_fold ja e por paciente), aninhada: a amostra
de fracao f esta contida na de 2f para a mesma semente. Fracoes < 1 repetem
com 3 sementes; reporta-se a media do erro por celula.
"""
from __future__ import annotations

import argparse
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
from evaluate_views import PTBXL, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402
from prune_features import queda_bloco  # noqa: E402

CONJUNTOS = {"serie": "results/pruning_tsfresh_exploratory", "multiview": "results/pruning_bestmx_exploratory"}
KS = (8, 16, 33, 67, 134, 268, 537, 1075)
FRACOES = (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1)


def ajustar(xt, yt, xv, yv, xs, semente=42):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(1):
        m = BoostingPorRotulo(max_iter=2000, semente=semente, **BOOSTING).fit(xt, yt, xv, yv)
        return m, m.predict_proba(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--output", type=Path, default=ROOT / "results/heatmap_dimension_data")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    quadros = carregar_quadros()
    quadros["tsfresh"] = pd.read_parquet(ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    caminhos = {c: {p["n_features"]: p["features"] for p in json.load(open(ROOT / d / "steps.json"))}
                for c, d in CONJUNTOS.items()}
    familias = {f.split("::")[0] for c in caminhos.values() for f in c[1075]} | {"catch22"}
    comuns = None
    for fam in sorted(familias):
        comuns = quadros[fam].index if comuns is None else comuns.intersection(quadros[fam].index)
    reg = PTBXLDataLoader(PTBXL).records
    reg = reg.loc[reg.ecg_id.isin(comuns)].reset_index(drop=True)
    masc = PTBXLDataLoader.split_masks(reg, "exploratory")
    ids = {n: reg.loc[m, "ecg_id"].to_numpy() for n, m in masc.items()}
    pac = reg.loc[masc["train"], "patient_id"].to_numpy()
    rot = reg.set_index("ecg_id")[list(SUPERCLASSES)]
    y = {n: rot.loc[ids[n]].to_numpy(np.int8) for n in ids}

    def matriz(feats):
        partes = {}
        for f in feats:
            fam, c = f.split("::", 1)
            partes.setdefault(fam, []).append(c)
        ordem = [f"{fam}::{c}" for fam, cs in partes.items() for c in cs]
        xs = {n: np.hstack([quadros[fam].loc[ids[n], cs].to_numpy(np.float32) for fam, cs in partes.items()])
              for n in ids}
        pos = {f: i for i, f in enumerate(ordem)}
        return xs, pos

    # estender cada caminho a 16 e 8 com o treino completo
    for c, cam in caminhos.items():
        atual = cam[33]
        for alvo in (16, 8):
            xs, pos = matriz(atual)
            m, _ = ajustar(xs["train"], y["train"], xs["validation"], y["validation"], xs["test"])
            base = roc_auc_score(y["validation"], m.predict_proba(xs["validation"]), average="macro")
            imp = queda_bloco(m, xs["validation"], y["validation"], base, range(len(atual)), 2, 42)
            ordem = np.argsort(-np.array(imp), kind="stable")[:alvo]
            atual = [atual[i] for i in sorted(ordem)]
            cam[alvo] = atual
        print(c, "k16/k8:", cam[8], flush=True)
    json.dump({c: {k: cam[k] for k in KS} for c, cam in caminhos.items()},
              open(args.output / "caminhos.json", "w"), indent=1)

    # subamostras aninhadas por paciente
    unicos = np.unique(pac)
    subs = {}
    for s in range(args.seeds):
        perm = np.random.default_rng(1000 + s).permutation(unicos)
        for f in FRACOES:
            if f == 1 and s > 0:
                continue
            escolhidos = set(perm[:max(1, int(round(f * len(unicos))))])
            subs[(f, s)] = np.flatnonzero(np.isin(pac, list(escolhidos)))

    tarefas = []
    for c, cam in caminhos.items():
        xs, pos = matriz(cam[1075])
        for k in KS:
            col = [pos[f] for f in cam[k]]
            for (f, s), idx in subs.items():
                tarefas.append(((c, k, f, s), xs["train"][np.ix_(idx, col)], y["train"][idx],
                                xs["validation"][:, col], xs["test"][:, col]))
    tarefas.sort(key=lambda t: -t[1].size)  # grandes primeiro
    print(f"{len(tarefas)} ajustes", flush=True)
    inicio = time.perf_counter()

    def rodar(chave, xt, yt, xv, xs_):
        _, p = ajustar(xt, yt, xv, y["validation"], xs_)
        return chave, len(yt), 1 - roc_auc_score(y["test"], p, average="macro")

    res = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(rodar)(t[0], *t[1:]) for t in tarefas)
    df = pd.DataFrame([{"conjunto": c, "k": k, "fracao": f, "semente": s, "n": n, "erro": e}
                       for (c, k, f, s), n, e in res])
    df.to_csv(args.output / "celulas.csv", index=False)
    print(f"{(time.perf_counter() - inicio) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
