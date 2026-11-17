"""Lista de features de uma tarefa aplicada a outra: a selecao transfere?

Motivacao (Eduardo, 17/nov): "mais views nao atrapalharem e trivial, porque o
boosting e bem resistente a features ruins". Procede — com 2.000 iteracoes e
127 folhas, acrescentar familias inuteis custa tempo, nao AUROC, e a corrida de
23 familias no PTB-XL confirmou (val 0,9330 contra 0,9333 das 15). Esse teste
nao separa nada.

O teste que separa e CRUZAR AS TAREFAS, com o mesmo k e o mesmo conjunto de
partida, mudando so onde a selecao foi feita:

    lista escolhida no PTB-XL, avaliada no PTB-XL   (consonante)
    lista escolhida no chapman8, avaliada no PTB-XL (transferida)

A figura dos datasets externos ja mostra a direcao oposta: no Chapman, a lista
do PTB-XL fica ATRAS de todas as familias e do tsfresh. Este script fecha o par.

Controle de placebo (secao 6.3): k features sorteadas das mesmas familias, 5
sementes. Se a lista transferida nao for melhor que o sorteio, a selecao de
outra tarefa nao vale mais que acaso.

Regime exploratorio: treino 1-7, early stopping no fold 8, teste no fold 9.
O fold 10 nao e lido. Boosting identico ao da poda (BOOSTING, max_iter=2000).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_augmented import BOOSTING  # noqa: E402
from evaluate_views import PTBXL, BoostingPorRotulo, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SEED = 42


def montar(familias):
    """Matriz do PTB-XL com as familias pedidas, e os nomes `familia::coluna`."""
    quadros = carregar_quadros()
    import pandas as pd
    if "tsfresh" in familias:
        quadros["tsfresh"] = pd.read_parquet(
            ROOT / "data/features/tsfresh_12leads.parquet").set_index("ecg_id")
    comuns = None
    for f in familias:
        idx = quadros[f].index
        comuns = idx if comuns is None else comuns.intersection(idx)
    reg = PTBXLDataLoader(PTBXL).records
    reg = reg.loc[reg["ecg_id"].isin(comuns)].reset_index(drop=True)
    masc = PTBXLDataLoader.split_masks(reg, "exploratory")
    ids = {n: reg.loc[m, "ecg_id"] for n, m in masc.items()}
    rot = reg.set_index("ecg_id")[list(SUPERCLASSES)]
    nomes, x = [], {n: [] for n in ids}
    for familia in familias:
        treino = quadros[familia].loc[ids["train"]]
        cols = treino.columns[treino.nunique(dropna=True) > 1]
        nomes += [f"{familia}::{c}" for c in cols]
        for n in ids:
            x[n].append(quadros[familia].loc[ids[n], cols].to_numpy(np.float32))
    return ({n: np.hstack(v) for n, v in x.items()},
            {n: rot.loc[ids[n]].to_numpy(np.int8) for n in ids},
            np.array(nomes))


def avaliar(x, y, cols, rotulo):
    t0 = time.perf_counter()
    modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
        x["train"][:, cols], y["train"], x["validation"][:, cols], y["validation"])
    esc = {n: modelo.predict_proba(x[n][:, cols]) for n in ("validation", "test")}
    auc = {n: float(roc_auc_score(y[n], esc[n], average="macro")) for n in esc}
    print(f"  {rotulo:46s} k={len(cols):4d}  val {auc['validation']:.4f}  "
          f"teste {auc['test']:.4f}  ({(time.perf_counter() - t0) / 60:.1f} min)", flush=True)
    return {"rotulo": rotulo, "k": int(len(cols)), **{f"{n}_macro_auroc": v for n, v in auc.items()},
            "test_per_label": {c: float(roc_auc_score(y["test"][:, i], esc["test"][:, i]))
                               for i, c in enumerate(SUPERCLASSES)}}


def main() -> None:
    conj = json.load(open(ROOT / "results/pruning_bestmx_exploratory/config.json"))["families"]
    x, y, nomes = montar(conj)
    pos = {n: i for i, n in enumerate(nomes)}
    print(f"PTB-XL: {len(nomes):,} features das {len(conj)} familias de "
          f"`melhor + multiplex`; treino {len(x['train']):,}\n")

    # lista escolhida no chapman8 a partir do MESMO conjunto de familias
    cha = json.load(open(ROOT / "results/external/chapman8/pruning_melhor_multiplex/selected.json"))
    cha = cha if isinstance(cha, list) else cha["features"]
    faltam = [f for f in cha if f not in pos]
    print(f"lista do chapman8: {len(cha)} features, {len(faltam)} sem par no PTB-XL")
    if faltam:
        print(f"  ausentes (exemplos): {faltam[:3]}")
    idx_cha = np.array([pos[f] for f in cha if f in pos])
    k = len(idx_cha)

    # lista do PTB-XL no mesmo k, para a comparacao ser so de "onde se escolheu"
    passos = json.load(open(ROOT / "results/pruning_bestmx_exploratory/steps.json"))
    passo = min(passos, key=lambda p: abs(p["n_features"] - k))
    idx_ptb = np.array([pos[f] for f in passo["features"] if f in pos])
    print(f"lista do PTB-XL mais proxima: k={passo['n_features']} "
          f"(val registrada {passo['validation_macro_auroc']:.4f})\n")

    linhas = [avaliar(x, y, idx_ptb, "selecao no PTB-XL (consonante)"),
              avaliar(x, y, idx_cha, "selecao no chapman8 (transferida)")]
    rng = np.random.default_rng(SEED)
    for s in range(5):
        linhas.append(avaliar(x, y, rng.choice(len(nomes), size=k, replace=False),
                              f"placebo: {k} sorteadas, semente {s}"))

    saida = ROOT / "results/transferencia_listas"
    saida.mkdir(parents=True, exist_ok=True)
    (saida / "exploratory.json").write_text(json.dumps(
        {"regime": "exploratory (1-7 / 8 / 9); fold 10 nao lido", "k": k,
         "familias": conj, "resultados": linhas}, indent=2, ensure_ascii=False) + "\n")
    pl = [r["validation_macro_auroc"] for r in linhas if r["rotulo"].startswith("placebo")]
    print(f"\n  placebo: val {np.mean(pl):.4f} +- {np.std(pl):.4f}")
    print(f"  -> {saida}/exploratory.json")


if __name__ == "__main__":
    main()
