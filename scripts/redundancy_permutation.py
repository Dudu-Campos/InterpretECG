"""Teste de permutacao: o agrupamento das familias organiza a redundancia entre elas?

Le a redundancia normalizada por par (`red_val`, `red_test` de
shapley_interaction.py --scale d2) e, para cada agrupamento dos jogadores,
calcula a razao media(intra) / media(inter). A distribuicao nula embaralha os
rotulos de grupo entre os jogadores, preservando o tamanho de cada grupo.
p unilateral: fracao das permutacoes com razao >= observada.

Agrupamentos:
    formato      serie / imagem / grafo (a taxonomia da literatura)
    invariancia  o grupo medido por measure_invariance.py

O jogador placebo, se houver, fica fora dos agrupamentos: v({placebo}) ~ 0 e a
normalizacao explode. Sua interacao crua e reportada a parte, como o custo de
estimacao puro.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

FORMATO = {
    "catch22": "serie", "amplitude+inter": "serie", "amplitude": "serie", "interderivacao": "serie",
    "espectro+escalograma": "imagem", "gaf": "imagem", "mtf": "imagem", "rqa": "imagem",
    "visibilidade": "grafo", "takens": "grafo", "transicao": "grafo",
    "line graph mV": "grafo", "direcionado mV": "grafo",
}
# Medido em results/invariance (x -> 2x, x + 0,5, x + 0,4x^3).
INVARIANCIA = {
    "visibilidade": "monotonica", "mtf": "monotonica", "transicao": "monotonica",
    "catch22": "afim", "gaf": "afim", "rqa": "afim",
    "espectro+escalograma": "escala",
    "takens": "afim",
    "amplitude+inter": "mV",   # misto (amplitude mV + interderivacao afim): ver lgsep
    "amplitude": "mV", "interderivacao": "afim",
    "line graph mV": "mV", "direcionado mV": "mV",
}


def razao(red: np.ndarray, rotulos: np.ndarray, i: np.ndarray, j: np.ndarray) -> float:
    mesmo = rotulos[i] == rotulos[j]
    return red[mesmo].mean() / red[~mesmo].mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "results/shapley_lda_lg_exploratory")
    ap.add_argument("--permutations", type=int, default=20000)
    ap.add_argument("--override", nargs="*", default=[],
                    help="jogador=grupo para reatribuir na invariancia (teste de sensibilidade)")
    args = ap.parse_args()

    pares = pd.read_csv(args.input / "interacao_d2_pares.csv")
    placebo = pares[(pares.i == "placebo") | (pares.j == "placebo")]
    pares = pares.drop(placebo.index)
    jogadores = sorted(set(pares.i) | set(pares.j))
    pos = {p: k for k, p in enumerate(jogadores)}
    i, j = pares.i.map(pos).to_numpy(), pares.j.map(pos).to_numpy()
    invariancia = dict(INVARIANCIA)
    for o in args.override:
        nome, grupo = o.split("=")
        invariancia[nome] = grupo

    rng = np.random.default_rng(42)
    saida = {"input": str(args.input), "permutations": args.permutations, "override": args.override}
    for nome, mapa in (("formato", FORMATO), ("invariancia", invariancia)):
        rotulos = np.array([mapa[p] for p in jogadores])
        for split in ("val", "test"):
            red = pares[f"red_{split}"].to_numpy()
            obs = razao(red, rotulos, i, j)
            nulos = np.array([razao(red, rng.permutation(rotulos), i, j)
                              for _ in range(args.permutations)])
            p = (1 + (nulos >= obs).sum()) / (1 + len(nulos))
            grupos = {g: round(float(red[(rotulos[i] == g) & (rotulos[j] == g)].mean()), 4)
                      for g in np.unique(rotulos) if ((rotulos[i] == g) & (rotulos[j] == g)).any()}
            saida[f"{nome}_{split}"] = {"razao": obs, "p": p, "nulo_p95": float(np.quantile(nulos, 0.95)),
                                        "intra_por_grupo": grupos,
                                        "inter": float(red[rotulos[i] != rotulos[j]].mean())}
            print(f"{nome:12s} {split:4s} razao {obs:.3f}  p = {p:.4f}  (nulo p95 {np.quantile(nulos, .95):.3f})  "
                  f"intra {grupos}  inter {red[rotulos[i] != rotulos[j]].mean():.3f}")
    if len(placebo):
        cru = placebo.assign(outro=np.where(placebo.i == "placebo", placebo.j, placebo.i))
        saida["placebo_I"] = cru.set_index("outro")[["I_val", "I_test"]].to_dict()
        reais = pares[["I_val", "I_test"]]
        print(f"placebo: I cru medio {cru.I_val.mean():+.4f} (val), contra {reais.I_val.mean():+.4f} "
              f"entre jogadores reais; min/max placebo {cru.I_val.min():+.4f} / {cru.I_val.max():+.4f}")
    sufixo = "_" + "_".join(o.replace("=", "-").replace(" ", "") for o in args.override) if args.override else ""
    (args.input / f"permutacao{sufixo}.json").write_text(json.dumps(saida, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
