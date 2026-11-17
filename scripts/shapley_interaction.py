"""Indice de interacao de Shapley entre pares (Grabisch & Roubens, 1999), exato.

Le as coalizoes ja avaliadas por shapley_views.py (nenhum modelo novo):

    I(i,j) = sum_{S subset N\\{i,j}} |S|! (n-|S|-2)! / (n-1)!
             * [v(S+i+j) - v(S+i) - v(S+j) + v(S)]

I < 0: substitutos (redundancia, o ganho de um some na presenca do outro).
I > 0: complementares (sinergia). E a medida de pares que a H1 pede.

Escala: o AUROC satura em 1, entao dois jogadores fortes "colidem" mesmo com
informacao independente e I sai negativo para todo par. Com --scale d2 o valor
vira d^2 = 2 * Phi^-1(AUROC)^2, a distancia de Mahalanobis do modelo gaussiano
de variancia igual -- exatamente o que a LDA estima, e ADITIVA para features
independentes. Nessa escala I = 0 e o null de independencia; I < 0 e
redundancia de verdade. Aplicado ao AUROC macro, e aproximacao (a media dos
rotulos nao e o rotulo medio).
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "results/shapley_lda_lg_exploratory")
    ap.add_argument("--scale", choices=["auroc", "d2"], default="auroc")
    args = ap.parse_args()
    from scipy.stats import norm
    transformar = (lambda a: a) if args.scale == "auroc" else (lambda a: 2 * norm.ppf(a) ** 2)
    import json
    jogadores = list(json.loads((args.input / "shapley.json").read_text())["players"])
    n = len(jogadores)
    c = pd.read_csv(args.input / "coalizoes.csv", keep_default_na=False)
    indice = {j: i for i, j in enumerate(jogadores)}
    # Nomes de jogador podem conter '+' (amplitude+inter): casar pelos nomes,
    # na ordem em que shapley_views.py os escreve.
    def dividir(texto: str) -> list[int]:
        partes = []
        for k, nome in enumerate(jogadores):
            if texto == nome or texto.startswith(nome + "+"):
                partes.append(k)
                texto = texto[len(nome) + 1:]
        assert texto == "", texto
        return partes

    valores = {}
    for linha in c.itertuples():
        chave = frozenset(dividir(linha.coalizao))
        valores[chave] = {"validation": transformar(linha.validation), "test": transformar(linha.test)}
    assert len(valores) == 2 ** n

    saida = {}
    for split in ("validation", "test"):
        m = np.zeros((n, n))
        for i, j in itertools.combinations(range(n), 2):
            resto = [k for k in range(n) if k not in (i, j)]
            total = 0.0
            for t in range(n - 1):
                peso = math.factorial(t) * math.factorial(n - t - 2) / math.factorial(n - 1)
                for s in itertools.combinations(resto, t):
                    s = frozenset(s)
                    total += peso * (valores[s | {i, j}][split] - valores[s | {i}][split]
                                     - valores[s | {j}][split] + valores[s][split])
            m[i, j] = m[j, i] = total
        saida[split] = pd.DataFrame(m, index=jogadores, columns=jogadores)
        saida[split].to_csv(args.input / f"interacao_{args.scale}_{split}.csv")

    # Redundancia normalizada: -I(i,j) / min(v({i}), v({j})), a fracao do
    # jogador mais fraco que o outro ja cobre. Torna pares de forca diferente
    # comparaveis; a magnitude crua de I cresce com a forca dos dois.
    sozinho = {s: [valores[frozenset({k})][s] for k in range(n)] for s in ("validation", "test")}
    pares = pd.DataFrame([{"i": jogadores[i], "j": jogadores[j],
                           "I_val": saida["validation"].iat[i, j], "I_test": saida["test"].iat[i, j],
                           "red_val": -saida["validation"].iat[i, j]
                                      / min(sozinho["validation"][i], sozinho["validation"][j]),
                           "red_test": -saida["test"].iat[i, j]
                                       / min(sozinho["test"][i], sozinho["test"][j])}
                          for i, j in itertools.combinations(range(n), 2)]).sort_values("I_val")
    pares.to_csv(args.input / f"interacao_{args.scale}_pares.csv", index=False)
    pd.set_option("display.width", 200)
    print(pares.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))


if __name__ == "__main__":
    main()
