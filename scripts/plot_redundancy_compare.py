"""O mesmo layout de redundância colorido duas vezes: por formato e por invariância.

Um único layout por força do grafo de r² (o limiar vem do cotovelo de
percolação gravado em results/redundancy_graph_v2/summary.json). O layout nunca
vê rótulo nenhum; muda só a cor. Painel esquerdo: formato (série, imagem,
grafo). Painel direito: grupo de invariância medido (results/invariance).
Embaixo, um painel por grupo de invariância com só ele destacado, para que
nenhuma leitura dependa de distinguir cores.

Paleta: slots 1-3 de `references/palette.md` (dataviz), o trio aprovado em
`--pairs all`. O quarto grupo (só escala) usa o slot 4 com marcador
triangular como codificação secundária, porque o slot 4 não é validado contra
todos os pares. O validador não roda aqui (sem node).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_redundancy_graph import CINZA_NO, DESTAQUE, MUDO, SEED, TINTA, arestas  # noqa: E402
from redundancy_graph import carregar, matriz_r2  # noqa: E402

FORMATO = {"serie": ("série", "#2a78d6", "o"), "imagem": ("imagem", "#eb6834", "o"),
           "grafo": ("grafo", "#1baf7a", "o")}
INVARIANCIA = {"monotonica": ("monotônica", "#2a78d6", "o"), "afim": ("afim", "#eb6834", "o"),
               "mV": ("lê mV", "#1baf7a", "o"), "escala": ("só escala", "#eda100", "^")}
FAMILIAS_DO_GRUPO = {
    "monotonica": "HVG, LPHVG, line graph e direcionado ordinais,\nmultiplex, MTF, transição",
    "afim": "catch22, GASF, GADF, RQA,\nTakens, interderivação",
    "mV": "amplitude, line graph mV,\ndirecionado mV",
    "escala": "espectrograma, escalograma",
}


def painel(ax, pontos, sub, coluna, legenda, indice, kappa, titulo):
    for chave, (nome, cor, marcador) in legenda.items():
        m = (sub[coluna] == chave).to_numpy()
        total = int((indice[coluna] == chave).sum())
        ax.scatter(pontos[m, 0], pontos[m, 1], s=17 if marcador == "o" else 22, zorder=3,
                   color=cor, marker=marcador, edgecolor="white", linewidth=0.4,
                   label=f"{nome} — {int(m.sum())} de {total}")
    ax.legend(frameon=False, fontsize=10, loc="center left", bbox_to_anchor=(0.0, 0.66), handletextpad=0.4,
              borderpad=0.3, labelspacing=0.5)
    ax.set_title(f"{titulo}\npropagação recupera {kappa[0]:.1%} (acaso {kappa[1]:.1%}, κ = {kappa[2]:.2f})",
                 fontsize=12.5, color=TINTA, loc="left", pad=8, linespacing=1.5)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--summary", type=Path, default=ROOT / "results/redundancy_graph_v2/summary.json")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--output", type=Path, default=ROOT / "results/redundancy_graph_v2/layout_formato_invariancia.png")
    args = ap.parse_args()
    resumo = json.loads(args.summary.read_text())
    limiar = resumo["threshold"]

    valores, indice = carregar(args.regime)
    r2 = matriz_r2(valores)
    linhas, colunas = np.where(np.triu(r2 >= limiar, k=1))
    pesos_todos = r2[linhas, colunas]
    grafo = nx.Graph()
    grafo.add_nodes_from(range(len(indice)))
    grafo.add_weighted_edges_from(zip(linhas.tolist(), colunas.tolist(), pesos_todos.tolist()))
    grafo.remove_nodes_from(list(nx.isolates(grafo)))
    isolados = len(indice) - grafo.number_of_nodes()
    print(f"{len(linhas):,} arestas; {grafo.number_of_nodes():,} nós; layout...", flush=True)
    layout = nx.spring_layout(grafo, weight="weight", seed=SEED, iterations=args.iterations,
                              k=2.2 / np.sqrt(grafo.number_of_nodes()))
    nos = np.array(sorted(grafo.nodes))
    pontos = np.array([layout[n] for n in nos])
    posicao = {n: i for i, n in enumerate(nos)}
    sub = indice.iloc[nos].reset_index(drop=True)
    dentro = np.isin(linhas, nos) & np.isin(colunas, nos)
    segmentos = np.stack([pontos[[posicao[a] for a in linhas[dentro]]],
                          pontos[[posicao[b] for b in colunas[dentro]]]], axis=1)
    pesos = pesos_todos[dentro]

    def kappa(nivel):
        visto = indice[f"previsto_{nivel}"].notna() if f"previsto_{nivel}" in indice else None
        acerto = resumo[nivel]["coesao"]
        props = indice.loc[visto, nivel].value_counts(normalize=True) if visto is not None else \
            indice[nivel].value_counts(normalize=True)
        acaso = float((props ** 2).sum())
        return acerto, acaso, (acerto - acaso) / (1 - acaso)

    import pandas as pd
    previstos = pd.read_parquet(args.summary.parent / "feature_index.parquet")
    indice = indice.merge(previstos[["feature", "previsto_view", "previsto_invariance"]], on="feature", how="left")

    fig = plt.figure(figsize=(16.0, 12.4))
    grade = fig.add_gridspec(2, 4, height_ratios=[2.3, 1], hspace=0.2, wspace=0.05,
                             left=0.015, right=0.985, top=0.93, bottom=0.07)
    limites = (pontos[:, 0].min(), pontos[:, 0].max(), pontos[:, 1].min(), pontos[:, 1].max())
    folga = 0.04 * max(limites[1] - limites[0], limites[3] - limites[2])

    for k, (coluna, legenda, titulo) in enumerate((
            ("view", FORMATO, "Por formato: série se dissolve em imagem e grafo"),
            ("invariance", INVARIANCIA, "Por invariância: amplitude e grafo mV formam um bloco"))):
        ax = fig.add_subplot(grade[0, 2 * k:2 * k + 2])
        arestas(ax, segmentos, pesos, limiar)
        painel(ax, pontos, sub, coluna, legenda, indice, kappa(coluna), titulo)
        ax.set_xlim(limites[0] - folga, limites[1] + folga)
        ax.set_ylim(limites[2] - folga, limites[3] + folga)

    for i, (grupo, (nome, _, _)) in enumerate(INVARIANCIA.items()):
        eixo = fig.add_subplot(grade[1, i])
        arestas(eixo, segmentos, pesos, limiar, forca=0.5)
        m = (sub["invariance"] == grupo).to_numpy()
        eixo.scatter(pontos[~m, 0], pontos[~m, 1], s=4, zorder=2, color=CINZA_NO, linewidth=0)
        eixo.scatter(pontos[m, 0], pontos[m, 1], s=10, zorder=3, color=DESTAQUE,
                     edgecolor="white", linewidth=0.3)
        eixo.set_xlim(limites[0] - folga, limites[1] + folga)
        eixo.set_ylim(limites[2] - folga, limites[3] + folga)
        eixo.set_title(f"{nome}  ({int(m.sum())})\n{FAMILIAS_DO_GRUPO[grupo]}", fontsize=9.4,
                       color=TINTA, loc="left", pad=5, linespacing=1.4)

    fig.text(0.5, 0.018,
             "Um único layout, que recebe só as arestas de r² e nunca vê família, formato ou invariância; "
             "os dois painéis superiores mudam apenas a cor.\n"
             f"{len(indice):,} features de 23 famílias, folds 1–7 (regime exploratório); "
             f"{len(linhas):,} arestas com r² ≥ {limiar:.2f} (cotovelo de percolação da componente gigante); "
             f"{isolados:,} features sem par acima do limiar não são desenhadas.",
             ha="center", fontsize=9.2, color=MUDO, linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
