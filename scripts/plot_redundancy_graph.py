"""Layout por força do grafo de similaridade r² entre features.

O painel grande responde à pergunta direta: as features das três visões se
clusterizam? Os painéis pequenos mostram família por família, porque o painel
grande já revela que as famílias NÃO são espacialmente compactas — e um rótulo
no centroide de uma família dispersa aponta para espaço vazio ou para o
aglomerado de outra, o que seria pior que não rotular.

Segue `attic/experiments/plot_rich_redundancy_layout.py` do PRISM: o layout
recebe apenas as arestas de r² e nunca vê a que família uma feature pertence,
então qualquer agrupamento visível é estrutura do dado, não do desenho.

Paleta: slots 1–3 do `references/palette.md` do skill dataviz, o único trio
documentado como aprovado em `--pairs all` (dispersão, onde qualquer par de
cores pode ficar adjacente). O validador não pôde ser executado aqui (sem
`node`), então o trio foi usado como documentado, não re-derivado. Cada painel
pequeno tem uma só categoria destacada, logo não há par de cores em jogo neles.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from redundancy_graph import carregar, matriz_r2  # noqa: E402

SEED = 42
COR_VISAO = {"serie": "#2a78d6", "imagem": "#eb6834", "grafo": "#1baf7a"}
DESTAQUE = "#2a78d6"
TINTA, MUDO, CINZA_NO = "#0b0b0b", "#7a7975", "#d6d5d1"

ORDEM = ["catch22", "image_spectrogram", "image_scalogram", "image_gasf",
         "image_gadf", "image_mtf", "image_recurrence", "graph_hvg",
         "graph_lphvg1", "graph_takens", "graph_transition"]
NOTA = {
    "image_gasf": "par com GADF", "image_gadf": "par com GASF",
    "graph_hvg": "par com LPHVG", "graph_lphvg1": "par com HVG",
    "image_mtf": "par com transição", "graph_transition": "par com MTF",
}


def arestas(ax, segmentos, pesos, limiar, forca=1.0):
    alpha = (0.04 + 0.32 * (pesos - limiar) / (1 - limiar + 1e-9)) * forca
    ax.add_collection(LineCollection(segmentos, colors="#8a8a8a",
                                     linewidths=0.30, alpha=alpha, zorder=1))
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--output", type=Path,
                    default=ROOT / "results/redundancy_graph/layout.png")
    args = ap.parse_args()

    valores, indice = carregar(args.regime)
    r2 = matriz_r2(valores)
    linhas, colunas = np.where(np.triu(r2 >= args.threshold, k=1))
    pesos_todos = r2[linhas, colunas]

    grafo = nx.Graph()
    grafo.add_nodes_from(range(len(indice)))
    grafo.add_weighted_edges_from(zip(linhas.tolist(), colunas.tolist(),
                                      pesos_todos.tolist()))
    grafo.remove_nodes_from(list(nx.isolates(grafo)))
    isolados = len(indice) - grafo.number_of_nodes()
    print(f"{len(linhas):,} arestas; {grafo.number_of_nodes():,} nós, "
          f"{isolados:,} isolados omitidos")

    print(f"layout por força ({args.iterations} iterações)...")
    layout = nx.spring_layout(grafo, weight="weight", seed=SEED,
                              iterations=args.iterations,
                              k=2.2 / np.sqrt(grafo.number_of_nodes()))
    nos = np.array(sorted(grafo.nodes))
    pontos = np.array([layout[n] for n in nos])
    posicao = {n: i for i, n in enumerate(nos)}
    sub = indice.iloc[nos].reset_index(drop=True)

    dentro = np.isin(linhas, nos) & np.isin(colunas, nos)
    segmentos = np.stack([
        pontos[[posicao[a] for a in linhas[dentro]]],
        pontos[[posicao[b] for b in colunas[dentro]]],
    ], axis=1)
    pesos = pesos_todos[dentro]

    fig = plt.figure(figsize=(16.0, 17.6))
    grade = fig.add_gridspec(4, 4, height_ratios=[2.55, 1, 1, 1],
                             hspace=0.16, wspace=0.05,
                             left=0.015, right=0.985, top=0.945, bottom=0.052)

    ax = fig.add_subplot(grade[0, :])
    arestas(ax, segmentos, pesos, args.threshold)
    for visao, cor in COR_VISAO.items():
        m = (sub["view"] == visao).to_numpy()
        if not m.any():
            continue
        total = int((indice["view"] == visao).sum())
        ax.scatter(pontos[m, 0], pontos[m, 1], s=19, zorder=3, color=cor,
                   edgecolor="white", linewidth=0.45,
                   label=f"{visao} — {int(m.sum())} de {total} desenhadas")
    ax.legend(frameon=False, fontsize=10.5, loc="lower left", handletextpad=0.5,
              borderpad=0.4, labelspacing=0.55)
    ax.set_title("Imagem e grafo ocupam o mesmo espaço: as visões não se separam",
                 fontsize=13.5, color=TINTA, loc="left", pad=10)

    limites = (pontos[:, 0].min(), pontos[:, 0].max(),
               pontos[:, 1].min(), pontos[:, 1].max())
    folga = 0.04 * max(limites[1] - limites[0], limites[3] - limites[2])
    for i, familia in enumerate(ORDEM):
        eixo = fig.add_subplot(grade[1 + i // 4, i % 4])
        arestas(eixo, segmentos, pesos, args.threshold, forca=0.5)
        m = (sub["family"] == familia).to_numpy()
        eixo.scatter(pontos[~m, 0], pontos[~m, 1], s=4.5, zorder=2,
                     color=CINZA_NO, linewidth=0)
        eixo.scatter(pontos[m, 0], pontos[m, 1], s=11, zorder=3,
                     color=DESTAQUE, edgecolor="white", linewidth=0.3)
        eixo.set_xlim(limites[0] - folga, limites[1] + folga)
        eixo.set_ylim(limites[2] - folga, limites[3] + folga)
        nome = familia.replace("image_", "").replace("graph_", "")
        extra = NOTA.get(familia)
        eixo.set_title(f"{nome}  ({int(m.sum())})" + (f"\n{extra}" if extra else "\n "),
                       fontsize=9.6, color=TINTA, loc="left", pad=5,
                       linespacing=1.45)
    fig.add_subplot(grade[3, 3]).axis("off")

    fig.text(0.5, 0.019,
             "O layout recebe apenas as arestas de r² — nunca vê a que família ou visão uma "
             "feature pertence, então todo agrupamento é estrutura do dado.\n"
             f"{len(indice):,} features dos folds 1–7 (regime exploratório), "
             f"{len(linhas):,} arestas com r² ≥ {args.threshold:g}; "
             f"{isolados:,} features sem nenhum par acima do limiar não são desenhadas.",
             ha="center", fontsize=9.2, color=MUDO, linespacing=1.5)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
