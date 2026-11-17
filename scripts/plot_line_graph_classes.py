"""L(HVG) de um batimento por superclasse do PTB-XL, em V1 e V5.

Layout com leitura fisica: cada no de L(HVG) e uma corda (i, j) da HVG,
desenhada no ponto medio da corda ((i + j) / 2, eixo x) e na altura do seu
nivel min(x_i, x_j) (eixo y, mV). Duas cordas se ligam quando compartilham uma
amostra (cunha). Cor do no: vao j - i (log). So entram as cordas inteiramente
dentro da janela mostrada (~1,2 s em torno de um pico R).

Exemplos: um registro tipico por classe, rotulo unico, verossimilhanca SCP 100,
sem anotacao de ruido, folds 1-7 (nada do teste).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402
from interpretecg.views.series_to_graph.line import line_graph_features  # noqa: E402
from interpretecg.views.series_to_graph.visibility import _hvg_edges  # noqa: E402

EXEMPLOS = [("NORM", 3, "normal"), ("MI", 177, "infarto anterosseptal (ASMI)"),
            ("STTC", 674, "isquemia (ISC_)"), ("CD", 424, "bloqueio de ramo direito (CRBBB)"),
            ("HYP", 542, "hipertrofia ventricular esquerda (LVH)")]
DERIV = ("V1", "V5")
FS = 100
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"


def janela(sinal: np.ndarray) -> tuple[int, int]:
    """~1,2 s em torno do pico R mais proximo de 4 s, achado em II."""
    x = sinal[:, LEADS.index("II")]
    x = np.abs(x - np.median(x))
    picos, _ = find_peaks(x, distance=40, height=np.percentile(x, 97))
    r = picos[np.argmin(np.abs(picos - 400))]
    return max(0, r - 35), min(len(x), r + 85)


def main() -> None:
    loader = PTBXLDataLoader(ROOT / "data/ptb-xl")
    fig, eixos = plt.subplots(len(EXEMPLOS), 2, figsize=(11, 13), sharey="col")
    norma = LogNorm(1, 120)
    cmap = plt.get_cmap("Blues")
    for lin, (classe, ecg, nome) in enumerate(EXEMPLOS):
        sinal = loader.load_signal(ecg)[0]
        a, b = janela(sinal)
        for col, d in enumerate(DERIV):
            ax = eixos[lin, col]
            x = sinal[:, LEADS.index(d)].astype(float)
            e = np.array(list(_hvg_edges(x)), dtype=np.int64)
            nivel_dp = line_graph_features(x, e)[1]["level_std"]
            e = e[(e[:, 0] >= a) & (e[:, 1] < b)]
            meio = (e[:, 0] + e[:, 1]) / 2 / FS
            nivel = np.minimum(x[e[:, 0]], x[e[:, 1]])
            vao = e[:, 1] - e[:, 0]
            # cunhas: pares de cordas que compartilham uma amostra
            seg = []
            for s in range(a, b):
                inc = np.flatnonzero((e[:, 0] == s) | (e[:, 1] == s))
                for p in range(len(inc)):
                    for q in range(p + 1, len(inc)):
                        u, v = inc[p], inc[q]
                        seg.append([(meio[u], nivel[u]), (meio[v], nivel[v])])
            ax.add_collection(LineCollection(seg, colors="#9ca3af", linewidths=0.35, alpha=0.55, zorder=1))
            t = np.arange(a, b) / FS
            ax.plot(t, x[a:b], color=TINTA, lw=1.1, zorder=3)
            sc = ax.scatter(meio, nivel, c=vao, cmap=cmap, norm=norma, s=9, edgecolor="none", zorder=2)
            ax.set_xlim(a / FS, (b - 1) / FS)
            ax.text(0.99, 0.04, f"{len(e)} nós · {len(seg)} cunhas · dp do nível (10 s) {nivel_dp:.2f} mV",
                    transform=ax.transAxes, ha="right", fontsize=7.5, color=MUDO)
            if lin == 0:
                ax.set_title(d, color=TINTA, fontsize=11.5, loc="left")
            if col == 0:
                ax.set_ylabel(f"{classe}\n(mV)", color=TINTA, fontsize=10.5, rotation=0, ha="right", va="center",
                              labelpad=8)
                ax.text(0.01, 0.96, f"{nome} · ecg_id {ecg}", transform=ax.transAxes, fontsize=8,
                        color=MUDO, va="top")
            if lin == len(EXEMPLOS) - 1:
                ax.set_xlabel("tempo no registro (s)", color=TINTA)
            ax.tick_params(colors=MUDO, labelsize=8)
            ax.grid(color=GRADE, lw=0.6)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
            for s_ in ("left", "bottom"):
                ax.spines[s_].set_color(GRADE)
    fig.suptitle("Line graph da HVG, um batimento por superclasse", x=0.01, y=0.995, ha="left", color=TINTA, fontsize=13)
    fig.text(0.01, 0.958, "cada ponto é uma corda horizontal da HVG, no ponto médio e na altura do seu nível "
             "min(xᵢ, xⱼ); linhas cinza ligam cordas que compartilham uma amostra (cunhas)\n"
             "cor = vão da corda; linha escura = ECG; mesma escala em mV por coluna; registros dos folds de treino",
             color=MUDO, fontsize=8.5)
    cax = fig.add_axes([0.80, 0.935, 0.17, 0.007])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.set_ticks([1, 10, 100]); cb.set_ticklabels(["10 ms", "100 ms", "1 s"])
    cb.ax.tick_params(labelsize=7.5, colors=MUDO); cb.outline.set_visible(False)
    cb.set_label("vão da corda", fontsize=8, color=MUDO, labelpad=2)
    fig.tight_layout(rect=(0.02, 0, 1, 0.915))
    saida = ROOT / "results/figures/line_graph_classes.png"
    fig.savefig(saida, dpi=170, facecolor="white")
    print(saida)


if __name__ == "__main__":
    main()
