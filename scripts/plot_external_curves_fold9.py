"""Figura erro x features nos datasets externos, regime exploratorio (teste = fold 9).

Mesmo formato da figura do PTB-XL (rasos): curvas da poda por conjunto, ponto do
catch22, fronteira de Pareto sobre todos os pontos rasos. Paineis com a mesma
extensao do eixo y em razao (a mesma distancia vertical e a mesma variacao
relativa do erro). Dados: results/external/<dataset>/curvas_fold9.csv
(scripts/external_dimension_curves.py).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"
CONJ = (("tsfresh", "tsfresh (só série)", "#2a78d6", "o"),
        ("melhor_multiplex", "melhor + multiplex (filtro do PTB-XL)", "#1baf7a", "D"),
        ("todas_familias", "todas as famílias", "#9b4dca", "s"))
DATASETS = (("chapman", "Chapman, 4 grupos de ritmo"), ("chapman8", "Chapman, 8 ritmos"),
            ("code15", "CODE-15%, 6 anomalias"))


def fronteira(pts: pd.DataFrame):
    pts = pts.sort_values(["k", "erro_fold9"])
    xs, ys, melhor = [], [], np.inf
    for _, r in pts.iterrows():
        if r.erro_fold9 < melhor - 1e-12:
            if xs:
                xs.append(r.k); ys.append(ys[-1])
            xs.append(r.k); ys.append(r.erro_fold9); melhor = r.erro_fold9
    return xs, ys


def main() -> None:
    feitos = [(d, t) for d, t in DATASETS if (ROOT / f"results/external/{d}/curvas_fold9.csv").exists()]
    dados = {d: pd.read_csv(ROOT / f"results/external/{d}/curvas_fold9.csv") for d, _ in feitos}
    razao = max(q.erro_fold9.max() / q.erro_fold9.min() for q in dados.values()) * 1.35
    fig, eixos = plt.subplots(1, len(feitos), figsize=(5.4 * len(feitos), 4.8), squeeze=False)
    for ax, (d, titulo) in zip(eixos[0], feitos):
        q = dados[d]
        xs, ys = fronteira(q)
        ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="fronteira de Pareto")
        for c, nome, cor, mk in CONJ:
            s = q[q.conjunto == c].sort_values("k")
            if len(s):
                ax.plot(s.k, s.erro_fold9, color=cor, marker=mk, ms=5, lw=1.8, mec="white", mew=0.8,
                        label=nome, zorder=3)
        c22 = q[q.conjunto == "catch22"]
        ax.scatter(c22.k, c22.erro_fold9, marker="X", s=90, color="#6b7280", edgecolor="white", linewidth=0.8,
                   zorder=4, label="catch22")
        centro = np.sqrt(q.erro_fold9.min() * q.erro_fold9.max())
        lo, hi = centro / np.sqrt(razao), centro * np.sqrt(razao)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(lo, hi)
        ticks = [t for t in (0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.15)
                 if lo <= t <= hi]
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").replace(".", ",")))
        ax.xaxis.set_major_locator(FixedLocator([33, 134, 537, 2150, 8943]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
        ax.set_title(titulo, loc="left", color=TINTA, fontsize=11)
        ax.set_xlabel("número de features (log)", color=TINTA)
        ax.tick_params(colors=MUDO, labelsize=8.5)
        ax.grid(color=GRADE, lw=0.7)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        for s_ in ("left", "bottom"):
            ax.spines[s_].set_color(GRADE)
    eixos[0][0].set_ylabel("1 − AUROC macro, fold 9 (log)", color=TINTA)
    h, l = eixos[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=8.8, loc="upper left", bbox_to_anchor=(0.01, 0.875), ncol=5)
    fig.suptitle("Features vs erro nos datasets externos (rasos)", x=0.01, ha="left", color=TINTA, fontsize=12.5)
    fig.text(0.01, 0.905, "regime exploratório: treino folds 1–7, parada e ranking da poda no fold 8, teste no fold 9 "
             "(o fold 10 não entra) · eixos y com a mesma extensão em razão",
             color=MUDO, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    saida = ROOT / "results/external/erro_vs_features_fold9.png"
    fig.savefig(saida, dpi=170, facecolor="white")
    print(saida)


if __name__ == "__main__":
    main()
