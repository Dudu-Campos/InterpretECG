"""Heatmap k x n de log(erro so-serie / erro multi-view), de results/heatmap_dimension_data/celulas.csv.

Verde: multi-view erra menos; azul: so-serie (tsfresh) erra menos (as cores das
curvas em plot_dimension_curves.py). Escala divergente simetrica, meio neutro.
Numero na celula: o log-razao (x100). Erro de cada conjunto = media sobre
sementes de subamostragem (3 para n < treino completo, 1 no completo).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/heatmap_dimension_data"
TINTA, MUDO = "#1f2328", "#6b7280"


def main() -> None:
    df = pd.read_csv(D / "celulas.csv")
    n_med = df.groupby("fracao").n.mean()
    m = df.groupby(["conjunto", "k", "fracao"]).erro.mean().unstack("conjunto")
    m["lr"] = np.log(m["serie"] / m["multiview"])
    # consistencia: em quantas sementes o sinal concorda com o da media
    por_semente = df.pivot_table(index=["k", "fracao", "semente"], columns="conjunto", values="erro")
    por_semente["lr"] = np.log(por_semente["serie"] / por_semente["multiview"])
    m.to_csv(D / "heatmap.csv")
    grade = m["lr"].unstack("fracao").sort_index(ascending=False)
    ks, fr = grade.index.to_numpy(), grade.columns.to_numpy()
    lim = np.abs(grade.to_numpy()).max()

    cmap = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f3f4f6", "#1baf7a"])
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    im = ax.imshow(grade.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i, k in enumerate(ks):
        for j, f in enumerate(fr):
            v = grade.iloc[i, j]
            sem = por_semente.loc[(k, f), "lr"]
            mesma = (np.sign(sem) == np.sign(v)).sum()
            txt = f"{100 * v:+.1f}".replace(".", ",")
            if len(sem) > 1:
                txt += f"\n{mesma}/{len(sem)}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color=TINTA if abs(v) < 0.6 * lim else "white")
    ax.set_xticks(range(len(fr)))
    ax.set_xticklabels([f"{int(round(n_med[f])):,}".replace(",", ".") for f in fr])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([f"{k:,}".replace(",", ".") for k in ks])
    ax.set_xlabel("registros de treino (folds 1–7, subamostra por paciente)", color=TINTA)
    ax.set_ylabel("número de features", color=TINTA)
    ax.tick_params(colors=MUDO, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("log(erro só-série / erro multi-view)", color=TINTA)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=MUDO)
    ax.set_title("Onde o multi-view erra menos que o tsfresh", loc="left", color=TINTA, fontsize=12.5, pad=40)
    ax.text(0, 1.03, "verde: multi-view (melhor + multiplex) melhor · azul: tsfresh melhor · número: log-razão ×100;\n"
            "embaixo, em quantas sementes o sinal concorda · erro = 1 − AUROC macro, fold 9",
            transform=ax.transAxes, color=MUDO, fontsize=8.5, va="bottom")
    fig.text(0.01, 0.01, "Seleção de features fixa: caminho da poda com o treino completo; n varia só no reajuste.",
             color=MUDO, fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(D / "heatmap_log_razao.png", dpi=170, facecolor="white")
    print(grade.round(4).to_string())
    print(f"-> {D / 'heatmap_log_razao.png'}")


if __name__ == "__main__":
    main()
