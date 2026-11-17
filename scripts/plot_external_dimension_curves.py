"""Erro (1 - AUROC macro) vs numero de features nos datasets externos, log-log.

Cada ponto e um passo da poda dependente da tarefa (prune_external.py). O eixo y
e a VALIDACAO (fold 8), o mesmo fold que ordenou as features: os valores sao
otimistas, igualmente para os dois conjuntos. O fold 10 nao entra (so o k
escolhido foi testado, em evaluate_external.py); o ponto escolhido e marcado.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"
CONJ = (("tsfresh", "só série (tsfresh)", "#2a78d6", "o"),
        ("melhor_multiplex", "multi-view: melhor + multiplex", "#1baf7a", "D"))
DATASETS = (("chapman", "Chapman, 4 grupos de ritmo"), ("chapman8", "Chapman, 8 ritmos"),
            ("code15", "CODE-15%, 6 anomalias"))


def main() -> None:
    feitos = [(d, t) for d, t in DATASETS
              if all((ROOT / f"results/external/{d}/pruning_{c}/selected.json").exists() for c, *_ in CONJ)]
    fig, eixos = plt.subplots(1, len(feitos), figsize=(5.2 * len(feitos), 4.6), squeeze=False)
    import numpy as np
    faixas = {}
    for d, _ in feitos:
        v = [1 - p["validation_macro_auroc"] for c, *_ in CONJ
             for p in json.load(open(ROOT / f"results/external/{d}/pruning_{c}/steps.json"))]
        faixas[d] = (min(v), max(v))
    # mesma razao (decadas por polegada) em todos os paineis: distancia vertical igual = razao de erro igual
    razao = max(hi / lo for lo, hi in faixas.values()) * 1.3
    for ax, (d, titulo) in zip(eixos[0], feitos):
        ys = []
        for c, nome, cor, mk in CONJ:
            passos = json.load(open(ROOT / f"results/external/{d}/pruning_{c}/steps.json"))
            k = [p["n_features"] for p in passos]
            e = [1 - p["validation_macro_auroc"] for p in passos]
            ys += e
            ax.plot(k, e, color=cor, marker=mk, ms=5, lw=1.6, label=nome, zorder=3)
            esc = json.load(open(ROOT / f"results/external/{d}/pruning_{c}/selected.json"))
            ax.scatter([esc["k"]], [1 - esc["validation_macro_auroc_fold8"]], s=150, facecolor="none",
                       edgecolor=cor, linewidth=1.6, zorder=4)
        ax.set_xscale("log"); ax.set_yscale("log")
        centro = np.sqrt(faixas[d][0] * faixas[d][1])
        lo, hi = centro / np.sqrt(razao), centro * np.sqrt(razao)
        ticks = [t for t in (0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07, 0.1, 0.15)
                 if lo <= t <= hi]
        ax.set_ylim(lo, hi)
        ax.yaxis.set_major_locator(FixedLocator(ticks)); ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3f}".rstrip("0").replace(".", ",")))
        ax.xaxis.set_major_locator(FixedLocator([33, 134, 537, 2150, 8943]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}".replace(",", ".")))
        ax.set_title(titulo, loc="left", color=TINTA, fontsize=11)
        ax.set_xlabel("número de features (log)", color=TINTA)
        ax.tick_params(colors=MUDO, labelsize=8.5)
        ax.grid(color=GRADE, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRADE)
    eixos[0][0].set_ylabel("1 − AUROC macro, validação (fold 8, log)", color=TINTA)
    h, l = eixos[0][0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.01, 0.865), ncol=2)
    fig.suptitle("Poda dependente da tarefa: erro vs número de features", x=0.01, ha="left", color=TINTA,
                 fontsize=12.5)
    fig.text(0.01, 0.885, "cada ponto é um passo da poda (treino folds 1–7, ranking e early stopping no fold 8); "
             "círculo = k escolhido; valores otimistas, igualmente para os dois conjuntos\n"
             "eixos y com a mesma extensão em razão: a mesma distância vertical é a mesma variação relativa do erro nos dois painéis",
             color=MUDO, fontsize=8.3)
    fig.tight_layout(rect=(0, 0, 1, 0.82))
    saida = ROOT / "results/external/erro_vs_features_externos.png"
    fig.savefig(saida, dpi=170, facecolor="white")
    print(saida)


if __name__ == "__main__":
    main()
