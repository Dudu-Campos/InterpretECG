"""Curvas de erro (1 - AUROC macro, fold 9) em funcao do numero de features, eixo x log.

Cada ponto e um passo da poda recursiva (prune_features.py: importancia por
permutacao no fold 8, reajuste a cada passo). Bandas: IC 95% por bootstrap
pareado de registros do fold 9 (mesmos indices para todas as curvas).
Regime exploratorio (1-7 / 8 / 9). Tambem grava a tabela por k com a
diferenca multi-view menos so-serie, quando os dois tem o mesmo k.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CURVAS = (  # (diretorio, nome, cor, marcador)
    ("results/pruning_tsfresh_exploratory", "só série (tsfresh)", "#2a78d6", "o"),
    ("results/pruning_exploratory", "multi-view: base + direcionado", "#eb6834", "s"),
    ("results/pruning_bestmx_exploratory", "multi-view: melhor + multiplex", "#1baf7a", "D"),
)
PONTOS = (  # referencias isoladas
    ("results/views_key_coalitions_exploratory/catch22_scores.npz", "catch22 (264)", 264),
)
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"


def auc(y, s):
    return roc_auc_score(y, s, average="macro")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--omit", nargs="*", default=[], help="diretorios de CURVAS a deixar fora")
    ap.add_argument("--name", default="erro_vs_features.png")
    ap.add_argument("--output", type=Path, default=ROOT / "results/dimension_curves")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    series, ids_ref, y = {}, None, None
    for d, nome, cor, mk in CURVAS:
        if d in args.omit:
            continue
        arquivos = sorted((ROOT / d).glob("k*_scores.npz"), key=lambda p: int(p.stem[1:].split("_")[0]))
        if not arquivos:
            continue
        pts = {}
        for p in arquivos:
            z = np.load(p)
            if ids_ref is None:
                ids_ref, y = z["ecg_id_test"], z["y_test"]
            assert (z["ecg_id_test"] == ids_ref).all()
            pts[int(p.stem[1:].split("_")[0])] = z["test"]
        series[nome] = (cor, mk, dict(sorted(pts.items())))
    refs = {}
    for caminho, nome, k in PONTOS:
        z = np.load(ROOT / caminho)
        ordem = pd.Series(np.arange(len(z["ecg_id_test"])), index=z["ecg_id_test"]).loc[ids_ref].to_numpy()
        refs[nome] = (k, z["test"][ordem])

    rng = np.random.default_rng(0)
    n = len(ids_ref)
    amostras = [rng.integers(0, n, n) for _ in range(args.boot)]
    amostras = [a for a in amostras if (y[a].sum(0) > 0).all()]

    def boot(s):
        return np.array([auc(y[a], s[a]) for a in amostras])

    linhas, dist = [], {}
    for nome, (_, _, pts) in series.items():
        for k, s in pts.items():
            b = boot(s)
            dist[(nome, k)] = b
            linhas.append({"curva": nome, "k": k, "auroc": auc(y, s), "erro": 1 - auc(y, s),
                           "erro_lo": 1 - np.quantile(b, .975), "erro_hi": 1 - np.quantile(b, .025)})
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(args.output / "curvas.csv", index=False)

    serie = CURVAS[0][1]
    difs = []
    for nome in list(series)[1:]:
        for k in series[nome][2]:
            if (serie, k) in dist:
                dd = dist[(nome, k)] - dist[(serie, k)]
                difs.append({"multi_view": nome, "k": k,
                             "dif_auroc": auc(y, series[nome][2][k]) - auc(y, series[serie][2][k]),
                             "ic_lo": np.quantile(dd, .025), "ic_hi": np.quantile(dd, .975)})
    pd.DataFrame(difs).to_csv(args.output / "diferencas.csv", index=False)
    print(pd.DataFrame(difs).round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    for nome, (cor, mk, pts) in series.items():
        t = tabela[tabela.curva == nome].sort_values("k")
        ax.fill_between(t.k, t.erro_lo, t.erro_hi, color=cor, alpha=0.14, linewidth=0)
        ax.plot(t.k, t.erro, color=cor, lw=2, marker=mk, ms=7, mec="white", mew=1.2, label=nome, zorder=3)
    for nome, (k, s) in refs.items():
        b = boot(s)
        e = 1 - auc(y, s)
        ax.errorbar([k], [e], yerr=[[e - (1 - np.quantile(b, .975))], [(1 - np.quantile(b, .025)) - e]],
                    fmt="X", color=MUDO, ms=9, capsize=3, label=nome, zorder=3)
    cnn = 1 - 0.9287
    ax.axhline(cnn, color=TINTA, lw=1, ls="--", zorder=1)
    ax.text(34, cnn * 0.985, "xresnet1d101 (sinal cru), 0,9287",
            color=TINTA, fontsize=9, va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    yt = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    ax.yaxis.set_major_locator(FixedLocator(yt))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ks = sorted(set(tabela.k))
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:,}".replace(",", ".") for k in ks], fontsize=8.5)
    ax.minorticks_off()
    ax.set_xlabel("número de features (log)", color=TINTA)
    ax.set_ylabel("1 − AUROC macro, fold 9 (log)", color=TINTA)
    ax.grid(color=GRADE, lw=0.7)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors=MUDO)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    ax.set_title("Erro em função da dimensão: poda recursiva por importância (fold 8)",
                 loc="left", color=TINTA, fontsize=12.5, pad=24)
    ax.text(0, 1.02, "regime exploratório (treino 1–7, seleção 8, teste 9) · bandas: IC 95% bootstrap pareado · "
            "menor é melhor", transform=ax.transAxes, color=MUDO, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.output / args.name, dpi=170, facecolor="white")
    print(tabela.round(4).to_string(index=False))
    print(f"-> {args.output / args.name}")


if __name__ == "__main__":
    main()
