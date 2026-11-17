r"""Fronteira de Pareto: numero de features (entrada do classificador) vs 1 - AUROC macro, fold 9.

Rasos: passos das podas (tsfresh, base + direcionado, melhor + multiplex),
k = 16 e 8 da grade do heatmap (treino completo), catch22 (264). Profundos:
entrada = sinal cru, 12 derivacoes x 1000 amostras = 12.000 valores; media de
5 sementes, barra = desvio. Parametros treinaveis contados dos checkpoints.
Regime exploratorio (treino 1-7, parada 8, teste 9) para todos.

A FIGURA mostra so o que a pergunta dela pede: as duas curvas de poda (tsfresh
e prism-ecg) mais a xresnet1d101 como referencia profunda. Os demais pontos
(catch22, inception1d, LSTM, Transformer, ECG-IMN) continuam sendo CALCULADOS e
gravados em pontos.csv, porque `plot_pareto_params.py` le esse arquivo para os
modelos profundos — tirar do calculo quebraria a outra figura. Por isso a
fronteira desenhada e recalculada apenas sobre os pontos visiveis.

Texto em ingles, com titulo e sem subtitulo: o bloco explicativo vai no
`\caption` do LaTeX.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pareto"
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"
RASOS = (("results/pruning_tsfresh_exploratory", "tsfresh (só série)", "#2a78d6", "o"),
         ("results/pruning_bestmx_exploratory", "melhor + multiplex", "#1baf7a", "D"))
EN = {"tsfresh (só série)": "tsfresh (boosting)", "melhor + multiplex": "prism-ecg (boosting)",
      "xresnet1d101": "xresnet1d101"}
VISIVEIS = ("tsfresh (só série)", "melhor + multiplex", "xresnet1d101")
PROFUNDOS = (("results/cnn_xresnet1d101_exploratory*/metrics.json", "xresnet1d101", 33_555_333),
             ("results/deep_lstm_bidir_exploratory_seed*/metrics.json", "LSTM bidirecional", 2_330_629),
             ("results/deep_inception1d_exploratory_seed*/metrics.json", "inception1d", 507_653),
             ("results/deep_transformer_exploratory_seed*/metrics.json", "Transformer", 1_469_061),
             ("results/imn_exploratory/multilabel5*.json", "ECG-IMN", 140_330))
ENTRADA = 12 * 1000


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pts = []
    grade = pd.read_csv(ROOT / "results/heatmap_dimension_data/celulas.csv")
    grade = grade[grade.fracao == 1]
    for d, nome, cor, mk in RASOS:
        for p in json.load(open(ROOT / d / "steps.json")):
            pts.append({"modelo": nome, "k": p["n_features"], "erro": 1 - p["test_macro_auroc"], "tipo": "raso"})
        conj = "serie" if "tsfresh" in d else "multiview"
        for k in (8, 16):
            e = grade[(grade.conjunto == conj) & (grade.k == k)].erro.iloc[0]
            pts.append({"modelo": nome, "k": k, "erro": e, "tipo": "raso"})
    pts.append({"modelo": "catch22", "k": 264, "erro": 1 - 0.8768, "tipo": "raso"})
    for pat, nome, par in PROFUNDOS:
        fs = [f for f in sorted(glob.glob(str(ROOT / pat))) if "confirm" not in f]
        v = [json.load(open(f)).get("test", {}) for f in fs]
        v = [x["macro_auroc"] if isinstance(x, dict) and "macro_auroc" in x else json.load(open(f))["test_macro_auroc"]
             for x, f in zip(v, fs)]
        pts.append({"modelo": nome, "k": ENTRADA, "erro": 1 - np.mean(v), "dp": np.std(v), "tipo": "profundo",
                    "parametros": par, "sementes": len(v)})
    df = pd.DataFrame(pts).sort_values(["k", "erro"])
    # Pareto: minimizar k e erro
    melhor, fronteira = np.inf, []
    for i, r in df.iterrows():
        if r.erro < melhor - 1e-12:
            fronteira.append(i)
            melhor = r.erro
    df["pareto"] = df.index.isin(fronteira)
    df.to_csv(OUT / "pontos.csv", index=False)

    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    # fronteira recalculada so sobre o que aparece na figura
    vis = df[df.modelo.isin(VISIVEIS)].sort_values(["k", "erro"])
    melhor, idx = np.inf, []
    for i, r in vis.iterrows():
        if r.erro < melhor - 1e-12:
            idx.append(i); melhor = r.erro
    f = vis.loc[idx].sort_values("k")
    xs, ys = [], []
    for _, r in f.iterrows():
        if xs:
            xs.append(r.k); ys.append(ys[-1])
        xs.append(r.k); ys.append(r.erro)
    ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="Pareto frontier")

    for d, nome, cor, mk in RASOS:
        t = df[df.modelo == nome].sort_values("k")
        ax.plot(t.k, t.erro, color=cor, lw=2.4, marker=mk, ms=8, mec="white", mew=1.2,
                label=EN[nome], zorder=3)
    ref = df[df.modelo == "xresnet1d101"]
    ax.errorbar(ref.k, ref.erro, yerr=ref.dp, fmt="s", color=TINTA, ms=8, capsize=3, zorder=4,
                label="xresnet1d101 (raw signal)")
    for _, r in ref.iterrows():
        ax.annotate(f"xresnet1d101 ({r.parametros / 1e6:.0f}M parameters)", (r.k, r.erro),
                    xytext=(-13, 0), textcoords="offset points", ha="right", va="center",
                    fontsize=12, color=TINTA)

    ax.set_xscale("log"); ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ks = [8, 16, 33, 67, 134, 268, 537, 1075, 2150, 4449]
    ax.xaxis.set_major_locator(FixedLocator(ks + [ENTRADA]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: "12,000\n(raw signal)" if v == ENTRADA else f"{int(v):,}"))
    ax.set_xlim(6, ENTRADA * 4.5)
    ax.tick_params(colors=MUDO, labelsize=11)
    ax.set_xlabel("number of features at the classifier input (log)", color=TINTA, fontsize=13)
    ax.set_ylabel("1 − macro AUROC, fold 9 (log)", color=TINTA, fontsize=13)
    ax.grid(color=GRADE, lw=0.7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRADE)
    ax.legend(frameon=False, fontsize=12, loc="upper right", labelspacing=0.55)
    ax.set_title("Features vs error: pruning curves against a deep reference (PTB-XL, fold 9)",
                 loc="left", color=TINTA, fontsize=14.5, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_features_erro.png", dpi=170, facecolor="white")
    # PDF vetorial para o LaTeX; o PNG serve para olhar rapido
    fig.savefig(OUT / "pareto_features_erro.pdf", facecolor="white")
    print("fronteira desenhada (so os pontos visiveis):")
    print(f[["modelo", "k", "erro"]].assign(auroc=lambda d: 1 - d.erro).round(4).to_string(index=False))
    print("\nprofundos calculados e gravados em pontos.csv (usados por plot_pareto_params.py):")
    print(df[df.tipo == "profundo"][["modelo", "erro", "dp", "parametros"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
