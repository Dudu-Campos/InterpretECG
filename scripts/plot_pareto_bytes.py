"""Fronteira de Pareto no PTB-XL, fold 10: tamanho do modelo em bytes vs 1 - AUROC macro.

Bytes na codificacao compacta, o que a predicao precisa:
- redes: numero de valores em ponto flutuante do state_dict x 4 bytes (float32);
- boosting: por no interno feature (uint16) + limiar (float32) + 2 filhos (uint16),
  por folha valor (float32), mais as medianas do imputador (float32 por feature).
  Contado dos modelos reajustados na semente 42 (results/model_size/*.pkl).
Erro: media de 5 sementes; barras = IC 95% por bootstrap de registros
(results/confirmatory/tabela_ptbxl_ic.csv, scripts/ptbxl_table_ci.py).
Saida: results/pareto/pareto_bytes_fold10.{png,pdf} e pareto_bytes_fold10.csv
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results/pareto"
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"

PT = {"xresnet1d101": "cnn_xresnet1d101/model.pt", "inception1d": "deep_inception1d_confirmatory_seed42/model.pt",
      "LSTM bidirecional": "deep_lstm_bidir_confirmatory_seed42/model.pt",
      "Transformer": "deep_transformer_confirmatory_seed42/model.pt",
      "ECG-IMN": "imn_exploratory/multilabel5_confirmatory.pt",
      "GNN L(HVG) direcionada": "line_gnn_confirmatory/hvg_dir_d64_l4_e50_conf_seed42.pt"}
PKL = {"catch22 + boosting": "catch22", "tsfresh + boosting": "tsfresh", "nosso boosting": "melhor_multiplex"}


def bytes_rede(arq: Path) -> int:
    sd = torch.load(arq, map_location="cpu")
    return 4 * sum(v.numel() for v in sd.values() if v.is_floating_point())


def bytes_boosting(arq: Path) -> int:
    m = pickle.load(open(arq, "rb"))
    internos = folhas = 0
    for mm in m.modelos:
        for preds in mm._predictors:
            for p in preds:
                folha = p.nodes["is_leaf"].astype(bool)
                folhas += int(folha.sum()); internos += int((~folha).sum())
    return internos * (2 + 4 + 2 + 2) + folhas * 4 + 4 * len(m.imputador.statistics_)


def main() -> None:
    t = pd.read_csv(ROOT / "results/confirmatory/tabela_ptbxl_ic.csv").set_index("modelo")
    linhas = []
    for nome in t.index:
        b = (bytes_boosting(ROOT / "results/model_size" / f"boosting_{PKL[nome]}.pkl") if nome in PKL
             else bytes_rede(ROOT / "results" / PT[nome]))
        tipo = "raso" if nome in PKL else ("gnn" if nome.startswith("GNN") else "profundo")
        linhas.append({"modelo": nome, "bytes": b, "erro": 1 - t.auroc[nome], "erro_lo": 1 - t.ic_hi[nome],
                       "erro_hi": 1 - t.ic_lo[nome], "tipo": tipo})
    d = pd.DataFrame(linhas).sort_values(["bytes", "erro"]).reset_index(drop=True)
    melhor, fr = np.inf, []
    for _, r in d.iterrows():
        if r.erro < melhor - 1e-12:
            fr.append(r.modelo); melhor = r.erro
    d["pareto"] = d.modelo.isin(fr)
    d.to_csv(OUT / "pareto_bytes_fold10.csv", index=False)
    print(d.assign(MB=d.bytes / 1e6).round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    f = d[d.pareto]
    xs, ys = [], []
    for _, r in f.iterrows():
        if xs:
            xs.append(r.bytes); ys.append(ys[-1])
        xs.append(r.bytes); ys.append(r.erro)
    ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="fronteira de Pareto")
    estilo = {"catch22 + boosting": ("#9aa4b2", "o", 7), "tsfresh + boosting": ("#2a78d6", "o", 7),
              "nosso boosting": ("#1baf7a", "D", 7)}
    for _, r in d.iterrows():
        err = [[r.erro - r.erro_lo], [r.erro_hi - r.erro]]
        if r.tipo == "raso":
            cor, mk, ms = estilo[r.modelo]
            ax.errorbar(r.bytes, r.erro, yerr=err, fmt=mk, color=cor, mec="white", mew=0.8, ms=ms, ecolor=cor,
                        elinewidth=1, capsize=2.5, zorder=4)
        elif r.tipo == "gnn":
            ax.errorbar(r.bytes, r.erro, yerr=err, fmt="*", color="#1baf7a", mec=TINTA, mew=0.6, ms=13,
                        ecolor="#1baf7a", elinewidth=1, capsize=2.5, zorder=5)
        else:
            ax.errorbar(r.bytes, r.erro, yerr=err, fmt="s", color=TINTA, ms=5.5, ecolor=TINTA, elinewidth=1,
                        capsize=2.5, zorder=4)
    # legenda por classe de modelo
    ax.scatter([], [], marker="D", color="#1baf7a", s=40, label="nosso: boosting multi-view")
    ax.scatter([], [], marker="*", color="#1baf7a", edgecolor=TINTA, linewidth=0.6, s=120,
               label="nosso: GNN na visão L(HVG)")
    ax.scatter([], [], marker="o", color="#2a78d6", s=40, label="boosting só-série (tsfresh, catch22)")
    ax.scatter([], [], marker="s", color=TINTA, s=30, label="redes no sinal cru")
    # valor reportado (outro protocolo de implementacao; nao entra na fronteira); bytes supondo float32
    REP_AUC, REP_B = 0.926, 17.98e6 * 4
    ax.axhline(1 - REP_AUC, color=MUDO, lw=0.9, ls="--", zorder=1)
    ax.scatter([REP_B], [1 - REP_AUC], marker="o", s=45, facecolor="white", edgecolor=TINTA, linewidth=1.1, zorder=5,
               label="WResHDual-ViT, reportado (float32 suposto)")
    rot = {"catch22 + boosting": ("catch22", (8, 0), "left"), "tsfresh + boosting": ("tsfresh", (8, 4), "left"),
           "nosso boosting": ("nosso boosting", (-9, -2), "right"), "inception1d": ("inception1d", (8, 7), "left"),
           "GNN L(HVG) direcionada": ("GNN L(HVG)", (10, -13), "left"), "Transformer": ("Transformer", (8, 0), "left"),
           "LSTM bidirecional": ("LSTM", (8, 0), "left"), "ECG-IMN": ("ECG-IMN", (-8, 0), "right"),
           "xresnet1d101": ("xresnet1d101", (-9, -9), "right")}
    for _, r in d.iterrows():
        txt, off, ha = rot[r.modelo]
        ax.annotate(txt, (r.bytes, r.erro), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=8, color=TINTA)
    ax.annotate("WResHDual-ViT", (REP_B, 1 - REP_AUC), xytext=(0, 8), textcoords="offset points", ha="center",
                fontsize=8, color=MUDO)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1.5e5, 4e8)
    ax.yaxis.set_major_locator(FixedLocator([0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ",")))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v / 1e6:g} MB" if v >= 1e6 else f"{v / 1e3:g} kB").replace(".", ",")))
    ax.tick_params(colors=MUDO, labelsize=8)
    ax.set_xlabel("tamanho do modelo, codificação compacta (bytes, log)", color=TINTA, fontsize=9)
    ax.set_ylabel("1 − AUROC macro, fold 10 (log)", color=TINTA, fontsize=9)
    ax.grid(color=GRADE, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pareto_bytes_fold10.{ext}", dpi=200, facecolor="white")


if __name__ == "__main__":
    main()
