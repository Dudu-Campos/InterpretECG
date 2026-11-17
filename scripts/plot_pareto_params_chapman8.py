"""Fronteira de Pareto no chapman8 (8 ritmos): parametros aprendidos vs 1 - AUROC macro, fold 10.

Mesma figura que plot_pareto_params.py (PTB-XL, fold 9), no protocolo da tabela
final do chapman8: treino 1-8, validacao 9, teste 10, 5 sementes (42-46).
- Boosting: limiares + valores de folha (campo `parametros` de
  boosting_<conjunto>_pruned.json, ajuste da semente 42); AUROC = media +- dp
  das 5 sementes (results/boosting_seeds_fold10.json).
- Redes: parametros treinaveis do checkpoint da semente 42 (floats do
  state_dict sem os buffers de BatchNorm); AUROC = media +- dp das sementes
  disponiveis.
- GNN: rede de Takens (melhor visao do chapman8, pre-registrada) e L(HVG)
  direcionada (para a discussao).
Saida: results/pareto/pareto_params_chapman8.{png,pdf,csv}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results/external/chapman8"
OUT = ROOT / "results/pareto"
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"
S = range(42, 47)


def params(arq: Path) -> int:
    sd = torch.load(arq, map_location="cpu", weights_only=False)
    return sum(v.numel() for k, v in sd.items() if v.is_floating_point()
               and not any(b in k for b in ("running_", "num_batches")))


def aucs_deep(arch: str) -> list[float]:
    v = []
    for s in S:
        f = D / f"deep/{arch}_confirmatory_seed{s}/metrics.json"
        if f.exists():
            v.append(json.load(open(f))["test"]["macro_auroc"])
    return v


def aucs_json(padrao: str) -> list[float]:
    return [json.load(open(D / padrao.format(s=s)))["test_macro_auroc"] for s in S
            if (D / padrao.format(s=s)).exists()]


def main() -> None:
    seeds = json.load(open(ROOT / "results/boosting_seeds_fold10.json"))
    linhas = []
    for nome, chave, arq in (("tsfresh + boosting", "chapman8/tsfresh", "tsfresh_pruned"),
                             ("nosso boosting", "chapman8/nosso", "todas_familias_pruned")):
        r = seeds[chave]
        linhas.append({"modelo": nome, "tipo": "raso", "k": r["k"], "sementes": len(r["por_semente"]),
                       "parametros": json.load(open(D / f"boosting_{arq}.json"))["parametros"],
                       "auroc": r["media"], "dp": r["dp"]})
    for nome, arch in (("xresnet1d101", "xresnet1d101"), ("inception1d", "inception1d"),
                       ("LSTM bidirecional", "lstm_bidir"), ("Transformer", "transformer")):
        v = aucs_deep(arch)
        linhas.append({"modelo": nome, "tipo": "profundo", "sementes": len(v), "auroc": np.mean(v),
                       "dp": np.std(v, ddof=1),
                       "parametros": params(D / f"deep/{arch}_confirmatory_seed42/model.pt")})
    v = aucs_json("deep/imn/chapman8_multilabel5_confirmatory_seed{s}.json")
    v = [json.load(open(D / "deep/imn/chapman8_multilabel5_confirmatory.json"))["test_macro_auroc"]] + v
    linhas.append({"modelo": "ECG-IMN", "tipo": "profundo", "sementes": len(v), "auroc": np.mean(v),
                   "dp": np.std(v, ddof=1), "parametros": params(D / "deep/imn/chapman8_multilabel5_confirmatory.pt")})
    for nome, g in (("GNN na rede de Takens", "takens"), ("GNN em L(HVG) direcionado", "hvg")):
        v = aucs_json(f"deep/gnn/{g}_dir_d64_l4_e50_conf_seed{{s}}.json")
        linhas.append({"modelo": nome, "tipo": "gnn", "sementes": len(v), "auroc": np.mean(v),
                       "dp": np.std(v, ddof=1), "parametros": params(D / f"deep/gnn/{g}_dir_d64_l4_e50_conf_seed42.pt")})
    d = pd.DataFrame(linhas)
    d["erro"] = 1 - d.auroc
    d = d.sort_values(["parametros", "erro"]).reset_index(drop=True)
    melhor, fr = np.inf, []
    for _, r in d.iterrows():
        if r.erro < melhor - 1e-12:
            fr.append(r.modelo); melhor = r.erro
    d["pareto"] = d.modelo.isin(fr)
    d.to_csv(OUT / "pareto_params_chapman8.csv", index=False)
    print(d[["modelo", "parametros", "auroc", "dp", "sementes", "pareto"]].round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9.2, 5.9))
    f = d[d.pareto]
    xs, ys = [], []
    for _, r in f.iterrows():
        if xs:
            xs.append(r.parametros); ys.append(ys[-1])
        xs.append(r.parametros); ys.append(r.erro)
    ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="fronteira de Pareto")
    for _, r in d.iterrows():
        if r.modelo == "tsfresh + boosting":
            ax.errorbar(r.parametros, r.erro, yerr=r.dp, fmt="o", color="#2a78d6", mec="white", mew=1, ms=8,
                        capsize=3, zorder=4, label=f"boosting: tsfresh, podado (k={int(r.k)})")
        elif r.modelo == "nosso boosting":
            ax.errorbar(r.parametros, r.erro, yerr=r.dp, fmt="D", color="#1baf7a", mec="white", mew=1, ms=8,
                        capsize=3, zorder=4, label=f"boosting: todas as famílias, podado (k={int(r.k)})")
    p = d[d.tipo == "profundo"]
    ax.errorbar(p.parametros, p.erro, yerr=p.dp, fmt="s", color=TINTA, ms=6, capsize=3, zorder=4,
                label="profundos (sinal cru)")
    g = d[d.modelo == "GNN em L(HVG) direcionado"].iloc[0]
    ax.errorbar(g.parametros, g.erro, yerr=g.dp, fmt="*", color="#1baf7a", mec=TINTA, mew=0.6, ms=13, capsize=3,
                zorder=5, label="GNN em L(HVG) (grafo como entrada)")
    t = d[d.modelo == "GNN na rede de Takens"].iloc[0]
    ax.errorbar(t.parametros, t.erro, yerr=t.dp, fmt="*", color="white", mec="#1baf7a", mew=1.2, ms=13, capsize=3,
                ecolor="#1baf7a", zorder=5, label="GNN na rede de Takens (melhor visão do boosting)")
    rot = {"tsfresh + boosting": ("tsfresh", (9, 0), "left"), "nosso boosting": ("nosso boosting", (-10, 0), "right"),
           "xresnet1d101": ("xresnet1d101", (9, 0), "left"), "inception1d": ("inception1d", (9, 8), "left"),
           "LSTM bidirecional": ("LSTM bidirecional", (9, 0), "left"), "Transformer": ("Transformer", (9, 0), "left"),
           "ECG-IMN": ("ECG-IMN", (8, -11), "left")}
    for _, r in d.iterrows():
        if r.modelo in rot:
            txt, off, ha = rot[r.modelo]
        else:
            txt, off, ha = f"{r.modelo} ({r.sementes} sem.)", ((-10, 0) if "L(HVG)" in r.modelo else (10, -2)), \
                ("right" if "L(HVG)" in r.modelo else "left")
        ax.annotate(txt, (r.parametros, r.erro), xytext=off, textcoords="offset points", ha=ha, va="center",
                    fontsize=8.5, color=TINTA)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} mil").replace(".", ",")))
    ax.set_xlim(1e5, 8e7)
    ax.tick_params(colors=MUDO, labelsize=8.5)
    ax.set_xlabel("parâmetros aprendidos (log)", color=TINTA)
    ax.set_ylabel("1 − AUROC macro, fold 10 (log)", color=TINTA)
    ax.grid(color=GRADE, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Parâmetros vs erro no Chapman (8 ritmos): boosting e redes no mesmo protocolo", loc="left",
                 color=TINTA, fontsize=12.5, pad=36)
    ns = sorted(set(d.sementes))
    ax.text(0, 1.02, "boosting: limiares + valores de folha (8 rótulos, semente 42) · redes: pesos treináveis · "
            f"média ± dp de {'–'.join(map(str, ns))} sementes · fold 10\n"
            "boosting: lista podada no próprio dataset, escolhida no fold 8 (treino 1–7)",
            transform=ax.transAxes, color=MUDO, fontsize=8.3)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pareto_params_chapman8.{ext}", dpi=170, facecolor="white")


if __name__ == "__main__":
    main()
