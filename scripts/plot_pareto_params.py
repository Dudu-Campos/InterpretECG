"""Fronteira de Pareto: parametros aprendidos vs 1 - AUROC macro, fold 9, rasos e profundos.

Boosting: limiares + valores de folha de todas as arvores dos 5 classificadores
(count_boosting_params.py). Profundos: parametros treinaveis dos checkpoints;
media +- dp de 5 sementes. Rotulo nos pontos rasos: numero de features (k).
Regime exploratorio (treino 1-7, parada 8, teste 9) para todos.

Fusao tardia inception1d + GNN em L(HVG) direcionado (16/nov): media simples
das probabilidades, SEM peso ajustado -- nada e escolhido, entao o fold 9
continua sendo avaliacao unica e nao selecao. A fusao e PAREADA POR SEMENTE
(42, 43, 44), nao um ensemble: cada ponto do grafico e um modelo, e somar 5
sementes de um com 3 do outro custaria 4,17 M em vez de 1,08 M. Parametros da
fusao = soma dos dois modelos pareados.
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
OUT = ROOT / "results/pareto"
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"
CORES = {"tsfresh (só série)": ("#2a78d6", "o"), "melhor + multiplex": ("#1baf7a", "D")}


def fusao_pareada(ROOT, prof):
    """inception1d + GNN em L(HVG) direcionado, media simples, semente a semente.

    Pareia pela semente em vez de somar ensembles: assim o ponto continua sendo
    "um modelo de cada", comparavel aos demais pontos da figura, e os parametros
    sao a soma dos dois. Nenhum peso e ajustado -- a media e fixa em 1/2.
    """
    import glob, re
    from sklearn.metrics import roc_auc_score
    inc = {int(re.search(r"seed(\d+)", f).group(1)): np.load(f)
           for f in glob.glob(str(ROOT / "results/deep_inception1d_exploratory_seed4*/test_scores.npz"))}
    g = {}
    for f in glob.glob(str(ROOT / "results/line_gnn_cv_exploratory/hvg_dir_d64_l4_e50_test9_val8*_test_scores.npz")):
        m = re.search(r"seed(\d+)", f)
        g[int(m.group(1)) if m else 42] = np.load(f)
    comuns = sorted(set(inc) & set(g))
    y, ids = inc[comuns[0]]["alvos"], inc[comuns[0]]["ecg_id"]
    v = []
    for s in comuns:
        assert np.array_equal(g[s]["ecg_id"], ids), "ordem dos registros difere"
        v.append(roc_auc_score(y, (inc[s]["escores"] + g[s]["scores"]) / 2, average="macro"))
    par_inc = float(prof.loc[prof.modelo == "inception1d", "parametros"].iloc[0])
    return {"modelo": "inception1d + GNN em L(HVG)", "parametros": par_inc + 571_141,
            "erro": 1 - float(np.mean(v)), "dp": float(np.std(v)), "sementes": len(v)}


def main() -> None:
    b = pd.read_csv(OUT / "boosting_params.csv")
    b["erro"] = 1 - b.auroc
    # tsfresh: so o melhor passo pela VALIDACAO (fold 8); a curva inteira esta em plot_dimension_curves.py
    import json
    passos = json.load(open(ROOT / "results/pruning_tsfresh_exploratory/steps.json"))
    k_ts = max(passos, key=lambda p: p["validation_macro_auroc"])["n_features"]
    passos_mv = json.load(open(ROOT / "results/pruning_bestmx_exploratory/steps.json"))
    k_mv = max(passos_mv, key=lambda p: p["validation_macro_auroc"])["n_features"]
    # um ponto por conjunto, escolhido pela validacao (estetica; a curva completa
    # esta em results/dimension_curves e em boosting_params.csv)
    b = b[((b.modelo == "tsfresh (só série)") & (b.k == k_ts)) | ((b.modelo == "melhor + multiplex") & (b.k == k_mv))]
    prof = pd.read_csv(OUT / "pontos.csv")
    prof = prof[prof.tipo == "profundo"].copy()
    # GNN sobre L(HVG) (train_line_gnn.py), 50 epocas, epoca escolhida no fold 8, teste fold 9
    import glob, json
    gnn = []
    for padrao, nome, par in (("hvg_d64_l4_e50_test9_val8*.json", "GNN em L(HVG)", 520_453),
                              ("hvg_dir_d64_l4_e50_test9_val8*.json", "GNN em L(HVG) direcionado", 571_141)):
        v = [json.load(open(f))["test_macro_auroc"] for f in glob.glob(str(ROOT / "results/line_gnn_cv_exploratory" / padrao))]
        gnn.append({"modelo": nome, "parametros": par, "erro": 1 - np.mean(v), "dp": np.std(v) if len(v) > 1 else 0.0,
                    "sementes": len(v)})
    gnn = pd.DataFrame(gnn)
    print(gnn)
    fus = pd.DataFrame([fusao_pareada(ROOT, prof)])
    print(fus)
    todos = pd.concat([b[["modelo", "parametros", "erro"]].assign(tipo="raso", k=b.k),
                       prof[["modelo", "parametros", "erro"]].assign(tipo="profundo"),
                       gnn[["modelo", "parametros", "erro"]].assign(tipo="gnn"),
                       fus[["modelo", "parametros", "erro"]].assign(tipo="fusao")]).sort_values(["parametros", "erro"])
    melhor, fr = np.inf, []
    for i, r in todos.reset_index(drop=True).iterrows():
        if r.erro < melhor - 1e-12:
            fr.append(r); melhor = r.erro
    fr = pd.DataFrame(fr)
    todos.assign(pareto=todos.set_index(["modelo", "parametros"]).index.isin(
        fr.set_index(["modelo", "parametros"]).index)).to_csv(OUT / "pareto_params.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.6, 6.4))
    xs, ys = [], []
    for _, r in fr.iterrows():
        if xs:
            xs.append(r.parametros); ys.append(ys[-1])
        xs.append(r.parametros); ys.append(r.erro)
    ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="fronteira de Pareto")
    ts = b[b.modelo == "tsfresh (só série)"]
    ax.scatter(ts.parametros, ts.erro, color=CORES["tsfresh (só série)"][0], marker="o", s=60, edgecolor="white",
               linewidth=1, zorder=4, label=f"boosting: tsfresh, melhor passo na validação (k={k_ts})")
    mv = b[b.modelo == "melhor + multiplex"]
    ax.scatter(mv.parametros, mv.erro, color=CORES["melhor + multiplex"][0], marker="D", s=60, edgecolor="white",
               linewidth=1, zorder=4, label=f"boosting: melhor + multiplex, melhor passo na validação (k={k_mv})")
    ax.errorbar(prof.parametros, prof.erro, yerr=prof.dp, fmt="s", color=TINTA, ms=6, capsize=3, zorder=4,
                label="profundos (sinal cru)")
    ax.errorbar(gnn.parametros, gnn.erro, yerr=gnn.dp, fmt="*", color="#1baf7a", mec=TINTA, mew=0.6, ms=13,
                capsize=3, zorder=5, label="GNN em L(HVG) (grafo como entrada)")
    ax.errorbar(fus.parametros, fus.erro, yerr=fus.dp, fmt="P", color="#eb6834", mec=TINTA, mew=0.6, ms=11,
                capsize=3, zorder=6, label="fusão tardia: inception1d + GNN em L(HVG), pesos iguais")
    for _, r in fus.iterrows():
        ax.annotate(f"fusão ({r.sementes} sem. pareadas)", (r.parametros, r.erro), xytext=(11, -2),
                    textcoords="offset points", va="center", fontsize=8.5, color=TINTA)

    # valor reportado na literatura (fora do nosso protocolo; nao entra no calculo da fronteira)
    REP_AUC, REP_PAR = 0.926, 17.98e6
    ax.axhline(1 - REP_AUC, color=MUDO, lw=1, ls="--", zorder=1)
    ax.scatter([REP_PAR], [1 - REP_AUC], marker="o", s=55, facecolor="white", edgecolor=TINTA, linewidth=1.2,
               zorder=5, label="WResHDual-ViT, reportado (AUROC 0,926; 17,98 M)")
    ax.annotate("WResHDual-ViT (reportado)", (REP_PAR, 1 - REP_AUC), xytext=(0, 10), textcoords="offset points", ha="center",
                va="center", fontsize=8.5, color=TINTA)
    dy_gnn = {"GNN em L(HVG) direcionado": -13, "GNN em L(HVG)": 2}
    for _, r in gnn.iterrows():
        ax.annotate(f"{r.modelo} ({r.sementes} sem.)", (r.parametros, r.erro),
                    xytext=(-11, dy_gnn.get(r.modelo, 0)), ha="right",
                    textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
    desloc = {"LSTM bidirecional": 7, "inception1d": 9, "ECG-IMN": -11}
    for _, r in prof.iterrows():
        ax.annotate(r.modelo, (r.parametros, r.erro), xytext=(9, desloc.get(r.modelo, 0)),
                    textcoords="offset points", va="center", fontsize=8.5, color=TINTA)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.065, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: (f"{v:.3f}" if v < 0.07 else f"{v:.2f}").replace(".", ",")))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: (f"{v / 1e6:g} M" if v >= 1e6 else f"{v / 1e3:g} mil").replace(".", ",")))
    ax.set_xlim(6e4, 9e7)
    ax.tick_params(colors=MUDO, labelsize=8.5)
    ax.set_xlabel("parâmetros aprendidos (log)", color=TINTA)
    ax.set_ylabel("1 − AUROC macro, fold 9 (log)", color=TINTA)
    ax.grid(color=GRADE, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.legend(frameon=False, fontsize=8.3, loc="upper right", borderaxespad=0.4,
              handletextpad=0.5, labelspacing=0.45)
    ax.set_title("Parâmetros vs erro: boosting e redes no mesmo protocolo", loc="left", color=TINTA,
                 fontsize=12.5, pad=46)
    ax.text(0, 1.02,
            "boosting: limiares + valores de folha (5 rótulos) · redes: pesos treináveis, média ± dp · fold 9\n"
            "boosting: um ponto por conjunto, o melhor passo da poda na validação\n"
            "fusão: média simples das probabilidades, pareada por semente — nenhum peso ajustado",
            transform=ax.transAxes, color=MUDO, fontsize=8.3, linespacing=1.5)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_params_erro.png", dpi=170, facecolor="white")
    # PDF vetorial para o LaTeX; o PNG serve para olhar rapido
    fig.savefig(OUT / "pareto_params_erro.pdf", facecolor="white")
    print(fr[["modelo", "parametros", "erro"]].assign(auroc=lambda d: 1 - d.erro).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
