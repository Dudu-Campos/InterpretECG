r"""Fronteira de Pareto no fold 10: parametros aprendidos vs 1 - AUROC macro.

Versao final da figura de parametros, agora no conjunto de teste selado. A de
fold 9 (`plot_pareto_params.py`) continua existindo para a analise exploratoria.

Estatistica por modelo: media das AUROCs macro das 5 sementes. Barras: IC 95%
por 2.000 reamostragens dos registros do fold 10, **as mesmas para todos**,
como em `ptbxl_table_ci.py` — reaproveita o mapeamento de arquivos daquele
script para nao divergir dele.

Fusao tardia inception1d + GNN em L(HVG) direcionado: media simples das
probabilidades, pesos iguais e nao ajustados, **pareada por semente** (nao e
ensemble: cada ponto da figura e um modelo de cada, e os parametros sao a soma
dos dois). Nao houve leitura nova do fold 10 — os dois modelos ja o leram nas
rodadas pre-registradas e aqui so se combinam escores gravados. A fusao nao
esta no pre-registro: entra declarada como post-hoc, registrada em
results/test_fold_usage.jsonl.

Parametros: redes e GNN contados dos checkpoints; boosting = limiares + valores
de folha dos modelos reajustados em results/model_size/*.pkl (note que sao os
confirmatorios, nao os exploratorios da figura do fold 9).

A figura sai em INGLES e SEM titulo nem subtitulo: o texto explicativo vai no
\caption do LaTeX, e repeti-lo na imagem duplicaria a legenda. A chave de
simbolos fica, porque sem ela o grafico nao se le. Separador decimal: ponto.

O bootstrap e caro (2.000 reamostragens x 10 modelos x 5 sementes x 5 rotulos),
entao fica em cache no .npz; so recalcula com --rebootstrap.
"""
from __future__ import annotations

import importlib.util as _u
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pareto"
TINTA, MUDO, GRADE = "#0b0b0b", "#7a7975", "#e3e2de"
# Convencao do projeto (16/nov): grafo = VERMELHO. As figuras de redundancia
# (plot_redundancy_graph.py, plot_redundancy_compare.py) ainda usam verde para
# grafo e ficam inconsistentes ate serem atualizadas.
VERDE, VERMELHO, AZUL, CINZA = "#1baf7a", "#e34948", "#2a78d6", "#6b7280"
# Cor separa autoria, nao familia de metodo: verde/vermelho/azul sao nossos,
# preto e cinza sao de terceiros. catch22 e tsfresh sao baselines publicados
# e estavam em verde, o que os fazia parecer parte da contribuicao.

# nome interno (como vem de ptbxl_table_ci.py) -> rotulo na figura, em ingles
EN = {
    "catch22 + boosting": "catch22 (boosting)", "tsfresh + boosting": "tsfresh (boosting)",
    "nosso boosting": "prism-ecg (boosting)", "xresnet1d101": "xresnet1d101",
    "inception1d": "inception1d", "LSTM bidirecional": "bidirectional LSTM",
    "Transformer": "Transformer", "ECG-IMN": "ECG-IMN",
    "GNN L(HVG) direcionada": "prism-ecg (deep)",
}
PARAMETROS = {
    "catch22 + boosting": 278_300, "tsfresh + boosting": 280_039,
    "nosso boosting": 263_600, "xresnet1d101": 33_555_333,
    "inception1d": 507_653, "LSTM bidirecional": 2_330_629,
    "Transformer": 1_469_061, "ECG-IMN": 140_330,
    "GNN L(HVG) direcionada": 571_141,
}
FUSAO = "fusão: inception1d + GNN L(HVG)"
EN[FUSAO] = "fusion: inception1d + prism-ecg (deep)"
PARAMETROS[FUSAO] = PARAMETROS["inception1d"] + PARAMETROS["GNN L(HVG) direcionada"]

TIPO = {"catch22 + boosting": "base_raso", "tsfresh + boosting": "base_raso",
        "nosso boosting": "raso", "GNN L(HVG) direcionada": "gnn", FUSAO: "fusao"}


def carregar():
    """Escores do fold 10 por modelo, na ordem canonica dos ecg_id."""
    spec = _u.spec_from_file_location("ci", ROOT / "scripts/ptbxl_table_ci.py")
    ci = _u.module_from_spec(spec)
    sys.argv = ["x"]
    try:
        spec.loader.exec_module(ci)
    except SystemExit:
        pass
    lab = pd.read_csv(ROOT / "export/ecg_fm_bundle/ptbxl/labels.csv").set_index("ecg_id")
    rot = ["NORM", "MI", "STTC", "CD", "HYP"]
    ids_ref, escores = None, {}
    for nome, (arqs, chave) in ci.MODELOS.items():
        lista = []
        for a in arqs:
            z = np.load(a)
            ids = (np.load(ci.R / f"confirmatory/{ci.ORDEM_BOOSTING[nome]}_scores.npz")["ecg_id_test"]
                   if nome in ci.ORDEM_BOOSTING else z["ecg_id"])
            if ids_ref is None:
                ids_ref = np.sort(ids)
            lista.append(pd.DataFrame(z[chave], index=ids).loc[ids_ref].to_numpy())
        escores[nome] = lista
    assert set(lab.loc[ids_ref, "strat_fold"]) == {10}, "o fold nao e o 10"
    # fusao: pesos iguais, semente a semente
    escores[FUSAO] = [(a + b) / 2 for a, b in
                      zip(escores["inception1d"], escores["GNN L(HVG) direcionada"])]
    return lab.loc[ids_ref, rot].to_numpy(), escores


def main() -> None:
    y, escores = carregar()
    auc = lambda yy, s: float(np.mean([roc_auc_score(yy[:, i], s[:, i]) for i in range(yy.shape[1])]))
    cache = OUT / "pareto_params_fold10_boot.npz"
    if cache.exists() and "--rebootstrap" not in sys.argv:
        z = np.load(cache)
        boot = {m: z[m] for m in z.files}
        assert set(boot) == set(escores), "cache com modelos diferentes; use --rebootstrap"
        print(f"bootstrap reaproveitado de {cache.name}")
    else:
        rng = np.random.default_rng(42)
        n, boot = len(y), {m: [] for m in escores}
        while len(boot[FUSAO]) < 2000:
            i = rng.integers(0, n, n)
            if (y[i].sum(0) == 0).any() or (y[i].sum(0) == n).any():
                continue
            for m, ss in escores.items():
                boot[m].append(np.mean([auc(y[i], s[i]) for s in ss]))

    linhas = []
    for m, ss in escores.items():
        lo, hi = np.percentile(boot[m], [2.5, 97.5])
        linhas.append({"modelo": m, "parametros": PARAMETROS[m],
                       "erro": 1 - np.mean([auc(y, s) for s in ss]),
                       "erro_lo": 1 - hi, "erro_hi": 1 - lo,
                       "tipo": TIPO.get(m, "profundo"), "sementes": len(ss)})
    d = pd.DataFrame(linhas).sort_values(["parametros", "erro"]).reset_index(drop=True)

    melhor, fr = np.inf, []
    for _, r in d.iterrows():
        if r.erro < melhor - 1e-12:
            fr.append(r); melhor = r.erro
    fr = pd.DataFrame(fr)
    d.assign(pareto=d.modelo.isin(fr.modelo)).to_csv(OUT / "pareto_params_fold10.csv", index=False)

    B = {m: np.array(v) for m, v in boot.items()}
    np.savez(OUT / "pareto_params_fold10_boot.npz", **B)
    print("\ndiferencas PAREADAS contra a fusao (mesmas reamostragens):")
    for m in ("xresnet1d101", "GNN L(HVG) direcionada", "inception1d", "nosso boosting"):
        dif = B[FUSAO] - B[m]
        lo, hi = np.percentile(dif, [2.5, 97.5])
        print(f"  fusão − {m:24s} {dif.mean():+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
              f"{'SIM' if lo > 0 else 'nao'}")

    fig, ax = plt.subplots(figsize=(11.6, 7.0))
    xs, ys = [], []
    for _, r in fr.iterrows():
        if xs:
            xs.append(r.parametros); ys.append(ys[-1])
        xs.append(r.parametros); ys.append(r.erro)
    ax.plot(xs, ys, color=TINTA, lw=1, ls=":", zorder=1, label="Pareto frontier")

    # a forma do marcador e codificacao secundaria: vermelho e verde sao o par
    # critico para daltonismo, e D / * / P / s os separam sem depender da cor.
    # O catch22 (erro 0,128) esticava o eixo em 24% sozinho, comprimindo a regiao
    # onde estao todos os modelos da fronteira. Ele nao esta na fronteira -- e
    # dominado pelo prism-ecg (boosting), que tem menos parametros E menos erro.
    # Em vez de some-lo, fica no TOPO com seta para cima e o valor anotado: o
    # leitor ve que existe e onde esta, sem a escala ser ditada por ele.
    FORA = "catch22 + boosting"
    lim_cima = d[d.modelo != FORA].erro_hi.max() * 1.035

    estilo = {"base_raso": dict(fmt="D", color=CINZA, ms=9),
              "raso": dict(fmt="D", color=VERDE, ms=9),
              "profundo": dict(fmt="s", color=TINTA, ms=8),
              "gnn": dict(fmt="*", color=VERMELHO, ms=17, mec=TINTA, mew=0.6),
              "fusao": dict(fmt="P", color=AZUL, ms=15, mec=TINTA, mew=0.6)}
    rotulo = {"base_raso": "baseline boosting (catch22, tsfresh)",
              "raso": "prism-ecg boosting (handcrafted views)",
              "profundo": "deep baselines (raw signal)",
              "gnn": "prism-ecg deep: GNN on L(HVG)",
              "fusao": "late fusion, equal weights (post-hoc)"}
    for t, st in estilo.items():
        g = d[(d.tipo == t) & (d.modelo != FORA)]
        if g.empty:
            continue
        ax.errorbar(g.parametros, g.erro, yerr=[g.erro - g.erro_lo, g.erro_hi - g.erro],
                    capsize=3, zorder=5 if t in ("gnn", "fusao", "raso") else 4, label=rotulo[t], **st)

    fora = d[d.modelo == FORA].iloc[0]
    ax.plot([fora.parametros], [lim_cima], marker="^", ms=13, color=CINZA,
            mec=TINTA, mew=0.6, zorder=6)
    ax.annotate(f"{EN[FORA]}: {fora.erro:.3f}", (fora.parametros, lim_cima),
                xytext=(13, 0), textcoords="offset points", ha="left", va="center",
                fontsize=11.5, color=CINZA)

    desloc = {"LSTM bidirecional": (9, 8), "inception1d": (9, 9), "ECG-IMN": (9, 11),
              "Transformer": (9, 0), "xresnet1d101": (9, 0), "catch22 + boosting": (9, 0),
              "tsfresh + boosting": (9, 9), "nosso boosting": (-10, -11),
              "GNN L(HVG) direcionada": (-13, -22), FUSAO: (11, -2)}
    for _, r in d.iterrows():
        dx, dy = desloc.get(r.modelo, (9, 0))
        ax.annotate(EN[r.modelo], (r.parametros, r.erro), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else "left", va="center", fontsize=11.5, color=TINTA)

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator([0.065, 0.07, 0.08, 0.09, 0.10, 0.11]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.3f}" if v < 0.07 else f"{v:.2f}"))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}k"))
    ax.set_xlim(4.2e4, 9e7)
    ax.set_ylim(d.erro_lo.min() * 0.96, lim_cima * 1.07)
    ax.tick_params(colors=MUDO, labelsize=11)
    ax.set_xlabel("learned parameters (log)", color=TINTA, fontsize=13)
    ax.set_ylabel("1 − macro AUROC, fold 10 (log)", color=TINTA, fontsize=13)
    ax.grid(color=GRADE, lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRADE)
    ax.legend(frameon=False, fontsize=11.5, loc="upper right", borderaxespad=0.4, labelspacing=0.55)
    ax.set_title("Parameters vs error on the sealed test set (PTB-XL, fold 10)",
                 loc="left", color=TINTA, fontsize=14.5, pad=14)
    fig.tight_layout()
    fig.savefig(OUT / "pareto_params_fold10.png", dpi=170, facecolor="white")
    fig.savefig(OUT / "pareto_params_fold10.pdf", facecolor="white")
    print(d[["modelo", "parametros", "erro", "erro_lo", "erro_hi"]].round(4).to_string(index=False))
    print("\nfronteira:", ", ".join(fr.modelo))


if __name__ == "__main__":
    main()
