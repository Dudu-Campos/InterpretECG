"""Figura ilustrativa: um registro real do PTB-XL (100 Hz), 12 derivações empilhadas.

Registro escolhido sem olhar rótulo além de NORM: o primeiro dos folds 1-7 com
likelihood 100 para NORM e sem nenhuma anotação de qualidade (static_noise,
baseline_drift, burst_noise, electrodes_problems). Mesma escala vertical em
todos os painéis, para que a amplitude seja comparável entre derivações.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

ROOT = Path(__file__).resolve().parents[1]
NOMES = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
MEMBROS, PRECORDIAIS = "#2a78d6", "#eb6834"
TINTA, MUDO, GRADE = "#1f2328", "#6b7280", "#e5e7eb"


def escolher(db: pd.DataFrame) -> int:
    qual = ["static_noise", "baseline_drift", "burst_noise", "electrodes_problems"]
    limpo = db[qual].isna().all(axis=1) & db.strat_fold.between(1, 7)
    norm = db.scp_codes.map(lambda s: ast.literal_eval(s).get("NORM", 0) == 100)
    return int(db[limpo & norm].index[0])


def compacta(t: np.ndarray, sinal: np.ndarray, saida: Path) -> None:
    """Uma coluna IEEE: 12 derivacoes deslocadas num eixo so, mesma escala em mV."""
    # linhas de base com folga: o minimo de uma derivacao fica 0,15 mV acima do maximo da seguinte
    bases = [0.0]
    for k in range(1, 12):
        bases.append(bases[-1] + sinal[:, k - 1].min() - sinal[:, k].max() - 0.15)
    plt.rcParams.update({"font.size": 7})
    fig, ax = plt.subplots(figsize=(3.5, 4.3))
    for k in range(12):
        base = bases[k]
        ax.plot(t, sinal[:, k] + base, color=MEMBROS if k < 6 else PRECORDIAIS, lw=0.6)
        ax.text(-0.15, base, NOMES[k], ha="right", va="center", fontsize=7, color=TINTA, fontweight="bold")
    # barra de escala: 1 s x 1 mV, no canto inferior direito, abaixo de V6
    x0, y0 = t[-1] - 1.0, bases[-1] + sinal[:, 11].min() - 1.25
    ax.plot([x0, x0 + 1], [y0, y0], color=TINTA, lw=1.0, solid_capstyle="butt")
    ax.plot([x0, x0], [y0, y0 + 1], color=TINTA, lw=1.0, solid_capstyle="butt")
    ax.text(x0 + 0.5, y0 - 0.15, "1 s", ha="center", va="top", fontsize=6.5, color=TINTA)
    ax.text(x0 - 0.08, y0 + 0.5, "1 mV", ha="right", va="center", fontsize=6.5, color=TINTA)
    ax.set_xlim(-0.05, t[-1])
    ax.set_ylim(y0 - 0.6, sinal[:, 0].max() + 0.1)
    ax.axis("off")
    fig.subplots_adjust(left=0.07, right=0.99, top=0.995, bottom=0.005)
    saida = saida.with_name(saida.stem + "_coluna.png")
    fig.savefig(saida, dpi=300, facecolor="white")
    fig.savefig(saida.with_suffix(".pdf"), facecolor="white")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ecg-id", type=int, default=None)
    ap.add_argument("--output", type=Path, default=ROOT / "results/figures/ecg_12_derivacoes.png")
    ap.add_argument("--compact", action="store_true",
                    help="largura de uma coluna IEEE (3,5 pol): derivacoes empilhadas num eixo, barra de escala, sem titulo")
    args = ap.parse_args()
    db = pd.read_csv(ROOT / "data/ptb-xl/ptbxl_database.csv", index_col="ecg_id")
    ecg_id = args.ecg_id or escolher(db)
    linha = db.loc[ecg_id]
    sinal, campos = wfdb.rdsamp(str(ROOT / "data/ptb-xl" / Path(linha.filename_lr).name)
                                if (ROOT / "data/ptb-xl" / (Path(linha.filename_lr).name + ".hea")).exists()
                                else str(ROOT / "data/ptb-xl" / linha.filename_lr))
    fs = campos["fs"]
    t = np.arange(sinal.shape[0]) / fs
    lim = np.ceil(np.abs(sinal).max() * 10) / 10

    if args.compact:
        compacta(t, sinal, args.output)
        print(f"ecg_id {ecg_id} -> {args.output} (+ .pdf)")
        return
    fig, eixos = plt.subplots(12, 1, figsize=(11, 13), sharex=True)
    for k, ax in enumerate(eixos):
        cor = MEMBROS if k < 6 else PRECORDIAIS
        ax.plot(t, sinal[:, k], color=cor, lw=1.1)
        ax.set_ylim(-lim, lim)
        ax.set_yticks([-1, 0, 1] if lim >= 1 else [-lim, 0, lim])
        ax.axhline(0, color=GRADE, lw=0.8, zorder=0)
        ax.grid(axis="x", color=GRADE, lw=0.6)
        ax.set_ylabel(NOMES[k], rotation=0, ha="right", va="center", fontsize=11,
                      color=TINTA, fontweight="bold", labelpad=14)
        ax.tick_params(colors=MUDO, labelsize=8)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRADE)
    eixos[-1].set_xlabel("tempo (s)", color=TINTA)
    eixos[-1].set_xlim(0, t[-1])
    eixos[-1].set_xticks(range(11))
    fig.suptitle(f"ECG multivariado: 12 derivações de um registro do PTB-XL",
                 x=0.08, ha="left", fontsize=14, color=TINTA, y=0.985)
    fig.text(0.08, 0.958, f"ecg_id {ecg_id} · NORM · {fs} Hz, 10 s · amplitude em mV, mesma escala em todos os painéis · "
             "azul = membros, laranja = precordiais", fontsize=9.5, color=MUDO)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.945, bottom=0.05, hspace=0.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, facecolor="white")
    print(f"ecg_id {ecg_id} -> {args.output}")


if __name__ == "__main__":
    main()
