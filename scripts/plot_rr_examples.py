"""Exemplo da visao serie -> serie RR: ECG (II) com os picos R e o tacograma, por ritmo.

Registros dos folds 1-7 sem anotacao de ruido, um por rotulo de ritmo do PTB-XL.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402
from interpretecg.views.series_to_series.rr import r_peaks, rr_features, rr_series  # noqa: E402

RITMOS = [("SR", "ritmo sinusal"), ("SARRH", "arritmia sinusal"), ("SBRAD", "bradicardia sinusal"),
          ("STACH", "taquicardia sinusal"), ("AFIB", "fibrilação atrial")]
TINTA, MUDO, GRADE, AZUL, LARANJA = "#1f2328", "#6b7280", "#e5e7eb", "#2a78d6", "#eb6834"


def main() -> None:
    db = pd.read_csv(ROOT / "data/ptb-xl/ptbxl_database.csv", index_col="ecg_id")
    ok = db[["static_noise", "baseline_drift", "burst_noise", "electrodes_problems"]].isna().all(axis=1) \
        & db.strat_fold.between(1, 7)
    loader = PTBXLDataLoader(ROOT / "data/ptb-xl")
    fig, eixos = plt.subplots(len(RITMOS), 2, figsize=(11, 9.5), gridspec_kw={"width_ratios": [3, 1.25]})
    for lin, (cod, nome) in enumerate(RITMOS):
        cods = db.scp_codes.map(ast.literal_eval)
        # ritmo presente e nenhum outro rotulo de ritmo concorrente, nem extrassistoles
        outros = {c for c, _ in RITMOS} - {cod} | {"PVC", "PAC", "AFLT"}
        m = ok & cods.map(lambda d: cod in d and not (set(d) & outros))
        ecg = int(db[m].index[0])
        x = loader.load_signal(ecg)[0][:, LEADS.index("II")]
        picos, _ = r_peaks(x)
        t_rr, rr = rr_series(x)
        f = rr_features(rr)
        t = np.arange(len(x)) / 100
        ax = eixos[lin, 0]
        ax.plot(t, x, color=AZUL, lw=0.9)
        ax.plot(picos / 100, np.interp(picos, np.arange(len(x)), x), "v", color=LARANJA, ms=5,
                label="pico R detectado")
        ax.set_xlim(0, 10)
        ax.set_ylabel(f"{cod}\n(mV)", rotation=0, ha="right", va="center", color=TINTA, labelpad=8)
        ax.text(0.005, 0.97, f"{nome} · ecg_id {ecg} · derivação II", transform=ax.transAxes, va="top",
                fontsize=8, color=MUDO)
        ax2 = eixos[lin, 1]
        ax2.plot(t_rr, rr, "-o", color=TINTA, ms=3.5, lw=1)
        ax2.set_ylim(250, 2000)
        ax2.set_xlim(0, 10)
        ax2.text(0.97, 0.95, f"FC {f['hr_mean']:.0f} bpm\nSDNN {f['sdnn']:.0f} ms\nRMSSD {f['rmssd']:.0f} ms\n"
                 f"pNN50 {f['pnn50']:.2f}", transform=ax2.transAxes, ha="right", va="top", fontsize=7.5,
                 color=MUDO)
        for a in (ax, ax2):
            a.tick_params(colors=MUDO, labelsize=8)
            a.grid(color=GRADE, lw=0.6)
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                a.spines[s].set_color(GRADE)
        if lin == 0:
            ax.set_title("ECG, derivação II, com os picos R", loc="left", color=TINTA, fontsize=10.5)
            ax2.set_title("tacograma: intervalo RR (ms)", loc="left", color=TINTA, fontsize=10.5)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        if lin == len(RITMOS) - 1:
            ax.set_xlabel("tempo (s)", color=TINTA)
            ax2.set_xlabel("tempo do batimento (s)", color=TINTA)
    fig.suptitle("Visão série → série: a série de intervalos RR", x=0.01, ha="left", color=TINTA, fontsize=13)
    fig.text(0.01, 0.945, "picos R por neurokit2 (método padrão) com refinamento parabólico sub-amostra; "
             "tacograma na mesma escala em todas as linhas; registros dos folds de treino, sem anotação de ruído",
             color=MUDO, fontsize=8.5)
    fig.tight_layout(rect=(0.01, 0, 1, 0.94))
    saida = ROOT / "results/figures/rr_exemplos.png"
    fig.savefig(saida, dpi=170, facecolor="white")
    print(saida)


if __name__ == "__main__":
    main()
