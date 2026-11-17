"""Serie -> serie: o tacograma (intervalos RR) e seu resumo de variabilidade.

Visao de ritmo do CLAUDE.md, secao 4.1. O sinal de 10 s vira a serie dos
intervalos entre picos R consecutivos (tipicamente 8 a 15 valores). Com tao
poucos pontos, sumarizadores genericos de serie (catch22) nao fazem sentido;
o readout e o conjunto classico de HRV de curto prazo mais medidas de
irregularidade usadas para fibrilacao atrial.

Deteccao: `neurokit2.ecg_peaks` (metodo padrao) sobre a derivacao II limpa
por `ecg_clean`. Testado a 100 Hz contra Pan-Tompkins, Hamilton e Elgendi:
so o padrao nao inventa picos nessa frequencia. A 100 Hz o pico tem
resolucao de 10 ms, grossa para RMSSD e pNN50; cada pico e refinado por
interpolacao parabolica sobre o sinal limpo (resolucao sub-amostra). Picos cujo QRS
(max |sinal| em +-50 ms) fica abaixo de 40% da mediana sao descartados
(falsos na linha de base).
"""
from __future__ import annotations

import numpy as np

LEAD_PADRAO = "II"
RR_MIN_MS, RR_MAX_MS = 250.0, 2500.0  # 240 a 24 bpm: fora disso e erro de deteccao


def r_peaks(x: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """Picos R em amostras (fracionarias) e o sinal limpo usado."""
    import neurokit2 as nk
    limpo = nk.ecg_clean(np.asarray(x, dtype=float), sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(limpo, sampling_rate=fs, method="neurokit")
    picos = np.asarray(info["ECG_R_Peaks"], dtype=int)
    picos = picos[(picos > 0) & (picos < len(limpo) - 1)]
    # picos falsos na linha de base (tipicos perto das bordas): energia do QRS em volta do pico,
    # max |sinal| em +-50 ms (independe da polaridade do QRS), < 40% da mediana dos picos
    if len(picos) >= 3:
        w = max(1, int(round(0.05 * fs)))
        amp = np.array([np.abs(limpo[max(0, p - w):p + w + 1]).max() for p in picos])
        picos = picos[amp >= 0.4 * np.median(amp)]
    a, b, c = limpo[picos - 1], limpo[picos], limpo[picos + 1]
    den = a - 2 * b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return picos + np.clip(delta, -0.5, 0.5), limpo


def rr_series(x: np.ndarray, fs: float = 100.0) -> tuple[np.ndarray, np.ndarray]:
    """(tempos dos batimentos em s, intervalos RR em ms), descartando RR fisiologicamente impossiveis."""
    picos, _ = r_peaks(x, fs)
    t = picos / fs
    rr = np.diff(t) * 1000.0
    ok = (rr >= RR_MIN_MS) & (rr <= RR_MAX_MS)
    return t[1:][ok], rr[ok]


def rr_features(rr: np.ndarray) -> dict[str, float]:
    """HRV de curto prazo e irregularidade sobre a serie RR (ms)."""
    nomes = ["n_rr", "hr_mean", "rr_mean", "rr_median", "rr_min", "rr_max", "rr_range_rel", "sdnn", "cv",
             "rmssd", "nrmssd", "pnn50", "pnn20", "sd1", "sd2", "sd1_sd2", "tpr", "dir_changes"]
    rr = np.asarray(rr, dtype=float)
    if len(rr) < 3:
        return {k: (float(len(rr)) if k == "n_rr" else np.nan) for k in nomes}
    d = np.diff(rr)
    media = rr.mean()
    sdnn = rr.std(ddof=1)
    rmssd = np.sqrt(np.mean(d ** 2))
    sd1 = np.sqrt(0.5) * d.std(ddof=1) if len(d) > 1 else np.nan
    sd2 = np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0.0)) if np.isfinite(sd1) else np.nan
    # pontos de virada: fracao de RR que e maximo ou minimo local (~2/3 para ruido i.i.d., baixo para ritmo regular com tendencia)
    meio = rr[1:-1]
    virada = ((meio > rr[:-2]) & (meio > rr[2:])) | ((meio < rr[:-2]) & (meio < rr[2:]))
    sinal = np.sign(d[d != 0])
    return {
        "n_rr": float(len(rr)), "hr_mean": 60000.0 / media, "rr_mean": media, "rr_median": float(np.median(rr)),
        "rr_min": rr.min(), "rr_max": rr.max(), "rr_range_rel": (rr.max() - rr.min()) / media,
        "sdnn": sdnn, "cv": sdnn / media, "rmssd": rmssd, "nrmssd": rmssd / media,
        "pnn50": float(np.mean(np.abs(d) > 50)), "pnn20": float(np.mean(np.abs(d) > 20)),
        "sd1": sd1, "sd2": sd2, "sd1_sd2": sd1 / sd2 if sd2 and np.isfinite(sd2) and sd2 > 0 else np.nan,
        "tpr": float(virada.mean()) if len(meio) else np.nan,
        "dir_changes": float(np.mean(sinal[1:] != sinal[:-1])) if len(sinal) > 1 else np.nan,
    }
