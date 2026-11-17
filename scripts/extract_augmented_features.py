"""Aumento de dados no SINAL, com re-extracao das features feitas a mao.

So registros de TREINO (folds 1-7) sao aumentados; validacao e teste ficam
intactos. Cada registro de treino ganha K copias; cada copia recebe:

    ganho global      g ~ U(0,9; 1,1), o mesmo em todas as derivacoes (aparelho)
    deriva de base    p = 0,5: senoide de 0,05-0,5 Hz, amplitude ate 0,1 mV
    ruido             p = 0,5: gaussiano, sigma = 0,01 mV

Deriva e ruido sao sorteados em I e II e nas precordiais; III, aVR, aVL e aVF
sao RECALCULADAS pelas relacoes de Einthoven e Goldberger, para que o sinal
aumentado continue sendo um ECG de 12 derivacoes consistente (CLAUDE.md, 6.4).

Fora, de proposito: inversao temporal (a seta do tempo e informativa, ver
`directed.py`), inversao de sinal (polaridade e diagnostica) e mascaramento.

Familias re-extraidas (o conjunto podado `base + direcionado`): catch22,
amplitude, interderivacao e as quatro direcionadas da HVG/LPHVG, com os MESMOS
nomes de coluna dos parquets originais (verificado no inicio).

Saida: data/features/augmented/<tag>/<familia>.parquet, com colunas
ecg_id, copia e as features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pycatch22
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lead_features import amplitude, interderivacao  # noqa: E402
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402
from interpretecg.views.series_to_graph.directed import directed_graph_features  # noqa: E402
from interpretecg.views.series_to_graph.visibility import _hvg_edges, _lphvg_edges  # noqa: E402

FS = 100.0
I, II = LEADS.index("I"), LEADS.index("II")
GRAFOS = {"graph_hvg_dir": lambda x: _hvg_edges(x),
          "graph_lphvg1_dir": lambda x: _lphvg_edges(x, 1)}
ORIGINAIS = {
    "catch22": ROOT / "data/features/catch22_12leads.parquet",
    "series_amplitude": ROOT / "data/features/series/series_amplitude.parquet",
    "series_interlead": ROOT / "data/features/series/series_interlead.parquet",
    **{f: ROOT / f"data/features/views/{f}.parquet"
       for g in GRAFOS for f in (g, f"{g}_mv")},
}


def derivar_membros(sinal: np.ndarray) -> np.ndarray:
    """Recalcula III, aVR, aVL, aVF a partir de I e II (amostras x 12)."""
    s = sinal.copy()
    a, b = s[:, I], s[:, II]
    s[:, LEADS.index("III")] = b - a
    s[:, LEADS.index("AVR")] = -(a + b) / 2
    s[:, LEADS.index("AVL")] = a - b / 2
    s[:, LEADS.index("AVF")] = b - a / 2
    return s


def aumentar(sinal: np.ndarray, rng: np.random.Generator, args) -> tuple[np.ndarray, dict]:
    n, n_deriv = sinal.shape
    t = np.arange(n) / FS
    independentes = [I, II] + [LEADS.index(f"V{k}") for k in range(1, 7)]
    perturbacao = np.zeros_like(sinal)
    params = {"gain": float(rng.uniform(1 - args.gain, 1 + args.gain))}
    if rng.random() < 0.5:
        f = rng.uniform(0.05, 0.5, n_deriv)
        amp = rng.uniform(0, args.wander, n_deriv)
        fase = rng.uniform(0, 2 * np.pi, n_deriv)
        perturbacao += amp * np.sin(2 * np.pi * f * t[:, None] + fase)
        params["wander"] = True
    if rng.random() < 0.5:
        perturbacao += rng.normal(0, args.noise, sinal.shape)
        params["noise"] = True
    saida = sinal.copy()
    saida[:, independentes] += perturbacao[:, independentes]
    saida = derivar_membros(saida) * params["gain"]
    return saida, params


def features(sinal: np.ndarray) -> dict[str, dict]:
    """Todas as familias do conjunto podado para um sinal (amostras x 12, mV)."""
    out = {"catch22": {}, "series_amplitude": amplitude(sinal),
           "series_interlead": interderivacao(sinal)}
    for j, d in enumerate(LEADS):
        r = pycatch22.catch22_all(sinal[:, j].astype(float).tolist(), catch24=False)
        out["catch22"] |= {f"{d}__{k}": float(v) for k, v in zip(r["names"], r["values"])}
    for nome in GRAFOS:
        out[nome], out[f"{nome}_mv"] = {}, {}
    for j, d in enumerate(LEADS):
        x = sinal[:, j].astype(float)
        for nome, gerar in GRAFOS.items():
            ordinal, mv = directed_graph_features(x, np.array(list(gerar(x))))
            out[nome] |= {f"{d}__{nome}__{k}": v for k, v in ordinal.items()}
            out[f"{nome}_mv"] |= {f"{d}__{nome}_mv__{k}": v for k, v in mv.items()}
    return out


def tarefa(ecg_id: int, sinal: np.ndarray, copia: int, args) -> tuple[dict, dict]:
    rng = np.random.default_rng([args.seed, ecg_id, copia])
    aumentado, params = aumentar(sinal.astype(np.float64), rng, args)
    return {f: {"ecg_id": ecg_id, "copia": copia} | v
            for f, v in features(aumentado).items()}, {"ecg_id": ecg_id, "copia": copia, **params}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--tag", default="k2_gain10_wander10_noise1")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--gain", type=float, default=0.10, help="ganho em 1 +- gain")
    ap.add_argument("--wander", type=float, default=0.10, help="amplitude maxima da deriva, mV")
    ap.add_argument("--noise", type=float, default=0.01, help="desvio do ruido, mV")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--check", action="store_true",
                    help="sem aumento, compara com os parquets originais e sai")
    args = ap.parse_args()

    dados = np.load(args.cache)
    ecg_ids, sinais = dados["ecg_id"], dados["sinais"]          # N x 12 x 1000
    registros = PTBXLDataLoader(ROOT / "data/ptb-xl").records.set_index("ecg_id").loc[ecg_ids]
    treino = np.flatnonzero(registros["strat_fold"].between(1, 7).to_numpy())
    if args.limit:
        treino = treino[:args.limit]

    if args.check:
        for k in treino[:3]:
            calc = features(sinais[k].T.astype(np.float64))
            for familia, caminho in ORIGINAIS.items():
                orig = pd.read_parquet(caminho).set_index("ecg_id").loc[ecg_ids[k]]
                assert list(orig.index) == list(calc[familia]), f"colunas diferentes: {familia}"
                v = np.array([calc[familia][c] for c in orig.index], float)
                dif = np.nanmax(np.abs(v - orig.to_numpy(float)) / (np.abs(orig.to_numpy(float)) + 1e-6))
                print(f"registro {ecg_ids[k]} {familia:22s} {len(v)} colunas, dif. relativa max {dif:.2e}")
        return

    jobs = [(int(ecg_ids[k]), sinais[k].T, c) for k in treino for c in range(1, args.copies + 1)]
    print(f"{len(treino):,} registros de treino x {args.copies} copias = {len(jobs):,} sinais", flush=True)
    saidas = Parallel(n_jobs=args.n_jobs, batch_size=16, verbose=5)(
        delayed(tarefa)(i, s, c, args) for i, s, c in jobs)

    destino = ROOT / "data/features/augmented" / args.tag
    destino.mkdir(parents=True, exist_ok=True)
    for familia in ORIGINAIS:
        pd.DataFrame([s[0][familia] for s in saidas]).to_parquet(destino / f"{familia}.parquet",
                                                                 index=False)
    pd.DataFrame([s[1] for s in saidas]).to_parquet(destino / "params.parquet", index=False)
    (destino / "manifest.json").write_text(json.dumps(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, indent=2) + "\n")
    print(f"-> {destino}")


if __name__ == "__main__":
    main()
