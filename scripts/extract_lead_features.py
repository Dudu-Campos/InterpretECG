"""Duas famílias de série que separam "sem amplitude" de "univariado".

Todas as famílias atuais são invariantes à escala de cada derivação: o catch22
z-normaliza (medido: 0 de 22 features mudam sob x -> 3x + 0,5), a GAF reescala
por mín-máx, a MTF e o grafo de transição usam quantis, o Haralick quantiza por
igual probabilidade, os visibility graphs dependem só da ordem dos valores e a
recorrência tem taxa fixa. Então o modelo não vê voltagem, e os critérios de
hipertrofia são de voltagem somada entre derivações (Sokolow-Lyon, Cornell).

Duas hipóteses para a distância até a CNN, que estas famílias separam:

- `series_amplitude` -- univariada, SEM normalização: média, desvio,
  pico a pico e percentis 5 e 95 por derivação, em mV (60 features).
- `series_interlead` -- multivariada e invariante a uma escala comum a todas as
  derivações: correlações entre as 8 derivações independentes (28), fração de
  variância das 3 primeiras componentes principais (3) e o módulo das cargas da
  primeira componente nas 8 derivações (8), que é a direção do dipolo (39).

E um placebo da mesma dimensão das duas juntas (`series_placebo`, 99 colunas
gaussianas, semente fixa), para o controle da seção 6.3.

As 4 derivações aritméticas (III, aVR, aVL, aVF) ficam fora da família
multivariada: são combinações lineares exatas de I e II, então só repetiriam
correlações já presentes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402

PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))
SEED = 42
INDEPENDENTES = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")


def amplitude(sinal: np.ndarray) -> dict[str, float]:
    saida = {}
    for j, d in enumerate(LEADS):
        x = sinal[:, j]
        p5, p95 = np.percentile(x, [5, 95])
        saida |= {f"{d}_mean": x.mean(), f"{d}_std": x.std(), f"{d}_ptp": np.ptp(x),
                  f"{d}_p05": p5, f"{d}_p95": p95}
    return saida


def interderivacao(sinal: np.ndarray) -> dict[str, float]:
    x = sinal[:, [LEADS.index(d) for d in INDEPENDENTES]]
    x = x - x.mean(0)
    saida = {}
    desvio = x.std(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(x, rowvar=False)
    for a in range(len(INDEPENDENTES)):
        for b in range(a + 1, len(INDEPENDENTES)):
            # Derivacao morta (V5 do registro 12722) da desvio zero: NaN, imputado depois.
            saida[f"corr_{INDEPENDENTES[a]}_{INDEPENDENTES[b]}"] = (
                corr[a, b] if desvio[a] > 0 and desvio[b] > 0 else np.nan)
    _, valores, vt = np.linalg.svd(x, full_matrices=False)
    energia = valores ** 2 / (valores ** 2).sum()
    for k in range(3):
        saida[f"pca_var_{k + 1}"] = energia[k]
    for j, d in enumerate(INDEPENDENTES):
        saida[f"pc1_load_{d}"] = abs(vt[0, j])
    return saida


def extrair(ecg_id: int, caminho: str) -> dict[str, float]:
    sinal, campos = wfdb.rdsamp(caminho)
    assert tuple(n.upper() for n in campos["sig_name"]) == LEADS, campos["sig_name"]
    return {"ecg_id": ecg_id} | amplitude(sinal) | interderivacao(sinal)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/series")
    ap.add_argument("--n-jobs", type=int, default=16)
    args = ap.parse_args()

    loader = PTBXLDataLoader(PTBXL)
    loader.assert_complete()
    registros = loader.records.reset_index(drop=True)
    tabela = pd.DataFrame(Parallel(n_jobs=args.n_jobs, batch_size=64)(
        delayed(extrair)(int(i), str(c))
        for i, c in zip(registros["ecg_id"], registros["record_path"])))

    args.output.mkdir(parents=True, exist_ok=True)
    colunas_amp = [c for c in tabela if c.split("_")[-1] in {"mean", "std", "ptp", "p05", "p95"}]
    colunas_int = [c for c in tabela if c.startswith(("corr_", "pca_var_", "pc1_load_"))]
    rng = np.random.default_rng(SEED)
    placebo = pd.DataFrame(rng.standard_normal((len(tabela), len(colunas_amp) + len(colunas_int))),
                           columns=[f"placebo_{i:03d}" for i in range(len(colunas_amp) + len(colunas_int))])
    placebo.insert(0, "ecg_id", tabela["ecg_id"])
    familias = {
        "series_amplitude": (tabela[["ecg_id", *colunas_amp]], "por derivacao, em mV, sem normalizacao"),
        "series_interlead": (tabela[["ecg_id", *colunas_int]],
                             "8 derivacoes independentes; invariante a escala comum"),
        "series_placebo": (placebo, f"gaussiana padrao, semente {SEED}, mesma dimensao das duas"),
    }
    for nome, (quadro, nota) in familias.items():
        quadro.to_parquet(args.output / f"{nome}.parquet", index=False)
        (args.output / f"{nome}.parquet.json").write_text(json.dumps({
            "family": nome, "view": "series", "feature_count": quadro.shape[1] - 1,
            "records": len(quadro), "note": nota,
            "leads": list(INDEPENDENTES if nome == "series_interlead" else LEADS),
            "seed": SEED}, indent=2) + "\n")
        print(f"{nome}: {quadro.shape[1] - 1} features, {len(quadro):,} registros, "
              f"NaN {int(quadro.isna().sum().sum())}")


if __name__ == "__main__":
    main()
