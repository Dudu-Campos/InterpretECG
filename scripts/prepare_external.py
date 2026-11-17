"""Prepara um dataset externo no formato que extract_external_features.py consome.

Saida em data/external/<dataset>/:
    signals.npz  ecg_id (N,), sinais (N, 12, 1000) float32 em mV, 100 Hz, ordem LEADS
    labels.csv   ecg_id, strat_fold (1-10), rotulos binarios, metadados
    prep.json    o que foi feito e quantos registros cairam em cada filtro

Divisao: 10 folds estratificados pelo rotulo, com semente fixa, separando por
paciente; treino 1-8, validacao 9, teste 10, como no protocolo do PTB-XL.
Nada aqui olha features nem resultados.

chapman  Zheng et al. (2020), 10.646 registros, 1 por paciente, 500 Hz, 10 s, uV.
         Rotulos: os 4 grupos de ritmo do artigo original (SB, AFIB, GSVT, SR).
code15   Ribeiro et al. (2020/2021), partes HDF5 baixadas, 400 Hz. Ficam so os
         exames de 10 s (4.000 amostras validas), 1 exame por paciente (sorteado).
         Rotulos: 1dAVb, RBBB, LBBB, SB, ST, AF (multirrotulo; nenhum = normal).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS  # noqa: E402

SEED = 42
GRUPOS_CHAPMAN = {"SB": "SB", "AFIB": "AFIB", "AF": "AFIB",
                  "SVT": "GSVT", "AT": "GSVT", "SAAWR": "GSVT", "ST": "GSVT", "AVNRT": "GSVT", "AVRT": "GSVT",
                  "SR": "SR", "SA": "SR"}
ROTULOS_CODE = ["1dAVb", "RBBB", "LBBB", "SB", "ST", "AF"]


def folds_estratificados(estrato: pd.Series, grupos: pd.Series, semente: int = SEED) -> pd.Series:
    """Fold 1-10 por grupo (paciente), estratificado pelo estrato do grupo."""
    from sklearn.model_selection import StratifiedGroupKFold
    fold = np.zeros(len(estrato), dtype=int)
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=semente)
    for k, (_, teste) in enumerate(cv.split(np.zeros(len(estrato)), estrato, grupos), start=1):
        fold[teste] = k
    return pd.Series(fold, index=estrato.index)


def ler_chapman(arquivo: Path) -> np.ndarray:
    x = pd.read_csv(arquivo).to_numpy(np.float64) / 1000.0  # uV -> mV
    assert x.shape == (5000, 12), (arquivo, x.shape)
    return resample_poly(x, 1, 5, axis=0).T.astype(np.float32)  # 12 x 1000


def chapman(saida: Path, n_jobs: int) -> dict:
    base = ROOT / "data/chapman"
    d = pd.read_excel(base / "Diagnostics.xlsx")
    d["rotulo"] = d.Rhythm.map(GRUPOS_CHAPMAN)
    assert d.rotulo.notna().all(), d.loc[d.rotulo.isna(), "Rhythm"].unique()
    d["ecg_id"] = np.arange(len(d))
    sinais = np.stack(Parallel(n_jobs=n_jobs, verbose=2)(
        delayed(ler_chapman)(base / "ECGData" / f"{f}.csv") for f in d.FileName))
    nan_reg = np.isnan(sinais).any(axis=(1, 2))
    mortas = (np.nan_to_num(sinais).std(axis=2) == 0)  # registro x derivacao
    sinais = np.nan_to_num(sinais)  # amostras faltantes -> 0; registros contados em prep.json
    rot = pd.get_dummies(d.rotulo).astype(int)[["SB", "AFIB", "GSVT", "SR"]]
    lab = pd.concat([d[["ecg_id", "FileName", "Rhythm", "PatientAge", "Gender"]], rot], axis=1)
    lab["strat_fold"] = folds_estratificados(d.rotulo, d.ecg_id)
    np.savez(saida / "signals.npz", ecg_id=d.ecg_id.to_numpy(), sinais=sinais)
    lab.to_csv(saida / "labels.csv", index=False)
    return {"records": len(d), "labels": list(rot.columns), "label_counts": rot.sum().to_dict(),
            "records_with_nan": int(nan_reg.sum()), "records_with_flat_lead": int(mortas.any(1).sum()),
            "flat_lead_count": int(mortas.sum()), "units": "uV/1000", "fs_in": 500,
            "resample": "scipy.signal.resample_poly(1, 5)"}


def code15(saida: Path, n_jobs: int) -> dict:
    import h5py
    base = ROOT / "data/code15"
    e = pd.read_csv(base / "exams.csv")
    partes = sorted(p.name for p in base.glob("exams_part*.hdf5"))
    e = e[e.trace_file.isin(partes)].copy()
    n_inicial = len(e)
    blocos, ids = [], []
    for p in partes:
        with h5py.File(base / p, "r") as h:
            t = h["tracings"][:]                     # N x 4096 x 12
            ex = h["exam_id"][:]
        valido = np.abs(t).sum(axis=2) > 0           # N x 4096
        primeiro = valido.argmax(1)
        ultimo = 4096 - valido[:, ::-1].argmax(1)
        compr = np.where(valido.any(1), ultimo - primeiro, 0)
        dez = compr >= 3900                          # exames de 10 s (4.000 amostras)
        for k in np.flatnonzero(dez):
            ini = max(0, min(primeiro[k], 4096 - 4000))
            x = t[k, ini:ini + 4000].astype(np.float64)
            blocos.append(resample_poly(x, 1, 4, axis=0).T.astype(np.float32))
            ids.append(int(ex[k]))
        print(p, len(ex), "exames,", int(dez.sum()), "de 10 s", flush=True)
    sinais = np.stack(blocos)
    ids = np.array(ids)
    e = e.set_index("exam_id").loc[ids].reset_index()
    n_dez = len(e)
    # 1 exame por paciente, sorteado com semente fixa
    rng = np.random.default_rng(SEED)
    e["_u"] = rng.random(len(e))
    manter = e.sort_values("_u").drop_duplicates("patient_id").index.sort_values()
    e, sinais = e.loc[manter].reset_index(drop=True), sinais[manter]
    e = e.rename(columns={"exam_id": "ecg_id"})
    rot = e[ROTULOS_CODE].astype(int)
    estrato = rot.astype(str).agg("".join, axis=1)
    raros = estrato.map(estrato.value_counts()) < 10
    estrato[raros] = "raro"
    e["strat_fold"] = folds_estratificados(estrato, e.patient_id)
    lab = pd.concat([e[["ecg_id", "patient_id", "age", "is_male", "trace_file", "strat_fold"]], rot], axis=1)
    np.savez(saida / "signals.npz", ecg_id=e.ecg_id.to_numpy(), sinais=sinais)
    lab.to_csv(saida / "labels.csv", index=False)
    mortas = sinais.std(axis=2) == 0
    return {"parts": partes, "exams_in_parts": n_inicial, "exams_10s": n_dez,
            "records_one_per_patient": len(e), "labels": ROTULOS_CODE,
            "label_counts": rot.sum().to_dict(), "normal_or_other": int((rot.sum(1) == 0).sum()),
            "records_with_flat_lead": int(mortas.any(1).sum()), "fs_in": 400,
            "resample": "scipy.signal.resample_poly(1, 4) sobre as 4.000 amostras validas"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", choices=["chapman", "code15"])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    saida = ROOT / "data/external" / args.dataset
    saida.mkdir(parents=True, exist_ok=True)
    info = {"chapman": chapman, "code15": code15}[args.dataset](saida, args.n_jobs)
    lab = pd.read_csv(saida / "labels.csv")
    info |= {"fold_counts": lab.strat_fold.value_counts().sort_index().to_dict(), "seed": SEED,
             "leads": list(LEADS), "split": "train 1-8 / validation 9 / test 10"}
    (saida / "prep.json").write_text(json.dumps(info, indent=2, default=int))
    print(json.dumps(info, indent=2, default=int))


if __name__ == "__main__":
    main()
