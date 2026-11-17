"""Sinais e rotulos dos datasets externos no mesmo formato do PTB-XL.

Os datasets foram preparados por scripts/prepare_external.py em
data/external/<dataset>/{signals.npz, labels.csv, prep.json}: 12 derivacoes,
1.000 amostras a 100 Hz, mV, folds 1-10 por paciente. O chapman8 compartilha
os sinais do chapman (symlink) e tem 31 registros a menos nos rotulos.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXTERNOS = ("chapman", "chapman8", "code15")


def carregar(dataset: str, raiz: Path = ROOT) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, list[str]]:
    """(ecg_id, sinais N x 12 x 1000 float32, rotulos alinhados, nomes dos rotulos).

    `rotulos` tem `strat_fold` e uma coluna 0/1 por rotulo, na mesma ordem dos
    sinais; registros sem rotulo (so no chapman8) saem dos sinais.
    """
    if dataset not in EXTERNOS:
        raise ValueError(f"dataset externo desconhecido: {dataset}")
    d = raiz / "data/external" / dataset
    nomes = json.load(open(d / "prep.json"))["labels"]
    lab = pd.read_csv(d / "labels.csv").set_index("ecg_id")
    z = np.load(d / "signals.npz")
    ids, sinais = z["ecg_id"], z["sinais"]
    manter = np.isin(ids, lab.index.to_numpy())
    ids, sinais = ids[manter], np.ascontiguousarray(sinais[manter], dtype=np.float32)
    rot = lab.loc[ids, ["strat_fold", *nomes]].reset_index()
    assert len(rot) == len(lab), "registros rotulados sem sinal"
    return ids, sinais, rot, nomes


def mascaras(fold: pd.Series, regime: str) -> dict[str, np.ndarray]:
    """confirmatory: 1-8 / 9 / 10; exploratory: 1-7 / 8 / 9 (igual ao PTB-XL)."""
    if regime == "confirmatory":
        m = {"train": fold.between(1, 8), "validation": fold.eq(9), "test": fold.eq(10)}
    elif regime == "exploratory":
        m = {"train": fold.between(1, 7), "validation": fold.eq(8), "test": fold.eq(9)}
    else:
        raise ValueError(regime)
    return {k: v.to_numpy() for k, v in m.items()}
