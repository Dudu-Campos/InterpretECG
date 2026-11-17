"""Pacote dos 3 datasets para avaliacao de foundation models por terceiros.

Mesmos registros, sinais (100 Hz, mV, 12 derivacoes, 10 s), rotulos e folds que
todos os modelos do projeto usaram. Saida: export/ecg_fm_bundle/ e o .zip.
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, SUPERCLASSES, PTBXLDataLoader  # noqa: E402

OUT = ROOT / "export/ecg_fm_bundle"


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def ptbxl() -> None:
    d = OUT / "ptbxl"; d.mkdir(parents=True, exist_ok=True)
    loader = PTBXLDataLoader(ROOT / "data/ptb-xl")
    loader.assert_complete()
    reg = loader.records.reset_index(drop=True)
    sinais = np.stack([loader.load_signal(r)[0].T.astype(np.float32) for _, r in reg.iterrows()])
    assert sinais.shape[1:] == (12, 1000)
    np.savez(d / "signals.npz", ecg_id=reg.ecg_id.to_numpy(), sinais=sinais)
    reg[["ecg_id", "patient_id", "strat_fold", *SUPERCLASSES]].astype(
        {"patient_id": int}).to_csv(d / "labels.csv", index=False)
    print("ptbxl", sinais.shape)


def externos() -> None:
    for nome, origem, extras in (("chapman", "chapman", ["FileName", "Rhythm"]),
                                 ("code15", "code15", ["patient_id"])):
        d = OUT / nome; d.mkdir(parents=True, exist_ok=True)
        src = ROOT / "data/external" / origem
        shutil.copyfile(src / "signals.npz", d / "signals.npz")
        lab = pd.read_csv(src / "labels.csv")
        rot = pd.read_json(src / "prep.json", typ="series")["labels"]
        arq = "labels_4grupos.csv" if nome == "chapman" else "labels.csv"
        lab[["ecg_id", *extras, "strat_fold", *rot]].to_csv(d / arq, index=False)
        if nome == "chapman":
            lab8 = pd.read_csv(ROOT / "data/external/chapman8/labels.csv")
            rot8 = pd.read_json(ROOT / "data/external/chapman8/prep.json", typ="series")["labels"]
            lab8[["ecg_id", "FileName", "Rhythm", "strat_fold", *rot8]].to_csv(d / "labels_8ritmos.csv", index=False)
        print(nome, "ok")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ptbxl()
    externos()
    linhas = [f"{sha(p)}  {p.relative_to(OUT)}" for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "SHA256SUMS"]
    (OUT / "SHA256SUMS").write_text("\n".join(linhas) + "\n")
    print("\n".join(linhas))
    print("ordem das derivacoes:", LEADS)


if __name__ == "__main__":
    main()
