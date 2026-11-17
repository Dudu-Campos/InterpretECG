"""Leitura direcionada no tempo da HVG e da LPHVG (e dos seus line graphs), por derivacao.

As familias de visibilidade e de line graph sao invariantes a inversao
temporal (verificado: zero diferencas entre x e x[::-1]); estas leem a seta do
tempo sobre o MESMO grafo (ver `views/series_to_graph/directed.py`). Quatro
familias, gravadas em data/features/views/ so no fim:

- `graph_hvg_dir`, `graph_lphvg1_dir`: ordinais (invariantes a escala);
- `graph_hvg_dir_mv`, `graph_lphvg1_dir_mv`: em mV.
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
from interpretecg.views.series_to_graph.directed import directed_graph_features  # noqa: E402
from interpretecg.views.series_to_graph.visibility import (  # noqa: E402
    _hvg_edges, _lphvg_edges)

PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))
GRAFOS = {"graph_hvg_dir": lambda x: _hvg_edges(x),
          "graph_lphvg1_dir": lambda x: _lphvg_edges(x, 1)}


def extrair(ecg_id: int, caminho: str) -> dict[str, dict]:
    sinal, campos = wfdb.rdsamp(caminho)
    assert tuple(n.upper() for n in campos["sig_name"]) == LEADS, campos["sig_name"]
    saida = {f: {"ecg_id": ecg_id} for g in GRAFOS for f in (g, f"{g}_mv")}
    for j, d in enumerate(LEADS):
        x = sinal[:, j].astype(float)
        for nome, gerar in GRAFOS.items():
            ordinal, mv = directed_graph_features(x, np.array(list(gerar(x))))
            saida[nome] |= {f"{d}__{nome}__{k}": v for k, v in ordinal.items()}
            saida[f"{nome}_mv"] |= {f"{d}__{nome}_mv__{k}": v for k, v in mv.items()}
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/views")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None, help="so para teste")
    args = ap.parse_args()

    loader = PTBXLDataLoader(PTBXL)
    loader.assert_complete()
    registros = loader.records.reset_index(drop=True)
    if args.limit:
        registros = registros.head(args.limit)
    linhas = Parallel(n_jobs=args.n_jobs, batch_size=16, verbose=5)(
        delayed(extrair)(int(i), str(c))
        for i, c in zip(registros["ecg_id"], registros["record_path"]))

    args.output.mkdir(parents=True, exist_ok=True)
    for familia in linhas[0]:
        quadro = pd.DataFrame([l[familia] for l in linhas])
        quadro.to_parquet(args.output / f"{familia}.parquet", index=False)
        (args.output / f"{familia}.parquet.json").write_text(json.dumps({
            "family": familia, "view": "graph", "leads": list(LEADS),
            "feature_count": quadro.shape[1] - 1,
            "features_per_lead": (quadro.shape[1] - 1) // len(LEADS),
            "source": "scripts/extract_directed_graph_features.py",
            "invariant_to_scale": not familia.endswith("_mv"),
            "invariant_to_time_reversal": False,
        }, indent=2) + "\n")
        print(f"{familia}: {quadro.shape[1] - 1} features, {len(quadro):,} registros, "
              f"NaN {int(quadro.isna().sum().sum())}")


if __name__ == "__main__":
    main()
