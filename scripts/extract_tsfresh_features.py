"""tsfresh (EfficientFCParameters) nas 12 derivacoes: a versao de alta dimensao do catch22.

777 features por serie, 9.324 por registro. A configuracao `efficient` e a
`comprehensive` sem as 6 entropias O(n^2), que custam 5,4x o tempo
(medido: 0,11 contra 0,60 s por serie de 1.000 amostras).

Diferenca de natureza em relacao ao catch22: o catch22 z-normaliza cada serie
(invariante a escala); o tsfresh nao, entao carrega amplitude em mV e se
sobrepoe a familia `series_amplitude`.

Sinal lido do cache em mV (data/features/graphs/hvg_edges.npz), na mesma ordem
de derivacoes que `interpretecg.data.LEADS`. Saida em float32:
data/features/tsfresh_12leads.parquet, colunas ecg_id e `<derivacao>__<feature>`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS  # noqa: E402


def bloco(ecg_ids: np.ndarray, sinais: np.ndarray) -> pd.DataFrame:
    warnings.filterwarnings("ignore")
    from tsfresh.feature_extraction import EfficientFCParameters, extract_features
    n, n_deriv, n_amostras = sinais.shape
    longo = pd.DataFrame({"id": np.repeat(np.arange(n * n_deriv), n_amostras),
                          "t": np.tile(np.arange(n_amostras), n * n_deriv),
                          "x": sinais.reshape(-1).astype(np.float64)})
    f = extract_features(longo, column_id="id", column_sort="t",
                         default_fc_parameters=EfficientFCParameters(),
                         n_jobs=0, disable_progressbar=True).sort_index()
    nomes = [c.removeprefix("x__") for c in f.columns]
    v = f.to_numpy(np.float32).reshape(n, n_deriv, len(nomes))
    colunas = [f"{d}__{k}" for d in LEADS for k in nomes]
    saida = pd.DataFrame(v.reshape(n, -1), columns=colunas)
    saida.insert(0, "ecg_id", ecg_ids)
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/tsfresh_12leads.parquet")
    ap.add_argument("--chunk", type=int, default=100)
    ap.add_argument("--n-jobs", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dados = np.load(args.cache)
    ecg_ids, sinais = dados["ecg_id"], dados["sinais"]
    if args.limit:
        ecg_ids, sinais = ecg_ids[:args.limit], sinais[:args.limit]
    partes = [slice(i, i + args.chunk) for i in range(0, len(ecg_ids), args.chunk)]
    print(f"{len(ecg_ids):,} registros em {len(partes)} blocos", flush=True)
    inicio = time.perf_counter()
    quadros = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(bloco)(ecg_ids[p], np.asarray(sinais[p])) for p in partes)
    quadro = pd.concat(quadros, ignore_index=True)
    assert quadro["ecg_id"].is_unique and len(quadro) == len(ecg_ids)
    quadro.to_parquet(args.output, index=False)
    (args.output.with_suffix(".json")).write_text(json.dumps({
        "settings": "tsfresh EfficientFCParameters", "tsfresh": __import__("tsfresh").__version__,
        "records": len(quadro), "columns": quadro.shape[1] - 1,
        "minutes": round((time.perf_counter() - inicio) / 60, 1)}, indent=2) + "\n")
    print(f"{quadro.shape} -> {args.output} ({(time.perf_counter() - inicio) / 60:.1f} min)")


if __name__ == "__main__":
    main()
