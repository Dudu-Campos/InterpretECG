"""Familias `graph_hvg_multiplex` e `graph_lphvg1_multiplex`: VG multiplex nas 8 derivacoes independentes.

Camadas I, II, V1-V6 (as mesmas de `series_interlead`); III, aVR, aVL e aVF
sao aritmetica de I e II (secao 6.4) e ficariam como camadas quase-copias.
Arestas da HVG lidas do cache (data/features/graphs/hvg_edges.npz); as da
LPHVG (rho=1) sao recalculadas dos sinais do mesmo cache. Nada depende de
rotulo ou de fold. Saida em data/features/views/, uma familia por Parquet,
como as demais.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS  # noqa: E402
from interpretecg.views.series_to_graph.multiplex import K_MAX, multiplex_features  # noqa: E402
from interpretecg.views.series_to_graph.visibility import _lphvg_edges  # noqa: E402

CAMADAS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
IDX = [LEADS.index(d) for d in CAMADAS]


def bloco(ids, sinais, hvg) -> list[tuple[dict, dict]]:
    saida = []
    for r in range(len(ids)):
        n = sinais.shape[-1]
        h = multiplex_features(hvg[r], list(CAMADAS), n)
        lp = [np.array(list(_lphvg_edges(sinais[r, d].astype(float), 1)), dtype=np.int64) for d in IDX]
        saida.append(({"ecg_id": int(ids[r]), **h},
                      {"ecg_id": int(ids[r]), **multiplex_features(lp, list(CAMADAS), n)}))
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/views")
    ap.add_argument("--chunk", type=int, default=200)
    ap.add_argument("--n-jobs", type=int, default=48)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    dados = np.load(args.cache)
    ids, offsets = dados["ecg_id"], dados["offsets"]
    if args.limit:
        ids = ids[:args.limit]
    arestas = dados["arestas"]
    sinais = dados["sinais"]
    nl = len(LEADS)

    def tarefa(p):
        hvg = [[arestas[offsets[r * nl + d]:offsets[r * nl + d + 1]] for d in IDX] for r in range(p.start, p.stop)]
        return delayed(bloco)(ids[p], np.asarray(sinais[p]), hvg)

    partes = [slice(i, min(i + args.chunk, len(ids))) for i in range(0, len(ids), args.chunk)]
    inicio = time.perf_counter()
    resultados = Parallel(n_jobs=args.n_jobs, verbose=5)(tarefa(p) for p in partes)
    linhas = [l for r in resultados for l in r]
    minutos = round((time.perf_counter() - inicio) / 60, 1)
    for k, familia in enumerate(("graph_hvg_multiplex", "graph_lphvg1_multiplex")):
        q = pd.DataFrame([l[k] for l in linhas])
        q.columns = ["ecg_id"] + [f"multiplex__{familia}__{c}" for c in q.columns[1:]]
        assert q.ecg_id.is_unique and len(q) == len(ids)
        destino = args.output / f"{familia}.parquet"
        q.astype({c: np.float32 for c in q.columns[1:]}).to_parquet(destino, index=False)
        Path(str(destino) + ".json").write_text(json.dumps({
            "family": familia, "view": "graph", "layers": CAMADAS, "k_max_mi": K_MAX,
            "records": len(q), "feature_count": q.shape[1] - 1, "minutes": minutos,
            "reference": "Lacasa, Nicosia & Latora (2015) Sci. Rep. 5:15508",
            "invariance": "monotonica por derivacao (so a ordem das amostras)"}, indent=2) + "\n")
        print(f"{familia}: {q.shape} -> {destino}")


if __name__ == "__main__":
    main()
