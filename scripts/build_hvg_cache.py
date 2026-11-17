"""Cache para a GNN no line graph: sinais crus e arestas da HVG por derivacao.

Grava um unico .npz com
    ecg_id   (N,)
    sinais   (N, 12, 1000) float32, mV
    arestas  (E, 2) int16, arestas (i, j) com i < j de todas as HVGs, concatenadas
    offsets  (N * 12 + 1,) int64, a HVG da derivacao d do registro r e
             arestas[offsets[r*12 + d] : offsets[r*12 + d + 1]]

O line graph nao e guardado: a GNN monta as cunhas na GPU a partir das
arestas (scripts/train_line_gnn.py). Nada aqui depende de rotulo ou de fold.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402
from interpretecg.external import EXTERNOS  # noqa: E402
from interpretecg.views.series_to_graph.visibility import _hvg_edges  # noqa: E402
from interpretecg.views.series_to_graph.takens import takens_recurrence_graph  # noqa: E402
from interpretecg.view_features import RECURRENCE_RATE, THEILER_WINDOW  # noqa: E402
import json  # noqa: E402

EMBEDDING = {d: (int(v["m"]), int(v["tau"])) for d, v in json.loads(
    (ROOT / "data/features/takens_embedding.json").read_text())["per_lead"].items()}


def arestas_takens(x: np.ndarray, d: int) -> np.ndarray:
    """Rede de recorrencia com os mesmos parametros de graph_takens / image_recurrence.

    No i = vetor de embedding que comeca na amostra i; arestas (i, j), i < j."""
    m, tau = EMBEDDING[LEADS[d]]
    g = takens_recurrence_graph(x.astype(float), m=m, tau=tau, recurrence_rate=RECURRENCE_RATE,
                                theiler_window=THEILER_WINDOW)
    a = np.array(sorted((min(u, v), max(u, v)) for u, v in g.edges()), dtype=np.int16).reshape(-1, 2)
    return a

PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))


def um(loader, registro) -> tuple[np.ndarray, list[np.ndarray]]:
    sinal = loader.load_signal(registro)[0].astype(np.float32)  # amostras x derivacoes
    assert sinal.shape == (1000, len(LEADS)), sinal.shape
    arestas = [np.array(list(_hvg_edges(sinal[:, d].astype(float))), dtype=np.int16)
               for d in range(len(LEADS))]
    return sinal.T, arestas


def um_sinal(sinal: np.ndarray, grafo: str = "hvg") -> tuple[np.ndarray, list[np.ndarray]]:
    """sinal: 12 x 1000 (formato de data/external)."""
    if grafo == "takens":
        return sinal, [arestas_takens(sinal[d], d) for d in range(len(LEADS))]
    arestas = [np.array(list(_hvg_edges(sinal[d].astype(float))), dtype=np.int16)
               for d in range(len(LEADS))]
    return sinal, arestas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dataset", choices=["ptbxl", *EXTERNOS], default="ptbxl",
                    help="externos: sinais de data/external/<dataset>/signals.npz")
    ap.add_argument("--graph", choices=["hvg", "takens"], default="hvg",
                    help="takens: rede de recorrencia (so para datasets externos)")
    args = ap.parse_args()

    if args.dataset == "ptbxl":
        loader = PTBXLDataLoader(PTBXL)
        loader.assert_complete()
        registros = loader.records.reset_index(drop=True)
        if args.limit:
            registros = registros.head(args.limit)
        ids = registros["ecg_id"].to_numpy()
        saidas = Parallel(n_jobs=args.n_jobs, batch_size=32, verbose=5)(
            delayed(um)(loader, r) for _, r in registros.iterrows())
    else:
        z = np.load(ROOT / "data/external" / args.dataset / "signals.npz")
        ids, todos = z["ecg_id"][:args.limit], z["sinais"][:args.limit]
        saidas = Parallel(n_jobs=args.n_jobs, batch_size=64, verbose=5)(
            delayed(um_sinal)(s, args.graph) for s in todos)

    sinais = np.stack([s for s, _ in saidas])
    listas = [a for _, arestas in saidas for a in arestas]
    offsets = np.zeros(len(listas) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(a) for a in listas])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, ecg_id=ids, sinais=sinais.astype(np.float32),
             arestas=np.concatenate(listas), offsets=offsets)
    tam = np.diff(offsets)
    print(f"{len(ids):,} registros, {offsets[-1]:,} arestas "
          f"(por derivacao: mediana {np.median(tam):.0f}, max {tam.max()}) -> {args.output}")


if __name__ == "__main__":
    main()
