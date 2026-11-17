"""Features das listas congeladas para um dataset externo (generalizacao).

Entrada: um .npz com
    ecg_id  (N,)            identificador inteiro do registro
    sinais  (N, 12, 1000)   float32, mV, 100 Hz, derivacoes na ordem de LEADS
preparado por scripts/prepare_external.py. Nada aqui depende de rotulo ou de
divisao: as features sao as mesmas funcoes usadas no PTB-XL, com os mesmos
nomes de coluna, para que as listas de results/confirmatory/frozen_feature_lists.json
se apliquem sem traducao.

Saida: <saida>/<familia>.parquet, uma por familia, coluna ecg_id + features.
tsfresh (baseline so-serie) e opcional com --tsfresh, porque e o passo mais caro.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pycatch22
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from extract_lead_features import amplitude, interderivacao  # noqa: E402
from interpretecg.data import LEADS  # noqa: E402
from interpretecg.view_features import FAMILY_OPERATORS  # noqa: E402
from interpretecg.views.series_to_graph.directed import directed_graph_features  # noqa: E402
from interpretecg.views.series_to_graph.line import line_graph_features  # noqa: E402
from interpretecg.views.series_to_graph.multiplex import multiplex_features  # noqa: E402
from interpretecg.views.series_to_graph.visibility import _hvg_edges, _lphvg_edges  # noqa: E402

GERADORES = {"hvg": lambda x: _hvg_edges(x), "lphvg1": lambda x: _lphvg_edges(x, 1)}
CAMADAS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")
FAMILIAS = (["catch22", "series_amplitude", "series_interlead", "graph_hvg", "graph_lphvg1"]
            + [f"graph_{g}_{t}{s}" for g in GERADORES for t in ("line", "dir") for s in ("", "_mv")]
            + [f"graph_{g}_multiplex" for g in GERADORES])
# Familias descartadas no PTB-XL (filtro de familias exploratorio), extraidas com --extra
# para a poda partir de todas (desvio de 11/nov, results/external/PREREGISTRATION.md).
# Embedding de Takens: o congelado do PTB-XL (folds 1-7), mesmo m e tau por derivacao.
EXTRAS = ["image_spectrogram", "image_scalogram", "image_gasf", "image_gadf", "image_mtf",
          "image_recurrence", "graph_takens", "graph_transition"]
EMBEDDING = {d: {"m": int(v["m"]), "tau": int(v["tau"])} for d, v in json.loads(
    (ROOT / "data/features/takens_embedding.json").read_text())["per_lead"].items()}


def um_registro(ecg_id: int, sinal: np.ndarray) -> dict[str, dict]:
    """sinal: amostras x 12, mV."""
    out = {f: {"ecg_id": int(ecg_id)} for f in FAMILIAS}
    out["series_amplitude"] |= amplitude(sinal)
    out["series_interlead"] |= interderivacao(sinal)
    arestas = {g: {} for g in GERADORES}
    for j, d in enumerate(LEADS):
        x = sinal[:, j].astype(float)
        r = pycatch22.catch22_all(x.tolist(), catch24=False)
        out["catch22"] |= {f"{d}__{k}": float(v) for k, v in zip(r["names"], r["values"])}
        for fam in ("graph_hvg", "graph_lphvg1"):
            out[fam] |= {f"{d}__{fam}__{k}": v for k, v in FAMILY_OPERATORS[fam](x, {}).items()}
        for g, gerar in GERADORES.items():
            e = np.array(list(gerar(x)), dtype=np.int64)
            arestas[g][d] = e
            for t, func in (("line", line_graph_features), ("dir", directed_graph_features)):
                nome = f"graph_{g}_{t}"
                ordinal, mv = func(x, e)
                out[nome] |= {f"{d}__{nome}__{k}": v for k, v in ordinal.items()}
                out[f"{nome}_mv"] |= {f"{d}__{nome}_mv__{k}": v for k, v in mv.items()}
    for g in GERADORES:
        fam = f"graph_{g}_multiplex"
        h = multiplex_features([arestas[g][d] for d in CAMADAS], list(CAMADAS), sinal.shape[0])
        out[fam] |= {f"multiplex__{fam}__{k}": v for k, v in h.items()}
    return out


def um_registro_extra(ecg_id: int, sinal: np.ndarray) -> dict[str, dict]:
    out = {f: {"ecg_id": int(ecg_id)} for f in EXTRAS}
    for j, d in enumerate(LEADS):
        x = sinal[:, j].astype(float)
        for fam in EXTRAS:
            out[fam] |= {f"{d}__{fam}__{k}": v for k, v in FAMILY_OPERATORS[fam](x, EMBEDDING[d]).items()}
    return out


def bloco(ids: np.ndarray, sinais: np.ndarray, extra: bool = False) -> list[dict]:
    f = um_registro_extra if extra else um_registro
    with threadpool_limits(1):
        return [f(i, s.T.astype(np.float64)) for i, s in zip(ids, sinais)]


def extrair_views(ids, sinais, args, familias) -> dict[str, pd.DataFrame]:
    partes = [slice(i, min(i + args.chunk, len(ids))) for i in range(0, len(ids), args.chunk)]
    res = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(bloco)(ids[p], np.asarray(sinais[p]), args.extra) for p in partes)
    linhas = [l for r in res for l in r]
    return {fam: pd.DataFrame([l[fam] for l in linhas]) for fam in familias}


def extrair_tsfresh(ids, sinais, args) -> pd.DataFrame:
    from extract_tsfresh_features import bloco as bloco_ts

    def ts(p):
        with threadpool_limits(1):
            return bloco_ts(ids[p], np.asarray(sinais[p]))
    partes = [slice(i, min(i + 100, len(ids))) for i in range(0, len(ids), 100)]
    return pd.concat(Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(ts)(p) for p in partes), ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--signals", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--chunk", type=int, default=50)
    ap.add_argument("--n-jobs", type=int, default=15)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tsfresh", action="store_true")
    ap.add_argument("--skip-views", action="store_true")
    ap.add_argument("--extra", action="store_true",
                    help="so as familias descartadas no PTB-XL (imagens, RQA, Takens, transicao)")
    ap.add_argument("--shard-size", type=int, default=0,
                    help="grava checkpoints a cada N registros em <saida>/shards e retoma de onde parou")
    args = ap.parse_args()
    familias = EXTRAS if args.extra else FAMILIAS

    dados = np.load(args.signals)
    ids, sinais = dados["ecg_id"], dados["sinais"]
    if args.limit:
        ids, sinais = ids[:args.limit], sinais[:args.limit]
    assert sinais.shape[1:] == (len(LEADS), 1000), sinais.shape
    args.output.mkdir(parents=True, exist_ok=True)
    tam = args.shard_size or len(ids)
    shards = [slice(i, min(i + tam, len(ids))) for i in range(0, len(ids), tam)]
    pasta = args.output / "shards"
    pasta.mkdir(exist_ok=True)
    inicio = time.perf_counter()
    for n, sh in enumerate(shards):
        if not args.skip_views and not all((pasta / f"{fam}.{n:03d}.parquet").exists() for fam in familias):
            t0 = time.perf_counter()
            for fam, q in extrair_views(ids[sh], sinais[sh], args, familias).items():
                q.to_parquet(pasta / f"{fam}.{n:03d}.parquet", index=False)
            print(f"shard {n + 1}/{len(shards)} views: {sh.stop - sh.start} registros, "
                  f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if args.tsfresh and not (pasta / f"tsfresh.{n:03d}.parquet").exists():
            t0 = time.perf_counter()
            extrair_tsfresh(ids[sh], sinais[sh], args).to_parquet(pasta / f"tsfresh.{n:03d}.parquet", index=False)
            print(f"shard {n + 1}/{len(shards)} tsfresh: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    finais = ([] if args.skip_views else list(familias)) + (["tsfresh"] if args.tsfresh else [])
    for fam in finais:
        q = pd.concat([pd.read_parquet(pasta / f"{fam}.{n:03d}.parquet") for n in range(len(shards))],
                      ignore_index=True)
        assert q.ecg_id.is_unique and len(q) == len(ids) and (q.ecg_id.to_numpy() == ids).all(), fam
        q.to_parquet(args.output / f"{fam}.parquet", index=False)
    minutos = round((time.perf_counter() - inicio) / 60, 1)
    (args.output / ("manifest_extra.json" if args.extra else "manifest.json")).write_text(json.dumps({
        "signals": str(args.signals), "records": int(len(ids)), "families": finais,
        "shard_size": args.shard_size, "minutes_this_run": minutos,
        "script": "scripts/extract_external_features.py"}, indent=2))
    print(f"{len(ids):,} registros, {len(finais)} familias, {minutos} min -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
