"""Grupo de invariancia de cada familia, medido e nao declarado.

Para uma amostra de derivacoes reais, cada familia e recalculada depois de
tres transformacoes do sinal:

    escala        x -> 2x
    deslocamento  x -> x + 0,5 mV
    monotonica    x -> x + 0,4 x^3   (estritamente crescente, nao afim)

Uma feature e invariante a T se |f(Tx) - f(x)| <= 1e-6 + 1e-4 |f(x)| em pelo
menos 95% das amostras. Por familia, reporta a fracao de features invariantes
a cada T e a mudanca relativa mediana. O grupo da familia e o maior grupo ao
qual (quase) todas as suas features sao invariantes:

    monotonica   invariante a qualquer reparametrizacao crescente da amplitude
    afim         invariante a escala e deslocamento, nao a monotonica
    escala       invariante a escala, nao a deslocamento
    nenhum       depende da amplitude em mV

Familias por registro (amplitude, interderivacao) recebem a transformacao em
todas as derivacoes. As familias `_abs` usam as constantes de treino de
data/features/absolute_calibration.json. Amostra dos folds 1-7, sem rotulo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402

PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))
TRANSFORMACOES = {
    "escala": lambda x: 2.0 * x,
    "deslocamento": lambda x: x + 0.5,
    "monotonica": lambda x: x + 0.4 * x ** 3,
}
RTOL, ATOL, QUORUM = 1e-4, 1e-6, 0.95


def familias_por_derivacao():
    import pycatch22
    from interpretecg.view_features import FAMILY_OPERATORS
    from interpretecg.views.series_to_graph.directed import directed_graph_features
    from interpretecg.views.series_to_graph.line import line_graph_features
    from interpretecg.views.series_to_graph.visibility import _hvg_edges, _lphvg_edges

    ops = {"catch22": lambda x, p: dict(zip(*pycatch22.catch22_all(list(x), catch24=False).values()))}
    ops |= {f: (lambda op: lambda x, p: op(x, p))(op) for f, op in FAMILY_OPERATORS.items()}
    for nome, gerar in (("hvg", _hvg_edges), ("lphvg1", lambda x: _lphvg_edges(x, 1))):
        for tipo, fn in (("line", line_graph_features), ("dir", directed_graph_features)):
            def op(x, p, fn=fn, gerar=gerar, mv=False):
                o, v = fn(x, np.array(list(gerar(x))))
                return v if mv else o
            ops[f"graph_{nome}_{tipo}"] = op
            ops[f"graph_{nome}_{tipo}_mv"] = (lambda op: lambda x, p: op(x, p, mv=True))(op)
    return ops


def familias_por_registro():
    from extract_lead_features import amplitude, interderivacao
    return {"series_amplitude": amplitude, "series_interlead": interderivacao}


def medir(caminho: str, derivacoes: list[str], params: dict) -> list[dict]:
    warnings.filterwarnings("ignore")
    sinal, _ = wfdb.rdsamp(caminho)
    linhas = []
    por_derivacao = familias_por_derivacao()
    for d in derivacoes:
        x = sinal[:, LEADS.index(d)].astype(float)
        for familia, op in por_derivacao.items():
            base = op(x, params[d])
            for t, f in TRANSFORMACOES.items():
                novo = op(f(x), params[d])
                for k, v in base.items():
                    linhas.append((familia, f"{d}__{k}", t, float(v), float(novo[k])))
    for familia, op in familias_por_registro().items():
        base = op(sinal)
        for t, f in TRANSFORMACOES.items():
            novo = op(f(sinal))
            for k, v in base.items():
                linhas.append((familia, k, t, float(v), float(novo[k])))
    return linhas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=int, default=24)
    ap.add_argument("--leads", nargs="+", default=["II", "V2", "V5"])
    ap.add_argument("--n-jobs", type=int, default=24)
    ap.add_argument("--output", type=Path, default=ROOT / "results/invariance")
    args = ap.parse_args()

    calib = json.loads((ROOT / "data/features/absolute_calibration.json").read_text())["per_lead"]
    registros = PTBXLDataLoader(PTBXL).records
    treino = registros.loc[PTBXLDataLoader.split_masks(registros, "exploratory")["train"]]
    amostra = treino.sample(args.records, random_state=42)
    caminhos = amostra["record_path"].astype(str).tolist()
    partes = Parallel(n_jobs=args.n_jobs)(delayed(medir)(c, args.leads, calib) for c in caminhos)
    q = pd.DataFrame([l for p in partes for l in p], columns=["familia", "feature", "T", "antes", "depois"])
    q = q[np.isfinite(q.antes) & np.isfinite(q.depois)]
    q["inv"] = (q.depois - q.antes).abs() <= ATOL + RTOL * q.antes.abs()
    q["rel"] = (q.depois - q.antes).abs() / (q.antes.abs() + 1e-9)

    por_feature = q.groupby(["familia", "feature", "T"]).inv.mean().ge(QUORUM).rename("inv").reset_index()
    tabela = por_feature.pivot_table(index="familia", columns="T", values="inv", aggfunc="mean")
    tabela = tabela[list(TRANSFORMACOES)]
    tabela.columns = [f"inv_{c}" for c in tabela.columns]
    rel = q.groupby(["familia", "T"]).rel.median().unstack()[list(TRANSFORMACOES)]
    tabela = tabela.join(rel.add_prefix("rel_"))

    def grupo(r, limiar=0.9):
        if r.inv_monotonica >= limiar and r.inv_escala >= limiar and r.inv_deslocamento >= limiar:
            return "monotonica"
        if r.inv_escala >= limiar and r.inv_deslocamento >= limiar:
            return "afim"
        if r.inv_escala >= limiar:
            return "escala"
        return "nenhum"
    tabela["grupo"] = tabela.apply(grupo, axis=1)
    tabela = tabela.sort_values(["grupo", "familia"])
    args.output.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(args.output / "invariance.csv")
    (args.output / "invariance.json").write_text(json.dumps({
        "records": args.records, "leads": args.leads, "seed": 42, "rtol": RTOL, "atol": ATOL,
        "quorum": QUORUM, "transforms": {"escala": "2x", "deslocamento": "x+0.5 mV",
                                         "monotonica": "x+0.4x^3"}}, indent=2) + "\n")
    pd.set_option("display.width", 200)
    print(tabela.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
