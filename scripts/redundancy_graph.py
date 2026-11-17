"""Grafo de similaridade r² entre features, e teste de coesão por visão.

Replica o protocolo do PRISM (`experiments/label_propagation_rich_r2_graph.py` e
`label_propagation_multitask_r2.py` em /exp/duducampos/Paintings): os nós são
features individuais, o peso da aresta é o r² par a par acima de um limiar, e a
coesão de uma família é medida escondendo repetidamente um subconjunto
estratificado dos rótulos de família e propagando os restantes pelo grafo. Uma
feature cujo rótulo recuperado é de outra família "migrou" — a família não é
coesa naquele ponto.

Duas diferenças em relação ao PRISM:

1. Aqui há uma hierarquia natural de dois níveis — 11 famílias dentro de 3
   visões (série, imagem, grafo) — então a propagação roda nos dois níveis. A
   seção 5 do protocolo (H1) é uma afirmação sobre o nível de VISÃO; o nível de
   família serve de controle, porque GASF/GADF e HVG/LPHVG são reparametrizações
   e devem ser inseparáveis entre si.
2. O r² é estimado só nos folds 1–7 (treino do regime exploratório). A medida é
   livre de rótulo clínico e portanto não gastaria o fold selado, mas estimá-la
   fora do treino não acrescenta nada e abre uma porta que não precisa existir.

Colunas constantes são excluídas, não zeradas: correlação indefinida viraria nó
isolado, que infla artificialmente a coesão (um nó sem aresta nunca migra).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import pathlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]

# O mount NFS de /exp trava com leituras concorrentes destes parquets (estado
# `D` em nfs_set_open_stateid_locked, sem recuperacao quando o servidor volta).
# INTERPRETECG_FEATURES aponta para uma copia em tmpfs; os resultados continuam
# sendo escritos no projeto, que sao arquivos pequenos e nunca travaram.
FEATURES = pathlib.Path(os.environ.get("INTERPRETECG_FEATURES", str(ROOT / "data/features")))
PTBXL = pathlib.Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))


sys.path.insert(0, str(ROOT / "src"))

from interpretecg.data import PTBXLDataLoader  # noqa: E402

SEED = 42
VIEW_OF = {
    "catch22": "serie",
    "image_spectrogram": "imagem", "image_scalogram": "imagem",
    "image_gasf": "imagem", "image_gadf": "imagem",
    "image_mtf": "imagem", "image_recurrence": "imagem",
    "graph_hvg": "grafo", "graph_lphvg1": "grafo",
    "graph_takens": "grafo", "graph_transition": "grafo",
    "graph_hvg_line": "grafo", "graph_lphvg1_line": "grafo",
    "graph_hvg_line_mv": "grafo", "graph_lphvg1_line_mv": "grafo",
    "graph_hvg_dir": "grafo", "graph_lphvg1_dir": "grafo",
    "graph_hvg_dir_mv": "grafo", "graph_lphvg1_dir_mv": "grafo",
    "graph_hvg_multiplex": "grafo", "graph_lphvg1_multiplex": "grafo",
    "series_amplitude": "serie", "series_interlead": "serie",
}
# Grupo de invariancia MEDIDO por scripts/measure_invariance.py (results/invariance):
# monotonica > afim > so escala > nenhuma (le mV).
INVARIANCE_OF = {
    **{f: "monotonica" for f in ("graph_hvg", "graph_lphvg1", "graph_hvg_line", "graph_lphvg1_line",
                                 "graph_hvg_dir", "graph_lphvg1_dir", "image_mtf", "graph_transition",
                                 "graph_hvg_multiplex", "graph_lphvg1_multiplex")},
    **{f: "afim" for f in ("catch22", "image_gasf", "image_gadf", "image_recurrence", "graph_takens",
                           "series_interlead")},
    **{f: "escala" for f in ("image_spectrogram", "image_scalogram")},
    **{f: "mV" for f in ("series_amplitude", "graph_hvg_line_mv", "graph_lphvg1_line_mv",
                         "graph_hvg_dir_mv", "graph_lphvg1_dir_mv")},
}
SERIE = ("series_amplitude", "series_interlead")      # o placebo fica de fora


def carregar(regime: str) -> tuple[np.ndarray, pd.DataFrame]:
    """Matriz de features dos folds de treino, mais o índice família/visão."""
    loader = PTBXLDataLoader(PTBXL)
    quadros, familias = {}, {}

    catch22 = pd.read_parquet(FEATURES / "catch22_12leads.parquet")
    quadros["catch22"] = catch22.set_index("ecg_id")
    for caminho in sorted((FEATURES / "views").glob("*.parquet")):
        quadros[caminho.stem] = pd.read_parquet(caminho).set_index("ecg_id")
    for nome in SERIE:
        quadros[nome] = pd.read_parquet(FEATURES / "series" / f"{nome}.parquet").set_index("ecg_id")

    comuns = None
    for nome, quadro in quadros.items():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = loader.records
    mascaras = PTBXLDataLoader.split_masks(
        registros.loc[registros["ecg_id"].isin(comuns)], regime
    )
    treino = registros.loc[registros["ecg_id"].isin(comuns)].loc[mascaras["train"], "ecg_id"]
    print(f"{len(quadros)} famílias, {len(comuns):,} registros em comum, "
          f"{len(treino):,} no treino do regime {regime}")

    partes, colunas, familias_col = [], [], []
    for nome, quadro in quadros.items():
        sub = quadro.loc[treino]
        constantes = sub.columns[sub.nunique(dropna=True) <= 1]
        sub = sub.drop(columns=constantes)
        if len(constantes):
            print(f"  {nome:20s} {len(constantes):3d} colunas constantes excluídas")
        partes.append(sub.to_numpy(dtype=np.float32))
        colunas += list(sub.columns)
        familias_col += [nome] * sub.shape[1]

    valores = np.hstack(partes)
    indice = pd.DataFrame({
        "feature": colunas, "family": familias_col,
        "view": [VIEW_OF[f] for f in familias_col],
        "invariance": [INVARIANCE_OF[f] for f in familias_col],
        "feature_index": np.arange(len(colunas)),
    })
    return valores, indice


def matriz_r2(valores: np.ndarray) -> np.ndarray:
    """r² par a par = quadrado da correlação de Pearson, com NaN imputado."""
    x = valores.astype(np.float64)
    medianas = np.nanmedian(x, axis=0)
    faltantes = ~np.isfinite(x)
    x[faltantes] = np.take(medianas, np.where(faltantes)[1])
    desvio = x.std(axis=0)
    padrao = (x - x.mean(axis=0)) / np.where(desvio > 0, desvio, 1.0)
    correlacao = (padrao.T @ padrao) / len(padrao)
    r2 = np.square(np.nan_to_num(correlacao, nan=0.0))
    np.fill_diagonal(r2, 0.0)
    return r2


def razao_intra_inter(r2: np.ndarray, rotulos: np.ndarray, nomes: list[str]) -> pd.DataFrame:
    """Redundância média dentro de cada grupo contra fora dele (H1)."""
    linhas = []
    for i, nome in enumerate(nomes):
        dentro = rotulos == i
        if dentro.sum() < 2:
            continue
        bloco = r2[np.ix_(dentro, dentro)]
        intra = bloco[np.triu_indices(dentro.sum(), k=1)].mean()
        inter = r2[np.ix_(dentro, ~dentro)].mean()
        linhas.append(dict(grupo=nome, n=int(dentro.sum()), intra=intra, inter=inter,
                           razao=intra / max(inter, 1e-12)))
    return pd.DataFrame(linhas).sort_values("razao", ascending=False)


def transicao(r2: np.ndarray, limiar: float) -> tuple[sparse.csr_matrix, int]:
    adjacencia = sparse.csr_matrix(np.where(r2 >= limiar, r2, 0.0))
    graus = np.asarray(adjacencia.sum(axis=1)).ravel()
    inversos = np.divide(1.0, graus, out=np.zeros_like(graus), where=graus > 0)
    return sparse.diags(inversos) @ adjacencia, int(adjacencia.nnz // 2)


def propagar(transicao_m, rotulos, rotulados, n_classes, alpha, max_iter, tol):
    semente = np.zeros((len(rotulos), n_classes))
    semente[rotulados, rotulos[rotulados]] = 1.0
    f = semente.copy()
    for _ in range(max_iter):
        novo = alpha * (transicao_m @ f) + (1 - alpha) * semente
        if np.abs(novo - f).max() < tol:
            f = novo
            break
        f = novo
    soma = f.sum(axis=1, keepdims=True)
    return np.divide(f, soma, out=np.zeros_like(f), where=soma > 0)


def coesao(r2, rotulos, nomes, limiar, repeticoes, fracao, alpha):
    transicao_m, arestas = transicao(r2, limiar)
    n = len(rotulos)
    densidade = 200 * arestas / (n * (n - 1))
    print(f"  grafo: {arestas:,} arestas com r² ≥ {limiar} (densidade {densidade:.2f}%)")

    rng = np.random.default_rng(SEED)
    somas = np.zeros((n, len(nomes)))
    contagens = np.zeros(n, dtype=np.int64)
    for _ in range(repeticoes):
        escondidos = np.zeros(n, dtype=bool)
        for i in range(len(nomes)):
            idx = np.where(rotulos == i)[0]
            k = max(1, int(round(fracao * len(idx))))
            escondidos[rng.choice(idx, size=k, replace=False)] = True
        dist = propagar(transicao_m, rotulos, ~escondidos, len(nomes), alpha, 100, 1e-6)
        somas[escondidos] += dist[escondidos]
        contagens[escondidos] += 1

    visto = contagens > 0
    media = np.zeros_like(somas)
    media[visto] = somas[visto] / contagens[visto, None]
    com_massa = visto & (media.sum(axis=1) > 0)
    previsto = np.full(n, -1, dtype=np.int32)
    previsto[com_massa] = media[com_massa].argmax(axis=1)
    return previsto, com_massa, media, densidade, arestas


def cotovelo_percolacao(r2: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Limiar na transicao de percolacao (o criterio do PRISM,
    attic/experiments/analyze_redundancy_graph_elbow.py): varre o limiar de r² e
    escolhe o ponto de maior |derivada| da fracao de nos na componente gigante.
    Substitui o 0,5 fixo, que nao tinha justificativa estatistica."""
    from scipy.sparse.csgraph import connected_components
    n = len(r2)
    esq, dir_ = np.triu_indices(n, 1)
    pares = r2[esq, dir_]
    limiares = np.concatenate([np.linspace(0.0, 0.1, 21), np.linspace(0.1, 0.6, 51)[1:],
                               np.linspace(0.6, 1.0, 21)[1:]])
    linhas = []
    for tau in limiares:
        m = pares >= tau
        adj = sparse.csr_matrix((np.ones(m.sum()), (esq[m], dir_[m])), shape=(n, n))
        _, rot = connected_components(adj, directed=False)
        linhas.append({"threshold": tau, "mean_degree": 2 * m.sum() / n,
                       "giant_fraction": np.bincount(rot).max() / n})
    curva = pd.DataFrame(linhas)
    inclinacao = np.gradient(curva.giant_fraction, curva.threshold)
    return float(curva.threshold.iloc[np.abs(inclinacao).argmax()]), curva


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--threshold", default="0.5",
                    help="numero, ou 'elbow' para a transicao de percolacao da componente gigante")
    ap.add_argument("--repeats", type=int, default=100)
    ap.add_argument("--holdout-fraction", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--output", type=Path, default=ROOT / "results/redundancy_graph")
    args = ap.parse_args()

    valores, indice = carregar(args.regime)
    print(f"\nmatriz r²: {valores.shape[1]} x {valores.shape[1]}")
    r2 = matriz_r2(valores)
    if args.threshold == "elbow":
        args.threshold, curva = cotovelo_percolacao(r2)
        args.output.mkdir(parents=True, exist_ok=True)
        curva.to_csv(args.output / "percolation_curve.csv", index=False)
        print(f"cotovelo de percolacao: r² >= {args.threshold:.3f}")
        print(curva.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    else:
        args.threshold = float(args.threshold)

    saida = {"regime": args.regime, "n_features": int(valores.shape[1]),
             "n_records_train": int(valores.shape[0]), "threshold": args.threshold,
             "repeats": args.repeats, "alpha": args.alpha}

    for nivel in ("view", "invariance", "family"):
        nomes = sorted(indice[nivel].unique())
        rotulos = indice[nivel].map({n: i for i, n in enumerate(nomes)}).to_numpy(np.int32)

        print(f"\n{'='*66}\nNÍVEL: {nivel}  ({len(nomes)} grupos)\n{'='*66}")
        razoes = razao_intra_inter(r2, rotulos, nomes)
        print(razoes.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

        previsto, visto, _, densidade, arestas = coesao(
            r2, rotulos, nomes, args.threshold, args.repeats,
            args.holdout_fraction, args.alpha)
        acerto = (previsto[visto] == rotulos[visto]).mean()
        isolados = int((~visto).sum())
        print(f"  coesão: {acerto:.1%} dos rótulos escondidos recuperados "
              f"({visto.sum():,} avaliados, {isolados:,} isolados no grafo)")
        print(f"  acaso (maior grupo): {np.bincount(rotulos).max()/len(rotulos):.1%}")

        por_grupo = []
        for i, nome in enumerate(nomes):
            m = visto & (rotulos == i)
            if m.sum():
                por_grupo.append(dict(grupo=nome, n=int(m.sum()),
                                      recuperado=float((previsto[m] == i).mean())))
        print("\n  por grupo:")
        print(pd.DataFrame(por_grupo).to_string(index=False,
              float_format=lambda v: f"{v:.3f}"))

        saida[nivel] = {"razoes": razoes.to_dict("records"), "coesao": float(acerto),
                        "por_grupo": por_grupo, "densidade": densidade,
                        "arestas": arestas, "isolados": isolados}

        indice[f"previsto_{nivel}"] = [nomes[p] if p >= 0 else None for p in previsto]

    args.output.mkdir(parents=True, exist_ok=True)
    indice.to_parquet(args.output / "feature_index.parquet", index=False)
    (args.output / "summary.json").write_text(json.dumps(saida, indent=2) + "\n")
    print(f"\n-> {args.output}/")


if __name__ == "__main__":
    main()
