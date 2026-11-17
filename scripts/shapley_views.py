"""Shapley exato das visoes, com LDA de shrinkage como leitor (GPU).

`todas as visoes` contra a base mede a soma das contribuicoes, e uma familia
negativa pode esconder o ganho de outra. O Shapley divide o credito: o valor de
cada jogador e a media do seu ganho marginal sobre todas as coalizoes.

Jogadores (9, logo 512 coalizoes): reparametrizacoes de uma mesma leitura
ficam juntas, para nao repartir credito entre variantes (secao 6.3).

Valor de uma coalizao: AUROC macro na VALIDACAO (fold 8), porque escolher
visoes e decisao de metodo; o fold 9 e calculado so como descricao. v(vazio)
= 0,5.

Leitor: LDA binaria por rotulo, covariancia intraclasse com shrinkage de
Ledoit-Wolf (a mesma formula do sklearn). E linear, entao mede acessibilidade
LINEAR (secao 6.1): a amplitude, que entra de forma nao-linear, tende a ser
subvalorizada. O boosting e o leitor que vale; este e o primeiro corte.

A covariancia de todas as features e calculada uma vez por rotulo; cada
coalizao recorta a submatriz e recalcula so o shrinkage, que depende de
somas por jogador ja pre-calculadas.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_views import PTBXL, ROOT, carregar_quadros  # noqa: E402
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

CORTE = 5.0
JOGADORES = {
    "catch22": ["catch22"],
    "amplitude+inter": ["series_amplitude", "series_interlead"],
    "espectro+escalograma": ["image_spectrogram", "image_scalogram"],
    "gaf": ["image_gasf", "image_gadf"],
    "mtf": ["image_mtf"],
    "rqa": ["image_recurrence"],
    "visibilidade": ["graph_hvg", "graph_lphvg1"],
    "takens": ["graph_takens"],
    "transicao": ["graph_transition"],
}
# Segunda rodada: as duas leituras em mV que vieram depois, como jogadores
# proprios. As partes ordinais do line graph e do direcionado ficam de fora
# (importancia <= 0,002 cada) para manter 11 jogadores, 2.048 coalizoes.
JOGADORES_LG = {
    **JOGADORES,
    "line graph mV": ["graph_hvg_line_mv", "graph_lphvg1_line_mv"],
    "direcionado mV": ["graph_hvg_dir_mv", "graph_lphvg1_dir_mv"],
}
# Terceira: amplitude (depende de mV) e interderivacao (afim: correlacoes e
# PCA) separadas, porque juntas formam um jogador de invariancia mista que
# decide sozinho o teste de agrupamento por invariancia.
JOGADORES_LG_SEP = {
    ("amplitude" if k == "amplitude+inter" else k): (["series_amplitude"] if k == "amplitude+inter" else v)
    for k, v in JOGADORES_LG.items()
}
JOGADORES_LG_SEP = {**{k: JOGADORES_LG_SEP[k] for k in list(JOGADORES_LG_SEP)[:2]},
                    "interderivacao": ["series_interlead"],
                    **{k: JOGADORES_LG_SEP[k] for k in list(JOGADORES_LG_SEP)[2:]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--players", choices=["views", "views+lg", "views+lg-sep"], default="views",
                    help="views: os 9 originais; views+lg: + line graph mV e direcionado mV")
    ap.add_argument("--pca", type=int, default=None,
                    help="reduz cada jogador a K componentes (PCA no treino) para igualar a "
                         "dimensao: controle do custo de estimacao, que sozinho gera interacao negativa")
    ap.add_argument("--placebo", action="store_true",
                    help="acrescenta um jogador de ruido gaussiano (series_placebo): suas interacoes "
                         "medem o custo de estimacao sem nenhuma informacao compartilhada")
    args = ap.parse_args()
    global JOGADORES
    if args.players == "views+lg":
        JOGADORES = dict(JOGADORES_LG)
    elif args.players == "views+lg-sep":
        JOGADORES = dict(JOGADORES_LG_SEP)
    if args.placebo:
        JOGADORES["placebo"] = ["series_placebo"]
    if args.output is None:
        nome = {"views": "results/shapley_lda_exploratory", "views+lg": "results/shapley_lda_lg",
                "views+lg-sep": "results/shapley_lda_lgsep"}[args.players]
        if args.players != "views":
            nome += f"{'_pca' + str(args.pca) if args.pca else ''}{'_placebo' if args.placebo else ''}_exploratory"
        args.output = ROOT / nome
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    quadros = carregar_quadros()
    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, "exploratory")
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    alvos = {n: registros.loc[m, list(SUPERCLASSES)].to_numpy(np.int8) for n, m in mascaras.items()}

    # Matriz unica, colunas marcadas pelo jogador; imputacao e escala so do treino.
    blocos, dono = {n: [] for n in ids}, []
    for j, (jogador, familias) in enumerate(JOGADORES.items()):
        for familia in familias:
            treino = quadros[familia].loc[ids["train"]]
            uteis = treino.columns[treino.nunique(dropna=True) > 1]
            for n in ids:
                blocos[n].append(quadros[familia].loc[ids[n], uteis].to_numpy(np.float64))
            dono += [j] * len(uteis)
    dono = np.array(dono)
    brutos = {n: np.hstack(b) for n, b in blocos.items()}
    imputador = SimpleImputer(strategy="median").fit(brutos["train"])
    x = {n: imputador.transform(v) for n, v in brutos.items()}
    media, desvio = x["train"].mean(0), x["train"].std(0)
    desvio[desvio == 0] = 1.0
    x = {n: np.clip((v - media) / desvio, -CORTE, CORTE) for n, v in x.items()}
    if args.pca:
        # PCA por jogador ajustada so no treino; os escores sao re-padronizados
        # pelo desvio de treino, para que o shrinkage veja blocos comparaveis.
        reduzidos, novo_dono = {n: [] for n in x}, []
        for j in range(len(JOGADORES)):
            cols = np.flatnonzero(dono == j)
            centro = x["train"][:, cols].mean(0)
            _, _, vt = np.linalg.svd(x["train"][:, cols] - centro, full_matrices=False)
            base = vt[:args.pca].T
            escala = ((x["train"][:, cols] - centro) @ base).std(0)
            for n in x:
                reduzidos[n].append((x[n][:, cols] - centro) @ base / escala)
            novo_dono += [j] * base.shape[1]
        x = {n: np.hstack(v) for n, v in reduzidos.items()}
        dono = np.array(novo_dono)
    x = {n: torch.as_tensor(v, device=dispositivo) for n, v in x.items()}
    n_treino, p = x["train"].shape
    print(f"{p} features em {len(JOGADORES)} jogadores; treino {n_treino:,} [{dispositivo}]")

    # Por rotulo: medias de classe, covariancia intraclasse completa e as somas
    # por jogador de ||x_k||^2 (para o beta do Ledoit-Wolf de qualquer coalizao).
    dono_t = torch.as_tensor(dono, device=dispositivo)
    por_rotulo = []
    for r in range(len(SUPERCLASSES)):
        y = torch.as_tensor(alvos["train"][:, r], device=dispositivo).bool()
        m1, m0 = x["train"][y].mean(0), x["train"][~y].mean(0)
        centrado = torch.where(y[:, None], x["train"] - m1, x["train"] - m0)
        cov = centrado.T @ centrado / n_treino
        quadrados = torch.zeros(n_treino, len(JOGADORES), device=dispositivo, dtype=torch.float64)
        quadrados.index_add_(1, dono_t, centrado ** 2)
        por_rotulo.append((m1 - m0, cov, quadrados))

    def valor(coalizao: tuple[int, ...]) -> dict:
        if not coalizao:
            return {"validation": 0.5, "test": 0.5}
        cols = torch.as_tensor(np.flatnonzero(np.isin(dono, coalizao)), device=dispositivo)
        escores = {n: [] for n in ("validation", "test")}
        for diferenca, cov, quadrados in por_rotulo:
            s = cov[cols][:, cols]
            k = len(cols)
            mu = torch.trace(s) / k
            delta = ((s ** 2).sum() - k * mu ** 2) / k
            normas = quadrados[:, list(coalizao)].sum(1)
            beta = ((normas ** 2).sum() - n_treino * (s ** 2).sum()) / (n_treino ** 2 * k)
            encolhimento = float(torch.clamp(beta, max=delta) / delta) if delta > 0 else 1.0
            sigma = (1 - encolhimento) * s + encolhimento * mu * torch.eye(k, device=dispositivo,
                                                                          dtype=s.dtype)
            w = torch.linalg.solve(sigma, diferenca[cols])
            for n in escores:
                escores[n].append((x[n][:, cols] @ w).cpu().numpy())
        return {n: float(roc_auc_score(alvos[n], np.column_stack(e), average="macro"))
                for n, e in escores.items()}

    jogadores = list(JOGADORES)
    n = len(jogadores)
    inicio = time.perf_counter()
    valores = {}
    for tamanho in range(n + 1):
        for c in itertools.combinations(range(n), tamanho):
            valores[c] = valor(c)
    print(f"{len(valores)} coalizoes em {time.perf_counter() - inicio:.0f} s")

    shapley = {}
    for split in ("validation", "test"):
        phi = {}
        for i in range(n):
            resto = [j for j in range(n) if j != i]
            total = 0.0
            for tamanho in range(n):
                peso = math.factorial(tamanho) * math.factorial(n - tamanho - 1) / math.factorial(n)
                for c in itertools.combinations(resto, tamanho):
                    total += peso * (valores[tuple(sorted(c + (i,)))][split] - valores[c][split])
            phi[jogadores[i]] = total
        shapley[split] = phi
    cheia = tuple(range(n))
    base = tuple(jogadores.index(k) for k in ("catch22", "amplitude+inter", "amplitude", "interderivacao")
                 if k in jogadores)
    tabela = pd.DataFrame({
        "shapley_val": shapley["validation"], "shapley_test": shapley["test"],
        "sozinho_val": {jogadores[i]: valores[(i,)]["validation"] for i in range(n)},
        "loo_val": {jogadores[i]: valores[cheia]["validation"]
                    - valores[tuple(j for j in cheia if j != i)]["validation"] for i in range(n)},
        "add_na_base_val": {jogadores[i]: (valores[tuple(sorted(set(base) | {i}))]["validation"]
                                           - valores[base]["validation"]) for i in range(n)},
    }).sort_values("shapley_val", ascending=False)
    print(tabela.to_string(float_format=lambda v: f"{v:+.4f}" if abs(v) < 0.5 else f"{v:.4f}"))
    print(f"coalizao cheia: val {valores[cheia]['validation']:.4f}  teste {valores[cheia]['test']:.4f}; "
          f"soma dos Shapley (val) {sum(shapley['validation'].values()) + 0.5:.4f}")

    args.output.mkdir(parents=True, exist_ok=True)
    tabela.to_csv(args.output / "shapley.csv")
    pd.DataFrame([{"coalizao": "+".join(jogadores[i] for i in c), "tamanho": len(c), **v}
                  for c, v in valores.items()]).to_csv(args.output / "coalizoes.csv", index=False)
    (args.output / "shapley.json").write_text(json.dumps({
        "reader": "LDA binaria por rotulo, shrinkage Ledoit-Wolf, z-score do treino, corte +-5"
                  + (f"; PCA por jogador, {args.pca} componentes (treino)" if args.pca else ""),
        "value": "AUROC macro; validacao = fold 8 (decisao), teste = fold 9 (descricao)",
        "players": JOGADORES, "shapley": shapley,
        "full": valores[cheia], "base_catch22_amplitude_inter": valores[base],
    }, indent=2) + "\n")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
