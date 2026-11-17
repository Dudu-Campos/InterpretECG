"""Avaliação exploratória por família e por visão, nos dois classificadores.

A seção 6.1 do protocolo pede os dois: a logística mede quanto de cada visão já
vem LINEARMENTE separável, e o gradient boosting mede quanto a visão torna
acessível a um modelo raso qualquer. A diferença entre os dois, por família, é
ela mesma uma medida — quanto da informação daquela representação está
codificada de forma não-linear.

Regime exploratório: folds 1–7 treino, 8 validação, 9 teste. O fold 10 não é
lido. Hiperparâmetros são escolhidos na validação; o teste é medido uma vez.

Colunas constantes são descartadas usando apenas o TREINO para decidir — uma
coluna constante no treino não carrega informação ajustável, e olhar o teste
para decidir isso seria vazamento, ainda que fraco.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "16")

import argparse
import pathlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

# O mount NFS de /exp trava com leituras concorrentes destes parquets (estado
# `D` em nfs_set_open_stateid_locked, sem recuperacao quando o servidor volta).
# INTERPRETECG_FEATURES aponta para uma copia em tmpfs; os resultados continuam
# sendo escritos no projeto, que sao arquivos pequenos e nunca travaram.
FEATURES = pathlib.Path(os.environ.get("INTERPRETECG_FEATURES", str(ROOT / "data/features")))
PTBXL = pathlib.Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))


sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import SPLIT_REGIMES, SUPERCLASSES, PTBXLDataLoader  # noqa: E402

SEED = 42
IMAGEM = ["image_spectrogram", "image_scalogram", "image_gasf", "image_gadf",
          "image_mtf", "image_recurrence"]
GRAFO = ["graph_hvg", "graph_lphvg1", "graph_takens", "graph_transition"]
SERIE = ["series_amplitude", "series_interlead", "series_placebo"]
# Variantes que preservam a amplitude (data/features/views_abs, constantes de
# treino em absolute_calibration.json). Os visibility graphs nao tem variante:
# sao ordinais, entao a visao de grafo `_abs` os reusa como estao.
IMAGEM_ABS = [f"{f}_abs" for f in IMAGEM]
GRAFO_ABS = ["graph_takens_abs", "graph_transition_abs"]
GRAFO_ABS_VISAO = ["graph_hvg", "graph_lphvg1"] + GRAFO_ABS


def carregar_quadros() -> dict[str, pd.DataFrame]:
    quadros = {"catch22": pd.read_parquet(
        FEATURES / "catch22_12leads.parquet").set_index("ecg_id")}
    for caminho in sorted((FEATURES / "views").glob("*.parquet")):
        quadros[caminho.stem] = pd.read_parquet(caminho).set_index("ecg_id")
    for nome in SERIE:
        quadros[nome] = pd.read_parquet(FEATURES / "series" / f"{nome}.parquet").set_index("ecg_id")
    # A extracao `_abs` grava checkpoints parciais; carregar um deles encolheria
    # em silencio a intersecao de registros de TODOS os conjuntos. So entra completo.
    for caminho in sorted((FEATURES / "views_abs").glob("*.parquet")):
        quadro = pd.read_parquet(caminho).set_index("ecg_id")
        if quadro.index.isin(quadros["catch22"].index).sum() < len(quadros["catch22"]):
            print(f"[ignorado: {caminho.stem} incompleto, {len(quadro):,} registros]")
            continue
        quadros[caminho.stem] = quadro
    return quadros


def conjuntos() -> dict[str, list[str]]:
    saida = {n: [n] for n in ["catch22"] + IMAGEM + GRAFO}
    saida["visao: imagem"] = IMAGEM
    saida["visao: grafo"] = GRAFO
    saida["imagem + grafo"] = IMAGEM + GRAFO
    # A pergunta de complementaridade e "X acrescenta ao catch22", nao
    # "X vence o catch22" -- entao os dois pares com a visao nativa entram
    # explicitamente, em vez de so aparecerem dentro de "todas".
    saida["catch22 + imagem"] = ["catch22"] + IMAGEM
    saida["catch22 + grafo"] = ["catch22"] + GRAFO
    saida["todas as visoes"] = ["catch22"] + IMAGEM + GRAFO
    # Amplitude contra relacoes entre derivacoes (scripts/extract_lead_features.py):
    # todas as familias acima sao invariantes a escala de cada derivacao, entao
    # estes conjuntos separam "falta voltagem" de "falta tratamento multivariado".
    # O placebo tem a dimensao das duas juntas (secao 6.3).
    for nome in SERIE:
        saida[nome] = [nome]
    saida["catch22 + amplitude"] = ["catch22", "series_amplitude"]
    saida["catch22 + interderivacao"] = ["catch22", "series_interlead"]
    saida["catch22 + amplitude + interderivacao"] = ["catch22", "series_amplitude",
                                                     "series_interlead"]
    saida["catch22 + placebo"] = ["catch22", "series_placebo"]
    saida["todas + amplitude + interderivacao"] = (["catch22"] + IMAGEM + GRAFO
                                                   + ["series_amplitude", "series_interlead"])
    saida["todas + placebo"] = ["catch22"] + IMAGEM + GRAFO + ["series_placebo"]
    # Visoes com amplitude: a pergunta e se elas acrescentam por cima da base
    # de serie forte (catch22 + amplitude + interderivacao), o que as visoes
    # invariantes quase nao fazem (+0,009 no boosting, +0,013 no MLP).
    for nome in IMAGEM_ABS + GRAFO_ABS:
        saida[nome] = [nome]
    saida["visao: imagem abs"] = IMAGEM_ABS
    saida["visao: grafo abs"] = GRAFO_ABS_VISAO
    saida["catch22 + visoes abs"] = ["catch22"] + IMAGEM_ABS + GRAFO_ABS_VISAO
    saida["catch22 + amplitude + interderivacao + visoes abs"] = (
        ["catch22", "series_amplitude", "series_interlead"] + IMAGEM_ABS + GRAFO_ABS_VISAO)
    # Com epsilon fixo, 2-10% das redes de recorrencia ficam quase vazias (taxa
    # < 0,001), justamente nos registros de amplitude alta; este conjunto tira
    # as duas leituras da mesma matriz para ver se elas pesam ou atrapalham.
    # Coalizoes-chave do Shapley com LDA (results/shapley_lda_exploratory): as
    # quatro familias fracas (MTF, transicao, RQA, Takens) tinham contribuicao
    # negativa na coalizao cheia; estes conjuntos conferem isso no boosting.
    base = ["catch22", "series_amplitude", "series_interlead"]
    boas = ["image_spectrogram", "image_scalogram", "image_gasf", "image_gadf",
            "graph_hvg", "graph_lphvg1"]
    saida["base + visoes boas"] = base + boas
    saida["amplitude + interderivacao + visoes boas"] = base[1:] + boas
    saida["base + espectro e escalograma"] = base + boas[:2]
    saida["base + gaf"] = base + boas[2:4]
    saida["base + visibilidade"] = base + boas[4:]
    # Line graph das visibility graphs (scripts/extract_line_graph_features.py),
    # que no PRISM somava acima do grafo: parte ordinal (invariante) e parte mV.
    linha = ["graph_hvg_line", "graph_lphvg1_line"]
    linha_mv = ["graph_hvg_line_mv", "graph_lphvg1_line_mv"]
    for nome in linha + linha_mv:
        saida[nome] = [nome]
    saida["visibilidade"] = boas[4:]
    saida["line graph"] = linha
    saida["base + line graph"] = base + linha
    saida["visibilidade + line graph"] = boas[4:] + linha
    saida["visibilidade + line graph + mv"] = boas[4:] + linha + linha_mv
    saida["base + visibilidade + line graph"] = base + boas[4:] + linha
    saida["base + visibilidade + line graph + mv"] = base + boas[4:] + linha + linha_mv
    saida["base + visoes boas + line graph + mv"] = base + boas + linha + linha_mv
    # Leitura direcionada no tempo (scripts/extract_directed_graph_features.py):
    # null controlado -- o mesmo grafo, lido com e sem a seta do tempo.
    direcao = ["graph_hvg_dir", "graph_lphvg1_dir"]
    direcao_mv = ["graph_hvg_dir_mv", "graph_lphvg1_dir_mv"]
    for nome in direcao + direcao_mv:
        saida[nome] = [nome]
    saida["direcionado"] = direcao + direcao_mv
    saida["L(HVG) nao direcionado"] = ["graph_hvg_line", "graph_hvg_line_mv"]
    saida["L(HVG) + direcionado"] = ["graph_hvg_line", "graph_hvg_line_mv",
                                     "graph_hvg_dir", "graph_hvg_dir_mv"]
    saida["visibilidade + line graph + mv + direcionado"] = (boas[4:] + linha + linha_mv
                                                             + direcao + direcao_mv)
    saida["base + direcionado"] = base + direcao + direcao_mv
    saida["base + visibilidade + line graph + mv + direcionado"] = (
        base + boas[4:] + linha + linha_mv + direcao + direcao_mv)
    # VG multiplex (scripts/extract_multiplex_features.py): a primeira leitura
    # ENTRE derivacoes alem das correlacoes de `series_interlead`. Pergunta: a
    # estrutura entre camadas acrescenta ao que o line graph le dentro de cada uma?
    multiplex = ["graph_hvg_multiplex", "graph_lphvg1_multiplex"]
    for nome in multiplex:
        saida[nome] = [nome]
    saida["multiplex"] = multiplex
    saida["catch22 + amplitude + multiplex"] = base[:2] + multiplex     # multiplex no lugar da interderivacao
    saida["base + multiplex"] = base + multiplex
    saida["base + direcionado + multiplex"] = base + direcao + direcao_mv + multiplex
    saida["base + visibilidade + line graph + mv + direcionado + multiplex"] = (
        base + boas[4:] + linha + linha_mv + direcao + direcao_mv + multiplex)
    # Todas as 23 familias invariantes e mV juntas (sem as `_abs`, que sao
    # variantes de controle, e sem o placebo).
    saida["todas as familias"] = (["catch22"] + IMAGEM + GRAFO + base[1:] + linha + linha_mv
                                  + direcao + direcao_mv + multiplex)
    for fraca in ("image_mtf", "graph_transition", "image_recurrence", "graph_takens"):
        saida[f"todas + amplitude + interderivacao sem {fraca}"] = [
            f for f in saida["todas + amplitude + interderivacao"] if f != fraca]
    sem_recorrencia = [f for f in IMAGEM_ABS + GRAFO_ABS_VISAO
                       if f not in ("image_recurrence_abs", "graph_takens_abs")]
    saida["catch22 + amplitude + interderivacao + visoes abs sem recorrencia"] = (
        ["catch22", "series_amplitude", "series_interlead"] + sem_recorrencia)
    return saida


class BoostingPorRotulo:
    """Um HistGradientBoosting por rotulo, com early stopping no fold de VALIDACAO.

    O `OneVsRestClassifier` nao repassa `X_val` ao `fit`, entao o boosting da
    grade padrao separa 10% aleatorios do TREINO para parar -- treina com 90% e
    para com dados que nao sao os de selecao. O MLP (`evaluate_mlp.py`) para no
    fold 8 e treina com 100%; esta classe da ao boosting o mesmo regime, para
    que a comparacao entre os dois nao favoreça o MLP por protocolo.
    """

    def __init__(self, imputador=True, semente=SEED, **params):
        self.params = params
        self.semente = semente
        self.imputador = SimpleImputer(strategy="median") if imputador else None

    def fit(self, x, y, x_val, y_val):
        if self.imputador is not None:
            x = self.imputador.fit_transform(x)
            x_val = self.imputador.transform(x_val)
        self.modelos = [HistGradientBoostingClassifier(
            early_stopping=True, n_iter_no_change=30, random_state=self.semente,
            **self.params).fit(x, y[:, i], X_val=x_val, y_val=y_val[:, i])
            for i in range(y.shape[1])]
        return self

    def predict_proba(self, x):
        if self.imputador is not None:
            x = self.imputador.transform(x)
        return np.column_stack([m.predict_proba(x)[:, 1] for m in self.modelos])


def grade_boosting_ampla() -> list[dict]:
    """As duas configuracoes da grade padrao mais 10 sorteadas, com semente fixa."""
    espaco = dict(learning_rate=(0.03, 0.06, 0.1), max_leaf_nodes=(15, 31, 63, 127),
                  min_samples_leaf=(20, 80), l2_regularization=(0.0, 1.0),
                  max_features=(1.0, 0.5))
    base = dict(min_samples_leaf=20, l2_regularization=0.0, max_features=1.0)
    grade = [dict(learning_rate=0.1, max_leaf_nodes=31, **base),
             dict(learning_rate=0.05, max_leaf_nodes=63, **base)]
    rng = np.random.default_rng(SEED)
    while len(grade) < 12:
        sorteio = {k: v[rng.integers(len(v))] for k, v in espaco.items()}
        sorteio = {k: (v.item() if hasattr(v, "item") else v) for k, v in sorteio.items()}
        if sorteio not in grade:
            grade.append(sorteio)
    return grade


def candidatos(n_features: int, grade: str = "padrao", fixa: dict | None = None):
    if grade == "boosting":
        # So o boosting fixo: para comparar coalizoes grandes sem as cinco
        # logisticas, que levam ~70 min cada com ~3.000 features.
        grade, so_boosting = "fixa", True
    else:
        so_boosting = False
    if grade in ("ampla", "fixa"):
        for c in (0.001, 0.01, 0.1, 1.0, 10.0) if not so_boosting else ():
            yield (f"logistica C={c}", "linear", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", OneVsRestClassifier(LogisticRegression(
                    C=c, class_weight="balanced", max_iter=5000,
                    random_state=SEED, solver="liblinear"), n_jobs=5))]))
        for p in ([fixa] if grade == "fixa" else grade_boosting_ampla()):
            rotulo = (f"boosting lr={p['learning_rate']} folhas={p['max_leaf_nodes']} "
                      f"msl={p['min_samples_leaf']} l2={p['l2_regularization']} "
                      f"mf={p['max_features']}")
            yield (rotulo, "nao-linear", BoostingPorRotulo(max_iter=2000, **p))
        return
    for c in (0.01, 0.1, 1.0):
        yield (f"logistica C={c}", "linear", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", OneVsRestClassifier(LogisticRegression(
                C=c, class_weight="balanced", max_iter=3000,
                random_state=SEED, solver="liblinear"), n_jobs=5))]))
    for taxa, folhas in ((0.1, 31), (0.05, 63)):
        yield (f"boosting lr={taxa} folhas={folhas}", "nao-linear", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", OneVsRestClassifier(HistGradientBoostingClassifier(
                learning_rate=taxa, max_leaf_nodes=folhas, max_iter=400,
                early_stopping=True, validation_fraction=0.1,
                random_state=SEED), n_jobs=1))]))


def arquivo(nome: str) -> str:
    return nome.replace(": ", "_").replace(" + ", "_").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--output", type=Path, default=ROOT / "results/views_exploratory")
    ap.add_argument("--only", nargs="+", default=None,
                    help="rodar apenas estes conjuntos (nome do arquivo, sem .json)")
    ap.add_argument("--skip-done", action="store_true",
                    help="pular conjuntos cujo JSON ja existe")
    ap.add_argument("--grade", choices=("padrao", "ampla", "fixa", "boosting"), default="padrao",
                    help="ampla: C de 1e-3 a 10 e 12 configuracoes de boosting com "
                         "early stopping no fold de validacao; fixa: a mesma logistica "
                         "e uma unica configuracao de boosting, dada em --boosting")
    ap.add_argument("--boosting", type=json.loads, default=None,
                    help='para --grade fixa, ex. \'{"learning_rate": 0.03, "max_leaf_nodes": 127, '
                         '"min_samples_leaf": 80, "l2_regularization": 0.0, "max_features": 0.5}\'')
    args = ap.parse_args()
    if (args.grade in ("fixa", "boosting")) != (args.boosting is not None):
        ap.error("--boosting e obrigatorio com --grade fixa, e so vale com ela")

    loader = PTBXLDataLoader(PTBXL)
    quadros = carregar_quadros()

    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = loader.records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    mascaras = PTBXLDataLoader.split_masks(registros, args.regime)
    ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
    alvos = {n: registros.loc[m, list(SUPERCLASSES)].to_numpy(np.int8)
             for n, m in mascaras.items()}
    folds = SPLIT_REGIMES[args.regime]
    print(f"{len(comuns):,} registros; treino {len(ids['train']):,} / "
          f"validacao {len(ids['validation']):,} / teste {len(ids['test']):,}\n")

    args.output.mkdir(parents=True, exist_ok=True)
    todos = conjuntos()
    # NAO chamar isto de `alvos`: esse nome ja e o dicionario de rotulos.
    selecionados = {n: f for n, f in todos.items()
                    if (args.only is None or arquivo(n) in args.only)
                    and not (args.skip_done
                             and (args.output / f"{arquivo(n)}.json").exists())}
    print(f"rodando {len(selecionados)} de {len(todos)} conjuntos: "
          f"{', '.join(selecionados)}\n")
    linhas = []
    for nome, familias in selecionados.items():
        partes = {n: [] for n in ids}
        colunas = 0
        for familia in familias:
            quadro = quadros[familia]
            treino = quadro.loc[ids["train"]]
            uteis = treino.columns[treino.nunique(dropna=True) > 1]
            colunas += len(uteis)
            for n in ids:
                partes[n].append(quadro.loc[ids[n], uteis].to_numpy(np.float32))
        dados = {n: np.hstack(partes[n]) for n in ids}

        melhor, registro = None, []
        inicio = time.perf_counter()
        for rotulo, tipo, modelo in candidatos(colunas, args.grade, args.boosting):
            if isinstance(modelo, BoostingPorRotulo):
                modelo.fit(dados["train"], alvos["train"],
                           dados["validation"], alvos["validation"])
            else:
                modelo.fit(dados["train"], alvos["train"])
            val = roc_auc_score(alvos["validation"],
                                modelo.predict_proba(dados["validation"]),
                                average="macro")
            registro.append({"modelo": rotulo, "familia_modelo": tipo,
                             "val_auroc": float(val)})
            if melhor is None or val > melhor[0]:
                melhor = (val, rotulo, tipo, modelo)
        val, rotulo, tipo, modelo = melhor
        escores = modelo.predict_proba(dados["test"])
        # Escores do teste para bootstrap pareado entre conjuntos (mesmos ids).
        np.savez(args.output / f"{arquivo(nome)}_scores.npz", test=escores,
                 ecg_id_test=ids["test"].to_numpy(), y_test=alvos["test"])
        teste = float(roc_auc_score(alvos["test"], escores, average="macro"))
        ap_macro = float(average_precision_score(alvos["test"], escores, average="macro"))
        por_classe = {c: float(roc_auc_score(alvos["test"][:, i], escores[:, i]))
                      for i, c in enumerate(SUPERCLASSES)}

        melhor_linear = max((r for r in registro if r["familia_modelo"] == "linear"),
                            key=lambda r: r["val_auroc"], default=None)
        decorrido = time.perf_counter() - inicio
        linhas.append(dict(conjunto=nome, n_features=colunas, melhor_modelo=rotulo,
                           val_auroc=float(val), test_auroc=teste,
                           test_ap=ap_macro, segundos=decorrido, **por_classe))
        print(f"  {nome:22s} {colunas:5d}f  val {val:.4f}  teste {teste:.4f}  "
              f"[{rotulo}]  ({decorrido/60:.1f} min)")

        (args.output / f"{arquivo(nome)}.json").write_text(json.dumps({
            "task": "PTB-XL diagnostic superclasses", "labels": list(SUPERCLASSES),
            "feature_count": colunas, "families": familias,
            "protocol": {"regime": args.regime,
                         "train_folds": list(range(folds["train"][0], folds["train"][1] + 1)),
                         "validation_fold": folds["validation"], "test_fold": folds["test"],
                         "classifier": rotulo, "preprocessing": "median imputation",
                         "grid": args.grade,
                         **({"fixed_boosting": args.boosting} if args.grade == "fixa" else {}),
                         "random_state": SEED},
            "selection_metric": "validation macro-AUROC", "selected_c": None,
            "model_comparison_validation": registro,
            "best_linear_validation": melhor_linear,
            "validation_macro_auroc": float(val),
            "test": {"macro_auroc": teste, "macro_average_precision": ap_macro,
                     "per_label": {c: {"auroc": v} for c, v in por_classe.items()}},
        }, indent=2) + "\n")

    linhas = []
    for nome in todos:
        caminho = args.output / f"{arquivo(nome)}.json"
        if not caminho.exists():
            continue
        d = json.loads(caminho.read_text())
        comp = d["model_comparison_validation"]
        lin = max((r for r in comp if r["familia_modelo"] == "linear"),
                  key=lambda r: r["val_auroc"], default={"val_auroc": float("nan")})
        linhas.append(dict(
            conjunto=nome, n_features=d["feature_count"],
            melhor_modelo=d["protocol"]["classifier"],
            val_auroc=d["validation_macro_auroc"],
            test_auroc=d["test"]["macro_auroc"],
            test_ap=d["test"]["macro_average_precision"],
            melhor_linear_val=lin["val_auroc"],
            ganho_nao_linear=d["validation_macro_auroc"] - lin["val_auroc"],
            **{c: v["auroc"] for c, v in d["test"]["per_label"].items()}))
    tabela = pd.DataFrame(linhas).sort_values("test_auroc", ascending=False)
    tabela.to_csv(args.output / "summary.csv", index=False, float_format="%.4f")
    print(f"\n{'='*92}")
    print(tabela[["conjunto", "n_features", "melhor_modelo", "val_auroc",
                  "test_auroc"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {args.output}/")


if __name__ == "__main__":
    main()
