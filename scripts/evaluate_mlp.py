"""MLP sobre as features de cada conjunto: há não-linearidade que o boosting não pega?

O boosting mede acessibilidade a um modelo de árvores, que corta eixo a eixo.
Interações suaves entre muitas features (combinações oblíquas, razões entre
derivações) são caras para árvores e baratas para um MLP. Se o MLP supera o
melhor de logística e boosting num conjunto, há estrutura não-linear ali que o
boosting não estava pegando; se empata com a logística, a visão já entrega o
que tem de forma linear.

Mesmo protocolo de `evaluate_views.py`: regime exploratório, folds 1–7 treino,
8 validação, 9 teste; o fold 10 não é lido. Colunas constantes decididas só no
treino. A validação é usada duas vezes — early stopping por época e escolha da
configuração — então `val_auroc` aqui é otimista e não deve ser comparado com o
`val_auroc` dos outros modelos; o teste não é afetado.

Rede neural varia com a semente muito mais que logística ou boosting. A
configuração é escolhida com uma semente; o teste reporta a média e o desvio
de várias sementes dessa configuração, e a AUROC do ensemble (média das
probabilidades), que é o número comparável a um modelo único bem ajustado.

Memória: o conjunto inteiro (≈15 mil × 3,2 mil float32, ≈200 MB) vai para a
GPU uma vez; os modelos têm no máximo ≈4 M parâmetros. O pico fica bem abaixo
de 1 GB, e `--gpu-fraction` limita o processo caso a placa esteja dividida.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_views import (PTBXL, ROOT, SEED, arquivo,  # noqa: E402
                            carregar_quadros, conjuntos)
from interpretecg.data import SPLIT_REGIMES, SUPERCLASSES, PTBXLDataLoader  # noqa: E402

# (camadas ocultas, dropout, dropout na entrada, weight decay). Pequena de
# propósito: 15 mil amostras não sustentam uma busca grande sem que a escolha
# na validação vire ruído. Toda a grade é fortemente regularizada porque a
# primeira versão (lr 1e-3, wd 1e-4, sem dropout na entrada) atingia o melhor
# ponto entre as épocas 6 e 17 no catch22 e melhorava monotonamente com mais
# dropout -- sobreajuste rápido, que tornaria o MLP um rival injustamente fraco.
CONFIGURACOES = [
    ((512, 512), 0.5, 0.0, 1e-2),
    ((512, 512), 0.5, 0.2, 1e-2),
    ((1024, 1024), 0.5, 0.2, 1e-2),
    ((1024, 1024), 0.6, 0.3, 5e-2),
    ((256, 256, 256), 0.5, 0.2, 1e-2),
    ((2048,), 0.5, 0.2, 1e-2),
]
TAXA = 3e-4
LOTE = 256
EPOCAS_MAX = 200
PACIENCIA = 20
# Features handcrafted têm caudas pesadas (dimensões de correlação, expoentes
# de grau). Depois de padronizar, cortar em ±CORTE desvios evita que poucos
# registros dominem os gradientes; logística e boosting não precisam disso.
CORTE = 5.0


def rotulo_config(camadas, dropout, dropout_entrada, decaimento) -> str:
    return (f"mlp {'x'.join(map(str, camadas))} dropout={dropout} "
            f"entrada={dropout_entrada} wd={decaimento:g}")


def rede(entrada: int, camadas, dropout: float, dropout_entrada: float) -> torch.nn.Module:
    blocos, largura = [torch.nn.Dropout(dropout_entrada)], entrada
    for oculta in camadas:
        blocos += [torch.nn.Linear(largura, oculta), torch.nn.BatchNorm1d(oculta),
                   torch.nn.GELU(), torch.nn.Dropout(dropout)]
        largura = oculta
    blocos.append(torch.nn.Linear(largura, len(SUPERCLASSES)))
    return torch.nn.Sequential(*blocos)


@torch.no_grad()
def prever(modelo, x: torch.Tensor) -> np.ndarray:
    modelo.eval()
    saida = torch.cat([torch.sigmoid(modelo(x[i:i + 4096]))
                       for i in range(0, len(x), 4096)])
    return saida.float().cpu().numpy()


def treinar(dados, alvos, configuracao, semente):
    """Treina com early stopping na AUROC macro da validação; devolve o melhor estado."""
    torch.manual_seed(semente)
    gerador = torch.Generator(device=dados["train"].device).manual_seed(semente)
    camadas, dropout, dropout_entrada, decaimento = configuracao
    modelo = rede(dados["train"].shape[1], camadas, dropout,
                  dropout_entrada).to(dados["train"].device)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=TAXA, weight_decay=decaimento)
    perda = torch.nn.BCEWithLogitsLoss()
    y = torch.as_tensor(alvos["train"], dtype=torch.float32, device=dados["train"].device)
    n = len(y)

    melhor, estado, epoca_melhor = -1.0, None, 0
    for epoca in range(EPOCAS_MAX):
        modelo.train()
        ordem = torch.randperm(n, generator=gerador, device=y.device)
        for i in range(0, n - LOTE + 1, LOTE):  # descarta o lote final: BatchNorm com 1 amostra quebra
            idx = ordem[i:i + LOTE]
            otimizador.zero_grad(set_to_none=True)
            perda(modelo(dados["train"][idx]), y[idx]).backward()
            otimizador.step()
        val = roc_auc_score(alvos["validation"], prever(modelo, dados["validation"]),
                            average="macro")
        if val > melhor:
            melhor, epoca_melhor = val, epoca
            estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        elif epoca - epoca_melhor >= PACIENCIA:
            break
    modelo.load_state_dict(estado)
    return modelo, float(melhor), epoca_melhor + 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory")
    ap.add_argument("--output", type=Path, default=ROOT / "results/views_mlp_exploratory")
    ap.add_argument("--baseline", type=Path, default=ROOT / "results/views_exploratory",
                    help="JSONs de evaluate_views.py, para comparar com logística e boosting")
    ap.add_argument("--only", nargs="+", default=None,
                    help="rodar apenas estes conjuntos (nome do arquivo, sem .json)")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--seeds", type=int, default=5,
                    help="sementes da configuração escolhida, para o teste")
    ap.add_argument("--gpu-fraction", type=float, default=0.5)
    args = ap.parse_args()

    if args.regime == "confirmatory":
        sys.exit("O fold 10 esta selado (CLAUDE.md, secao 6.1); use catch22-baseline "
                 "com justificativa para a rodada confirmatoria.")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dispositivo.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.gpu_fraction)
    torch.backends.cudnn.benchmark = False

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
          f"validacao {len(ids['validation']):,} / teste {len(ids['test']):,}  "
          f"[{dispositivo}]\n")

    args.output.mkdir(parents=True, exist_ok=True)
    todos = conjuntos()
    selecionados = {n: f for n, f in todos.items()
                    if (args.only is None or arquivo(n) in args.only)
                    and not (args.skip_done
                             and (args.output / f"{arquivo(n)}.json").exists())}
    print(f"rodando {len(selecionados)} de {len(todos)} conjuntos\n")

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
        brutos = {n: np.hstack(partes[n]) for n in ids}
        imputador = SimpleImputer(strategy="median").fit(brutos["train"])
        escala = StandardScaler().fit(imputador.transform(brutos["train"]))
        dados = {n: torch.as_tensor(
                     np.clip(escala.transform(imputador.transform(x)), -CORTE, CORTE),
                     dtype=torch.float32, device=dispositivo)
                 for n, x in brutos.items()}

        inicio = time.perf_counter()
        registro = []
        for configuracao in CONFIGURACOES:
            _, val, epocas = treinar(dados, alvos, configuracao, SEED)
            registro.append({"modelo": rotulo_config(*configuracao),
                             "familia_modelo": "mlp", "val_auroc": val, "epocas": epocas})
        escolha = max(range(len(CONFIGURACOES)), key=lambda i: registro[i]["val_auroc"])
        configuracao = CONFIGURACOES[escolha]
        rotulo = rotulo_config(*configuracao)

        escores, por_semente = [], []
        for k in range(args.seeds):
            modelo, val, epocas = treinar(dados, alvos, configuracao, SEED + k)
            p = prever(modelo, dados["test"])
            escores.append(p)
            por_semente.append({"seed": SEED + k, "val_auroc": val, "epocas": epocas,
                                "test_auroc": float(roc_auc_score(alvos["test"], p,
                                                                  average="macro"))})
            del modelo
        ensemble = np.mean(escores, axis=0)
        teste = float(roc_auc_score(alvos["test"], ensemble, average="macro"))
        ap_macro = float(average_precision_score(alvos["test"], ensemble, average="macro"))
        por_classe = {c: float(roc_auc_score(alvos["test"][:, i], ensemble[:, i]))
                      for i, c in enumerate(SUPERCLASSES)}
        testes = np.array([s["test_auroc"] for s in por_semente])
        decorrido = time.perf_counter() - inicio
        print(f"  {nome:22s} {colunas:5d}f  val {registro[escolha]['val_auroc']:.4f}  "
              f"teste ens {teste:.4f}  sementes {testes.mean():.4f}±{testes.std(ddof=1):.4f}  "
              f"[{rotulo}]  ({decorrido/60:.1f} min)", flush=True)

        (args.output / f"{arquivo(nome)}.json").write_text(json.dumps({
            "task": "PTB-XL diagnostic superclasses", "labels": list(SUPERCLASSES),
            "feature_count": colunas, "families": familias,
            "protocol": {"regime": args.regime,
                         "train_folds": list(range(folds["train"][0], folds["train"][1] + 1)),
                         "validation_fold": folds["validation"], "test_fold": folds["test"],
                         "classifier": f"{rotulo} (ensemble de {args.seeds} sementes)",
                         "preprocessing": f"median imputation, standard scaling, clip ±{CORTE}",
                         "optimizer": {"name": "AdamW", "lr": TAXA, "batch": LOTE,
                                       "max_epochs": EPOCAS_MAX, "patience": PACIENCIA,
                                       "early_stopping": "validation macro-AUROC"},
                         "random_state": SEED},
            "selection_metric": "validation macro-AUROC (semente unica)", "selected_c": None,
            "model_comparison_validation": registro,
            "seeds": por_semente,
            "test_auroc_seed_mean": float(testes.mean()),
            "test_auroc_seed_std": float(testes.std(ddof=1)) if len(testes) > 1 else None,
            "validation_macro_auroc": registro[escolha]["val_auroc"],
            "test": {"macro_auroc": teste, "macro_average_precision": ap_macro,
                     "per_label": {c: {"auroc": v} for c, v in por_classe.items()}},
        }, indent=2) + "\n")
        del dados
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()

    linhas = []
    for nome in todos:
        caminho = args.output / f"{arquivo(nome)}.json"
        if not caminho.exists():
            continue
        d = json.loads(caminho.read_text())
        linha = dict(conjunto=nome, n_features=d["feature_count"],
                     mlp=d["protocol"]["classifier"].split(" (")[0],
                     mlp_teste=d["test"]["macro_auroc"],
                     mlp_sementes=d["test_auroc_seed_mean"],
                     mlp_dp=d["test_auroc_seed_std"])
        base = args.baseline / f"{arquivo(nome)}.json"
        if base.exists():
            # So o teste e comparavel: a validacao do MLP foi usada no early stopping.
            b = json.loads(base.read_text())
            linha.update(rasos_modelo=b["protocol"]["classifier"],
                         rasos_teste=b["test"]["macro_auroc"],
                         ganho_mlp=d["test"]["macro_auroc"] - b["test"]["macro_auroc"])
        linhas.append(linha)
    tabela = pd.DataFrame(linhas).sort_values("mlp_teste", ascending=False)
    tabela.to_csv(args.output / "summary.csv", index=False, float_format="%.4f")
    print(f"\n{'='*92}")
    print(tabela.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n-> {args.output}/")


if __name__ == "__main__":
    main()
