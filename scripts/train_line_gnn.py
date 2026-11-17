"""GNN no line graph da HVG, por derivacao, contra a CNN no mesmo fold 9.

Motivacao: 21 numeros por derivacao que resumem L(HVG) em mV (familia
`graph_hvg_line_mv`) ja chegaram a 0,901 de AUROC com boosting, acima do
catch22. Se uma sumarizacao fixa leva tanto, um modelo que le o grafo inteiro
pode levar mais. A referencia e a xresnet1d101 no regime exploratorio
(fold 9: 0,9287).

Grafo. Para cada derivacao, os nos de L(G) sao as arestas (i, j) da HVG
(i < j). Dois nos se ligam quando as arestas compartilham uma amostra s (uma
cunha). As cunhas sao montadas na GPU a cada lote, a partir do cache de
arestas (scripts/build_hvg_cache.py).

Atributos de no: x_i, x_j, min(x_i, x_j), |x_i - x_j|, log(j - i), (i + j)/2000.
A amplitude e padronizada por media e desvio ESCALARES do treino, como na CNN:
nada e normalizado por amostra, entao a escala em mV fica legivel.
Atributo de mensagem: o valor da amostra compartilhada x_s.

Modelo: L camadas residuais h <- h + MLP([h, media_vizinhos(W h + e_s)]),
pesos compartilhados entre as 12 derivacoes, mais um embedding da derivacao
somado na entrada. Leitura: media e maximo por derivacao, concatenadas nas 12
derivacoes, cabeca MLP para os 5 rotulos.

Variante temporal (--directed): as cunhas sao tipadas pela posicao da amostra
compartilhada s em cada corda, do ponto de vista do no que recebe a mensagem:
leque para frente (s e a ponta esquerda das duas), cadeia para tras (a vizinha
termina em s, onde esta comeca), cadeia para frente (a vizinha comeca em s, onde
esta termina) e leque para tras (s e a ponta direita das duas). Cada relacao tem
o seu W_r e a sua media, no estilo R-GCN: agregado = soma_r media_r(W_r h). Com
--directed desligado as quatro relacoes se fundem numa media so, que e o modelo
original. Atributos, leitura e treino nao mudam: a comparacao isola a direcao da
passagem de mensagens.

Protocolo: regime exploratorio (folds 1-7 / 8 / 9). A epoca reportada e a de
melhor AUROC de validacao (fold 8); o fold 9 e lido uma vez, no fim, com
esse checkpoint. O fold 10 nao e lido.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402
from interpretecg.external import EXTERNOS  # noqa: E402

PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))
N_DERIV = 12
N_AMOSTRAS = 1000
N_ATRIB = 6


# --- grafo -----------------------------------------------------------------

class Cache:
    def __init__(self, caminho: Path):
        dados = np.load(caminho)
        self.ecg_id = dados["ecg_id"]
        self.sinais = torch.from_numpy(dados["sinais"])        # N x 12 x 1000
        self.arestas = torch.from_numpy(dados["arestas"])          # int16; vira int64 na GPU
        self.offsets = dados["offsets"]

    def lote(self, linhas: np.ndarray, dispositivo):
        """Arestas de B x 12 grafos com indice de no global (grafo * 1000 + amostra)."""
        grafos = (linhas[:, None] * N_DERIV + np.arange(N_DERIV)[None]).ravel()
        ini, fim = self.offsets[grafos], self.offsets[grafos + 1]
        tam = fim - ini
        idx = np.concatenate([np.arange(a, b) for a, b in zip(ini, fim)])
        arestas = self.arestas[idx].to(dispositivo, non_blocking=True).long()
        grafo = torch.repeat_interleave(torch.arange(len(grafos), device=dispositivo),
                                        torch.as_tensor(tam, device=dispositivo))
        sinais = self.sinais[linhas].to(dispositivo, non_blocking=True)
        return sinais, arestas, grafo


def cunhas(arestas: torch.Tensor, grafo: torch.Tensor):
    """Pares ordenados (origem, destino) de nos de L(G) que compartilham uma amostra.

    Retorna tambem o indice global da amostra compartilhada e a relacao
    2 * lado_destino + lado_origem, com lado 0 = ponta esquerda e 1 = direita:
    0 leque para frente, 1 cadeia para tras, 2 cadeia para frente, 3 leque para tras.
    """
    n_arestas = len(arestas)
    no = (grafo[:, None] * N_AMOSTRAS + arestas).reshape(-1)        # 2E incidencias
    aresta = torch.arange(n_arestas, device=arestas.device).repeat_interleave(2)
    ordem = torch.argsort(no)
    no, aresta, lado = no[ordem], aresta[ordem], ordem % 2
    _, contagem = torch.unique_consecutive(no, return_counts=True)
    inicio = torch.cumsum(contagem, 0) - contagem
    k = contagem.repeat_interleave(contagem)                         # grau por incidencia
    ini = inicio.repeat_interleave(contagem)
    origem = torch.arange(len(no), device=no.device).repeat_interleave(k)
    deslocamento = torch.arange(len(origem), device=no.device) - torch.repeat_interleave(
        torch.cumsum(k, 0) - k, k)
    destino = ini.repeat_interleave(k) + deslocamento
    manter = origem != destino
    origem, destino = origem[manter], destino[manter]
    return aresta[origem], aresta[destino], no[origem], 2 * lado[destino] + lado[origem]


def atributos(sinais: torch.Tensor, arestas: torch.Tensor, grafo: torch.Tensor,
              media: float, desvio: float):
    x = ((sinais - media) / desvio).reshape(-1)                      # (B*12*1000,)
    base = grafo * N_AMOSTRAS
    xi, xj = x[base + arestas[:, 0]], x[base + arestas[:, 1]]
    vao = (arestas[:, 1] - arestas[:, 0]).float()
    return torch.stack([xi, xj, torch.minimum(xi, xj), (xi - xj).abs(),
                        torch.log(vao), (arestas.sum(1).float() / 2000.0)], 1), x


# --- modelo ----------------------------------------------------------------

class Camada(nn.Module):
    def __init__(self, d: int, dropout: float, relacoes: int = 1):
        super().__init__()
        self.mensagem = nn.ModuleList(nn.Linear(d, d) for _ in range(relacoes))
        self.borda = nn.Linear(relacoes, d)
        self.atualiza = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, 2 * d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d, d))

    def forward(self, h, adjacencias, x_medio):
        # media_vizinhos(W h + w x_s + b) = W (A h) + w media(x_s) + b, com A normalizada
        # por linha: a multiplicacao esparsa evita materializar pares x dimensao.
        # Uma adjacencia por relacao (uma so no modelo nao direcionado).
        with torch.autocast("cuda", enabled=False):
            agregado = sum(torch.sparse.mm(a, w(h).float())
                           for a, w in zip(adjacencias, self.mensagem))
        agregado = agregado + self.borda(x_medio)
        return h + self.atualiza(torch.cat([h, agregado.to(h.dtype)], 1))


class LineGNN(nn.Module):
    def __init__(self, d: int = 64, camadas: int = 4, dropout: float = 0.1, n_saidas: int = 5,
                 direcionado: bool = False):
        super().__init__()
        self.relacoes = 4 if direcionado else 1
        self.entrada = nn.Sequential(nn.Linear(N_ATRIB, d), nn.GELU(), nn.Linear(d, d))
        self.derivacao = nn.Embedding(N_DERIV, d)
        self.camadas = nn.ModuleList(Camada(d, dropout, self.relacoes) for _ in range(camadas))
        self.cabeca = nn.Sequential(nn.LayerNorm(2 * d * N_DERIV), nn.Dropout(0.25),
                                    nn.Linear(2 * d * N_DERIV, 256), nn.GELU(),
                                    nn.Dropout(0.5), nn.Linear(256, n_saidas))
        self.d = d

    def forward(self, sinais, arestas, grafo, media, desvio):
        n_grafos = sinais.shape[0] * N_DERIV
        atrib, x = atributos(sinais, arestas, grafo, media, desvio)
        origem, destino, compartilhado, relacao = cunhas(arestas, grafo)
        if self.relacoes == 1:
            relacao = torch.zeros_like(relacao)
        n = len(arestas)
        adjacencias, x_medio = [], []
        for r in range(self.relacoes):
            m = relacao == r
            o, d = origem[m], destino[m]
            grau = torch.bincount(d, minlength=n).clamp(min=1).float()
            adjacencias.append(torch.sparse_coo_tensor(torch.stack([d, o]), 1.0 / grau[d],
                                                       (n, n)).coalesce())
            x_medio.append(torch.zeros(n, device=x.device).index_add_(0, d, x[compartilhado[m]]) / grau)
        x_medio = torch.stack(x_medio, 1)
        h = self.entrada(atrib) + self.derivacao(grafo % N_DERIV)
        for camada in self.camadas:
            h = camada(h, adjacencias, x_medio)
        n_por_grafo = torch.bincount(grafo, minlength=n_grafos).clamp(min=1).float()
        media_g = torch.zeros(n_grafos, self.d, device=h.device, dtype=h.dtype).index_add_(
            0, grafo, h) / n_por_grafo[:, None]
        max_g = torch.full((n_grafos, self.d), -1e4, device=h.device, dtype=h.dtype).scatter_reduce(
            0, grafo[:, None].expand_as(h), h, reduce="amax", include_self=True)
        leitura = torch.cat([media_g, max_g], 1).reshape(sinais.shape[0], -1)
        return self.cabeca(leitura)


class HVGGNN(nn.Module):
    """Ablacao: a mesma GNN, mas sobre a propria HVG (nos = amostras, arestas = visibilidade).

    Atributos de no: x_t padronizado e t/1000. Mensagens pelas arestas da HVG nos
    dois sentidos; atributo de mensagem: log do vao |j - i| (no line graph e o
    valor da amostra compartilhada). Com --directed, 2 relacoes: vizinho no
    passado e vizinho no futuro. Camadas, leitura e cabeca identicas a LineGNN.

    Com faixas > 0 (post-hoc da rede de Takens, 15/nov/2026): cada sentido se
    divide em `faixas` relacoes pelo vao |j - i|, em faixas logaritmicas fixas
    entre 1 e 1000 amostras, para que o vao entre em cada mensagem e nao so
    como media por no. faixas = 0 reproduz o modelo original.
    """

    def __init__(self, d: int = 64, camadas: int = 4, dropout: float = 0.1, n_saidas: int = 5,
                 direcionado: bool = False, faixas: int = 0):
        super().__init__()
        self.sentidos = 2 if direcionado else 1
        self.faixas = faixas
        self.relacoes = self.sentidos * max(faixas, 1)
        if faixas:
            self.register_buffer("cortes", torch.logspace(0, 3, faixas + 1)[1:-1], persistent=False)
        self.entrada = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.derivacao = nn.Embedding(N_DERIV, d)
        self.camadas = nn.ModuleList(Camada(d, dropout, self.relacoes) for _ in range(camadas))
        self.cabeca = nn.Sequential(nn.LayerNorm(2 * d * N_DERIV), nn.Dropout(0.25),
                                    nn.Linear(2 * d * N_DERIV, 256), nn.GELU(),
                                    nn.Dropout(0.5), nn.Linear(256, n_saidas))
        self.d = d

    def forward(self, sinais, arestas, grafo, media, desvio):
        n_grafos = sinais.shape[0] * N_DERIV
        x = ((sinais - media) / desvio).reshape(-1)
        n = len(x)
        no_grafo = torch.arange(n, device=x.device) // N_AMOSTRAS
        t = (torch.arange(n, device=x.device) % N_AMOSTRAS).float() / N_AMOSTRAS
        base = grafo * N_AMOSTRAS
        u, v = base + arestas[:, 0], base + arestas[:, 1]            # u no passado de v
        vao = torch.log((arestas[:, 1] - arestas[:, 0]).float())
        origem, destino = torch.cat([u, v]), torch.cat([v, u])
        relacao = torch.cat([torch.zeros_like(u), torch.ones_like(v)])  # 0: vem do passado, 1: do futuro
        vao2 = torch.cat([vao, vao])
        if self.sentidos == 1:
            relacao = torch.zeros_like(relacao)
        if self.faixas:
            faixa = torch.bucketize(torch.exp(vao2), self.cortes.to(vao2.dtype), right=True)
            relacao = relacao * self.faixas + faixa
        adjacencias, x_medio = [], []
        for r in range(self.relacoes):
            m = relacao == r
            o, d = origem[m], destino[m]
            grau = torch.bincount(d, minlength=n).clamp(min=1).float()
            adjacencias.append(torch.sparse_coo_tensor(torch.stack([d, o]), 1.0 / grau[d], (n, n)).coalesce())
            x_medio.append(torch.zeros(n, device=x.device).index_add_(0, d, vao2[m]) / grau)
        x_medio = torch.stack(x_medio, 1)
        h = self.entrada(torch.stack([x, t], 1)) + self.derivacao(no_grafo % N_DERIV)
        for camada in self.camadas:
            h = camada(h, adjacencias, x_medio)
        media_g = h.reshape(n_grafos, N_AMOSTRAS, self.d).mean(1)
        max_g = h.reshape(n_grafos, N_AMOSTRAS, self.d).amax(1)
        return self.cabeca(torch.cat([media_g, max_g], 1).reshape(sinais.shape[0], -1))


# --- treino ----------------------------------------------------------------

@torch.no_grad()
def prever(modelo, cache, linhas, media, desvio, lote, dispositivo) -> np.ndarray:
    modelo.eval()
    saidas = []
    for i in range(0, len(linhas), lote):
        s, a, g = cache.lote(linhas[i:i + lote], dispositivo)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            saidas.append(torch.sigmoid(modelo(s, a, g, media, desvio).float()).cpu().numpy())
    return np.concatenate(saidas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ptbxl", *EXTERNOS], default="ptbxl",
                    help="externos: rotulos de data/external/<dataset>; --cache deve ser o cache da HVG do dataset")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "results/line_gnn_exploratory")
    ap.add_argument("--tag", default="hvg_d64_l4")
    ap.add_argument("--test-fold", type=int, default=9,
                    help="rotacao de validacao cruzada dentro dos folds 1-9; o fold 10 nunca entra")
    ap.add_argument("--val-fold", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--directed", action="store_true",
                    help="passagem de mensagens tipada pelas 4 relacoes temporais das cunhas")
    ap.add_argument("--graph", choices=["line", "hvg", "takens"], default="line",
                    help="line: L(HVG) (padrao); hvg: ablacao sobre a propria HVG (results/ablation_gnn/REGISTRO.md); "
                         "takens: a mesma HVGGNN sobre a rede de recorrencia (cache de build_hvg_cache.py --graph takens)")
    ap.add_argument("--span-bins", type=int, default=0,
                    help="(hvg/takens) divide cada sentido em N relacoes por faixa logaritmica do vao, 1-1000 "
                         "amostras; 0 = modelo original. Post-hoc da rede de Takens, 15/nov/2026")
    ap.add_argument("--limit-train", type=int, default=None, help="so para teste rapido")
    ap.add_argument("--confirm-test-fold", default=None, metavar="JUSTIFICATIVA",
                    help="regime confirmatorio (treino 1-8, validacao 9, teste 10); exige justificativa, "
                         "registrada em results/test_fold_usage.jsonl (ver results/confirmatory/PREREGISTRATION.md)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dispositivo = torch.device("cuda")

    cache = Cache(args.cache)
    if args.dataset == "ptbxl":
        registros = PTBXLDataLoader(PTBXL).records.set_index("ecg_id").loc[cache.ecg_id].reset_index()
        rotulos = list(SUPERCLASSES)
    else:
        rotulos = json.load(open(ROOT / "data/external" / args.dataset / "prep.json"))["labels"]
        lab = pd.read_csv(ROOT / "data/external" / args.dataset / "labels.csv").set_index("ecg_id")
        # registros do cache sem rotulo (chapman8) ficam com fold NaN e caem fora de todas as mascaras
        registros = lab.reindex(cache.ecg_id)[["strat_fold", *rotulos]].reset_index()
        registros[rotulos] = registros[rotulos].fillna(0)
    fold = registros["strat_fold"]
    if args.confirm_test_fold:
        if args.limit_train or (args.test_fold, args.val_fold) != (10, 9):
            sys.exit("confirmatorio exige --test-fold 10 --val-fold 9 e nenhum --limit-train")
        mascaras = {"train": fold.between(1, 8), "validation": fold.eq(9), "test": fold.eq(10)}
    else:
        if 10 in (args.test_fold, args.val_fold) or args.test_fold == args.val_fold:
            sys.exit("folds invalidos: o fold 10 e selado e teste != validacao")
        mascaras = {"train": fold.between(1, 9) & ~fold.isin([args.test_fold, args.val_fold]),
                    "validation": fold.eq(args.val_fold), "test": fold.eq(args.test_fold)}
    linhas = {n: np.flatnonzero(m.to_numpy()) for n, m in mascaras.items()}
    if args.limit_train:
        linhas = {n: v[:args.limit_train] for n, v in linhas.items()}
    alvos = registros[rotulos].to_numpy(np.float32)
    treino_sinais = cache.sinais[linhas["train"]]
    media, desvio = float(treino_sinais.mean()), float(treino_sinais.std())
    print(f"treino {len(linhas['train']):,} / val {len(linhas['validation']):,} / "
          f"teste {len(linhas['test']):,}; escala do treino media {media:.4f} desvio {desvio:.4f}")

    Modelo = HVGGNN if args.graph in ("hvg", "takens") else LineGNN
    extra = {"faixas": args.span_bins} if args.graph in ("hvg", "takens") else {}
    if args.span_bins and not extra:
        sys.exit("--span-bins so vale para --graph hvg ou takens")
    modelo = Modelo(args.hidden, args.layers, args.dropout, n_saidas=len(rotulos),
                    direcionado=args.directed, **extra).to(dispositivo)
    print(f"{sum(p.numel() for p in modelo.parameters()):,} parametros")
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.wd)
    passos = args.epochs * math.ceil(len(linhas["train"]) / args.batch)
    agenda = torch.optim.lr_scheduler.OneCycleLR(otimizador, max_lr=args.lr, total_steps=passos)
    alvo_t = torch.as_tensor(alvos, device=dispositivo)

    args.output.mkdir(parents=True, exist_ok=True)
    pesos = args.output / f"{args.tag}.pt"
    historico, melhor = [], -1.0
    for epoca in range(1, args.epochs + 1):
        modelo.train()
        inicio, perdas = time.perf_counter(), []
        ordem = rng.permutation(linhas["train"])
        for i in range(0, len(ordem), args.batch):
            lote = ordem[i:i + args.batch]
            s, a, g = cache.lote(lote, dispositivo)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = modelo(s, a, g, media, desvio)
            perda = F.binary_cross_entropy_with_logits(logits.float(), alvo_t[lote])
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otimizador.step()
            agenda.step()
            perdas.append(perda.item())
        val = prever(modelo, cache, linhas["validation"], media, desvio, 64, dispositivo)
        auc = float(roc_auc_score(alvos[linhas["validation"]], val, average="macro"))
        historico.append({"epoch": epoca, "loss": float(np.mean(perdas)), "val_auroc": auc})
        if auc > melhor:
            melhor = auc
            torch.save(modelo.state_dict(), pesos)
        print(f"epoca {epoca:3d}  perda {np.mean(perdas):.4f}  val {auc:.4f}  "
              f"melhor {melhor:.4f}  ({time.perf_counter() - inicio:.0f} s, "
              f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB)", flush=True)

    modelo.load_state_dict(torch.load(pesos))
    teste = prever(modelo, cache, linhas["test"], media, desvio, 64, dispositivo)
    y = alvos[linhas["test"]]
    resultado = {
        "model": (("HVGGNN direcionada (2 relacoes temporais)" if args.directed else "HVGGNN")
                  + (f", vao em {args.span_bins} faixas por sentido" if args.span_bins else "")
                  + (" sobre a rede de recorrencia de Takens" if args.graph == "takens" else " sobre a HVG")
                  + ", 12 derivacoes, pesos compartilhados") if args.graph in ("hvg", "takens") else
                 (("LineGNN direcionada (4 relacoes temporais)" if args.directed else "LineGNN")
                  + " sobre L(HVG), 12 derivacoes, pesos compartilhados"),
        "regime": ("confirmatory: treino 1-8, validacao 9, teste 10" if args.confirm_test_fold else
                   f"exploratory CV: teste fold {args.test_fold}, validacao fold {args.val_fold}, "
                   "treino nos demais de 1-9; fold 10 nao lido"),
        "test_fold": args.test_fold, "val_fold": args.val_fold,
        "selection": f"epoca de melhor AUROC macro no fold {args.val_fold}",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "scale": {"mean": media, "std": desvio},
        "best_epoch": int(max(historico, key=lambda r: r["val_auroc"])["epoch"]),
        "validation_macro_auroc": melhor,
        "test_macro_auroc": float(roc_auc_score(y, teste, average="macro")),
        "test_per_label": {c: float(roc_auc_score(y[:, i], teste[:, i]))
                           for i, c in enumerate(rotulos)},
        "dataset": args.dataset, "labels": rotulos,
        "history": historico,
        "finished": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / f"{args.tag}.json").write_text(json.dumps(resultado, indent=2) + "\n")
    np.savez(args.output / f"{args.tag}_test_scores.npz", ecg_id=cache.ecg_id[linhas["test"]],
             scores=teste)
    if args.confirm_test_fold:
        with (ROOT / "results/test_fold_usage.jsonl").open("a") as uso:
            uso.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                  "reason": args.confirm_test_fold, "test_fold": 10,
                                  "model": f"{resultado['model']} {args.tag}", "dataset": args.dataset,
                                  "feature_count": None,
                                  "test_macro_auroc": resultado["test_macro_auroc"],
                                  "output_dir": str(args.output)}) + "\n")
    print(f"val {melhor:.4f}  teste {resultado['test_macro_auroc']:.4f}  "
          f"(epoca {resultado['best_epoch']}) -> {args.output}")


if __name__ == "__main__":
    main()
