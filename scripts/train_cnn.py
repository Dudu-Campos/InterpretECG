"""Baseline profundo de referência (seção 6.2): xresnet1d101 na série crua.

Reproduz o protocolo do benchmark de Strodthoff et al. (2021, IEEE JBHI,
repositório ecg_ptbxl_benchmarking, `fastai_model`), em PyTorch puro:

- entrada de 100 Hz, padronizada por média e desvio ESCALARES do treino;
- treino com um recorte aleatório de 2,5 s (250 amostras) por registro por época;
- avaliação em janelas de 2,5 s com passo de 1,25 s, agregadas por MÁXIMO;
- 50 épocas, one-cycle (lr máx. 1e-2, wd 1e-2), batch 128, BCE multirrótulo;
- cabeça com concat pooling, uma camada oculta de 128, dropout 0,25/0,5;
- SEM early stopping: o modelo avaliado é o da última época, como no benchmark.

Por isso nenhuma escolha depende da validação nem do teste: a configuração é
a publicada, congelada antes de qualquer número. A validação (fold 9 no regime
confirmatório) é só reportada, como checagem de sanidade.

A CNN é concorrente de configuração fixa, então é a única exceção à regra de
não tocar o fold 10 antes da rodada final (CLAUDE.md, seção 6.1, decisão de
29/out/2026). Ela continua sujeita ao registro em
`results/test_fold_usage.jsonl` e à exigência de justificativa explícita.

Desvios conhecidos do original: PyTorch em vez de fastai v1 (otimizador AdamW
com momentos do one-cycle 0,95→0,85, mesmo agendamento), precisão mista bf16,
uma única semente.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from joblib import Parallel, delayed
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
# Mesmo mecanismo de evaluate_views.py: o NFS de /exp trava em leituras grandes,
# entao dados e codigo podem vir de uma copia local.
PTBXL = Path(os.environ.get("INTERPRETECG_PTBXL", str(ROOT / "data/ptb-xl")))
sys.path.insert(0, os.environ.get("INTERPRETECG_SRC", str(ROOT / "src")))
from interpretecg.baselines.deep1d import Inception1d, LSTMBidir, TransformerECG  # noqa: E402
from interpretecg.data import (LEADS, SEALED_FOLD, SPLIT_REGIMES,  # noqa: E402
                               SUPERCLASSES, PTBXLDataLoader)
from interpretecg.external import EXTERNOS, carregar as carregar_externo, mascaras as mascaras_externas  # noqa: E402

SEED = 42
JANELA = 250
PASSO = 125
EPOCAS = 50
LOTE = 128
TAXA = 1e-2
DECAIMENTO = 1e-2


# --- xresnet1d (Strodthoff, ecg_ptbxl_benchmarking/code/models/xresnet1d.py) ---

def conv_bn(entrada, saida, kernel, stride=1, ativacao=True, zero_bn=False):
    bn = nn.BatchNorm1d(saida)
    nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
    camadas = [nn.Conv1d(entrada, saida, kernel, stride=stride,
                         padding=(kernel - 1) // 2, bias=False), bn]
    if ativacao:
        camadas.append(nn.ReLU(inplace=True))
    return nn.Sequential(*camadas)


class ResBlock(nn.Module):
    def __init__(self, expansao, entrada, oculta, stride, kernel):
        super().__init__()
        nf, ni = oculta * expansao, entrada * expansao
        self.convs = nn.Sequential(
            conv_bn(ni, oculta, 1),
            conv_bn(oculta, oculta, kernel, stride=stride),
            conv_bn(oculta, nf, 1, ativacao=False, zero_bn=True))
        atalho = []
        if stride != 1:
            atalho.append(nn.AvgPool1d(2, ceil_mode=True))
        if ni != nf:
            atalho.append(conv_bn(ni, nf, 1, ativacao=False))
        self.atalho = nn.Sequential(*atalho)

    def forward(self, x):
        return F.relu(self.convs(x) + self.atalho(x))


class XResNet1d(nn.Module):
    def __init__(self, camadas, n_saidas, entrada=12, expansao=4, kernel=5,
                 tronco=(32, 32, 64), ocultas_cabeca=(128,), dropout=0.5):
        super().__init__()
        tamanhos = [entrada, *tronco]
        self.tronco = nn.Sequential(
            *[conv_bn(tamanhos[i], tamanhos[i + 1], kernel, stride=2 if i == 0 else 1)
              for i in range(3)],
            nn.MaxPool1d(3, stride=2, padding=1))
        blocos = [64 // expansao, 64, 128, 256, 512]
        estagios = []
        for i, n in enumerate(camadas):
            estagios += [ResBlock(expansao, blocos[i] if j == 0 else blocos[i + 1],
                                  blocos[i + 1], 1 if (i == 0 or j > 0) else 2, kernel)
                         for j in range(n)]
        self.estagios = nn.Sequential(*estagios)
        nf = blocos[-1] * expansao
        larguras = [2 * nf, *ocultas_cabeca, n_saidas]
        taxas = [dropout / 2] * (len(larguras) - 2) + [dropout]
        cabeca = []
        for i, p in enumerate(taxas):
            cabeca += [nn.BatchNorm1d(larguras[i]), nn.Dropout(p),
                       nn.Linear(larguras[i], larguras[i + 1])]
            if i < len(taxas) - 1:
                cabeca.append(nn.ReLU(inplace=True))
        self.cabeca = nn.Sequential(*cabeca)

    def forward(self, x):
        x = self.estagios(self.tronco(x))
        x = torch.cat([F.adaptive_avg_pool1d(x, 1), F.adaptive_max_pool1d(x, 1)], 1)
        return self.cabeca(x.flatten(1))


def xresnet1d101(n_saidas):
    return XResNet1d([3, 4, 23, 3], n_saidas)


# Uma familia de arquitetura por linha, todas na mesma receita do benchmark
# (janelas de 2,5 s, one-cycle, 50 epocas, lote 128, maximo sobre janelas).
# Taxa de pico: a do `fastai_configs.py` do Strodthoff (lstm_bidir 1e-3, demais
# o padrao 1e-2); o transformer nao esta no benchmark e usa o padrao do
# DeepECG-Kit (1e-3), porque 1e-2 e alto demais para atencao com AdamW.
ARQUITETURAS = {
    "xresnet1d101": (xresnet1d101, TAXA, "xresnet1d101 (Strodthoff et al. 2021, fastai_model)"),
    "inception1d": (Inception1d, TAXA, "inception1d (Strodthoff et al. 2021, fastai_model)"),
    "lstm_bidir": (LSTMBidir, 1e-3, "lstm_bidir (Strodthoff et al. 2021, fastai_model, lr 1e-3)"),
    "transformer": (TransformerECG, 1e-3, "transformer (DeepECG-Kit@5ecd8c17, lr 1e-3)"),
}


# --- dados ---

def carregar_sinais(loader, registros, cache: Path) -> np.ndarray:
    ids = registros["ecg_id"].to_numpy()
    if cache.exists():
        dados = np.load(cache)
        if np.array_equal(dados["ecg_id"], ids):
            return dados["sinais"]
    linhas = [r for _, r in registros.iterrows()]
    sinais = Parallel(n_jobs=8, prefer="threads")(
        delayed(lambda r: loader.load_signal(r)[0].astype(np.float32))(r) for r in linhas)
    sinais = np.stack(sinais).transpose(0, 2, 1)  # registros × derivações × amostras
    np.savez(cache, ecg_id=ids, sinais=sinais)
    return sinais


@torch.no_grad()
def prever(modelo, sinais: torch.Tensor) -> np.ndarray:
    """Janelas de 2,5 s com passo de 1,25 s, agregadas por máximo (benchmark)."""
    modelo.eval()
    inicios = range(0, sinais.shape[-1] - JANELA + 1, PASSO)
    saidas = []
    for i in range(0, len(sinais), 256):
        lote = sinais[i:i + 256]
        janelas = torch.stack([lote[..., s:s + JANELA] for s in inicios], 1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = modelo(janelas.flatten(0, 1))
        p = torch.sigmoid(logits.float()).view(len(lote), len(inicios), -1)
        saidas.append(p.amax(1))
    return torch.cat(saidas).cpu().numpy()


def metricas(alvos, escores, nomes=SUPERCLASSES) -> dict:
    return {"macro_auroc": float(roc_auc_score(alvos, escores, average="macro")),
            "macro_average_precision": float(average_precision_score(alvos, escores,
                                                                     average="macro")),
            "per_label": {c: {"auroc": float(roc_auc_score(alvos[:, i], escores[:, i]))}
                          for i, c in enumerate(nomes)}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regime", default="exploratory", choices=sorted(SPLIT_REGIMES))
    ap.add_argument("--confirm-test-fold", default=None, metavar="JUSTIFICATIVA",
                    help=f"obrigatorio no regime confirmatorio, que le o fold {SEALED_FOLD}")
    ap.add_argument("--dataset", choices=["ptbxl", *EXTERNOS], default="ptbxl",
                    help="externos: data/external/<dataset>, mesmos folds 1-8/9/10 (ou 1-7/8/9)")
    ap.add_argument("--cache", type=Path, default=None,
                    help="arquivo .npz com os sinais (fora do NFS)")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--arch", choices=sorted(ARQUITETURAS), default="xresnet1d101")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="semente de inicializacao e de ordem dos lotes; 42 e a rodada original")
    args = ap.parse_args()
    if args.regime == "confirmatory" and not args.confirm_test_fold:
        sys.exit(f"O regime confirmatorio avalia no fold {SEALED_FOLD}, que esta selado. "
                 "Passe --confirm-test-fold \"<justificativa>\".")
    construtor, taxa, descricao = ARQUITETURAS[args.arch]
    prefixo = "cnn_xresnet1d101" if args.arch == "xresnet1d101" else f"deep_{args.arch}"
    if args.dataset == "ptbxl":
        if args.cache is None:
            sys.exit("--cache e obrigatorio no PTB-XL")
        saida = args.output or ROOT / "results" / (
            prefixo + ("" if args.regime == "confirmatory" else f"_{args.regime}"))
    else:
        saida = args.output or ROOT / "results/external" / args.dataset / "deep" / (
            f"{args.arch}_{args.regime}_seed{args.seed}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dispositivo = torch.device("cuda")
    if args.dataset == "ptbxl":
        loader = PTBXLDataLoader(PTBXL)
        loader.assert_complete()
        registros = loader.records.reset_index(drop=True)
        sinais = carregar_sinais(loader, registros, args.cache)
        mascaras = {n: m.to_numpy() for n, m in PTBXLDataLoader.split_masks(registros, args.regime).items()}
        nomes, ids, completo = list(SUPERCLASSES), registros["ecg_id"].to_numpy(), loader.is_complete
    else:
        ids, sinais, registros, nomes = carregar_externo(args.dataset)
        mascaras, completo = mascaras_externas(registros["strat_fold"], args.regime), True
    alvos = {n: registros.loc[m, nomes].to_numpy(np.float32) for n, m in mascaras.items()}
    treino_np = sinais[mascaras["train"]]
    media, desvio = float(treino_np.mean()), float(treino_np.std())
    x = {n: torch.as_tensor((sinais[m] - media) / desvio, device=dispositivo)
         for n, m in mascaras.items()}
    y_treino = torch.as_tensor(alvos["train"], device=dispositivo)
    print(f"treino {len(x['train']):,} / validacao {len(x['validation']):,} / "
          f"teste {len(x['test']):,}  [{args.regime}]", flush=True)

    modelo = construtor(len(nomes)).to(dispositivo)
    print(f"{args.arch}: {sum(p.numel() for p in modelo.parameters()):,} parametros, lr {taxa}", flush=True)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=taxa, weight_decay=DECAIMENTO,
                                   betas=(0.95, 0.99))
    passos = len(x["train"]) // LOTE
    agenda = torch.optim.lr_scheduler.OneCycleLR(
        otimizador, max_lr=taxa, total_steps=EPOCAS * passos, pct_start=0.3,
        div_factor=25.0, final_div_factor=1e4, base_momentum=0.85, max_momentum=0.95)
    gerador = torch.Generator(device=dispositivo).manual_seed(args.seed)
    n, comprimento = len(x["train"]), x["train"].shape[-1]
    inicio = time.perf_counter()
    for epoca in range(EPOCAS):
        modelo.train()
        ordem = torch.randperm(n, generator=gerador, device=dispositivo)
        cortes = torch.randint(0, comprimento - JANELA + 1, (n,), generator=gerador,
                               device=dispositivo)
        indices = cortes[:, None] + torch.arange(JANELA, device=dispositivo)
        perdas = []
        for i in range(passos):
            idx = ordem[i * LOTE:(i + 1) * LOTE]
            lote = torch.gather(x["train"][idx], 2,
                                indices[idx][:, None, :].expand(-1, len(LEADS), -1))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                perda = F.binary_cross_entropy_with_logits(modelo(lote).float(), y_treino[idx])
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            otimizador.step()
            agenda.step()
            perdas.append(perda.item())
        if epoca % 10 == 9 or epoca == EPOCAS - 1:
            val = roc_auc_score(alvos["validation"], prever(modelo, x["validation"]),
                                average="macro")
            print(f"  epoca {epoca + 1:2d}  perda {np.mean(perdas):.4f}  "
                  f"val AUROC {val:.4f}  ({(time.perf_counter() - inicio) / 60:.1f} min)",
                  flush=True)

    folds = SPLIT_REGIMES[args.regime]
    validacao = metricas(alvos["validation"], prever(modelo, x["validation"]), nomes)
    escores_teste = prever(modelo, x["test"])
    teste = metricas(alvos["test"], escores_teste, nomes)
    saida.mkdir(parents=True, exist_ok=True)
    np.savez(saida / "test_scores.npz", ecg_id=ids[mascaras["test"]],
             escores=escores_teste, alvos=alvos["test"])
    torch.save(modelo.state_dict(), saida / "model.pt")
    resultado = {
        "task": "PTB-XL diagnostic superclasses" if args.dataset == "ptbxl" else args.dataset,
        "dataset": args.dataset, "labels": nomes,
        "feature_count": None, "families": ["raw_signal"],
        "protocol": {"regime": args.regime,
                     "train_folds": list(range(folds["train"][0], folds["train"][1] + 1)),
                     "validation_fold": folds["validation"], "test_fold": folds["test"],
                     "classifier": descricao,
                     "preprocessing": "standardization by scalar train mean/std",
                     "training": {"epochs": EPOCAS, "batch": LOTE, "max_lr": taxa,
                                  "weight_decay": DECAIMENTO, "schedule": "one-cycle",
                                  "crop": JANELA, "eval_stride": PASSO,
                                  "aggregate": "max", "early_stopping": None,
                                  "precision": "bf16"},
                     "random_state": args.seed},
        "split_sizes": {k: int(len(v)) for k, v in x.items()},
        "selection_metric": None, "selected_c": None,
        "validation": validacao,
        "validation_macro_auroc": validacao["macro_auroc"],
        "test": teste,
        "training_minutes": (time.perf_counter() - inicio) / 60,
    }
    (saida / "metrics.json").write_text(json.dumps(resultado, indent=2) + "\n")

    if args.regime == "confirmatory":
        uso = ROOT / "results" / "test_fold_usage.jsonl"
        with uso.open("a") as stream:
            stream.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": args.confirm_test_fold, "test_fold": SEALED_FOLD,
                "model": args.arch if args.dataset == "ptbxl" else f"{args.arch} [{args.dataset}]",
                "dataset": args.dataset, "feature_count": None,
                "test_macro_auroc": teste["macro_auroc"],
                "complete_dataset": completo, "output_dir": str(saida),
            }) + "\n")
        print(f"Registrado uso #{sum(1 for _ in uso.open())} do fold {SEALED_FOLD} em {uso}")
    print(f"\nvalidacao {validacao['macro_auroc']:.4f}  teste {teste['macro_auroc']:.4f}  "
          f"-> {saida}/")


if __name__ == "__main__":
    main()
