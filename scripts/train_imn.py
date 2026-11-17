"""ECG-IMN (vizinho mais proximo, CBMS 2026) no nosso protocolo exploratorio.

Configuracao do repositorio original (scripts/train_imn_v2_*.sh), sem ajuste:
z-score por derivacao e por amostra, penalidade L1 1e-4 nos pesos gerados,
AdamW lr 1e-3 e weight decay 1e-4, dropout 0,2, lote 64, StepLR(10, 0,1), ate
100 epocas com early stopping (paciencia 10) no AUROC de validacao, pos_weight
neg/pos por classe. 100 Hz, 12 derivacoes.

Protocolo nosso: 5 superclasses multirrotulo, treino nos folds 1-7, early
stopping e checkpoint no fold 8, fold 9 lido uma vez no fim. O fold 10 nao e lido.

Com --regime confirmatory (treino 1-8, early stopping no fold 9, teste no fold
10) roda como concorrente de configuracao fixa (CLAUDE.md 6.1, decisao de
09/nov/2026: a epoca e escolhida so na validacao, como no protocolo publicado, e
isso e declarado). Exige --confirm-test-fold e registra o uso em
results/test_fold_usage.jsonl. Use o cache completo de sinais
(data/features/cnn_signals_100hz.npz): o do HVG nao cobre todos os registros.

Com --binary TASK reproduz a tarefa original (NORM vs X, so registros com
exatamente um dos dois rotulos), ainda nos nossos folds.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.baselines.imn import ECG_IMN, per_lead_zscore  # noqa: E402
from interpretecg.data import SEALED_FOLD, SUPERCLASSES, PTBXLDataLoader  # noqa: E402
from interpretecg.external import EXTERNOS, carregar as carregar_externo, mascaras as mascaras_externas  # noqa: E402


def auroc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p, average="macro"))


@torch.no_grad()
def prever(modelo, x: torch.Tensor, dispositivo, lote: int = 256) -> np.ndarray:
    modelo.eval()
    saidas = []
    for i in range(0, len(x), lote):
        xb = per_lead_zscore(x[i:i + lote].to(dispositivo))
        saidas.append(torch.sigmoid(modelo(xb)[0]).cpu().numpy())
    return np.concatenate(saidas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["ptbxl", *EXTERNOS], default="ptbxl",
                    help="externos: sinais e rotulos de data/external/<dataset> (ignora --cache)")
    ap.add_argument("--cache", type=Path, default=ROOT / "data/features/graphs/hvg_edges.npz")
    ap.add_argument("--output", type=Path, default=ROOT / "results/imn_exploratory")
    ap.add_argument("--binary", choices=["MI", "STTC", "CD", "HYP"], default=None)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regime", choices=["exploratory", "confirmatory"], default="exploratory")
    ap.add_argument("--confirm-test-fold", default=None, metavar="JUSTIFICATIVA")
    args = ap.parse_args()
    if args.regime == "confirmatory" and not args.confirm_test_fold:
        sys.exit(f"O regime confirmatorio avalia no fold {SEALED_FOLD}, que esta selado. "
                 "Passe --confirm-test-fold \"<justificativa>\".")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dispositivo = torch.device("cuda")

    if args.dataset == "ptbxl":
        dados = np.load(args.cache)
        ids = dados["ecg_id"]
        registros = PTBXLDataLoader(ROOT / "data/ptb-xl").records.set_index("ecg_id").loc[ids].reset_index()
        sinais = torch.from_numpy(dados["sinais"])                   # N x 12 x 1000, mV
        rotulos = list(SUPERCLASSES)
    else:
        if args.binary:
            sys.exit("--binary so existe no PTB-XL")
        ids, sinais_np, registros, rotulos = carregar_externo(args.dataset)
        sinais = torch.from_numpy(sinais_np)
        args.cache = ROOT / "data/external" / args.dataset / "signals.npz"
    alvos = registros[rotulos].to_numpy(np.float32)
    manter = np.ones(len(registros), bool)
    if args.binary:
        norm, pos = alvos[:, rotulos.index("NORM")] > 0, alvos[:, rotulos.index(args.binary)] > 0
        manter = norm ^ pos
        alvos, rotulos = pos[:, None].astype(np.float32), [args.binary]
    if args.dataset == "ptbxl":
        mascaras = {n: m.to_numpy() for n, m in PTBXLDataLoader.split_masks(registros, args.regime).items()}
    else:
        mascaras = mascaras_externas(registros["strat_fold"], args.regime)
    linhas = {n: np.flatnonzero(m & manter) for n, m in mascaras.items()}
    tag = f"binary_norm_vs_{args.binary.lower()}" if args.binary else "multilabel5"
    if args.regime == "confirmatory":
        tag += "_confirmatory"
    if args.seed != 42:
        tag += f"_seed{args.seed}"
    if args.dataset != "ptbxl":
        tag = f"{args.dataset}_{tag}"
        if args.output == ROOT / "results/imn_exploratory":
            args.output = ROOT / "results/external" / args.dataset / "deep" / "imn"
    print(f"{tag}: treino {len(linhas['train']):,} / val {len(linhas['validation']):,} / "
          f"teste {len(linhas['test']):,}", flush=True)

    k = alvos.shape[1]
    modelo = ECG_IMN(n_outputs=k).to(dispositivo)
    y_treino = alvos[linhas["train"]]
    pos_weight = torch.as_tensor((len(y_treino) - y_treino.sum(0)) / np.maximum(y_treino.sum(0), 1),
                                 device=dispositivo)
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    agenda = torch.optim.lr_scheduler.StepLR(otimizador, step_size=10, gamma=0.1)
    alvo_t = torch.as_tensor(alvos, device=dispositivo)

    args.output.mkdir(parents=True, exist_ok=True)
    pesos = args.output / f"{tag}.pt"
    historico, melhor, sem_melhora = [], -1.0, 0
    for epoca in range(1, args.epochs + 1):
        modelo.train()
        inicio, perdas = time.perf_counter(), []
        for lote in np.array_split(rng.permutation(linhas["train"]),
                                   int(np.ceil(len(linhas["train"]) / 64))):
            xb = per_lead_zscore(sinais[lote].to(dispositivo))
            logits, w, _ = modelo(xb)
            perda = (F.binary_cross_entropy_with_logits(logits, alvo_t[lote], pos_weight=pos_weight)
                     + 1e-4 * w.abs().mean())
            otimizador.zero_grad(set_to_none=True)
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())
        agenda.step()
        val = auroc(alvos[linhas["validation"]], prever(modelo, sinais[linhas["validation"]], dispositivo))
        historico.append({"epoch": epoca, "loss": float(np.mean(perdas)), "val_auroc": val})
        if val > melhor:
            melhor, sem_melhora = val, 0
            torch.save(modelo.state_dict(), pesos)
        else:
            sem_melhora += 1
        print(f"epoca {epoca:3d}  perda {np.mean(perdas):.4f}  val {val:.4f}  melhor {melhor:.4f}  "
              f"({time.perf_counter() - inicio:.0f} s)", flush=True)
        if sem_melhora >= args.patience:
            break

    modelo.load_state_dict(torch.load(pesos))
    teste = prever(modelo, sinais[linhas["test"]], dispositivo)
    y = alvos[linhas["test"]]
    resultado = {
        "model": "ECG-IMN (Thambawita et al., CBMS 2026), configuracao do repositorio original",
        "source": "github.com/vlbthambawita/mesomorphic_ecg@53d9bc74",
        "task": tag, "dataset": args.dataset, "labels": rotulos,
        "regime": ("exploratory (folds 1-7 / 8 / 9); fold 10 nao lido" if args.regime == "exploratory"
                   else "confirmatory (folds 1-8 / 9 early stopping / 10)"),
        "seed": args.seed, "cache": str(args.cache), "n_test": int(len(linhas["test"])),
        "best_epoch": int(max(historico, key=lambda r: r["val_auroc"])["epoch"]),
        "validation_macro_auroc": melhor,
        "test_macro_auroc": auroc(y, teste),
        "test_per_label": {c: float(roc_auc_score(y[:, i], teste[:, i])) for i, c in enumerate(rotulos)},
        "history": historico, "finished": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / f"{tag}.json").write_text(json.dumps(resultado, indent=2) + "\n")
    np.savez(args.output / f"{tag}_test_scores.npz", ecg_id=ids[linhas["test"]], scores=teste)
    if args.regime == "confirmatory":
        uso = ROOT / "results" / "test_fold_usage.jsonl"
        with uso.open("a") as stream:
            stream.write(json.dumps({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": args.confirm_test_fold, "test_fold": SEALED_FOLD, "model": f"ECG-IMN {tag}", "dataset": args.dataset,
                "feature_count": None, "test_macro_auroc": resultado["test_macro_auroc"],
                "output_dir": str(args.output)}) + "\n")
        print(f"Registrado uso #{sum(1 for _ in uso.open())} do fold {SEALED_FOLD} em {uso}")
    print(f"val {melhor:.4f}  teste {resultado['test_macro_auroc']:.4f}  "
          + " ".join(f"{c} {v:.4f}" for c, v in resultado["test_per_label"].items()))


if __name__ == "__main__":
    main()
