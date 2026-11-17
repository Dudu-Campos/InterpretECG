"""Mudanca de distribuicao por aparelho: treina numa familia de eletrocardiografo
e testa na outra, sem tocar no fold 10.

O PTB-XL tem duas familias de aparelho: AT (AT-6, AT-60) e CS (CS-12, CS-12E,
CS100). Fabricantes diferentes tem ganho, filtragem e ruido diferentes, entao
a troca e uma mudanca de distribuicao natural. Os folds continuam os mesmos
(estratificados por paciente), so restritos por aparelho:

    treino      folds 1-7 da origem
    validacao   fold 8 da origem (so early stopping; nada e escolhido no alvo)
    teste ID    fold 9 da origem
    teste OOD   fold 9 do alvo

nas duas direcoes, AT -> CS e CS -> AT. A pergunta (secao 6.4 reformulada, na
linha de Teney et al. 2022) e se as visoes que quase nao somam AUROC dentro da
distribuicao somam fora dela, e se a escolha de normalizacao troca acuracia
por robustez: as visoes invariantes sao imunes a ganho por construcao, as
`_abs` e a familia de amplitude nao.

Leitores sem selecao de hiperparametro, para que nada dependa da distribuicao
de validacao: boosting com a configuracao que a grade ampla escolheu para o
catch22, e (com --mlp) um MLP de configuracao fixa, ensemble de 3 sementes.
Os escores de teste sao salvos para a analise de diversidade de erro.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_views import (PTBXL, ROOT, BoostingPorRotulo, arquivo,  # noqa: E402
                            carregar_quadros, conjuntos)
from interpretecg.data import SUPERCLASSES, PTBXLDataLoader  # noqa: E402

BOOSTING = {"learning_rate": 0.03, "max_leaf_nodes": 127, "min_samples_leaf": 80,
            "l2_regularization": 0.0, "max_features": 0.5}
MLP = ((1024, 1024), 0.5, 0.2, 1e-2)
SEMENTES_MLP = (0, 1, 2)
PADRAO = [
    "catch22", "series_amplitude", "catch22_placebo", "catch22_amplitude",
    "catch22_amplitude_interderivacao",
    "image_spectrogram", "image_scalogram", "image_gasf", "image_gadf", "image_mtf",
    "image_recurrence", "graph_hvg", "graph_lphvg1", "graph_takens", "graph_transition",
    "image_spectrogram_abs", "image_scalogram_abs", "image_gasf_abs", "image_gadf_abs",
    "image_mtf_abs", "image_recurrence_abs", "graph_takens_abs", "graph_transition_abs",
    "visao_imagem", "visao_grafo", "visao_imagem_abs", "visao_grafo_abs",
    "todas_as_visoes", "catch22_visoes_abs", "todas_amplitude_interderivacao",
    "catch22_amplitude_interderivacao_visoes_abs_sem_recorrencia", "todas_placebo",
]


def familia_aparelho(device: str) -> str | None:
    return "AT" if device.startswith("AT") else "CS" if device.startswith("CS") else None


def auroc(y: np.ndarray, p: np.ndarray) -> dict:
    return {"macro": float(roc_auc_score(y, p, average="macro")),
            "per_label": {c: float(roc_auc_score(y[:, i], p[:, i]))
                          for i, c in enumerate(SUPERCLASSES)}}


def mlp(dados: dict, alvos: dict) -> dict:
    import torch
    from evaluate_mlp import CORTE, prever, treinar
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imputador = SimpleImputer(strategy="median").fit(dados["train"])
    escala = StandardScaler().fit(imputador.transform(dados["train"]))
    tensores = {n: torch.as_tensor(np.clip(escala.transform(imputador.transform(x)), -CORTE, CORTE),
                                   dtype=torch.float32, device=dispositivo)
                for n, x in dados.items()}
    escores = {n: [] for n in ("test_id", "test_ood")}
    for semente in SEMENTES_MLP:
        modelo, _, _ = treinar(tensores, alvos, MLP, semente)
        for n in escores:
            escores[n].append(prever(modelo, tensores[n]))
    return {n: np.mean(v, axis=0) for n, v in escores.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", default=PADRAO)
    ap.add_argument("--output", type=Path, default=ROOT / "results/shift_device_exploratory")
    ap.add_argument("--mlp", action="store_true", help="tambem o MLP (GPU)")
    ap.add_argument("--skip-done", action="store_true")
    args = ap.parse_args()

    quadros = carregar_quadros()
    comuns = None
    for quadro in quadros.values():
        comuns = quadro.index if comuns is None else comuns.intersection(quadro.index)
    registros = PTBXLDataLoader(PTBXL).records
    registros = registros.loc[registros["ecg_id"].isin(comuns)].reset_index(drop=True)
    registros["aparelho"] = registros["device"].map(familia_aparelho)
    fold = registros["strat_fold"]

    nomes = {arquivo(n): (n, f) for n, f in conjuntos().items()}
    faltando = [n for n in args.only if n not in nomes]
    if faltando:
        sys.exit(f"conjuntos desconhecidos: {faltando}")
    args.output.mkdir(parents=True, exist_ok=True)
    leitores = ["boosting"] + (["mlp"] if args.mlp else [])

    for origem, alvo in (("AT", "CS"), ("CS", "AT")):
        mascaras = {
            "train": fold.between(1, 7) & registros["aparelho"].eq(origem),
            "validation": fold.eq(8) & registros["aparelho"].eq(origem),
            "test_id": fold.eq(9) & registros["aparelho"].eq(origem),
            "test_ood": fold.eq(9) & registros["aparelho"].eq(alvo),
        }
        ids = {n: registros.loc[m, "ecg_id"] for n, m in mascaras.items()}
        alvos = {n: registros.loc[m, list(SUPERCLASSES)].to_numpy(np.int8)
                 for n, m in mascaras.items()}
        print(f"\n{origem} -> {alvo}: " + ", ".join(f"{n} {len(v):,}" for n, v in ids.items()))
        prevalencia = {n: dict(zip(SUPERCLASSES, alvos[n].mean(0).round(3).tolist()))
                       for n in ("train", "test_id", "test_ood")}

        for arq in args.only:
            nome, familias = nomes[arq]
            destino = args.output / f"{origem}_to_{alvo}__{arq}.json"
            if args.skip_done and destino.exists():
                continue
            partes = {n: [] for n in ids}
            for familia in familias:
                quadro = quadros[familia]
                treino = quadro.loc[ids["train"]]
                uteis = treino.columns[treino.nunique(dropna=True) > 1]
                for n in ids:
                    partes[n].append(quadro.loc[ids[n], uteis].to_numpy(np.float32))
            dados = {n: np.hstack(p) for n, p in partes.items()}

            saida = {"set": nome, "families": familias, "source": origem, "target": alvo,
                     "regime": "exploratory, folds 1-7/8/9 restritos por aparelho",
                     "feature_count": int(dados["train"].shape[1]),
                     "split_sizes": {n: int(len(v)) for n, v in ids.items()},
                     "prevalence": prevalencia, "readers": {}}
            escores_salvos = {}
            for leitor in leitores:
                inicio = time.perf_counter()
                if leitor == "boosting":
                    modelo = BoostingPorRotulo(max_iter=2000, **BOOSTING).fit(
                        dados["train"], alvos["train"], dados["validation"], alvos["validation"])
                    escores = {n: modelo.predict_proba(dados[n]) for n in ("test_id", "test_ood")}
                else:
                    escores = mlp(dados, alvos)
                r = {n: auroc(alvos[n], escores[n]) for n in escores}
                r["drop_macro"] = r["test_id"]["macro"] - r["test_ood"]["macro"]
                r["minutes"] = round((time.perf_counter() - inicio) / 60, 1)
                saida["readers"][leitor] = r
                escores_salvos |= {f"{leitor}_{n}": v for n, v in escores.items()}
                print(f"  {origem}->{alvo} {nome:58s} {leitor:8s} ID {r['test_id']['macro']:.4f} "
                      f"OOD {r['test_ood']['macro']:.4f}  queda {r['drop_macro']:+.4f}  "
                      f"HYP {r['test_id']['per_label']['HYP']:.3f}->{r['test_ood']['per_label']['HYP']:.3f}"
                      f"  ({r['minutes']} min)", flush=True)
            np.savez_compressed(destino.with_suffix(".npz"), **escores_salvos,
                                ecg_id_test_id=ids["test_id"].to_numpy(),
                                ecg_id_test_ood=ids["test_ood"].to_numpy())
            destino.write_text(json.dumps(saida, indent=2) + "\n")


if __name__ == "__main__":
    main()
