"""Constantes de treino para as familias `_abs`, que preservam a amplitude.

Todas as visoes de imagem e grafo descartam a escala de cada derivacao em algum
ponto (min-max na GAF, quantis da propria serie na MTF e no grafo de transicao,
taxa de recorrencia fixa no RP e na rede de Takens, quantizacao equal-probability
por imagem no Haralick). As variantes `_abs` trocam cada normalizacao por
amostra por uma constante unica, estimada aqui so nos folds 1-7 (treino do
regime exploratorio), e aplicada igual a todo registro. Nada vem de validacao
ou teste, entao a extracao nao e transdutiva -- e continua nao sendo no regime
confirmatorio, cujo treino (1-8) contem estes folds.

Por derivacao:
- `scale`: p99,9 do |x| maximo por registro, sobre TODOS os registros de
  treino. O maximo literal e de 9 a 18x o registro mediano (artefatos), e
  deixaria o ECG tipico em ~5-10% da faixa da GAF. Acima do p99,9 a GAF satura.
- `mtf_edges`, `transition_edges`: quantis do conjunto de amostras de treino
  (8 e 10 estados), no lugar dos quantis da propria serie.
- `epsilon`: percentil 10, entre registros, do limiar que da taxa de
  recorrencia 0,05 com o embedding congelado -- entao 90% dos registros ficam
  em taxa <= 0,05, o teto das familias atuais, e o registro mediano em ~0,01.
  Com a mediana dos limiares, metade dos registros passava de 0,05 com cauda
  pesada (p99 de 0,4 a 0,6), e a rede de Takens custava 20-80 s por derivacao
  nesses registros, sem ganho de leitura. O mesmo epsilon serve a imagem e ao
  grafo de recorrencia, que continuam lendo a mesma matriz (secao 4.4).
- `haralick[familia]`: 15 cortes de cinza (16 niveis), quantis equiprovaveis
  do conjunto de pixels das imagens de treino.

As escalas usam os 14.955 registros de treino; o resto, uma amostra de 400
(semente 42), como o embedding de `takens_embedding.json`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from joblib import Parallel, delayed
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interpretecg.data import LEADS, PTBXLDataLoader  # noqa: E402
from interpretecg.view_features import (  # noqa: E402
    ABSOLUTE_IMAGES, RECURRENCE_RATE, THEILER_WINDOW)
from interpretecg.views.series_to_image.transforms import _delay_embedding  # noqa: E402

SEED = 42
NIVEIS = 16
ESTADOS_MTF = 8
ESTADOS_TRANSICAO = 10
PIXELS_POR_IMAGEM = 4000
PERCENTIL_EPSILON = 10


def ler(caminho: str) -> np.ndarray:
    sinal, campos = wfdb.rdsamp(caminho)
    assert tuple(n.upper() for n in campos["sig_name"]) == LEADS, campos["sig_name"]
    return sinal


def limiar_recorrencia(x: np.ndarray, m: int, tau: int) -> float:
    pontos = _delay_embedding(x, m, tau)
    linhas, colunas = np.triu_indices(len(pontos), k=THEILER_WINDOW + 1)
    return float(np.quantile(cdist(pontos, pontos)[linhas, colunas], RECURRENCE_RATE))


def pixels(sinal: np.ndarray, parametros: dict, semente: int) -> dict:
    rng = np.random.default_rng(semente)
    saida = {}
    for j, d in enumerate(LEADS):
        for familia, imagem in ABSOLUTE_IMAGES.items():
            valores = imagem(sinal[:, j], parametros[d]).ravel()
            saida[(d, familia)] = rng.choice(valores, min(PIXELS_POR_IMAGEM, len(valores)),
                                             replace=False)
    return saida


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/ptb-xl")
    ap.add_argument("--embedding", type=Path, default=ROOT / "data/features/takens_embedding.json")
    ap.add_argument("--output", type=Path, default=ROOT / "data/features/absolute_calibration.json")
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--n-jobs", type=int, default=12)
    args = ap.parse_args()

    loader = PTBXLDataLoader(args.data_dir)
    loader.assert_complete()
    treino = loader.records.loc[loader.records["strat_fold"].between(1, 7)].reset_index(drop=True)
    embedding = json.loads(args.embedding.read_text())["per_lead"]

    maximos = np.array(Parallel(n_jobs=args.n_jobs, batch_size=64)(
        delayed(lambda c: np.abs(ler(c)).max(0))(c) for c in treino["record_path"]))
    escala = np.percentile(maximos, 99.9, axis=0)
    print(f"escala (p99,9 de {len(maximos):,} registros): "
          + ", ".join(f"{d} {s:.2f}" for d, s in zip(LEADS, escala)))

    amostra = treino.sample(n=args.sample, random_state=SEED).sort_values("ecg_id")
    sinais = Parallel(n_jobs=args.n_jobs)(delayed(ler)(c) for c in amostra["record_path"])
    sinais = np.stack(sinais)  # registros x amostras x derivacoes

    parametros = {}
    for j, d in enumerate(LEADS):
        valores = sinais[:, :, j].ravel()
        m, tau = embedding[d]["m"], embedding[d]["tau"]
        limiares = Parallel(n_jobs=args.n_jobs)(
            delayed(limiar_recorrencia)(s[:, j], m, tau) for s in sinais)
        parametros[d] = {
            "m": m, "tau": tau, "scale": float(escala[j]),
            "epsilon": float(np.percentile(limiares, PERCENTIL_EPSILON)),
            "mtf_edges": np.quantile(valores, np.arange(1, ESTADOS_MTF) / ESTADOS_MTF).tolist(),
            "transition_edges": np.quantile(
                valores, np.arange(1, ESTADOS_TRANSICAO) / ESTADOS_TRANSICAO).tolist(),
        }
        print(f"{d:4s} epsilon {parametros[d]['epsilon']:.3f}")

    amostras = Parallel(n_jobs=args.n_jobs)(
        delayed(pixels)(s, parametros, SEED + i) for i, s in enumerate(sinais))
    for d in LEADS:
        parametros[d]["haralick"] = {
            familia: np.quantile(np.concatenate([a[(d, familia)] for a in amostras]),
                                 np.arange(1, NIVEIS) / NIVEIS).tolist()
            for familia in ABSOLUTE_IMAGES}

    args.output.write_text(json.dumps({
        "source": f"folds 1-7 (regime exploratorio); escala sobre {len(maximos)} registros, "
                  f"o resto sobre {args.sample} amostrados",
        "seed": SEED,
        "scale": "p99,9 do |x| maximo por registro",
        "epsilon": f"percentil {PERCENTIL_EPSILON} do limiar de taxa de recorrencia {RECURRENCE_RATE}, "
                   f"janela de Theiler {THEILER_WINDOW}",
        "haralick_levels": NIVEIS,
        "pixels_per_image": PIXELS_POR_IMAGEM,
        "per_lead": parametros,
    }, indent=1) + "\n")
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
