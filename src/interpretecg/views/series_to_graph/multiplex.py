"""Visibility graph multiplex: uma camada por derivacao, os mesmos nos (amostras).

Lacasa, Nicosia & Latora (2015), *Network structure of multivariate time
series*, Sci. Rep. 5:15508. Cada derivacao gera sua HVG sobre os mesmos n
instantes; o multiplex le a estrutura ENTRE camadas, que nenhuma familia
univariada ve.

Features (camadas na ordem dada, L camadas, L(L-1)/2 pares):

por par (a, b)
    jaccard_a_b    |E_a & E_b| / |E_a | E_b|, sem as arestas (i, i+1), que
                   toda HVG tem e que inflariam a sobreposicao por igual
    kcorr_a_b      Spearman entre as sequencias de grau k_a(t), k_b(t)
    kmi_a_b        informacao mutua (bits) da distribuicao conjunta de graus,
                   graus truncados em K_MAX (Lacasa et al. 2015, I_ab)

globais
    jaccard_mean/std, kcorr_mean/std, kmi_mean/std
    mult_ge2, mult_ge4, mult_all   fracao das arestas da uniao (sem (i,i+1))
                                   presentes em >= 2, >= 4 e em todas as camadas
    mult_mean                      multiplicidade media dessas arestas
    union_frac, core_frac          |uniao| e |intersecao| sobre a media de |E_a|
    participation_mean/std         coeficiente de participacao de Battiston et
                                   al. (2014), P_i = L/(L-1) [1 - sum_a (k_i^a/o_i)^2]

Tudo depende so da ordem das amostras em cada camada: invariante a qualquer
transformacao crescente aplicada a cada derivacao.
"""

from __future__ import annotations

import itertools

import numpy as np
from scipy.stats import spearmanr

K_MAX = 16


def _graus(arestas: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(arestas.ravel(), minlength=n)


def _mi(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.minimum(a, K_MAX), np.minimum(b, K_MAX)
    conj = np.zeros((K_MAX + 1, K_MAX + 1))
    np.add.at(conj, (a, b), 1)
    conj /= conj.sum()
    pa, pb = conj.sum(1, keepdims=True), conj.sum(0, keepdims=True)
    nz = conj > 0
    return float((conj[nz] * np.log2(conj[nz] / (pa @ pb)[nz])).sum())


def multiplex_features(camadas: list[np.ndarray], nomes: list[str], n: int) -> dict[str, float]:
    """`camadas[a]` e um array (E_a, 2) de arestas (i, j), i < j, sobre n nos."""
    L = len(camadas)
    graus = [_graus(np.asarray(e, dtype=np.int64), n) for e in camadas]
    codigos = []
    for e in camadas:
        e = np.asarray(e, dtype=np.int64)
        e = e[e[:, 1] - e[:, 0] > 1]                      # sem (i, i+1)
        codigos.append(np.unique(e[:, 0] * n + e[:, 1]))

    saida: dict[str, float] = {}
    jac, kc, km = [], [], []
    for a, b in itertools.combinations(range(L), 2):
        inter = np.intersect1d(codigos[a], codigos[b], assume_unique=True).size
        uniao = codigos[a].size + codigos[b].size - inter
        j = inter / uniao if uniao else np.nan
        if graus[a].std() > 0 and graus[b].std() > 0:
            c = float(spearmanr(graus[a], graus[b]).statistic)
        else:
            c = np.nan                                    # camada degenerada (derivacao morta)
        m = _mi(graus[a], graus[b])
        par = f"{nomes[a]}_{nomes[b]}"
        saida[f"jaccard_{par}"], saida[f"kcorr_{par}"], saida[f"kmi_{par}"] = j, c, m
        jac.append(j), kc.append(c), km.append(m)
    for nome, v in (("jaccard", jac), ("kcorr", kc), ("kmi", km)):
        saida[f"{nome}_mean"], saida[f"{nome}_std"] = float(np.nanmean(v)), float(np.nanstd(v))

    todos = np.concatenate(codigos)
    _, mult = np.unique(todos, return_counts=True)
    tamanho_medio = np.mean([c.size for c in codigos])
    saida["mult_ge2"] = float((mult >= 2).mean())
    saida["mult_ge4"] = float((mult >= 4).mean())
    saida["mult_all"] = float((mult == L).mean())
    saida["mult_mean"] = float(mult.mean())
    saida["union_frac"] = float(mult.size / tamanho_medio)
    saida["core_frac"] = float((mult == L).sum() / tamanho_medio)

    k = np.stack(graus).astype(float)                    # L x n
    o = k.sum(0)
    p = L / (L - 1) * (1 - ((k / o) ** 2).sum(0))
    saida["participation_mean"], saida["participation_std"] = float(p.mean()), float(p.std())
    return saida
