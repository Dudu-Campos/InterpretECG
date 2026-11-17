"""Baselines profundos adicionais na serie crua, uma familia de arquitetura cada.

Proveniencia:

- `inception1d` e `lstm_bidir`: Strodthoff et al. (2021), github.com/helme/
  ecg_ptbxl_benchmarking, `code/models/inception1d.py`, `rnn1d.py`,
  `basic_conv1d.py` (cabeca) e `fastai_model.py` (hiperparametros). Portados
  sem fastai; mesma topologia e mesma cabeca (`lin_ftrs_head=[128]`,
  `ps_head=0.5`, BatchNorm -> Dropout -> Linear). O `kernel_size` do inception
  e 8 x 5 = 40, como em `fastai_model.py`, o que da nucleos (39, 19, 9).
- `transformer`: DeepECG-Kit (Hicks et al., CBMS 2026), github.com/stevenah/
  deepecg-kit@5ecd8c17, `deepecgkit/models/transformer_ecg.py` e
  `layers.PositionalEncoding`, licenca Apache-2.0. Mesmos padroes do kit
  (d_model 128, 8 cabecas, 4 camadas, feedforward 512, dropout 0,1).

Todas recebem (lote, 12, amostras) e devolvem logitos (lote, n_saidas).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def _bn_drop_lin(ni: int, no: int, p: float, ativacao: bool) -> list[nn.Module]:
    camadas = [nn.BatchNorm1d(ni), nn.Dropout(p), nn.Linear(ni, no)]
    return camadas + ([nn.ReLU(inplace=True)] if ativacao else [])


def _cabeca(nf: int, n_saidas: int, ocultas=(128,), ps: float = 0.5) -> nn.Sequential:
    """create_head1d / cabeca do RNN1d: dropout ps/2 nas ocultas e ps na ultima."""
    larguras = [nf, *ocultas, n_saidas]
    taxas = [ps / 2] * (len(larguras) - 2) + [ps]
    camadas = []
    for i, p in enumerate(taxas):
        camadas += _bn_drop_lin(larguras[i], larguras[i + 1], p, i < len(taxas) - 1)
    return nn.Sequential(*camadas)


# --- inception1d ------------------------------------------------------------

def _conv(ni: int, nf: int, k: int, stride: int = 1) -> nn.Conv1d:
    return nn.Conv1d(ni, nf, k, stride=stride, padding=(k - 1) // 2, bias=False)


class _BlocoInception(nn.Module):
    def __init__(self, ni: int, nf: int, nucleos, gargalo: int = 32):
        super().__init__()
        self.gargalo = _conv(ni, gargalo, 1)
        self.convs = nn.ModuleList(_conv(gargalo, nf, k) for k in nucleos)
        self.pool = nn.Sequential(nn.MaxPool1d(3, 1, padding=1), _conv(ni, nf, 1))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d((len(nucleos) + 1) * nf), nn.ReLU())

    def forward(self, x):
        g = self.gargalo(x)
        return self.bn_relu(torch.cat([c(g) for c in self.convs] + [self.pool(x)], 1))


class _Atalho(nn.Module):
    def __init__(self, ni: int, nf: int):
        super().__init__()
        self.conv, self.bn = _conv(ni, nf, 1), nn.BatchNorm1d(nf)

    def forward(self, entrada, saida):
        return torch.relu(saida + self.bn(self.conv(entrada)))


class Inception1d(nn.Module):
    def __init__(self, n_saidas: int, entrada: int = 12, kernel_size: int = 40, profundidade: int = 6,
                 gargalo: int = 32, filtros: int = 32):
        super().__init__()
        nucleos = [k - 1 if k % 2 == 0 else k for k in (kernel_size, kernel_size // 2, kernel_size // 4)]
        largura = (len(nucleos) + 1) * filtros
        self.blocos = nn.ModuleList(
            _BlocoInception(entrada if d == 0 else largura, filtros, nucleos, gargalo)
            for d in range(profundidade))
        self.atalhos = nn.ModuleList(
            _Atalho(entrada if d == 0 else largura, largura) for d in range(profundidade // 3))
        self.cabeca = _cabeca(2 * largura, n_saidas)

    def forward(self, x):
        residuo = x
        for d, bloco in enumerate(self.blocos):
            x = bloco(x)
            if d % 3 == 2:
                x = self.atalhos[d // 3](residuo, x)
                residuo = x
        x = torch.cat([x.amax(-1), x.mean(-1)], 1)          # AdaptiveConcatPool1d: max, media
        return self.cabeca(x)


# --- lstm bidirecional --------------------------------------------------------

class LSTMBidir(nn.Module):
    def __init__(self, n_saidas: int, entrada: int = 12, oculta: int = 256, camadas: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(entrada, oculta, camadas, bidirectional=True)
        self.oculta = oculta
        self.cabeca = _cabeca(6 * oculta, n_saidas)

    def forward(self, x):
        h, _ = self.lstm(x.permute(2, 0, 1))               # T x B x 2H
        h = h.permute(1, 2, 0)                              # B x 2H x T
        ultimo = torch.cat([h[:, :self.oculta, -1], h[:, self.oculta:, 0]], 1)
        return self.cabeca(torch.cat([h.mean(-1), h.amax(-1), ultimo], 1))


# --- transformer (DeepECG-Kit) -------------------------------------------------

class _CodificacaoPosicional(nn.Module):
    def __init__(self, d: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(1, max_len, d)
        pe[0, :, 0::2], pe[0, :, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerECG(nn.Module):
    def __init__(self, n_saidas: int, entrada: int = 12, d: int = 128, cabecas: int = 8,
                 camadas: int = 4, feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.projecao = nn.Linear(entrada, d)
        self.posicao = _CodificacaoPosicional(d, dropout=dropout)
        self.codificador = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, cabecas, feedforward, dropout, batch_first=True), camadas)
        self.classifier = nn.Sequential(nn.Linear(d, feedforward // 2), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(feedforward // 2, n_saidas))

    def forward(self, x):
        h = self.codificador(self.posicao(self.projecao(x.transpose(1, 2))))
        return self.classifier(h.mean(1))
