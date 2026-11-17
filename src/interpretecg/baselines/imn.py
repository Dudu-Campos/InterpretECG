"""ECG-IMN (Interpretable Mesomorphic Network), portado para o nosso protocolo.

Proveniencia: github.com/vlbthambawita/mesomorphic_ecg, commit
53d9bc74b538b74cb5be185cee522e8d9f8a7714 (08/fev/2026),
`models/imn_v2_transition_one_linear.py`, classe `ECG_IMN`. Licenca CC-BY-4.0
(model card do repositorio); atribuicao: Thambawita et al., ECG-IMN, CBMS 2026,
arXiv:2602.09566.

Unica mudanca na arquitetura: `n_outputs` em vez de `output_dim = 1` fixo, para
o problema multirrotulo de 5 superclasses (um mapa de pesos W por classe, logo
um modelo linear local por classe e por amostra). Com n_outputs=1 a rede e
identica a original.

Diferencas de protocolo em relacao ao original, todas impostas pelo nosso
(ver scripts/train_imn.py): tarefa multirrotulo de 5 classes em vez de 4
binarias NORM-vs-X restritas a registros com exatamente um dos dois rotulos;
folds 1-7 / 8 / 9 em vez de 1-8 / (9+10 como validacao de selecao).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ECG_IMN(nn.Module):
    """Hiper-rede que gera, por amostra, W [B, K, 12, L] e b [B, K]; logito = sum(W * x) + b."""

    def __init__(self, input_channels: int = 12, signal_len: int = 1000, dropout: float = 0.2,
                 n_outputs: int = 1):
        super().__init__()
        self.C, self.L = input_channels, signal_len
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(16), nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(32), nn.GELU(), nn.MaxPool2d(kernel_size=(1, 2)))
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 15), padding=(1, 7), bias=False),
            nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(kernel_size=(1, 2)))
        self.dropout = nn.Dropout(dropout)
        self.transition = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(), nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.GELU(), nn.Upsample(scale_factor=(1, 2), mode="nearest"),
            nn.Conv2d(16, n_outputs, kernel_size=3, padding=1, bias=True))
        self.bias_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bias_head = nn.Linear(64, n_outputs)

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        feat = self.dropout(self.conv3(self.conv2(self.conv1(x.unsqueeze(1)))))
        w = self.transition(feat)                                    # [B, K, 12, L]
        b = self.bias_head(self.bias_pool(feat).view(batch, -1))     # [B, K]
        logits = (w * x.unsqueeze(1)).sum(dim=(2, 3)) + b
        return logits, w, b


def per_lead_zscore(x: torch.Tensor) -> torch.Tensor:
    """Normalizacao do original: z-score por derivacao e por amostra (descarta a voltagem)."""
    return (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, keepdim=True).clamp_min(1e-6)
