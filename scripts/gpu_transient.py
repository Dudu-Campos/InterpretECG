"""Estresse por DEGRAU de carga: ocioso <-> potencia maxima, em ciclos curtos.

A queda relatada pelo Eduardo nao foi sob carga constante: foi na transicao
entre o fim de uma seed e o inicio da proxima. Isso aponta para di/dt — a
corrente salta de ~30 W para ~350 W em milissegundos, a tensao de alimentacao
afunda por um instante, e a protecao da fonte pode disparar. Carga sustentada
nao reproduz isso, por mais quente que fique: a sonda de 09/nov manteve 344 W
e 76 C por 10 minutos sem nenhum incidente.

Entao o eixo da rampa aqui nao e tamanho de modelo, e **frequencia de
transicao**. Cada estagio encurta o periodo, de 20 s por ciclo ate 1 s, o que
leva de 3 para 60 degraus por minuto.

O estagio final (`--mode seeds` no runner) troca o ciclador sintetico por
treinos reais curtos em sequencia, que e a reproducao fiel do cenario
observado — mesmo degrau, mais o custo de subir e derrubar o processo CUDA.

O que observar esta no jornal, nao aqui: o contador
`hw_power_brake_slowdown`, que nesta placa tem 0 us de vida util.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

# (segundos de rajada, segundos ocioso) -- periodo decrescente
ESTAGIOS = [
    (10.0, 10.0),
    (5.0, 5.0),
    (2.0, 2.0),
    (1.0, 1.0),
    (0.5, 0.5),
    (0.25, 0.75),
]


def agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def rajada(a: torch.Tensor, b: torch.Tensor, ate: float) -> int:
    """Multiplicacoes grandes em laco ate o instante `ate`. Devolve iteracoes."""
    n = 0
    while time.monotonic() < ate:
        for _ in range(8):
            a = torch.matmul(a, b)
            a = a * 0.5 + 0.1   # evita que o valor exploda ou vire NaN
        torch.cuda.synchronize()
        n += 8
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path,
                    default=Path("/var/tmp/interpretecg_transient.jsonl"))
    ap.add_argument("--size", type=int, default=16384, help="lado da matriz; 16384 bf16 mede 309 W, contra 283 em 8192")
    ap.add_argument("--stage-minutes", type=float, default=5.0)
    ap.add_argument("--rounds", type=int, default=99)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("sem GPU")
    args.log.parent.mkdir(parents=True, exist_ok=True)
    arquivo = args.log.open("a", buffering=1)

    def registrar(reg: dict) -> None:
        reg["t"] = agora()
        arquivo.write(json.dumps(reg, ensure_ascii=False) + "\n")
        arquivo.flush()
        os.fsync(arquivo.fileno())

    tipo = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.backends.cuda.matmul.allow_tf32 = True
    a = torch.randn(args.size, args.size, device="cuda", dtype=tipo)
    b = torch.randn(args.size, args.size, device="cuda", dtype=tipo)
    torch.cuda.synchronize()

    registrar({"evento": "inicio", "pid": os.getpid(), "size": args.size,
               "dtype": args.dtype, "estagios": ESTAGIOS,
               "minutos_por_estagio": args.stage_minutes, "rounds": args.rounds,
               "memoria_mib": round(torch.cuda.memory_allocated() / 2**20, 1)})

    ciclo_global = 0
    try:
        for rodada in range(1, args.rounds + 1):
            for i, (t_rajada, t_ocioso) in enumerate(ESTAGIOS, 1):
                fim_estagio = time.monotonic() + args.stage_minutes * 60
                registrar({"evento": "estagio_inicio", "rodada": rodada, "estagio": i,
                           "rajada_s": t_rajada, "ocioso_s": t_ocioso,
                           "degraus_por_min": round(60 / (t_rajada + t_ocioso), 1)})
                ciclos = 0
                while time.monotonic() < fim_estagio:
                    t0 = time.monotonic()
                    registrar({"evento": "sobe", "rodada": rodada, "estagio": i,
                               "ciclo": ciclo_global})
                    its = rajada(a, b, t0 + t_rajada)
                    registrar({"evento": "desce", "rodada": rodada, "estagio": i,
                               "ciclo": ciclo_global, "matmuls": its})
                    time.sleep(t_ocioso)
                    ciclos += 1
                    ciclo_global += 1
                registrar({"evento": "estagio_fim", "rodada": rodada, "estagio": i,
                           "ciclos": ciclos, "ciclo_global": ciclo_global})
    except RuntimeError as erro:
        registrar({"EVENTO": "ERRO_CUDA", "detalhe": str(erro)[:800],
                   "ciclo_global": ciclo_global})
        raise
    finally:
        registrar({"evento": "fim", "ciclos_totais": ciclo_global})
        arquivo.close()


if __name__ == "__main__":
    main()
