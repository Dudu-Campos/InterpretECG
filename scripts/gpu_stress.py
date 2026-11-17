"""Rampa de estresse: treina a GNN do line graph com configuracoes crescentes.

Medicao que define o desenho (sonda de 1 epoca, 09/nov): a configuracao BASE
(d64 l4 b32) ja chega a 344,8 W de 350 W, 76 C e 100% de utilizacao, com as
razoes de throttle 0x4 (limite de potencia) e 0x20 (limite termico) ativas.
Ou seja, potencia e temperatura ja estao no teto; o eixo que ainda tem folga e
memoria — 17,7 GB de 24,6 GB. Entao a rampa cresce `hidden`, `layers` e
`batch`, que e o que empurra a memoria, ate a placa recusar.

A rampa termina em configuracoes que devem faltar memoria. Isso e proposital e
**nao** e queda: OOM e excecao limpa do PyTorch, a GPU segue respondendo. O
runner separa os dois casos lendo o stderr, e so chama de queda o que o jornal
confirmar (nvidia-smi sem resposta, ou Xid no kernel).

Cada degrau roda um treino completo e curto de verdade, nao um laco sintetico:
se a placa aguentar, sobra resultado util em vez de calor.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# (hidden, layers, batch) -- cresce o que ocupa memoria
RAMPA = [
    (64, 4, 32),
    (96, 5, 32),
    (128, 6, 32),
    (128, 6, 48),
    (192, 8, 48),
    (256, 8, 64),
]


def agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def classificar(codigo: int, stderr: str) -> str:
    """OOM, erro de CUDA e queda sao coisas diferentes."""
    baixo = stderr.lower()
    if codigo == 0:
        return "ok"
    if "out of memory" in baixo or "outofmemoryerror" in baixo:
        return "sem_memoria"
    if "cuda error" in baixo or "device-side assert" in baixo:
        return "erro_cuda"
    if "unable to determine the device handle" in baixo or "no cuda-capable device" in baixo:
        return "gpu_sumiu"
    return "erro_outro"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", type=Path,
                    default=Path("/var/tmp/interpretecg_stress_runs.jsonl"))
    ap.add_argument("--mirror", type=Path,
                    default=ROOT / "results/gpu_stress/stress_runs.jsonl")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=99,
                    help="quantas vezes repetir a rampa inteira")
    ap.add_argument("--output", type=Path, default=Path("/var/tmp/gpu_stress_runs"))
    args = ap.parse_args()

    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    arquivo = args.log.open("a", buffering=1)

    def registrar(reg: dict) -> None:
        reg["timestamp"] = agora()
        arquivo.write(json.dumps(reg, ensure_ascii=False) + "\n")
        arquivo.flush()
        os.fsync(arquivo.fileno())
        try:
            import shutil
            shutil.copy2(args.log, args.mirror)
        except OSError:
            pass

    registrar({"evento": "rampa_inicio", "pid": os.getpid(),
               "rampa": [{"hidden": h, "layers": l, "batch": b} for h, l, b in RAMPA],
               "epochs_por_degrau": args.epochs, "rounds": args.rounds})

    for rodada in range(1, args.rounds + 1):
        for i, (hidden, layers, batch) in enumerate(RAMPA, 1):
            tag = f"stress_r{rodada}_d{hidden}_l{layers}_b{batch}"
            cmd = [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/train_line_gnn.py"),
                   "--epochs", str(args.epochs), "--hidden", str(hidden),
                   "--layers", str(layers), "--batch", str(batch),
                   "--tag", tag, "--output", str(args.output)]
            registrar({"evento": "degrau_inicio", "rodada": rodada, "degrau": i,
                       "hidden": hidden, "layers": layers, "batch": batch, "tag": tag})
            t0 = time.monotonic()
            r = subprocess.run(cmd, capture_output=True, text=True)
            decorrido = time.monotonic() - t0
            situacao = classificar(r.returncode, r.stderr)

            ultima = ""
            for linha in reversed(r.stdout.strip().splitlines()):
                if linha.startswith("epoca") or linha.startswith("val "):
                    ultima = linha.strip()
                    break
            registrar({"evento": "degrau_fim", "rodada": rodada, "degrau": i,
                       "hidden": hidden, "layers": layers, "batch": batch,
                       "situacao": situacao, "returncode": r.returncode,
                       "segundos": round(decorrido, 1), "ultima_linha": ultima,
                       "stderr_fim": r.stderr.strip()[-600:] if r.returncode else ""})

            if situacao in ("erro_cuda", "gpu_sumiu"):
                registrar({"evento": "rampa_abortada", "motivo": situacao,
                           "detalhe": "o jornal da GPU tem o estado no momento da falha"})
                arquivo.close()
                sys.exit(1)

    registrar({"evento": "rampa_fim"})
    arquivo.close()


if __name__ == "__main__":
    main()
