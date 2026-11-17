"""Jornal da GPU, voltado a TRANSIENTE de carga, nao a carga sustentada.

Motivo do desenho (relato do Eduardo, 09/nov): a queda observada nao aconteceu
sob carga constante, e sim na transicao de tensao entre o fim de uma seed e o
inicio da proxima — um degrau de corrente, nao uma temperatura alta.

Tres consequencias:

1. **Contador, nao amostra.** Uma assercao do freio externo pode durar
   microssegundos e passar despercebida amostrando a 10 Hz. A placa mantem
   contadores acumulados em us (`clocks_event_reasons_counters.*`) que nao
   perdem evento. O jornal registra os contadores e grita quando algum sobe.
   Linha de base medida em 09/nov nesta 3090: `hw_power_brake_slowdown` em
   **0 us de vida util** — qualquer valor acima de zero e novo.
2. **`hw_power_brake_slowdown` e o indicador certo.** A documentacao da NVIDIA
   o descreve como "External Power Brake Assertion is triggered (e.g. by the
   system power supply)": e a protecao da FONTE agindo, que e o que um degrau
   de carga com queda de tensao produz. Nao confundir com `sw_power_cap`, que
   e o limite normal de 350 W e fica ativo o tempo todo sob carga.
3. **Fluxo continuo, nao um processo por amostra.** `nvidia-smi -lms` entrega
   amostras sem pagar a criacao de processo a cada leitura, o que permite
   100 ms de intervalo. A morte do fluxo e, ela mesma, o registro da queda.

A 3090 nao expoe tensao de nucleo (`nvidia-smi -q -d VOLTAGE` vem vazio em
placas de consumo), entao a tensao e observada indiretamente: potencia, clock
e o freio externo.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

CAMPOS = [
    "power.draw", "clocks.sm", "temperature.gpu", "utilization.gpu",
    "memory.used", "pstate",
    "clocks_event_reasons.sw_power_cap",
    "clocks_event_reasons.hw_slowdown",
    "clocks_event_reasons.hw_thermal_slowdown",
    "clocks_event_reasons.hw_power_brake_slowdown",
    "clocks_event_reasons.sw_thermal_slowdown",
    "clocks_event_reasons_counters.sw_power_cap",
    "clocks_event_reasons_counters.hw_power_brake_slowdown",
    "clocks_event_reasons_counters.hw_thermal_slowdown",
]
NUMERICOS = {"power.draw", "clocks.sm", "temperature.gpu", "utilization.gpu",
             "memory.used"}
CONTADORES = [c for c in CAMPOS if c.startswith("clocks_event_reasons_counters")]
CURTO = {
    "power.draw": "W", "clocks.sm": "sm", "temperature.gpu": "C",
    "utilization.gpu": "util", "memory.used": "mem", "pstate": "p",
    "clocks_event_reasons.sw_power_cap": "cap",
    "clocks_event_reasons.hw_slowdown": "hw",
    "clocks_event_reasons.hw_thermal_slowdown": "hwterm",
    "clocks_event_reasons.hw_power_brake_slowdown": "BRAKE",
    "clocks_event_reasons.sw_thermal_slowdown": "swterm",
    "clocks_event_reasons_counters.sw_power_cap": "n_cap",
    "clocks_event_reasons_counters.hw_power_brake_slowdown": "N_BRAKE",
    "clocks_event_reasons_counters.hw_thermal_slowdown": "n_hwterm",
}


def agora() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def converter(nome: str, bruto: str):
    bruto = bruto.strip()
    if nome in NUMERICOS or nome in CONTADORES:
        try:
            return float(bruto)
        except ValueError:
            return None
    return bruto


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--journal", type=Path,
                    default=Path("/var/tmp/interpretecg_gpu_journal.jsonl"))
    ap.add_argument("--mirror", type=Path,
                    default=Path("/exp/duducampos/InterpretECG/results/gpu_stress/"
                                 "gpu_journal.jsonl"))
    ap.add_argument("--ms", type=int, default=100, help="intervalo de amostragem")
    ap.add_argument("--mirror-every", type=int, default=600)
    ap.add_argument("--max-hours", type=float, default=72.0)
    ap.add_argument("--quiet-samples", action="store_true",
                    help="so grava mudancas de estado e eventos, nao toda amostra")
    args = ap.parse_args()

    args.journal.parent.mkdir(parents=True, exist_ok=True)
    try:
        args.mirror.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    arquivo = args.journal.open("a", buffering=1)
    trava = threading.Lock()

    def escrever(reg: dict) -> None:
        reg["t"] = agora()
        with trava:
            arquivo.write(json.dumps(reg, ensure_ascii=False) + "\n")
            arquivo.flush()
            os.fsync(arquivo.fileno())

    consulta = ",".join(CAMPOS)
    cmd = ["nvidia-smi", f"--query-gpu={consulta}",
           "--format=csv,noheader,nounits", "-lms", str(args.ms)]
    escrever({"evento": "inicio", "ms": args.ms, "pid": os.getpid(), "campos": CAMPOS})

    processo = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1)
    base_contadores: dict[str, float] = {}
    anterior_estado = None
    recentes: deque = deque(maxlen=40)
    inicio = time.monotonic()
    n = 0

    try:
        for linha in processo.stdout:
            n += 1
            valores = [v.strip() for v in linha.strip().split(",")]
            if len(valores) != len(CAMPOS):
                escrever({"evento": "linha_malformada", "saida": linha.strip()[:300]})
                continue
            amostra = {CURTO[c]: converter(c, v) for c, v in zip(CAMPOS, valores)}

            # Primeira leitura fixa a base dos contadores de vida util.
            if not base_contadores:
                base_contadores = {CURTO[c]: amostra[CURTO[c]] or 0.0 for c in CONTADORES}
                escrever({"evento": "base_contadores", **base_contadores})

            # Um contador que sobe e o evento que estamos cacando.
            subiu = {}
            for c in CONTADORES:
                chave = CURTO[c]
                atual = amostra.get(chave)
                if atual is None:
                    continue
                delta = atual - base_contadores[chave]
                if delta > 0:
                    subiu[chave] = delta
                    base_contadores[chave] = atual
            if "N_BRAKE" in subiu:
                escrever({"EVENTO": "FREIO_EXTERNO_DA_FONTE",
                          "delta_us": subiu["N_BRAKE"], "amostra": amostra,
                          "antes": list(recentes)})
            elif subiu and any(k != "n_cap" for k in subiu):
                escrever({"EVENTO": "contador_subiu", "delta_us": subiu,
                          "amostra": amostra})

            estado = (amostra["cap"], amostra["hw"], amostra["hwterm"],
                      amostra["BRAKE"], amostra["swterm"], amostra["p"])
            if estado != anterior_estado:
                escrever({"evento": "mudanca_de_estado", **amostra})
                anterior_estado = estado
            elif not args.quiet_samples:
                escrever(amostra)

            recentes.append({k: amostra.get(k) for k in
                             ("W", "sm", "C", "util", "p", "BRAKE", "cap")})

            if n % args.mirror_every == 0:
                try:
                    shutil.copy2(args.journal, args.mirror)
                except OSError:
                    pass
            if time.monotonic() - inicio > args.max_hours * 3600:
                break
    except KeyboardInterrupt:
        pass
    finally:
        codigo = processo.poll()
        erro = ""
        try:
            processo.kill()
            erro = (processo.stderr.read() or "")[:500]
        except Exception:
            pass
        # O fluxo morrer sozinho e o registro da queda.
        escrever({"EVENTO": "fluxo_encerrou", "returncode": codigo,
                  "stderr": erro.strip(), "amostras": n,
                  "ultimas": list(recentes)})
        try:
            shutil.copy2(args.journal, args.mirror)
        except OSError:
            pass
        arquivo.close()


if __name__ == "__main__":
    main()
