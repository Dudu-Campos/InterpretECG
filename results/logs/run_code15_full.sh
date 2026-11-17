#!/bin/bash
# CODE-15% inteiro (results/external/PREREGISTRATION.md, 11/nov). Retomavel: cada etapa pula o que ja existe.
# Variaveis de ambiente:
#   ETAPA=download   para depois do preparo (download + descompactacao + prepare_external)
#   ETAPA=extract    so extracao + poda (exige data/external/code15/signals.npz)
#   ETAPA=all        tudo (padrao)
#   NJOBS_EXTRACT    processos da extracao (padrao 14)
#   NJOBS_PRUNE      processos da permutacao na poda (padrao 10)
# Guia de execucao em outra maquina: results/logs/CODE15_EXTRACAO.md
cd /exp/duducampos/InterpretECG
set -x
ETAPA=${ETAPA:-all}
NJOBS_EXTRACT=${NJOBS_EXTRACT:-14}
NJOBS_PRUNE=${NJOBS_PRUNE:-10}

if [ "$ETAPA" != extract ]; then
  for p in $(seq 0 17); do
    [ -f data/code15/exams_part$p.hdf5 ] && continue
    for tentativa in 1 2 3 4 5 6; do
      curl -sL -C - --retry 5 -o data/code15/exams_part$p.zip "https://zenodo.org/records/4916206/files/exams_part$p.zip?download=1"
      unzip -tq data/code15/exams_part$p.zip && break
      echo "parte $p incompleta/corrompida (tentativa $tentativa)"; sleep 60
    done
    unzip -tq data/code15/exams_part$p.zip || { echo "parte $p falhou"; exit 1; }
    (cd data/code15 && unzip -o -q exams_part$p.zip) && echo "parte $p ok"
  done
  [ -f data/external/code15/signals.npz ] || .venv/bin/python scripts/prepare_external.py code15 > results/logs/prepare_code15_full.log 2>&1 || exit 1
  echo "prepare ok $(date)"
  [ "$ETAPA" = download ] && { echo "ETAPA=download concluida"; exit 0; }
fi

[ -f data/external/code15/signals.npz ] || { echo "falta signals.npz: rode ETAPA=download antes"; exit 1; }
TRAVA=data/features/external/code15.LOCK
if [ -f $TRAVA ] && [ "$(cat $TRAVA)" != "$(hostname)" ]; then
  echo "extracao ja reservada por $(cat $TRAVA); apague $TRAVA se essa maquina parou"; exit 1
fi
mkdir -p data/features/external && hostname > $TRAVA
export OMP_NUM_THREADS=1
nice -n 5 .venv/bin/python -u scripts/extract_external_features.py --signals data/external/code15/signals.npz \
  --output data/features/external/code15 --n-jobs $NJOBS_EXTRACT --tsfresh --shard-size 10000 \
  >> results/logs/extract_code15_full.log 2>&1 || exit 1
echo "extract ok $(date)"
export OMP_NUM_THREADS=$NJOBS_PRUNE  # o ajuste do boosting usa OpenMP; a permutacao limita cada worker a 1 thread
for c in melhor_multiplex tsfresh; do
  [ -f results/external/code15/pruning_$c/selected.json ] && continue
  .venv/bin/python -u scripts/prune_external.py code15 $c --n-jobs $NJOBS_PRUNE > results/logs/prune_code15_full_$c.log 2>&1 || exit 1
  echo "prune $c ok $(date)"
done
rm -f $TRAVA
echo "fila CODE-15% completa $(date)"
