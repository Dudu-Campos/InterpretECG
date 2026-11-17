#!/bin/bash
# Fila da maquina de 128 threads (guia: results/logs/CODE15_EXTRACAO.md). Retomavel em todas as etapas.
#  1. Chapman: familias descartadas no PTB-XL (--extra) e poda "todas_familias" (chapman e chapman8)
#  2. espera o preparo do CODE-15% terminar na refri
#  3. CODE-15%: extracao (15 familias + tsfresh) e podas melhor_multiplex e tsfresh (run_code15_full.sh)
#  4. CODE-15%: familias extras e poda "todas_familias"
# Nao le o fold 10 de nenhum dataset: o ajuste final (evaluate_external.py) e feito depois, uma vez.
cd /exp/duducampos/InterpretECG
set -x
NJOBS_EXTRACT=${NJOBS_EXTRACT:-120}
NJOBS_PRUNE=${NJOBS_PRUNE:-100}
PY=${PY:-.venv/bin/python}

poda() {  # dataset conjunto
  [ -f results/external/$1/pruning_$2/selected.json ] && return 0
  OMP_NUM_THREADS=$NJOBS_PRUNE $PY -u scripts/prune_external.py $1 $2 --n-jobs $NJOBS_PRUNE \
    > results/logs/prune_$1_$2.log 2>&1 || { echo "poda $1 $2 falhou"; exit 1; }
  echo "poda $1 $2 ok $(date)"
}

# 1. Chapman
OMP_NUM_THREADS=1 nice -n 5 $PY -u scripts/extract_external_features.py --extra \
  --signals data/external/chapman/signals.npz --output data/features/external/chapman \
  --n-jobs $NJOBS_EXTRACT --chunk 20 --shard-size 5000 >> results/logs/extract_chapman_extra.log 2>&1 || exit 1
echo "chapman extra ok $(date)"
poda chapman todas_familias
poda chapman8 todas_familias

# 2. espera o preparo do CODE-15% (feito na refri, ETAPA=download)
until grep -q "ETAPA=download concluida" results/logs/run_code15_download.log 2>/dev/null; do sleep 300; done
echo "CODE-15% preparado $(date)"

# 3. CODE-15%: extracao principal e podas
TRAVA=data/features/external/code15.LOCK
if [ -f $TRAVA ] && [ "$(cat $TRAVA)" != "$(hostname)" ]; then
  echo "extracao do CODE-15% reservada por $(cat $TRAVA); apague $TRAVA se essa maquina parou"; exit 1
fi
hostname > $TRAVA
OMP_NUM_THREADS=1 nice -n 5 $PY -u scripts/extract_external_features.py \
  --signals data/external/code15/signals.npz --output data/features/external/code15 \
  --n-jobs $NJOBS_EXTRACT --tsfresh --shard-size 10000 >> results/logs/extract_code15_full.log 2>&1 || exit 1
echo "code15 extract ok $(date)"
poda code15 melhor_multiplex
poda code15 tsfresh

# 4. CODE-15%: familias extras e poda a partir de todas
OMP_NUM_THREADS=1 nice -n 5 $PY -u scripts/extract_external_features.py --extra \
  --signals data/external/code15/signals.npz --output data/features/external/code15 \
  --n-jobs $NJOBS_EXTRACT --chunk 20 --shard-size 10000 >> results/logs/extract_code15_extra.log 2>&1 || exit 1
echo "code15 extra ok $(date)"
poda code15 todas_familias
rm -f $TRAVA
echo "fila da maquina de 128 threads completa $(date)"
