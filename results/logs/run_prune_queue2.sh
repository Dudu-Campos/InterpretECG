#!/bin/bash
# Fila reordenada (11/nov): espera a poda multi-view do chapman (4 grupos), depois chapman8 e o tsfresh do chapman.
cd /exp/duducampos/InterpretECG
export OMP_NUM_THREADS=6
while kill -0 1469222 2>/dev/null; do sleep 30; done
echo "chapman melhor_multiplex terminou"
for job in "chapman8 melhor_multiplex" "chapman8 tsfresh" "chapman tsfresh"; do
  set -- $job
  .venv/bin/python -u scripts/prune_external.py $1 $2 --n-jobs 6 > results/logs/prune_$1_$2.log 2>&1
  echo "$1 $2 exit $?"
done
