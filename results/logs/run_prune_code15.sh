#!/bin/bash
cd /exp/duducampos/InterpretECG
export OMP_NUM_THREADS=8
for c in melhor_multiplex tsfresh; do
  .venv/bin/python -u scripts/prune_external.py code15 $c --n-jobs 8 > results/logs/prune_code15_$c.log 2>&1
  echo "code15 $c exit $?"
done
