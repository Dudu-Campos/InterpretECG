#!/bin/bash
cd /exp/duducampos/InterpretECG
export OMP_NUM_THREADS=6
for c in melhor_multiplex tsfresh; do
  .venv/bin/python -u scripts/prune_external.py chapman $c --n-jobs 6 > results/logs/prune_chapman_$c.log 2>&1
  echo "$c exit $?"
done
