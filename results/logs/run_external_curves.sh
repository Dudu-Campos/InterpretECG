#!/bin/bash
# Curvas erro x features dos externos no fold 9 (scripts/external_dimension_curves.py). Retomavel.
cd /exp/duducampos/InterpretECG
for d in chapman chapman8; do
  .venv/bin/python -u scripts/external_dimension_curves.py $d --n-jobs 4 --threads 4 > results/logs/external_curves_$d.log 2>&1
  echo "$d exit $? $(date)"
done
.venv/bin/python scripts/plot_external_curves_fold9.py && echo "figura ok $(date)"
