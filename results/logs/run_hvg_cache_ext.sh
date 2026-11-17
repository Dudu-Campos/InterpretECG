#!/bin/bash
cd /exp/duducampos/InterpretECG
for d in chapman code15; do
  .venv/bin/python -u scripts/build_hvg_cache.py --dataset $d --n-jobs 14 --output data/features/graphs/hvg_edges_$d.npz > results/logs/hvg_cache_$d.log 2>&1
  echo "$d exit $? $(date)" >> results/logs/run_hvg_cache_ext.log
done
