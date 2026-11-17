#!/bin/bash
cd /exp/duducampos/InterpretECG
set -e
.venv/bin/python -u scripts/evaluate_external.py code15 --lists pruned --sets todas_familias
echo "eval ok $(date)"
.venv/bin/python -u scripts/boosting_seeds_fold10.py code15 --sets todas_familias --threads 4
echo "sementes ok $(date)"
