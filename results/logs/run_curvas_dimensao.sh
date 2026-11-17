#!/bin/bash
# Curvas de desempenho x numero de features com dimensoes casadas (regime exploratorio).
cd /exp/duducampos/InterpretECG
GRADE="2150 1075 537 268 134 67 33"
.venv/bin/python -u scripts/prune_features.py --families tsfresh --schedule 4449 $GRADE \
    --n-jobs 16 --output results/pruning_tsfresh_exploratory > results/logs/prune_tsfresh.log 2>&1
.venv/bin/python -u scripts/prune_features.py --families catch22 series_amplitude series_interlead \
    graph_hvg graph_lphvg1 graph_hvg_line graph_lphvg1_line graph_hvg_line_mv graph_lphvg1_line_mv \
    graph_hvg_dir graph_lphvg1_dir graph_hvg_dir_mv graph_lphvg1_dir_mv graph_hvg_multiplex graph_lphvg1_multiplex \
    --schedule $GRADE --n-jobs 16 --output results/pruning_bestmx_exploratory > results/logs/prune_bestmx.log 2>&1
