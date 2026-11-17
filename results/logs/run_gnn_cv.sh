#!/usr/bin/env bash
# GNN L(HVG), validacao cruzada em 5 rotacoes dentro dos folds 1-9 (fold 10 selado), 50 epocas.
cd /exp/duducampos/InterpretECG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for par in "9 8" "8 7" "7 6" "6 5" "5 4"; do
  set -- $par
  tag="hvg_d64_l4_e50_test${1}_val${2}"
  [ -f results/line_gnn_cv_exploratory/${tag}.json ] && continue
  .venv/bin/python -u scripts/train_line_gnn.py --epochs 50 --test-fold $1 --val-fold $2 \
      --tag $tag --output results/line_gnn_cv_exploratory > results/logs/gnn_cv_${tag}.log 2>&1 \
      || echo "FALHOU $tag" >> results/logs/gnn_cv_falhas.log
done
