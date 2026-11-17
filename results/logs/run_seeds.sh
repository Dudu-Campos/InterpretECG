#!/bin/bash
# 5 sementes por modelo no regime exploratorio 1-7 / 8 / 9 (fold 10 nao lido).
# 42 = rodadas originais; 43-46 novas. Baselines novos (inception1d, lstm_bidir,
# transformer) rodam as 5. Ordem: baratos primeiro, GNNs por ultimo.
cd /exp/duducampos/InterpretECG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 1064605 2>/dev/null; do sleep 60; done
falha() { echo "$(date -Is) $1" >> results/logs/seeds_falhas.log; }
CACHE=data/features/cnn_signals_100hz.npz
for s in 43 44 45 46; do
  d=results/cnn_xresnet1d101_exploratory_seed$s
  [ -f $d/metrics.json ] || .venv/bin/python -u scripts/train_cnn.py --cache $CACHE --seed $s --output $d > results/logs/seeds_cnn_$s.log 2>&1 || falha "cnn $s"
done
for a in inception1d lstm_bidir transformer; do
  for s in 42 43 44 45 46; do
    d=results/deep_${a}_exploratory_seed$s
    [ -f $d/metrics.json ] || .venv/bin/python -u scripts/train_cnn.py --arch $a --cache $CACHE --seed $s --output $d > results/logs/seeds_${a}_$s.log 2>&1 || falha "$a $s"
  done
done
for s in 43 44 45 46; do
  [ -f results/imn_exploratory/multilabel5_seed$s.json ] || .venv/bin/python -u scripts/train_imn.py --seed $s > results/logs/seeds_imn_$s.log 2>&1 || falha "imn $s"
done
for s in 43 44 45 46; do
  t=hvg_dir_d64_l4_e50_test9_val8_seed$s
  [ -f results/line_gnn_cv_exploratory/$t.json ] || .venv/bin/python -u scripts/train_line_gnn.py --directed --epochs 50 --seed $s --tag $t --output results/line_gnn_cv_exploratory > results/logs/seeds_gnn_dir_$s.log 2>&1 || falha "gnn_dir $s"
done
for s in 43 44 45 46; do
  t=hvg_d64_l4_e50_test9_val8_seed$s
  [ -f results/line_gnn_cv_exploratory/$t.json ] || .venv/bin/python -u scripts/train_line_gnn.py --epochs 50 --seed $s --tag $t --output results/line_gnn_cv_exploratory > results/logs/seeds_gnn_$s.log 2>&1 || falha "gnn $s"
done
