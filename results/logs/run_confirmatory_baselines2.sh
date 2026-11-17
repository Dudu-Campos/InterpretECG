#!/bin/bash
# Segunda leva no fold 10 (decisao do Eduardo, 09/nov/2026): transformer (receita do
# benchmark + lr padrao do DeepECG-Kit, declarado) e ECG-IMN (early stopping no fold 9,
# protocolo publicado, declarado). Espera a primeira leva terminar.
cd /exp/duducampos/InterpretECG
while kill -0 1191370 2>/dev/null; do sleep 60; done
CACHE=data/features/cnn_signals_100hz.npz
falha() { echo "$(date -Is) $1" >> results/logs/confirmatory_falhas.log; }
for s in 42 43 44 45 46; do
  d=results/deep_transformer_confirmatory_seed$s
  [ -f $d/metrics.json ] || .venv/bin/python -u scripts/train_cnn.py --regime confirmatory --arch transformer --seed $s --cache $CACHE --output $d \
    --confirm-test-fold "baseline transformer semente $s: arquitetura e padroes do DeepECG-Kit@5ecd8c17, receita do benchmark Strodthoff com lr 1e-3 (padrao do kit) fixado a priori, 50 epocas sem early stopping; decisao do Eduardo 09/nov/2026, declarada no artigo" > results/logs/confirm_transformer_$s.log 2>&1 || falha "transformer $s"
done
for s in 42 43 44 45 46; do
  f=results/imn_exploratory/multilabel5_confirmatory$([ $s != 42 ] && echo _seed$s).json
  [ -f $f ] || .venv/bin/python -u scripts/train_imn.py --regime confirmatory --seed $s --cache $CACHE \
    --confirm-test-fold "baseline ECG-IMN semente $s: configuracao do repositorio original (mesomorphic_ecg@53d9bc74), early stopping no AUROC do fold 9 como no protocolo publicado, fold 10 nao usado para selecao; decisao do Eduardo 09/nov/2026, declarada no artigo" > results/logs/confirm_imn_$s.log 2>&1 || falha "imn $s"
done
