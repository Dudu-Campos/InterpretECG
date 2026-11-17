#!/bin/bash
# Baselines de configuracao fixa no fold 10 (excecao de 29/out/2026, CLAUDE.md 6.1).
# Regime confirmatorio: treino 1-8, validacao 9 (so reportada), teste 10. Cada uso e
# registrado em results/test_fold_usage.jsonl pelo proprio train_cnn.py.
cd /exp/duducampos/InterpretECG
CACHE=data/features/cnn_signals_100hz.npz
falha() { echo "$(date -Is) $1" >> results/logs/confirmatory_falhas.log; }
J="configuracao publicada e congelada (Strodthoff et al. 2021, fastai_configs.py), 50 epocas sem early stopping; excecao de concorrente de configuracao fixa, CLAUDE.md 6.1"
for a in inception1d lstm_bidir; do
  for s in 42 43 44 45 46; do
    d=results/deep_${a}_confirmatory_seed$s
    [ -f $d/metrics.json ] || .venv/bin/python -u scripts/train_cnn.py --regime confirmatory --arch $a --seed $s --cache $CACHE --output $d \
      --confirm-test-fold "baseline $a semente $s, $J" > results/logs/confirm_${a}_$s.log 2>&1 || falha "$a $s"
  done
done
for s in 43 44 45 46; do
  d=results/cnn_xresnet1d101_seed$s
  [ -f $d/metrics.json ] || .venv/bin/python -u scripts/train_cnn.py --regime confirmatory --seed $s --cache $CACHE --output $d \
    --confirm-test-fold "baseline xresnet1d101 semente $s, $J" > results/logs/confirm_xresnet1d101_$s.log 2>&1 || falha "xresnet $s"
done
