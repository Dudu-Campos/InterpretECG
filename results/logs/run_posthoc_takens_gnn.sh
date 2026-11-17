#!/bin/bash
# Post-hoc da GNN na rede de Takens (chapman8), regime exploratorio: treino 1-7,
# validacao 8 (escolha da epoca), teste 9. O fold 10 nao e lido.
# A: modelo original (--span-bins 0) e B: vao em 12 faixas por sentido.
#
# Regularizacao aumentada em 15/nov nos DOIS bracos (dropout 0.1 -> 0.3,
# wd 1e-2 -> 5e-2), depois de o braco B sobreajustar na primeira rodada:
# perda de treino 0,005 na epoca 39 com a validacao estacionada desde a ~13.
# Decisao tomada olhando so o fold 8; ver results/external/PREREGISTRATION.md.
# Mudar os dois mantem a comparacao isolando os vaos, que e a hipotese em teste.
#
# 3 sementes cada. Uso: GPU=0 results/logs/run_posthoc_takens_gnn.sh  (retomavel)
cd /exp/duducampos/InterpretECG
OUT=results/external/chapman8/posthoc_takens/gnn
DROPOUT=0.3
WD=5e-2
mkdir -p $OUT
for s in 42 43 44; do
  for b in 0 12; do
    tag=takens_dir_d64_l4_e50_do30_wd5e2_bins${b}_test9_val8_seed$s
    [ -f $OUT/$tag.json ] && continue
    CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -u scripts/train_line_gnn.py --dataset chapman8 --graph takens \
      --cache data/features/graphs/takens_edges_chapman.npz --directed --epochs 50 --hidden 64 --layers 4 \
      --dropout $DROPOUT --wd $WD \
      --span-bins $b --seed $s --test-fold 9 --val-fold 8 --output $OUT --tag $tag \
      > results/logs/posthoc_takens_do30_bins${b}_seed$s.log 2>&1
    echo "bins $b semente $s exit $? $(date)"
  done
done
