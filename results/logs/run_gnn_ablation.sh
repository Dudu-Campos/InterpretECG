#!/bin/bash
# Ablacao HVG x L(HVG) (results/ablation_gnn/REGISTRO.md). Espera a fila confirmatoria liberar a GPU.
cd /exp/duducampos/InterpretECG
while kill -0 1302116 2>/dev/null; do sleep 120; done
echo "GPU livre $(date)"
roda() {  # graph directed seed outdir tag
  [ -f $4/$5.json ] && { echo "ja existe $5"; return; }
  .venv/bin/python -u scripts/train_line_gnn.py --graph $1 $2 --epochs 50 --hidden 64 --layers 4 --seed $3 \
    --test-fold 9 --val-fold 8 --output $4 --tag $5 > results/logs/gnn_abl_$5.log 2>&1
  echo "$5 exit $? $(date)"
}
A=results/ablation_gnn; L=results/line_gnn_cv_exploratory
for s in 42 43 44; do roda hvg "" $s $A hvgnode_d64_l4_e50_test9_val8_seed$s; done
for s in 42 43 44; do roda hvg --directed $s $A hvgnode_dir_d64_l4_e50_test9_val8_seed$s; done
for s in 43 44; do roda line "" $s $L hvg_d64_l4_e50_test9_val8_seed$s; done
roda line --directed 44 $L hvg_dir_d64_l4_e50_test9_val8_seed44
echo "ablacao completa $(date)"
