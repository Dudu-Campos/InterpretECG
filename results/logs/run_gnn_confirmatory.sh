#!/bin/bash
# Rodada confirmatoria das GNNs (results/confirmatory/PREREGISTRATION.md, secao 2). Uma por vez na GPU (duas estouraram a memoria de 12 GB).
cd /exp/duducampos/InterpretECG
RAZAO="GNN em L(HVG) %s semente %s: configuracao congelada no pre-registro de 09/nov/2026 (results/confirmatory/PREREGISTRATION.md), identica a exploratoria; epoca escolhida no fold 9, fold 10 nao usado para selecao; decisao do Eduardo 09/nov/2026"
tarefas=()
for s in 42 43 44 45 46; do tarefas+=("dir $s"); done
for s in 42 43 44 45 46; do tarefas+=("nodir $s"); done
printf '%s\n' "${tarefas[@]}" | xargs -P 1 -L 1 bash -c '
  v=$0; s=$1
  if [ "$v" = dir ]; then flag=--directed; tag=hvg_dir_d64_l4_e50_conf_seed$s; nome=direcionada; else flag=; tag=hvg_d64_l4_e50_conf_seed$s; nome="nao direcionada"; fi
  razao=$(printf "'"$RAZAO"'" "$nome" "$s")
  .venv/bin/python -u scripts/train_line_gnn.py $flag --epochs 50 --hidden 64 --layers 4 --seed $s \
     --test-fold 10 --val-fold 9 --confirm-test-fold "$razao" \
     --output results/line_gnn_confirmatory --tag $tag > results/logs/gnn_conf_$tag.log 2>&1
'
