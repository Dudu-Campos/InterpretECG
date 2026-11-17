#!/bin/bash
# Fila da GPU para a tabela final (pre-registro externo, acrescimo de 13/nov/2026).
# Retomavel: pula o que ja tem resultado. Ordem: baratos primeiro, GNN do CODE por ultimo.
cd /exp/duducampos/InterpretECG
PY=.venv/bin/python
LOG=results/logs/run_deep_externos.log
MOT="configuracao congelada do PTB-XL sem ajuste, pre-registro externo (acrescimo de 13/nov/2026); decisao do Eduardo 13/nov/2026"

cnn() {  # ds arch seed
  out=results/external/$1/deep/$2_confirmatory_seed$3
  [ -f $out/metrics.json ] && return
  $PY -u scripts/train_cnn.py --dataset $1 --arch $2 --seed $3 --regime confirmatory --output $out \
    --confirm-test-fold "baseline $2 [$1] semente $3: $MOT" > results/logs/deep_$1_$2_seed$3.log 2>&1
  echo "cnn $1 $2 $3 exit $? $(date)" >> $LOG
}
imn() {  # ds seed
  suf=""; [ $2 != 42 ] && suf="_seed$2"
  [ -f results/external/$1/deep/imn/$1_multilabel5_confirmatory$suf.json ] && return
  $PY -u scripts/train_imn.py --dataset $1 --seed $2 --regime confirmatory \
    --confirm-test-fold "baseline ECG-IMN [$1] semente $2: $MOT" > results/logs/deep_$1_imn_seed$2.log 2>&1
  echo "imn $1 $2 exit $? $(date)" >> $LOG
}
gnn() {  # ds graph seed   (graph: line = L(HVG); takens = rede de recorrencia)
  if [ $2 = takens ]; then cache=data/features/graphs/takens_edges_${1/chapman8/chapman}.npz; g="--graph takens"; tag=takens_dir_d64_l4_e50_conf_seed$3
  else cache=data/features/graphs/hvg_edges_${1/chapman8/chapman}.npz; g=""; tag=hvg_dir_d64_l4_e50_conf_seed$3; fi
  out=results/external/$1/deep/gnn
  [ -f $out/$tag.json ] && return
  until [ -f $cache ]; do sleep 300; done
  $PY -u scripts/train_line_gnn.py --dataset $1 --cache $cache $g --directed --epochs 50 --hidden 64 --layers 4 \
    --seed $3 --test-fold 10 --val-fold 9 --output $out --tag $tag \
    --confirm-test-fold "GNN $2 direcionada [$1] semente $3: $MOT" > results/logs/deep_$1_gnn_$2_seed$3.log 2>&1
  echo "gnn $1 $2 $3 exit $? $(date)" >> $LOG
}

# espera a rodada que a fila anterior deixou em andamento
[ -n "$ESPERA" ] && while kill -0 $ESPERA 2>/dev/null; do sleep 30; done

for ds in chapman8 code15; do
  for a in xresnet1d101 inception1d lstm_bidir transformer; do for s in 42 43 44 45 46; do cnn $ds $a $s; done; done
  for s in 42 43 44 45 46; do imn $ds $s; done
done
for s in 42 43 44 45 46; do gnn chapman8 takens $s; done      # melhor visao do chapman8
for s in 42 43 44 45 46; do gnn chapman8 line $s; done        # L(HVG), para a discussao
# code15: melhor visao decidida apos a extracao das familias extras
for s in 42 43 44 45 46; do gnn code15 line $s; done
echo "fila completa $(date)" >> $LOG
