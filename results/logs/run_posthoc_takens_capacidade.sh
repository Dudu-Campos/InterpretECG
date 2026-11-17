#!/bin/bash
# Varredura de CAPACIDADE do post-hoc dos vaos (chapman8), semente 42.
# Motivo: dropout 0.1 -> 0.3 nao mudou a forma da curva (pico na epoca ~15 e
# queda depois, nas duas configuracoes), e o pico mudou +0.0006. O excesso esta
# no numero de matrizes de mensagem (12 faixas x 2 sentidos = 24 relacoes),
# nao na coadaptacao das ativacoes.
#
# A ESCOLHA DA CONFIGURACAO E FEITA NO FOLD 8. O numero do fold 9 e gravado
# pelo script mas nao entra na decisao -- os dois ficam no JSON para auditoria.
#
# Parametros (hidden, bins) -> M de parametros:
#   64/0  0.538   64/12 0.910   64/6  0.707
#   32/0  0.236   32/12 0.332   32/6  0.280
# dropout e wd voltam aos originais (0.1 / 1e-2): empilhar o dropout 0.3
# confundiria duas mudancas, e ele ja mostrou nao ser a alavanca.
cd /exp/duducampos/InterpretECG
OUT=results/external/chapman8/posthoc_takens/gnn_capacidade
mkdir -p $OUT
for cfg in "32 12" "32 6" "64 6" "32 0"; do
  set -- $cfg; h=$1; b=$2
  tag=takens_dir_d${h}_l4_e50_bins${b}_test9_val8_seed42
  [ -f $OUT/$tag.json ] && continue
  CUDA_VISIBLE_DEVICES=${GPU:-0} .venv/bin/python -u scripts/train_line_gnn.py --dataset chapman8 --graph takens \
    --cache data/features/graphs/takens_edges_chapman.npz --directed --epochs 50 --hidden $h --layers 4 \
    --dropout 0.1 --wd 1e-2 --span-bins $b --seed 42 --test-fold 9 --val-fold 8 \
    --output $OUT --tag $tag > results/logs/posthoc_cap_d${h}_bins${b}.log 2>&1
  echo "hidden $h bins $b exit $? $(date)"
done
echo "varredura de capacidade completa $(date)"
