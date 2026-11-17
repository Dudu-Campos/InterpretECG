#!/bin/bash
# Generalizacao (results/external/PREREGISTRATION.md): download, preparo e extracao.
cd /exp/duducampos/InterpretECG
set -x
# download da parte 2 do CODE-15% em paralelo (rede), retomando o arquivo parcial
( curl -sL -C - -o data/code15/exams_part2.zip "https://zenodo.org/records/4916206/files/exams_part2.zip?download=1" \
  && unzip -tq data/code15/exams_part2.zip && echo "part2 ok" ) > results/logs/download_part2.log 2>&1 &
DL=$!
# Chapman: extracao (CPU)
nice -n 5 .venv/bin/python scripts/extract_external_features.py --signals data/external/chapman/signals.npz \
  --output data/features/external/chapman --n-jobs 13 --tsfresh > results/logs/extract_chapman.log 2>&1
echo "chapman extract exit $?"
wait $DL
grep -q "part2 ok" results/logs/download_part2.log || { echo "download part2 falhou"; exit 1; }
cd data/code15 && for p in 0 1 2; do [ -f exams_part$p.hdf5 ] || unzip -o -q exams_part$p.zip; done && cd ../..
.venv/bin/python scripts/prepare_external.py code15 > results/logs/prepare_code15.log 2>&1
echo "code15 prepare exit $?"
nice -n 5 .venv/bin/python scripts/extract_external_features.py --signals data/external/code15/signals.npz \
  --output data/features/external/code15 --n-jobs 13 --tsfresh > results/logs/extract_code15.log 2>&1
echo "code15 extract exit $?"
