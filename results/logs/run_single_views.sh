#!/bin/bash
cd /exp/duducampos/InterpretECG
.venv/bin/python -u scripts/external_single_views.py chapman8 --n-jobs 6 --threads 2 > results/logs/single_views_chapman8.log 2>&1
.venv/bin/python -u scripts/external_single_views.py code15 --n-jobs 4 --threads 3 > results/logs/single_views_code15.log 2>&1
