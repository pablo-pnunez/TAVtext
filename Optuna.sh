#!/bin/bash
MAXTSTS=8

for TEST in $(seq 1 $MAXTSTS) ;do

  nohup venv/bin/python3 -u  Main_opt.py > "scripts/optuna/test_"$TEST".txt" &
  sleep 2
done
