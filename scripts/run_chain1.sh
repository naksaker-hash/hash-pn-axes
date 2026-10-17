#!/bin/zsh
cd ~/Downloads/HASH_DL_morph
PY=~/Downloads/M31_PN_YOLO/.venv/bin/python
$PY scripts/08_pa_regressor.py cv > notes/pa_cnn_cv.log 2>&1 && echo "pa cv done" >> notes/chain1.log
$PY scripts/08_pa_regressor.py full > notes/pa_cnn_full.log 2>&1 && echo "pa full done" >> notes/chain1.log
./scripts/run_round1.sh && echo "round1 done" >> notes/chain1.log
