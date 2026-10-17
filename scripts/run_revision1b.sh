#!/bin/zsh
# revision round 1b: seeds and ablations (rerun after the env word-splitting bug)
cd ~/Downloads/HASH_DL_morph
PY=~/Downloads/M31_PN_YOLO/.venv/bin/python
run() { echo "$(date +%H:%M) start $1" >> notes/rev1b.log; env ${=2} $PY scripts/08_pa_regressor.py $3 > notes/pa_$1.log 2>&1 && echo "$(date +%H:%M) done $1" >> notes/rev1b.log || echo "$(date +%H:%M) FAIL $1" >> notes/rev1b.log; }
run s1_cv "PA_TAG=_s1 PA_SEED=1" cv
run s2_cv "PA_TAG=_s2 PA_SEED=2" cv
run noaug_cv "PA_TAG=_noaug PA_AUG=none" cv
run rotnolabel_cv "PA_TAG=_rotnolabel PA_AUG=rot_nolabel" cv
run ha_cv "PA_TAG=_ha PA_CH=ha" cv
echo "$(date +%H:%M) ALL DONE" >> notes/rev1b.log
