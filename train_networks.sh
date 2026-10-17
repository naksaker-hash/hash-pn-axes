#!/bin/sh
# Every axis network run used in the MLST paper beyond the base run of run_all.sh (GPU; Apple MPS was used).
# Outputs go to data/pa_cnn_cv<TAG>.csv/.json (cross validation) or data/pa_cnn_full<TAG>.* (all 350 labels).
set -e
PY=${PY:-python}
cv()   { env "$@" $PY scripts/08_pa_regressor.py cv; }
full() { env "$@" $PY scripts/08_pa_regressor.py full; }
# seeds 1 to 4 and three further final networks (scatter of the catalogue axes)
for s in 1 2 3 4; do cv PA_TAG=_s$s PA_SEED=$s; done
for s in 1 2; do full PA_TAG=_s$s PA_SEED=$s; done
# two further assignments of the longitude blocks to folds
for fs in 1 2; do cv PA_TAG=_fold$fs PA_FOLDSEED=$fs; done
# ablations, three seeds each: no augmentation, untransformed labels, Halpha only, from scratch, full cutout rotated
cv PA_TAG=_noaug PA_AUG=none; cv PA_TAG=_rotnolabel PA_AUG=rot_nolabel; cv PA_TAG=_ha PA_CH=ha
for s in 1 2; do cv PA_TAG=_noaug_s$s PA_AUG=none PA_SEED=$s; cv PA_TAG=_ha_s$s PA_CH=ha PA_SEED=$s; done
for s in 0 1 2; do cv PA_TAG=_scratch_s$s PA_PRETRAIN=0 PA_SEED=$s; cv PA_TAG=_rotfull_s$s PA_ROTFULL=1 PA_SEED=$s; done
# alternatives: von Mises head and D8 steerable CNN (escnn), three seeds each
for s in 0 1 2; do cv PA_TAG=_vm_s$s PA_HEAD=vm PA_SEED=$s; cv PA_TAG=_d8_s$s PA_ARCH=d8 PA_SEED=$s; done
cv PA_TAG=_s0d8tta PA_SEED=0
# label checks: without the MASH labels and without the ten labels of exactly 0 deg
cv PA_TAG=_nomash PA_EXCL=mash; full PA_TAG=_nomash PA_EXCL=mash; cv PA_TAG=_nozero PA_EXCL=zero
# learning curve
for f in 0.25 0.5 0.75; do t=$(echo $f | sed 's/0\.//; s/^5$/50/'); cv PA_TAG=_f$t PA_FRAC=$f; done
