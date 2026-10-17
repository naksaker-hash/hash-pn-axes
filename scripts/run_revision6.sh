#!/bin/bash
# Revision 6 (deep review) cross validation runs, resumable: a run is skipped when its output json exists.
# Usage: run_revision6.sh main | d8     (the two queues run in parallel); logs in notes/rev6.log and notes/rev6_d8.log
cd "$(dirname "$0")/.."
PY=.venv/bin/python
Q=${1:-main}; LOG=notes/rev6.log; [ "$Q" = d8 ] && LOG=notes/rev6_d8.log
run() { tag=$1; shift
  if [ -f "data/pa_cnn_cv$tag.json" ]; then echo "=== skip $tag" >> $LOG; return; fi
  echo "=== $(date '+%H:%M:%S') $tag $*" >> $LOG; env "$@" PA_TAG=$tag $PY scripts/08_pa_regressor.py cv >> $LOG 2>&1; }
if [ "$Q" = main ]; then
  for s in 0 1 2; do run _vm_s$s PA_HEAD=vm PA_SEED=$s; done
  for s in 3 4; do run _s$s PA_SEED=$s; done
  run _s0d8tta PA_SEED=0
  for fs in 1 2; do run _fold$fs PA_FOLDSEED=$fs; done
  for s in 1 2; do run _noaug_s$s PA_AUG=none PA_SEED=$s; run _ha_s$s PA_CH=ha PA_SEED=$s; done
else
  for s in 0 1 2; do run _d8_s$s PA_ARCH=d8 PA_SEED=$s; done
fi
echo "=== $(date '+%H:%M:%S') done" >> $LOG
