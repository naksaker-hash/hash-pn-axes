#!/bin/bash
# ResNet18 from scratch (no ImageNet weights), three seeds, to separate pretraining from equivariance. Log notes/rev6b.log
cd "$(dirname "$0")/.."
for s in 0 1 2; do tag=_scratch_s$s; [ -f data/pa_cnn_cv$tag.json ] && continue
  echo "=== $(date '+%H:%M:%S') $tag" >> notes/rev6b.log; PA_PRETRAIN=0 PA_SEED=$s PA_TAG=$tag .venv/bin/python scripts/08_pa_regressor.py cv >> notes/rev6b.log 2>&1; done
echo "=== $(date '+%H:%M:%S') done" >> notes/rev6b.log
