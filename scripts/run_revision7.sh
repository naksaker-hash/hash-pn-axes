#!/bin/bash
# Revision 7: standard network with rotation of the full cutout before cropping (no padded corners), three seeds.
cd "$(dirname "$0")/.."
for s in 0 1 2; do tag=_rotfull_s$s; [ -f data/pa_cnn_cv$tag.json ] && continue
  echo "=== $(date '+%H:%M:%S') $tag" >> notes/rev7.log; PA_ROTFULL=1 PA_SEED=$s PA_TAG=$tag .venv/bin/python scripts/08_pa_regressor.py cv >> notes/rev7.log 2>&1; done
echo "=== $(date '+%H:%M:%S') done" >> notes/rev7.log
