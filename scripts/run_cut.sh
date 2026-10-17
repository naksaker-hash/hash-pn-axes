#!/bin/zsh
# Rerun of every analysis whose outputs the MLST paper uses, with the size cut of 2 arcsec (author request 2026-10-17)
cd ${0:a:h}/..
export CUT=2
PY=${PY:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
mkdir -p notes
for s in 15_catalogue 30_revision5_analyses 38_revision7_analyses 39a_conservative90_from27 39b_conservative90_from29 40_revision8_limits 41_inject3d_rev8 43_revision8_misc 46_revision9 47_revision10 49_cut_stats; do
  echo "=== $(date '+%H:%M:%S') $s"; $PY scripts/$s.py > notes/cut_$s.log 2>&1 || { echo "FAILED $s"; tail -5 notes/cut_$s.log; }
done
echo "=== $(date '+%H:%M:%S') done"
