#!/bin/zsh
# round 1: model size and seed variance
cd ~/Downloads/HASH_DL_morph
PY=~/Downloads/M31_PN_YOLO/.venv/bin/python
for a in "yolo26s-seg.pt 0" "yolo26n-seg.pt 1" "yolo26n-seg.pt 2"; do
  set -- ${=a}
  n=${1%.pt}_s$2
  $PY scripts/04_train.py $1 $2 --epochs 80 > notes/train_$n.log 2>&1
  $PY scripts/05_measure.py $n > /dev/null 2>&1
  $PY scripts/06_eval.py $n > notes/eval_$n.txt 2>&1
  echo "$n done" >> notes/round1.log
done
