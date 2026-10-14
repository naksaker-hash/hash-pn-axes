#!/bin/sh
# Full pipeline. Run from the repository root with the environment of requirements.txt.
# Steps 01 to 04 download about 1.1 GB of cutouts and train on a GPU (Apple MPS used for the paper).
set -e
PY=${PY:-python}
$PY scripts/00_fetch_binaries.py           # close binary central stars (D. Jones compilation)
$PY scripts/01_fetch.py                    # Halpha and r cutouts from CDS hips2fits
$PY scripts/01b_continuum_shs.py           # r band for SHS-only fields (DECaPS DR2, Pan-STARRS DR1)
$PY scripts/02_pseudolabels.py             # classical isophote masks, moments
$PY scripts/03_dataset.py                  # YOLO-seg dataset, longitude block split
for s in 0 1 2; do $PY scripts/04_train.py yolo26n-seg.pt $s --epochs 80; $PY scripts/05_measure.py yolo26n-seg_s$s; $PY scripts/06_eval.py yolo26n-seg_s$s; done
$PY scripts/08_pa_regressor.py cv          # axis network, fivefold CV (set PA_SEED, PA_AUG, PA_CH, PA_TAG for seeds and ablations)
$PY scripts/08_pa_regressor.py full        # axis network on all 350 labels
$PY scripts/09_morph_classifier.py         # classification network, fivefold CV
$PY scripts/10_independent_pa_test.py
$PY scripts/15_catalogue.py
$PY scripts/18_revision_analyses.py        # alignment samples, Tan et al. and binary tests, injections
$PY scripts/19_baselines.py
$PY scripts/17_seg_summary.py
$PY scripts/20_seg_morph_extra.py yolo26n-seg_s0 yolo26n-seg_s1 yolo26n-seg_s2
$PY scripts/21_class_comparison.py
$PY scripts/22_revision2_analyses.py
$PY scripts/13_text_numbers.py
$PY scripts/14_tables.py
$PY scripts/fig_paper.py
