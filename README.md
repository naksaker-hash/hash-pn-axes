# Symmetry axes of Galactic planetary nebulae measured with deep learning

Code, catalogue and trained networks for the paper *Deep learning of the symmetry axes and morphologies of planetary nebulae in the HASH database* (N. Aksaker, Machine Learning: Science and Technology, submitted). This version reproduces the submitted manuscript (selection down to a HASH major diameter of 2 arcsec, 292 bipolars outside the bulge region, injection limits on a grid of 0.01 with 10,000 samples per step, three dimensional injections in `scripts/41_inject3d_rev8.py`).

A ResNet18 axis network is trained on visual position angles to measure the symmetry axes of HASH True planetary nebulae (PNe) in IPHAS, VPHAS+ and SHS H-alpha images, using the exact transformation of an axis under rotation and reflection as augmentation and eight-rotation test-time averaging. A YOLO26 nano segmentation network and a ResNet18 classification network provide masks and morphological classes. The axes are used to test the alignment of bipolar and elliptical PNe with the Galactic plane.

## Contents

| Path | Content |
|---|---|
| `data/catalogue_pa.csv` | Position angle catalogue for 2368 PNe (columns described below) |
| `data/pa_cnn_cv.csv` | Out-of-fold axis predictions, folds and visual labels for the 350 labelled PNe |
| `data/pa_final.csv` | Adopted axes (out-of-fold for labelled PNe) with sample flags used in the alignment tests |
| `data/morph_cv.csv` | Out-of-fold class probabilities of the classification network |
| `data/tan_comparison.csv` | Network and visual axes of the Tan et al. (2023) binary and high-adf PNe |
| `data/supp_binaries.csv` | PNe with known close binary nuclei matched to HASH (compilation of D. Jones, accessed 2026-10-13) |
| `data/*.json` | All numbers quoted in the paper; the authoritative outputs are those written by `scripts/run_cut.sh` (CUT=2) |
| `data/hash_master.csv` | Input sample: HASH True PNe with diameters, classes and the Galactic components and distances of Erzincan et al. (2025) |
| `data/hash_coverage.csv`, `data/mocs/` | Survey coverage (CDS MOCs) |
| `refs/pa_tables/` | Visual position angles from Weidmann & Diaz (2008), Rees & Zijlstra (2013), Corradi et al. (1998) and Tan et al. (2023), with a README on frames and axis definitions |
| `scripts/` | Pipeline (see `run_all.sh` for the order) |

Trained weights are attached to the GitHub release: `pa_cnn_full.pt` (axis network trained on all 350 labels) and `yolo26n-seg_s{0,1,2}.pt` (segmentation networks, three seeds).

## Catalogue columns (`data/catalogue_pa.csv`)

HASH identifier, PN G designation, name, coordinates, HASH class and diameters (arcsec), survey of the cutout, axes and axial ratio of the classical mask, `mask_trusted` (mask passed the quality criteria), `PA_moment` (mask moment, deg east of north), `PA_cnn` (axis network, deg east of north), `GPA_cnn` (Galactic position angle, deg from Galactic north through east), `R_TTA` (consistency over eight rotations), `HASH_PA` (visual position angle in HASH, where available), and the flags `flag_reliable` (B, E, I or A class, HASH major diameter >= 2 arcsec, R_TTA >= 0.8), `flag_below_labels` (HASH major diameter below 4 arcsec, smaller than any training label), `flag_round` (R or S class), `flag_SHS` (SHS image, axes carry an instrumental orientation preference) and `flag_out_of_fold` (labelled PN, out-of-fold axis given).

## Reproducing

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
mkdir -p weights   # place yolo26n-seg.pt (Ultralytics, COCO pretrained) here
PY=python ./run_all.sh
```

`run_all.sh` runs the full pipeline, including the later revision analyses (scripts 25 to 50) and finally `scripts/run_cut.sh`, which regenerates every number that depends on the size cut with `CUT=2`. Scripts that write tables or figures (`14_tables.py`, `fig_*.py`, `44_tab_3d.py`, `49_cut_stats.py`) write into a `paper_mlst/` folder, which is not part of this repository. The ten labels of exactly 0 deg (probable placeholders) are kept in training; a network trained without them is `PA_TAG=_nozero` (see `28_revision4_training.py`).

Image cutouts are downloaded from the CDS hips2fits service (about 1.1 GB) and are not included. Training used an Apple M1 GPU (MPS), which is not bit-for-bit deterministic, so medians can differ by a few tenths of a degree between runs.

## Data sources

HASH PN database (Parker et al. 2016; hashpn.space); IPHAS DR2, VPHAS+ DR4, SHS, DECaPS DR2 and Pan-STARRS DR1 through CDS hips2fits; visual position angles from the papers listed above; close binary central stars from the compilation of D. Jones (http://www.drdjones.net/bcspn/; Jones & Boffin 2017).

## Licence

Code under the MIT licence. The catalogue may be used with citation of the paper. Literature tables remain the work of their original authors and should be cited accordingly.
