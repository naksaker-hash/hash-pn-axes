"""Train a YOLO-seg model on data/yolo_seg.

Usage: python 04_train.py <weights e.g. yolo26n-seg.pt> <seed> [--epochs N] [--imgsz N] [--tag T]
Channels are physical (Halpha, Halpha minus r, r), so HSV jitter is off.
Morphology is invariant to rotation and flips, so full rotations and both flips are on;
the PA is measured afterwards on unaugmented images.
"""
import argparse, pathlib
from ultralytics import YOLO

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument('weights'); ap.add_argument('seed', type=int)
ap.add_argument('--epochs', type=int, default=80)
ap.add_argument('--imgsz', type=int, default=256)
ap.add_argument('--batch', type=int, default=16)
ap.add_argument('--tag', default='')
a = ap.parse_args()
name = f"{pathlib.Path(a.weights).stem}{a.tag}_s{a.seed}"
YOLO(str(ROOT / 'weights' / a.weights)).train(
    data=str(ROOT / 'data' / 'yolo_seg' / 'data.yaml'), project=str(ROOT / 'runs'), name=name,
    epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=False, device='mps',
    workers=4, cache='ram', plots=True, exist_ok=True,
    hsv_h=0, hsv_s=0, hsv_v=0, degrees=180, translate=0.1, scale=0.3, shear=0, perspective=0,
    fliplr=0.5, flipud=0.5, mosaic=0, mixup=0, cutmix=0, copy_paste=0, close_mosaic=0,
    max_det=10, patience=25, amp=False, overlap_mask=False,
)
