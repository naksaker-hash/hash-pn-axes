"""Run a trained YOLO-seg model on all cutouts and measure shape and PA from the predicted mask.

Usage: python 05_measure.py <run name in runs/> [--conf 0.1]
The prediction nearest the image centre is kept. The PNG is north-up (flipped FITS),
so the mask is flipped back before measuring on the FITS WCS.
Output: data/dl_<run>.csv
"""
import argparse, pathlib, importlib, sys
import numpy as np, pandas as pd, cv2
from astropy.io import fits
from astropy.wcs import WCS
from ultralytics import YOLO

sys.path.insert(0, str(pathlib.Path(__file__).parent))
pl = importlib.import_module('02_pseudolabels')
ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASSES = ['R', 'E', 'B', 'S']

ap = argparse.ArgumentParser()
ap.add_argument('run'); ap.add_argument('--conf', type=float, default=0.1)
a = ap.parse_args()
model = YOLO(str(ROOT / 'runs' / a.run / 'weights' / 'best.pt'))
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
DS = ROOT / 'data' / 'yolo_seg' / 'images'
rows = []
for _, r in t.iterrows():
    stem = r.file.replace('.fits', '')
    png = DS / r.split / f'{stem}.png'
    if not png.exists():  # dropped from train/val because it had no trusted mask
        img = importlib.import_module('03_dataset').image(ROOT / 'data' / 'cutouts' / r.file, r.kcont)
        png = ROOT / 'data' / 'tmp_pred.png'
        cv2.imwrite(str(png), img[:, :, ::-1])
    res = model.predict(str(png), imgsz=256, conf=a.conf, device='mps', verbose=False, retina_masks=True)[0]
    out = dict(idPNMain=r.idPNMain, file=r.file, split=r.split, good=r.good)
    if res.masks is not None and len(res.boxes):
        n = res.orig_shape[0]
        ctr = res.boxes.xywh[:, :2].cpu().numpy()
        k = int(np.argmin(np.hypot(*(ctr - (n - 1) / 2).T)))
        m = res.masks.data[k].cpu().numpy() > 0.5
        m = np.flipud(m)
        h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
        w = WCS(h[0].header)
        mm = pl.moments(m, r.pixscale)
        probs = res.boxes.cls.cpu().numpy()
        out.update(dl_conf=float(res.boxes.conf[k]), dl_class=CLASSES[int(probs[k])],
                   dl_npix=int(m.sum()), dl_maj=mm['maj'], dl_min=mm['mnr'],
                   dl_ndet=len(res.boxes))
        if np.isfinite(mm['maj']):
            out['dl_q'] = mm['mnr'] / mm['maj']
            out['dl_pa_eq'], out['dl_pa_gal'] = pl.sky_pa(w, mm['x0'], mm['y0'], mm['pa_pix'])
        np.savez_compressed(ROOT / 'data' / 'masks' / f'{stem}_dl.npz', mask=m.astype(np.uint8))
        np.savez_compressed(ROOT / 'data' / 'masks' / f'{stem}_dl_{a.run}.npz', mask=m.astype(np.uint8))
    rows.append(out)
pd.DataFrame(rows).to_csv(ROOT / 'data' / f'dl_{a.run}.csv', index=False)
print(len(rows), 'done')
