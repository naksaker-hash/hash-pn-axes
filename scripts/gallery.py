"""Gallery of cutouts with pseudo-label mask contours and the HASH diameter circle."""
import sys, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
start, n = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (0, 24)
out = sys.argv[3] if len(sys.argv) > 3 else str(ROOT / 'notes' / 'gallery.png')
sub = t.iloc[start:start + n]
cols = 6
f, ax = plt.subplots(int(np.ceil(len(sub) / cols)), cols, figsize=(3 * cols, 3.2 * np.ceil(len(sub) / cols)))
for a, (_, r) in zip(ax.flat, sub.iterrows()):
    h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
    d = h[0].data
    lo, hi = np.nanpercentile(d, [5, 99.8])
    a.imshow(np.arcsinh(np.clip((d - lo) / (hi - lo), 0, 1) * 10), origin='lower', cmap='gray')
    m = np.load(ROOT / 'data' / 'masks' / r.file.replace('.fits', '.npz'))['mask']
    a.contour(m, [0.5], colors='lime', linewidths=0.8)
    c = (d.shape[0] - 1) / 2
    if np.isfinite(r.MajDiam):
        a.add_patch(plt.Circle((c, c), r.MajDiam / 2 / r.pixscale, fill=False, color='r', lw=0.7, ls='--'))
    a.set_title(f"{r.Name} {r.mainClass} {r.survey}\nD {r.MajDiam:.0f} mask {r.maj_mask:.0f} q {r.q_mask:.2f}", fontsize=7)
    a.axis('off')
for a in ax.flat[len(sub):]:
    a.axis('off')
plt.tight_layout()
plt.savefig(out, dpi=70)
