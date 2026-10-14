"""Classical pseudo-label masks for HASH PN cutouts.

Halpha minus scaled r (continuum subtraction), background from sigma clipping,
smoothed isophote at NSIG sigma, keep the connected component(s) touching the
central search circle (HASH MajDiam), fill holes. Shape from second moments.
Output: data/masks/<id>_<survey>.npz and data/pseudolabels.csv
"""
import sys, glob, pathlib
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[1]
MASKS = ROOT / 'data' / 'masks'
MASKS.mkdir(exist_ok=True)
NSIG = 3.0
PEAKFRAC = 0.10  # Tylenda et al. 2003 convention for PN diameters


def continuum_subtract(ha, r):
    if r is None:
        return ha.copy(), np.nan
    # scale from point sources: bright pixels in r that are not in the centre
    good = np.isfinite(ha) & np.isfinite(r)
    _, rmed, rstd = sigma_clipped_stats(r[good])
    _, hmed, _ = sigma_clipped_stats(ha[good])
    sel = good & (r > rmed + 20 * rstd)
    yy, xx = np.indices(ha.shape)
    c = (ha.shape[0] - 1) / 2
    sel &= np.hypot(yy - c, xx - c) > ha.shape[0] / 6
    if sel.sum() < 30:
        k = np.nan
    else:
        k = np.median((ha[sel] - hmed) / (r[sel] - rmed))
    if not np.isfinite(k) or k <= 0:
        return ha - hmed, np.nan
    return (ha - hmed) - k * (r - rmed), k


def moments(mask, pixscale):
    yy, xx = np.nonzero(mask)
    if len(xx) < 5:
        return dict(maj=np.nan, mnr=np.nan, pa_pix=np.nan)
    x0, y0 = xx.mean(), yy.mean()
    cov = np.cov(np.vstack([xx - x0, yy - y0]))
    ev, evec = np.linalg.eigh(cov)
    a, b = 4 * np.sqrt(ev[1]), 4 * np.sqrt(max(ev[0], 0))  # full axes of a uniform ellipse
    vx, vy = evec[:, 1]
    return dict(x0=x0, y0=y0, maj=a * pixscale, mnr=b * pixscale,
                pa_pix=np.degrees(np.arctan2(vy, vx)))


def sky_pa(w, x0, y0, pa_pix, L=10):
    """Equatorial PA (east of north) of the pixel-frame axis, and Galactic PA."""
    t = np.radians(pa_pix)
    p1 = w.pixel_to_world(x0 - L * np.cos(t), y0 - L * np.sin(t))
    p2 = w.pixel_to_world(x0 + L * np.cos(t), y0 + L * np.sin(t))
    pa_eq = p1.position_angle(p2).deg % 180
    pa_gal = p1.galactic.position_angle(p2.galactic).deg % 180
    return pa_eq, pa_gal


def process(fn, row):
    h = fits.open(fn)
    ha = h[0].data.astype(float)
    r = h['R'].data.astype(float) if 'R' in h else None
    w = WCS(h[0].header)
    pixscale = abs(h[0].header['CDELT1']) * 3600
    d, k = continuum_subtract(ha, r)
    n = d.shape[0]
    yy, xx = np.indices(d.shape)
    c = (n - 1) / 2
    rad = np.hypot(yy - c, xx - c)
    diam = row.MajDiam if np.isfinite(row.MajDiam) else 10.0
    rsearch = max(0.65 * diam / pixscale, 4)
    bgreg = rad > 0.8 * diam / pixscale
    _, bmed, bstd = sigma_clipped_stats(d[bgreg & np.isfinite(d)], sigma=3)
    sm = ndi.gaussian_filter(np.nan_to_num(d - bmed), 1.5)
    # noise of the smoothed image, measured the same way
    _, _, sstd = sigma_clipped_stats(sm[bgreg], sigma=3)
    peak = sm[rad < rsearch].max()
    thr = max(NSIG * sstd, PEAKFRAC * peak)
    seg = sm > thr
    lab, nl = ndi.label(seg)
    keep = np.unique(lab[(rad < rsearch) & seg])
    keep = keep[keep > 0]
    mask = np.isin(lab, keep)
    mask = ndi.binary_fill_holes(ndi.binary_opening(mask, iterations=1))
    np.savez_compressed(MASKS / (fn.stem + '.npz'), mask=mask.astype(np.uint8))
    m = moments(mask, pixscale)
    out = dict(idPNMain=row.idPNMain, file=fn.name, survey=h[0].header['SURVEY'],
               pixscale=pixscale, kcont=k, npix=int(mask.sum()),
               peak_snr=float(peak / sstd), thr_peak=bool(PEAKFRAC * peak > NSIG * sstd),
               frac_edge=float(mask[[0, -1], :].mean() + mask[:, [0, -1]].mean()),
               frac_in_search=float((mask & (rad < rsearch)).sum() / max(mask.sum(), 1)),
               maj_mask=m['maj'], min_mask=m['mnr'])
    if np.isfinite(m['maj']):
        out['q_mask'] = m['mnr'] / m['maj']
        out['pa_eq'], out['pa_gal'] = sky_pa(w, m['x0'], m['y0'], m['pa_pix'])
    return out


if __name__ == '__main__':
    cov = pd.read_csv(ROOT / 'data' / 'hash_coverage.csv').set_index('idPNMain')
    files = sorted(pathlib.Path(ROOT / 'data' / 'cutouts').glob('*.fits'))
    if len(sys.argv) > 1:
        files = files[:int(sys.argv[1])]
    rows = []
    for fn in files:
        hid = int(fn.stem.split('_')[0])
        try:
            row = cov.loc[hid].copy()
            row['idPNMain'] = hid
            rows.append(process(fn, row))
        except Exception as e:
            print(fn.name, 'FAIL', e)
    t = pd.DataFrame(rows).merge(cov[['Name', 'mainClass', 'MajDiam', 'MinDiam', 'Glon', 'Glat']].reset_index(), on='idPNMain')
    t.to_csv(ROOT / 'data' / 'pseudolabels.csv', index=False)
    print(len(t), 'done')
