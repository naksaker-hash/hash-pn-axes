"""Stronger moment baselines for the axis comparison (referee round 1).

For every cutout: (a) moments of the YOLO26n seed 0 mask, (b) moments of a low isophote mask
(3 sigma only, no peak fraction), (c) flux weighted second moments of the smoothed continuum subtracted
image above 1 sigma inside the search radius. Scored against the visual axes on the cross validation set
(HASH EPA) and on the never trained Rees & Zijlstra and Corradi objects.
Output: data/baselines.json, data/baselines_pa.csv
"""
import json, pathlib, importlib, sys, re
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pl = importlib.import_module('02_pseudolabels')
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
cov = pd.read_csv(ROOT / 'data' / 'hash_coverage.csv').set_index('idPNMain')
rows = []
for _, r in t.iterrows():
    h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
    ha = h[0].data.astype(float)
    rr = h['R'].data.astype(float) if 'R' in h else None
    w = WCS(h[0].header)
    d, _ = pl.continuum_subtract(ha, rr)
    n = d.shape[0]; c = (n - 1) / 2
    yy, xx = np.indices(d.shape); rad = np.hypot(yy - c, xx - c)
    diam = r.MajDiam if np.isfinite(r.MajDiam) else 10.0
    rs = max(0.65 * diam / r.pixscale, 4)
    bg = rad > 0.8 * diam / r.pixscale
    _, bm, _ = sigma_clipped_stats(d[bg & np.isfinite(d)], sigma=3)
    sm = ndi.gaussian_filter(np.nan_to_num(d - bm), 1.5)
    _, _, ss = sigma_clipped_stats(sm[bg], sigma=3)
    o = dict(idPNMain=r.idPNMain)
    # (b) low isophote
    seg = sm > 3 * ss
    lab, _ = ndi.label(seg); keep = np.unique(lab[(rad < rs) & seg]); keep = keep[keep > 0]
    mk = ndi.binary_fill_holes(np.isin(lab, keep))
    mm = pl.moments(mk, r.pixscale)
    if np.isfinite(mm['maj']):
        o['pa_low'] = pl.sky_pa(w, mm['x0'], mm['y0'], mm['pa_pix'])[0]
    # (c) flux weighted moments inside the search radius
    wt = np.clip(sm, 0, None) * ((rad < rs) & (sm > ss))
    W = wt.sum()
    if W > 0:
        x0, y0 = (xx * wt).sum() / W, (yy * wt).sum() / W
        cxx = ((xx - x0) ** 2 * wt).sum() / W; cyy = ((yy - y0) ** 2 * wt).sum() / W; cxy = ((xx - x0) * (yy - y0) * wt).sum() / W
        ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
        o['pa_flux'] = pl.sky_pa(w, x0, y0, np.degrees(np.arctan2(evec[1, 1], evec[0, 1])))[0]
    rows.append(o)
b = pd.DataFrame(rows)
dl = pd.read_csv(ROOT / 'data' / 'dl_yolo26n-seg_s0.csv')[['idPNMain', 'dl_pa_eq']]
x = t[['idPNMain', 'Name', 'mainClass', 'MajDiam', 'good', 'pa_eq']].merge(b, on='idPNMain').merge(dl, on='idPNMain', how='left')
x.to_csv(ROOT / 'data' / 'baselines_pa.csv', index=False)

cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')[['idPNMain', 'EPA', 'cnn_pa']]
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['idPNMain'] = m.idPNMain.astype(int); m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['png'] = 'PN G' + m.PNG
m['nkey'] = m.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
fullp = pd.read_csv(ROOT / 'data' / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']]
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv')
co['nkey'] = co.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
sets = {'cv': x.merge(cv, on='idPNMain').rename(columns={'EPA': 'ref'})}
# same matching as 10_independent_pa_test.py: RZ13 by PN G name, Corradi by name with PN G (approximate) fallback
jr = m[['png', 'idPNMain', 'EPA']].merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png')
j1 = m[['nkey', 'png', 'idPNMain', 'EPA']].merge(co[['nkey', 'EPA_deg']], on='nkey')
j2 = m[['png', 'idPNMain', 'EPA']].merge(co[['PNG_approx_from_lb', 'EPA_deg']].rename(columns={'PNG_approx_from_lb': 'png'}), on='png')
jc = pd.concat([j1, j2[~j2.idPNMain.isin(j1.idPNMain)]]).drop_duplicates('idPNMain')
for name, j in [('RZ13', jr), ('Corradi98', jc)]:
    j = j[j.EPA.isna()].merge(x, on='idPNMain').merge(fullp, on='idPNMain').rename(columns={'EPA_deg': 'ref'})
    sets[name] = j[j.MajDiam >= 4]
res = {}
for sname, s in sets.items():
    res[sname] = {}
    for cl in ['all', 'B', 'E']:
        ss_ = s if cl == 'all' else s[s.mainClass == cl]
        r_ = {}
        for lab_, col in [('network', 'cnn_pa'), ('classical', 'pa_eq'), ('yolo_mask', 'dl_pa_eq'), ('low_isophote', 'pa_low'), ('flux_weighted', 'pa_flux')]:
            dd = dpa(ss_[col], ss_.ref).dropna()
            r_[lab_] = dict(n=len(dd), median=float(dd.median()) if len(dd) else None, gt60=float((dd > 60).mean()) if len(dd) else None)
        if sname == 'cv':
            g = ss_[ss_.good]
            dd = dpa(g.pa_eq, g.ref).dropna(); r_['classical_trusted_masks'] = dict(n=len(dd), median=float(dd.median()), gt60=float((dd > 60).mean()))
            dd = dpa(g.cnn_pa, g.ref).dropna(); r_['network_trusted_masks'] = dict(n=len(dd), median=float(dd.median()), gt60=float((dd > 60).mean()))
        res[sname][cl] = r_
json.dump(res, open(ROOT / 'data' / 'baselines.json', 'w'), indent=1)
for sname in res:
    for cl in res[sname]:
        print(sname, cl, {k: (v['n'], round(v['median'], 1) if v['median'] is not None else None, round(v['gt60'], 2) if v['gt60'] is not None else None) for k, v in res[sname][cl].items()})
