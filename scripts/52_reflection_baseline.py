"""Training free reflection symmetry baseline (external referee, cf. Shishkin & Michaelis 2026).

On the same smoothed continuum subtracted image and search radius as the flux weighted moments of
19_baselines.py, the image is mirrored across lines through the flux centroid at trial angles in steps of
1 deg, and the angle with the smallest normalised mirror asymmetry sum|I - I_refl| / sum I is adopted as the
axis. Because a bipolar nebula is mirror symmetric about both its polar axis and its waist, the second
deepest minimum near 90 deg from the first is recorded too, and a training free choice between the two lines keeps
the one nearer the flux weighted (or low isophote) moment axis. Scored against the visual axes for the cross
validation PNe (HASH EPA) and the unseen PNe of Rees & Zijlstra (2013) and Corradi et al. (1998), with the
block bootstrap of 38_revision7_analyses.py for the gain of the seed averaged network.
Output: data/reflection_baseline.csv, data/reflection_baseline.json
"""
import json, pathlib, importlib, sys, re
import numpy as np, pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clipped_stats
from scipy import ndimage as ndi

ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
sys.path.insert(0, str(ROOT / 'scripts'))
pl = importlib.import_module('02_pseudolabels')
wrap = lambda a: (np.asarray(a, float) + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(np.asarray(a, float) - np.asarray(b, float)))
t = pd.read_csv(D / 'pseudolabels.csv')
m = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['idPNMain'] = m.idPNMain.astype(int); m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['png'] = 'PN G' + m.PNG
m['nkey'] = m.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
jr = m[['png', 'idPNMain', 'EPA']].merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png')
j1 = m[['nkey', 'png', 'idPNMain', 'EPA']].merge(co[['nkey', 'EPA_deg']], on='nkey')
j2 = m[['png', 'idPNMain', 'EPA']].merge(co[['PNG_approx_from_lb', 'EPA_deg']].rename(columns={'PNG_approx_from_lb': 'png'}), on='png')
jc = pd.concat([j1, j2[~j2.idPNMain.isin(j1.idPNMain)]]).drop_duplicates('idPNMain')
need = set(cv.idPNMain) | set(jr[jr.EPA.isna()].idPNMain) | set(jc[jc.EPA.isna()].idPNMain)

TH = np.arange(0, 180, 1.0)
rows = []
for _, r in t[t.idPNMain.isin(need)].iterrows():
    h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
    ha = h[0].data.astype(float); rr = h['R'].data.astype(float) if 'R' in h else None; w = WCS(h[0].header)
    d, _ = pl.continuum_subtract(ha, rr)
    n = d.shape[0]; c = (n - 1) / 2
    yy, xx = np.indices(d.shape); rad = np.hypot(yy - c, xx - c)
    diam = r.MajDiam if np.isfinite(r.MajDiam) else 10.0
    rs = max(0.65 * diam / r.pixscale, 4)
    bg = rad > 0.8 * diam / r.pixscale
    _, bm, _ = sigma_clipped_stats(d[bg & np.isfinite(d)], sigma=3)
    sm = ndi.gaussian_filter(np.nan_to_num(d - bm), 1.5)
    _, _, ss = sigma_clipped_stats(sm[bg], sigma=3)
    img = np.clip(sm, 0, None) * (sm > ss)
    wt = img * (rad < rs); W = wt.sum()
    o = dict(idPNMain=r.idPNMain)
    if W > 0:
        x0, y0 = (xx * wt).sum() / W, (yy * wt).sum() / W
        ap = np.hypot(yy - y0, xx - x0) < rs
        I = img * ap; S = I.sum()
        dx, dy = xx - x0, yy - y0
        A = []
        for th in np.radians(TH):
            c2, s2 = np.cos(2 * th), np.sin(2 * th)
            sx = x0 + c2 * dx + s2 * dy; sy = y0 + s2 * dx - c2 * dy  # mirror image of each pixel across the line
            Ir = ndi.map_coordinates(img, [sy, sx], order=1, mode='constant') * ap
            A.append(np.abs(I - Ir).sum() / S)
        A = np.array(A); k = int(np.argmin(A))
        perp = (k + 90) % 180; win = [(perp + j) % 180 for j in range(-20, 21)]; k2 = win[int(np.argmin(A[win]))]
        o['pa_refl'] = pl.sky_pa(w, x0, y0, TH[k])[0]; o['pa_refl2'] = pl.sky_pa(w, x0, y0, TH[k2])[0]
        o['asym_min'] = float(A[k]); o['asym_second'] = float(A[k2])
    rows.append(o)
b = pd.DataFrame(rows); b.to_csv(D / 'reflection_baseline.csv', index=False)

bp = pd.read_csv(D / 'baselines_pa.csv')[['idPNMain', 'mainClass', 'MajDiam', 'pa_eq', 'pa_low', 'pa_flux']]
ids = cv.idPNMain.values
seeds = [cv.set_index('idPNMain').reindex(ids)] + [pd.read_csv(D / f'pa_cnn_cv_s{s}.csv').set_index('idPNMain').reindex(ids) for s in (1, 2, 3, 4)]
c5 = cv[['idPNMain', 'EPA', 'Glon']].copy(); c5['d_net5'] = np.mean([dpa(s.cnn_pa.values, cv.EPA.values) for s in seeds], 0)
c5 = c5.merge(bp, on='idPNMain', how='left').merge(b, on='idPNMain', how='left')
c5['d_refl'] = dpa(c5.pa_refl, c5.EPA); c5['d_refl_best2'] = np.fmin(c5.d_refl, dpa(c5.pa_refl2, c5.EPA))
def guided(df, ref):
    pick = np.where(dpa(df.pa_refl, df[ref]) <= dpa(df.pa_refl2, df[ref]), df.pa_refl, df.pa_refl2)
    return np.where(df[ref].notna(), pick, df.pa_refl)
c5['d_refl_flux'] = dpa(guided(c5, 'pa_flux'), c5.EPA); c5['d_refl_low'] = dpa(guided(c5, 'pa_low'), c5.EPA)
blk = (c5.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0
rng = np.random.default_rng(71); out = {}
for cl in ['all', 'B', 'E']:
    s = c5 if cl == 'all' else c5[c5.mainClass == cl]; s = s[s.d_refl.notna()]
    ii = s.index.values; groups = [ii[blk[ii] == q] for q in np.unique(blk[ii])]
    g = [np.median(c5.d_refl.values[p]) - np.median(c5.d_net5.values[p]) for p in
         (np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(4000))]
    for col in ['d_refl_flux', 'd_refl_low']:
        gg = [np.median(c5[col].values[p]) - np.median(c5.d_net5.values[p]) for p in
              (np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(4000))]
        out[f'cv {cl} {col}'] = dict(median=float(s[col].median()), gt60=float((s[col] > 60).mean()),
                                     diff_vs_net=float(s[col].median() - s.d_net5.median()), ci95=[float(v) for v in np.percentile(gg, [2.5, 97.5])])
    out[f'cv {cl}'] = dict(n=int(len(s)), median=float(s.d_refl.median()), gt60=float((s.d_refl > 60).mean()),
                          net5_gt60=float((s.d_net5 > 60).mean()),
                          near90=float((s.d_refl > 70).mean()), median_best_of_two=float(s.d_refl_best2.median()),
                          net5_median=float(s.d_net5.median()), gain=float(s.d_refl.median() - s.d_net5.median()),
                          ci95=[float(v) for v in np.percentile(g, [2.5, 97.5])])
fullp = pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']]
for name, j in [('RZ13', jr), ('Corradi98', jc)]:
    j = j[j.EPA.isna()].merge(bp[['idPNMain', 'mainClass', 'MajDiam']], on='idPNMain').merge(b, on='idPNMain').merge(fullp, on='idPNMain')
    j = j[(j.MajDiam >= 4) & j.pa_refl.notna()].merge(bp[['idPNMain', 'pa_flux']], on='idPNMain', how='left')
    dd = dpa(j.pa_refl, j.EPA_deg); dg = dpa(guided(j, 'pa_flux'), j.EPA_deg); dn = dpa(j.cnn_pa, j.EPA_deg)
    out[name] = dict(n=int(len(j)), median=float(np.median(dd)), gt60=float((dd > 60).mean()), network_median=float(np.median(dn)),
                     network_gt60=float((dn > 60).mean()), guided_flux_median=float(np.median(dg)), guided_flux_gt60=float((dg > 60).mean()))
json.dump(out, open(D / 'reflection_baseline.json', 'w'), indent=1)
for k, v in out.items(): print(k, v)
