"""Galactic PA alignment of B and E PNe with CNN symmetry-axis PAs, by Galactic component.

GPA = 90 deg is parallel to the plane. Statistics on doubled angles: Rayleigh test, mean axis,
fraction within 30 deg of GPA 90 (random 1/3), and a V-test toward GPA 90 (mean resultant
projected on the plane direction; positive = parallel, negative = perpendicular).
Sample: resolved (MajDiam >= 6 arcsec), CNN TTA consistency R >= 0.8; ellipticals also need q < 0.85.
Also scores the RZ13 hand GPAs with the same statistics as a reference.
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
for c in ['DRAJ2000', 'DDECJ2000', 'MajDiam']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['idPNMain'] = m.idPNMain.astype(int)
p = pd.read_csv(ROOT / 'data' / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa', 'cnn_R']]
q = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')[['idPNMain', 'q_mask', 'survey']]
x = m[['idPNMain', 'Name', 'mainClass', 'Component', 'Glon', 'Glat', 'MajDiam', 'DRAJ2000', 'DDECJ2000', 'PNG']] \
    .merge(p, on='idPNMain').merge(q, on='idPNMain')
c = SkyCoord(x.DRAJ2000.values * u.deg, x.DDECJ2000.values * u.deg)
rot = c.galactic.position_angle(c.directional_offset_by(0 * u.deg, 1 * u.arcsec).galactic).deg
x['gpa'] = (x.cnn_pa + rot) % 180


def circ(pa):
    pa = np.asarray(pd.Series(pa).dropna(), float)
    n = len(pa)
    if n < 5:
        return dict(n=n)
    th = np.radians(2 * pa)
    C, S = np.cos(th).sum(), np.sin(th).sum()
    R = np.hypot(C, S) / n
    mean = (np.degrees(np.arctan2(S, C)) / 2) % 180
    # V test toward GPA 90 (doubled angle 180): V = sum cos(2 pa - 180)
    V = np.sum(np.cos(th - np.pi)) / n
    u_ = V * np.sqrt(2 * n)
    return dict(n=n, mean=float(mean), R=float(R), rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                frac_plane30=float(np.mean(dpa(pa, 90) < 30)), V=float(V), V_p_parallel=float(stats.norm.sf(u_)),
                V_p_perp=float(stats.norm.cdf(u_)))


sel = (x.MajDiam >= 6) & (x.cnn_R >= 0.8) & (((x.mainClass == 'E') & (x.q_mask < 0.85)) | (x.mainClass == 'B'))
res = {}
for comp in ['Bulge', 'Thin Disk', 'Thick Disk']:
    for cl in ['B', 'E']:
        s = x[sel & (x.Component == comp) & (x.mainClass == cl)]
        res[f'{comp}|{cl}'] = circ(s.gpa)
        print(f'{comp:10s} {cl}', {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"{comp}|{cl}"].items()})
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
for cl in ['bipolar', 'polar', 'non-polar']:
    res[f'RZ13 hand|{cl}'] = circ(rz[rz.morph_class == cl].GPA_deg)
    print(f'RZ13 hand {cl:9s}', {k: round(v, 3) if isinstance(v, float) else v for k, v in res[f"RZ13 hand|{cl}"].items()})
# same statistics for the CNN on the RZ13 bipolars that are in HASH
rzb = x.assign(png='PN G' + x.PNG).merge(rz[rz.morph_class == 'bipolar'][['PNG', 'GPA_deg']].rename(columns={'PNG': 'png'}), on='png')
res['RZ13 bipolars|CNN'] = circ(rzb.gpa)
print('RZ13 bipolars CNN  ', {k: round(v, 3) if isinstance(v, float) else v for k, v in res['RZ13 bipolars|CNN'].items()})
# sky-defined bulge region (|l|, |b| < 10 deg, as RZ13) and the disc outside it
for c_ in ['Glon', 'Glat']:
    x[c_] = pd.to_numeric(x[c_], errors='coerce')
x = x.assign(png='PN G' + x.PNG).merge(rz[['PNG', 'morph_class']].rename(columns={'PNG': 'png'}), on='png', how='left')
l = ((x.Glon + 180) % 360) - 180
bs = (np.abs(l) < 10) & (np.abs(x.Glat) < 10)
base = (x.MajDiam >= 6) & (x.cnn_R >= 0.8)
Bm = base & (x.mainClass == 'B')
Em = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
for name, mk in [('bulge sky|B all', Bm & bs), ('bulge sky|B in RZ13 bipolar', Bm & bs & (x.morph_class == 'bipolar')),
                 ('bulge sky|B in RZ13 polar or non-polar', Bm & bs & x.morph_class.isin(['polar', 'non-polar'])),
                 ('bulge sky|B new', Bm & bs & x.morph_class.isna()), ('bulge sky|E', Em & bs),
                 ('disc|B', Bm & ~bs), ('disc|E', Em & ~bs)]:
    res[name] = circ(x[mk].gpa)
    print(f'{name:40s}', {k: round(v, 3) if isinstance(v, float) else v for k, v in res[name].items()})
cat = (x.MajDiam >= 6) & (x.cnn_R >= 0.8)
res['catalogue'] = dict(n=int(cat.sum()), by_class=x[cat].mainClass.value_counts().to_dict())
json.dump(res, open(ROOT / 'data' / 'alignment_cnn.json', 'w'), indent=1)
x[Bm | Em].assign(bulge_sky=bs[Bm | Em]).to_csv(ROOT / 'data' / 'alignment_cnn_sample.csv', index=False)
