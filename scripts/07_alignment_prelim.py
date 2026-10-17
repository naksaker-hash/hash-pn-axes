"""Preliminary Galactic PA statistics for elongated B and E PNe, DL masks vs hand EPA."""
import pathlib, sys
import pandas as pd, numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else 'yolo26n-seg_s0'
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
d = pd.read_csv(ROOT / 'data' / f'dl_{run}.csv')
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
for c in ['EPA', 'DRAJ2000', 'DDECJ2000']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['idPNMain'] = m.idPNMain.astype(int)
x = t.merge(d.drop(columns=['split', 'good']), on=['idPNMain', 'file']).merge(
    m[['idPNMain', 'EPA', 'DRAJ2000', 'DDECJ2000', 'Component']], on='idPNMain')
c = SkyCoord(x.DRAJ2000.values * u.deg, x.DDECJ2000.values * u.deg)
rot = c.galactic.position_angle(c.directional_offset_by(0 * u.deg, 1 * u.arcsec).galactic).deg
x['EPA_gal'] = (x.EPA + rot) % 180
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
chk = x[x.EPA.notna() & x.dl_pa_eq.notna()]
assert np.allclose(dpa(chk.dl_pa_eq, chk.EPA), dpa(chk.dl_pa_gal, chk.EPA_gal), atol=0.5), 'frame conversion mismatch'


def stats(pa):
    pa = np.asarray(pd.Series(pa).dropna())
    n = len(pa)
    if n == 0:
        return 'n   0'
    th = np.radians(2 * pa)
    R = np.hypot(np.cos(th).sum(), np.sin(th).sum()) / n
    mean = (np.degrees(np.arctan2(np.sin(th).sum(), np.cos(th).sum())) / 2) % 180
    near = np.mean(dpa(pa, 90) < 30)
    return f'n {n:3d}  mean {mean:5.1f}  R {R:.2f}  Rayleigh p {np.exp(-n * R ** 2):.2g}  frac|GPA-90|<30 {near:.2f}'


sel = (x.dl_q < 0.8) & (x.dl_maj > 6)
print('random expectation for frac|GPA-90|<30 is 0.33; GPA 90 = parallel to the plane')
for comp in ['Bulge', 'Thin Disk', 'Thick Disk']:
    for cl in ['B', 'E']:
        s = x[(x.Component == comp) & (x.mainClass == cl)]
        print(f'{comp:10s} {cl} DL   ', stats(s[sel[s.index]].dl_pa_gal))
        print(f'{comp:10s} {cl} hand ', stats(s.EPA_gal))
x[['idPNMain', 'Name', 'mainClass', 'Component', 'Glon', 'Glat', 'dl_maj', 'dl_q', 'dl_pa_eq', 'dl_pa_gal', 'EPA', 'EPA_gal']].to_csv(
    ROOT / 'data' / f'alignment_{run}.csv', index=False)
