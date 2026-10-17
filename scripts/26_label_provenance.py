"""Image source of every visual label (referee round 4). The HASH position angles are matched by coordinates
(30 arcsec) to Weidmann & Diaz (2008), whose catalogue gives the image source (DSS2, MASH, CAM98 = adopted from
Corradi et al. 1998, HST). Labels without a match that equal exactly 0 deg are flagged as suspect placeholders.
Output: data/label_provenance.csv (all HASH PAs with a cutout of at least 4 arcsec)
"""
import pathlib
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['idPNMain'] = m.idPNMain.astype(int)
for c in ['DRAJ2000', 'DDECJ2000']:
    m[c] = m[c].astype(float)
t = t.merge(m[['idPNMain', 'EPA', 'DRAJ2000', 'DDECJ2000']], on='idPNMain')
lab = t[t.EPA.notna() & (t.MajDiam >= 4)].reset_index(drop=True)
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv').dropna(subset=['EPA_deg', 'RAJ2000_simbad'])
c1 = SkyCoord(lab.DRAJ2000.values * u.deg, lab.DDECJ2000.values * u.deg)
c2 = SkyCoord(w.RAJ2000_simbad.values * u.deg, w.DEJ2000_simbad.values * u.deg)
i, d, _ = c1.match_to_catalog_sky(c2)
ok = d.arcsec < 30
lab['label_source'] = np.where(ok, w.source.values[i], 'other')
lab['weidmann_epa'] = np.where(ok, w.EPA_deg.values[i], np.nan)
lab['suspect_zero'] = (~ok) & (lab.EPA == 0)
lab[['idPNMain', 'Name', 'survey', 'EPA', 'label_source', 'weidmann_epa', 'suspect_zero']].to_csv(
    ROOT / 'data' / 'label_provenance.csv', index=False)
print(lab.label_source.value_counts().to_dict(), 'suspect zero', int(lab.suspect_zero.sum()))
