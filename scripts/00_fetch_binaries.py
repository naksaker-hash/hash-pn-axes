"""Download and parse the compilation of binary central stars of D. Jones (http://www.drdjones.net/bcspn/)
and match it to the HASH sample. Output: refs/binaries/bcspn_jones.csv"""
import re, html, pathlib
import numpy as np, pandas as pd, requests
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'refs' / 'binaries'; OUT.mkdir(parents=True, exist_ok=True)
s = requests.get('http://www.drdjones.net/bcspn/', timeout=60, headers={'User-Agent': 'Mozilla/5.0'}).text
rows = []
for r in re.findall(r'<tr[^>]*>(.*?)</tr>', s, re.S):
    c = [html.unescape(re.sub('<[^>]+>', '', x)).strip() for x in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)]
    if len(c) >= 4 and re.match(r'\d+:\d+:\d', c[1] or ''):
        rows.append(dict(name=c[0].lstrip('*'), candidate=c[0].startswith('*'), ra=c[1], dec=c[2], period_d=c[3],
                         phot=c[4] if len(c) > 4 else ''))
b = pd.DataFrame(rows).drop_duplicates(['ra', 'dec'])
bc = SkyCoord(b.ra, b.dec, unit=(u.hourangle, u.deg))
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
hc = SkyCoord(pd.to_numeric(m.DRAJ2000).values * u.deg, pd.to_numeric(m.DDECJ2000).values * u.deg)
i, d, _ = bc.match_to_catalog_sky(hc)
b['idPNMain'] = np.where(d.arcsec < 15, m.idPNMain.values[i], None); b['sep'] = d.arcsec
b.to_csv(OUT / 'bcspn_jones.csv', index=False)
print(len(b), 'binaries,', b.idPNMain.notna().sum(), 'matched to HASH')
