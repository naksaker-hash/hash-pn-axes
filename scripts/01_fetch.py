"""Fetch Halpha and r cutouts for HASH True PNe from CDS hips2fits (resumable)."""
import io, sys, time, pathlib
import numpy as np, pandas as pd, requests
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'data' / 'cutouts'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'https://alasky.cds.unistra.fr/hips-image-services/hips2fits'
SURVEYS = [  # priority order: (flag column, tag, Halpha HiPS, r HiPS or None)
    ('IPHAS_DR2_halpha', 'IPHAS', 'CDS/P/IPHAS/DR2/halpha', 'CDS/P/IPHAS/DR2/r'),
    ('VPHAS_DR4_Halpha', 'VPHAS', 'CDS/P/VPHAS/DR4/Halpha', 'CDS/P/VPHAS/DR4/r'),
    ('SHS', 'SHS', 'CDS/P/SHS', None),
]
NPX = 256


def get(hips, ra, dec, fov):
    p = dict(hips=hips, ra=ra, dec=dec, fov=fov, width=NPX, height=NPX,
             projection='TAN', format='fits')
    for k in range(4):
        try:
            q = requests.get(URL, params=p, timeout=180)
            q.raise_for_status()
            return fits.open(io.BytesIO(q.content))
        except Exception as e:
            err = e
            time.sleep(5 * (k + 1))
    raise err


def job(r):
    for flag, tag, ha, rb in SURVEYS:
        if not r[flag]:
            continue
        fn = OUT / f'{int(r.idPNMain)}_{tag}.fits'
        if fn.exists():
            return f'{r.idPNMain} skip'
        diam = r.MajDiam if np.isfinite(r.MajDiam) else 10.0
        fov = max(3 * diam, 30.0) / 3600
        try:
            hh = get(ha, r.DRAJ2000, r.DDECJ2000, fov)
            hdus = [fits.PrimaryHDU(hh[0].data.astype(np.float32), hh[0].header)]
            hdus[0].header['BAND'] = 'HALPHA'
            if rb:
                hr = get(rb, r.DRAJ2000, r.DDECJ2000, fov)
                h2 = fits.ImageHDU(hr[0].data.astype(np.float32), hr[0].header, name='R')
                hdus.append(h2)
            hdus[0].header['HASHID'] = int(r.idPNMain)
            hdus[0].header['SURVEY'] = tag
            hdus[0].header['FOVDEG'] = fov
            tmp = fn.with_suffix('.tmp')
            fits.HDUList(hdus).writeto(tmp, overwrite=True)
            tmp.rename(fn)
            return f'{r.idPNMain} {tag} ok'
        except Exception as e:
            return f'{r.idPNMain} {tag} FAIL {e}'
    return f'{r.idPNMain} none'


if __name__ == '__main__':
    m = pd.read_csv(ROOT / 'data' / 'hash_coverage.csv')
    m = m[m.anyHa]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else len(m)
    with ThreadPoolExecutor(4) as ex:
        for i, s in enumerate(ex.map(job, [r for _, r in m.head(n).iterrows()])):
            print(i, s, flush=True)
