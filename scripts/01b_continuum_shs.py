"""Add an r-band continuum HDU to SHS-only cutouts (DECaPS DR2 r, else Pan-STARRS DR1 r)."""
import pathlib, sys
import numpy as np, pandas as pd
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from importlib import import_module
fetch = import_module('01_fetch')

ROOT = pathlib.Path(__file__).resolve().parents[1]


def job(r):
    fn = ROOT / 'data' / 'cutouts' / f'{int(r.idPNMain)}_SHS.fits'
    if not fn.exists():
        return f'{r.idPNMain} missing'
    h = fits.open(fn)
    if 'R' in h:
        return f'{r.idPNMain} has R'
    hips = 'CDS/P/DECaPS/DR2/r' if r.DECaPS_DR2_r else ('CDS/P/PanSTARRS/DR1/r' if r.DDECJ2000 > -30 else None)
    if hips is None:
        return f'{r.idPNMain} no r'
    try:
        hr = fetch.get(hips, r.DRAJ2000, r.DDECJ2000, h[0].header['FOVDEG'])
        d = hr[0].data.astype(np.float32)
        if not np.isfinite(d).any() or np.nanstd(d) == 0:
            return f'{r.idPNMain} empty r'
        hdus = [fits.PrimaryHDU(h[0].data, h[0].header), fits.ImageHDU(d, hr[0].header, name='R')]
        hdus[1].header['RSRC'] = hips
        tmp = fn.with_suffix('.tmp')
        fits.HDUList(hdus).writeto(tmp, overwrite=True)
        tmp.rename(fn)
        return f'{r.idPNMain} ok {hips}'
    except Exception as e:
        return f'{r.idPNMain} FAIL {e}'


if __name__ == '__main__':
    m = pd.read_csv(ROOT / 'data' / 'hash_coverage.csv')
    m = m[m.SHS & ~m.IPHAS_DR2_halpha & ~m.VPHAS_DR4_Halpha]
    with ThreadPoolExecutor(4) as ex:
        for s in ex.map(job, [r for _, r in m.iterrows()]):
            print(s, flush=True)
