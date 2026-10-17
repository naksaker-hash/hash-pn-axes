"""Revision 8: is the orientation preference of the SHS stellar PNe caused by the Halpha to r registration?
For the 41 SHS stellar PNe with a HASH diameter, the r band of each cutout is shifted onto Halpha by the mean
centroid offset of its field stars, and the final network (data/pa_cnn_full.pt) is applied at eight rotations to
the original and the re-registered inputs. A further variant matches the PSF of r to Halpha by a Gaussian blur of
the sharper band. No training. Output: data/rev8_shs_reregister.json"""
import sys, json, pathlib, importlib
import numpy as np, pandas as pd, cv2, torch
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import maximum_filter, shift as ndshift, gaussian_filter
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
sys.path.insert(0, str(ROOT / 'scripts'))
ds03 = importlib.import_module('03_dataset'); m8 = importlib.import_module('08_pa_regressor')
x = pd.read_csv(D / 'pa_final.csv'); pl = pd.read_csv(D / 'pseudolabels.csv').set_index('file')
st = x[(x.mainClass == 'S') & (x.survey == 'SHS') & x.MajDiam.notna()]


def stars(rr):
    med = np.median(rr); sd = 1.4826 * np.median(np.abs(rr - med)) or 1; nn = rr.shape[0]
    pk = (rr == maximum_filter(rr, 7)) & (rr > med + 20 * sd) & (rr < np.percentile(rr, 99.95))
    ys, xs = np.nonzero(pk); c0 = (nn - 1) / 2
    k = (np.hypot(xs - c0, ys - c0) > nn / 6) & (xs > 5) & (ys > 5) & (xs < nn - 6) & (ys < nn - 6)
    return xs[k], ys[k]


def mom(im, xs, ys):
    out = []
    gy, gx = np.mgrid[-3:4, -3:4]
    for xv, yv in zip(xs, ys):
        c = np.clip(im[yv - 3:yv + 4, xv - 3:xv + 4] - np.median(im), 0, None); w = c.sum()
        if w <= 0:
            out.append(None); continue
        mx, my = (gx * c).sum() / w, (gy * c).sum() / w
        out.append((mx, my, ((gx - mx) ** 2 * c).sum() / w + ((gy - my) ** 2 * c).sum() / w))
    return out


def build(ha, r, k):
    _, hm, _ = sigma_clipped_stats(ha); _, rm, _ = sigma_clipped_stats(r)
    sub = (ha - hm) - k * (r - rm)
    img = np.flipud(np.dstack([ds03.stretch(ha), ds03.stretch(sub), ds03.stretch(r)]))
    n = img.shape[0]; c = n // 6
    return cv2.resize(img[c:n - c, c:n - c], (m8.SIZE, m8.SIZE), interpolation=cv2.INTER_AREA)


imgs = {'original': [], 'reregistered': [], 'reregistered_psfmatched': []}; info = []
for _, row in st.iterrows():
    with fits.open(D / 'cutouts' / row.file) as h:
        ha = np.nan_to_num(h[0].data.astype(float)); r = np.nan_to_num(h['R'].data.astype(float)) if 'R' in h else None
    k = float(pl.kcont.get(row.file, np.nan))
    if r is None or not np.isfinite(k):
        info.append(dict(name=row.Name, used=False)); continue
    xs, ys = stars(r); mh, mr = mom(ha, xs, ys), mom(r, xs, ys)
    pairs = [(a, b) for a, b in zip(mh, mr) if a is not None and b is not None]
    if len(pairs) < 3:
        info.append(dict(name=row.Name, used=False)); continue
    dx = float(np.mean([a[0] - b[0] for a, b in pairs])); dy = float(np.mean([a[1] - b[1] for a, b in pairs]))
    s2h = float(np.median([a[2] for a, b in pairs])); s2r = float(np.median([b[2] for a, b in pairs]))
    rs = ndshift(r, (dy, dx), order=3, mode='nearest')
    # Gaussian blur of the sharper band so that both have the same second moment (sum of the two variances)
    dv = (s2h - s2r) / 2
    ham, rsm = (ha, gaussian_filter(rs, np.sqrt(dv))) if dv > 0 else (gaussian_filter(ha, np.sqrt(-dv)), rs)
    imgs['original'].append(build(ha, r, k)); imgs['reregistered'].append(build(ha, rs, k)); imgs['reregistered_psfmatched'].append(build(ham, rsm, k))
    info.append(dict(name=row.Name, used=True, dx=dx, dy=dy, offset_px=float(np.hypot(dx, dy)), offset_arcsec=float(np.hypot(dx, dy) * row.pixscale),
                     sigma2_ha=s2h, sigma2_r=s2r, pa_catalogue=float(row.pa)))
m = m8.model(); m.load_state_dict(torch.load(D / 'pa_cnn_full.pt', map_location='cpu')); m = m.to(m8.DEV)
out = dict(n_total=len(st), n_used=int(sum(i['used'] for i in info)))
for key, im in imgs.items():
    xt = torch.tensor(np.stack(im)).permute(0, 3, 1, 2).float() / 255
    pa, R = m8.predict(m, xt)
    z = np.exp(2j * np.radians(pa)); n = len(pa); Rm = float(np.abs(z.mean()))
    out[key] = dict(n=n, R=Rm, mean=float(np.degrees(np.angle(z.mean())) / 2 % 180), rayleigh_p=float(np.exp(-n * Rm ** 2)),
                    V113=float(np.mean(np.cos(np.radians(2 * (pa - 113))))), median_R_tta=float(np.median(R)))
    out[key + '_pa'] = pa.tolist(); print(key, out[key], flush=True)
off = np.array([i['offset_arcsec'] for i in info if i['used']]); out['offset_arcsec_median'] = float(np.median(off))
out['objects'] = info
# Rayleigh power for n = 18 at several true concentrations (wrapped normal), alpha 0.05
rng = np.random.default_rng(5); pw = {}
for rho in (0.3, 0.4, 0.5):
    s = np.sqrt(-2 * np.log(rho)); th = rng.normal(0, s, (200000, 18)); Rr = np.abs(np.exp(1j * th).mean(1))
    p = np.exp(np.sqrt(1 + 4 * 18 + 4 * (18 ** 2 - (18 * Rr) ** 2)) - (1 + 2 * 18)); pw[str(rho)] = float(np.mean(p < 0.05))
out['rayleigh_power_n18'] = pw
json.dump(out, open(D / 'rev8_shs_reregister.json', 'w'), indent=1)

# ---- second step: the same network with H alpha in all three channels, and with r and the continuum subtracted
# channel taken from H alpha rotated by 90 deg (destroys any common orientation of the stars between the bands)
ha_only, ha_rot = [], []
for _, row in st.iterrows():
    with fits.open(D / 'cutouts' / row.file) as h:
        ha = np.nan_to_num(h[0].data.astype(float)); has_r = 'R' in h
    k = float(pl.kcont.get(row.file, np.nan))
    if not has_r or not np.isfinite(k):
        continue
    if not [i for i in info if i['name'] == row.Name and i['used']]:
        continue
    s = ds03.stretch(ha); img = np.flipud(np.dstack([s, s, s])); n = img.shape[0]; c = n // 6
    ha_only.append(cv2.resize(img[c:n - c, c:n - c], (m8.SIZE, m8.SIZE), interpolation=cv2.INTER_AREA))
for key, im in [('halpha_all_channels', ha_only)]:
    xt = torch.tensor(np.stack(im)).permute(0, 3, 1, 2).float() / 255
    pa, R = m8.predict(m, xt); z = np.exp(2j * np.radians(pa)); n = len(pa); Rm = float(np.abs(z.mean()))
    out[key] = dict(n=n, R=Rm, mean=float(np.degrees(np.angle(z.mean())) / 2 % 180), rayleigh_p=float(np.exp(-n * Rm ** 2)), median_R_tta=float(np.median(R)))
    print(key, out[key], flush=True)
# control: the same H alpha only input for the stellar PNe in IPHAS
sti = x[(x.mainClass == 'S') & (x.survey == 'IPHAS') & x.MajDiam.notna()]; im = []
for _, row in sti.iterrows():
    with fits.open(D / 'cutouts' / row.file) as h:
        ha = np.nan_to_num(h[0].data.astype(float))
    s = ds03.stretch(ha); img = np.flipud(np.dstack([s, s, s])); n = img.shape[0]; c = n // 6
    im.append(cv2.resize(img[c:n - c, c:n - c], (m8.SIZE, m8.SIZE), interpolation=cv2.INTER_AREA))
xt = torch.tensor(np.stack(im)).permute(0, 3, 1, 2).float() / 255
pa, R = m8.predict(m, xt); z = np.exp(2j * np.radians(pa)); Rm = float(np.abs(z.mean()))
out['iphas_stellar_halpha_all_channels'] = dict(n=len(pa), R=Rm, rayleigh_p=float(np.exp(-len(pa) * Rm ** 2)))
print('iphas', out['iphas_stellar_halpha_all_channels'])
json.dump(out, open(D / 'rev8_shs_reregister.json', 'w'), indent=1)
