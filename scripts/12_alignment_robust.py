"""Robustness of the alignment result.

1. Injection: for the disc bipolar sample, a fraction f of objects gets an axis drawn from the RZ13
   bulge bipolar distribution (doubled-angle mean resultant 0.40 about GPA 90), the rest are uniform.
   Every simulated axis gets an error drawn from the out-of-fold CNN residuals (signed, with flips).
   Detection = V test toward GPA 90 at p < 0.05 (one sided). Gives power(f) and the 95 per cent upper
   limit on f from the observed V.
2. Splits of the observed samples: size, survey, latitude, CNN consistency, bulge sky vs not.
Output: data/alignment_robust.json
"""
import json, pathlib, importlib, sys
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    al = importlib.import_module('11_alignment')
x = al.x.copy()
for c in ['Glon', 'Glat']:
    x[c] = pd.to_numeric(x[c], errors='coerce')
rng = np.random.default_rng(1)
wrap = lambda a: (a + 90) % 180 - 90

cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
resid = wrap(cv.cnn_pa - cv.EPA).dropna().values  # signed out-of-fold residuals (deg)
resid_B = wrap(cv[cv.mainClass == 'B'].cnn_pa - cv[cv.mainClass == 'B'].EPA).dropna().values

R_bulge = 0.40
sig2 = np.sqrt(-2 * np.log(R_bulge))  # wrapped normal on doubled angles with mean resultant R_bulge


def V(pa):
    return np.mean(np.cos(np.radians(2 * pa) - np.pi))


def vtest_p(pa):
    n = len(pa)
    return stats.norm.sf(V(pa) * np.sqrt(2 * n))


l = ((x.Glon + 180) % 360) - 180
bulge_sky = (np.abs(l) < 10) & (np.abs(x.Glat) < 10)
base = (x.MajDiam >= 6) & (x.cnn_R >= 0.8)
B = base & (x.mainClass == 'B')
disc = x[B & ~bulge_sky]
n = len(disc)
Vobs = V(disc.gpa.values)

res = {'n_disc_B': n, 'V_obs': float(Vobs), 'V_obs_p': float(vtest_p(disc.gpa.values))}
power, vq = {}, {}
for f in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    det, Vs = 0, []
    for k in range(2000):
        na = rng.binomial(n, f)
        al_ = 90 + np.degrees(rng.normal(0, sig2, na)) / 2
        un = rng.uniform(0, 180, n - na)
        pa = np.r_[al_, un] + rng.choice(resid_B, n)
        Vs.append(V(pa))
        det += vtest_p(pa) < 0.05
    power[f] = det / 2000
    vq[f] = float(np.mean(np.array(Vs) <= Vobs))  # P(V <= observed | f)
    print(f'f {f:.1f}: power {power[f]:.2f}  P(V<=Vobs) {vq[f]:.3f}', flush=True)
res['power'] = power
res['P_V_le_obs'] = vq
fs = sorted(vq)
up = next((f for f in fs if vq[f] < 0.05), None)
res['f_upper95_grid'] = up
print('95 per cent upper limit on the aligned fraction (grid):', up)

splits = {}
def add(name, mask):
    s = x[mask]
    splits[name] = {k: v for k, v in al.circ(s.gpa).items() if k in ['n', 'mean', 'R', 'rayleigh_p', 'V', 'V_p_parallel', 'frac_plane30']}
    print(f'{name:40s}', {k: round(v, 3) if isinstance(v, float) else v for k, v in splits[name].items()})

for lo, hi in [(6, 12), (12, 25), (25, 60), (60, 1e4)]:
    add(f'disc B MajDiam {lo}-{hi}', B & ~bulge_sky & x.MajDiam.between(lo, hi, inclusive='left'))
for sv in ['IPHAS', 'VPHAS', 'SHS']:
    add(f'disc B survey {sv}', B & ~bulge_sky & (x.survey == sv))
add('disc B |b|<2', B & ~bulge_sky & (np.abs(x.Glat) < 2))
add('disc B |b|>=2', B & ~bulge_sky & (np.abs(x.Glat) >= 2))
add('disc B cnn_R>=0.95', (x.MajDiam >= 6) & (x.cnn_R >= 0.95) & (x.mainClass == 'B') & ~bulge_sky)
add('disc B inner |l|<60', B & ~bulge_sky & (np.abs(l) < 60))
add('disc B outer |l|>=60', B & ~bulge_sky & (np.abs(l) >= 60))
for lo, hi in [(6, 12), (12, 1e4)]:
    add(f'bulge-sky B MajDiam {lo}-{hi}', B & bulge_sky & x.MajDiam.between(lo, hi, inclusive='left'))
add('bulge-sky B no size cut', (x.cnn_R >= 0.8) & (x.mainClass == 'B') & bulge_sky)
add('bulge-sky E q<0.85', base & (x.mainClass == 'E') & (x.q_mask < 0.85) & bulge_sky)
add('disc E q<0.85', base & (x.mainClass == 'E') & (x.q_mask < 0.85) & ~bulge_sky)
res['splits'] = splits
json.dump(res, open(ROOT / 'data' / 'alignment_robust.json', 'w'), indent=1)
