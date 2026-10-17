"""Clean close binaries (revision 7): limits for the bulge region with the residuals against Rees & Zijlstra (2013),
as stated in Section 2.6, and expected signals for a tight alignment along the plane. Output: data/rev7_binaries_rz.json"""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
rng = np.random.default_rng(81)
wrap = lambda a: (a + 90) % 180 - 90
x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
mres = (cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
err_cv = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png'); xr = xr[~xr.trained & (xr.MajDiam >= 4)]
err_rz = wrap(xr.pa - xr.EPA_deg).dropna().values
b = pd.read_csv(D / 'binaries_clean.csv'); b = b[b.in_alignment_test]
Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu)), axis=-1)
fg = np.round(np.arange(0.02, 1.001, 0.02), 2)


def limit(pa_obs, e, nsim=3000, R_al=0.83, mu=90.0):
    n = len(pa_obs); Vobs = Vat(pa_obs, mu); s2 = np.sqrt(-2 * np.log(R_al)); P = []
    for f in fg:
        al = rng.random((nsim, n)) < f
        pa = np.where(al, mu + np.degrees(rng.normal(0, s2, (nsim, n))) / 2, rng.uniform(0, 180, (nsim, n))) + rng.choice(e, (nsim, n))
        P.append(np.mean(Vat(pa, mu) <= Vobs))
    P = np.array(P); i = np.flatnonzero(P < 0.05)
    if not len(i):
        return None
    j = i[0]; return float(fg[j]) if j == 0 else float(fg[j - 1] + (P[j - 1] - 0.05) * (fg[j] - fg[j - 1]) / (P[j - 1] - P[j]))


out = dict(Ecos2_cv=float(np.mean(np.cos(np.radians(2 * err_cv)))), Ecos2_rz=float(np.mean(np.cos(np.radians(2 * err_rz)))))
for nm, mk in [('outside', ~b.bulge_sky), ('bulge', b.bulge_sky)]:
    s = b[mk]; sB = s[s.mainClass == 'B']; n = len(s)
    r = dict(n=n, n_B=len(sB), expected_sigma_plane_cv=float(0.83 * out['Ecos2_cv'] * np.sqrt(2 * n)),
             f95_cv=limit(s.gpa.values, err_cv), f95_B_cv=limit(sB.gpa.values, err_cv))
    if nm == 'bulge':
        r.update(expected_sigma_plane_rz=float(0.83 * out['Ecos2_rz'] * np.sqrt(2 * n)), f95_rz=limit(s.gpa.values, err_rz), f95_B_rz=limit(sB.gpa.values, err_rz))
    out[nm] = r
json.dump(out, open(D / 'rev7_binaries_rz.json', 'w'), indent=1); print(json.dumps(out, indent=1))
