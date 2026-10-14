"""Final injection tests (referee round 3), all scored with the V test at the direction of the injected alignment.

Samples: bipolars outside the bulge region, other bipolars of the bulge region, PNe with known close binary nuclei
(all and confirmed only) outside and inside the bulge region.
Aligned populations: tight (R 0.83 about 107.5 deg, like Tan et al. 2023) and the visual concentration of the
selected bipolars of Rees & Zijlstra (2013) about 90 deg.
Error models: cross validation residuals of selected bipolars ('cv'), residuals against the visual axes of
Rees & Zijlstra ('rz', for bulge samples), or none.
Output: data/injection_final.json (limits and full grids)
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(31)
wrap = lambda a: (a + 90) % 180 - 90
x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
rv = json.load(open(D / 'rev1.json'))
mres = (cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
err = {'cv': wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values}
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png')
xr = xr[~xr.trained & (xr.MajDiam >= 4)]
err['rz'] = wrap(xr.pa - xr.EPA_deg).dropna().values
err['none'] = None
R_vis = rv['samples']['RZ13 bipolar, visual, same objects as cuts']['R']

base = (x.MajDiam >= 6) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B')
bj = pd.read_csv(D / 'supp_binaries.csv')
bsel = bj[bj.in_alignment_test]
samples = {
    'outside bulge region B': x[B & ~x.bulge_sky].gpa.values,
    'bulge region other B': x[B & x.bulge_sky & x.morph_class.isna()].gpa.values,
    'binaries outside bulge region, all': bsel[~bsel.bulge_sky].gpa.values,
    'binaries outside bulge region, confirmed': bsel[~bsel.bulge_sky & ~bsel.candidate].gpa.values,
    'binaries bulge region, all': bsel[bsel.bulge_sky].gpa.values,
    'binaries bulge region, confirmed': bsel[bsel.bulge_sky & ~bsel.candidate].gpa.values,
}
pops = {'tight': (0.83, 107.5), 'visual RZ13': (R_vis, 90.0)}
grid = np.round(np.arange(0, 1.001, 0.02), 2)


def V(pa, mu):
    return np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu)))


def run(pa_obs, R_al, mu, e, nsim=3000):
    n = len(pa_obs); Vobs = V(pa_obs, mu); sig2 = np.sqrt(-2 * np.log(R_al)); res = {}
    for f in grid:
        Vs = np.empty(nsim)
        for k in range(nsim):
            na = rng.binomial(n, f)
            pa = np.r_[mu + np.degrees(rng.normal(0, sig2, na)) / 2, rng.uniform(0, 180, n - na)]
            if e is not None:
                pa = pa + rng.choice(e, n)
            Vs[k] = V(pa, mu)
        res[float(f)] = float(np.mean(Vs <= Vobs))
    return dict(n=n, V_obs=float(Vobs), f95=next((f for f in sorted(res) if res[f] < 0.05), None), grid=res)


out = {'R_visual_RZ13_selected': R_vis, 'Ecos2': {k: (float(np.mean(np.cos(np.radians(2 * v)))) if v is not None else 1.0) for k, v in err.items()},
       'n': {k: int(len(v)) for k, v in samples.items()}}
for sname, pa in samples.items():
    for pname, (R_al, mu) in pops.items():
        for ename in (['cv', 'none'] if 'outside' in sname else ['cv', 'rz', 'none']):
            out[f'{sname} | {pname} | {ename}'] = run(pa, R_al, mu, err[ename])
            r = out[f'{sname} | {pname} | {ename}']
            print(f'{sname:42s} {pname:12s} {ename:5s} n {r["n"]:3d} f95 {r["f95"]}', flush=True)
json.dump(out, open(D / 'injection_final.json', 'w'), indent=1)
