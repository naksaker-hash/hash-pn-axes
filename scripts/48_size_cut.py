"""Size cut test (author question: can PNe smaller than 6 arcsec be used?). CPU only. Output: data/size_cut.json
For cuts of 6, 5, 4 and 3 arcsec: sample sizes, cross validation and external errors by size bin, the instrumental
check on axisless stellar PNe and on small axes in the frame of the images, the V tests and the injection limit."""
import json, pathlib, re
import numpy as np, pandas as pd
from scipy import stats
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'; rng = np.random.default_rng(111)
wrap = lambda a: (a + 90) % 180 - 90; dpa = lambda a, b: np.abs(wrap(a - b))
Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * (pa - mu))), axis=-1)
def circ(pa, mu=90.0):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    if n < 2: return dict(n=n)
    z = np.exp(2j * np.radians(pa)).mean(); R = abs(z); V = np.mean(np.cos(np.radians(2 * (pa - mu))))
    return dict(n=n, R=float(R), mean=float(np.degrees(np.angle(z)) / 2 % 180), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))))
x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
cv['d'] = dpa(cv.cnn_pa, cv.EPA); cv['res'] = wrap(cv.cnn_pa - cv.EPA)
out = {}
# errors by size in cross validation (labels exist down to 4 arcsec) and against Rees & Zijlstra (unseen bulge PNe)
bins = [(0, 4), (4, 5), (5, 6), (6, 10), (10, 1e9)]
cvb = cv[(cv.cnn_R >= 0.8)]
out['cv_by_size'] = {f'{a}-{b}': dict(n=int(((cvb.MajDiam >= a) & (cvb.MajDiam < b)).sum()), median=float(cvb[(cvb.MajDiam >= a) & (cvb.MajDiam < b)].d.median()) if ((cvb.MajDiam >= a) & (cvb.MajDiam < b)).sum() else None,
                             gt60=float((cvb[(cvb.MajDiam >= a) & (cvb.MajDiam < b)].d > 60).mean()) if ((cvb.MajDiam >= a) & (cvb.MajDiam < b)).sum() else None) for a, b in bins}
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
xr = xr[xr.rz_class.notna() & (xr.R_tta >= 0.8) & (xr.survey != 'SHS')].copy(); xr['d'] = dpa(xr.gpa, xr.GPA_deg)
out['rz_by_size'] = {f'{a}-{b}': dict(n=int(((xr.MajDiam >= a) & (xr.MajDiam < b)).sum()), median=float(xr[(xr.MajDiam >= a) & (xr.MajDiam < b)].d.median()) if ((xr.MajDiam >= a) & (xr.MajDiam < b)).sum() else None,
                             gt60=float((xr[(xr.MajDiam >= a) & (xr.MajDiam < b)].d > 60).mean()) if ((xr.MajDiam >= a) & (xr.MajDiam < b)).sum() else None) for a, b in bins}
# instrumental control: axisless stellar PNe and small axes in the frame of the images (equatorial PA), IPHAS and VPHAS+
inst = {}
for sv in ('IPHAS', 'VPHAS'):
    st = x[(x.mainClass == 'S') & (x.survey == sv) & (x.R_tta >= 0.8) & x.MajDiam.notna()]
    inst[f'stellar {sv}'] = circ(st.pa)
    for a, b in [(3, 4), (4, 5), (5, 6), (6, 1e9)]:
        s = x[x.mainClass.isin(['B', 'E']) & (x.survey == sv) & (x.R_tta >= 0.8) & (x.MajDiam >= a) & (x.MajDiam < b)]
        inst[f'B+E {sv} {a}-{b} equatorial'] = circ(s.pa)
out['instrumental'] = inst
# samples and tests
bj = pd.read_csv(D / 'binaries_clean.csv')
res = {}
for cut in (6, 5, 4, 3):
    base = (x.MajDiam >= cut) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
    B = base & (x.mainClass == 'B')
    ob = x[B & ~x.bulge_sky]; bb = x[B & x.bulge_sky & x.morph_class.isna()]; rzs = x[B & (x.morph_class == 'bipolar')] if 'morph_class' in x else None
    xrz = xr[(xr.rz_class == 'bipolar') & (xr.MajDiam >= cut)]
    cat = x[(x.MajDiam >= cut) & (x.R_tta >= 0.8) & x.mainClass.isin(['B', 'E', 'I', 'A'])]
    bsel = bj[(bj.MajDiam >= cut) & (bj.R_tta >= 0.8) & (bj.survey != 'SHS') & bj.mainClass.isin(['B', 'E', 'I', 'A']) & bj.short]
    r = dict(catalogue=len(cat), catalogue_new=int(cat.EPA.isna().sum()), outside_B=circ(ob.gpa), bulge_other_B=circ(bb.gpa),
             rz_bipolars_visual=circ(xrz.GPA_deg), rz_bipolars_network=circ(xrz.gpa), binaries_outside=circ(bsel[~bsel.bulge_sky].gpa), binaries_bulge=circ(bsel[bsel.bulge_sky].gpa))
    # error pool: cross validation residuals of selected bipolars down to the cut (non SHS)
    m = (cv.mainClass == 'B') & (cv.MajDiam >= max(cut, 4)) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS'); pool = cv[m].res.dropna().values
    gpa = ob.gpa.values; n = len(gpa); s2 = np.sqrt(-2 * np.log(0.83)); Vo = Vat(gpa, 90.0)
    lim = None
    for f in np.round(np.arange(0.02, 0.6, 0.01), 2):
        al = rng.random((6000, n)) < f
        pa = np.where(al, 90 + np.degrees(rng.normal(0, s2, (6000, n))) / 2, rng.uniform(0, 180, (6000, n))) + rng.choice(pool, (6000, n))
        if np.mean(Vat(pa, 90.0) <= Vo) < 0.05: lim = float(f); break
    r['limit_tight_plane_grid'] = lim; r['pool_n'] = len(pool); r['pool_Ecos2'] = float(np.mean(np.cos(np.radians(2 * pool))))
    res[str(cut)] = r; print(cut, json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
out['cuts'] = res
json.dump(out, open(D / 'size_cut.json', 'w'), indent=1)
print(json.dumps(out['cv_by_size'])); print(json.dumps(out['rz_by_size'])); print(json.dumps(out['instrumental'], indent=0)[:2500])
