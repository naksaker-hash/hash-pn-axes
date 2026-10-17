"""Copy of 29_revision4b_analyses.py with the tight injections along the plane (revision 7). Output: data/rev7_conservative90_29.json"""
"""Revision 4b analyses (second referee of round 4). Output: data/rev4b.json

a. external errors of unseen PNe (Rees & Zijlstra 2013, Corradi et al. 1998) at several R_TTA thresholds and by
   size, class and survey, together with the cross validation errors by the same strata
b. Monte Carlo spread of the injection limits for the bipolars outside the bulge region (ten repetitions)
c. injection with errors drawn per object from the cross validation residuals of the same survey and size bin
d. bipolars outside the bulge region for several thresholds of the network bipolar probability
e. bootstrap 68 per cent intervals of R and of the mean axis for the main samples
"""
import json, re, pathlib
import numpy as np, pandas as pd
from scipy import stats
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R), V=float(V),
                V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
out = {}

# ---------- a. external errors by threshold and stratum ----------
h = x[['idPNMain', 'Name', 'png', 'mainClass', 'MajDiam', 'survey', 'pa', 'R_tta', 'trained']].copy()
h['nkey'] = h.Name.map(norm)
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(norm)
e_rz = h.merge(rz[['PNG', 'EPA_deg']], left_on='png', right_on='PNG').assign(cat='RZ13')
b1 = h.merge(co[['nkey', 'EPA_deg']], on='nkey')
b2 = h.merge(co[['PNG_approx_from_lb', 'EPA_deg']], left_on='png', right_on='PNG_approx_from_lb')
e_co = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain').assign(cat='Corradi98')
ext = pd.concat([e_rz, e_co]).drop_duplicates('idPNMain')
ext = ext[~ext.trained & ext.mainClass.isin(['B', 'E', 'I', 'A']) & (ext.MajDiam >= CUT)].copy()
ext['d'] = dpa(ext.pa, ext.EPA_deg)
cvs = cv[cv.mainClass.isin(['B', 'E', 'I', 'A']) & (cv.MajDiam >= CUT)].copy()
cvs['d'] = cvs.d_cnn; cvs['R_tta'] = cvs.cnn_R
summ = lambda t: dict(n=int(len(t)), median=float(t.d.median()) if len(t) else None,
                      gt60=float((t.d > 60).mean()) if len(t) else None, gt30=float((t.d > 30).mean()) if len(t) else None)
a = {}
for nm, t in [('external', ext), ('cv', cvs)]:
    r = {}
    for thr in [0.0, 0.5, 0.8, 0.9, 0.95]:
        r[f'R_TTA>={thr}'] = summ(t[t.R_tta >= thr])
    r['R_TTA<0.8'] = summ(t[t.R_tta < 0.8])
    s = t[t.R_tta >= 0.8]
    for lo, hi in [(6, 10), (10, 20), (20, 1e9)]:
        r[f'size {lo}-{hi}'] = summ(s[(s.MajDiam >= lo) & (s.MajDiam < hi)])
    for cl in ['B', 'E']:
        r[f'class {cl}'] = summ(s[s.mainClass == cl])
    for sv in ['IPHAS', 'VPHAS', 'SHS']:
        r[f'survey {sv}'] = summ(s[s.survey == sv])
    pos, neg = t[t.d > 60].R_tta.values, t[t.d <= 60].R_tta.values
    r['auc_gt60'] = float(stats.mannwhitneyu(neg, pos).statistic / (len(pos) * len(neg))) if len(pos) else None
    a[nm] = r
out['errors_by_stratum'] = a

# ---------- injection helpers ----------
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B')
ob = x[B & ~x.bulge_sky]
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
cvb = cv[mres].copy(); cvb['res'] = wrap(cvb.cnn_pa - cvb.EPA)
grid = np.round(np.arange(0, 1.001, 0.02), 2)


def limit(pa_obs, R_al, mu, draw_err, rng, nsim=3000):
    n = len(pa_obs); Vobs = np.mean(np.cos(np.radians(2 * (np.asarray(pa_obs) - mu)))); sig2 = np.sqrt(-2 * np.log(R_al))
    for f in grid:
        Vs = np.empty(nsim)
        for k in range(nsim):
            al = rng.random(n) < f
            pa = np.where(al, mu + np.degrees(rng.normal(0, sig2, n)) / 2, rng.uniform(0, 180, n)) + draw_err(rng)
            Vs[k] = np.mean(np.cos(np.radians(2 * (pa - mu))))
        if np.mean(Vs <= Vobs) < 0.05:
            return float(f)
    return None


res_all = cvb.res.values
pooled = lambda rng: rng.choice(res_all, len(ob))
# ---------- b. Monte Carlo spread ----------
b = {}
for nm, (R_al, mu) in [('tight', (0.83, 90.0)), ('visual_RZ13', (0.29, 90.0))]:
    lims = [limit(ob.gpa.values, R_al, mu, pooled, np.random.default_rng(100 + s)) for s in range(10)]
    b[nm] = dict(limits=lims, min=min(lims), max=max(lims), median=float(np.median(lims)))
out['injection_mc'] = b

# ---------- c. stratified errors ----------
sbin = lambda d: np.digitize(d, [10, 20])
cvb['stratum'] = cvb.survey + '_' + sbin(cvb.MajDiam).astype(str)
ob_str = (ob.survey + '_' + sbin(ob.MajDiam).astype(str)).values
pools = {k: g.res.values for k, g in cvb.groupby('stratum')}
fallback = {sv: cvb[cvb.survey == sv].res.values for sv in ['IPHAS', 'VPHAS']}
strata_pools = [pools.get(k, fallback[k.split('_')[0]]) if len(pools.get(k, [])) >= 5 else fallback[k.split('_')[0]] for k in ob_str]
strat = lambda rng: np.array([p[rng.integers(len(p))] for p in strata_pools])
rng = np.random.default_rng(7)
out['injection_stratified'] = dict(strata={k: int(len(v)) for k, v in pools.items()},
                                   tight=limit(ob.gpa.values, 0.83, 90.0, strat, rng, nsim=2000),
                                   visual_RZ13=limit(ob.gpa.values, 0.29, 90.0, strat, rng, nsim=2000))

# ---------- d. bipolar probability thresholds ----------
out['p_B_thresholds'] = {str(t): circ(ob[ob.p_B >= t].gpa) for t in [0.5, 0.7, 0.9]}

# ---------- e. bootstrap intervals ----------
rng = np.random.default_rng(11)
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['GPA_deg'])
bj = pd.read_csv(D / 'supp_binaries.csv'); bsel = bj[bj.in_alignment_test]
E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
samp = {'RZ13 selected, visual': x[(x.morph_class == 'bipolar') & base].gpa_rz,
        'RZ13 selected, network': x[(x.morph_class == 'bipolar') & base].gpa,
        'Tan all, network': tan.gpa, 'Tan all, visual': tan.GPA_deg,
        'bulge region other B': x[B & x.bulge_sky & x.morph_class.isna()].gpa,
        'outside bulge region B': ob.gpa, 'outside bulge region E': x[E & ~x.bulge_sky].gpa,
        'binaries outside bulge region': bsel[~bsel.bulge_sky].gpa}
e = {}
for nm, v in samp.items():
    v = np.asarray(v.dropna(), float); Rs, ms = [], []
    for _ in range(2000):
        z = np.exp(2j * np.radians(rng.choice(v, len(v)))).mean(); Rs.append(abs(z)); ms.append(np.degrees(np.angle(z)) / 2 % 180)
    m0 = circ(v)['mean']; ms = m0 + wrap(np.array(ms) - m0)
    e[nm] = dict(n=len(v), R=circ(v)['R'], R68=[float(np.percentile(Rs, 16)), float(np.percentile(Rs, 84))],
                 mean=m0, mean68=[float(np.percentile(ms, 16) % 180), float(np.percentile(ms, 84) % 180)])
out['bootstrap'] = e

json.dump(out, open(D / 'rev7_conservative90_29.json', 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:1800])
