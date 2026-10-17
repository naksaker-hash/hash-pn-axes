"""Revision 8 (third deep review): all injection limits for the 266 bipolars outside the bulge region on one procedure.
Every limit is the linear interpolation of the crossing of 0.05 on a grid of 0.01 in f with 10000 samples per step,
for the five error models, the error free case, the tight (R 0.83) and weaker (R_WEAK, the bias corrected visual R of the selected bipolars of Rees & Zijlstra) alignments, and for the
alignment injected at 90 or 107.5 deg and scored at 90 (confirmatory) or 107.5 (oracle). Also the bulge known
binaries: limits, expected limits and the probability of the observed V statistic if all are aligned.
Output: data/rev8_limits.json"""
import json, re, pathlib
import numpy as np, pandas as pd
CUT = float(__import__('os').environ.get('CUT', '2'))
R_WEAK = float(__import__('os').environ.get('R_WEAK', '0.38'))  # bias corrected visual R of the 28 selected bipolars of Rees & Zijlstra (data/cut4.json)  # minimum HASH major diameter (arcsec) of the selected axes
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(81)
wrap = lambda a: (a + 90) % 180 - 90
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv'); cat = pd.read_csv(D / 'catalogue_pa.csv')
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
ob = x[base & (x.mainClass == 'B') & ~x.bulge_sky]; gpa = ob.gpa.values; n = len(gpa)
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
cvb = cv[mres].copy(); cvb['res'] = wrap(cvb.cnn_pa - cvb.EPA); cvres = cvb.res.dropna().values

# ---- error pools (as in scripts 39a and 39b) ----
h = x[['idPNMain', 'Name', 'png', 'mainClass', 'MajDiam', 'survey', 'pa', 'R_tta', 'trained']].copy(); h['nkey'] = h.Name.map(norm)
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(norm)
b1 = h.merge(co[['nkey', 'EPA_deg']], on='nkey')
b2 = h.merge(co[['PNG_approx_from_lb', 'EPA_deg']], left_on='png', right_on='PNG_approx_from_lb')
e_co = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain')
rel = lambda t: t[t.mainClass.isin(['B', 'E', 'I', 'A']) & (t.MajDiam >= CUT) & (t.R_tta >= 0.8) & ~t.trained]
co_new = rel(e_co); co_new = co_new[co_new.survey != 'SHS']; err_ext = wrap(co_new.pa - co_new.EPA_deg).values
bins = [0.8, 0.9, 0.95, 1.0001]
cvr = cv[cv.mainClass.isin(['B', 'E', 'I', 'A']) & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8)].copy(); cvr['bin'] = pd.cut(cvr.cnn_R, bins, right=False)
un = cat[(cat.flag_reliable == 1) & (cat.flag_out_of_fold == 0)].copy(); un['bin'] = pd.cut(un.R_TTA, bins, right=False)
cvb2 = cvr[(cvr.survey != 'SHS') & (cvr.mainClass == 'B')]; ub = un[(un.HASH_class == 'B') & (un.flag_SHS == 0)]
wub = ub.bin.value_counts(normalize=True).sort_index(); wlb = cvb2.bin.value_counts(normalize=True).sort_index()
wb = cvb2.bin.map(wub / wlb).astype(float).to_numpy().copy(); wb /= wb.sum()
err_rw = wrap(cvb2.cnn_pa - cvb2.EPA).values[rng.choice(len(cvb2), 20000, p=wb)]
gt = (np.abs(cvres) > 60).mean()
err_fail = np.where(rng.random(20000) < gt, rng.choice(cvres[np.abs(cvres) > 60], 20000), rng.choice(cvres, 20000))
sbin = lambda d: np.digitize(d, [10, 20])
cvb['stratum'] = cvb.survey + '_' + sbin(cvb.MajDiam).astype(str)
ob_str = (ob.survey + '_' + sbin(ob.MajDiam).astype(str)).values
pools = {k: g.res.values for k, g in cvb.groupby('stratum')}
fallback = {sv: cvb[cvb.survey == sv].res.values for sv in ['IPHAS', 'VPHAS']}
spools = [pools[k] if len(pools.get(k, [])) >= 5 else fallback[k.split('_')[0]] for k in ob_str]
pool_draw = lambda pool: (lambda ns, nn: rng.choice(pool, (ns, nn)))
def strat_draw(ns, nn):
    return np.stack([rng.choice(p, ns) for p in spools], 1)
models = {'cv': pool_draw(cvres), 'external Corradi': pool_draw(err_ext), 'reweighted cv': pool_draw(err_rw),
          'survey and size bins': strat_draw, 'doubled failures': pool_draw(err_fail), 'no errors': lambda ns, nn: np.zeros((ns, nn))}
Ecos = {k: float(np.mean(np.cos(np.radians(2 * v)))) for k, v in [('cv', cvres), ('external Corradi', err_ext), ('reweighted cv', err_rw), ('doubled failures', err_fail)]}

Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * (pa - mu))), axis=-1)


def P_of_f(pa_obs, R_al, mu_inj, mu_test, f, draw, ns):
    nn = len(pa_obs); sig2 = np.sqrt(-2 * np.log(R_al)) if R_al < 1 else 0.0
    al = rng.random((ns, nn)) < f
    pa = np.where(al, mu_inj + np.degrees(rng.normal(0, sig2, (ns, nn))) / 2, rng.uniform(0, 180, (ns, nn))) + draw(ns, nn)
    return float(np.mean(Vat(pa, mu_test) <= Vat(pa_obs, mu_test)))


def limit(pa_obs, R_al, mu_inj, mu_test, draw, ns=10000, step=0.01, ns_coarse=1500):
    """Coarse search on 0.05, then a fine grid of `step` with ns samples around the bracket and linear interpolation."""
    coarse = np.round(np.arange(0, 1.0001, 0.05), 3); c = None
    for f in coarse:
        if P_of_f(pa_obs, R_al, mu_inj, mu_test, f, draw, ns_coarse) < 0.05:
            c = f; break
    if c is None:
        return dict(f95=None)
    fg = np.round(np.arange(max(0, c - 0.10), min(1, c + 0.06) + 1e-9, step), 3)
    P = np.array([P_of_f(pa_obs, R_al, mu_inj, mu_test, f, draw, ns) for f in fg])
    i = np.flatnonzero(P < 0.05)
    if not len(i):
        return dict(f95=None)
    j = i[0]
    if j == 0:
        return dict(f95=float(fg[0]), note='at lower edge')
    cr = float(fg[j - 1] + (P[j - 1] - 0.05) * (fg[j] - fg[j - 1]) / (P[j - 1] - P[j]))
    slope = (P[j] - P[j - 1]) / (fg[j] - fg[j - 1])
    return dict(f95=cr, mc_se=float(np.sqrt(0.05 * 0.95 / ns) / abs(slope)) if slope else None, grid=fg.tolist(), P=P.tolist())


out = dict(n=n, Ecos2=Ecos, cv_gt60=float(gt))
lim = {}
for m, draw in models.items():
    r = {}
    for lab, R_al, mi, mt in [('tight_90_90', 0.83, 90.0, 90.0), ('tight_107_90', 0.83, 107.5, 90.0), ('tight_107_107', 0.83, 107.5, 107.5),
                              ('weak_90_90', R_WEAK, 90.0, 90.0)]:
        r[lab] = limit(gpa, R_al, mi, mt, draw); print(m, lab, r[lab].get('f95'), r[lab].get('mc_se'), flush=True)
    lim[m] = r
out['limits'] = lim
# ---- bulge known binaries ----
bj = pd.read_csv(D / 'binaries_clean.csv'); bs = bj[bj.in_alignment_test & bj.bulge_sky]
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
xr = xr[xr.rz_class.notna()]; err_rz = wrap(xr.gpa - xr.GPA_deg).dropna().values
bm = {}
for nm, s in [('all', bs), ('B', bs[bs.mainClass == 'B'])]:
    pa_o = s.gpa.values; nn = len(pa_o); r = dict(n=nn)
    for en, pool in [('cv', cvres), ('RZ13', err_rz)]:
        Vobs = Vat(pa_o, 90.0); sig2 = np.sqrt(-2 * np.log(0.83)); ns = 200000
        pa = 90 + np.degrees(rng.normal(0, sig2, (ns, nn))) / 2 + rng.choice(pool, (ns, nn))
        Vall = Vat(pa, 90.0); p_all = float(np.mean(Vall <= Vobs))
        Vn = Vat(rng.uniform(0, 180, (20000, nn)) + rng.choice(pool, (20000, nn)), 90.0)
        Ec = float(np.mean(np.cos(np.radians(2 * pool))))
        f_exp = np.clip((Vn + 1.645 / np.sqrt(2 * nn)) / (0.83 * Ec), 0, None)
        r[en] = dict(V_obs=float(Vobs), P_all_aligned=p_all, P_all_aligned_upper_if_zero=float(1 / ns) if p_all == 0 else None,
                     Ecos2=Ec, expected_f95=float(np.median(f_exp)), expected_band68=list(np.percentile(f_exp, [16, 84])),
                     f95=limit(pa_o, 0.83, 90.0, 90.0, pool_draw(pool))['f95'])
        print('bulge binaries', nm, en, r[en]['P_all_aligned'], r[en]['f95'], r[en]['expected_f95'], flush=True)
    bm[nm] = r
out['bulge_binaries'] = bm
json.dump(out, open(D / 'rev8_limits.json', 'w'), indent=1)
