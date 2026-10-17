"""Revision 4 analyses (referee round 4). Output: data/rev4.json

a. image source of the labels: orientation of the equatorial labels per source, in particular MASH (SHS based),
   and the cross validation error per source; suspect placeholder labels
b. errors of the catalogue axes of unlabelled PNe: external visual axes of PNe never used in training that pass
   the reliability criteria, and cross validation errors reweighted to the R_TTA distribution of the
   unlabelled catalogue
c. Holm correction over the pre-specified family of network alignment tests
d. known close binaries of the bipolar class only, and with P_B >= 0.7, with injection limits
e. injection limits for the bipolars outside the bulge region with more conservative error models
f. other bipolars of the bulge region by statistical distance
g. sky coverage lost by excluding SHS, and the IPHAS and VPHAS+ bipolars outside the bulge region separately
h. precision and recall of the bipolar class of the classification network
i. stellar PNe in VPHAS+ by consistency
"""
import json, re, pathlib
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(41)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    if n < 3:
        return dict(n=n)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


def inject(pa_obs, R_al, mu, err, nsim=3000, grid=np.round(np.arange(0, 1.001, 0.02), 2)):
    """Smallest aligned fraction for which the observed V statistic at mu is reached in fewer than 5 per cent."""
    pa_obs = np.asarray(pa_obs, float); n = len(pa_obs)
    Vobs = np.mean(np.cos(np.radians(2 * (pa_obs - mu)))); sig2 = np.sqrt(-2 * np.log(R_al))
    for f in grid:
        Vs = np.empty(nsim)
        for k in range(nsim):
            na = rng.binomial(n, f)
            pa = np.r_[mu + np.degrees(rng.normal(0, sig2, na)) / 2, rng.uniform(0, 180, n - na)]
            if err is not None:
                pa = pa + rng.choice(err, n)
            Vs[k] = np.mean(np.cos(np.radians(2 * (pa - mu))))
        if np.mean(Vs <= Vobs) < 0.05:
            return float(f)
    return None


x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
cat = pd.read_csv(D / 'catalogue_pa.csv')
prov = pd.read_csv(D / 'label_provenance.csv')
out = {}

# ---------- a. label sources ----------
cvp = cv.merge(prov[['idPNMain', 'label_source', 'suspect_zero']], on='idPNMain')
a = {}
for s in ['MASH', 'DSS2', 'CAM98', 'HST', 'other']:
    t = cvp[cvp.label_source == s]
    a[s] = dict(n=len(t), equatorial=circ(t.EPA, mu=113), surveys=t.survey.value_counts().to_dict(),
                cv_median=float(t.d_cnn.median()), cv_gt60=float((t.d_cnn > 60).mean()))
a['MASH, SHS stellar direction 113'] = circ(cvp[cvp.label_source == 'MASH'].EPA, mu=113)
sz = cvp[cvp.suspect_zero]
a['suspect_zero'] = dict(n=len(sz), names=sz.Name.tolist(), cv_median=float(sz.d_cnn.median()))
a['bipolar labels, network and moments differ by more than 45 deg'] = int(
    ((cvp.mainClass == 'B') & (dpa(cvp.pa_eq, cvp.EPA) > 45)).sum())
out['labels'] = a

# ---------- b. catalogue errors for unlabelled PNe ----------
h = x[['idPNMain', 'Name', 'png', 'mainClass', 'MajDiam', 'survey', 'pa', 'R_tta', 'trained']].copy()
h['nkey'] = h.Name.map(norm)
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(norm)
e_rz = h.merge(rz[['PNG', 'EPA_deg']], left_on='png', right_on='PNG')
b1 = h.merge(co[['nkey', 'EPA_deg']], on='nkey')
b2 = h.merge(co[['PNG_approx_from_lb', 'EPA_deg']], left_on='png', right_on='PNG_approx_from_lb')
e_co = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain')
rel = lambda t: t[t.mainClass.isin(['B', 'E', 'I', 'A']) & (t.MajDiam >= 6) & (t.R_tta >= 0.8) & ~t.trained]
b = {}
for name, t in [('Corradi98', e_co), ('RZ13', e_rz)]:
    r = rel(t); d = dpa(r.pa, r.EPA_deg)
    b[name] = dict(n=len(r), median=float(d.median()), gt60=float((d > 60).mean()), gt30=float((d > 30).mean()))
both = pd.concat([rel(e_co), rel(e_rz)]); d = dpa(both.pa, both.EPA_deg)
b['both'] = dict(n=len(both), median=float(d.median()), gt60=float((d > 60).mean()), gt30=float((d > 30).mean()))
nonshs = both[both.survey != 'SHS']; d = dpa(nonshs.pa, nonshs.EPA_deg)
b['both, not SHS'] = dict(n=len(nonshs), median=float(d.median()), gt60=float((d > 60).mean()))
# cross validation errors of reliable labelled PNe reweighted to the R_TTA bins of the unlabelled catalogue
bins = [0.8, 0.9, 0.95, 1.0001]
cvr = cv[cv.mainClass.isin(['B', 'E', 'I', 'A']) & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8)].copy()
cvr['bin'] = pd.cut(cvr.cnn_R, bins, right=False)
un = cat[(cat.flag_reliable == 1) & (cat.flag_out_of_fold == 0)].copy()
un['bin'] = pd.cut(un.R_TTA, bins, right=False)
wl = cvr.bin.value_counts(normalize=True).sort_index(); wu = un.bin.value_counts(normalize=True).sort_index()
w = cvr.bin.map(wu / wl).astype(float).to_numpy().copy(); w /= w.sum()
dd = cvr.d_cnn.values; o = np.argsort(dd); cw = np.cumsum(w[o])
b['rtta_bins'] = dict(labelled=wl.round(3).tolist(), unlabelled=wu.round(3).tolist())
b['cv_reliable'] = dict(n=len(cvr), median=float(np.median(dd)), gt60=float((dd > 60).mean()))
b['cv_reweighted'] = dict(median=float(dd[o][np.searchsorted(cw, 0.5)]), gt60=float(w[dd > 60].sum()))
out['catalogue_errors'] = b

# ---------- c. Holm correction over the pre-specified network tests ----------
base = (x.MajDiam >= 6) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B'); E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['GPA_deg'])
tsel = tan[(tan.MajDiam >= 6) & (tan.R_tta >= 0.8) & (tan.survey != 'SHS')]
bj = pd.read_csv(D / 'supp_binaries.csv'); bsel = bj[bj.in_alignment_test]
fam = {
    'RZ13 selected bipolars, network': circ(x[(x.morph_class == 'bipolar') & base].gpa)['V_p'],
    'Tan et al. all, network, at 107.5': circ(tan.gpa, 107.5)['V_p'],
    'Tan et al. selected, network, at 107.5': circ(tsel.gpa, 107.5)['V_p'],
    'bulge region other bipolars': circ(x[B & x.bulge_sky & x.morph_class.isna()].gpa)['V_p'],
    'bulge region ellipticals': circ(x[E & x.bulge_sky].gpa)['V_p'],
    'outside bulge region bipolars': circ(x[B & ~x.bulge_sky].gpa)['V_p'],
    'outside bulge region ellipticals': circ(x[E & ~x.bulge_sky].gpa)['V_p'],
    'known binaries outside bulge region': circ(bsel[~bsel.bulge_sky].gpa)['V_p'],
    'known binaries bulge region': circ(bsel[bsel.bulge_sky].gpa)['V_p'],
}
k = len(fam); ps = sorted(fam.items(), key=lambda kv: kv[1]); holm = {}; run = 0
for i, (n_, p) in enumerate(ps):
    run = max(run, min(1, (k - i) * p)); holm[n_] = dict(p=float(p), p_holm=float(run))
out['holm'] = dict(k=k, tests=holm, min_p_holm=min(v['p_holm'] for v in holm.values()))

# ---------- d. known binaries of the bipolar class ----------
bj2 = bsel.merge(x[['idPNMain', 'p_B']], on='idPNMain', how='left')
cvres = wrap(cv[(cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')].cnn_pa
             - cv[(cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')].EPA).values
dsel = {}
for reg, rm in [('outside', ~bj2.bulge_sky), ('bulge', bj2.bulge_sky)]:
    for nm, cm in [('HASH bipolar', bj2.mainClass == 'B'), ('P_B >= 0.7', bj2.p_B >= 0.7),
                   ('HASH bipolar, confirmed', (bj2.mainClass == 'B') & ~bj2.candidate)]:
        s = bj2[rm & cm]
        r = circ(s.gpa)
        if len(s) >= 5:
            r['f95_tight_cv'] = inject(s.gpa, 0.83, 107.5, cvres)
        dsel[f'{reg}, {nm}'] = r
out['binaries_bipolar'] = dsel

# ---------- e. conservative injection for the bipolars outside the bulge region ----------
ob = x[B & ~x.bulge_sky].gpa.values
co_new = rel(e_co); co_new = co_new[co_new.survey != 'SHS']
err_ext = wrap(co_new.pa - co_new.EPA_deg).values
# cross validation residuals resampled with the R_TTA weights of the unlabelled catalogue
# (bipolars outside SHS, weighted to the unlabelled reliable bipolars outside SHS)
cvb = cvr[(cvr.survey != 'SHS') & (cvr.mainClass == 'B')]
ub = un[(un.HASH_class == 'B') & (un.flag_SHS == 0)]
wub = ub.bin.value_counts(normalize=True).sort_index(); wlb = cvb.bin.value_counts(normalize=True).sort_index()
wb = cvb.bin.map(wub / wlb).astype(float).to_numpy().copy(); wb /= wb.sum()
err_rw = wrap(cvb.cnn_pa - cvb.EPA).values[rng.choice(len(cvb), 20000, p=wb)]
# cross validation residuals with the fraction of gross failures about doubled: a further fraction gt of the
# errors is drawn from the residuals beyond 60 deg
gt = (np.abs(cvres) > 60).mean()
err_fail = np.where(rng.random(20000) < gt, rng.choice(cvres[np.abs(cvres) > 60], 20000), rng.choice(cvres, 20000))
e = {'n': len(ob), 'ext_errors_n': len(err_ext), 'ext_errors_gt60': float((np.abs(err_ext) > 60).mean()),
     'cv_gt60': float(gt)}
for nm, err in [('cv', cvres), ('external Corradi', err_ext), ('reweighted cv', err_rw), ('doubled failures', err_fail)]:
    e[nm] = dict(tight=inject(ob, 0.83, 107.5, err), visual_RZ13=inject(ob, 0.29, 90.0, err),
                 Ecos2=float(np.mean(np.cos(np.radians(2 * err)))), gt60=float((np.abs(err) > 60).mean()))
out['injection_conservative'] = e

# ---------- f. other bipolars of the bulge region by distance ----------
ob_b = x[B & x.bulge_sky & x.morph_class.isna()]
out['bulge_distance'] = {'beyond 5 kpc': circ(ob_b[ob_b.rgeo_kpc > 5].gpa),
                         '6 to 10 kpc': circ(ob_b[(ob_b.rgeo_kpc > 6) & (ob_b.rgeo_kpc < 10)].gpa),
                         'bulge component': circ(ob_b[ob_b.Component == 'Bulge'].gpa),
                         'below 35 arcsec and beyond 5 kpc': circ(ob_b[(ob_b.rgeo_kpc > 5) & (ob_b.MajDiam < 35)].gpa)}

# ---------- g. sky coverage lost to the SHS exclusion ----------
bb = (x.MajDiam >= 6) & (x.R_tta >= 0.8) & (x.mainClass == 'B') & ~x.bulge_sky
g = dict(outside_B_all_surveys=int(bb.sum()), outside_B_SHS=int((bb & (x.survey == 'SHS')).sum()))
g['SHS_median_abs_b'] = float(x[bb & (x.survey == 'SHS')].Glat.abs().median())
g['kept_median_abs_b'] = float(x[bb & (x.survey != 'SHS')].Glat.abs().median())
g['kept_abs_b_q98'] = float(x[bb & (x.survey != 'SHS')].Glat.abs().quantile(0.98))
g['SHS_glon_range'] = [float(x[bb & (x.survey == 'SHS')].Glon.min()), float(x[bb & (x.survey == 'SHS')].Glon.max())]
for sv in ['IPHAS', 'VPHAS']:
    g[sv] = circ(x[B & ~x.bulge_sky & (x.survey == sv)].gpa)
lon = ((x.Glon + 180) % 360) - 180
g['VPHAS fourth quadrant'] = circ(x[B & ~x.bulge_sky & (x.survey == 'VPHAS') & (lon < 0)].gpa)
g['inner disc |l| < 30'] = circ(x[B & ~x.bulge_sky & (np.abs(lon) < 30)].gpa)
out['coverage'] = g

# ---------- h. bipolar class of the classification network ----------
mc = pd.read_csv(D / 'morph_cv.csv')
tp = ((mc.cnn_class == 'B') & (mc.mainClass == 'B')).sum()
out['bipolar_class'] = dict(precision=float(tp / (mc.cnn_class == 'B').sum()), recall=float(tp / (mc.mainClass == 'B').sum()))
for thr in [0.5, 0.7]:
    s = mc.p_B >= thr
    out['bipolar_class'][f'P_B>={thr}'] = dict(precision=float((s & (mc.mainClass == 'B')).sum() / s.sum()),
                                                recall=float((s & (mc.mainClass == 'B')).sum() / (mc.mainClass == 'B').sum()))

# ---------- i. stellar PNe in VPHAS+ ----------
st = x[(x.mainClass == 'S') & (x.survey == 'VPHAS') & x.MajDiam.notna()]
out['stellar_vphas'] = {'all': circ(st.pa, 143), 'R_TTA >= 0.8': circ(st[st.R_tta >= 0.8].pa, 143),
                        'R_TTA < 0.8': circ(st[st.R_tta < 0.8].pa, 143),
                        'median R_TTA': float(st.R_tta.median()), 'median MajDiam': float(st.MajDiam.median())}

json.dump(out, open(D / 'rev4.json', 'w'), indent=1, default=float)
for kk, v in out.items():
    print(kk, json.dumps(v, default=float)[:1500])
