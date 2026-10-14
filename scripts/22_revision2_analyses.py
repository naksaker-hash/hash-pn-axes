"""Revision 2 analyses (referee round 2). Output: data/rev2.json, data/supp_binaries.csv

a. injection limits for the known close binary samples (Jones compilation) and the expected signal
b. close binary fraction among bipolar PNe (Bayes, Miszalski et al. 2009a,b, HASH bipolar share)
c. injection limits with the visual concentration of the selected RZ13 bipolars, at 90 deg, with and without errors,
   for the PNe outside the bulge region and for the other bulge region bipolars; bulge samples also with the
   external RZ13 residuals as the error model
d. bulge membership: other bulge region bipolars within the RZ13 size limit (<= 35 arcsec) and those in the
   bulge component of Erzincan et al.
e. R_TTA: Spearman correlation and ROC AUC for gross failures; alignment at R_TTA >= 0.9 and 0.95
f. Tan et al. sample with the stated selection, V tests at 107.5 and 90 deg
g. classification comparison at the decision level (balanced accuracy, purity and recall at P(B) >= 0.7)
h. stellar PNe by survey: mean direction and Rayleigh p, and Galactic axes of VPHAS+ stellar PNe in the bulge region
i. finer size bins of the CV error; j. reliable axes without any published position angle
"""
import json, re, pathlib
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(23)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
out = {}
x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
rv = json.load(open(D / 'rev1.json'))
base = (x.MajDiam >= 6) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B')


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    if n < 3:
        return dict(n=n)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


def inject(n, Vobs, R_al, mu, err, grid=np.round(np.arange(0, 1.01, 0.02), 2), nsim=3000, mu_test=90):
    sig2 = np.sqrt(-2 * np.log(R_al)); res = {}
    for f in grid:
        Vs = []
        for _ in range(nsim):
            na = rng.binomial(n, f)
            pa = np.r_[mu + np.degrees(rng.normal(0, sig2, na)) / 2, rng.uniform(0, 180, n - na)]
            if err is not None:
                pa = pa + rng.choice(err, n)
            Vs.append(np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu_test))))
        res[float(f)] = float(np.mean(np.array(Vs) <= Vobs))
    f95 = next((f for f in sorted(res) if res[f] < 0.05), None)
    return dict(f95=f95, P_le_obs=res)


mres = (cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
resid_cv = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
# external residuals: never trained RZ13 objects (network vs RZ13 visual), all classes, >= 4 arcsec
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png')
xr = xr[~xr.trained & (xr.MajDiam >= 4)]
resid_rz = wrap(xr.pa - xr.EPA_deg).dropna().values
out['resid'] = dict(cv_n=len(resid_cv), cv_Ecos2=float(np.mean(np.cos(np.radians(2 * resid_cv)))),
                    rz_n=len(resid_rz), rz_Ecos2=float(np.mean(np.cos(np.radians(2 * resid_rz)))))
R_vis_sel = rv['samples']['RZ13 bipolar, visual, same objects as cuts']['R']

# a. known binaries
bj = pd.read_csv(ROOT / 'refs' / 'binaries' / 'bcspn_jones.csv').dropna(subset=['idPNMain'])
bj['idPNMain'] = bj.idPNMain.astype(int)
xb = x.merge(bj[['idPNMain', 'name', 'candidate', 'period_d']], on='idPNMain')
sel_b = (xb.MajDiam >= 6) & (xb.R_tta >= 0.8) & (xb.survey != 'SHS') & xb.mainClass.isin(['B', 'E', 'I', 'A'])
xb[['idPNMain', 'Name', 'name', 'PNG', 'candidate', 'period_d', 'mainClass', 'survey', 'MajDiam', 'pa', 'gpa', 'R_tta', 'bulge_sky']].assign(
    in_alignment_test=sel_b.values).to_csv(D / 'supp_binaries.csv', index=False)
out['binaries'] = {}
for name, m_ in [('outside bulge region', sel_b & ~xb.bulge_sky), ('bulge region', sel_b & xb.bulge_sky)]:
    s = xb[m_]; c_ = circ(s.gpa, 107.5); c90 = circ(s.gpa, 90)
    out['binaries'][name] = dict(stats_107=c_, stats_90=c90,
                                 tight_errors=inject(len(s), c_['V'], 0.83, 107.5, resid_cv, mu_test=107.5)['f95'],
                                 tight_noerr=inject(len(s), c_['V'], 0.83, 107.5, None, mu_test=107.5)['f95'],
                                 expected_V_all_aligned=float(0.83 * out['resid']['cv_Ecos2']),
                                 expected_u_all_aligned=float(0.83 * out['resid']['cv_Ecos2'] * np.sqrt(2 * len(s))))

# b. close binary fraction among bipolars
fB = float((x.mainClass == 'B').mean())
out['binary_fraction_bipolars'] = dict(P_bin=[0.12, 0.21], P_B_given_bin=[0.30, 0.60], P_B_HASH=fB,
                                       P_bin_given_B=[0.12 * 0.30 / fB, 0.21 * 0.60 / fB])

# c. weaker (visual RZ13 selected) concentration, at 90 deg
out['injection_visual_R'] = {}
for sname, m_, er_name, er in [('outside bulge region B', B & ~x.bulge_sky, 'cv', resid_cv),
                               ('bulge region B not in RZ13', B & x.bulge_sky & x.morph_class.isna(), 'cv', resid_cv),
                               ('bulge region B not in RZ13', B & x.bulge_sky & x.morph_class.isna(), 'rz', resid_rz)]:
    s = x[m_]; Vobs = circ(s.gpa)['V']
    out['injection_visual_R'][f'{sname} | {er_name} errors'] = dict(n=len(s), R=R_vis_sel,
        f95_errors=inject(len(s), Vobs, R_vis_sel, 90, er)['f95'], f95_noerr=inject(len(s), Vobs, R_vis_sel, 90, None)['f95'])
s = x[B & x.bulge_sky & x.morph_class.isna()]
out['injection_tight_bulge_rz_errors'] = inject(len(s), circ(s.gpa, 107.5)['V'], 0.83, 107.5, resid_rz, mu_test=107.5)['f95']
s = x[B & ~x.bulge_sky]
out['injection_tight_outside_at90'] = inject(len(s), circ(s.gpa)['V'], 0.83, 90, resid_cv)['f95']

# d. bulge membership
ob = B & x.bulge_sky & x.morph_class.isna()
out['bulge_membership'] = {'other B, MajDiam <= 35': circ(x[ob & (x.MajDiam <= 35)].gpa),
                           'other B, Erzincan bulge component': circ(x[ob & (x.Component == 'Bulge')].gpa),
                           'all bulge region B, Erzincan bulge component': circ(x[B & x.bulge_sky & (x.Component == 'Bulge')].gpa),
                           'other B, distance > 5 kpc': circ(x[ob & (x.rgeo_kpc > 5)].gpa)}

# e. R_TTA
cvd = cv.assign(d=dpa(cv.cnn_pa, cv.EPA))
out['rtta'] = dict(spearman=float(stats.spearmanr(1 - cvd.cnn_R, cvd.d).correlation),
                   auc_gross=float(roc_auc_score((cvd.d > 60).astype(int), 1 - cvd.cnn_R)),
                   pass08_labelled=float((x[x.trained & (x.MajDiam >= 6)].R_tta >= 0.8).mean()),
                   pass08_unlabelled=float((x[~x.trained & (x.MajDiam >= 6)].R_tta >= 0.8).mean()))
for thr in [0.9, 0.95]:
    bb = (x.MajDiam >= 6) & (x.R_tta >= thr) & (x.survey != 'SHS') & (x.mainClass == 'B')
    out['rtta'][f'outside B, R_tta >= {thr}'] = circ(x[bb & ~x.bulge_sky].gpa)
    out['rtta'][f'bulge other B, R_tta >= {thr}'] = circ(x[bb & x.bulge_sky & x.morph_class.isna()].gpa)

# f. Tan with the stated selection
tan = pd.read_csv(D / 'tan_comparison.csv')
tan = tan[tan.GPA_deg.notna()]
ts = tan[(tan.MajDiam >= 6) & (tan.R_tta >= 0.8)]
out['tan_selected'] = dict(n=len(ts), at107=circ(ts.gpa, 107.5), at90=circ(ts.gpa, 90), visual_at107=circ(ts.GPA_deg, 107.5))

# g. classification at the decision level
mc = pd.read_csv(D / 'morph_cv.csv')
m0 = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain'); m0['idPNMain'] = m0.idPNMain.astype(int)
m0['png'] = 'PN G' + m0.PNG; m0['nkey'] = m0.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
mc = mc.merge(m0[['idPNMain', 'png', 'nkey']], on='idPNMain')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
g = {}
for name, j, truth in [('Corradi98', mc.merge(co[['nkey', 'class']], on='nkey'), lambda j: j['class'].str.startswith('B')),
                       ('RZ13', mc.merge(rz[['PNG', 'morph_class']], left_on='png', right_on='PNG'), lambda j: j.morph_class == 'bipolar')]:
    y = truth(j).values
    for lab, pred in [('HASH B', (j.mainClass == 'B').values), ('network B', (j.cnn_class == 'B').values),
                      ('network P(B) >= 0.7', (j.p_B >= 0.7).values), ('HASH B and P(B) >= 0.7', ((j.mainClass == 'B') & (j.p_B >= 0.7)).values)]:
        tpr = float((pred & y).sum() / max(y.sum(), 1)); tnr = float((~pred & ~y).sum() / max((~y).sum(), 1))
        g[f'{name} | {lab}'] = dict(n=len(j), n_true=int(y.sum()), balanced_acc=(tpr + tnr) / 2, recall=tpr,
                                    purity=float((pred & y).sum() / max(pred.sum(), 1)))
    hb = j[j.mainClass == 'B']; yb = truth(hb).values
    if 0 < yb.sum() < len(yb):
        g[f'{name} | AUC of P(B) within HASH B'] = dict(n=len(hb), auc=float(roc_auc_score(yb, hb.p_B)))
    if name == 'Corradi98':
        g['Corradi98 | bipolars not B in HASH'] = dict(n=int(y.sum()), frac=float(((j.mainClass != 'B') & y).sum() / y.sum()))
out['class_decision'] = g

# h. stellar PNe by survey, and Galactic axes of VPHAS+ stellar PNe in the bulge region
st = {}
for sv in ['IPHAS', 'VPHAS', 'SHS']:
    s = x[(x.mainClass == 'S') & (x.survey == sv) & x.MajDiam.notna()]
    st[sv] = circ(s.pa)
st['VPHAS bulge region, Galactic'] = circ(x[(x.mainClass == 'S') & (x.survey == 'VPHAS') & x.bulge_sky].gpa)
st['VPHAS outside bulge, Galactic'] = circ(x[(x.mainClass == 'S') & (x.survey == 'VPHAS') & ~x.bulge_sky].gpa)
out['stellar'] = st

# i. finer size bins
out['cv_by_size_fine'] = {f'{lo}-{hi}': dict(n=int(((cvd.MajDiam >= lo) & (cvd.MajDiam < hi)).sum()),
                                            median=float(cvd[(cvd.MajDiam >= lo) & (cvd.MajDiam < hi)].d.median()))
                          for lo, hi in [(4, 6), (6, 10), (10, 20), (20, 40), (40, 1e4)]}

# j. reliable axes without any published position angle
cat = pd.read_csv(D / 'catalogue_pa.csv')
rel = cat[cat.flag_reliable == 1]
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv'); w['png'] = w.PNG_from_name.fillna(w.PNG_approx_from_lb)
lit = set('PN G' + rel.PNG) & (set(rz.PNG) | set(w.png) | set(co.PNG_approx_from_lb) | set(tan.PNG))
names_c = set(co.nkey)
relk = rel.assign(png='PN G' + rel.PNG, nkey=rel.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())))
haslit = relk.HASH_PA.notna() | relk.png.isin(lit) | relk.nkey.isin(names_c)
out['catalogue_new'] = dict(reliable=len(rel), reliable_not_shs=int((rel.flag_SHS == 0).sum()),
                            no_hash_pa=int(rel.HASH_PA.isna().sum()), no_published_pa=int((~haslit).sum()),
                            no_published_pa_not_shs=int((~haslit & (relk.flag_SHS == 0)).sum()))
json.dump(out, open(D / 'rev2.json', 'w'), indent=1, default=float)
print(json.dumps(out, indent=1, default=float)[:6000])
