"""Revision 7 analyses (second deep review). Output: data/rev7.json, data/binaries_clean.csv

a. clean close binary sample: only the main table of the compilation of D. Jones (Galactic close binaries), with a
   numeric orbital period of at most 5 d; confirmatory V test at the plane, B only subsets, expected signals and
   injection limits at the plane and at 107.5 deg
b. limits for the 266 bipolars outside the bulge region with the confirmatory direction (90 deg), interpolated, with
   Monte Carlo error, expected limit and its band, and the propagated uncertainty with the bootstrap spread of rho
c. 3D injections with filtering: intrinsic axes drawn for all PNe, those within the visibility cut of the line of
   sight removed from the bipolar class, cuts of 30, 45 and 60 deg, field, pole and girdle models, interpolated limits
d. consistency measures on a common axis set with paired bootstrap, per seed AUC and failures passing the cut
e. von Mises head: coverage at 50, 68, 90 and 95 per cent and held out likelihood against a constant kappa
f. gains against the best moment baseline with the five seed mean error and a block bootstrap
g. reproductions: attenuation of R predicted by the error model; classification purity with Wilson intervals
h. PSF of field stars in Halpha, r and the continuum subtracted image, and the Halpha to r centroid offset, per survey
"""
import json, pathlib, re, html as htmlmod
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import i0e
from sklearn.metrics import roc_auc_score
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(71)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
out = {}


def circ(pa, mu=90.0):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    Rc = np.sqrt(max((n * R ** 2 - 1) / (n - 1), 0)) if n > 1 else np.nan
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R), R_bias_corrected=float(Rc),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu)), axis=-1)
x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
err = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
Ec = float(np.mean(np.cos(np.radians(2 * err))))
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
ob = x[base & (x.mainClass == 'B') & ~x.bulge_sky]
gpa = ob.gpa.values; n = len(gpa)


def Pgrid(pa_obs, R_al, mu_inj, mu_test, fgrid, nsim, e=err):
    nn = len(pa_obs); Vobs = Vat(pa_obs, mu_test); sig2 = np.sqrt(-2 * np.log(R_al)); P = []
    for f in fgrid:
        al = rng.random((nsim, nn)) < f
        pa = np.where(al, mu_inj + np.degrees(rng.normal(0, sig2, (nsim, nn))) / 2, rng.uniform(0, 180, (nsim, nn)))
        pa = pa + rng.choice(e, (nsim, nn))
        P.append(np.mean(Vat(pa, mu_test) <= Vobs))
    return np.array(P), float(Vobs)


def crossing(fgrid, P, level=0.05):
    i = np.flatnonzero(P < level)
    if not len(i):
        return None
    if i[0] == 0:
        return float(fgrid[0])
    j = i[0]; return float(fgrid[j - 1] + (P[j - 1] - level) * (fgrid[j] - fgrid[j - 1]) / (P[j - 1] - P[j]))


# ---------- a. clean close binaries ----------
h = open(ROOT / 'refs' / 'binaries' / 'bcspn.html', errors='ignore').read()
tab0 = re.findall(r'<table.*?</table>', h, re.S)[0]
rows = []
for r in re.findall(r'<tr[^>]*>(.*?)</tr>', tab0, re.S):
    c = [htmlmod.unescape(re.sub('<[^>]+>', '', x_)).strip() for x_ in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)]
    if len(c) >= 4 and re.match(r'\d+:\d+:\d', c[1] or ''):
        pstr = c[3]; mnum = re.search(r'\d+(\.\d+)?', pstr)
        rows.append(dict(name=c[0].lstrip('*').strip(), candidate=c[0].startswith('*'), ra=c[1], dec=c[2], period_str=pstr,
                         period_d=float(mnum.group(0)) if mnum else np.nan))
bt = pd.DataFrame(rows).drop_duplicates(['ra', 'dec'])
bc = SkyCoord(bt.ra.values, bt.dec.values, unit=(u.hourangle, u.deg))
m = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
hc = SkyCoord(pd.to_numeric(m.DRAJ2000).values * u.deg, pd.to_numeric(m.DDECJ2000).values * u.deg)
ii, dd, _ = bc.match_to_catalog_sky(hc)
bt['idPNMain'] = np.where(dd.arcsec < 15, m.idPNMain.values[ii].astype(float), np.nan)
bt['short'] = bt.period_d <= 5
xb = x.merge(bt.dropna(subset=['idPNMain']).assign(idPNMain=lambda d_: d_.idPNMain.astype(int))[['idPNMain', 'name', 'candidate', 'period_str', 'period_d', 'short']], on='idPNMain')
sel = (xb.MajDiam >= CUT) & (xb.R_tta >= 0.8) & (xb.survey != 'SHS') & xb.mainClass.isin(['B', 'E', 'I', 'A']) & xb.short
xb.assign(in_alignment_test=sel.values)[['idPNMain', 'Name', 'name', 'PNG', 'candidate', 'period_str', 'period_d', 'short', 'mainClass', 'survey',
                                          'MajDiam', 'pa', 'gpa', 'R_tta', 'bulge_sky', 'in_alignment_test']].to_csv(D / 'binaries_clean.csv', index=False)
old = pd.read_csv(D / 'supp_binaries.csv'); old = old[old.in_alignment_test]
a = dict(main_table_rows=len(bt), main_table_matched=int(bt.idPNMain.notna().sum()), main_table_short=int(bt.short.sum()),
         old_in_test=int(len(old)), old_outside=int((~old.bulge_sky).sum()))
fg = np.round(np.arange(0.02, 1.001, 0.02), 2)
for nm, mk in [('outside', sel & ~xb.bulge_sky), ('bulge', sel & xb.bulge_sky)]:
    s = xb[mk]; sB = s[s.mainClass == 'B']
    r = dict(n=len(s), n_candidates=int(s.candidate.sum()), n_B=len(sB), n_E=int((s.mainClass == 'E').sum()),
             at90=circ(s.gpa), at107=circ(s.gpa, 107.5), B_at90=circ(sB.gpa) if len(sB) > 1 else None,
             expected_sigma_90=float(0.83 * Ec * np.sqrt(2 * len(s)) * np.cos(np.radians(2 * 17.5))),
             expected_sigma_107=float(0.83 * Ec * np.sqrt(2 * len(s))))
    for lab, mi, mt in [('tight_107_107', 107.5, 107.5), ('tight_90_90', 90.0, 90.0), ('tight_107_90', 107.5, 90.0)]:
        P, _ = Pgrid(s.gpa.values, 0.83, mi, mt, fg, 3000); r[f'f95_{lab}'] = crossing(fg, P)
        if len(sB) > 2:
            P, _ = Pgrid(sB.gpa.values, 0.83, mi, mt, fg, 3000); r[f'f95_B_{lab}'] = crossing(fg, P)
    r['names'] = s.Name.tolist()
    a[nm] = r
# family of two with the clean sample
p1 = circ(gpa)['V_p']; p2 = a['outside']['at90']['V_p']; ps = sorted([p1, p2])
a['holm'] = dict(p_bipolars=p1, p_binaries=p2, holm=[min(1, 2 * ps[0]), max(min(1, 2 * ps[0]), min(1, ps[1]))])
out['binaries'] = a

# ---------- b. limits for the 266 bipolars ----------
lim = {}
fgf = np.round(np.arange(0.06, 0.30, 0.01), 2)
for lab, mi, mt in [('90_90', 90.0, 90.0), ('107_90', 107.5, 90.0), ('107_107', 107.5, 107.5)]:
    P, Vobs = Pgrid(gpa, 0.83, mi, mt, fgf, 10000)
    c0 = crossing(fgf, P); k = np.argmin(np.abs(fgf - c0)); slope = np.gradient(P, fgf)[k]
    lim[lab] = dict(V_obs=Vobs, crossing=c0, mc_se=float(np.sqrt(0.05 * 0.95 / 10000) / abs(slope)))
# expected limit and its band under the null, at 90/90 (analytic inversion, which matches the simulation)
Vn = Vat(rng.uniform(0, 180, (20000, n)) + rng.choice(err, (20000, n)), 90.0)
f_exp = (Vn + 1.645 / np.sqrt(2 * n)) / (0.83 * Ec)
lim['expected_90'] = dict(median=float(np.median(f_exp)), band68=list(np.percentile(f_exp, [16, 84])))
Vn107 = Vat(rng.uniform(0, 180, (20000, n)) + rng.choice(err, (20000, n)), 107.5)
f_exp107 = (Vn107 + 1.645 / np.sqrt(2 * n)) / (0.83 * Ec)
lim['expected_107'] = dict(median=float(np.median(f_exp107)), band68=list(np.percentile(f_exp107, [16, 84])))
# rho spread from the bootstrap of the 14 visual axes of Tan et al.
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg'])
tv = tan.GPA_deg.values
Rb = [np.abs(np.mean(np.exp(2j * np.radians(tv[rng.integers(0, len(tv), len(tv))])))) for _ in range(4000)]
rho_sd = float(np.std(Rb))
for lab, mt, cosfac in [('90', 90.0, 1.0), ('107', 107.5, 1.0)]:
    Vobs = Vat(gpa, mt); fs = []
    for _ in range(4000):
        eb = rng.choice(err, len(err)); rho = np.clip(rng.normal(0.83, rho_sd), 0.4, 0.99)
        fs.append((Vobs + 1.645 / np.sqrt(2 * n)) / (rho * np.mean(np.cos(np.radians(2 * eb)))))
    lim[f'propagated_{lab}'] = dict(rho_sd=rho_sd, median=float(np.median(fs)), band68=list(np.percentile(fs, [16, 84])), band95=list(np.percentile(fs, [2.5, 97.5])))
lim['counts'] = dict(n=n, at_90=float(lim['90_90']['crossing'] * n), conservative=float(0.24 * n))
out['limits'] = lim

# ---------- c. 3D injections with filtering ----------
R0 = 8.2
l = np.radians(ob.Glon.values); b = np.radians(ob.Glat.values)
dist_known = ob.rgeo_kpc.values
uvec = np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], 1)
nhat = np.stack([-np.sin(b) * np.cos(l), -np.sin(b) * np.sin(l), np.cos(b)], 1)
ehat = np.stack([-np.sin(l), np.cos(l), np.zeros_like(l)], 1)
known = np.isfinite(dist_known); logd = np.log(dist_known[known])


def field_dir(dist, pitch=np.radians(11)):
    P3 = np.array([-R0, 0, 0]) + dist[:, None] * uvec
    ph = np.stack([-P3[:, 1], P3[:, 0], np.zeros(n)], 1); ph /= np.linalg.norm(ph, axis=1, keepdims=True)
    rh = np.stack([P3[:, 0], P3[:, 1], np.zeros(n)], 1); rh /= np.linalg.norm(rh, axis=1, keepdims=True)
    return np.cos(pitch) * ph + np.sin(pitch) * rh


def fisher(mu, kappa):
    k = len(mu); xi = rng.random(k); w = 1 + np.log(xi + (1 - xi) * np.exp(-2 * kappa)) / kappa
    ph = rng.uniform(0, 2 * np.pi, k); s = np.sqrt(np.clip(1 - w ** 2, 0, None))
    t1 = np.where((np.abs(mu[:, 0]) < 0.9)[:, None], np.cross(mu, [1, 0, 0]), np.cross(mu, [0, 1, 0])); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(mu, t1)
    return w[:, None] * mu + s[:, None] * (np.cos(ph)[:, None] * t1 + np.sin(ph)[:, None] * t2)


def girdle(kappa_b, k):
    """Axes in the Galactic plane with random azimuth, latitude spread like a Fisher distribution about the plane."""
    az = rng.uniform(0, 2 * np.pi, k)
    lat = np.arcsin(np.clip(fisher(np.tile([1.0, 0, 0], (k, 1)), kappa_b)[:, 2], -1, 1))
    return np.stack([np.cos(lat) * np.cos(az), np.cos(lat) * np.sin(az), np.sin(lat)], 1)


def realisation(model, kappa, f, cut):
    """One sample of n observed bipolars: intrinsic axes are drawn for candidates at the observed positions, each is
    aligned with probability f, and candidates whose axis lies within the cut of the line of sight are discarded
    (they would not be classed as bipolar) until n visible axes are collected. Returns the GPAs and the visible
    aligned fraction."""
    dist = np.where(known, dist_known, np.exp(rng.choice(logd, n)))  # missing distances drawn from the known ones
    fd = field_dir(dist) if model == 'field' else None
    pa = np.full(n, np.nan); alv = np.zeros(n, bool); todo = np.ones(n, bool); cosc = np.cos(np.radians(cut))
    while todo.any():
        k = np.flatnonzero(todo); al = rng.random(len(k)) < f
        iso = rng.normal(size=(len(k), 3)); iso /= np.linalg.norm(iso, axis=1, keepdims=True)
        if model == 'field':
            ali = fisher(fd[k], kappa)
        elif model == 'pole':
            ali = fisher(np.tile([0, 0, 1.0], (len(k), 1)), kappa)
        else:
            ali = girdle(kappa, len(k))
        ax = np.where(al[:, None], ali, iso)
        vis = np.abs(np.sum(ax * uvec[k], 1)) < cosc  # angle to the line of sight larger than the cut
        kk = k[vis]; axv = ax[vis]
        pa[kk] = np.degrees(np.arctan2(np.sum(axv * ehat[kk], 1), np.sum(axv * nhat[kk], 1))) % 180
        alv[kk] = al[vis]; todo[kk] = False
    return pa, alv.mean()


V90obs = Vat(gpa, 90.0); Robs = circ(gpa)['R']
g3 = {}
import os
if os.path.exists(D / 'rev7_inject3d.json') and os.environ.get('REDO3D') != '1':
    g3 = json.load(open(D / 'rev7_inject3d.json'))
fgrid3 = np.round(np.arange(0, 1.001, 0.05), 2)
for model, kappa in ([] if g3 else [('field', 20.0), ('field', 5.0), ('pole', 20.0), ('pole', 5.0), ('girdle', 20.0), ('girdle', 5.0)]):
    for cut in (30, 45, 60):
        PV, PR, fvis = [], [], []
        for f in fgrid3:
            Vs, Rs, fv = [], [], []
            for _ in range(1500):
                pa, fa = realisation(model, kappa, f, cut)
                pa = pa + rng.choice(err, n); th = np.radians(2 * pa)
                Vs.append(np.mean(np.cos(th - np.pi))); Rs.append(np.abs(np.mean(np.exp(1j * th)))); fv.append(fa)
            Vs, Rs = np.array(Vs), np.array(Rs)
            st, ob_ = (Vs, V90obs) if model in ('field', 'girdle') else (-Vs, -V90obs)
            PV.append(np.mean(st <= ob_)); PR.append(np.mean(Rs <= Robs)); fvis.append(float(np.mean(fv)))
        PV, PR = np.array(PV), np.array(PR)
        cV, cR = crossing(fgrid3, PV), crossing(fgrid3, PR)
        vis_at = (lambda c: float(np.interp(c, fgrid3, fvis)) if c is not None else None)
        g3[f'{model} kappa {kappa:g} cut {cut}'] = dict(f95_V=cV, f95_R=cR, visible_aligned_at_f95_V=vis_at(cV), visible_aligned_at_f95_R=vis_at(cR),
                                                       P_V=PV.tolist(), P_R=PR.tolist(), f_visible=fvis)
        print(model, kappa, cut, cV, cR, vis_at(cV), flush=True)
out['inject3d'] = g3

# ---------- d. consistency measures on a common axis set ----------
seeds = [cv] + [pd.read_csv(D / f'pa_cnn_cv_s{s}.csv') for s in (1, 2, 3, 4)]
ids = cv.idPNMain.values
al = [s_.set_index('idPNMain').reindex(ids) for s_ in seeds]
Vz = np.mean([np.exp(2j * np.radians(s_.cnn_pa.values)) for s_ in al], 0)
ens_pa = (np.degrees(np.angle(Vz)) / 2) % 180; agree = np.abs(Vz)
meanR = np.mean([s_.cnn_R.values for s_ in al], 0)
d_ens = dpa(ens_pa, cv.EPA.values); g_ens = d_ens > 60
per_seed = []
for s_ in al:
    dd_ = dpa(s_.cnn_pa.values, cv.EPA.values); gg = dd_ > 60; hi = s_.cnn_R.values >= 0.8
    per_seed.append(dict(auc=float(roc_auc_score(gg, 1 - s_.cnn_R.values)), gross=int(gg.sum()), gross_pass=int((gg & hi).sum()),
                         rate_above=float(gg[hi].mean())))
cons = dict(per_seed=per_seed, auc_mean=float(np.mean([p['auc'] for p in per_seed])),
            rate_above_range=[float(min(p['rate_above'] for p in per_seed)), float(max(p['rate_above'] for p in per_seed))])
# common axis set: the ensemble mean axis, ranked by mean R_TTA and by the agreement of the seeds
a1 = roc_auc_score(g_ens, 1 - meanR); a2 = roc_auc_score(g_ens, 1 - agree); diffs = []
for _ in range(4000):
    k = rng.integers(0, len(ids), len(ids))
    if g_ens[k].any() and (~g_ens[k]).any():
        diffs.append(roc_auc_score(g_ens[k], 1 - agree[k]) - roc_auc_score(g_ens[k], 1 - meanR[k]))
cons['common_ensemble'] = dict(auc_meanR=float(a1), auc_agreement=float(a2), diff95=list(np.percentile(diffs, [2.5, 97.5])), n_gross=int(g_ens.sum()))


def rc_area(score, err_):
    o = np.argsort(-score); keep = np.linspace(0.3, 1, 36)
    return float(np.mean([np.median(err_[o[:int(round(k_ * len(o)))]]) for k_ in keep]))


cons['rc_area'] = dict(meanR=rc_area(meanR, d_ens), agreement=rc_area(agree, d_ens), oracle=rc_area(-d_ens, d_ens),
                       random=float(np.median(d_ens)))
out['consistency'] = cons

# ---------- e. von Mises head ----------
from scipy.stats import vonmises
vmr = {}
for s in (0, 1, 2):
    c_ = pd.read_csv(D / f'pa_cnn_cv_vm_s{s}.csv')
    k_ = c_.cnn_kappa.values; z = np.radians(2 * wrap(c_.cnn_pa - c_.EPA).values)
    cov = {}
    for lev in (0.5, 0.68, 0.9, 0.95):
        lo = vonmises.ppf((1 - lev) / 2, k_); hi_ = vonmises.ppf(1 - (1 - lev) / 2, k_); cov[str(lev)] = float(np.mean((z >= lo) & (z <= hi_)))
    nll = lambda kk: float(np.mean(-kk * np.cos(z) + np.log(2 * np.pi * i0e(kk)) + kk))
    # constant kappa fitted out of fold: for each fold, the maximum likelihood kappa of the other folds
    kc = np.empty(len(z))
    for f in np.unique(c_.fold):
        o_ = c_.fold.values != f; grid = np.linspace(0.2, 20, 400)
        best = grid[np.argmin([np.mean(-g * np.cos(z[o_]) + np.log(2 * np.pi * i0e(g)) + g) for g in grid])]
        kc[c_.fold.values == f] = best
    vmr[s] = dict(coverage=cov, nll_predicted=nll(k_), nll_constant=nll(kc), spearman_kappa_R=float(stats.spearmanr(k_, c_.cnn_R).correlation))
out['vonmises'] = vmr

# ---------- f. gains against the best moment baseline ----------
bp = pd.read_csv(D / 'baselines_pa.csv')
c5 = cv[['idPNMain', 'EPA', 'mainClass', 'Glon']].copy()
c5['d_net5'] = np.mean([dpa(s_.cnn_pa.values, cv.EPA.values) for s_ in al], 0)
c5 = c5.merge(bp[['idPNMain', 'pa_eq', 'pa_low', 'pa_flux', 'dl_pa_eq']], on='idPNMain', how='left')
blk = (c5.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0
gains = {}
for cl in ['all', 'B', 'E']:
    s = c5 if cl == 'all' else c5[c5.mainClass == cl]
    med = {col: float(np.nanmedian(dpa(s[col], s.EPA))) for col in ['pa_eq', 'pa_low', 'pa_flux', 'dl_pa_eq']}
    best = min(med, key=med.get)
    ii_ = s.index.values; bb = blk[ii_]; groups = [ii_[bb == q] for q in np.unique(bb)]
    dm = dpa(c5[best], c5.EPA).values; dn = c5.d_net5.values; dcl = dpa(c5.pa_eq, c5.EPA).values; gb, gc = [], []
    for _ in range(4000):
        pick = np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))])
        gb.append(np.nanmedian(dm[pick]) - np.median(dn[pick])); gc.append(np.nanmedian(dcl[pick]) - np.median(dn[pick]))
    gains[cl] = dict(net5=float(np.median(s.d_net5)), moments=med, best=best, gain_best=float(med[best] - np.median(s.d_net5)),
                     gain_best95=list(np.percentile(gb, [2.5, 97.5])), gain_classical=float(med['pa_eq'] - np.median(s.d_net5)),
                     gain_classical95=list(np.percentile(gc, [2.5, 97.5])))
out['gains'] = gains

# ---------- g. reproductions and classification ----------
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
rzs = xr[(xr.rz_class == 'bipolar') & (xr.MajDiam >= CUT) & (xr.R_tta >= 0.8) & (xr.survey != 'SHS')]
tsel = tan[(tan.MajDiam >= CUT) & (tan.R_tta >= 0.8) & (tan.survey != 'SHS')]
rep = {}
for nm, vis, net in [('RZ13 selected', rzs.GPA_deg, rzs.gpa), ('Tan all', tan.GPA_deg, tan.gpa), ('Tan selected', tsel.GPA_deg, tsel.gpa)]:
    Rv, Rn = circ(vis)['R'], circ(net)['R']
    rep[nm] = dict(n=len(vis), R_visual=Rv, R_network=Rn, R_predicted=float(Rv * Ec), Ecos2=Ec)
out['reproductions'] = rep


def wilson(k, nn, z=1.96):
    p = k / nn; dnm = 1 + z ** 2 / nn; c = (p + z ** 2 / (2 * nn)) / dnm; w = z * np.sqrt(p * (1 - p) / nn + z ** 2 / (4 * nn ** 2)) / dnm
    return [float(c - w), float(c + w)]


out['purity'] = dict(network=dict(k=39, n=47, ci=wilson(39, 47)), hash=dict(k=49, n=74, ci=wilson(49, 74)))


# ---------- h. PSF per channel and Halpha to r registration ----------
def star_moments(im, xs, ys):
    res = []
    for yv, xv in zip(ys, xs):
        cut = im[yv - 3:yv + 4, xv - 3:xv + 4]; cut = cut - np.median(im); cut = np.clip(cut, 0, None); w = cut.sum()
        if w <= 0:
            res.append(None); continue
        gy, gx = np.mgrid[-3:4, -3:4]
        mx, my = (gx * cut).sum() / w, (gy * cut).sum() / w
        cxx = ((gx - mx) ** 2 * cut).sum() / w; cyy = ((gy - my) ** 2 * cut).sum() / w; cxy = ((gx - mx) * (gy - my) * cut).sum() / w
        if not cxx + cyy > 0:
            res.append(None); continue
        ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy); e = np.hypot(cxx - cyy, 2 * cxy) / (cxx + cyy)
        res.append((mx, my, e * np.exp(2j * np.radians(np.degrees(np.arctan2(-np.cos(ang), np.sin(ang))) % 180))))
    return res


pl = pd.read_csv(D / 'pseudolabels.csv')
from scipy.ndimage import maximum_filter
psf = {}
for sv in ['SHS', 'IPHAS', 'VPHAS']:
    files = rng.choice(pl[(pl.survey == sv) & pl.kcont.notna()].file.values, 150, replace=False)
    acc = {'ha': [], 'r': [], 'sub': [], 'off': []}
    for fn in files:
        with fits.open(D / 'cutouts' / fn) as hd:
            if len(hd) < 2 or hd[1].data is None:
                continue
            ha = np.nan_to_num(hd[0].data.astype(float)); rr = np.nan_to_num(hd[1].data.astype(float))
        k = float(pl.set_index('file').kcont.get(fn, np.nan))
        if not np.isfinite(k):
            continue
        sub = ha - k * rr; nn_ = rr.shape[0]
        med = np.median(rr); sd = 1.4826 * np.median(np.abs(rr - med)) or 1
        pk = (rr == maximum_filter(rr, 7)) & (rr > med + 20 * sd) & (rr < np.percentile(rr, 99.95))
        ys, xs = np.nonzero(pk); c0 = (nn_ - 1) / 2
        keep = (np.hypot(xs - c0, ys - c0) > nn_ / 6) & (xs > 5) & (ys > 5) & (xs < nn_ - 6) & (ys < nn_ - 6)
        xs, ys = xs[keep], ys[keep]
        if len(xs) < 3:
            continue
        mh, mr, ms = star_moments(ha, xs, ys), star_moments(rr, xs, ys), star_moments(np.abs(sub), xs, ys)
        for key, mm in [('ha', mh), ('r', mr), ('sub', ms)]:
            v = [q[2] for q in mm if q is not None]
            if len(v) >= 3:
                acc[key].append(np.mean(v))
        offs = [complex(a_[0] - b_[0], a_[1] - b_[1]) for a_, b_ in zip(mh, mr) if a_ is not None and b_ is not None]
        if len(offs) >= 3:
            o = np.mean(offs); acc['off'].append(o)
    rsv = {}
    for key in ['ha', 'r', 'sub']:
        z = np.array(acc[key]); un = z / np.abs(z)
        rsv[key] = dict(n=len(z), R=float(np.abs(un.mean())), mean_pa=float(np.degrees(np.angle(un.mean())) / 2 % 180),
                        mean_e=float(np.abs(z.mean())), rayleigh_p=float(np.exp(-len(z) * np.abs(un.mean()) ** 2)))
    o = np.array(acc['off']); po = np.degrees(np.arctan2(-o.real, o.imag)) % 180  # direction of the offset as an axis east of north
    uo = np.exp(2j * np.radians(po))
    rsv['offset'] = dict(n=len(o), median_px=float(np.median(np.abs(o))), R=float(np.abs(uo.mean())), mean_pa=float(np.degrees(np.angle(uo.mean())) / 2 % 180))
    psf[sv] = rsv; print(sv, json.dumps(rsv)[:400], flush=True)
out['psf_channels'] = psf
st = x[(x.mainClass == 'S') & (x.survey == 'IPHAS') & x.MajDiam.notna()]
out['iphas_stellar_power'] = dict(n=int(len(st)), p_if_R05=float(np.exp(-len(st) * 0.5 ** 2)))
json.dump(out, open(D / 'rev7.json', 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k not in ('inject3d', 'psf_channels')}, default=float, indent=1)[:9000])
