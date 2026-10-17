"""Revision 6 analyses (deep review by four referees). Output: data/rev6.json

a. block (cluster) bootstrap of the cross validation median and paired bootstrap of network minus moments
b. consistency selection in cross validation: gross failures above and below R_TTA 0.8, AUC with interval
c. test time rotations by seed
d. confirmatory family of two tests and paired agreement for the reproductions of Rees & Zijlstra and Tan et al.
e. close binary fraction among bipolars with consistent observed frame inputs
f. upper limits: interpolated crossing with Monte Carlo error, expected limit, limits against rho and the
   propagated uncertainty of the error pool and of rho
g. 3D injections: axes along the local azimuthal field or the Galactic pole, projected, with a bipolar
   visibility cut on inclination
h. PSF elongation of field stars per survey, image frame tests of axisless PNe per survey and in the bulge region
i. classifier calibration, distances and components, bias corrected R, Monte Carlo error of the null check
"""
import json, pathlib, re
import numpy as np, pandas as pd
from scipy import stats
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(61)
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


x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
blk = (cv.Glon // 10).astype(int).to_numpy().copy(); blk[blk == 35] = 0; cv['blk'] = blk

# ---------- a. block bootstrap and paired differences ----------
a = {}
ub = np.unique(blk); idx_by = {b: np.flatnonzero(blk == b) for b in ub}
for cl in ['all', 'B', 'E']:
    s = cv if cl == 'all' else cv[cv.mainClass == cl]
    ii = s.index.values; bb = blk[ii]; groups = [ii[bb == b] for b in np.unique(bb)]
    meds, diffs = [], []
    for _ in range(4000):
        pick = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        meds.append(np.median(cv.d_cnn.values[pick])); diffs.append(np.nanmedian(cv.d_mom.values[pick]) - np.median(cv.d_cnn.values[pick]))
    a[f'cv {cl}'] = dict(n=len(s), median=float(s.d_cnn.median()), block68=list(np.percentile(meds, [16, 84])),
                         block95=list(np.percentile(meds, [2.5, 97.5])), mom_minus_net=float(np.nanmedian(s.d_mom) - s.d_cnn.median()),
                         mom_minus_net_block95=list(np.percentile(diffs, [2.5, 97.5])), n_blocks=len(groups))
# external tests: paired iid bootstrap (objects scored as in 10_independent_pa_test.py)
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
m = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['idPNMain'] = m.idPNMain.astype(int); m['MajDiam'] = pd.to_numeric(m.MajDiam, errors='coerce')
m['png'] = 'PN G' + m.PNG; m['nkey'] = m.Name.map(norm)
h = m[['idPNMain', 'mainClass', 'MajDiam', 'png', 'nkey', 'EPA']].merge(pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']], on='idPNMain', how='left') \
    .merge(pd.read_csv(D / 'pseudolabels.csv')[['idPNMain', 'pa_eq']], on='idPNMain', how='left')
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(norm)
ext = {'RZ13': h.merge(rz[['PNG', 'EPA_deg']], left_on='png', right_on='PNG')}
b1 = h.merge(co[['nkey', 'EPA_deg']], on='nkey')
b2 = h.merge(co[['PNG_approx_from_lb', 'EPA_deg']], left_on='png', right_on='PNG_approx_from_lb')
ext['Corradi98'] = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain')
for nm, e in ext.items():
    e = e[e.EPA.isna() & (e.MajDiam >= 4) & e.cnn_pa.notna() & e.pa_eq.notna()]
    dn, dm = dpa(e.cnn_pa, e.EPA_deg).values, dpa(e.pa_eq, e.EPA_deg).values
    bs = [np.median(dm[k]) - np.median(dn[k]) for k in (rng.integers(0, len(dn), len(dn)) for _ in range(4000))]
    bn = [np.median(dn[k]) for k in (rng.integers(0, len(dn), len(dn)) for _ in range(4000))]
    a[nm] = dict(n=len(e), median=float(np.median(dn)), boot95=list(np.percentile(bn, [2.5, 97.5])),
                 mom_minus_net=float(np.median(dm) - np.median(dn)), mom_minus_net_95=list(np.percentile(bs, [2.5, 97.5])))
out['bootstrap'] = a

# ---------- b. consistency selection in cross validation ----------
g = cv.d_cnn > 60; hi = cv.cnn_R >= 0.8
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(g, 1 - cv.cnn_R); aucs = []
for _ in range(4000):
    k = rng.integers(0, len(cv), len(cv))
    if g.values[k].any() and (~g.values[k]).any():
        aucs.append(roc_auc_score(g.values[k], 1 - cv.cnn_R.values[k]))
out['rtta_cv'] = dict(n_gross=int(g.sum()), gross_above=int((g & hi).sum()), n_above=int(hi.sum()), n_below=int((~hi).sum()),
                      rate_above=float(g[hi].mean()), rate_below=float(g[~hi].mean()), auc=float(auc), auc95=list(np.percentile(aucs, [2.5, 97.5])))

# ---------- c. test time rotations by seed ----------
tt = {}
for tag in ['', '_s1', '_s2']:
    c = pd.read_csv(D / f'pa_cnn_cv{tag}.csv')
    tt[tag or '_s0'] = {k: float(np.median(dpa(c[col], c.EPA))) for k, col in [('1', 'cnn_pa_tta1'), ('4', 'cnn_pa_tta4'), ('8', 'cnn_pa')]}
out['tta_by_seed'] = tt

# ---------- d. confirmatory family and paired agreement ----------
base = (x.MajDiam >= 6) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B')
ob = x[B & ~x.bulge_sky]
bj = pd.read_csv(D / 'supp_binaries.csv'); bsel = bj[bj.in_alignment_test]; bo = bsel[~bsel.bulge_sky]
fam = {'outside bulge region bipolars at 90': circ(ob.gpa)['V_p'], 'known binaries outside bulge region at 90': circ(bo.gpa)['V_p']}
ps = np.array(list(fam.values())); order = np.argsort(ps); holm = np.empty(2)
run_max = 0
for r, i in enumerate(order):
    run_max = max(run_max, min(1, (2 - r) * ps[i])); holm[i] = run_max
out['confirmatory'] = dict(tests={k: dict(p=float(p), p_holm=float(hh)) for (k, p), hh in zip(fam.items(), holm)},
                           binaries_outside_at_107p5=circ(bo.gpa, 107.5), binaries_outside_at_90=circ(bo.gpa))
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg'])
agree = {}
for nm, s in [('Tan all', tan), ('Tan selected', tan[(tan.MajDiam >= 6) & (tan.R_tta >= 0.8) & (tan.survey != 'SHS')])]:
    d = wrap(s.gpa - s.GPA_deg).values
    shift = [np.degrees(np.angle(np.mean(np.exp(2j * np.radians(d[k]))))) / 2 for k in (rng.integers(0, len(d), len(d)) for _ in range(4000))]
    agree[nm] = dict(n=len(d), median_abs=float(np.median(np.abs(d))), frac_within10=float(np.mean(np.abs(d) < 10)),
                     mean_shift=float(np.degrees(np.angle(np.mean(np.exp(2j * np.radians(d))))) / 2), shift95=list(np.percentile(shift, [2.5, 97.5])))
xr = x.merge(rz[['PNG', 'EPA_deg', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rzc'}), on='png')
rzs = xr[(xr.rzc == 'bipolar') & (xr.MajDiam >= 6) & (xr.R_tta >= 0.8) & (xr.survey != 'SHS')]
d = wrap(rzs.gpa - rzs.GPA_deg).values
shift = [np.degrees(np.angle(np.mean(np.exp(2j * np.radians(d[k]))))) / 2 for k in (rng.integers(0, len(d), len(d)) for _ in range(4000))]
agree['RZ13 selected bipolars'] = dict(n=len(d), median_abs=float(np.median(np.abs(d))), frac_within10=float(np.mean(np.abs(d) < 10)),
                                       mean_shift=float(np.degrees(np.angle(np.mean(np.exp(2j * np.radians(d))))) / 2), shift95=list(np.percentile(shift, [2.5, 97.5])))
out['reproduction_agreement'] = agree

# ---------- e. close binary fraction among bipolars ----------
cuts = x[x.mainClass.notna()]
res_ = cuts[cuts.mainClass != 'S']
shares = dict(all_imaged=float((cuts.mainClass == 'B').mean()), resolved=float((res_.mainClass == 'B').mean()),
              resolved_ge6=float((res_[res_.MajDiam >= 6].mainClass == 'B').mean()),
              selected_axisym=float((x[base & x.mainClass.isin(['B', 'E'])].mainClass == 'B').mean()))
fb = {}
for k, pB in shares.items():
    fb[k] = dict(P_B=pB, observed_frame=[0.12 * 0.30 / pB, 0.21 * 0.30 / pB], with_intrinsic_60=[0.12 * 0.30 / pB, 0.21 * 0.60 / pB])
out['binary_fraction'] = fb

# ---------- f. upper limits ----------
mres = (cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
err = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
Ec = float(np.mean(np.cos(np.radians(2 * err))))
gpa = ob.gpa.values; n = len(gpa)
Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu)), axis=-1)


def Pgrid(R_al, mu, fgrid, nsim, e=err, pa_obs=gpa):
    nn = len(pa_obs); Vobs = Vat(pa_obs, mu); sig2 = np.sqrt(-2 * np.log(R_al)); P = []
    for f in fgrid:
        al = rng.random((nsim, nn)) < f
        pa = np.where(al, mu + np.degrees(rng.normal(0, sig2, (nsim, nn))) / 2, rng.uniform(0, 180, (nsim, nn)))
        pa = pa + rng.choice(e, (nsim, nn))
        P.append(np.mean(Vat(pa, mu) <= Vobs))
    return np.array(P), float(Vobs)


def crossing(fgrid, P):
    i = np.flatnonzero(P < 0.05)
    if not len(i) or i[0] == 0:
        return None
    j = i[0]; return float(fgrid[j - 1] + (P[j - 1] - 0.05) * (fgrid[j] - fgrid[j - 1]) / (P[j - 1] - P[j]))


lim = {}
fg = np.round(np.arange(0.10, 0.26, 0.01), 2)
P, Vobs = Pgrid(0.83, 107.5, fg, 20000)
c0 = crossing(fg, P)
slope = np.gradient(P, fg)[np.argmin(np.abs(fg - c0))]
lim['tight 107.5 cv'] = dict(V_obs=Vobs, crossing=c0, mc_se=float(np.sqrt(0.05 * 0.95 / 20000) / abs(slope)), grid={float(f): float(p) for f, p in zip(fg, P)})
# expected limit: median over null realisations of the observed statistic
Vnull = Vat(rng.uniform(0, 180, (4000, n)) + rng.choice(err, (4000, n)), 107.5)
lim['tight 107.5 cv']['expected_crossing_analytic'] = float((np.median(Vnull) + 1.645 / np.sqrt(2 * n)) / (0.83 * Ec))
lim['tight 107.5 cv']['observed_crossing_analytic'] = float((Vobs + 1.645 / np.sqrt(2 * n)) / (0.83 * Ec))
# limit against rho, simulated at 90 deg and 107.5 deg
lr = {}
for mu in (90.0, 107.5):
    for rho in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        fgr = np.round(np.arange(0.02, 1.001, 0.02), 2)
        Pr, _ = Pgrid(rho, mu, fgr, 2000)
        lr[f'{mu} {rho}'] = crossing(fgr, Pr)
lim['f95_vs_rho'] = lr
# propagated uncertainty of the error pool and of rho (analytic inversion, which matches the simulation)
fs = []
for _ in range(4000):
    eb = rng.choice(err, len(err)); rho = np.clip(rng.normal(0.83, 0.05), 0.5, 0.99)
    fs.append((Vobs + 1.645 / np.sqrt(2 * n)) / (rho * np.mean(np.cos(np.radians(2 * eb)))))
lim['tight propagated'] = dict(Ecos2=Ec, Ecos2_95=list(np.percentile([np.mean(np.cos(np.radians(2 * rng.choice(err, len(err))))) for _ in range(4000)], [2.5, 97.5])),
                               f95_median=float(np.median(fs)), f95_68=list(np.percentile(fs, [16, 84])), f95_95=list(np.percentile(fs, [2.5, 97.5])))
# null check Monte Carlo error
Vn0 = Vat(rng.uniform(0, 180, (200000, n)), 107.5)
lim['null_check'] = dict(P_null_mc=float(np.mean(Vn0 <= Vobs)), P_null_analytic=float(stats.norm.cdf(Vobs * np.sqrt(2 * n))), se_3000=float(np.sqrt(0.7 * 0.3 / 3000)))
out['limits'] = lim

# ---------- g. 3D injections ----------
R0 = 8.2
l = np.radians(ob.Glon.values); b = np.radians(ob.Glat.values)
dist = ob.rgeo_kpc.fillna(ob.rgeo_kpc.median()).clip(0.3, 15).values
u = np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], 1)
P3 = np.array([-R0, 0, 0]) + dist[:, None] * u
nhat = np.stack([-np.sin(b) * np.cos(l), -np.sin(b) * np.sin(l), np.cos(b)], 1)
ehat = np.stack([-np.sin(l), np.cos(l), np.zeros_like(l)], 1)
phi_hat = np.stack([-P3[:, 1], P3[:, 0], np.zeros(n)], 1); phi_hat /= np.linalg.norm(phi_hat, axis=1, keepdims=True)
r_hat = np.stack([P3[:, 0], P3[:, 1], np.zeros(n)], 1); r_hat /= np.linalg.norm(r_hat, axis=1, keepdims=True)
pitch = np.radians(11)
field = np.cos(pitch) * phi_hat + np.sin(pitch) * r_hat
pole = np.tile([0, 0, 1.0], (n, 1))
cos_vis = np.cos(np.radians(30))  # bipolar only when the axis is more than 30 deg from the line of sight


def fisher(mu, kappa, size):
    xi = rng.random(size); w = 1 + np.log(xi + (1 - xi) * np.exp(-2 * kappa)) / kappa
    ph = rng.uniform(0, 2 * np.pi, size); s = np.sqrt(np.clip(1 - w ** 2, 0, None))
    a = np.abs(mu[:, 0]) < 0.9
    t1 = np.where(a[:, None], np.cross(mu, [1, 0, 0]), np.cross(mu, [0, 1, 0])); t1 /= np.linalg.norm(t1, axis=-1, keepdims=True)
    t2 = np.cross(mu, t1)
    return w[..., None] * mu + s[..., None] * (np.cos(ph)[..., None] * t1 + np.sin(ph)[..., None] * t2)


def draw_axes(mu_dir, kappa, aligned):
    """Axes for one realisation: aligned objects from a Fisher distribution, others isotropic, all visible."""
    ax = np.empty((n, 3)); todo = np.ones(n, bool)
    while todo.any():
        k = np.flatnonzero(todo)
        iso = rng.normal(size=(len(k), 3)); iso /= np.linalg.norm(iso, axis=1, keepdims=True)
        fi = fisher(mu_dir[k], kappa, len(k))
        cand = np.where(aligned[k][:, None], fi, iso)
        ok = np.abs(np.sum(cand * u[k], 1)) < cos_vis
        ax[k[ok]] = cand[ok]; todo[k[ok]] = False
    return (np.degrees(np.arctan2(np.sum(ax * ehat, 1), np.sum(ax * nhat, 1)))) % 180


V90obs = Vat(gpa, 90.0); Robs = circ(gpa)['R']
g3 = {}
for name, mu_dir in [('field', field), ('pole', pole)]:
    for kappa in (20.0, 5.0):
        res = {}
        for f in np.round(np.arange(0, 1.001, 0.05), 2):
            Vs, Rs = [], []
            for _ in range(800):
                pa = draw_axes(mu_dir, kappa, rng.random(n) < f) + rng.choice(err, n)
                th = np.radians(2 * pa); Vs.append(np.mean(np.cos(th - np.pi))); Rs.append(np.abs(np.mean(np.exp(1j * th))))
            Vs, Rs = np.array(Vs), np.array(Rs)
            stat = Vs if name == 'field' else -Vs
            obs = V90obs if name == 'field' else -V90obs
            res[float(f)] = dict(P_V=float(np.mean(stat <= obs)), P_R=float(np.mean(Rs <= Robs)), mean_gpa_R=float(np.mean(Rs)))
        fV = next((f for f in sorted(res) if res[f]['P_V'] < 0.05), None); fR = next((f for f in sorted(res) if res[f]['P_R'] < 0.05), None)
        g3[f'{name} kappa {kappa:g}'] = dict(f95_V=fV, f95_R=fR, grid=res)
        print(name, kappa, fV, fR, flush=True)
# projected concentration of a fully aligned visible population, for reference
for name, mu_dir in [('field', field), ('pole', pole)]:
    pa = draw_axes(mu_dir, 20.0, np.ones(n, bool)); g3[f'{name} fully aligned kappa 20 projected'] = circ(pa)
out['inject3d'] = g3

# ---------- h. PSF elongation of field stars and image frame null tests ----------
def star_pa(fn):
    with fits.open(D / 'cutouts' / fn) as hd:
        im = hd[0].data.astype(float)
    if im is None or not np.isfinite(im).any():
        return None
    im = np.nan_to_num(im, nan=np.nanmedian(im)); nn_ = im.shape[0]
    med = np.median(im); sd = 1.4826 * np.median(np.abs(im - med)) or 1
    from scipy.ndimage import maximum_filter
    pk = (im == maximum_filter(im, 7)) & (im > med + 20 * sd) & (im < np.percentile(im, 99.95))
    yy, xx = np.nonzero(pk); c = (nn_ - 1) / 2
    keep = (np.hypot(xx - c, yy - c) > nn_ / 6) & (xx > 5) & (yy > 5) & (xx < nn_ - 6) & (yy < nn_ - 6)
    vs = []
    for yv, xv in zip(yy[keep], xx[keep]):
        cut = im[yv - 3:yv + 4, xv - 3:xv + 4] - med
        cut = np.clip(cut, 0, None); w = cut.sum()
        if w <= 0:
            continue
        gy, gx = np.mgrid[-3:4, -3:4]
        mx, my = (gx * cut).sum() / w, (gy * cut).sum() / w
        cxx = ((gx - mx) ** 2 * cut).sum() / w; cyy = ((gy - my) ** 2 * cut).sum() / w; cxy = ((gx - mx) * (gy - my) * cut).sum() / w
        ang = 0.5 * np.arctan2(2 * cxy, cxx - cyy)  # direction of the major axis in pixel coordinates (x right, y up = north)
        if not cxx + cyy > 0:
            continue
        e = np.hypot(cxx - cyy, 2 * cxy) / (cxx + cyy)
        pa = np.degrees(np.arctan2(-np.cos(ang), np.sin(ang))) % 180  # east of north, east to decreasing x
        vs.append(e * np.exp(2j * np.radians(pa)))
    return np.mean(vs) if len(vs) >= 3 else None


pl = pd.read_csv(D / 'pseudolabels.csv')
psf = {}
for sv in ['SHS', 'IPHAS', 'VPHAS']:
    files = pl[pl.survey == sv].file.values
    files = rng.choice(files, min(250, len(files)), replace=False)
    zs = [z for z in (star_pa(f) for f in files) if z is not None and np.isfinite(z) and abs(z) > 0]
    zs = np.array(zs); un = zs / np.abs(zs)
    psf[sv] = dict(n_fields=len(zs), median_ellipticity=float(np.median(np.abs(zs))), R=float(np.abs(un.mean())),
                   mean_pa=float(np.degrees(np.angle(un.mean())) / 2 % 180), rayleigh_p=float(np.exp(-len(zs) * np.abs(un.mean()) ** 2)))
    print(sv, psf[sv], flush=True)
out['psf'] = psf
nul = {}
for sv in ['SHS', 'IPHAS', 'VPHAS']:
    for cls in [['S'], ['R'], ['S', 'R']]:
        s = x[(x.survey == sv) & x.mainClass.isin(cls) & x.MajDiam.notna()]
        nul[f'{sv} {"+".join(cls)} equatorial'] = circ(s.pa, 113)
s = x[(x.survey == 'VPHAS') & x.mainClass.isin(['S']) & x.bulge_sky]
nul['VPHAS stellar bulge region equatorial'] = circ(s.pa)
s = x[(x.survey == 'VPHAS') & x.mainClass.isin(['S', 'R']) & x.bulge_sky]
nul['VPHAS stellar and round bulge region equatorial'] = circ(s.pa)
bb = x[B & x.bulge_sky]
nul['bulge region selected bipolars psi range'] = [float(bb.psi.min() % 180), float(bb.psi.max() % 180)]
nul['psi concentration bulge'] = float(np.abs(np.mean(np.exp(2j * np.radians(bb.psi)))))
nul['psi concentration outside'] = float(np.abs(np.mean(np.exp(2j * np.radians(ob.psi)))))
out['image_frame'] = nul

# ---------- i. classifier calibration, distances, components, bias corrected R ----------
mc = pd.read_csv(D / 'morph_cv.csv')
yb = (mc.mainClass == 'B').values; pb = mc.p_B.values
bins = np.linspace(0, 1, 11); rel = []
for lo, hi_ in zip(bins[:-1], bins[1:]):
    k = (pb >= lo) & (pb < hi_ + (1e-9 if hi_ == 1 else 0))
    if k.sum():
        rel.append(dict(lo=float(lo), hi=float(hi_), n=int(k.sum()), mean_p=float(pb[k].mean()), frac_B=float(yb[k].mean())))
ece = sum(r['n'] * abs(r['mean_p'] - r['frac_B']) for r in rel) / len(pb)
out['classifier_calibration'] = dict(bins=rel, ece=float(ece))
hm = pd.read_csv(D / 'hash_master.csv', low_memory=False)
snr = pd.to_numeric(hm.plx_snr, errors='coerce')
out['distances'] = dict(n=len(hm), with_rgeo=int(hm.rgeo_kpc.notna().sum()), with_gaia=int(hm.GaiaEDR3.notna().sum()),
                        plx_snr_gt3=int((snr > 3).sum()), components=hm.Component.value_counts().to_dict(),
                        bulge_region_selected_B=int(len(bb)), bulge_region_selected_B_plx_snr_gt3=int((pd.to_numeric(hm.set_index('idPNMain').reindex(bb.idPNMain).plx_snr, errors='coerce') > 3).sum()))
comp = {}
for cname in ['Thin Disk', 'Thick Disk']:
    comp[cname] = circ(ob[ob.Component == cname].gpa)
out['components_outside'] = comp
out['bias_corrected'] = {k: circ(v) for k, v in {'RZ13 selected visual': rzs.GPA_deg, 'RZ13 selected network': rzs.gpa,
                                                  'Tan selected network': tan[(tan.MajDiam >= 6) & (tan.R_tta >= 0.8) & (tan.survey != 'SHS')].gpa,
                                                  'Tan all network': tan.gpa}.items()}
json.dump(out, open(D / 'rev6.json', 'w'), indent=1, default=float)
for k, v in out.items():
    if k not in ('limits', 'inject3d'):
        print(k, json.dumps(v, default=float)[:900])
print('limits', json.dumps({k: v for k, v in out['limits'].items() if k != 'tight 107.5 cv'}, default=float)[:1500])
print('tight', {k: v for k, v in out['limits']['tight 107.5 cv'].items() if k != 'grid'})
print('inject3d', {k: (v.get('f95_V'), v.get('f95_R')) if 'grid' in v else v for k, v in out['inject3d'].items()})
