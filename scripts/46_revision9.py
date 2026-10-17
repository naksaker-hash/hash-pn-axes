"""Revision 9 (fourth deep review, paper_mlst/DEEP_REVIEW4.md). CPU only. Output: data/rev9.json
a. bulge binaries: probability of the observed V statistic if all are tightly aligned (4e6 samples), for the full 22, the
   bipolar class, the bulge members of Erzincan et al. and the nebulae of at most 15 arcsec, along the plane and at 107.5,
   for a weaker alignment (R_WEAK), simulated expected limits, the visual axes of the bulge binaries
b. close binary share among the HASH bipolars with the strict and the HASH class numerators
c. SHS cause: second moment of the central source in the H alpha cutouts of stellar PNe, no network, per survey;
   sign checked on large HASH ellipticals with a visual axis
d. accuracy intervals: block bootstrap of the mean of the five seed medians, bootstrap of the external medians and of
   the external network minus moment differences
e. learned moment baseline: gradient boosting on the four moment axes and mask shape, folds of seed 0, label
   transformation in feature space and averaging over feature rotations at prediction
f. other bulge bipolars: limits on the weaker alignment, rev8 procedure
g. limits against the concentration of the aligned part for Fig. 8b, rev8 procedure, along the plane and about 107.5
   scored at the plane
h. small items: Holm power of the binary test, Monte Carlo error range of the limits, Tan attenuation probability,
   RZ13 expected signal with the rev8 pool, fold structure, balanced accuracy interval, the visual scatter on the same
   objects, the V test verification at the plane, the share of known binaries in the HASH bipolar class
"""
import json, re, pathlib, importlib, sys
import numpy as np, pandas as pd
from scipy import stats
from astropy.io import fits
CUT = float(__import__('os').environ.get('CUT', '2'))
R_WEAK = float(__import__('os').environ.get('R_WEAK', '0.38'))  # minimum HASH major diameter (arcsec) of the selected axes
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
sys.path.insert(0, str(ROOT / 'scripts'))
rng = np.random.default_rng(91)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
key = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * (pa - mu))), axis=-1)
out = {}
x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
master = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain'); master['idPNMain'] = master.idPNMain.astype(int)
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
cvres = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
err_rz = wrap(xr[xr.rz_class.notna()].gpa - xr[xr.rz_class.notna()].GPA_deg).dropna().values


def P_of_f(pa_obs, R_al, mu_inj, mu_test, f, pool, ns):
    nn = len(pa_obs); sig2 = np.sqrt(-2 * np.log(R_al))
    al = rng.random((ns, nn)) < f
    pa = np.where(al, mu_inj + np.degrees(rng.normal(0, sig2, (ns, nn))) / 2, rng.uniform(0, 180, (ns, nn))) + rng.choice(pool, (ns, nn))
    return float(np.mean(Vat(pa, mu_test) <= Vat(pa_obs, mu_test)))


def limit(pa_obs, R_al, mu_inj, mu_test, pool, ns=10000):
    """The procedure of script 40: coarse search on 0.05 with 1500 samples, then 0.01 with ns samples and interpolation."""
    c = None
    for f in np.round(np.arange(0, 1.0001, 0.05), 3):
        if P_of_f(pa_obs, R_al, mu_inj, mu_test, f, pool, 1500) < 0.05:
            c = f; break
    if c is None:
        return None
    fg = np.round(np.arange(max(0, c - 0.10), min(1, c + 0.06) + 1e-9, 0.01), 3)
    P = np.array([P_of_f(pa_obs, R_al, mu_inj, mu_test, f, pool, ns) for f in fg])
    i = np.flatnonzero(P < 0.05)
    if not len(i):
        return None
    j = i[0]
    return float(fg[0]) if j == 0 else float(fg[j - 1] + (P[j - 1] - 0.05) * (fg[j] - fg[j - 1]) / (P[j - 1] - P[j]))


def p_all(pa_o, pool, mu=90.0, R_al=0.83, ns=4_000_000, chunk=200_000):
    nn = len(pa_o); s2 = np.sqrt(-2 * np.log(R_al)); Vo = Vat(pa_o, mu); hits = 0
    for _ in range(ns // chunk):
        pa = mu + np.degrees(rng.normal(0, s2, (chunk, nn))) / 2 + rng.choice(pool, (chunk, nn))
        hits += int(np.sum(Vat(pa, mu) <= Vo))
    return hits / ns


# ---------- a ----------
bj = pd.read_csv(D / 'binaries_clean.csv').merge(master[['idPNMain', 'Component']], on='idPNMain', how='left')
bs = bj[bj.in_alignment_test & bj.bulge_sky]
subs = {'all': bs, 'B': bs[bs.mainClass == 'B'], 'bulge component': bs[bs.Component == 'Bulge'], 'D <= 15': bs[bs.MajDiam <= 15]}
a = {}
for nm, s in subs.items():
    pa_o = s.gpa.values; r = dict(n=len(s))
    for en, pool in [('cv', cvres), ('rz', err_rz)]:
        ns = 4_000_000 if nm == 'all' else 400_000
        r[en] = dict(p90=p_all(pa_o, pool, 90.0, ns=ns), p107=p_all(pa_o, pool, 107.5, ns=ns), p_weak=p_all(pa_o, pool, 90.0, R_al=R_WEAK, ns=400_000))
    print('bulge binaries', nm, r, flush=True); a[nm] = r
# simulated expected limits for unaligned axes (null realisations, rev8 procedure on a reduced sample count)
def expected_limit(nn, pool, nreal=60):
    fs = []
    for _ in range(nreal):
        pa = rng.uniform(0, 180, nn) + rng.choice(pool, nn)
        v = limit(pa, 0.83, 90.0, 90.0, pool, ns=3000); fs.append(np.nan if v is None else v)
    fs = np.array(fs); return dict(median=float(np.nanmedian(fs)), frac_none=float(np.mean(np.isnan(fs))))
a['expected_limits'] = {f'{nm} {en}': expected_limit(len(subs[nm]), pool) for nm in ('all', 'B') for en, pool in [('cv', cvres), ('rz', err_rz)]}
print(a['expected_limits'], flush=True)
vis = x.set_index('idPNMain').reindex(bs.idPNMain); vv = ((vis.EPA + vis.psi) % 180).dropna().values
z = np.exp(2j * np.radians(vv)).mean(); a['visual_axes'] = dict(n=len(vv), R=float(abs(z)), V_p=float(stats.norm.sf(np.mean(np.cos(np.radians(2 * (vv - 90)))) * np.sqrt(2 * len(vv)))))
a['n_bulge_component'] = int((bs.Component == 'Bulge').sum()); a['n_candidates'] = int(bj[bj.in_alignment_test].candidate.sum())
a['n_candidates_bulge'] = int(bs.candidate.sum())
out['bulge_binaries'] = a

N_OUT = int(((x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS') & (x.mainClass == 'B') & ~x.bulge_sky).sum())
# ---------- b ----------
sh = bj[bj.short & (bj.survey != 'SHS') & bj.mainClass.isin(['B', 'E', 'R', 'I', 'A'])]
cut = (x.MajDiam >= CUT) & (x.survey != 'SHS') & ~x.bulge_sky & x.mainClass.isin(['B', 'E', 'R', 'I', 'A'])
PB_sel = float((x[cut].mainClass == 'B').mean())
PBb_all = float((sh.mainClass == 'B').mean()); PBb_out = float((sh[~sh.bulge_sky].mainClass == 'B').mean())
b = dict(P_B_selection=PB_sel, n_short_matched=len(sh), P_HASHB_given_bin=PBb_all, n_out=int((~sh.bulge_sky).sum()), P_HASHB_given_bin_outside=PBb_out)
for nm, num, den in [('strict', 0.30, 0.297), ('hash', PBb_all, PB_sel)]:
    b[nm] = dict(frac=[0.12 * num / den, 0.21 * num / den], count=[N_OUT * 0.12 * num / den, N_OUT * 0.21 * num / den])
out['binary_share'] = b; print(b, flush=True)

# ---------- c ----------
def pa_of(c):
    n0, n1 = c.shape; gy, gx = np.mgrid[:n0, :n1]; w = c.sum()
    mx, my = (gx * c).sum() / w, (gy * c).sum() / w; dx, dy = gx - mx, gy - my
    sxx = (dx * dx * c).sum() / w; syy = (dy * dy * c).sum() / w; sxy = (dx * dy * c).sum() / w
    return (90 + 0.5 * np.degrees(np.arctan2(2 * sxy, sxx - syy))) % 180
def central(fn):
    im = np.nan_to_num(fits.getdata(D / 'cutouts' / fn).astype(float)); n = im.shape[0]
    gy, gx = np.mgrid[:n, :n] - (n - 1) / 2
    return pa_of(np.clip(im - np.median(im), 0, None) * (np.hypot(gx, gy) < n / 6))
ax = lambda a_: (float(np.degrees(np.angle(np.mean(np.exp(2j * np.radians(a_))))) / 2 % 180), float(abs(np.mean(np.exp(2j * np.radians(a_))))))
c = {}
el = x[(x.mainClass == 'E') & x.EPA.notna() & (x.MajDiam >= 20) & (x.q_mask < 0.8)].head(80)
cen = np.array([central(f) for f in el.file]); c['sign_check_ellipticals'] = dict(n=len(el), mean_cos2=float(np.mean(np.cos(np.radians(2 * (cen - el.EPA))))))
for sv in ('SHS', 'IPHAS', 'VPHAS'):
    st = x[(x.mainClass == 'S') & (x.survey == sv) & x.MajDiam.notna()]
    cen = np.array([central(f) for f in st.file]); m_, R_ = ax(cen); n = len(cen)
    c[sv] = dict(n=n, mean=m_, R=R_, rayleigh_p=float(np.exp(-n * R_ ** 2)), agree_network=float(np.mean(np.cos(np.radians(2 * (st.pa.values - cen))))))
# 1 sigma of the mean axis of the network axes of the SHS stellar PNe (bootstrap)
st = x[(x.mainClass == 'S') & (x.survey == 'SHS') & x.MajDiam.notna()].pa.values; m0 = ax(st)[0]
bm = [m0 + wrap(ax(st[rng.integers(0, len(st), len(st))])[0] - m0) for _ in range(4000)]
c['network_mean_sd'] = float(np.std(bm)); c['offset_122_in_sd'] = float(abs(wrap(122 - m0)) / np.std(bm))
out['shs_central_moment'] = c; print(c, flush=True)

# ---------- d ----------
ids = cv.idPNMain.values
seeds = [pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids) for t in ['', '_s1', '_s2', '_s3', '_s4']]
E = np.array([dpa(s.cnn_pa.values, cv.EPA.values) for s in seeds])
blk = (cv.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0; groups = [np.flatnonzero(blk == q) for q in np.unique(blk)]
bsd = []
for _ in range(4000):
    pk = np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]); bsd.append(np.mean(np.median(E[:, pk], 1)))
d = dict(mean_of_medians=float(np.mean(np.median(E, 1))), block95=list(np.percentile(bsd, [2.5, 97.5])), block68=list(np.percentile(bsd, [16, 84])))
bp = pd.read_csv(D / 'baselines_pa.csv'); fullp = pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']]
m = master.copy(); m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['png'] = 'PN G' + m.PNG; m['nkey'] = m.Name.map(key)
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(key)
jr = m[['png', 'idPNMain', 'EPA']].merge(rz[['PNG', 'EPA_deg']].rename(columns={'PNG': 'png'}), on='png')
j1 = m[['nkey', 'png', 'idPNMain', 'EPA']].merge(co[['nkey', 'EPA_deg']], on='nkey')
j2 = m[['png', 'idPNMain', 'EPA']].merge(co[['PNG_approx_from_lb', 'EPA_deg']].rename(columns={'PNG_approx_from_lb': 'png'}), on='png')
jc = pd.concat([j1, j2[~j2.idPNMain.isin(j1.idPNMain)]]).drop_duplicates('idPNMain')
for name, j in [('RZ13', jr), ('Corradi98', jc)]:
    j = j[j.EPA.isna()].merge(bp, on='idPNMain').merge(fullp, on='idPNMain'); j = j[j.MajDiam >= 4]
    dn = dpa(j.cnn_pa, j.EPA_deg).values; dm = dpa(j.pa_eq, j.EPA_deg).values; ok = np.isfinite(dm)
    bn = [np.median(dn[rng.integers(0, len(dn), len(dn))]) for _ in range(4000)]
    kk = np.flatnonzero(ok); bdiff = [np.median(dm[k]) - np.median(dn[k]) for k in (kk[rng.integers(0, len(kk), len(kk))] for _ in range(4000))]
    d[name] = dict(n=len(dn), median=float(np.median(dn)), ci95=list(np.percentile(bn, [2.5, 97.5])), diff_classical=float(np.median(dm[ok]) - np.median(dn[ok])), diff95=list(np.percentile(bdiff, [2.5, 97.5])))
out['accuracy_intervals'] = d; print(d, flush=True)

# ---------- e ----------
from sklearn.ensemble import HistGradientBoostingRegressor
s0 = cv.set_index('idPNMain'); bb = bp.set_index('idPNMain')
X0 = s0.join(bb[['pa_low', 'pa_flux', 'dl_pa_eq']], how='left')
angs = ['pa_eq', 'pa_low', 'pa_flux', 'dl_pa_eq']; X0 = X0[X0[angs].notna().all(1)]
inv = np.column_stack([X0.q_mask.fillna(1), np.log(X0.MajDiam), np.log1p(X0.peak_snr.fillna(0)), (X0.maj_mask / X0.MajDiam).fillna(1),
                       (X0.survey == 'SHS').astype(float), (X0.survey == 'IPHAS').astype(float)]
                      + [np.cos(np.radians(2 * (X0[p] - X0[q]))) for i, p in enumerate(angs) for q in angs[i + 1:]])
A = X0[angs].values; y = X0.EPA.values; fold = X0.fold.values; pred = np.zeros(len(y))
for f in range(5):
    tr = fold != f; te = fold == f; Xs, Ys = [], []
    for _ in range(32):
        phi = rng.uniform(0, 180, tr.sum()); fl = rng.random(tr.sum()) < 0.5
        A2 = np.where(fl[:, None], -A[tr], A[tr]) + phi[:, None]; y2 = np.where(fl, -y[tr], y[tr]) + phi
        Xs.append(np.column_stack([inv[tr], np.cos(np.radians(2 * A2)), np.sin(np.radians(2 * A2))])); Ys.append(np.column_stack([np.cos(np.radians(2 * y2)), np.sin(np.radians(2 * y2))]))
    Xs = np.vstack(Xs); Ys = np.vstack(Ys)
    mc = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xs, Ys[:, 0])
    ms = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(Xs, Ys[:, 1])
    zz = 0
    for phi in np.arange(0, 180, 22.5):
        A2 = A[te] + phi; Xt = np.column_stack([inv[te], np.cos(np.radians(2 * A2)), np.sin(np.radians(2 * A2))])
        zz = zz + np.exp(-2j * np.radians(phi)) * (mc.predict(Xt) + 1j * ms.predict(Xt))
    pred[te] = (np.degrees(np.angle(zz)) / 2) % 180
el_ = dpa(pred, y); cls = X0.mainClass.values
e = {nm: dict(n=int(mm.sum()), median=float(np.median(el_[mm])), gt60=float(np.mean(el_[mm] > 60)), network_seed0=float(np.median(dpa(X0.cnn_pa.values, y)[mm])))
     for nm, mm in [('all', np.ones(len(y), bool)), ('B', cls == 'B'), ('E', cls == 'E')]}
out['learned_moments'] = e; print(e, flush=True)

# ---------- f ----------
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
ob_b = x[base & (x.mainClass == 'B') & x.bulge_sky & x.morph_class.isna()].gpa.values
out['other_bulge_bipolars'] = dict(n=len(ob_b), weak_cv=limit(ob_b, R_WEAK, 90.0, 90.0, cvres), weak_rz=limit(ob_b, R_WEAK, 90.0, 90.0, err_rz))
print(out['other_bulge_bipolars'], flush=True)

# ---------- g ----------
gpa = x[base & (x.mainClass == 'B') & ~x.bulge_sky].gpa.values
g = {}
for rho in (0.2, 0.29, 0.38, 0.4, 0.5, 0.6, 0.7, 0.83, 0.95):
    g[str(rho)] = dict(plane=limit(gpa, rho, 90.0, 90.0, cvres), at107_scored_plane=limit(gpa, rho, 107.5, 90.0, cvres))
    print('rho', rho, g[str(rho)], flush=True)
out['f95_vs_rho'] = g

# ---------- h ----------
h = {}
bo = bj[bj.in_alignment_test & ~bj.bulge_sky]; nb = len(bo); s2 = np.sqrt(-2 * np.log(0.83))
pa = 90 + np.degrees(rng.normal(0, s2, (200000, nb))) / 2 + rng.choice(cvres, (200000, nb))
h['holm_binary_power'] = float(np.mean(stats.norm.sf(Vat(pa, 90.0) * np.sqrt(2 * nb)) < 0.025))
L = json.load(open(D / 'rev8_limits.json'))['limits']
se = [v['mc_se'] for mm in L.values() for v in mm.values() if v.get('mc_se')]; h['mc_se_range'] = [float(min(se)), float(max(se))]
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg']); Rb = lambda a_: abs(np.mean(np.exp(2j * np.radians(a_))))
sims = np.array([Rb(tan.GPA_deg.values + rng.choice(cvres, len(tan))) for _ in range(20000)]); h['tan_P_R_le_obs'] = float(np.mean(sims <= Rb(tan.gpa.values)))
h['rz_expected_sigma_bulge'] = float(0.83 * np.mean(np.cos(np.radians(2 * err_rz))) * np.sqrt(2 * len(bs)))
gc = blk == 0; e0 = dpa(cv.cnn_pa, cv.EPA).values
h['fold_gc'] = dict(n=int(gc.sum()), frac=float(gc.mean()), median_gc=float(np.median(e0[gc])), median_rest=float(np.median(e0[~gc])))
mc_ = pd.read_csv(D / 'morph_cv.csv'); cls4 = ['E', 'R', 'B', 'S']; mc_ = mc_[mc_.mainClass.isin(cls4)]
bacc = lambda d_: np.mean([np.mean(d_[d_.mainClass == k].cnn_class == k) for k in cls4])
mb = mc_.copy(); mb['blk'] = (mb.Glon // 10).fillna(-1).astype(int)
gg = [mb[mb.blk == q] for q in mb.blk.unique()]
h['balanced_accuracy'] = dict(value=float(bacc(mc_)), ci95=list(np.percentile([bacc(pd.concat([gg[q] for q in rng.integers(0, len(gg), len(gg))])) for _ in range(1000)], [2.5, 97.5])))
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv'); w['k'] = w.PNG_from_name.fillna(w.PNG_approx_from_lb)
co2 = co.copy(); co2['k'] = co2.PNG_approx_from_lb
yy = w[w.source != 'CAM98'].merge(co2, on='k', suffixes=('_w', '_c')).drop_duplicates('k').set_index('k')
lab = x.set_index('idPNMain').loc[ids].copy(); lab['k'] = 'PN G' + lab.PNG.astype(str); sub = lab[lab.k.isin(yy.index)]
Wv = sub.EPA.values; Cv = yy.loc[sub.k, 'EPA_deg_c'].values; Nv = s0.loc[sub.index, 'cnn_pa'].values
h['visual_same_objects'] = dict(n=len(sub), net_vs_W=float(np.median(dpa(Nv, Wv))), net_vs_C=float(np.median(dpa(Nv, Cv))), W_vs_C=float(np.median(dpa(Wv, Cv))))
Vn = Vat(rng.uniform(0, 180, (200000, len(gpa))), 90.0); Vo = Vat(gpa, 90.0)
h['verification_90'] = dict(V_obs=float(Vo), sim=float(np.mean(Vn <= Vo)), analytic=float(stats.norm.cdf(Vo * np.sqrt(2 * len(gpa)))))
out['misc'] = h; print(h, flush=True)
json.dump(out, open(D / 'rev9.json', 'w'), indent=1, default=float)

