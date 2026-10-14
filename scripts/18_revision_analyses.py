"""Revision 1 analyses (referee round 1). Output: data/rev1.json, data/pa_final.csv, data/tan_comparison.csv

Axis adopted for each PN: the out-of-fold prediction of the axis network for PNe with a HASH position angle
(never their own training label), otherwise the prediction of the network trained on all labels.
Alignment samples: HASH major diameter >= 6 arcsec, R_TTA >= 0.8, IPHAS or VPHAS+ (SHS excluded because
stellar PNe show a preferred direction in SHS), B, and E with mask axial ratio < 0.85. Same cuts for every
sample, including the Rees & Zijlstra (2013) bipolars.
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(11)
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
wrap = lambda a: (a + 90) % 180 - 90
out = {}

# ---------- adopted axes ----------
m = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
for c in ['DRAJ2000', 'DDECJ2000', 'MajDiam', 'EPA', 'Glon', 'Glat', 'AngDiam_pc', 'rgeo_kpc']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['idPNMain'] = m.idPNMain.astype(int)
full = pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa', 'cnn_R']]
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
oof = cv[['idPNMain', 'cnn_pa', 'cnn_R']].rename(columns={'cnn_pa': 'oof_pa', 'cnn_R': 'oof_R'})
pl = pd.read_csv(D / 'pseudolabels.csv')[['idPNMain', 'survey', 'q_mask', 'pa_eq', 'file', 'pixscale']]
x = m[['idPNMain', 'PNG', 'Name', 'mainClass', 'Component', 'Glon', 'Glat', 'MajDiam', 'EPA', 'DRAJ2000', 'DDECJ2000',
       'AngDiam_pc', 'rgeo_kpc']].merge(pl, on='idPNMain').merge(full, on='idPNMain').merge(oof, on='idPNMain', how='left')
x['trained'] = x.oof_pa.notna()
x['pa'] = np.where(x.trained, x.oof_pa, x.cnn_pa)
x['R_tta'] = np.where(x.trained, x.oof_R, x.cnn_R)
c = SkyCoord(x.DRAJ2000.values * u.deg, x.DDECJ2000.values * u.deg)
x['psi'] = c.galactic.position_angle(c.directional_offset_by(0 * u.deg, 1 * u.arcsec).galactic).deg
x['gpa'] = (x.pa + x.psi) % 180
l = ((x.Glon + 180) % 360) - 180
x['bulge_sky'] = (np.abs(l) < 10) & (np.abs(x.Glat) < 10)
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
x['png'] = 'PN G' + x.PNG
x = x.merge(rz[['PNG', 'morph_class', 'GPA_deg']].rename(columns={'PNG': 'png', 'GPA_deg': 'gpa_rz'}), on='png', how='left')
x = x.merge(pd.read_csv(D / 'morph_cv.csv')[['idPNMain', 'p_B']], on='idPNMain', how='left')
x.to_csv(D / 'pa_final.csv', index=False)

base = (x.MajDiam >= 6) & (x.R_tta >= 0.8)
noshs = x.survey != 'SHS'
B = base & noshs & (x.mainClass == 'B')
E = base & noshs & (x.mainClass == 'E') & (x.q_mask < 0.85)


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float)
    n = len(pa)
    if n < 3:
        return dict(n=n)
    th = np.radians(2 * pa)
    C, S = np.cos(th).sum(), np.sin(th).sum()
    R = np.hypot(C, S) / n
    Z = n * R ** 2
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    u_ = V * np.sqrt(2 * n)
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))), V=float(V),
                V_p_parallel=float(stats.norm.sf(u_)), V_p_two=float(2 * stats.norm.sf(abs(u_))),
                frac_plane30=float(np.mean(dpa(pa, 90) < 30)))


# ---------- 1. per-survey null test on stellar and round PNe ----------
out['null_by_survey'] = {}
for cl in ['S', 'R']:
    for sv in ['IPHAS', 'VPHAS', 'SHS']:
        s = x[(x.mainClass == cl) & (x.survey == sv) & (x.MajDiam.notna())]
        r_ = circ(s.pa)  # equatorial axes: an instrumental preference shows in this frame
        out['null_by_survey'][f'{cl}|{sv}'] = {k: r_[k] for k in ['n', 'mean', 'R', 'rayleigh_p'] if k in r_}
for sv in ['IPHAS', 'VPHAS', 'SHS']:
    s = x[base & (x.survey == sv)]
    r_ = circ(s.pa)
    out['null_by_survey'][f'reliable|{sv}'] = {k: r_[k] for k in ['n', 'mean', 'R', 'rayleigh_p'] if k in r_}
# pixel grid check: fourfold harmonic of equatorial axes
th4 = np.radians(4 * x[base].pa.dropna())
R4 = np.hypot(np.cos(th4).mean(), np.sin(th4).mean()); n4 = len(th4)
out['pixel_grid_R4'] = dict(n=int(n4), R4=float(R4), p=float(np.exp(-n4 * R4 ** 2)))

# ---------- 2. alignment samples, identical cuts ----------
samples = {
    'RZ13 bipolar, visual, all': rz[rz.morph_class == 'bipolar'].GPA_deg,
    'RZ13 polar, visual, all': rz[rz.morph_class == 'polar'].GPA_deg,
    'RZ13 non-polar, visual, all': rz[rz.morph_class == 'non-polar'].GPA_deg,
    'RZ13 bipolar, network, no cuts': x[x.morph_class == 'bipolar'].gpa,
    'RZ13 bipolar, network, cuts': x[(x.morph_class == 'bipolar') & base & noshs].gpa,
    'RZ13 bipolar, network, cuts, never trained': x[(x.morph_class == 'bipolar') & base & noshs & ~x.trained].gpa,
    'RZ13 bipolar, visual, same objects as cuts': x[(x.morph_class == 'bipolar') & base & noshs].gpa_rz,
    'bulge region B all': x[B & x.bulge_sky].gpa,
    'bulge region B not in RZ13': x[B & x.bulge_sky & x.morph_class.isna()].gpa,
    'bulge region E': x[E & x.bulge_sky].gpa,
    'outside bulge region B': x[B & ~x.bulge_sky].gpa,
    'outside bulge region E': x[E & ~x.bulge_sky].gpa,
    'outside bulge region B, SHS only': x[base & (x.survey == 'SHS') & (x.mainClass == 'B') & ~x.bulge_sky].gpa,
    'outside bulge region E, SHS only': x[base & (x.survey == 'SHS') & (x.mainClass == 'E') & (x.q_mask < 0.85) & ~x.bulge_sky].gpa,
    'outside bulge region B, network P(B) >= 0.7': x[B & ~x.bulge_sky & (x.p_B >= 0.7)].gpa,
    'bulge region B, network P(B) >= 0.7': x[B & x.bulge_sky & (x.p_B >= 0.7)].gpa,
    'outside bulge region B, thin or thick disc': x[B & ~x.bulge_sky & x.Component.isin(['Thin Disk', 'Thick Disk'])].gpa,
}
out['samples'] = {k: circ(v) for k, v in samples.items()}
out['samples_V107'] = {k: circ(v, 107.5)['V_p_parallel'] for k, v in samples.items() if len(pd.Series(v).dropna()) >= 3}

# ---------- 3. two sample test: RZ13 bipolars (cuts) vs new bulge region bipolars ----------
a = x[(x.morph_class == 'bipolar') & base & noshs].gpa.dropna().values
bnew = x[B & x.bulge_sky & x.morph_class.isna()].gpa.dropna().values
Vf = lambda p: np.mean(np.cos(np.radians(2 * p) - np.pi))
dV = Vf(a) - Vf(bnew)
pool = np.r_[a, bnew]
perm = [Vf(q[:len(a)]) - Vf(q[len(a):]) for q in (rng.permutation(pool) for _ in range(20000))]
out['two_sample_V'] = dict(n_rz=len(a), n_new=len(bnew), V_rz=float(Vf(a)), V_new=float(Vf(bnew)),
                           p_one_sided=float(np.mean(np.array(perm) >= dV)))

# ---------- 4. Tan et al. (2023) binaries ----------
tan = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'tan2023_bulge_binaries.csv')
tan = tan[~tan.note.fillna('').str.contains('excluded')]  # M 1-34, a suspected binary excluded by Tan et al.
tj = tan.merge(x, left_on='PNG', right_on='png', how='left', suffixes=('', '_h'))
tj['d_gpa'] = dpa(tj.gpa, tj.GPA_deg)
tj[['PNG', 'Name', 'subsample', 'GPA_deg', 'gpa', 'd_gpa', 'R_tta', 'MajDiam', 'survey', 'mainClass', 'trained']].to_csv(D / 'tan_comparison.csv', index=False)
tok = tj[tj.GPA_deg.notna() & tj.gpa.notna()]
cut = tok[(tok.MajDiam >= 4) & (tok.R_tta >= 0.8)]
out['tan'] = dict(n_with_gpa=len(tok), median_d=float(tok.d_gpa.median()), n_within10=int((tok.d_gpa <= 10).sum()),
                  cut_n=len(cut), cut_median_d=float(cut.d_gpa.median()) if len(cut) else None,
                  network=circ(tok.gpa, 107.5), network_cut=circ(cut.gpa, 107.5), visual=circ(tok.GPA_deg, 107.5))

# ---------- 5. known close binaries (Jones compilation) ----------
bj = pd.read_csv(ROOT / 'refs' / 'binaries' / 'bcspn_jones.csv').dropna(subset=['idPNMain'])
bj['idPNMain'] = bj.idPNMain.astype(int)
xb = x.merge(bj[['idPNMain', 'candidate', 'period_d']], on='idPNMain')
xb_ok = xb[(xb.MajDiam >= 6) & (xb.R_tta >= 0.8) & (xb.survey != 'SHS') & (xb.mainClass != 'R') & (xb.mainClass != 'S')]
out['binaries'] = dict(n_matched=len(xb), n_cuts=len(xb_ok),
                       outside_bulge=circ(xb_ok[~xb_ok.bulge_sky].gpa), bulge_region=circ(xb_ok[xb_ok.bulge_sky].gpa),
                       outside_bulge_confirmed=circ(xb_ok[~xb_ok.bulge_sky & ~xb_ok.candidate].gpa))
xb_ok.to_csv(D / 'binaries_sample.csv', index=False)

# ---------- 6. injection: RZ13 strength (cuts) and Tan-like, fine grid ----------
resid_all = wrap(cv.cnn_pa - cv.EPA).dropna()
mres = (cv.mainClass == 'B') & (cv.MajDiam >= 6) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
resid = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
out['resid'] = dict(n=len(resid), E_cos2=float(np.mean(np.cos(np.radians(2 * resid)))))


def inject(n, Vobs, R_al, mu, err, grid, nsim=3000):
    sig2 = np.sqrt(-2 * np.log(R_al))
    res = {}
    for f in grid:
        Vs = []
        for _ in range(nsim):
            na = rng.binomial(n, f)
            pa = np.r_[mu + np.degrees(rng.normal(0, sig2, na)) / 2, rng.uniform(0, 180, n - na)]
            if err is not None:
                pa = pa + rng.choice(err, n)
            Vs.append(np.mean(np.cos(np.radians(2 * pa) - np.pi)))
        Vs = np.array(Vs)
        res[round(float(f), 3)] = dict(P_le_obs=float(np.mean(Vs <= Vobs)),
                                       power=float(np.mean(stats.norm.sf(Vs * np.sqrt(2 * n)) < 0.05)))
    up = next((f for f in sorted(res) if res[f]['P_le_obs'] < 0.05), None)
    return dict(grid=res, f95=up)


grid = np.round(np.arange(0, 0.62, 0.02), 2)
R_rz_cut = out['samples']['RZ13 bipolar, network, cuts']['R']
inj = {}
for sname in ['outside bulge region B', 'bulge region B not in RZ13']:
    s = out['samples'][sname]
    inj[sname] = {
        'RZ13 network concentration, errors': inject(s['n'], s['V'], max(R_rz_cut, 0.05), 90, resid, grid),
        'Tan like R 0.83 about 107.5, errors': inject(s['n'], s['V'], 0.83, 107.5, resid, grid),
        'Tan like R 0.83 about 107.5, no errors': inject(s['n'], s['V'], 0.83, 107.5, None, grid),
    }
out['injection'] = {k: {kk: dict(f95=vv['f95']) for kk, vv in v.items()} for k, v in inj.items()}
out['injection_grids'] = {k: {kk: vv['grid'] for kk, vv in v.items()} for k, v in inj.items()}

# ---------- 7. physical radii from the Erzincan et al. distances ----------
gr = {'RZ13 bipolar cuts': x[(x.morph_class == 'bipolar') & base & noshs], 'new bulge region B': x[B & x.bulge_sky & x.morph_class.isna()]}
out['phys_size'] = {k: dict(n=len(g), n_dist=int(g.AngDiam_pc.notna().sum()), median_diam_pc=float(g.AngDiam_pc.median()),
                            median_majdiam=float(g.MajDiam.median()), median_abs_b=float(g.Glat.abs().median()),
                            frac_dist_gt5kpc=float((g.rgeo_kpc > 5).mean())) for k, g in gr.items()}
a_, b_ = gr['RZ13 bipolar cuts'].AngDiam_pc.dropna(), gr['new bulge region B'].AngDiam_pc.dropna()
out['phys_size']['MW_p'] = float(stats.mannwhitneyu(a_, b_).pvalue) if len(a_) > 2 and len(b_) > 2 else None

# ---------- 8. multiple testing: two sided V over the new samples ----------
new = ['bulge region B all', 'bulge region B not in RZ13', 'bulge region E', 'outside bulge region B', 'outside bulge region E']
pmin = min(out['samples'][k]['V_p_two'] for k in new)
out['multiple'] = dict(samples=new, smallest_two_sided=float(pmin), sidak=float(1 - (1 - pmin) ** len(new)))

out['counts'] = dict(alignment_B_bulge=int((B & x.bulge_sky).sum()), alignment_B_out=int((B & ~x.bulge_sky).sum()),
                     trained_in_B_out=int((B & ~x.bulge_sky & x.trained).sum()),
                     catalogue_reliable_nonround=int((base & x.mainClass.isin(['B', 'E', 'I', 'A'])).sum()))
json.dump(out, open(D / 'rev1.json', 'w'), indent=1, default=float)
for k in ['null_by_survey', 'pixel_grid_R4', 'two_sample_V', 'tan', 'binaries', 'resid', 'injection', 'phys_size', 'multiple', 'counts']:
    print(k, json.dumps(out[k], default=float)[:900])
for k, v in out['samples'].items():
    print(f'{k:50s}', {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items() if kk in ['n', 'mean', 'R', 'rayleigh_p', 'V_p_parallel', 'V_p_two']})
