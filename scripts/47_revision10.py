"""Revision 10 (fifth deep review; counts use the sample of the current size cut, paper_mlst/DEEP_REVIEW5.md). CPU only. Output: data/rev10.json
a. bulge binaries by size cuts fixed in advance (20 arcsec, the largest binary of Tan et al.; below 35 arcsec, the limit
   of Rees & Zijlstra) and the compact bulge members
b. close binary share with numerator and denominator from the same population
c. expected limits of the bulge binaries with 200 realisations, a missing limit counted as 1
d. learned moment baseline over three seeds and its paired difference from the network (seed 0, same PNe)
e. probability at the lower end of the binary share; single copy error without augmentation; seed averaged rows of
   Table 1; central moment Rayleigh probability with the formula used elsewhere; VPHAS+ control"""
import json, pathlib, importlib, sys
import numpy as np, pandas as pd
from scipy import stats
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'; sys.path.insert(0, str(ROOT / 'scripts'))
rng = np.random.default_rng(101)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * (pa - mu))), axis=-1)
x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
master = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain'); master['idPNMain'] = master.idPNMain.astype(int)
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
cvres = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
err_rz = wrap(xr[xr.rz_class.notna()].gpa - xr[xr.rz_class.notna()].GPA_deg).dropna().values
out = {}


def P_of_f(pa_obs, R_al, f, pool, ns, mu=90.0):
    nn = len(pa_obs); s2 = np.sqrt(-2 * np.log(R_al)); al = rng.random((ns, nn)) < f
    pa = np.where(al, mu + np.degrees(rng.normal(0, s2, (ns, nn))) / 2, rng.uniform(0, 180, (ns, nn))) + rng.choice(pool, (ns, nn))
    return float(np.mean(Vat(pa, mu) <= Vat(pa_obs, mu)))


def limit(pa_obs, pool, R_al=0.83, ns=3000):
    c = None
    for f in np.round(np.arange(0, 1.0001, 0.05), 3):
        if P_of_f(pa_obs, R_al, f, pool, 1000) < 0.05:
            c = f; break
    if c is None:
        return 1.0
    fg = np.round(np.arange(max(0, c - 0.10), min(1, c + 0.06) + 1e-9, 0.01), 3)
    P = np.array([P_of_f(pa_obs, R_al, f, pool, ns) for f in fg]); i = np.flatnonzero(P < 0.05)
    if not len(i):
        return 1.0
    j = i[0]
    return float(fg[0]) if j == 0 else float(fg[j - 1] + (P[j - 1] - 0.05) * (fg[j] - fg[j - 1]) / (P[j - 1] - P[j]))


def p_all(pa_o, pool, ns=400_000):
    nn = len(pa_o); s2 = np.sqrt(-2 * np.log(0.83))
    pa = 90 + np.degrees(rng.normal(0, s2, (ns, nn))) / 2 + rng.choice(pool, (ns, nn))
    return float(np.mean(Vat(pa, 90.0) <= Vat(pa_o, 90.0)))


# a
bj = pd.read_csv(D / 'binaries_clean.csv').merge(master[['idPNMain', 'Component']], on='idPNMain', how='left')
bs = bj[bj.in_alignment_test & bj.bulge_sky]
a = {}
for nm, s in [('D <= 20', bs[bs.MajDiam <= 20]), ('D < 35', bs[bs.MajDiam < 35]), ('D <= 15', bs[bs.MajDiam <= 15]),
              ('compact bulge members', bs[(bs.MajDiam <= 15) & (bs.Component == 'Bulge')])]:
    z = np.exp(2j * np.radians(s.gpa.values)).mean()
    a[nm] = dict(n=len(s), cv=p_all(s.gpa.values, cvres), rz=p_all(s.gpa.values, err_rz), R=float(abs(z)), mean=float(np.degrees(np.angle(z)) / 2 % 180),
                 n_disc=int(s.Component.isin(['Thin Disk', 'Thick Disk']).sum() if s.Component.notna().any() else 0),
                 components=s.Component.value_counts().to_dict())
    print(nm, a[nm], flush=True)
m2 = x[x.Name == 'M 2-19']; a['M 2-19'] = dict(MajDiam=float(m2.MajDiam.iloc[0]), gpa=float(m2.gpa.iloc[0]))
out['bulge_size'] = a

gpa_n = x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS') & (x.mainClass == 'B') & ~x.bulge_sky].gpa.values
# b
sh = bj[bj.short & (bj.survey != 'SHS') & bj.mainClass.isin(['B', 'E', 'R', 'I', 'A'])]
pop = x[(x.survey != 'SHS') & x.mainClass.isin(['B', 'E', 'R', 'I', 'A'])]
num = float((sh.mainClass == 'B').mean()); den = float((pop.mainClass == 'B').mean())
out_ = sh[~sh.bulge_sky & (sh.MajDiam >= CUT)]; pop2 = pop[~pop.bulge_sky & (pop.MajDiam >= CUT)]
b = dict(num_all=num, n_all=len(sh), den_all=den, frac=[0.12 * num / den, 0.21 * num / den], count=[len(gpa_n) * 0.12 * num / den, len(gpa_n) * 0.21 * num / den],
         num_out=float((out_.mainClass == 'B').mean()), n_out=len(out_), den_out=float((pop2.mainClass == 'B').mean()))
b['frac_out'] = [0.12 * b['num_out'] / b['den_out'], 0.21 * b['num_out'] / b['den_out']]
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
gpa = x[base & (x.mainClass == 'B') & ~x.bulge_sky].gpa.values
b['P_at_lower_end'] = P_of_f(gpa, 0.83, 0.12 * 0.30 / 0.297, cvres, 20000)
out['binary_share'] = b; print(b, flush=True)

# c
c = {}
for nm, n_ in [('all', len(bs)), ('B', int((bs.mainClass == 'B').sum()))]:
    for en, pool in [('cv', cvres), ('rz', err_rz)]:
        fs = np.array([limit(rng.uniform(0, 180, n_) + rng.choice(pool, n_), pool) for _ in range(200)])
        c[f'{nm} {en}'] = dict(median=float(np.median(fs)), frac_none=float(np.mean(fs >= 1.0)))
        print('expected', nm, en, c[f'{nm} {en}'], flush=True)
out['expected_limits'] = c

# d
from sklearn.ensemble import HistGradientBoostingRegressor
bp = pd.read_csv(D / 'baselines_pa.csv').set_index('idPNMain'); s0 = cv.set_index('idPNMain')
X0 = s0.join(bp[['pa_low', 'pa_flux', 'dl_pa_eq']], how='left'); angs = ['pa_eq', 'pa_low', 'pa_flux', 'dl_pa_eq']; X0 = X0[X0[angs].notna().all(1)]
inv = np.column_stack([X0.q_mask.fillna(1), np.log(X0.MajDiam), np.log1p(X0.peak_snr.fillna(0)), (X0.maj_mask / X0.MajDiam).fillna(1),
                       (X0.survey == 'SHS').astype(float), (X0.survey == 'IPHAS').astype(float)]
                      + [np.cos(np.radians(2 * (X0[p] - X0[q]))) for i, p in enumerate(angs) for q in angs[i + 1:]])
A = X0[angs].values; y = X0.EPA.values; fold = X0.fold.values
preds = []
for seed in (0, 1, 2):
    r_ = np.random.default_rng(seed); pred = np.zeros(len(y))
    for f in range(5):
        tr = fold != f; te = fold == f; Xs, Ys = [], []
        for _ in range(32):
            phi = r_.uniform(0, 180, tr.sum()); fl = r_.random(tr.sum()) < 0.5
            A2 = np.where(fl[:, None], -A[tr], A[tr]) + phi[:, None]; y2 = np.where(fl, -y[tr], y[tr]) + phi
            Xs.append(np.column_stack([inv[tr], np.cos(np.radians(2 * A2)), np.sin(np.radians(2 * A2))])); Ys.append(np.column_stack([np.cos(np.radians(2 * y2)), np.sin(np.radians(2 * y2))]))
        Xs = np.vstack(Xs); Ys = np.vstack(Ys)
        mc = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(Xs, Ys[:, 0])
        ms = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=seed).fit(Xs, Ys[:, 1])
        zz = 0
        for phi in np.arange(0, 180, 22.5):
            A2 = A[te] + phi; Xt = np.column_stack([inv[te], np.cos(np.radians(2 * A2)), np.sin(np.radians(2 * A2))])
            zz = zz + np.exp(-2j * np.radians(phi)) * (mc.predict(Xt) + 1j * ms.predict(Xt))
        pred[te] = (np.degrees(np.angle(zz)) / 2) % 180
    preds.append(dpa(pred, y))
eg = np.array(preds); en = dpa(X0.cnn_pa.values, y)
blk = (X0.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0; groups = [np.flatnonzero(blk == q) for q in np.unique(blk)]
bsd = [np.median(eg[0][k]) - np.median(en[k]) for k in (np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(4000))]
out['learned_moments'] = dict(medians=[float(np.median(e)) for e in eg], network_seed0=float(np.median(en)), diff_seed0=float(np.median(eg[0]) - np.median(en)),
                              diff95=list(np.percentile(bsd, [2.5, 97.5])), gt60=[float(np.mean(e > 60)) for e in eg])
print(out['learned_moments'], flush=True)

# e
ids = cv.idPNMain.values
na = [pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids) for t in ['_noaug', '_noaug_s1', '_noaug_s2']]
out['noaug_single_copy'] = [float(np.median(dpa(s.cnn_pa_tta1.values, cv.EPA.values))) for s in na]
seeds = [pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids) for t in ['', '_s1', '_s2', '_s3', '_s4']]
E = np.mean([dpa(s.cnn_pa.values, cv.EPA.values) for s in seeds], 0)
out['seed_averaged'] = {cl: dict(n=int(mm.sum()), median=float(np.median(E[mm])), gt60=float(np.mean(E[mm] > 60)))
                        for cl, mm in [('all', np.ones(len(E), bool)), ('B', (cv.mainClass == 'B').values), ('E', (cv.mainClass == 'E').values)]}
r9 = json.load(open(D / 'rev9.json'))
zar = lambda n, R: float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n)))
cm = r9['shs_central_moment']; out['central_moment_rayleigh_zar'] = {sv: zar(cm[sv]['n'], cm[sv]['R']) for sv in ('SHS', 'IPHAS')}
json.dump(out, open(D / 'rev10.json', 'w'), indent=1, default=float); print(json.dumps({k: out[k] for k in ('noaug_single_copy', 'seed_averaged', 'central_moment_rayleigh_zar')}, indent=0))
