"""Revision 8 (third deep review), small recomputations. Output: data/rev8_misc.json
a. von Mises kappa AUC and Spearman with R_TTA for the three seeds
b. paired block bootstrap of the median error of each configuration against the standard network on the same PNe
c. classification: bootstrap interval of the AUC of P_B within the HASH bipolars against Corradi et al. and the
   paired purity difference; reliability curve of P_B against the HASH bipolar class
d. attenuation of R for the reproductions with the finite n bias, and with the error pool against Rees & Zijlstra
e. known binaries: the bulge subset bipolar in the morphology column of the compilation, and the binaries with a
   HASH visual axis
f. close binary fraction among the HASH bipolars without a second purity factor"""
import json, re, pathlib, html as htmlmod
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
rng = np.random.default_rng(83)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
key = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())
out = {}


def circ(pa, mu=90.0):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); R = np.abs(np.exp(1j * th).mean()); V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, R=float(R), mean=float(np.degrees(np.angle(np.exp(1j * th).mean())) / 2 % 180), V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


# a
cv = pd.read_csv(D / 'pa_cnn_cv.csv'); a = {}
for s in (0, 1, 2):
    c = pd.read_csv(D / f'pa_cnn_cv_vm_s{s}.csv'); g = dpa(c.cnn_pa, c.EPA) > 60
    a[s] = dict(auc_kappa=float(roc_auc_score(g, -c.cnn_kappa)), auc_rtta=float(roc_auc_score(g, 1 - c.cnn_R)),
                spearman_kappa_R=float(stats.spearmanr(c.cnn_kappa, c.cnn_R).correlation))
d8R = [float(np.median(pd.read_csv(D / f'pa_cnn_cv_d8_s{s}.csv').cnn_R)) for s in (0, 1, 2)]
out['vm_kappa'] = dict(per_seed=a, auc_kappa_range=[min(v['auc_kappa'] for v in a.values()), max(v['auc_kappa'] for v in a.values())],
                       auc_rtta_range=[min(v['auc_rtta'] for v in a.values()), max(v['auc_rtta'] for v in a.values())],
                       spearman_range=[min(v['spearman_kappa_R'] for v in a.values()), max(v['spearman_kappa_R'] for v in a.values())],
                       d8_median_R_tta=d8R)
# b
ids = cv.idPNMain.values; epa = cv.EPA.values
blk = (cv.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0; groups = [np.flatnonzero(blk == q) for q in np.unique(blk)]
def mean_err(tags):
    return np.nanmean([dpa(pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids).cnn_pa.values, epa) for t in tags], 0)
std = mean_err(['', '_s1', '_s2', '_s3', '_s4']); std3 = mean_err(['', '_s1', '_s2'])
seeds = [pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids) for t in ['', '_s1', '_s2', '_s3', '_s4']]
V = np.mean([np.exp(2j * np.radians(s_.cnn_pa.values)) for s_ in seeds], 0); ens = dpa((np.degrees(np.angle(V)) / 2) % 180, epa)
cfg = {'no augmentation': mean_err(['_noaug', '_noaug_s1', '_noaug_s2']), 'Halpha alone': mean_err(['_ha', '_ha_s1', '_ha_s2']),
       'von Mises head': mean_err(['_vm_s0', '_vm_s1', '_vm_s2']), 'from scratch': mean_err(['_scratch_s0', '_scratch_s1', '_scratch_s2']),
       'D8': mean_err(['_d8_s0', '_d8_s1', '_d8_s2']), 'rotate full': mean_err(['_rotfull_s0', '_rotfull_s1', '_rotfull_s2']), 'ensemble': ens}
b = {}
for nm, e in cfg.items():
    d0 = float(np.nanmedian(e) - np.median(std3)); bs = []
    for _ in range(4000):
        pk = np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))])
        bs.append(np.nanmedian(e[pk]) - np.median(std3[pk]))
    b[nm] = dict(median_config=float(np.nanmedian(e)), median_standard_3seeds=float(np.median(std3)), diff=d0, diff95=list(np.percentile(bs, [2.5, 97.5])))
    print(nm, b[nm], flush=True)
out['paired_vs_standard'] = b
# c
mc = pd.read_csv(D / 'morph_cv.csv')
m0 = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain'); m0['idPNMain'] = m0.idPNMain.astype(int); m0['nkey'] = m0.Name.map(key)
mc = mc.merge(m0[['idPNMain', 'nkey']], on='idPNMain')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(key)
j = mc.merge(co[['nkey', 'class']], on='nkey'); hb = j[j.mainClass == 'B']; y = hb['class'].str.startswith('B').values; p = hb.p_B.values
auc = roc_auc_score(y, p); aucs = []
for _ in range(4000):
    k = rng.integers(0, len(y), len(y))
    if 0 < y[k].sum() < len(k):
        aucs.append(roc_auc_score(y[k], p[k]))
# purity of P_B >= 0.7 and of the HASH bipolar class among all PNe of Corradi et al., paired over the same objects
ya = j['class'].str.startswith('B').values; sn = j.p_B.values >= 0.7; sh = (j.mainClass == 'B').values; dpur = []
for _ in range(4000):
    k = rng.integers(0, len(ya), len(ya))
    dpur.append(ya[k][sn[k]].mean() - ya[k][sh[k]].mean())
out['classification'] = dict(n_hash_B=len(y), auc=float(auc), auc95=list(np.percentile(aucs, [2.5, 97.5])), n_all=len(ya),
                             purity_net=float(ya[sn].mean()), n_net=int(sn.sum()), purity_hash=float(ya[sh].mean()), n_hash=int(sh.sum()),
                             purity_diff=float(ya[sn].mean() - ya[sh].mean()), purity_diff95=list(np.percentile(dpur, [2.5, 97.5])))
edges = np.linspace(0, 1, 11); yb = (mc.mainClass == 'B').values; pb = mc.p_B.values; rel = []
for lo, hi in zip(edges[:-1], edges[1:]):
    s = (pb >= lo) & (pb < hi if hi < 1 else pb <= hi)
    rel.append(dict(lo=float(lo), hi=float(hi), n=int(s.sum()), mean_p=float(pb[s].mean()) if s.any() else None, frac_B=float(yb[s].mean()) if s.any() else None))
out['reliability_P_B'] = rel
print(out['classification'])
# d
x = pd.read_csv(D / 'pa_final.csv')
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
xr_all = xr[xr.rz_class.notna()]; err_rz = wrap(xr_all.gpa - xr_all.GPA_deg).dropna().values
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
Ec_cv = float(np.mean(np.cos(np.radians(2 * wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna()))))
Ec_rz = float(np.mean(np.cos(np.radians(2 * err_rz))))
rzs = xr[(xr.rz_class == 'bipolar') & (xr.MajDiam >= CUT) & (xr.R_tta >= 0.8) & (xr.survey != 'SHS')]
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg'])
att = dict(Ecos2_cv=Ec_cv, Ecos2_rz=Ec_rz)
for nm, vis, net in [('RZ13 selected', rzs.GPA_deg, rzs.gpa), ('Tan all', tan.GPA_deg, tan.gpa)]:
    n = len(vis); Rv = circ(vis)['R']; Rn = circ(net)['R']; r = dict(n=n, R_visual=Rv, R_network=Rn)
    for en, Ec in [('cv', Ec_cv), ('rz', Ec_rz)]:
        rho = Rv * Ec; r[f'rho_{en}'] = float(rho); r[f'expected_Rbar_{en}'] = float(np.sqrt(rho ** 2 + (1 - rho ** 2) / n))
    att[nm] = r
out['attenuation'] = att; print(att)
# e
h = open(ROOT / 'refs' / 'binaries' / 'bcspn.html', errors='ignore').read(); t0 = re.findall(r'<table.*?</table>', h, re.S)[0]; morph = {}
for r in re.findall(r'<tr[^>]*>(.*?)</tr>', t0, re.S):
    c = [htmlmod.unescape(re.sub('<[^>]+>', '', q)).strip() for q in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)]
    if len(c) >= 7 and re.match(r'\d+:\d+:\d', c[1] or ''):
        morph[c[0].lstrip('*').strip()] = c[6]
bj = pd.read_csv(D / 'binaries_clean.csv'); bt = bj[bj.in_alignment_test].copy(); bt['jones_morph'] = bt.name.map(morph)
bt['jones_bipolar'] = bt.jones_morph.fillna('').str.contains('bipolar|Bipolar|B', regex=True) & ~bt.jones_morph.fillna('').str.contains('–')
e = dict(morph_values=bt.jones_morph.value_counts().to_dict())
for reg, mk in [('outside', ~bt.bulge_sky), ('bulge', bt.bulge_sky)]:
    s = bt[mk]; sj = s[s.jones_bipolar]
    e[reg] = dict(n=len(s), n_jones_bipolar=len(sj), jones_bipolar=circ(sj.gpa) if len(sj) > 1 else None,
                  n_with_hash_axis=int(x.set_index('idPNMain').reindex(s.idPNMain).EPA.notna().sum()),
                  hash_axes=circ(x.set_index('idPNMain').reindex(s.idPNMain).dropna(subset=['EPA']).pipe(lambda d_: (d_.EPA + d_.psi) % 180)) if True else None)
bt[['Name', 'name', 'mainClass', 'jones_morph', 'bulge_sky', 'gpa', 'R_tta', 'MajDiam']].to_csv(D / 'rev8_binaries_morph.csv', index=False)
out['binaries'] = e; print(json.dumps(e, indent=0)[:1500])
N_OUT = int(((x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS') & (x.mainClass == 'B') & ~x.bulge_sky).sum())
# f
out['binary_fraction'] = dict(P_B_resolved=0.297, P_bin=[0.12, 0.21], P_B_given_bin=0.30, P_bin_given_B=[0.12 * 0.30 / 0.297, 0.21 * 0.30 / 0.297],
                              count_sample=[N_OUT * 0.12 * 0.30 / 0.297, N_OUT * 0.21 * 0.30 / 0.297])
json.dump(out, open(D / 'rev8_misc.json', 'w'), indent=1)
