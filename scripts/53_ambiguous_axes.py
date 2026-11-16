"""Ambiguous and axisless nebulae (PASP referee, 2026-11-16). CPU only. Output: data/ambiguous_axes.json, data/ambiguous_axes.csv
a. ambiguity without visual labels: spread of the four moment axes (classical, segmentation, low isophote, flux
   weighted); network and moment errors against the visual axis for consistent and ambiguous nebulae, five seeds
b. waist dominated bipolars: classical moment more than 45 deg from the visual axis
c. visual disagreement: disc PNe measured independently by Weidmann & Diaz (2008, not copied from Corradi et al.)
   and Corradi et al. (1998), network distance to each where they differ by more than 20 deg
d. axisless nebulae: share with R_TTA >= 0.8 per HASH class and Rayleigh test of the stellar and round GPAs
"""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
rng = np.random.default_rng(51)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(np.asarray(a, float) - np.asarray(b, float)))
seeds = [pd.read_csv(D / f) .set_index('idPNMain') for f in ('pa_cnn_cv.csv', 'pa_cnn_cv_s1.csv', 'pa_cnn_cv_s2.csv', 'pa_cnn_cv_s3.csv', 'pa_cnn_cv_s4.csv')]
cv = seeds[0]
err_net = np.mean([dpa(s.loc[cv.index, 'cnn_pa'], cv.EPA) for s in seeds], axis=0)  # seed averaged error per PN
cv = cv.assign(err_net=err_net, err_mom=dpa(cv.pa_eq, cv.EPA))
bl = pd.read_csv(D / 'baselines_pa.csv').set_index('idPNMain')
cv = cv.join(bl[['pa_low', 'pa_flux', 'dl_pa_eq']])
out = {}


def summ(sel):
    s = cv[sel]
    return dict(n=int(len(s)), net_median=float(np.median(s.err_net)), net_gt60=float(np.mean(s.err_net > 60)),
                mom_median=float(np.median(s.err_mom)), mom_gt60=float(np.mean(s.err_mom > 60)),
                best_mom_median=float(np.median(np.min([dpa(s[c], s.EPA) for c in ('pa_eq', 'pa_low', 'pa_flux', 'dl_pa_eq')], axis=0))))


# a. spread of the moment axes: largest pairwise difference of the four, needs all four
m = cv[['pa_eq', 'dl_pa_eq', 'pa_low', 'pa_flux']].dropna()
cols = m.columns
spread = pd.Series(np.max([dpa(m[a], m[b]) for i, a in enumerate(cols) for b in cols[i + 1:]], axis=0), index=m.index)
cv['spread'] = spread
has = cv.spread.notna()
out['moment_spread'] = {}
for cl in ('all', 'B', 'E'):
    c = has if cl == 'all' else has & (cv.mainClass == cl)
    out['moment_spread'][cl] = dict(consistent_lt20=summ(c & (cv.spread < 20)), ambiguous_gt45=summ(c & (cv.spread > 45)))
# b. waist dominated bipolars
out['waist_bipolars'] = dict(classical_gt45=summ((cv.mainClass == 'B') & (cv.err_mom > 45)),
                             classical_le45=summ((cv.mainClass == 'B') & (cv.err_mom <= 45)))
# c. visual disagreement on the same disc PNe
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv'); w['k'] = w.PNG_from_name.fillna(w.PNG_approx_from_lb)
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['k'] = co.PNG_approx_from_lb
yy = w[w.source != 'CAM98'].merge(co, on='k', suffixes=('_w', '_c')).drop_duplicates('k').set_index('k')
hm = pd.read_csv(D / 'hash_master.csv').set_index('idPNMain')
lab = cv.copy(); lab['k'] = 'PN G' + hm.loc[lab.index, 'PNG'].astype(str)
sub = lab[lab.k.isin(yy.index)]
W = sub.EPA.values; C = yy.loc[sub.k, 'EPA_deg_c'].values
N = [s.loc[sub.index, 'cnn_pa'].values for s in seeds]
dWC = dpa(W, C); dis = dWC > 20
nW = np.mean([dpa(n, W) for n in N], axis=0); nC = np.mean([dpa(n, C) for n in N], axis=0)
out['visual_disagreement'] = dict(n_same=int(len(sub)), n_disagree=int(dis.sum()), W_vs_C_median_disagree=float(np.median(dWC[dis])) if dis.any() else None,
                                  net_closer_to_W=int(np.sum(nW[dis] < nC[dis])), net_closer_to_C=int(np.sum(nC[dis] < nW[dis])),
                                  net_vs_W_median=float(np.median(nW[dis])) if dis.any() else None, net_vs_C_median=float(np.median(nC[dis])) if dis.any() else None,
                                  names=list(sub.Name.values[dis]))
# d. axisless nebulae in the whole catalogue
cat = pd.read_csv(D / 'catalogue_pa.csv')
cat = cat[cat.HASH_MajDiam >= 2]
d = {}
for cl in ('B', 'E', 'R', 'S'):
    s = cat[cat.HASH_class == cl]
    d[cl] = dict(n=int(len(s)), frac_RTTA_ge08=float(np.mean(s.R_TTA >= 0.8)), median_RTTA=float(np.median(s.R_TTA)))
out['axis_presence'] = d
for cl in ('S', 'R'):
    s = cat[(cat.HASH_class == cl) & (cat.R_TTA >= 0.8) & (cat.survey != 'SHS')]
    z = np.exp(2j * np.radians(s.GPA_cnn.values)); R = abs(z.mean()); n = len(s)
    out['axis_presence'][cl + '_kept_nonSHS'] = dict(n=int(n), R=float(R), rayleigh_p=float(np.exp(-n * R ** 2)))
cv[['Name', 'mainClass', 'EPA', 'spread', 'err_net', 'err_mom']].to_csv(D / 'ambiguous_axes.csv')
json.dump(out, open(D / 'ambiguous_axes.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
