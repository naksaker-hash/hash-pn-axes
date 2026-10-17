"""Revision 4 retraining results (referee round 4). Run after run_revision4.sh. Output: data/rev4_training.json

a. cross validation with the MASH labels or the suspect labels left out of training, scored on all 350
b. SHS preference of the stellar PNe and alignment of the bipolars outside the bulge region with the final network
   trained without the MASH labels (out of fold axes of the same run for labelled PNe)
c. scatter of the catalogue axes over three final networks with different seeds, and alignment per seed
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


x = pd.read_csv(D / 'pa_final.csv')
prov = pd.read_csv(D / 'label_provenance.csv')[['idPNMain', 'label_source', 'suspect_zero']]
base = pd.read_csv(D / 'pa_cnn_cv.csv').merge(prov, on='idPNMain')
out = {}

# ---------- a. label sensitivity ----------
for tag in ['_nomash', '_nozero']:
    c = pd.read_csv(D / f'pa_cnn_cv{tag}.csv').merge(prov, on='idPNMain')
    c['d'] = dpa(c.cnn_pa, c.EPA)
    r = {}
    for nm, s in [('all', c), ('B', c[c.mainClass == 'B']), ('E', c[c.mainClass == 'E']),
                  ('MASH labels', c[c.label_source == 'MASH']), ('not MASH', c[c.label_source != 'MASH']),
                  ('not suspect', c[~c.suspect_zero])]:
        r[nm] = dict(n=len(s), median=float(s.d.median()), gt60=float((s.d > 60).mean()))
    for nm, s in [('base, MASH labels', base[base.label_source == 'MASH']), ('base, not MASH', base[base.label_source != 'MASH']),
                  ('base, not suspect', base[~base.suspect_zero])]:
        r[nm] = dict(n=len(s), median=float(s.d_cnn.median()), gt60=float((s.d_cnn > 60).mean()))
    r['median diff to base axes'] = float(dpa(c.set_index('idPNMain').cnn_pa, base.set_index('idPNMain').cnn_pa).median())
    out[f'cv{tag}'] = r

# ---------- b. final network without the MASH labels ----------
nm = pd.read_csv(D / 'pa_cnn_all_nomash.csv')[['idPNMain', 'cnn_pa', 'cnn_R']]
oof = pd.read_csv(D / 'pa_cnn_cv_nomash.csv')[['idPNMain', 'cnn_pa', 'cnn_R']].rename(columns={'cnn_pa': 'oof', 'cnn_R': 'oofR'})
y = x.merge(nm.rename(columns={'cnn_pa': 'pa_nm', 'cnn_R': 'R_nm'}), on='idPNMain').merge(oof, on='idPNMain', how='left')
y['pa_nm'] = np.where(y.oof.notna(), y.oof, y.pa_nm); y['R_nm'] = np.where(y.oof.notna(), y.oofR, y.R_nm)
y['gpa_nm'] = (y.pa_nm + y.psi) % 180
st = y[(y.mainClass == 'S') & y.MajDiam.notna()]
b = {}
for sv in ['SHS', 'VPHAS', 'IPHAS']:
    b[f'stellar {sv}, base'] = circ(st[st.survey == sv].pa, 113)
    b[f'stellar {sv}, without MASH labels'] = circ(st[st.survey == sv].pa_nm, 113)
sel = (y.MajDiam >= 6) & (y.R_nm >= 0.8) & (y.survey != 'SHS') & (y.mainClass == 'B') & ~y.bulge_sky
b['outside bulge region B, without MASH labels'] = circ(y[sel].gpa_nm)
sel_rz = (y.MajDiam >= 6) & (y.R_nm >= 0.8) & (y.survey != 'SHS') & (y.morph_class == 'bipolar')
b['RZ13 selected bipolars, without MASH labels'] = circ(y[sel_rz].gpa_nm)
rel = (y.MajDiam >= 6) & (y.R_tta >= 0.8) & y.mainClass.isin(['B', 'E', 'I', 'A'])
b['reliable, median diff to base'] = float(dpa(y[rel].pa_nm, y[rel].pa).median())
out['nomash_full'] = b

# ---------- c. seed scatter of the catalogue axes ----------
s0 = pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']]
s1 = pd.read_csv(D / 'pa_cnn_all_s1.csv')[['idPNMain', 'cnn_pa']].rename(columns={'cnn_pa': 'p1'})
s2 = pd.read_csv(D / 'pa_cnn_all_s2.csv')[['idPNMain', 'cnn_pa']].rename(columns={'cnn_pa': 'p2'})
z = x.merge(s0.rename(columns={'cnn_pa': 'p0'}), on='idPNMain').merge(s1, on='idPNMain').merge(s2, on='idPNMain')
un = z[(z.MajDiam >= 6) & (z.R_tta >= 0.8) & z.mainClass.isin(['B', 'E', 'I', 'A']) & ~z.trained]
mx = np.max(np.c_[dpa(un.p0, un.p1), dpa(un.p0, un.p2), dpa(un.p1, un.p2)], axis=1)
ang = np.radians(2 * un[['p0', 'p1', 'p2']].values)
Rs = np.hypot(np.cos(ang).mean(1), np.sin(ang).mean(1))
cstd = np.degrees(np.sqrt(-2 * np.log(np.clip(Rs, 1e-9, 1)))) / 2
c = dict(n=len(un), median_pairwise=float(np.median(np.r_[dpa(un.p0, un.p1), dpa(un.p0, un.p2), dpa(un.p1, un.p2)])),
         median_circ_std=float(np.median(cstd)), frac_max_gt20=float((mx > 20).mean()), frac_max_gt45=float((mx > 45).mean()))
ob = z[(z.MajDiam >= 6) & (z.R_tta >= 0.8) & (z.survey != 'SHS') & (z.mainClass == 'B') & ~z.bulge_sky & ~z.trained]
for k in ['p0', 'p1', 'p2']:
    c[f'outside bulge region B unlabelled, seed {k}'] = circ((ob[k] + ob.psi) % 180)
out['seed_scatter'] = c

json.dump(out, open(D / 'rev4_training.json', 'w'), indent=1)
for k, v in out.items():
    print(k, json.dumps(v)[:1400])
