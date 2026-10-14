"""Independent PA test: Rees & Zijlstra 2013 (bulge) and Corradi et al. 1998 (disc).

Only HASH objects without an EPA (so never used to train the CNN) are scored.
Also reports human-to-human agreement (Weidmann & Diaz 2008 = HASH EPA vs RZ13 / Corradi) on the
objects both measured, as the floor.
Usage: python 10_independent_pa_test.py [yolo run for mask PAs]
"""
import sys, re, json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
run = sys.argv[1] if len(sys.argv) > 1 else 'yolo26n-seg_s0'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
norm = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())

m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['idPNMain'] = m.idPNMain.astype(int)
m['png'] = 'PN G' + m.PNG; m['nkey'] = m.Name.map(norm)
cnn = pd.read_csv(ROOT / 'data' / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa', 'cnn_R']]
pl = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')[['idPNMain', 'pa_eq', 'q_mask', 'maj_mask']]
dl = pd.read_csv(ROOT / 'data' / f'dl_{run}.csv')[['idPNMain', 'dl_pa_eq', 'dl_q', 'dl_maj']]
h = m[['idPNMain', 'Name', 'mainClass', 'MajDiam', 'png', 'nkey', 'EPA']].merge(cnn, on='idPNMain', how='left') \
    .merge(pl, on='idPNMain', how='left').merge(dl, on='idPNMain', how='left')
h['MajDiam'] = pd.to_numeric(h.MajDiam, errors='coerce')

rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
a = h.merge(rz[['PNG', 'EPA_deg', 'morph_class']], left_on='png', right_on='PNG')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv')
co['nkey'] = co.Name.map(norm)
b1 = h.merge(co[['nkey', 'EPA_deg', 'class']], on='nkey')
b2 = h.merge(co[['PNG_approx_from_lb', 'EPA_deg', 'class', 'nkey']].rename(columns={'nkey': 'ck'}),
             left_on='png', right_on='PNG_approx_from_lb')
b = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain')

res = {}
for name, x in [('RZ13', a), ('Corradi98', b)]:
    r = {'matched': len(x)}
    both = x[x.EPA.notna()]
    d = dpa(both.EPA, both.EPA_deg)
    r['human_floor'] = dict(n=len(d), median=float(d.median()), gt60=float((d > 60).mean()))
    new = x[x.EPA.isna() & (x.MajDiam >= 4)]
    for lab, col in [('cnn', 'cnn_pa'), ('mask_moment_classical', 'pa_eq'), ('mask_moment_yolo', 'dl_pa_eq')]:
        d = dpa(new[col], new.EPA_deg).dropna()
        r[lab] = dict(n=len(d), median=float(d.median()), frac_lt20=float((d < 20).mean()), gt60=float((d > 60).mean()))
    for cl in ['B', 'E']:
        s = new[new.mainClass == cl]
        for lab, col in [('cnn', 'cnn_pa'), ('moment', 'pa_eq')]:
            d = dpa(s[col], s.EPA_deg).dropna()
            r[f'{lab}_{cl}'] = dict(n=len(d), median=float(d.median()), gt60=float((d > 60).mean()))
    res[name] = r
    print(name, json.dumps(r, indent=1))
json.dump(res, open(ROOT / 'data' / 'pa_independent_test.json', 'w'), indent=1)
