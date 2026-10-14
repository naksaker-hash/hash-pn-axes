"""Evaluate DL measurements against pseudo-labels, HASH diameters, HASH EPA and HASH classes.

Usage: python 06_eval.py <run name>   (needs data/dl_<run>.csv from 05_measure.py)
Output: printed summary and data/eval_<run>.json
"""
import sys, json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
run = sys.argv[1]
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
d = pd.read_csv(ROOT / 'data' / f'dl_{run}.csv')
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['EPA'] = pd.to_numeric(m.EPA, errors='coerce')
m['idPNMain'] = m.idPNMain.astype(int)
x = t.merge(d.drop(columns=['split', 'good']), on=['idPNMain', 'file']).merge(m[['idPNMain', 'EPA']], on='idPNMain')
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)


def iou(stem):
    try:
        a = np.load(ROOT / 'data' / 'masks' / f'{stem}.npz')['mask'] > 0
        b = np.load(ROOT / 'data' / 'masks' / f'{stem}_dl.npz')['mask'] > 0
    except FileNotFoundError:
        return np.nan
    return (a & b).sum() / max((a | b).sum(), 1)


x['iou'] = [iou(f.replace('.fits', '')) for f in x.file]
res = {}
for split in ['test', 'val', 'train']:
    s = x[x.split == split]
    g = s[s.good]
    r = {'n': len(s), 'detected': float(s.dl_maj.notna().mean())}
    r['iou_good_median'] = float(g.iou.median())
    r['ratio_dl_hash_median_all'] = float((s.dl_maj / s.MajDiam).median())
    r['ratio_dl_hash_iqr_all'] = [float(v) for v in (s.dl_maj / s.MajDiam).quantile([.25, .75])]
    r['ratio_cl_hash_iqr_all'] = [float(v) for v in (s.maj_mask / s.MajDiam).quantile([.25, .75])]
    bad = s[~s.good]
    r['ratio_dl_hash_iqr_classical_failed'] = [float(v) for v in (bad.dl_maj / bad.MajDiam).quantile([.25, .5, .75])]
    e = s[s.EPA.notna() & (s.dl_maj > 6)]
    for lab, col, qcol in [('dl', 'dl_pa_eq', 'dl_q'), ('classical', 'pa_eq', 'q_mask')]:
        ee = e[e[qcol] < 0.85]
        dd = dpa(ee[col], ee.EPA)
        r[f'pa_{lab}'] = dict(n=int(dd.notna().sum()), median=float(dd.median()), frac_lt20=float((dd < 20).mean()))
    c = s[s.mainClass.isin(['R', 'E', 'B', 'S']) & s.dl_class.notna()]
    r['class_acc'] = float((c.mainClass == c.dl_class).mean())
    r['confusion'] = pd.crosstab(c.mainClass, c.dl_class).to_dict()
    res[split] = r
    print(f'== {split}: n {r["n"]} detected {r["detected"]:.3f} IoU(good) {r["iou_good_median"]:.2f}')
    print(f'   Maj/HASH median {r["ratio_dl_hash_median_all"]:.2f} IQR dl {np.round(r["ratio_dl_hash_iqr_all"],2)} classical {np.round(r["ratio_cl_hash_iqr_all"],2)}')
    print(f'   classical-failed objects: DL Maj/HASH q25/50/75 {np.round(r["ratio_dl_hash_iqr_classical_failed"],2)}')
    print(f'   PA vs EPA: DL {r["pa_dl"]}  classical {r["pa_classical"]}')
    print(f'   class acc {r["class_acc"]:.2f}')
    print(pd.crosstab(c.mainClass, c.dl_class))
json.dump(res, open(ROOT / 'data' / f'eval_{run}.json', 'w'), indent=1)
