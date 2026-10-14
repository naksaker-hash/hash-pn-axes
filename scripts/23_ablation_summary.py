"""Seed spread, ablations and test time augmentation for the axis network. Output: data/ablation.json

Medians of the absolute difference from the visual axis in the fivefold cross validation for the base run,
two further seeds, no augmentation, rotation without label transformation and the H-alpha band alone, with
bootstrap 68 per cent intervals (2000 resamples) of the base medians, and the base run with 1, 4 and 8 rotations
at prediction.
"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
rng = np.random.default_rng(5)
out = {}
for tag, name in [('', 'base'), ('_s1', 'seed 1'), ('_s2', 'seed 2'), ('_noaug', 'no augmentation'),
                  ('_rotnolabel', 'rotation without label transformation'), ('_ha', 'H-alpha only')]:
    c = pd.read_csv(D / f'pa_cnn_cv{tag}.csv')
    c['d'] = dpa(c.cnn_pa, c.EPA)
    r = {}
    for cl in ['all', 'B', 'E']:
        s = c if cl == 'all' else c[c.mainClass == cl]
        r[cl] = dict(n=len(s), median=float(s.d.median()), gt60=float((s.d > 60).mean()))
    out[name] = r
c = pd.read_csv(D / 'pa_cnn_cv.csv')
for cl in ['all', 'B', 'E']:
    s = c if cl == 'all' else c[c.mainClass == cl]
    d = dpa(s.cnn_pa, s.EPA).values
    bs = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
    out['base'][cl]['boot68'] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
for k in (1, 4):
    col = f'cnn_pa_tta{k}'
    out[f'base, {k} rotations'] = {cl: dict(median=float(dpa((c if cl == 'all' else c[c.mainClass == cl])[col],
                                                             (c if cl == 'all' else c[c.mainClass == cl]).EPA).median()))
                                   for cl in ['all', 'B', 'E']}
seeds = [out[k]['all']['median'] for k in ['base', 'seed 1', 'seed 2']]
out['seed_range_all'] = [min(seeds), max(seeds)]
json.dump(out, open(D / 'ablation.json', 'w'), indent=1)
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {cl: (round(vv['median'], 1), vv.get('boot68')) for cl, vv in v.items()})
