"""Revision 9: paired block bootstrap of the median error of the D8 steerable CNN against the ResNet18 from scratch
(three seeds each, seed averaged per object error). Adds key d8_vs_scratch to data/rev9.json."""
import json, pathlib
import numpy as np, pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'; rng = np.random.default_rng(92)
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
cv = pd.read_csv(D / 'pa_cnn_cv.csv'); ids = cv.idPNMain.values; epa = cv.EPA.values
me = lambda tags: np.nanmean([dpa(pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(ids).cnn_pa.values, epa) for t in tags], 0)
d8 = me(['_d8_s0', '_d8_s1', '_d8_s2']); sc = me(['_scratch_s0', '_scratch_s1', '_scratch_s2'])
blk = (cv.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0; groups = [np.flatnonzero(blk == q) for q in np.unique(blk)]
bs = [np.nanmedian(d8[k]) - np.nanmedian(sc[k]) for k in (np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(4000))]
r = dict(median_d8=float(np.nanmedian(d8)), median_scratch=float(np.nanmedian(sc)), diff=float(np.nanmedian(d8) - np.nanmedian(sc)), diff95=list(np.percentile(bs, [2.5, 97.5])))
j = json.load(open(D / 'rev9.json')); j['d8_vs_scratch'] = r; json.dump(j, open(D / 'rev9.json', 'w'), indent=1); print(r)
