"""Gradient boosting moment baseline against the seed averaged network error (same 346 PNe, block bootstrap), as for all other paired differences."""
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
rng = np.random.default_rng(101)
seeds = [pd.read_csv(D / f'pa_cnn_cv{t}.csv').set_index('idPNMain').reindex(X0.index) for t in ['', '_s1', '_s2', '_s3', '_s4']]
en = np.mean([dpa(q.cnn_pa.values, y) for q in seeds], 0); eb = eg.mean(0)
blk = (X0.Glon // 10).astype(int).values.copy(); blk[blk == 35] = 0; groups = [np.flatnonzero(blk == q) for q in np.unique(blk)]
bsd = [np.median(eb[k]) - np.median(en[k]) for k in (np.concatenate([groups[q] for q in rng.integers(0, len(groups), len(groups))]) for _ in range(4000))]
o = dict(n=int(len(y)), gb_medians=[float(np.median(e)) for e in eg], gb_seedavg=float(np.median(eb)), net_seedavg=float(np.median(en)),
         diff=float(np.median(eb) - np.median(en)), diff95=[float(v) for v in np.percentile(bsd, [2.5, 97.5])])
print(o); json.dump(o, open(D / 'gb_seedavg.json', 'w'), indent=1)
