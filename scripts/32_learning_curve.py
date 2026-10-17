"""Learning curve of the axis network: cross validation with 25, 50, 75 and 100 per cent of each training fold.
Output: data/learning_curve.json, paper_mlst/figures/learning_curve.pdf
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
rng = np.random.default_rng(9)
out = {}
for frac, tag in [(0.25, '_f25'), (0.5, '_f50'), (0.75, '_f75'), (1.0, '')]:
    c = pd.read_csv(D / f'pa_cnn_cv{tag}.csv'); d = dpa(c.cnn_pa, c.EPA).values
    b = c[c.mainClass == 'B']; db = dpa(b.cnn_pa, b.EPA).values
    bs = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
    out[str(frac)] = dict(n_train_mean=float(frac * 350 * 4 / 5), median=float(np.median(d)), boot68=[float(np.percentile(bs, 16)), float(np.percentile(bs, 84))],
                          median_B=float(np.median(db)), gt60=float((d > 60).mean()))
s = [json.load(open(D / f'pa_cnn_cv{t}.json'))['all']['cnn_median'] for t in ['', '_s1', '_s2']]
out['seed_range_full'] = [min(s), max(s)]
json.dump(out, open(D / 'learning_curve.json', 'w'), indent=1)
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9, 'pdf.fonttype': 42})
f, a = plt.subplots(figsize=(3.4, 2.5))
fr = [0.25, 0.5, 0.75, 1.0]; n = [out[str(x)]['n_train_mean'] for x in fr]
m = [out[str(x)]['median'] for x in fr]; lo = [out[str(x)]['boot68'][0] for x in fr]; hi = [out[str(x)]['boot68'][1] for x in fr]
a.fill_between(n, lo, hi, color='#2a78d6', alpha=0.2, lw=0)
a.plot(n, m, 'o-', color='#2a78d6', label='All')
a.plot(n, [out[str(x)]['median_B'] for x in fr], 's--', color='#eb6834', label='Bipolar')
a.set_xlabel('Training PNe per fold'); a.set_ylabel(r'Median $|\Delta \mathrm{PA}|$ (deg)'); a.legend(frameon=False)
plt.tight_layout(); plt.savefig(ROOT / 'paper_mlst' / 'figures' / 'learning_curve.pdf'); plt.close()
print(json.dumps(out, indent=1))
