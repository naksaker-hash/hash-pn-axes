"""ERBIAS confusion figure of the classification network. Rows are the HASH main classes in the ERBIAS order
(E, R, B, I, A, S) and columns the network classes (E, R, B, S). R, E, B and S rows are the out of fold
predictions of the cross validation; I and A rows come from the network trained on all four classes (morph_IA.csv),
since these classes were never used in training. Panel b gives the agreement against the HASH major diameter.
Output: paper_mlst/figures/class_matrix.pdf, data/class_matrix.json
"""
import json, pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
cv = pd.read_csv(D / 'morph_cv.csv'); ia = pd.read_csv(D / 'morph_IA.csv')
rows, cols = ['E', 'R', 'B', 'I', 'A', 'S'], ['E', 'R', 'B', 'S']
names = {'E': 'Elliptical', 'R': 'Round', 'B': 'Bipolar', 'I': 'Irregular', 'A': 'Asymmetric', 'S': 'Stellar'}
allp = pd.concat([cv[['mainClass', 'cnn_class', 'MajDiam']], ia[['mainClass', 'cnn_class', 'MajDiam']]], ignore_index=True)
C = pd.crosstab(allp.mainClass, allp.cnn_class).reindex(index=rows, columns=cols, fill_value=0)
F = C.div(C.sum(axis=1), axis=0)
bins = [0, 5, 10, 20, 40, 1e9]; lab = ['<5', '5–10', '10–20', '20–40', '>40']
cv['bin'] = pd.cut(cv.MajDiam, bins, right=False, labels=lab)
agree = cv.groupby('bin', observed=False).apply(lambda g: (g.cnn_class == g.mainClass).mean())
nb = cv.groupby('bin', observed=False).size()
out = dict(counts=C.to_dict(orient='index'), fractions=F.round(4).to_dict(orient='index'),
           n_rows={r: int(C.loc[r].sum()) for r in rows},
           agreement_by_size={k: float(v) for k, v in agree.items()}, n_by_size={k: int(v) for k, v in nb.items()},
           diag={r: float(F.loc[r, r]) for r in cols})
json.dump(out, open(D / 'class_matrix.json', 'w'), indent=1)

plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9, 'pdf.fonttype': 42})
f, (a, b) = plt.subplots(1, 2, figsize=(6.97, 2.9), gridspec_kw=dict(width_ratios=[1.25, 1]))
im = a.imshow(F.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i, r in enumerate(rows):
    for j, c in enumerate(cols):
        v = F.values[i, j]
        a.text(j, i, f'{100 * v:.0f}%\n({C.values[i, j]})', ha='center', va='center', fontsize=7,
               color='white' if v > 0.55 else 'black')
a.axhline(2.5, color='0.3', lw=0.8, ls='--'); a.axhline(4.5, color='0.3', lw=0.8, ls='--')
a.set_xticks(range(4)); a.set_xticklabels([names[c] for c in cols])
a.set_yticks(range(6)); a.set_yticklabels([f'{names[r]} ({C.loc[r].sum()})' for r in rows])
a.set_xlabel('Network class'); a.set_ylabel('HASH class')
a.set_title('(a) HASH class against network class', fontsize=9)
plt.colorbar(im, ax=a, fraction=0.046, pad=0.03, label='Fraction of HASH class')
x = np.arange(len(lab))
b.bar(x, agree.values, color='#2a78d6', width=0.6)
for xi, v, n in zip(x, agree.values, nb.values):
    b.text(xi, v + 0.02, f'{n}', ha='center', fontsize=7)
b.set_xticks(x); b.set_xticklabels(lab); b.set_ylim(0, 1)
b.set_xlabel('HASH major diameter (arcsec)'); b.set_ylabel('Agreement with HASH class')
b.set_title('(b) Agreement against size', fontsize=9)
for s in ['top', 'right']:
    b.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(ROOT / 'paper_mlst' / 'figures' / 'class_matrix.pdf'); plt.close()
print(C); print(F.round(2)); print(agree.round(3).to_dict(), nb.to_dict())
