"""Compare the HASH morphological classes (ERBIAS scheme, Parker et al. 2006) with the classification network.

1. Confusion matrix (out of fold) overall, by size, by survey and by HASH sub-class letters.
2. External reference: the high resolution classes of Rees & Zijlstra (2013; bipolar, polar, non polar) and of
   Corradi et al. (1998; B1, B2 bipolar, E1, E2 elliptical). For each, the fraction called B by HASH and by the
   network, and the area under the ROC curve of the network bipolar probability.
3. Gallery of the main disagreements. Output: data/class_comparison.json, paper/figures/class_disagree.pdf
"""
import json, re, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'pdf.fonttype': 42})
from astropy.io import fits
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CL = ['R', 'E', 'B', 'S']
mc = pd.read_csv(ROOT / 'data' / 'morph_cv.csv')
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['idPNMain'] = m.idPNMain.astype(int); m['png'] = 'PN G' + m.PNG
m['nkey'] = m.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
mc = mc.merge(m[['idPNMain', 'subClass', 'png', 'nkey']], on='idPNMain', how='left')
out = {}


def summary(s):
    ct = pd.crosstab(s.mainClass, s.cnn_class).reindex(index=CL, columns=CL, fill_value=0)
    agree = float((s.mainClass == s.cnn_class).mean())
    rec = {c: float(ct.loc[c, c] / max(ct.loc[c].sum(), 1)) for c in CL}
    prec = {c: float(ct.loc[c, c] / max(ct[c].sum(), 1)) for c in CL}
    return dict(n=len(s), agreement=agree, recall_of_HASH_class=rec, precision=prec, confusion=ct.to_dict())


out['all'] = summary(mc)
out['by_size'] = {f'{lo}-{hi}': summary(mc[(mc.MajDiam >= lo) & (mc.MajDiam < hi)])['agreement']
                  for lo, hi in [(0, 5), (5, 10), (10, 20), (20, 40), (40, 1e4)]}
out['by_survey'] = {sv: summary(mc[mc.survey == sv])['agreement'] for sv in ['IPHAS', 'VPHAS', 'SHS']}
# HASH sub-class letters: a asymmetry, m multiple shells, p point symmetry, r ring, s internal structure
sub = mc.subClass.fillna('')
out['by_subclass_letter'] = {}
for L in 'amprs':
    has = sub.str.contains(L)
    for cl in ['B', 'E']:
        s = mc[(mc.mainClass == cl) & has]
        if len(s) >= 10:
            out['by_subclass_letter'][f'HASH {cl} with {L}'] = dict(n=len(s), called_B=float((s.cnn_class == 'B').mean()),
                                                                     called_E=float((s.cnn_class == 'E').mean()), called_R=float((s.cnn_class == 'R').mean()))
# P(B) by HASH class
out['median_pB_by_HASH_class'] = mc.groupby('mainClass').p_B.median().to_dict()

# external high resolution classes
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['nkey'] = co.Name.map(lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower()))
ext = {}
j = mc.merge(rz[['PNG', 'morph_class']], left_on='png', right_on='PNG')
for k in ['bipolar', 'polar', 'non-polar']:
    s = j[j.morph_class == k]
    ext[f'RZ13 {k}'] = dict(n=len(s), HASH_B=float((s.mainClass == 'B').mean()), network_B=float((s.cnn_class == 'B').mean()),
                            median_pB=float(s.p_B.median()))
yb = (j.morph_class == 'bipolar').astype(int)
ext['RZ13 AUC bipolar vs rest'] = dict(network_pB=float(roc_auc_score(yb, j.p_B)),
                                       HASH_B_as_score=float(roc_auc_score(yb, (j.mainClass == 'B').astype(int))))
jc = mc.merge(co[['nkey', 'class']], on='nkey')
jc['cbip'] = jc['class'].str.startswith('B')
for k, s in [('Corradi bipolar (B1, B2)', jc[jc.cbip]), ('Corradi elliptical (E1, E2)', jc[~jc.cbip])]:
    ext[k] = dict(n=len(s), HASH_B=float((s.mainClass == 'B').mean()), network_B=float((s.cnn_class == 'B').mean()),
                  median_pB=float(s.p_B.median()))
ext['Corradi AUC bipolar vs elliptical'] = dict(network_pB=float(roc_auc_score(jc.cbip.astype(int), jc.p_B)),
                                                HASH_B_as_score=float(roc_auc_score(jc.cbip.astype(int), (jc.mainClass == 'B').astype(int))))
out['external'] = ext
json.dump(out, open(ROOT / 'data' / 'class_comparison.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != 'all'}, indent=1)[:3500])
print('agreement', out['all']['agreement'], 'recall', out['all']['recall_of_HASH_class'], 'precision', out['all']['precision'])

# gallery of disagreements: HASH B called E, HASH E called B, HASH R called E (most confident, >= 10 arcsec)
pairs = [('B', 'E'), ('E', 'B'), ('R', 'E'), ('E', 'R')]
f, ax = plt.subplots(len(pairs), 5, figsize=(6.97, 1.55 * len(pairs)))
for i, (h, n_) in enumerate(pairs):
    s = mc[(mc.mainClass == h) & (mc.cnn_class == n_) & (mc.MajDiam >= 10) & (mc.MajDiam <= 200) & (mc.peak_snr > 30)].sort_values(f'p_{n_}', ascending=False).head(5)
    for jx, (_, r) in enumerate(s.iterrows()):
        a = ax[i, jx]
        hh = fits.open(ROOT / 'data' / 'cutouts' / r.file); d = hh[0].data.astype(float)
        nn = d.shape[0]; c = nn // 2; w = int(0.75 * r.MajDiam / r.pixscale) + 3
        d = d[max(c - w, 0):c + w, max(c - w, 0):c + w]
        lo, hi = np.nanpercentile(d, [2, 99.5])
        a.imshow(np.arcsinh(np.clip((d - lo) / (hi - lo), 0, 1) * 8), origin='lower', cmap='gray_r')
        a.set_xticks([]); a.set_yticks([])
        a.set_title(f"{r.Name}\nHASH {h}, network {n_} ({r[f'p_{n_}']:.2f})", fontsize=6.5)
    for jx in range(len(s), 5):
        ax[i, jx].axis('off')
plt.tight_layout(); plt.savefig(ROOT / 'paper' / 'figures' / 'class_disagree.pdf'); plt.close()
