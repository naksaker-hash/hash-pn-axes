"""Values added after the claim check of the MLST version. Output: data/claim_check.json

a. cutouts with an r band but no usable continuum scale factor k (fewer than 30 star pixels or k <= 0), whose
   network input repeats the Halpha band in all three channels
b. median HASH major diameter of the 84 Rees & Zijlstra (2013) PNe scored as unseen (no HASH PA, D >= 4 arcsec)
c. fraction passing R_TTA >= 0.8 among labelled and unlabelled PNe of at least 6 arcsec in the catalogue classes
   (B, E, I, A), and the number of cross validated PNe with a HASH diameter in Fig. 2b
"""
import json, re, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
out = {}

pl = pd.read_csv(D / 'pseudolabels.csv')
cov = pd.read_csv(D / 'hash_coverage.csv') if (D / 'hash_coverage.csv').exists() else None
from astropy.io import fits
has_r = []
for f in pl.file:
    with fits.open(D / 'cutouts' / f) as h:
        has_r.append(len(h) > 1 and h[1].data is not None)
pl['has_r'] = has_r
nok = pl[pl.has_r & pl.kcont.isna()]
out['n_with_r'] = int(pl.has_r.sum()); out['n_no_k'] = int(len(nok))
out['n_no_k_by_survey'] = nok.survey.value_counts().to_dict()

m = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['EPA'] = pd.to_numeric(m.EPA, errors='coerce'); m['MajDiam'] = pd.to_numeric(m.MajDiam, errors='coerce')
m['png'] = 'PN G' + m.PNG; m['idPNMain'] = m.idPNMain.astype(int)
cnn = pd.read_csv(D / 'pa_cnn_all.csv')[['idPNMain', 'cnn_pa']]
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
a = m.merge(cnn, on='idPNMain', how='left').merge(rz[['PNG', 'EPA_deg']], left_on='png', right_on='PNG')
new = a[a.EPA.isna() & (a.MajDiam >= 4) & a.cnn_pa.notna()]
out['rz13_unseen_n'] = int(len(new)); out['rz13_unseen_median_majdiam'] = float(new.MajDiam.median())
out['rz13_all_in_hash_n'] = int(len(a)); out['rz13_all_in_hash_median_majdiam'] = float(a.MajDiam.median())
px = pd.read_csv(D / 'pa_final.csv'); big = px[(px.MajDiam >= 6) & px.mainClass.isin(list('BEIA'))]
out['pass08_labelled_BEIA'] = float((big[big.EPA.notna()].R_tta >= 0.8).mean())
out['pass08_unlabelled_BEIA'] = float((big[big.EPA.isna()].R_tta >= 0.8).mean())
out['fig2b_n'] = int(sum(json.load(open(D / 'class_matrix.json'))['n_by_size'].values()))
json.dump(out, open(D / 'claim_check.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
