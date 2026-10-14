"""Build the PA catalogue (all 2368 cutouts) and the appendix sample table.
Output: data/catalogue_pa.csv, paper/tab_catalogue.tex, data/catalogue_summary.json
"""
import json, pathlib
import numpy as np, pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

ROOT = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
for c in ['DRAJ2000', 'DDECJ2000', 'MajDiam', 'MinDiam', 'EPA', 'Glon', 'Glat']:
    m[c] = pd.to_numeric(m[c], errors='coerce')
m['idPNMain'] = m.idPNMain.astype(int)
# adopted axes: out of fold for PNe with a HASH position angle, otherwise the network trained on all labels
p = pd.read_csv(ROOT / 'data' / 'pa_final.csv')[['idPNMain', 'pa', 'R_tta', 'gpa', 'trained']].rename(
    columns={'pa': 'cnn_pa', 'R_tta': 'cnn_R', 'gpa': 'gpa_cnn'})
q = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')[['idPNMain', 'survey', 'maj_mask', 'min_mask', 'q_mask', 'pa_eq', 'good']]
x = m[['idPNMain', 'PNG', 'Name', 'DRAJ2000', 'DDECJ2000', 'Glon', 'Glat', 'mainClass', 'MajDiam', 'MinDiam', 'EPA']] \
    .merge(q, on='idPNMain').merge(p, on='idPNMain')
x['flag_reliable'] = ((x.MajDiam >= 6) & (x.cnn_R >= 0.8) & x.mainClass.isin(['B', 'E', 'I', 'A'])).astype(int)
x['flag_round'] = x.mainClass.isin(['R', 'S']).astype(int)
x['flag_SHS'] = (x.survey == 'SHS').astype(int)
x['flag_out_of_fold'] = x.trained.astype(int)
cat = x.rename(columns={'idPNMain': 'HASH_ID', 'DRAJ2000': 'RA_deg', 'DDECJ2000': 'Dec_deg', 'mainClass': 'HASH_class',
                        'MajDiam': 'HASH_MajDiam', 'MinDiam': 'HASH_MinDiam', 'EPA': 'HASH_PA', 'cnn_pa': 'PA_cnn',
                        'cnn_R': 'R_TTA', 'maj_mask': 'Maj_mask', 'min_mask': 'Min_mask', 'q_mask': 'q_mask',
                        'pa_eq': 'PA_moment', 'good': 'mask_trusted', 'gpa_cnn': 'GPA_cnn'})
cols = ['HASH_ID', 'PNG', 'Name', 'RA_deg', 'Dec_deg', 'Glon', 'Glat', 'HASH_class', 'survey', 'HASH_MajDiam', 'HASH_MinDiam',
        'Maj_mask', 'Min_mask', 'q_mask', 'mask_trusted', 'PA_moment', 'PA_cnn', 'GPA_cnn', 'R_TTA', 'HASH_PA',
        'flag_reliable', 'flag_round', 'flag_SHS', 'flag_out_of_fold']
cat = cat[cols].sort_values('PNG')
cat.to_csv(ROOT / 'data' / 'catalogue_pa.csv', index=False, float_format='%.2f')

rel = cat[cat.flag_reliable == 1]
summ = dict(n_rows=len(cat), n_reliable=len(rel), n_reliable_no_hash_pa=int(rel.HASH_PA.isna().sum()), n_reliable_not_shs=int((rel.flag_SHS == 0).sum()),
            by_class=rel.HASH_class.value_counts().to_dict())
json.dump(summ, open(ROOT / 'data' / 'catalogue_summary.json', 'w'), indent=1)

f = lambda v, d=1: '' if pd.isna(v) else f'{v:.{d}f}'
rows = []
for _, r in rel.head(10).iterrows():
    rows.append(f"{r.HASH_ID} & {r.PNG} & {r.Name} & {r.HASH_class} & {r.survey.replace('VPHAS', 'VPHAS+')} & {f(r.HASH_MajDiam)} & {f(r.Maj_mask)} & "
                f"{f(r.q_mask, 2)} & {f(r.PA_moment, 0)} & {f(r.PA_cnn, 0)} & {f(r.GPA_cnn, 0)} & {f(r.R_TTA, 2)} \\\\")
tab = r"""\begin{table*}
\caption{The first ten reliable entries of the position angle catalogue. The full catalogue of 2368 PNe is available as supplementary material. Diameters are in arcsec and angles in degrees, PA east of north and GPA from Galactic north through east.}
\label{tab:catalogue}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{rllllrrrrrrr}
\hline
HASH ID & PN G & Name & Class & Survey & $D_{\mathrm{HASH}}$ & $D_{\mathrm{mask}}$ & $q_{\mathrm{mask}}$ & PA$_{\mathrm{mom}}$ & PA$_{\mathrm{net}}$ & GPA$_{\mathrm{net}}$ & $R_{\mathrm{TTA}}$\\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
(ROOT / 'paper' / 'tab_catalogue.tex').write_text(tab)
print(summ)
