"""Derived numbers quoted in the text that no other script writes. Output: data/text_numbers.json"""
import json, pathlib
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
m['MajDiam'] = pd.to_numeric(m.MajDiam, errors='coerce'); m['png'] = 'PN G' + m.PNG
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv')
j = m.merge(rz, left_on='png', right_on='PNG')
cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
al = json.load(open(ROOT / 'data' / 'alignment_cnn.json'))
new = ['bulge sky|B all', 'bulge sky|B new', 'bulge sky|E', 'disc|B', 'disc|E']
pmin = min(min(al[k]['V_p_parallel'], al[k]['V_p_perp']) for k in new)
out = dict(rz13_in_hash=len(j), rz13_median_majdiam=float(j.MajDiam.median()),
           rz13_frac_majdiam_lt10=float((j.MajDiam < 10).mean()),
           cv_median_majdiam=float(cv.MajDiam.median()),
           n_new_tests=2 * len(new), smallest_p_new=float(pmin),
           chance_min_p=float(1 - (1 - pmin) ** (2 * len(new))))
print(out)
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# HASH main class of the RZ13 PNe by RZ13 class
ct = pd.crosstab(j.morph_class, j.mainClass.fillna('?'))
out['rz13_class_crosstab'] = {k: {kk: int(vv) for kk, vv in v.items()} for k, v in ct.to_dict().items()}
nB = int(ct['B'].sum()) if 'B' in ct else 0
out['rz13_hashB_n'] = nB
out['rz13_hashB_bipolar_frac'] = float(ct.loc['bipolar', 'B'] / nB)

# (dilution test superseded by the matched two sample test in 18_revision_analyses.py)

# human to human agreement: Weidmann & Diaz own measurements vs Corradi 1998 (rows not copied from Corradi)
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv'); w['key'] = w.PNG_from_name.fillna(w.PNG_approx_from_lb)
co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co['key'] = co.PNG_approx_from_lb
y = w[w.source != 'CAM98'].merge(co, on='key', suffixes=('_w', '_c'))
dd = np.abs((y.EPA_deg_w - y.EPA_deg_c + 90) % 180 - 90)
out['human_weidmann_vs_corradi'] = dict(n=len(dd), median=float(dd.median()), gt60=float((dd > 60).mean()))
print(out['human_weidmann_vs_corradi'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# PA error breakdowns from the cross validation (Section 3.2)
cv['d'] = np.abs((cv.cnn_pa - cv.EPA + 90) % 180 - 90)
out['cv_fold_median'] = {int(k): float(v) for k, v in cv.groupby('fold').d.median().items()}
out['cv_by_RTTA'] = {}
for lo, hi in [(0, 0.8), (0.8, 0.95), (0.95, 1.01)]:
    s_ = cv[(cv.cnn_R >= lo) & (cv.cnn_R < hi)]
    out['cv_by_RTTA'][f'{lo}-{hi}'] = dict(n=len(s_), median=float(s_.d.median()), gt60=float((s_.d > 60).mean()))
out['cv_by_size'] = {}
for lo, hi in [(4, 10), (10, 20), (20, 40), (40, 1e4)]:
    s_ = cv[(cv.MajDiam >= lo) & (cv.MajDiam < hi)]
    out['cv_by_size'][f'{lo}-{hi}'] = dict(n=len(s_), median=float(s_.d.median()))
out['cv_by_survey'] = {sv: dict(n=int((cv.survey == sv).sum()), median=float(cv[cv.survey == sv].d.median()))
                       for sv in ['IPHAS', 'VPHAS', 'SHS']}
print(out['cv_fold_median'], out['cv_by_RTTA'], out['cv_by_size'], out['cv_by_survey'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# why RZ13 bipolars drop out of the selection
pf = pd.read_csv(ROOT / 'data' / 'pa_final.csv')
rb = pf[pf.morph_class == 'bipolar']
out['rz13_bipolar_cuts'] = dict(in_hash=len(rb), lt6=int((rb.MajDiam < 6).sum()), rtta_lt08=int((rb.R_tta < 0.8).sum()),
                                shs=int((rb.survey == 'SHS').sum()),
                                pass_all=int(((rb.MajDiam >= 6) & (rb.R_tta >= 0.8) & (rb.survey != 'SHS')).sum()))
print(out['rz13_bipolar_cuts'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# provenance of the labels and the height of the selected RZ13 bipolars at 8 kpc
w2 = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv'); w2['key'] = w2.PNG_from_name.fillna(w2.PNG_approx_from_lb)
mm = m.assign(EPA=pd.to_numeric(m.EPA, errors='coerce'), key='PN G' + m.PNG)
xx = mm[mm.EPA.notna()].merge(w2[['key', 'EPA_deg']], on='key')
out['labels_provenance'] = dict(n_hash_pa=int(mm.EPA.notna().sum()), matched_weidmann=len(xx),
                                identical=float((np.abs((xx.EPA - xx.EPA_deg + 90) % 180 - 90) == 0).mean()))
r1 = json.load(open(ROOT / 'data' / 'rev1.json'))
out['rz13_sel_height_pc_at_8kpc'] = float(8000 * np.tan(np.radians(r1['phys_size']['RZ13 bipolar cuts']['median_abs_b'])))
print(out['labels_provenance'], out['rz13_sel_height_pc_at_8kpc'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# labels with a cutout
plc = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')[['idPNMain', 'MajDiam']]
mm2 = m.assign(EPA=pd.to_numeric(m.EPA, errors='coerce'), idPNMain=m.idPNMain.astype(int))
lab_cut = mm2[mm2.EPA.notna()].merge(plc, on='idPNMain', suffixes=('', '_c'))
out['labels_with_cutout'] = dict(n=len(lab_cut), n_ge4=int((lab_cut.MajDiam_c >= 4).sum()))
print(out['labels_with_cutout'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)

# RZ13 bipolars: consistency failures among those at least 6 arcsec; candidate binaries in the alignment test
out['rz13_bipolar_cuts']['ge6_rtta_lt08'] = int(((rb.MajDiam >= 6) & (rb.R_tta < 0.8)).sum())
sb = pd.read_csv(ROOT / 'data' / 'supp_binaries.csv')
out['binaries_candidates_in_test'] = int((sb.in_alignment_test & sb.candidate).sum())
print(out['rz13_bipolar_cuts'], out['binaries_candidates_in_test'])
json.dump(out, open(ROOT / 'data' / 'text_numbers.json', 'w'), indent=1)
