"""Supplementary table of exploratory alignment tests (from data/rev5.json) and run times of the axis network
(from the modification times of the output files of run_revision4.sh). Output: paper_mlst/tab_s_explore.tex, data/compute.json
"""
import json, pathlib, os
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
r = json.load(open(D / 'rev5.json'))['exploratory']
fp = lambda p: f'{p:.2f}'
def lab(k):
    for a, b in [('P_B >= ', '$P_{\\mathrm{B}} \\geq$ '), ('R_TTA >= ', '$R_{\\mathrm{TTA}} \\geq$ '), ('|l| < 30 deg', '$|l| < 30\\degr$')]:
        k = k.replace(a, b)
    return k[0].upper() + k[1:]


rows = '\n'.join(lab(k) + f" & {v['n']} & {v['mean']:.0f} & {v['R']:.2f} & {fp(v['V_p'])} " + '\\\\' for k, v in r.items())
tab = r"""\begin{table}
\caption{Exploratory alignment tests, not corrected for the number of tests. The mean Galactic position angle, the mean resultant length $R$ and the one sided V test probability for alignment with the plane are given. All samples obey the selection of Section 2.6 of the main paper, and the other bipolars of the bulge region are those not in the sample of Rees \& Zijlstra (2013).}
\label{tab:s_explore}
\small
\begin{tabular}{lrrrr}
\hline
Sample & $n$ & Mean GPA & $R$ & V test $p$\\
\hline
""" + rows + r"""
\hline
\end{tabular}
\end{table}
"""
(ROOT / 'paper_mlst' / 'tab_s_explore.tex').write_text(tab)
t = {f: os.path.getmtime(D / f) for f in ['pa_cnn_cv_nomash.csv', 'pa_cnn_cv_nozero.csv', 'pa_cnn_all_s1.csv', 'pa_cnn_all_s2.csv']}
comp = dict(cv_five_folds_min=(t['pa_cnn_cv_nozero.csv'] - t['pa_cnn_cv_nomash.csv']) / 60,
            full_train_and_predict_min=(t['pa_cnn_all_s2.csv'] - t['pa_cnn_all_s1.csv']) / 60, hardware='Apple M1, PyTorch MPS')
comp['per_model_min'] = comp['cv_five_folds_min'] / 5
pl = pd.read_csv(D / 'pseudolabels.csv')
comp['seg_split_all'] = pl.split.value_counts().to_dict(); comp['seg_split_accepted'] = pl[pl.good].split.value_counts().to_dict()
comp['accepted_by_class'] = pl[pl.good].mainClass.value_counts().to_dict()
json.dump(comp, open(D / 'compute.json', 'w'), indent=1)
print(comp)
