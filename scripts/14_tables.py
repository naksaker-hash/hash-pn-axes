"""Write the paper tables from the JSON outputs (revision 1).
Output: paper/tab_paacc.tex, paper/tab_align.tex, paper/tab_tan.tex"""
import json, pathlib
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
bl = json.load(open(D / 'baselines.json'))
rv = json.load(open(D / 'rev1.json'))
r2 = json.load(open(D / 'rev2.json'))
pc = lambda v: f'{int(100 * v + 0.5)}'
f1 = lambda v: '' if v is None else f'{v:.1f}'

rows = []
names = {'cv': 'Cross validation', 'Corradi98': r'\citet{corradi1998}', 'RZ13': r'\citet{rees2013}'}
for sname in ['cv', 'Corradi98', 'RZ13']:
    for i, (cl, lab) in enumerate([('all', 'All'), ('B', 'Bipolar'), ('E', 'Elliptical')]):
        r = bl[sname][cl]
        rows.append(f"{names[sname] if i == 0 else ''} & {lab} & {r['network']['n']} & {f1(r['network']['median'])} & {pc(r['network']['gt60'])} & "
                    f"{f1(r['classical']['median'])} & {pc(r['classical']['gt60'])} & {f1(r['yolo_mask']['median'])} & "
                    f"{f1(r['low_isophote']['median'])} & {f1(r['flux_weighted']['median'])} \\\\")
tab1 = r"""\begin{table*}
\caption{Differences between measured and visual position angles. The median absolute difference (deg) and the percentage of axes misplaced by more than $60\degr$ are given for the axis network and for the moments of the classical masks, followed by the medians for the moments of the masks of the segmentation network, of a mask at three times the noise and of the flux weighted image. The external tests include only PNe without a HASH position angle and with a HASH major diameter of at least 4 arcsec. The number $n$ refers to the axis network, and the moments, which need a mask, are measured for up to four fewer PNe.}
\label{tab:paacc}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrrrrrrr}
\hline
Test & Class & $n$ & \multicolumn{2}{c}{Axis network} & \multicolumn{2}{c}{Classical mask} & Segmentation mask & Low isophote & Flux weighted\\
 & & & median & $>60\degr$ & median & $>60\degr$ & median & median & median\\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\end{table*}
"""
(ROOT / 'paper' / 'tab_paacc.tex').write_text(tab1)

spec = [('RZ13 bipolar, visual, all', r'\citet{rees2013} bipolar, visual, all'),
        ('RZ13 bipolar, visual, same objects as cuts', r'\citet{rees2013} bipolar, visual, selected'),
        ('RZ13 bipolar, network, cuts', r'\citet{rees2013} bipolar, network, selected'),
        ('RZ13 bipolar, network, cuts, never trained', r'\quad of which not in the labels'),
        ('RZ13 polar, visual, all', r'\citet{rees2013} polar, visual, all'),
        ('bulge region B all', 'Bulge region bipolar, all'),
        ('bulge region B not in RZ13', r'Bulge region bipolar, not in \citet{rees2013}'),
        ('bulge region B, network P(B) >= 0.7', r'Bulge region bipolar, $P_{\mathrm{B}} \geq 0.7$'),
        ('bulge region E', 'Bulge region elliptical'),
        ('outside bulge region B', 'Outside bulge region, bipolar'),
        ('outside bulge region B, network P(B) >= 0.7', r'Outside bulge region, bipolar, $P_{\mathrm{B}} \geq 0.7$'),
        ('outside bulge region B, thin or thick disc', 'Outside bulge region, bipolar, disc component'),
        ('outside bulge region E', 'Outside bulge region, elliptical'),
        ('outside bulge region E, SHS only', 'Outside bulge region, elliptical, SHS only')]
fp = lambda p: (f'{p:.2f}' if p >= 0.1 else f'{p:.2g}') if p >= 0.001 else '$<0.001$'
fpu = lambda p: fp(p) if p <= 0.999 else '$>0.999$'
arows = []
for k, lab in spec:
    r = rv['samples'][k]
    arows.append(f"{lab} & {r['n']} & {r['mean']:.0f} & {r['R']:.2f} & {fp(r['rayleigh_p'])} & {fpu(r['V_p_parallel'])} & {fpu(r['V_p_two'])} \\\\")
from scipy import stats as _st
for k, lab in [('outside bulge region', 'Known close binaries, outside bulge region'), ('bulge region', 'Known close binaries, bulge region')]:
    r = r2['binaries'][k]['stats_90']
    arows.append(f"{lab} & {r['n']} & {r['mean']:.0f} & {r['R']:.2f} & {fp(r['rayleigh_p'])} & {fpu(r['V_p'])} & {fpu(2 * _st.norm.sf(abs(r['V']) * (2 * r['n']) ** 0.5))} \\\\")
r4 = json.load(open(ROOT / 'data' / 'rev4.json'))
for k, lab in [('outside, HASH bipolar', r'\quad bipolar only, outside bulge region'), ('bulge, HASH bipolar', r'\quad bipolar only, bulge region')]:
    r = r4['binaries_bipolar'][k]
    arows.append(f"{lab} & {r['n']} & {r['mean']:.0f} & {r['R']:.2f} & {fp(r['rayleigh_p'])} & {fpu(r['V_p'])} & {fpu(2 * _st.norm.sf(abs(r['V']) * (2 * r['n']) ** 0.5))} \\\\")
arows[-4:] = [arows[-4], arows[-2], arows[-3], arows[-1]]  # each bipolar only row under its binary row
tab2 = r"""\begin{table*}
\caption{Galactic position angle statistics. The mean axis, the mean resultant length $R$ of the doubled angles, the Rayleigh probability and the one sided and two sided V test probabilities for alignment with the plane ($\mathrm{GPA} = 90\degr$) are given. Network samples contain PNe with a HASH major diameter of at least 6 arcsec, $R_{\mathrm{TTA}} \geq 0.8$ and IPHAS or VPHAS+ images unless stated, and the ellipticals in addition a mask axial ratio below 0.85, except the known close binaries, which include all axisymmetric classes. Selected PNe of \citet{rees2013} pass the same criteria. The bulge region is $|l| < 10\degr$ and $|b| < 10\degr$, and $P_{\mathrm{B}}$ is the bipolar probability of the classification network.}
\label{tab:align}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrr}
\hline
Sample & $n$ & Mean GPA & $R$ & Rayleigh & V test & V test\\
 & & (deg) & & $p$ & one sided $p$ & two sided $p$\\
\hline
""" + '\n'.join(arows) + r"""
\hline
\end{tabular}
\end{table*}
"""
(ROOT / 'paper' / 'tab_align.tex').write_text(tab2)

tan = pd.read_csv(D / 'tan_comparison.csv')
tan = tan[tan.GPA_deg.notna()].sort_values('PNG')
trows = []
for _, r in tan.iterrows():
    sub = 'binary' if r.subsample.startswith('a') else 'high adf'
    trows.append(f"{r.Name} & {sub} & {r.mainClass} & {r.MajDiam:.1f} & {r.GPA_deg:.1f} & {r.gpa:.0f} & {r.d_gpa:.0f} & {r.R_tta:.2f} \\\\")
tab3 = r"""\begin{table}
\caption{Galactic position angles of the binary and high abundance discrepancy PNe of \citet{tan2023} from their visual measurements and from the axis network, with the HASH class, the HASH major diameter (arcsec) and $R_{\mathrm{TTA}}$. All 14 axes come from VPHAS+ images.}
\label{tab:tan}
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lllrrrrr}
\hline
Name & Type & Class & $D$ & GPA$_{\mathrm{vis}}$ & GPA$_{\mathrm{net}}$ & $|\Delta|$ & $R_{\mathrm{TTA}}$\\
\hline
""" + '\n'.join(trows) + r"""
\hline
\end{tabular}
\end{table}
"""
(ROOT / 'paper' / 'tab_tan.tex').write_text(tab3)
print(tab1, tab2, tab3)
