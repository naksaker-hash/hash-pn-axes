"""Supplementary table of the three dimensional injection limits from data/rev8_inject3d.json -> paper_mlst/tab_s_3d.tex"""
import json, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
g = json.load(open(ROOT / 'data' / 'rev8_inject3d.json'))
names = {'field': 'Spiral field', 'girdle': 'Plane, random azimuth', 'pole': 'Galactic pole'}
rows = []
for m in ('field', 'girdle', 'pole'):
    for k in (20, 5):
        r = [g[f'{m} kappa {k} cut {c}']['V'] for c in (30, 45, 60)]
        cells = ' & '.join(f"{100 * v['f95_intrinsic']:.0f} & {100 * v['f95_visible']:.0f}" for v in r)
        rows.append(f"{names[m]} & {k} & {cells} \\\\")
tab = r"""\begin{table}
\caption{The 95 per cent upper limits (per cent) on three dimensional alignments of the 292 bipolars outside the bulge region, as the aligned fraction of all intrinsic axes and of the observed bipolars, for visibility cuts of $30\degr$, $45\degr$ and $60\degr$ from the line of sight. The limits are interpolated on a grid in steps of 0.025 with 1500 samples per step, with Monte Carlo errors below 1 per cent, and the pole model is scored with the V test perpendicular to the plane.}
\label{tab:s_3d}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrrrr}
\hline
Model & $\kappa$ & \multicolumn{2}{c}{Cut $30\degr$} & \multicolumn{2}{c}{Cut $45\degr$} & \multicolumn{2}{c}{Cut $60\degr$}\\
 & & intrinsic & observed & intrinsic & observed & intrinsic & observed\\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}}
\end{table}
"""
(ROOT / 'paper_mlst' / 'tab_s_3d.tex').write_text(tab); print(tab)
