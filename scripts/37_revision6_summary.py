"""Summary of the revision 6 cross validation runs. Output: data/rev6_runs.json and paper_mlst/tab_ablation.tex

Runs (data/pa_cnn_cv<TAG>.csv): seeds 0 to 4 of the standard network ('', _s1 to _s4), seed 0 repeated with the
dihedral test average (_s0d8tta, one fold diverged and the run is not used), two further assignments of blocks to folds (_fold1, _fold2), the ablations
without augmentation (_noaug, _noaug_s1, _noaug_s2), with untransformed labels (_rotnolabel) and with the Halpha
band alone (_ha, _ha_s1, _ha_s2), the von Mises head (_vm_s0 to _vm_s2) and the D8 steerable CNN (_d8_s0 to _d8_s2).
"""
import json, pathlib, sys, importlib, os
import numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
load = lambda tag: pd.read_csv(D / f'pa_cnn_cv{tag}.csv') if (D / f'pa_cnn_cv{tag}.csv').exists() else None


def stat(c, col='cnn_pa'):
    if c is None:
        return None
    d = dpa(c[col], c.EPA)
    r = dict(all=float(np.median(d)), gt60=float(np.mean(d > 60)))
    for cl in ['B', 'E']:
        s = c.mainClass == cl
        r[cl] = float(np.median(d[s])); r[f'{cl}_gt60'] = float(np.mean(d[s] > 60))
    return r


def summ(rows):
    rows = [r for r in rows if r is not None]
    if not rows:
        return None
    out = {k: dict(mean=float(np.mean([r[k] for r in rows])), sd=float(np.std([r[k] for r in rows], ddof=1)) if len(rows) > 1 else 0.0,
                   min=float(np.min([r[k] for r in rows])), max=float(np.max([r[k] for r in rows]))) for k in rows[0]}
    out['n_runs'] = len(rows)
    return out


out = {}
seeds = [load(t) for t in ['', '_s1', '_s2', '_s3', '_s4']]
out['standard_5seeds'] = summ([stat(c) for c in seeds])
out['standard_per_seed'] = [stat(c) for c in seeds if c is not None]
# ensemble of the five seeds
base = seeds[0]
V = np.mean([np.exp(2j * np.radians(c.set_index('idPNMain').reindex(base.idPNMain).cnn_pa.values)) for c in seeds if c is not None], 0)
ens = base.assign(cnn_pa=(np.degrees(np.angle(V)) / 2) % 180)
out['ensemble'] = stat(ens); out['ensemble_agreement_median'] = float(np.median(np.abs(V)))
pair = [dpa(seeds[i].cnn_pa.values, seeds[j].set_index('idPNMain').reindex(seeds[i].idPNMain).cnn_pa.values)
        for i in range(5) for j in range(i + 1, 5) if seeds[i] is not None and seeds[j] is not None]
out['seed_pairwise'] = dict(median=float(np.median(np.concatenate(pair))), max=float(np.max(np.concatenate(pair))))
g = dpa(base.cnn_pa, base.EPA) > 60
out['ensemble_agreement_auc_vs_seed0'] = float(roc_auc_score(g, 1 - np.abs(V)))
ge = dpa(ens.cnn_pa, ens.EPA) > 60
out['ensemble_agreement_auc'] = float(roc_auc_score(ge, 1 - np.abs(V)))  # against the failures of the ensemble mean axis
# test time copies by seed
out['tta'] = summ([dict(t1=float(np.median(dpa(c.cnn_pa_tta1, c.EPA))), t4=float(np.median(dpa(c.cnn_pa_tta4, c.EPA))),
                        t8=float(np.median(dpa(c.cnn_pa, c.EPA)))) for c in seeds if c is not None])
d8t = [load(t) for t in ['_s3', '_s4', '_fold1', '_fold2']]  # _s0d8tta is not used: one fold diverged on MPS
out['tta16'] = summ([dict(t8=float(np.median(dpa(c.cnn_pa, c.EPA))), t16=float(np.median(dpa(c.cnn_pa_d8, c.EPA)))) for c in d8t if c is not None and 'cnn_pa_d8' in c])
# fold assignments
out['fold_assignments'] = summ([stat(c) for c in [seeds[0], load('_fold1'), load('_fold2')]])
# ablations
out['noaug'] = summ([stat(c) for c in [load('_noaug'), load('_noaug_s1'), load('_noaug_s2')]])
out['ha'] = summ([stat(c) for c in [load('_ha'), load('_ha_s1'), load('_ha_s2')]])
out['rotnolabel'] = stat(load('_rotnolabel'))
out['vm'] = summ([stat(c) for c in [load('_vm_s0'), load('_vm_s1'), load('_vm_s2')]])
out['d8'] = summ([stat(c) for c in [load('_d8_s0'), load('_d8_s1'), load('_d8_s2')]])
out['scratch'] = summ([stat(c) for c in [load('_scratch_s0'), load('_scratch_s1'), load('_scratch_s2')]])
out['rotfull'] = summ([stat(c) for c in [load('_rotfull_s0'), load('_rotfull_s1'), load('_rotfull_s2')]])
lc = json.load(open(D / 'learning_curve.json')) if (D / 'learning_curve.json').exists() else None
# von Mises concentration as an uncertainty
vm0 = load('_vm_s0')
if vm0 is not None:
    dv = dpa(vm0.cnn_pa, vm0.EPA)
    out['vm_kappa'] = dict(spearman=float(stats.spearmanr(dv, -vm0.cnn_kappa).correlation), auc=float(roc_auc_score(dv > 60, -vm0.cnn_kappa)),
                           median_kappa=float(vm0.cnn_kappa.median()))
    # coverage of the predicted von Mises interval for 2 theta: |2 dtheta| within the central 68 per cent
    k = vm0.cnn_kappa.values; z = np.radians(2 * ((vm0.cnn_pa - vm0.EPA + 90) % 180 - 90)).values
    from scipy.stats import vonmises
    lo = vonmises.ppf(0.16, k); hi = vonmises.ppf(0.84, k)
    out['vm_kappa']['coverage68'] = float(np.mean((z >= lo) & (z <= hi)))
# R_TTA for comparison
out['rtta'] = dict(spearman=float(stats.spearmanr(dpa(base.cnn_pa, base.EPA), 1 - base.cnn_R).correlation), auc=float(roc_auc_score(g, 1 - base.cnn_R)))
# survey errors of the labelled PNe
out['cv_by_survey'] = {sv: dict(n=int((base.survey == sv).sum()), median=float(np.median(dpa(base.cnn_pa, base.EPA)[base.survey == sv])))
                       for sv in ['IPHAS', 'VPHAS', 'SHS']}
# derived numbers of the discussion
r6 = json.load(open(D / 'rev6.json'))
fb = r6['binary_fraction']['resolved']
out['binary_counts'] = dict(n=266, low=float(fb['observed_frame'][0] * 266), high=float(fb['observed_frame'][1] * 266),
                            limit=float(r6['limits']['tight 107.5 cv']['crossing'] * 266), conservative=float(0.24 * 266), P_B=fb['P_B'])
# parameters of the D8 network
os.environ['PA_ARCH'] = 'd8'
sys.path.insert(0, str(ROOT / 'scripts'))
m = importlib.import_module('08_pa_regressor')
out['d8_params'] = int(sum(p.numel() for p in m.D8Net().parameters()))
json.dump(out, open(D / 'rev6_runs.json', 'w'), indent=1)

# ablation table (mean and spread over seeds)
f = lambda r, k: (f"{r[k]['mean']:.1f} $\\pm$ {r[k]['sd']:.1f}" if r and r.get('n_runs', 1) > 1 else (f"{r[k]['mean']:.1f}" if r else '--'))
g60 = lambda r, k: (f"{100 * r[k]['mean']:.0f}" if r else '--')
rows = [('ResNet18, standard (five seeds)', out['standard_5seeds']), ('Ensemble of the five seeds', None),
        ('Two further assignments of blocks to folds$^a$', out['fold_assignments']),
        ('No augmentation', out['noaug']), ('H$\\alpha$ band alone', out['ha']), ('Untransformed labels$^b$', None),
        ('von Mises head', out['vm']), ('Full cutout rotated before cropping', out['rotfull']), ('ResNet18 from scratch', out['scratch']), ('D8 steerable CNN (dihedral group of order 16), from scratch', out['d8'])]
lines = []
for name, r in rows:
    if name.startswith('Ensemble'):
        e = out['ensemble']; lines.append(f"{name} & 1 & {e['all']:.1f} & {100 * e['gt60']:.0f} & {e['B']:.1f} & {e['E']:.1f} \\\\")
    elif name.startswith('Untransformed'):
        e = out['rotnolabel']; lines.append(f"{name} & 1 & {e['all']:.1f} & {100 * e['gt60']:.0f} & {e['B']:.1f} & {e['E']:.1f} \\\\")
    elif r is not None:
        lines.append(f"{name} & {r['n_runs']} & {f(r, 'all')} & {g60(r, 'gt60')} & {f(r, 'B')} & {f(r, 'E')} \\\\")
lm = json.load(open(D / 'rev9.json'))['learned_moments'] if (D / 'rev9.json').exists() else None
l10 = json.load(open(D / 'rev10.json'))['learned_moments'] if (D / 'rev10.json').exists() else None
if lm:
    lines.append(f"Gradient boosting on the four moment axes and mask shape$^d$ & 3 & {np.mean(l10['medians']):.1f} $\\pm$ {np.std(l10['medians'], ddof=1):.1f} & {100 * np.mean(l10['gt60']):.0f} & -- & -- \\\\")
t = out['tta']; t16 = out['tta16']
tta_line = (f"Test time copies 1, 4 and 8 (five seeds) & 5 & {t['t1']['mean']:.1f}, {t['t4']['mean']:.1f}, {t['t8']['mean']:.1f} & & & \\\\")
lines.append(tta_line)
if t16:
    lines.append(f"Test time copies 8 and 16 with reflections$^c$ & {t16['n_runs']} & {t16['t8']['mean']:.1f}, {t16['t16']['mean']:.1f} & & & \\\\")
tab = r"""\begin{table}
\caption{Median absolute difference from the visual axes (deg) and percentage of axes misplaced by more than $60\degr$ in the cross validation of the 350 labelled PNe, with the mean and standard deviation over the runs. $^a$With the standard assignment. $^b$Close to the $45\degr$ of random axes. $^c$The four runs that stored the 16 copies, since a repeat of seed 0 diverged on the MPS backend. $^d$Folds of seed 0 and the 346 PNe with all four moments, three augmentation seeds.}
\label{tab:ablation}
\setlength{\tabcolsep}{4pt}
\resizebox{\textwidth}{!}{%
\begin{tabular}{lrrrrr}
\hline
Configuration & Runs & All & $>60\degr$ & Bipolar & Elliptical\\
\hline
""" + '\n'.join(lines) + r"""
\hline
\end{tabular}}
\end{table}
"""
(ROOT / 'paper_mlst' / 'tab_ablation.tex').write_text(tab)
print(json.dumps(out, indent=1)[:6000])
print(tab)
