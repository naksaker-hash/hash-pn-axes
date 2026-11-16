"""Astronomy and Computing figure: network and moment errors for nebulae with ambiguous moment axes, and the
consistency R_TTA per HASH class. Inputs: data/ambiguous_axes.csv, data/catalogue_pa.csv. Output: paper_ac/figures/ambiguous.pdf"""
import pathlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_ac' / 'figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'axes.spines.top': False, 'axes.spines.right': False})
CNN, MOM, MUTED = '#2a78d6', '#eb6834', '#9a9a9a'
a = pd.read_csv(ROOT / 'data' / 'ambiguous_axes.csv')
cat = pd.read_csv(ROOT / 'data' / 'catalogue_pa.csv'); cat = cat[cat.HASH_MajDiam >= 2]
f, ax = plt.subplots(1, 2, figsize=(6.97, 2.6))
grid = np.linspace(0, 90, 181)
amb = a[a.spread > 45]; con = a[a.spread < 20]
for s, ls, lab in [(amb, '-', 'ambiguous moments'), (con, '--', 'consistent moments')]:
    ax[0].plot(grid, [(s.err_net <= g).mean() for g in grid], color=CNN, ls=ls, lw=1.8 if ls == '-' else 1.2, label=f'Network, {lab} ({len(s)})')
    ax[0].plot(grid, [(s.err_mom <= g).mean() for g in grid], color=MOM, ls=ls, lw=1.8 if ls == '-' else 1.2, label=f'Classical moments, {lab}')
ax[0].plot(grid, grid / 90, color=MUTED, lw=1, ls=':', label='Random axes')
ax[0].set_xlim(0, 90); ax[0].set_ylim(0, 1); ax[0].set_xticks([0, 15, 30, 45, 60, 75, 90])
ax[0].set_xlabel(r'$|\Delta \mathrm{PA}|$ from the visual axis (deg)'); ax[0].set_ylabel('Cumulative fraction')
ax[0].legend(frameon=False, loc='lower right', fontsize=6.5); ax[0].set_title('(a) Labelled PNe in cross validation', fontsize=9)
bins = np.linspace(0, 1, 21)
for cl, c, name in [('B', '#2a78d6', 'Bipolar'), ('E', '#3aa76d', 'Elliptical'), ('R', '#eb6834', 'Round'), ('S', '#555555', 'Stellar')]:
    v = cat[cat.HASH_class == cl].R_TTA.dropna().values
    ax[1].hist(v, bins=bins, histtype='step', density=True, color=c, lw=1.3, label=f'{name} ({len(v)}, {np.mean(v >= 0.8) * 100:.0f}% above 0.8)')
ax[1].axvline(0.8, color=MUTED, ls=':', lw=1)
ax[1].set_xlim(0, 1); ax[1].set_xlabel(r'Consistency $R_{\mathrm{TTA}}$'); ax[1].set_ylabel('Normalised number')
ax[1].legend(frameon=False, loc='upper left', fontsize=6.5); ax[1].set_title('(b) All HASH PNe by class', fontsize=9)
plt.tight_layout(); plt.savefig(OUT / 'ambiguous.pdf'); plt.savefig(OUT / 'ambiguous.png', dpi=150); plt.close()
