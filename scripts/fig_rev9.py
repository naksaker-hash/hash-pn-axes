"""Revision 9 figures for the MLST paper: flow (2 x 3 panels), pa_errors with the flux weighted moments. Output: paper_mlst/figures"""
import pathlib, json, importlib, sys, io, contextlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_mlst' / 'figures'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'axes.spines.top': False, 'axes.spines.right': False})
CNN, MOM, HAND, MUTED = '#2a78d6', '#eb6834', '#555555', '#9a9a9a'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
COL1, COL2 = 3.32, 6.97  # MNRAS column widths (inches)


def fig_pa_errors():
    cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv').merge(pd.read_csv(ROOT / 'data' / 'baselines_pa.csv')[['idPNMain', 'pa_flux']], on='idPNMain', how='left')
    cv['d'] = dpa(cv.cnn_pa, cv.EPA); cv['dm'] = dpa(cv.pa_eq, cv.EPA); cv['df'] = dpa(cv.pa_flux, cv.EPA)
    f, ax = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    grid = np.linspace(0, 90, 181)
    for a, cl, name in zip(ax, ['B', 'E'], ['Bipolar', 'Elliptical']):
        s = cv[cv.mainClass == cl]
        for col, c, ls, lab in [('d', CNN, '-', 'Axis network'), ('dm', MOM, '-', 'Classical mask moments'), ('df', MOM, '--', 'Flux weighted moments')]:
            v = s[col].dropna().values
            a.plot(grid, [(v <= g).mean() for g in grid], color=c, lw=2 if ls == '-' else 1.4, ls=ls, label=lab)
        a.plot(grid, grid / 90, color=MUTED, lw=1, ls=':', label='Random axes')
        a.set_xlim(0, 90); a.set_ylim(0, 1); a.set_xticks([0, 15, 30, 45, 60, 75, 90])
        a.set_xlabel(r'$|\Delta \mathrm{PA}|$ (deg)')
        a.set_title(f'{name} ($n$ = {len(s)})', fontsize=9)
    ax[0].set_ylabel('Cumulative fraction')
    ax[0].legend(frameon=False, loc='lower right', fontsize=7.5)
    plt.tight_layout(); plt.savefig(OUT / 'pa_errors.pdf'); plt.close()


def fig_flow(name='M 1-28'):
    """Overview: segmentation row and axis network row, all from real outputs for one bipolar PN."""
    import torch, torchvision.transforms.functional as TF
    pa = importlib.import_module('08_pa_regressor')
    ds = importlib.import_module('03_dataset')
    cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
    r = cv[cv.Name == name].iloc[0]
    h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
    ha = h[0].data.astype(float)
    rgb = ds.image(ROOT / 'data' / 'cutouts' / r.file, r.kcont).copy()  # north up uint8
    mpl = np.load(ROOT / 'data' / 'masks' / r.file.replace('.fits', '.npz'))['mask']
    mdl = np.load(ROOT / 'data' / 'masks' / r.file.replace('.fits', '_dl.npz'))['mask']
    n = ha.shape[0]; c = (n - 1) / 2; L = 0.62 * r.MajDiam / r.pixscale; W = 1.15 * L
    # axis network outputs for this object, per rotation
    m = pa.model(); m.load_state_dict(torch.load(ROOT / 'data' / 'pa_cnn_full.pt', map_location=pa.DEV)); m.eval()
    x = pa.load_images(cv[cv.Name == name])
    angs = []
    with torch.no_grad():
        for k in range(8):
            phi = 45 * k
            o = m(TF.rotate(x[0], phi)[None].to(pa.DEV)).cpu()[0]
            angs.append(((np.degrees(np.arctan2(o[1], o[0])) / 2 - phi) % 180).item())
    p_full, R_full = pa.predict(m, x)

    def axis(a, pa_, col, ls='-', lw=1.3, cx=c, cy=c, LL=L, flipy=False):
        t = np.radians(pa_)
        dy = np.cos(t) * (-1 if flipy else 1)
        a.plot([cx + LL * np.sin(t), cx - LL * np.sin(t)], [cy - LL * dy, cy + LL * dy], color=col, ls=ls, lw=lw)

    f, ax = plt.subplots(2, 3, figsize=(COL2 * 0.82, 4.1))
    lo, hi = np.nanpercentile(ha, [5, 99.7])
    a = ax[0, 0]; a.imshow(np.arcsinh(np.clip((ha - lo) / (hi - lo), 0, 1) * 10), origin='lower', cmap='gray_r')
    a.set_title(r'(a) H$\alpha$ cutout', fontsize=8)
    a = ax[0, 1]; a.imshow(rgb, origin='upper'); a.set_title(r'(b) Input H$\alpha$, H$\alpha - kr$, $r$', fontsize=8)
    a = ax[0, 2]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r'); a.contour(np.flipud(mpl), [0.5], colors=MOM, linewidths=0.9)
    a.set_title('(c) Classical mask', fontsize=8)
    a = ax[1, 0]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r'); a.contour(np.flipud(mdl), [0.5], colors='#2e9e5b', linewidths=0.9)
    a.set_title('(d) Segmentation network', fontsize=8)
    # (e) augmentation: rotated input with the transformed label
    phi = 60
    xr = TF.rotate(torch.tensor(rgb).permute(2, 0, 1), phi).permute(1, 2, 0).numpy()
    a = ax[1, 1]; a.imshow(xr[:, :, 1], origin='upper', cmap='gray_r')
    axis(a, r.EPA + phi, HAND, ':', 1.6, flipy=True)
    a.set_title(r'(e) Rotate by $\phi$, label PA + $\phi$', fontsize=8)
    # (f) axes on the image
    a = ax[1, 2]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r')
    for pa_, col, ls, lw in [(r.EPA, HAND, ':', 1.6), (r.pa_eq, MOM, '-', 1.2), (p_full[0], CNN, '-', 1.2)]:
        axis(a, pa_, col, ls, lw, flipy=True)
    a.set_title(f'(f) Network, moments, visual', fontsize=8)
    for a in list(ax.flat):
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(c - W, c + W)
    for a in [ax[0, 0]]:
        a.set_ylim(c - W, c + W)
    for a in [ax[0, 1], ax[0, 2], ax[1, 0], ax[1, 1], ax[1, 2]]:
        a.set_ylim(c + W, c - W)
    plt.tight_layout()
    plt.savefig(OUT / 'flow.pdf'); plt.close()
    return dict(name=name, angs=angs, pa=float(p_full[0]), R=float(R_full[0]), epa=float(r.EPA), mom=float(r.pa_eq))



def fig_tan_compare():
    """Network axes against the visual axes of Tan et al. (2023b) for their 14 binary and high adf PNe."""
    import os
    cut = float(os.environ.get('CUT', '2'))
    t = pd.read_csv(ROOT / 'data' / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg'])
    sel = (t.MajDiam >= cut) & (t.R_tta >= 0.8) & (t.survey != 'SHS')
    net = t.GPA_deg + ((t.gpa - t.GPA_deg + 90) % 180 - 90)  # network axis shifted by 180 deg where this brings it closer
    f, a = plt.subplots(figsize=(COL2 * 0.52, 3.0))
    a.plot([60, 175], [60, 175], color=MUTED, lw=0.8, ls=':')
    for sub, mk, lab in [('a_confirmed_SP_binary', 'o', 'close binary'), ('b_high_adf', 's', 'high adf')]:
        m = t.subsample == sub
        a.scatter(t.GPA_deg[m & sel], net[m & sel], s=28, marker=mk, color=CNN, label=f'{lab}, selected')
        a.scatter(t.GPA_deg[m & ~sel], net[m & ~sel], s=28, marker=mk, facecolor='none', edgecolor=CNN, label=f'{lab}, not selected')
    for _, r in t.iterrows():
        d = (r.gpa - r.GPA_deg + 90) % 180 - 90
        if abs(d) > 25:
            off = {'H 2-13': (6, -6), 'Bl M': (6, -3), 'M 3-42': (7, -2)}.get(r.Name, (6, 2))
            a.annotate(r.Name, (r.GPA_deg, r.GPA_deg + d), fontsize=6.5, xytext=off, textcoords='offset points', color=HAND,
                       ha='right' if off[0] < 0 else 'left', va='center')
    a.axhline(90, color=MUTED, lw=0.5); a.axvline(90, color=MUTED, lw=0.5)
    a.set_xlabel('Visual GPA of Tan et al. (deg)'); a.set_ylabel('Network GPA (deg)')
    a.set_xlim(80, 178); a.set_ylim(0, 210)
    a.legend(frameon=False, fontsize=6.5, loc='upper left')
    plt.tight_layout(); plt.savefig(OUT / 'tan_compare.pdf'); plt.close()
    return dict(n=len(t), n_sel=int(sel.sum()))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    import sys as _s
    if len(_s.argv) > 1:
        for n_ in _s.argv[1:]: print(n_, globals()[f"fig_{n_}"]())
    else:
        fig_pa_errors(); print("flow", fig_flow())
