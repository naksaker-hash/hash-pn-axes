"""Paper figures (STIX fonts, TrueType embedding). Output: paper/figures/*.pdf"""
import pathlib, json, importlib, sys, io, contextlib
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper' / 'figures'; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'axes.spines.top': False, 'axes.spines.right': False})
CNN, MOM, HAND, MUTED = '#2a78d6', '#eb6834', '#555555', '#9a9a9a'
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)
COL1, COL2 = 3.32, 6.97  # MNRAS column widths (inches)


def fig_pa_errors():
    cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
    cv['d'] = dpa(cv.cnn_pa, cv.EPA); cv['dm'] = dpa(cv.pa_eq, cv.EPA)
    f, ax = plt.subplots(1, 2, figsize=(COL2, 2.5), sharey=True)
    grid = np.linspace(0, 90, 181)
    for a, cl, name in zip(ax, ['B', 'E'], ['Bipolar', 'Elliptical']):
        s = cv[cv.mainClass == cl]
        for col, c, lab in [('d', CNN, 'Axis network'), ('dm', MOM, 'Classical mask moments')]:
            v = s[col].dropna().values
            a.plot(grid, [(v <= g).mean() for g in grid], color=c, lw=2, label=lab)
        a.plot(grid, grid / 90, color=MUTED, lw=1, ls='--', label='Random axes')
        a.set_xlim(0, 90); a.set_ylim(0, 1); a.set_xticks([0, 15, 30, 45, 60, 75, 90])
        a.set_xlabel(r'$|\Delta \mathrm{PA}|$ (deg)')
        a.set_title(f'{name} ($n$ = {len(s)})', fontsize=9)
    ax[0].set_ylabel('Cumulative fraction')
    ax[0].legend(frameon=False, loc='lower right', fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'pa_errors.pdf'); plt.close()


def fig_gallery():
    """Bipolars where the moment axis and the CNN axis differ: image, mask, both axes, hand axis."""
    cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
    cv['d'] = dpa(cv.cnn_pa, cv.EPA); cv['dm'] = dpa(cv.pa_eq, cv.EPA)
    s = cv[(cv.mainClass == 'B') & (cv.dm > 50) & (cv.d < 15) & (cv.MajDiam >= 10)].sort_values('MajDiam', ascending=False).head(8)
    f, ax = plt.subplots(2, 4, figsize=(COL2, 3.7))
    for a, (_, r) in zip(ax.flat, s.iterrows()):
        h = fits.open(ROOT / 'data' / 'cutouts' / r.file)
        d = h[0].data.astype(float)
        if 'R' in h and np.isfinite(r.kcont):
            d = (d - np.nanmedian(d)) - r.kcont * (h['R'].data - np.nanmedian(h['R'].data))
        lo, hi = np.nanpercentile(d, [5, 99.7])
        a.imshow(np.arcsinh(np.clip((d - lo) / (hi - lo), 0, 1) * 10), origin='lower', cmap='gray_r')
        m = np.load(ROOT / 'data' / 'masks' / r.file.replace('.fits', '.npz'))['mask']
        a.contour(m, [0.5], colors=MOM, linewidths=0.7)
        c = (d.shape[0] - 1) / 2; L = 0.62 * r.MajDiam / r.pixscale
        # north up, east left: PA east of north maps to (dx, dy) = (-sin, cos)
        for pa, col, ls, lw in [(r.EPA, HAND, ':', 1.6), (r.pa_eq, MOM, '-', 1.2), (r.cnn_pa, CNN, '-', 1.2)]:
            t = np.radians(pa)
            a.plot([c + L * np.sin(t), c - L * np.sin(t)], [c - L * np.cos(t), c + L * np.cos(t)], color=col, ls=ls, lw=lw)
        a.set_title(r.Name, fontsize=8); a.set_xticks([]); a.set_yticks([])
        a.set_xlim(c - 1.1 * L, c + 1.1 * L); a.set_ylim(c - 1.1 * L, c + 1.1 * L)
    for a in ax.flat[len(s):]:
        a.axis('off')
    plt.tight_layout(); plt.savefig(OUT / 'gallery_axes.pdf'); plt.close()
    return s.Name.tolist()


def _circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); R = np.hypot(np.cos(th).mean(), np.sin(th).mean())
    from scipy import stats
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return n, R, stats.norm.sf(V * np.sqrt(2 * n))


def fig_alignment():
    x = pd.read_csv(ROOT / 'data' / 'pa_final.csv')
    tan = pd.read_csv(ROOT / 'data' / 'tan_comparison.csv')
    base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
    B = base & (x.mainClass == 'B'); E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
    rzc = x[(x.morph_class == 'bipolar') & base]
    panels = [('Rees and Zijlstra bipolars, visual', rzc.gpa_rz, HAND, 90),
              ('Same objects, network', rzc.gpa, CNN, 90),
              ('Tan et al. binary and high adf PNe', tan[tan.GPA_deg.notna()].gpa, CNN, 107.5),
              ('Bulge region, other bipolars', x[B & x.bulge_sky & x.morph_class.isna()].gpa, CNN, 90),
              ('Outside bulge region, bipolars', x[B & ~x.bulge_sky].gpa, CNN, 90),
              ('Outside bulge region, ellipticals', x[E & ~x.bulge_sky].gpa, CNN, 90)]
    f, ax = plt.subplots(2, 3, figsize=(COL2, 4.0))
    bins = np.arange(0, 181, 15)
    for a, (t, v, c, mu) in zip(ax.flat, panels):
        v = np.asarray(pd.Series(v).dropna())
        n, R, pv = _circ(v, mu)
        a.hist(v, bins=bins, density=True, color=c, edgecolor='white', linewidth=1)
        a.axhline(1 / 180, color=MUTED, ls='--', lw=0.8)
        a.axvline(mu, color=MUTED, lw=0.6)
        a.set_xlim(0, 180); a.set_xticks([0, 45, 90, 135, 180]); a.set_yticks([])
        a.set_xlabel('GPA (deg)', fontsize=8)
        a.set_title(t, fontsize=8, pad=14)
        a.text(0.5, 1.02, f"$n$ = {n}   $R$ = {R:.2f}   $p_V$ = {pv:.2g}" if pv >= 0.001 else f"$n$ = {n}   $R$ = {R:.2f}   $p_V$ < 0.001",
               transform=a.transAxes, ha='center', va='bottom', fontsize=7.5)
    for a in ax[:, 0]:
        a.set_ylabel('Relative frequency', fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'alignment.pdf'); plt.close()


def fig_injection():
    r = json.load(open(ROOT / 'data' / 'injection_final.json'))
    f, a = plt.subplots(figsize=(COL1, 2.6))
    a.axvspan(0.14, 0.49, color='#e6e6e6', lw=0)
    a.text(0.315, 0.97, 'close binary fraction\namong bipolar PNe', ha='center', va='top', fontsize=7, color=HAND)
    for key, col, ls, lab in [('outside bulge region B | tight | cv', CNN, '-', 'Tight, with errors'),
                              ('outside bulge region B | tight | none', CNN, '--', 'Tight, no errors'),
                              ('outside bulge region B | visual RZ13 | cv', MOM, '-', 'Rees and Zijlstra concentration')]:
        items = sorted((float(k), v) for k, v in r[key]['grid'].items())
        a.plot([i_[0] for i_ in items], [i_[1] for i_ in items], color=col, ls=ls, lw=2, label=lab)
    a.axhline(0.05, color=MUTED, ls=':', lw=0.8)
    a.set_xlabel('Aligned fraction $f$'); a.set_ylabel(r'$P(V \leq V_\mathrm{obs})$')
    a.set_xlim(0, 0.6); a.set_ylim(0, 1.02)
    a.legend(frameon=False, fontsize=7, loc='upper right', bbox_to_anchor=(1, 0.8))
    plt.tight_layout(); plt.savefig(OUT / 'injection.pdf'); plt.close()


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

    f, ax = plt.subplots(2, 4, figsize=(COL2, 3.9))
    lo, hi = np.nanpercentile(ha, [5, 99.7])
    a = ax[0, 0]; a.imshow(np.arcsinh(np.clip((ha - lo) / (hi - lo), 0, 1) * 10), origin='lower', cmap='gray_r')
    a.set_title(r'(a) H$\alpha$ cutout', fontsize=8)
    a = ax[0, 1]; a.imshow(rgb, origin='upper'); a.set_title(r'(b) Input H$\alpha$, H$\alpha - kr$, $r$', fontsize=8)
    a = ax[0, 2]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r'); a.contour(np.flipud(mpl), [0.5], colors=MOM, linewidths=0.9)
    a.set_title('(c) Pseudo label, 10 per cent of peak', fontsize=8)
    a = ax[0, 3]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r'); a.contour(np.flipud(mdl), [0.5], colors=CNN, linewidths=0.9)
    a.set_title('(d) Segmentation network', fontsize=8)
    # (e) augmentation: rotated input with the transformed label
    phi = 60
    xr = TF.rotate(torch.tensor(rgb).permute(2, 0, 1), phi).permute(1, 2, 0).numpy()
    a = ax[1, 0]; a.imshow(xr[:, :, 1], origin='upper', cmap='gray_r')
    axis(a, r.EPA + phi, HAND, ':', 1.6, flipy=True)
    a.set_title(r'(e) Rotate by $\phi$, label PA + $\phi$', fontsize=8)
    # (f) TTA vectors on the doubled-angle circle
    a = ax[1, 1]; t = np.linspace(0, 2 * np.pi, 200); a.plot(np.cos(t), np.sin(t), color=MUTED, lw=0.6)
    for g in angs:
        a.plot([0, np.cos(np.radians(2 * g))], [0, np.sin(np.radians(2 * g))], color=CNN, lw=0.8, alpha=0.6)
    v = np.mean([[np.cos(np.radians(2 * g)), np.sin(np.radians(2 * g))] for g in angs], 0)
    a.annotate('', xy=v, xytext=(0, 0), arrowprops=dict(arrowstyle='->', color=CNN, lw=1.8))
    a.set_aspect('equal'); a.set_xlim(-1.15, 1.15); a.set_ylim(-1.15, 1.15)
    a.set_title(f'(f) 8 rotations, $R_{{\\rm TTA}}$ = {R_full[0]:.2f}', fontsize=8)
    a.set_xlabel(r'$\cos 2\theta$', fontsize=7); a.set_ylabel(r'$\sin 2\theta$', fontsize=7)
    # (g) axes on the image
    a = ax[1, 2]; a.imshow(rgb[:, :, 1], origin='upper', cmap='gray_r')
    for pa_, col, ls, lw in [(r.EPA, HAND, ':', 1.6), (r.pa_eq, MOM, '-', 1.2), (p_full[0], CNN, '-', 1.2)]:
        axis(a, pa_, col, ls, lw, flipy=True)
    a.set_title('(g) Axis network, moments, visual', fontsize=8)
    for a in list(ax[0]) + [ax[1, 0], ax[1, 2]]:
        a.set_xticks([]); a.set_yticks([]); a.set_xlim(c - W, c + W)
    for a in [ax[0, 0]]:
        a.set_ylim(c - W, c + W)
    for a in [ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 2]]:
        a.set_ylim(c + W, c - W)
    # (h) reliability: cross validation error against the TTA consistency
    cvp = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
    cvp['d'] = dpa(cvp.cnn_pa, cvp.EPA)
    a = ax[1, 3]
    u = (1 - cvp.cnn_R).clip(lower=1e-4)
    a.scatter(u, cvp.d, s=4, color=CNN, alpha=0.45, lw=0)
    qs = np.quantile(u, np.linspace(0, 1, 7))
    for lo, hi in zip(qs[:-1], qs[1:]):
        q = cvp[(u >= lo) & (u <= hi)]
        a.plot([lo, hi], [q.d.median()] * 2, color=HAND, lw=2)
    a.set_xscale('log'); a.set_xlim(qs[-1] * 1.3, qs[0] * 0.8); a.set_ylim(0, 90); a.set_yticks([0, 30, 60, 90])
    a.set_xlabel(r'$1 - R_{\rm TTA}$', fontsize=7); a.set_ylabel(r'$|\Delta {\rm PA}|$ (deg)', fontsize=7)
    a.set_title('(h) Error against consistency', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT / 'flow.pdf'); plt.close()
    return dict(name=name, angs=angs, pa=float(p_full[0]), R=float(R_full[0]), epa=float(r.EPA), mom=float(r.pa_eq))


def fig_sky():
    """Bipolar PNe above the size cut with R_TTA >= 0.8: IPHAS and VPHAS+ (analysed) and SHS (excluded)."""
    x = pd.read_csv(ROOT / 'data' / 'pa_final.csv')
    s = x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.mainClass == 'B')].copy()
    s['l'] = ((s.Glon + 180) % 360) - 180
    f, a = plt.subplots(figsize=(COL2, 2.3))
    for sv, col, mk, lab in [('IPHAS', CNN, 'o', 'IPHAS'), ('VPHAS', MOM, 'o', 'VPHAS+'), ('SHS', MUTED, 'x', 'SHS, excluded')]:
        t = s[s.survey == sv]
        a.scatter(t.l, t.Glat, s=6 if mk == 'o' else 9, c=col, marker=mk, lw=0.6 if mk == 'x' else 0, label=f'{lab} ({len(t)})')
    a.add_patch(plt.Rectangle((-10, -10), 20, 20, fill=False, ec=HAND, lw=0.8, ls='--'))
    a.text(0, -10.9, 'bulge region', ha='center', va='top', fontsize=7, color=HAND)
    a.set_xlim(180, -180); a.set_ylim(-16, 16); a.set_xticks([180, 120, 60, 0, -60, -120, -180])
    a.set_xticklabels(['180', '120', '60', '0', '300', '240', '180'])
    a.set_xlabel('Galactic longitude (deg)'); a.set_ylabel('Galactic latitude (deg)')
    a.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=3, markerscale=1.5)
    plt.tight_layout(); plt.savefig(OUT / 'sky.pdf'); plt.close()
    return {sv: int((s.survey == sv).sum()) for sv in ['IPHAS', 'VPHAS', 'SHS']}, int((s.Glat.abs() > 16).sum())


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'scripts'))
    fig_pa_errors()
    print('gallery', fig_gallery())
    fig_alignment()
    fig_injection()
    print('sky', fig_sky())
    print('flow', fig_flow())
    print('figures written to', OUT)
