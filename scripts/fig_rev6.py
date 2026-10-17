"""Figures added in revision 6 (deep review). Output: paper_mlst/figures/*.pdf

axis_schematic   the doubled angle target, its transformation under rotation and reflection, and the network
uncertainty      error against the fraction of axes kept for R_TTA, seed spread and von Mises kappa
failures         gallery of the gross failures of the axis network in cross validation (supplement)
"""
import pathlib, sys, importlib, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Arc
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper_mlst' / 'figures'
plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 9,
                     'pdf.fonttype': 42, 'axes.linewidth': 0.6, 'xtick.direction': 'in', 'ytick.direction': 'in',
                     'axes.spines.top': False, 'axes.spines.right': False})
CNN, MOM, HAND, MUTED, ALT = '#2a78d6', '#eb6834', '#555555', '#9a9a9a', '#2b9e5f'
COL1, COL2 = 3.32, 6.97
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)


def arrow(ax, p, q, c, lw=1.4, ms=8, ls='-'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=ms, color=c, lw=lw, ls=ls, shrinkA=0, shrinkB=0))


def fig_axis_schematic():
    f, ax = plt.subplots(1, 3, figsize=(COL2, 2.45), gridspec_kw=dict(width_ratios=[1, 1, 1.45]))
    # (a) an axis in the image: theta and theta + 180 deg are the same answer; rotation adds phi
    a = ax[0]; a.set_xlim(-1.25, 1.25); a.set_ylim(-1.25, 1.25); a.set_aspect('equal'); a.axis('off')
    th, phi = 25.0, 40.0
    for t, c, alpha in [(th, HAND, 1.0), (th + phi, CNN, 1.0)]:
        r = np.radians(t)
        d = np.array([-np.sin(r), np.cos(r)])  # position angle east of north, north up and east to the left
        a.add_patch(Ellipse((0, 0), 0.75, 1.75, angle=t, fc='none', ec=c, lw=1.0, alpha=0.8))
        a.plot([-d[0], d[0]], [-d[1], d[1]], color=c, lw=1.6)
    a.annotate('', xy=(0, 1.15), xytext=(0, 0.8), arrowprops=dict(arrowstyle='-|>', color='k', lw=0.6))
    a.text(0.06, 1.12, 'N', fontsize=8); a.annotate('', xy=(-1.15, 0), xytext=(-0.8, 0), arrowprops=dict(arrowstyle='-|>', color='k', lw=0.6))
    a.text(-1.2, 0.07, 'E', fontsize=8)
    a.add_patch(Arc((0, 0), 1.1, 1.1, theta1=90, theta2=90 + th, color=HAND, lw=0.8))
    a.text(-0.17, 0.62, r'$\theta$', color=HAND, fontsize=9)
    a.add_patch(Arc((0, 0), 1.45, 1.45, theta1=90 + th, theta2=90 + th + phi, color=CNN, lw=0.8))
    a.text(-0.62, 0.62, r'$\phi$', color=CNN, fontsize=9)
    a.text(0.0, -1.25, r'axis at $\theta \equiv \theta + 180\degree$', ha='center', fontsize=8)
    a.set_title('(a) Image rotated by $\\phi$', fontsize=9)
    # (b) the doubled angle circle: rotation adds 2 phi, reflection mirrors
    b = ax[1]; b.set_xlim(-1.35, 1.35); b.set_ylim(-1.35, 1.35); b.set_aspect('equal'); b.axis('off')
    tt = np.linspace(0, 2 * np.pi, 200); b.plot(np.cos(tt), np.sin(tt), color=MUTED, lw=0.8)
    b.plot([-1.25, 1.25], [0, 0], color=MUTED, lw=0.4); b.plot([0, 0], [-1.25, 1.25], color=MUTED, lw=0.4)
    for ang, c, lab, off in [(2 * th, HAND, r'$2\theta$', (0.06, 0.04)), (2 * (th + phi), CNN, r'$2\theta + 2\phi$', (-0.05, 0.08)),
                             (-2 * th, ALT, r'$-2\theta$', (0.06, -0.14))]:
        r = np.radians(ang); p = np.array([np.cos(r), np.sin(r)])
        arrow(b, (0, 0), tuple(p), c, lw=1.3, ms=7)
        b.text(*(1.12 * p + np.array(off)), lab, color=c, fontsize=8.5, ha='center', va='center')
    b.text(1.27, -0.12, r'$\cos 2\theta$', fontsize=8, ha='right'); b.text(0.05, 1.22, r'$\sin 2\theta$', fontsize=8)
    b.text(0.0, -1.38, 'reflection $2\\theta \\rightarrow -2\\theta$', ha='center', fontsize=8, color=ALT)
    b.set_title('(b) Doubled angle label', fontsize=9)
    # (c) the network and the test time average
    c = ax[2]; c.set_xlim(0, 10); c.set_ylim(0, 6); c.axis('off')
    def box(x0, y0, w, h, txt, fc):
        c.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0.08', fc=fc, ec='none'))
        c.text(x0 + w / 2, y0 + h / 2, txt, ha='center', va='center', fontsize=8.5)
    box(0.0, 3.6, 1.7, 1.6, 'image\nrotated\nby $k\\phi$', '#e8e8e8')
    box(2.15, 3.6, 1.9, 1.6, 'ResNet18\n512\nfeatures', '#d6e6f7')
    box(4.5, 3.6, 2.9, 1.6, 'linear 512$\\rightarrow$2,\nnormalised to\n$(\\cos 2\\hat\\theta, \\sin 2\\hat\\theta)$', '#d6e6f7')
    box(7.85, 3.6, 2.1, 1.6, 'rotate\nback by\n$-k\\phi$', '#e8e8e8')
    for x0, x1 in [(1.7, 2.15), (4.05, 4.5), (7.4, 7.85)]:
        arrow(c, (x0 + 0.05, 4.4), (x1 - 0.05, 4.4), 'k', lw=0.8, ms=6)
    box(0.0, 0.3, 9.95, 2.4, 'loss $1 - \\cos 2\\Delta\\theta$, label moved with the image\n'
        'prediction is the mean over 8 rotations\n'
        'its length is $R_{\\mathrm{TTA}}$', '#fdf0e6')
    arrow(c, (8.9, 3.5), (8.9, 2.8), 'k', lw=0.8, ms=6)
    c.set_title('(c) Network, training and prediction', fontsize=9)
    plt.tight_layout(w_pad=0.4); plt.savefig(OUT / 'axis_schematic.pdf'); plt.close()


def fig_uncertainty():
    """Median error and gross failure fraction of the kept axes against the fraction kept, for the mean axis of the
    five seeds ranked by the mean R_TTA and by the agreement of the seeds, and for the von Mises head ranked by its
    concentration, with an ideal and a random ranking."""
    D = ROOT / 'data'
    base = pd.read_csv(D / 'pa_cnn_cv.csv'); ids = base.idPNMain.values
    seeds = [base] + [pd.read_csv(D / f'pa_cnn_cv_s{s}.csv') for s in (1, 2, 3, 4)]
    al = [s.set_index('idPNMain').reindex(ids) for s in seeds]
    V = np.mean([np.exp(2j * np.radians(s.cnn_pa.values)) for s in al], 0)
    err = dpa((np.degrees(np.angle(V)) / 2) % 180, base.EPA.values)
    meanR = np.mean([s.cnn_R.values for s in al], 0)
    vm = pd.read_csv(D / 'pa_cnn_cv_vm_s0.csv').set_index('idPNMain').reindex(ids)
    dvm = dpa(vm.cnn_pa.values, base.EPA.values)
    rng = np.random.default_rng(3)
    curves = [(r'Mean $R_{\mathrm{TTA}}$', meanR, err, CNN, '-'), ('Agreement of the seeds', np.abs(V), err, ALT, '-'),
              (r'von Mises $\kappa$ (own axes)', vm.cnn_kappa.values, dvm, MOM, '--'),
              ('Ideal ranking', -err, err, HAND, ':'), ('Random ranking', None, err, MUTED, ':')]
    f, ax = plt.subplots(1, 2, figsize=(COL2, 2.5))
    keep = np.linspace(0.3, 1, 36); out = {}
    for lab, s, e, c, ls in curves:
        if s is None:  # random ranking: expectation over 500 permutations
            M = np.array([[np.median(e[o_[:int(round(k * len(e)))]]) for k in keep] for o_ in (rng.permutation(len(e)) for _ in range(500))])
            G = np.array([[np.mean(e[o_[:int(round(k * len(e)))]] > 60) for k in keep] for o_ in (rng.permutation(len(e)) for _ in range(500))])
            med, gf = list(M.mean(0)), list(G.mean(0))
        else:
            o = np.argsort(-s); med, gf = [], []
            for k in keep:
                sel = o[:int(round(k * len(o)))]; med.append(np.median(e[sel])); gf.append(np.mean(e[sel] > 60))
        ax[0].plot(keep, med, color=c, lw=1.6, ls=ls, label=lab); ax[1].plot(keep, np.array(gf) * 100, color=c, lw=1.6, ls=ls)
        out[lab] = dict(median_keep80=float(med[np.argmin(np.abs(keep - 0.8))]), gross_keep80=float(gf[np.argmin(np.abs(keep - 0.8))]))
    for a, yl in zip(ax, [r'Median $|\Delta\mathrm{PA}|$ (deg)', 'Misplaced by more than $60\\degree$ (per cent)']):
        a.set_xlabel('Fraction of labelled PNe kept'); a.set_ylabel(yl); a.set_xlim(0.3, 1)
    f.legend(*ax[0].get_legend_handles_labels(), loc='lower center', ncol=5, frameon=False, fontsize=7.5)
    plt.tight_layout(rect=(0, 0.09, 1, 1)); plt.savefig(OUT / 'uncertainty.pdf'); plt.close()
    json.dump(out, open(D / 'uncertainty_curves_rev7.json', 'w'), indent=1); print(out)

def fig_failures():
    """The eight largest labelled PNe misplaced by more than 60 deg in cross validation."""
    sys.path.insert(0, str(ROOT / 'scripts')); ds03 = importlib.import_module('03_dataset')
    cv = pd.read_csv(ROOT / 'data' / 'pa_cnn_cv.csv')
    cv['d'] = dpa(cv.cnn_pa, cv.EPA)
    s = cv[cv.d > 60].sort_values('MajDiam', ascending=False).head(8)
    f, ax = plt.subplots(2, 4, figsize=(COL2, 3.7))
    for a, (_, r) in zip(ax.ravel(), s.iterrows()):
        img = ds03.image(ROOT / 'data' / 'cutouts' / r.file, r.kcont)
        n = img.shape[0]; c0 = int(n / 2 - n * 1.4 / 6); c1 = int(n / 2 + n * 1.4 / 6)
        a.imshow(img[c0:c1, c0:c1, 1], cmap='gray_r', origin='upper')
        m = (c1 - c0) / 2
        for pa, col, ls in [(r.cnn_pa, CNN, '-'), (r.EPA, 'k', ':'), (r.pa_eq, MOM, '-')]:
            t = np.radians(pa); dx, dy = -np.sin(t) * m * 0.85, -np.cos(t) * m * 0.85
            a.plot([m - dx, m + dx], [m - dy, m + dy], color=col, lw=1.3, ls=ls)
        a.set_title(f'{r.Name} ({r.mainClass}) $R_{{\\mathrm{{TTA}}}}$ {r.cnn_R:.2f}', fontsize=7.5)
        a.set_xticks([]); a.set_yticks([])
    plt.tight_layout(); plt.savefig(OUT / 'failures.pdf'); plt.close()
    print(s[['Name', 'mainClass', 'MajDiam', 'survey', 'cnn_R', 'd']].to_string())


def fig_injection():
    """(a) Probability of a V statistic at the plane no larger than the observed one against the aligned fraction for
    the bipolars outside the bulge region, for alignments along the plane. (b) The 95 per cent limit against the
    concentration of the aligned part."""
    D = ROOT / 'data'; r6 = json.load(open(D / 'rev6.json')); rng = np.random.default_rng(5)
    x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
    base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
    gpa = x[base & (x.mainClass == 'B') & ~x.bulge_sky].gpa.values; n = len(gpa)
    mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
    err = ((cv[mres].cnn_pa - cv[mres].EPA + 90) % 180 - 90).dropna().values
    Vat = lambda pa: np.mean(np.cos(np.radians(2 * pa) - np.pi), axis=-1)
    Vobs = Vat(gpa); fg = np.round(np.arange(0, 0.701, 0.01), 2); NS = 10000
    def curve(R_al, e):
        s2 = np.sqrt(-2 * np.log(R_al)); P = []
        for f_ in fg:
            al = rng.random((NS, n)) < f_
            pa = np.where(al, 90 + np.degrees(rng.normal(0, s2, (NS, n))) / 2, rng.uniform(0, 180, (NS, n)))
            if e is not None:
                pa = pa + rng.choice(e, (NS, n))
            P.append(np.mean(Vat(pa) <= Vobs))
        return P
    f, ax = plt.subplots(1, 2, figsize=(COL2, 2.6))
    a = ax[0]
    r9 = json.load(open(D / 'rev9.json')); lo, hi = r9['binary_share']['strict']['frac']; lo2, hi2 = json.load(open(D / 'rev10.json'))['binary_share']['frac']
    a.axvspan(lo2, hi2, color='#efefef', lw=0); a.axvspan(lo, hi, color='#d6d6d6', lw=0)
    a.text((lo + hi) / 2, 0.98, 'strict share', rotation=90, ha='center', va='top', fontsize=7, color=HAND)
    a.text((lo2 + hi2) / 2, 0.98, 'HASH class share', rotation=90, ha='center', va='top', fontsize=7, color=HAND)
    for R_al, e, col, ls, lab in [(0.83, err, CNN, '-', 'Tight, with errors'), (0.83, None, CNN, '--', 'Tight, no errors'),
                                  (0.38, err, MOM, '-', 'Weaker, with errors')]:
        a.plot(fg, curve(R_al, e), color=col, ls=ls, lw=2, label=lab)
    a.axhline(0.05, color=HAND, ls=':', lw=1.0)
    a.set_xlabel('Aligned fraction $f$'); a.set_ylabel(r'$P(V \leq V_\mathrm{obs})$')
    a.set_xlim(0, 0.7); a.set_ylim(0, 1.02)
    a.legend(frameon=False, fontsize=7, loc='upper right', handlelength=1.6)
    a.set_title('(a)', fontsize=9, loc='left')
    b = ax[1]
    for kk, col, lab in [('plane', CNN, 'along the plane'), ('at107_scored_plane', MOM, r'about $107.5\degree$')]:
        pts = sorted((float(k), v[kk]) for k, v in r9['f95_vs_rho'].items() if v[kk] is not None)
        b.plot([p_[0] for p_ in pts], [p_[1] for p_ in pts], color=col, lw=1.8, marker='o', ms=3, label=lab)
    for xv, txt, y0 in [(0.83, 'Tan et al.', 0.22), (0.38, 'Rees & Zijlstra', 0.42)]:
        b.axvline(xv, color=MUTED, ls=':', lw=0.8)
        b.text(xv - 0.012, y0, txt, fontsize=7, color=HAND, ha='right', va='bottom', rotation=90)
    b.set_xlabel('Concentration $R$ of the aligned part'); b.set_ylabel('95 per cent upper limit on $f$')
    b.set_xlim(0.15, 1.0); b.set_ylim(0, 0.85); b.legend(frameon=False, fontsize=7.5, loc='upper right')
    b.set_title('(b)', fontsize=9, loc='left')
    plt.tight_layout(); plt.savefig(OUT / 'injection.pdf'); plt.close()

def _circ(pa, mu=90):
    from scipy import stats
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); R = np.hypot(np.cos(th).mean(), np.sin(th).mean())
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return n, R, stats.norm.sf(V * np.sqrt(2 * n))


def fig_alignment():
    """Distributions of Galactic position angles. The panel of the PNe of Tan et al. shows no V test probability,
    since their direction was found from the same objects."""
    x = pd.read_csv(ROOT / 'data' / 'pa_final.csv')
    tan = pd.read_csv(ROOT / 'data' / 'tan_comparison.csv')
    base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
    B = base & (x.mainClass == 'B'); E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
    rzc = x[(x.morph_class == 'bipolar') & base]
    panels = [('Rees & Zijlstra bipolars, visual', rzc.gpa_rz, HAND, 90, False),
              ('Same objects, network', rzc.gpa, CNN, 90, False),
              ('Tan et al. binary and high adf PNe', tan[tan.GPA_deg.notna()].gpa, CNN, 107.5, False),  # visual axes added in grey below
              ('Bulge region, other bipolars', x[B & x.bulge_sky & x.morph_class.isna()].gpa, CNN, 90, True),
              ('Outside bulge region, bipolars', x[B & ~x.bulge_sky].gpa, CNN, 90, True),
              ('Outside bulge region, ellipticals', x[E & ~x.bulge_sky].gpa, CNN, 90, True)]
    f, ax = plt.subplots(2, 3, figsize=(COL2, 4.0))
    bins = np.arange(0, 181, 15)
    for a, (tt, v, c, mu, showp) in zip(ax.flat, panels):
        v = np.asarray(pd.Series(v).dropna())
        n, R, pv = _circ(v, mu)
        if tt.startswith('Tan et al.'):
            a.hist(np.asarray(tan[tan.GPA_deg.notna()].GPA_deg) % 180, bins=bins, density=True, color=HAND, alpha=0.45, edgecolor='white', linewidth=1)
            a.hist(v, bins=bins, density=True, histtype='step', color=c, linewidth=1.8)
        else:
            a.hist(v, bins=bins, density=True, color=c, edgecolor='white', linewidth=1)
        a.axhline(1 / 180, color=MUTED, ls='--', lw=0.8)
        a.axvline(mu, color=MUTED, lw=0.6)
        a.set_xlim(0, 180); a.set_xticks([0, 45, 90, 135, 180]); a.set_yticks([])
        a.set_xlabel('GPA (deg)', fontsize=8)
        a.set_title(tt, fontsize=8, pad=14)
        lab = f"$n$ = {n}   $R$ = {R:.2f}"
        if showp:
            lab += f"   $p_V$ = {pv:.2g}" if pv >= 0.001 else "   $p_V$ < 0.001"
        a.text(0.5, 1.02, lab, transform=a.transAxes, ha='center', va='bottom', fontsize=7.5)
    for a in ax[:, 0]:
        a.set_ylabel('Relative frequency', fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'alignment.pdf'); plt.close()


def fig_reliability():
    """Reliability of the bipolar probability of the classification network against the HASH bipolar class."""
    D = ROOT / 'data'; rel = json.load(open(D / 'rev8_misc.json'))['reliability_P_B']
    rel = [r for r in rel if r['n']]
    f, a = plt.subplots(figsize=(COL2 * 0.45, 2.6))
    a.plot([0, 1], [0, 1], color=MUTED, ls=':', lw=1)
    a.plot([r['mean_p'] for r in rel], [r['frac_B'] for r in rel], color=CNN, marker='o', ms=4, lw=1.6)
    for r in rel:
        a.text(r['mean_p'], 1.04, str(r['n']), fontsize=6.5, ha='center', va='center', color=HAND)
    a.set_xlabel(r'Mean $P_{\mathrm{B}}$ in the bin'); a.set_ylabel('Fraction of HASH bipolars')
    a.set_xlim(0, 1); a.set_ylim(0, 1.1); a.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.tight_layout(); plt.savefig(OUT / 'reliability.pdf'); plt.close()


if __name__ == '__main__':
    for name in sys.argv[1:]:
        globals()[f'fig_{name}']()
