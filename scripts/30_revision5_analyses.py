"""Revision 5 analyses (third referee report on the MLST version). Output: data/rev5.json

a. composition of the five cross validation folds (size, survey, class, median diameter) and their medians
b. SHS preference: equatorial and Galactic concentration of the stellar SHS axes, and by declination and longitude
c. spatial correlation of the axes of the bipolars outside the bulge region: pair correlation of doubled angles at
   small separations, and the V test probability from a permutation of whole 10 deg longitude blocks
d. exploratory subsamples with one sided V test probabilities (supplementary table)
e. independent checks of the statistics code: Rayleigh and V tests against astropy.stats.circstats, Holm against
   statsmodels, the bootstrap against scipy.stats.bootstrap, and the injection null against the analytic V test
"""
import json, pathlib
import numpy as np, pandas as pd
from scipy import stats
from astropy.stats import circstats
from statsmodels.stats.multitest import multipletests
from astropy.coordinates import SkyCoord
import astropy.units as u
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(51)
dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)


def circ(pa, mu=90):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
out = {}

# ---------- a. folds ----------
fo = {}
for f, g in cv.groupby('fold'):
    fo[int(f)] = dict(n=len(g), median=float(g.d_cnn.median()), median_MajDiam=float(g.MajDiam.median()),
                      frac_B=float((g.mainClass == 'B').mean()), surveys=g.survey.value_counts().to_dict(),
                      glon_range=[float(g.Glon.min()), float(g.Glon.max())])
fo['spearman_fold_median_vs_size'] = float(stats.spearmanr([fo[k]['median_MajDiam'] for k in range(5)],
                                                            [fo[k]['median'] for k in range(5)]).correlation)
# spread of fold medians expected for random folds of the same sizes
sizes = [fo[k]['n'] for k in range(5)]; obs_range = max(fo[k]['median'] for k in range(5)) - min(fo[k]['median'] for k in range(5))
d = cv.d_cnn.values; rr = []
for _ in range(5000):
    pmt = rng.permutation(len(d)); meds = []; o = 0
    for n_ in sizes:
        meds.append(np.median(d[pmt[o:o + n_]])); o += n_
    rr.append(max(meds) - min(meds))
fo['observed_range'] = float(obs_range); fo['random_range_median'] = float(np.median(rr))
fo['frac_random_range_ge_observed'] = float(np.mean(np.array(rr) >= obs_range))
out['folds'] = fo

# ---------- b. SHS preference ----------
st = x[(x.mainClass == 'S') & (x.survey == 'SHS') & x.MajDiam.notna()].copy()
b = dict(equatorial=circ(st.pa, 113), galactic=circ(st.gpa))
for lo, hi in [(-90, -45), (-45, -30), (-30, 10)]:
    b[f'dec {lo} to {hi}'] = circ(st[(st.DDECJ2000 >= lo) & (st.DDECJ2000 < hi)].pa, 113)
lon = ((st.Glon + 180) % 360) - 180
b['l < 0'] = circ(st[lon < 0].pa, 113); b['l >= 0'] = circ(st[lon >= 0].pa, 113)
# all SHS axes of large selected PNe, whose axes are well defined, against the same direction
big = x[(x.survey == 'SHS') & (x.MajDiam >= 20) & (x.R_tta >= 0.8)]
b['SHS selected >= 20 arcsec, equatorial'] = circ(big.pa, 113)
b['SHS stellar median R_TTA'] = float(st.R_tta.median())
out['shs'] = b

# ---------- c. spatial correlation ----------
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
ob = x[base & (x.mainClass == 'B') & ~x.bulge_sky].reset_index(drop=True)
c = SkyCoord(ob.Glon.values * u.deg, ob.Glat.values * u.deg, frame='galactic')
sep = c[:, None].separation(c[None, :]).deg
th = np.radians(2 * ob.gpa.values)
cosd = np.cos(th[:, None] - th[None, :])
iu = np.triu_indices(len(ob), 1)
sc = {}
for lo, hi in [(0, 1), (1, 3), (3, 10), (10, 180)]:
    m = (sep[iu] >= lo) & (sep[iu] < hi)
    v = cosd[iu][m]
    sc[f'{lo}-{hi} deg'] = dict(pairs=int(m.sum()), mean_cos2d=float(v.mean()), sem=float(v.std() / np.sqrt(max(m.sum(), 1))))
# block permutation: each 10 deg longitude block gets one random rotation of all its axes
blocks = (ob.Glon // 10).astype(int).values
Vobs = np.mean(np.cos(th - np.pi)); ub = np.unique(blocks); Vs = []
for _ in range(20000):
    rot = dict(zip(ub, rng.uniform(0, 2 * np.pi, len(ub))))
    Vs.append(np.mean(np.cos(th + np.array([rot[k] for k in blocks]) - np.pi)))
sc['n_blocks'] = int(len(ub)); sc['V_obs'] = float(Vobs)
sc['V_p_block_permutation'] = float(np.mean(np.array(Vs) >= Vobs)); sc['V_p_analytic'] = circ(ob.gpa)['V_p']
out['spatial'] = sc

# ---------- d. exploratory subsamples ----------
B = base & (x.mainClass == 'B'); E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
obb = x[B & x.bulge_sky & x.morph_class.isna()]
ex = {
    'bulge region, all bipolars': x[B & x.bulge_sky].gpa,
    'bulge region, other bipolars below 35 arcsec': obb[obb.MajDiam < 35].gpa,
    'bulge region, other bipolars of the bulge component': obb[obb.Component == 'Bulge'].gpa,
    'bulge region, other bipolars beyond 5 kpc': obb[obb.rgeo_kpc > 5].gpa,
    'bulge region, other bipolars 6 to 10 kpc': obb[(obb.rgeo_kpc > 6) & (obb.rgeo_kpc < 10)].gpa,
    'bulge region, bipolars P_B >= 0.7': x[B & x.bulge_sky & (x.p_B >= 0.7)].gpa,
    'outside, bipolars P_B >= 0.5': x[B & ~x.bulge_sky & (x.p_B >= 0.5)].gpa,
    'outside, bipolars P_B >= 0.7': x[B & ~x.bulge_sky & (x.p_B >= 0.7)].gpa,
    'outside, bipolars P_B >= 0.9': x[B & ~x.bulge_sky & (x.p_B >= 0.9)].gpa,
    'outside, bipolars thin or thick disc': x[B & ~x.bulge_sky & x.Component.isin(['Thin Disk', 'Thick Disk'])].gpa,
    'outside, bipolars R_TTA >= 0.9': x[B & ~x.bulge_sky & (x.R_tta >= 0.9)].gpa,
    'outside, bipolars R_TTA >= 0.95': x[B & ~x.bulge_sky & (x.R_tta >= 0.95)].gpa,
    'outside, bipolars IPHAS': x[B & ~x.bulge_sky & (x.survey == 'IPHAS')].gpa,
    'outside, bipolars VPHAS+': x[B & ~x.bulge_sky & (x.survey == 'VPHAS')].gpa,
    'outside, bipolars |l| < 30 deg': x[B & ~x.bulge_sky & (np.abs(((x.Glon + 180) % 360) - 180) < 30)].gpa,
}
exr = {k: circ(v) for k, v in ex.items()}
out['exploratory'] = exr
out['exploratory_min_p'] = min(v['V_p'] for v in exr.values())
out['exploratory_n_tests'] = len(exr)
out['components'] = x.Component.value_counts().to_dict()

# ---------- e. verification ----------
ver = {}
for nm, v in [('outside B', ob.gpa.values), ('RZ13 selected network', x[(x.morph_class == 'bipolar') & base].gpa.values)]:
    ours = circ(v)
    a2 = np.radians(2 * v)
    ver[nm] = dict(rayleigh_ours=ours['rayleigh_p'], rayleigh_astropy=float(circstats.rayleightest(a2)),
                   V_p_ours=ours['V_p'], V_p_astropy=float(circstats.vtest(a2, mu=np.radians(180))))
fam = json.load(open(D / 'rev4.json'))['holm']['tests']
ps = np.array([v['p'] for v in fam.values()]); ours = np.array([v['p_holm'] for v in fam.values()])
ver['holm_max_abs_diff_vs_statsmodels'] = float(np.max(np.abs(multipletests(ps, method='holm')[1] - ours)))
v = x[(x.morph_class == 'bipolar') & base].gpa_rz.dropna().values
Rf = lambda s, axis=-1: np.abs(np.exp(2j * np.radians(s)).mean(axis=axis))
bs = stats.bootstrap((v,), Rf, n_resamples=4000, confidence_level=0.68, method='percentile', random_state=3)
ver['bootstrap_R68_RZ13_visual_scipy'] = [float(bs.confidence_interval.low), float(bs.confidence_interval.high)]
inj = json.load(open(D / 'injection_final.json'))
r = inj['outside bulge region B | tight | none']
ver['injection_null_P_at_f0'] = r['grid']['0.0']; ver['injection_expected_1_minus_Vp_at_107.5'] = float(1 - circ(ob.gpa, 107.5)['V_p'])
out['verification'] = ver

json.dump(out, open(D / 'rev5.json', 'w'), indent=1, default=float)
for k, vv in out.items():
    print(k, json.dumps(vv, default=float)[:1600])
