"""Three dimensional injections of revision 8, run in parallel with the P(f) grids, the visible aligned fraction and
Monte Carlo errors saved. Output: data/rev8_inject3d.json. Same models as script 38 section c, finer grid in f."""
import json, pathlib, re, html as htmlmod
import numpy as np, pandas as pd
from scipy import stats
from scipy.special import i0e
from sklearn.metrics import roc_auc_score
from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
CUT = float(__import__('os').environ.get('CUT', '2'))  # minimum HASH major diameter (arcsec) of the selected axes

ROOT = pathlib.Path(__file__).resolve().parents[1]
D = ROOT / 'data'
rng = np.random.default_rng(71)
wrap = lambda a: (a + 90) % 180 - 90
dpa = lambda a, b: np.abs(wrap(a - b))
out = {}


def circ(pa, mu=90.0):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    th = np.radians(2 * pa); C, S = np.cos(th).sum(), np.sin(th).sum(); R = np.hypot(C, S) / n
    V = np.mean(np.cos(th - np.radians(2 * mu)))
    Rc = np.sqrt(max((n * R ** 2 - 1) / (n - 1), 0)) if n > 1 else np.nan
    return dict(n=n, mean=float((np.degrees(np.arctan2(S, C)) / 2) % 180), R=float(R), R_bias_corrected=float(Rc),
                rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                V=float(V), V_p=float(stats.norm.sf(V * np.sqrt(2 * n))))


Vat = lambda pa, mu: np.mean(np.cos(np.radians(2 * pa) - np.radians(2 * mu)), axis=-1)
x = pd.read_csv(D / 'pa_final.csv')
cv = pd.read_csv(D / 'pa_cnn_cv.csv')
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS')
err = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
Ec = float(np.mean(np.cos(np.radians(2 * err))))
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
ob = x[base & (x.mainClass == 'B') & ~x.bulge_sky]
gpa = ob.gpa.values; n = len(gpa)


def Pgrid(pa_obs, R_al, mu_inj, mu_test, fgrid, nsim, e=err):
    nn = len(pa_obs); Vobs = Vat(pa_obs, mu_test); sig2 = np.sqrt(-2 * np.log(R_al)); P = []
    for f in fgrid:
        al = rng.random((nsim, nn)) < f
        pa = np.where(al, mu_inj + np.degrees(rng.normal(0, sig2, (nsim, nn))) / 2, rng.uniform(0, 180, (nsim, nn)))
        pa = pa + rng.choice(e, (nsim, nn))
        P.append(np.mean(Vat(pa, mu_test) <= Vobs))
    return np.array(P), float(Vobs)


def crossing(fgrid, P, level=0.05):
    i = np.flatnonzero(P < level)
    if not len(i):
        return None
    if i[0] == 0:
        return float(fgrid[0])
    j = i[0]; return float(fgrid[j - 1] + (P[j - 1] - level) * (fgrid[j] - fgrid[j - 1]) / (P[j - 1] - P[j]))


# ---------- c. 3D injections with filtering ----------
R0 = 8.2
l = np.radians(ob.Glon.values); b = np.radians(ob.Glat.values)
dist_known = ob.rgeo_kpc.values
uvec = np.stack([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)], 1)
nhat = np.stack([-np.sin(b) * np.cos(l), -np.sin(b) * np.sin(l), np.cos(b)], 1)
ehat = np.stack([-np.sin(l), np.cos(l), np.zeros_like(l)], 1)
known = np.isfinite(dist_known); logd = np.log(dist_known[known])


def field_dir(dist, pitch=np.radians(11)):
    P3 = np.array([-R0, 0, 0]) + dist[:, None] * uvec
    ph = np.stack([-P3[:, 1], P3[:, 0], np.zeros(n)], 1); ph /= np.linalg.norm(ph, axis=1, keepdims=True)
    rh = np.stack([P3[:, 0], P3[:, 1], np.zeros(n)], 1); rh /= np.linalg.norm(rh, axis=1, keepdims=True)
    return np.cos(pitch) * ph + np.sin(pitch) * rh


def fisher(mu, kappa):
    k = len(mu); xi = rng.random(k); w = 1 + np.log(xi + (1 - xi) * np.exp(-2 * kappa)) / kappa
    ph = rng.uniform(0, 2 * np.pi, k); s = np.sqrt(np.clip(1 - w ** 2, 0, None))
    t1 = np.where((np.abs(mu[:, 0]) < 0.9)[:, None], np.cross(mu, [1, 0, 0]), np.cross(mu, [0, 1, 0])); t1 /= np.linalg.norm(t1, axis=1, keepdims=True)
    t2 = np.cross(mu, t1)
    return w[:, None] * mu + s[:, None] * (np.cos(ph)[:, None] * t1 + np.sin(ph)[:, None] * t2)


def girdle(kappa_b, k):
    """Axes in the Galactic plane with random azimuth, latitude spread like a Fisher distribution about the plane."""
    az = rng.uniform(0, 2 * np.pi, k)
    lat = np.arcsin(np.clip(fisher(np.tile([1.0, 0, 0], (k, 1)), kappa_b)[:, 2], -1, 1))
    return np.stack([np.cos(lat) * np.cos(az), np.cos(lat) * np.sin(az), np.sin(lat)], 1)


def realisation(model, kappa, f, cut):
    """One sample of n observed bipolars: intrinsic axes are drawn for candidates at the observed positions, each is
    aligned with probability f, and candidates whose axis lies within the cut of the line of sight are discarded
    (they would not be classed as bipolar) until n visible axes are collected. Returns the GPAs and the visible
    aligned fraction."""
    dist = np.where(known, dist_known, np.exp(rng.choice(logd, n)))  # missing distances drawn from the known ones
    fd = field_dir(dist) if model == 'field' else None
    pa = np.full(n, np.nan); alv = np.zeros(n, bool); todo = np.ones(n, bool); cosc = np.cos(np.radians(cut))
    while todo.any():
        k = np.flatnonzero(todo); al = rng.random(len(k)) < f
        iso = rng.normal(size=(len(k), 3)); iso /= np.linalg.norm(iso, axis=1, keepdims=True)
        if model == 'field':
            ali = fisher(fd[k], kappa)
        elif model == 'pole':
            ali = fisher(np.tile([0, 0, 1.0], (len(k), 1)), kappa)
        else:
            ali = girdle(kappa, len(k))
        ax = np.where(al[:, None], ali, iso)
        vis = np.abs(np.sum(ax * uvec[k], 1)) < cosc  # angle to the line of sight larger than the cut
        kk = k[vis]; axv = ax[vis]
        pa[kk] = np.degrees(np.arctan2(np.sum(axv * ehat[kk], 1), np.sum(axv * nhat[kk], 1))) % 180
        alv[kk] = al[vis]; todo[kk] = False
    return pa, alv.mean()



V90obs = Vat(gpa, 90.0); Robs = circ(gpa)['R']
fgrid3 = np.round(np.arange(0, 0.6001, 0.025), 3)
NS = 1500


def run(cfg):
    global rng
    model, kappa, cut = cfg
    import zlib; rng = np.random.default_rng(zlib.crc32(f"{model}{kappa}{cut}".encode()))
    PV, PR, fvis = [], [], []
    for f in fgrid3:
        Vs, Rs, fv = [], [], []
        for _ in range(NS):
            pa, fa = realisation(model, kappa, f, cut)
            pa = pa + rng.choice(err, n); th = np.radians(2 * pa)
            Vs.append(np.mean(np.cos(th - np.pi))); Rs.append(np.abs(np.mean(np.exp(1j * th)))); fv.append(fa)
        Vs, Rs = np.array(Vs), np.array(Rs)
        st, ob_ = (Vs, V90obs) if model in ('field', 'girdle') else (-Vs, -V90obs)
        PV.append(np.mean(st <= ob_)); PR.append(np.mean(Rs <= Robs)); fvis.append(float(np.mean(fv)))
    PV, PR = np.array(PV), np.array(PR)
    res = {}
    for lab, P in (('V', PV), ('R', PR)):
        c = crossing(fgrid3, P)
        if c is None:
            res[lab] = None; continue
        slope = np.gradient(P, fgrid3)[int(np.argmin(np.abs(fgrid3 - c)))]
        se = float(np.sqrt(0.05 * 0.95 / NS) / abs(slope)) if slope != 0 else None
        res[lab] = dict(f95_intrinsic=c, se=se, f95_visible=float(np.interp(c, fgrid3, fvis)),
                        se_visible=float(se * np.gradient(fvis, fgrid3)[int(np.argmin(np.abs(fgrid3 - c)))]) if se else None)
    test = 'V test at the plane' if model != 'pole' else 'V test perpendicular to the plane'
    print(model, kappa, cut, res, flush=True)
    return f'{model} kappa {kappa:g} cut {cut}', dict(test=test, **res, f_grid=fgrid3.tolist(), P_V=PV.tolist(), P_R=PR.tolist(), f_visible=fvis)


if __name__ == '__main__':
    from multiprocessing import Pool
    cfgs = [(m, k, c) for m in ('field', 'pole', 'girdle') for k in (20.0, 5.0) for c in (30, 45, 60)]
    with Pool(6) as p:
        g3 = dict(p.map(run, cfgs))
    g3['meta'] = dict(n_samples=NS, grid_step=0.025, interpolation='linear crossing of 0.05')
    json.dump(g3, open(D / 'rev8_inject3d.json', 'w'), indent=1)
