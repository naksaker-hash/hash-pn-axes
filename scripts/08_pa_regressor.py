"""CNN regression of the symmetry-axis position angle, trained on the HASH hand-measured EPA.

The target is the axis (period 180 deg), encoded as (cos 2PA, sin 2PA); the loss is 1 - cos(2 dPA).
Images are the north-up PNGs (east left). Rotating the image counterclockwise by phi adds phi to the
PA (east of north); a left-right or up-down flip maps PA to -PA. Augmentation draws a random
rotation and flip for every sample, so each label is used at every orientation.

Usage: python 08_pa_regressor.py cv      5-fold CV grouped by 10 deg longitude block
       python 08_pa_regressor.py full    train on all labels, predict all cutouts
"""
import sys, pathlib, json, importlib
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
ds03 = importlib.import_module('03_dataset')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
SIZE, EPOCHS, BS = 160, 60, 32
import os
AUG = os.environ.get('PA_AUG', 'full')      # full | none | rot_nolabel
CH = os.environ.get('PA_CH', '3')           # 3 | ha
SEED = int(os.environ.get('PA_SEED', '0'))
TAG = os.environ.get('PA_TAG', '')          # suffix for output files
FRAC = float(os.environ.get('PA_FRAC', '1'))  # fraction of each training fold used (learning curve)
EXCL = os.environ.get('PA_EXCL', '')        # '' | mash | zero: labels left out of training (data/label_provenance.csv)
HEAD = os.environ.get('PA_HEAD', 'vec')     # vec: unit vector, loss 1 - cos 2dPA | vm: von Mises on 2PA with predicted kappa
ARCH = os.environ.get('PA_ARCH', 'resnet18')  # resnet18 | d8: D8 steerable CNN (e2cnn, Weiler & Cesa 2019) from scratch
FOLDSEED = int(os.environ.get('PA_FOLDSEED', '0'))  # seed of the assignment of longitude blocks to folds
PRETRAIN = os.environ.get('PA_PRETRAIN', '1') == '1'  # 0: ResNet18 trained from scratch (random initialisation)
ROTFULL = os.environ.get('PA_ROTFULL', '0') == '1'  # 1: rotate the full cutout and crop afterwards, so no padded corners appear
FULL = int(round(SIZE * 1.5))  # full cutout resampled so that its central two thirds are SIZE pixels
if ARCH == 'd8':
    SIZE = 161  # an odd size keeps the 90 deg rotations exact about the central pixel
torch.manual_seed(SEED); np.random.seed(SEED)


def load_images(t):
    imgs = []
    for _, r in t.iterrows():
        img = ds03.image(ROOT / 'data' / 'cutouts' / r.file, r.kcont)  # north up, uint8 HxWx3
        n = img.shape[0]
        if ROTFULL:
            img = cv2.resize(img, (FULL, FULL), interpolation=cv2.INTER_AREA)
        else:
            c = n // 6  # FOV is 3 MajDiam: keep the central two thirds
            img = cv2.resize(img[c:n - c, c:n - c], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    x = torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    if CH == 'ha':
        x = x[:, :1].repeat(1, 3, 1, 1)
    return x


def crop(x):
    """Central SIZE pixels of a full resampled cutout (identity for pre-cropped inputs)."""
    if x.shape[-1] == SIZE:
        return x
    o = (x.shape[-1] - SIZE) // 2
    return x[..., o:o + SIZE, o:o + SIZE]


def augment(x, pa):
    if AUG == 'none':
        return crop(x), pa
    phi = torch.rand(len(x)) * 360
    flip = torch.rand(len(x)) < 0.5
    out = torch.empty_like(x)
    for i in range(len(x)):
        xi = TF.hflip(x[i]) if flip[i] else x[i]
        out[i] = TF.rotate(xi, float(phi[i]), interpolation=TF.InterpolationMode.BILINEAR)
    out = crop(out)
    if AUG == 'rot_nolabel':  # ablation: rotate and flip the image but keep the original label
        return out, pa
    pa2 = torch.where(flip, -pa, pa) + phi
    return out, pa2


def model():
    if ARCH == 'd8':
        return D8Net().to(DEV)
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if PRETRAIN else None)
    m.fc = nn.Linear(512, 3 if HEAD == 'vm' else 2)
    return m.to(DEV)


class D8Net(nn.Module):
    """D8 steerable CNN whose output is one frequency 2 field averaged over the image, i.e. an axis that rotates
    with the image by construction. A fixed map (sigma, beta), found numerically once, aligns the field with
    the doubled position angle convention of the labels."""
    def __init__(self):
        super().__init__()
        from e2cnn import gspaces, nn as enn
        g = gspaces.FlipRot2dOnR2(N=8)
        self.enn = enn
        ft = lambda n: enn.FieldType(g, n * [g.regular_repr])
        self.tin = enn.FieldType(g, 3 * [g.trivial_repr])
        layers, prev = [], self.tin
        for n, k in [(4, 7), (8, 5), (16, 5), (16, 3), (32, 3)]:
            out = ft(n)
            layers += [enn.R2Conv(prev, out, k, padding=k // 2), enn.InnerBatchNorm(out), enn.ReLU(out)]
            if n != 32:
                layers.append(enn.PointwiseAvgPoolAntialiased(out, sigma=0.66, stride=2))
            prev = out
        self.tout = enn.FieldType(g, [g.irrep(1, 2)])
        layers.append(enn.R2Conv(prev, self.tout, 1))
        self.body = enn.SequentialModule(*layers)
        self.register_buffer('sigma', torch.tensor(1.0)); self.register_buffer('beta', torch.tensor(0.0))

    def forward(self, x):
        y = self.body(self.enn.GeometricTensor(x, self.tin)).tensor.mean((2, 3))
        a = self.sigma * torch.atan2(y[:, 1], y[:, 0]) + self.beta
        r = y.norm(dim=1)
        return torch.stack([r * torch.cos(a), r * torch.sin(a)], 1)


def blobs(n):
    """Elongated Gaussian test images at random angles, used to calibrate the D8 output."""
    yy, xx = np.mgrid[:SIZE, :SIZE] - (SIZE - 1) / 2; out = []
    for th in np.random.default_rng(1).uniform(0, np.pi, n):
        u = xx * np.cos(th) + yy * np.sin(th); v = -xx * np.sin(th) + yy * np.cos(th)
        im = np.exp(-0.5 * ((u / (SIZE / 5)) ** 2 + (v / (SIZE / 14)) ** 2)); out.append(np.stack([im, 0.7 * im, 0.3 * im]))
    return torch.tensor(np.array(out), dtype=torch.float32)


def calibrate_d8(m):
    """Find the sign and offset that map the frequency 2 field to the label convention: a counterclockwise
    rotation by phi adds 2 phi to the doubled angle and a horizontal flip negates it."""
    m.eval(); x = blobs(16)
    ang = lambda z: torch.atan2(z[:, 1], z[:, 0])
    with torch.no_grad():
        m.sigma.fill_(1.0); m.beta.fill_(0.0)
        a0 = ang(m(x.to(DEV)).cpu())
        a90 = ang(m(torch.rot90(x, 1, (2, 3)).to(DEV)).cpu())   # rot90 over (H, W) is counterclockwise on screen
        af = ang(m(TF.hflip(x).to(DEV)).cpu())
    wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    # a 90 deg rotation adds 180 deg to the doubled angle for either sense, so test the sense with 45 deg
    with torch.no_grad():
        a45 = ang(m(TF.rotate(x, 45, interpolation=TF.InterpolationMode.BILINEAR).to(DEV)).cpu())
    sigma = 1.0  # exact test with a frequency 1 field: TF.rotate by +90 deg turns it by +90 deg, so the sense is positive
    c = np.angle(np.mean(np.exp(1j * (af + a0).numpy())))   # af = -a0 + c
    m.sigma.fill_(sigma); m.beta.fill_(float(-sigma * c / 2))
    with torch.no_grad():
        b0 = ang(m(x.to(DEV)).cpu()); bf = ang(m(TF.hflip(x).to(DEV)).cpu()); b45 = ang(m(TF.rotate(x, 45, interpolation=TF.InterpolationMode.BILINEAR).to(DEV)).cpu())
    print('D8 calibration sigma', sigma, 'rot45 err deg', float(np.degrees(np.median(np.abs(wrap((b45 - b0 - np.pi / 2).numpy()))))),
          'flip err deg', float(np.degrees(np.median(np.abs(wrap((bf + b0).numpy()))))), 'rot90 check', float(np.degrees(np.median(np.abs(wrap((a90 - a0 - np.pi).numpy()))))), flush=True)
    # 45 deg rotations are only approximate on the pixel grid for random weights, so only the exact checks are enforced
    if np.degrees(np.median(np.abs(wrap((bf + b0).numpy())))) > 1 or np.degrees(np.median(np.abs(wrap((a90 - a0 - np.pi).numpy())))) > 1:
        raise RuntimeError('D8 calibration failed')
    m.train()


def loss_fn(out, pa):
    th = torch.deg2rad(2 * pa).to(DEV)
    v = out[:, :2] / out[:, :2].norm(dim=1, keepdim=True).clamp_min(1e-6)
    c = v[:, 0] * torch.cos(th) + v[:, 1] * torch.sin(th)
    if HEAD == 'vm':  # negative log likelihood of a von Mises distribution of 2PA
        k = nn.functional.softplus(out[:, 2]) + 1e-3
        return (-k * c + torch.log(torch.special.i0e(k)) + k).mean()
    return (1 - c).mean()


def predict(m, x, tta=8, flips=False, kappa=False):
    """Average axis vectors over rotations (rotation de-applied), which also gives a spread. With flips the
    reflected image is added at every rotation (the D8 set of 16 copies). With kappa the mean predicted von
    Mises concentration is returned as well."""
    m.eval(); vs = []; ks = []
    with torch.no_grad():
        for fl in ([False, True] if flips else [False]):
            xf = torch.stack([TF.hflip(xi) for xi in x]) if fl else x
            for k in range(tta):
                phi = 360 * k / tta
                xr = crop(torch.stack([TF.rotate(xi, phi, interpolation=TF.InterpolationMode.BILINEAR) for xi in xf]))
                o = torch.cat([m(xr[i:i + 64].to(DEV)).cpu() for i in range(0, len(xr), 64)])
                ang = torch.rad2deg(torch.atan2(o[:, 1], o[:, 0])) / 2 - phi  # undo rotation
                if fl:
                    ang = -ang  # undo the reflection, which negates the position angle
                vs.append(torch.stack([torch.cos(torch.deg2rad(2 * ang)), torch.sin(torch.deg2rad(2 * ang))], 1))
                if o.shape[1] > 2:
                    ks.append(nn.functional.softplus(o[:, 2]) + 1e-3)
    v = torch.stack(vs).mean(0)
    pa = (torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) / 2) % 180
    if kappa:
        return pa.numpy(), v.norm(dim=1).numpy(), (torch.stack(ks).mean(0).numpy() if ks else np.full(len(x), np.nan))
    return pa.numpy(), v.norm(dim=1).numpy()  # resultant length 1 = all rotations agree


def train(x, pa):
    m = model()
    if ARCH == 'd8':
        calibrate_d8(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(len(x))
        for i in range(0, len(x), BS):
            b = idx[i:i + BS]
            xb, pb = augment(x[b], pa[b])
            opt.zero_grad()
            l = loss_fn(m(xb.to(DEV)), pb)
            l.backward(); opt.step()
        sched.step()
    return m


dpa = lambda a, b: np.abs((a - b + 90) % 180 - 90)

if __name__ == '__main__':
    mode = sys.argv[1]
    t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
    m0 = pd.read_csv(ROOT / 'data' / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain')
    m0['EPA'] = pd.to_numeric(m0.EPA, errors='coerce'); m0['idPNMain'] = m0.idPNMain.astype(int)
    t = t.merge(m0[['idPNMain', 'EPA']], on='idPNMain')
    lab = t[t.EPA.notna() & (t.MajDiam >= 4)].reset_index(drop=True)
    x = load_images(lab); pa = torch.tensor(lab.EPA.values, dtype=torch.float32)
    keep = np.ones(len(lab), bool)
    if EXCL:
        prov = pd.read_csv(ROOT / 'data' / 'label_provenance.csv').set_index('idPNMain')
        src = prov.loc[lab.idPNMain]
        keep = ~(src.label_source == 'MASH').values if EXCL == 'mash' else ~src.suspect_zero.values
        print('labels left out of training', int((~keep).sum()), flush=True)
    if mode == 'cv':
        blocks = (lab.Glon // 10).astype(int).to_numpy().copy()
        blocks[blocks == 35] = 0  # l = 350-360 and 0-10 deg are neighbours across the Galactic centre
        ub = np.random.default_rng(FOLDSEED).permutation(np.unique(blocks))
        fold = {b: i % 5 for i, b in enumerate(ub)}
        lab['fold'] = [fold[b] for b in blocks]
        lab['cnn_pa'] = np.nan; lab['cnn_R'] = np.nan
        for f in range(5):
            tr, te = (lab.fold != f).values & keep, (lab.fold == f).values
            if FRAC < 1:
                idx = np.flatnonzero(tr); sub = np.random.default_rng(100 + f).choice(idx, int(round(FRAC * len(idx))), replace=False)
                tr = np.zeros(len(lab), bool); tr[sub] = True
            m = train(x[tr], pa[tr])
            p, R, kap = predict(m, x[te], kappa=True)
            lab.loc[te, 'cnn_pa'] = p; lab.loc[te, 'cnn_R'] = R; lab.loc[te, 'cnn_kappa'] = kap
            pd8, Rd8 = predict(m, x[te], flips=True)
            lab.loc[te, 'cnn_pa_d8'] = pd8; lab.loc[te, 'cnn_R_d8'] = Rd8
            for k in (1, 4):
                pk, _ = predict(m, x[te], tta=k)
                lab.loc[te, f'cnn_pa_tta{k}'] = pk
            print(f'fold {f}: n {te.sum()} median dPA {np.median(dpa(p, lab.EPA[te])):.1f}', flush=True)
        lab['d_cnn'] = dpa(lab.cnn_pa, lab.EPA); lab['d_mom'] = dpa(lab.pa_eq, lab.EPA)
        res = {}
        for cl in ['B', 'E', 'all']:
            s = lab if cl == 'all' else lab[lab.mainClass == cl]
            res[cl] = dict(n=len(s), cnn_median=float(s.d_cnn.median()), cnn_gt60=float((s.d_cnn > 60).mean()),
                           mom_median=float(s.d_mom.median()), mom_gt60=float((s.d_mom > 60).mean()))
            print(cl, res[cl])
        json.dump(res, open(ROOT / 'data' / f'pa_cnn_cv{TAG}.json', 'w'), indent=1)
        lab.to_csv(ROOT / 'data' / f'pa_cnn_cv{TAG}.csv', index=False)
    else:
        m = train(x[keep], pa[keep])
        torch.save(m.state_dict(), ROOT / 'data' / f'pa_cnn_full{TAG}.pt')
        allx = load_images(t)
        p, R = predict(m, allx)
        t[['idPNMain', 'file', 'mainClass', 'MajDiam', 'EPA']].assign(cnn_pa=p, cnn_R=R).to_csv(
            ROOT / 'data' / f'pa_cnn_all{TAG}.csv', index=False)
