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
torch.manual_seed(SEED); np.random.seed(SEED)


def load_images(t):
    imgs = []
    for _, r in t.iterrows():
        img = ds03.image(ROOT / 'data' / 'cutouts' / r.file, r.kcont)  # north up, uint8 HxWx3
        n = img.shape[0]
        c = n // 6  # FOV is 3 MajDiam: keep the central two thirds
        img = cv2.resize(img[c:n - c, c:n - c], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        imgs.append(img)
    x = torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).float() / 255
    if CH == 'ha':
        x = x[:, :1].repeat(1, 3, 1, 1)
    return x


def augment(x, pa):
    if AUG == 'none':
        return x, pa
    phi = torch.rand(len(x)) * 360
    flip = torch.rand(len(x)) < 0.5
    out = torch.empty_like(x)
    for i in range(len(x)):
        xi = TF.hflip(x[i]) if flip[i] else x[i]
        out[i] = TF.rotate(xi, float(phi[i]), interpolation=TF.InterpolationMode.BILINEAR)
    if AUG == 'rot_nolabel':  # ablation: rotate and flip the image but keep the original label
        return out, pa
    pa2 = torch.where(flip, -pa, pa) + phi
    return out, pa2


def model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(512, 2)
    return m.to(DEV)


def loss_fn(out, pa):
    v = out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)
    th = torch.deg2rad(2 * pa).to(DEV)
    return (1 - (v[:, 0] * torch.cos(th) + v[:, 1] * torch.sin(th))).mean()


def predict(m, x, tta=8):
    """Average axis vectors over rotations (rotation de-applied), which also gives a spread."""
    m.eval(); vs = []
    with torch.no_grad():
        for k in range(tta):
            phi = 360 * k / tta
            xr = torch.stack([TF.rotate(xi, phi, interpolation=TF.InterpolationMode.BILINEAR) for xi in x])
            o = torch.cat([m(xr[i:i + 64].to(DEV)).cpu() for i in range(0, len(xr), 64)])
            ang = torch.rad2deg(torch.atan2(o[:, 1], o[:, 0])) / 2 - phi  # undo rotation
            vs.append(torch.stack([torch.cos(torch.deg2rad(2 * ang)), torch.sin(torch.deg2rad(2 * ang))], 1))
    v = torch.stack(vs).mean(0)
    pa = (torch.rad2deg(torch.atan2(v[:, 1], v[:, 0])) / 2) % 180
    return pa.numpy(), v.norm(dim=1).numpy()  # resultant length 1 = all rotations agree


def train(x, pa):
    m = model()
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
    if mode == 'cv':
        blocks = (lab.Glon // 10).astype(int).to_numpy().copy()
        blocks[blocks == 35] = 0  # l = 350-360 and 0-10 deg are neighbours across the Galactic centre
        ub = np.random.default_rng(0).permutation(np.unique(blocks))
        fold = {b: i % 5 for i, b in enumerate(ub)}
        lab['fold'] = [fold[b] for b in blocks]
        lab['cnn_pa'] = np.nan; lab['cnn_R'] = np.nan
        for f in range(5):
            tr, te = (lab.fold != f).values, (lab.fold == f).values
            m = train(x[tr], pa[tr])
            p, R = predict(m, x[te])
            lab.loc[te, 'cnn_pa'] = p; lab.loc[te, 'cnn_R'] = R
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
        m = train(x, pa)
        torch.save(m.state_dict(), ROOT / 'data' / f'pa_cnn_full{TAG}.pt')
        allx = load_images(t)
        p, R = predict(m, allx)
        t[['idPNMain', 'file', 'mainClass', 'MajDiam', 'EPA']].assign(cnn_pa=p, cnn_R=R).to_csv(
            ROOT / 'data' / f'pa_cnn_all{TAG}.csv', index=False)
