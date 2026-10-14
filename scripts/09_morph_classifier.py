"""ResNet18 morphology classifier (R, E, B, S) on the north-up cutouts, 5-fold CV by longitude block.

Same images and augmentation as 08_pa_regressor.py (random rotation and flip; morphology is invariant).
Class-weighted cross entropy; 8-rotation test-time averaging of the softmax.
Output: data/morph_cv.csv (per-object probabilities), data/morph_cv.json (accuracy, confusion, macro F1)
"""
import sys, json, pathlib, importlib
import numpy as np, pandas as pd, torch, torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
pa = importlib.import_module('08_pa_regressor')
DEV, EPOCHS, BS = pa.DEV, 25, 32
CLASSES = ['R', 'E', 'B', 'S']
torch.manual_seed(0); np.random.seed(0)


def model():
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(512, len(CLASSES))
    return m.to(DEV)


def train(x, y):
    m = model()
    w = torch.tensor(len(y) / (len(CLASSES) * np.bincount(y.numpy(), minlength=len(CLASSES))), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w.to(DEV), label_smoothing=0.05)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    for ep in range(EPOCHS):
        m.train()
        idx = torch.randperm(len(x))
        for i in range(0, len(x), BS):
            b = idx[i:i + BS]
            xb, _ = pa.augment(x[b], torch.zeros(len(b)))
            opt.zero_grad()
            l = lossf(m(xb.to(DEV)), y[b].to(DEV))
            l.backward(); opt.step()
        sched.step()
    return m


def predict(m, x, tta=8):
    m.eval(); ps = []
    with torch.no_grad():
        for k in range(tta):
            xr = torch.stack([TF.rotate(xi, 360 * k / tta) for xi in x])
            ps.append(torch.cat([m(xr[i:i + 64].to(DEV)).softmax(1).cpu() for i in range(0, len(xr), 64)]))
    return torch.stack(ps).mean(0).numpy()


if __name__ == '__main__':
    t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
    t = t[t.mainClass.isin(CLASSES)].reset_index(drop=True)
    x = pa.load_images(t)
    y = torch.tensor([CLASSES.index(c) for c in t.mainClass])
    blocks = (t.Glon // 10).astype(int).values
    ub = np.random.default_rng(0).permutation(np.unique(blocks))
    fold = {b: i % 5 for i, b in enumerate(ub)}
    t['fold'] = [fold[b] for b in blocks]
    P = np.zeros((len(t), len(CLASSES)))
    for f in range(5):
        tr, te = (t.fold != f).values, (t.fold == f).values
        m = train(x[tr], y[tr])
        P[te] = predict(m, x[te])
        acc = (P[te].argmax(1) == y[te].numpy()).mean()
        print(f'fold {f}: n {te.sum()} acc {acc:.3f}', flush=True)
    for i, c in enumerate(CLASSES):
        t[f'p_{c}'] = P[:, i]
    t['cnn_class'] = [CLASSES[i] for i in P.argmax(1)]
    acc = float((t.cnn_class == t.mainClass).mean())
    conf = pd.crosstab(t.mainClass, t.cnn_class)
    f1 = []
    for c in CLASSES:
        tp = ((t.cnn_class == c) & (t.mainClass == c)).sum()
        prec = tp / max((t.cnn_class == c).sum(), 1); rec = tp / max((t.mainClass == c).sum(), 1)
        f1.append(2 * prec * rec / max(prec + rec, 1e-9))
    # accuracy by size: resolved objects only
    big = t.MajDiam >= 10
    res = dict(n=len(t), acc=acc, macro_f1=float(np.mean(f1)), f1=dict(zip(CLASSES, map(float, f1))),
               acc_maj_ge10=float((t[big].cnn_class == t[big].mainClass).mean()), n_maj_ge10=int(big.sum()),
               confusion=conf.to_dict())
    print(json.dumps({k: v for k, v in res.items() if k != 'confusion'}, indent=1)); print(conf)
    json.dump(res, open(ROOT / 'data' / 'morph_cv.json', 'w'), indent=1)
    t.drop(columns=[c for c in t.columns if c.startswith('Unnamed')]).to_csv(ROOT / 'data' / 'morph_cv.csv', index=False)
