"""Segmentation and morphology checks requested in referee round 1. Output: data/seg_morph_extra.json

Segmentation: stricter detection, the predicted mask must contain the image centre or have its centroid
within half the HASH major radius of it, for each YOLO26n seed (masks regenerated per seed by 05_measure).
Morphology: balanced accuracy of the classification network, and a baseline gradient boosting classifier
on simple mask features (axial ratio, mask size, HASH size, survey) with the same longitude folds.
"""
import json, pathlib, sys
import numpy as np, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
out = {}
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
for run in sys.argv[1:]:
    d = pd.read_csv(ROOT / 'data' / f'dl_{run}.csv').merge(t[['idPNMain', 'file', 'MajDiam', 'pixscale']], on=['idPNMain', 'file'])
    d = d[d.split == 'test']
    ok = []
    for _, r in d.iterrows():
        f = ROOT / 'data' / 'masks' / f"{r.file.replace('.fits', '')}_dl_{run}.npz"
        if not f.exists() or not np.isfinite(r.dl_maj):
            ok.append(False); continue
        mk = np.load(f)['mask'] > 0
        n = mk.shape[0]; c = (n - 1) / 2
        yy, xx = np.nonzero(mk)
        rmax = 0.5 * (r.MajDiam if np.isfinite(r.MajDiam) else 10) / 2 / r.pixscale
        ok.append(bool(mk[int(round(c)), int(round(c))]) or np.hypot(xx.mean() - c, yy.mean() - c) < max(rmax, 3))
    d['strict'] = ok
    ratio = d.dl_maj / d.MajDiam
    out[run] = dict(n_test=len(d), detected_any=float(d.dl_maj.notna().mean()), detected_strict=float(np.mean(ok)),
                    multi_det=int((d.dl_ndet > 1).sum()), ratio_gt3=int((ratio > 3).sum()), ratio_lt03=int((ratio < 0.3).sum()))
    print(run, out[run])

mc = pd.read_csv(ROOT / 'data' / 'morph_cv.csv')
cls = ['R', 'E', 'B', 'S']
rec = {c: float(((mc.cnn_class == c) & (mc.mainClass == c)).sum() / (mc.mainClass == c).sum()) for c in cls}
out['morph_cnn'] = dict(n=len(mc), acc=float((mc.cnn_class == mc.mainClass).mean()), balanced_acc=float(np.mean(list(rec.values()))),
                        recall=rec, majority_class_frac=float(mc.mainClass.value_counts(normalize=True).max()))
from sklearn.ensemble import HistGradientBoostingClassifier
feat = mc[['idPNMain', 'mainClass', 'fold', 'MajDiam', 'maj_mask', 'q_mask', 'survey']].copy()
feat['logD'] = np.log10(feat.MajDiam.clip(lower=0.5)); feat['logM'] = np.log10(feat.maj_mask.clip(lower=0.5))
feat['sv'] = feat.survey.map({'IPHAS': 0, 'VPHAS': 1, 'SHS': 2})
X = feat[['logD', 'logM', 'q_mask', 'sv']].values; y = feat.mainClass.values
pred = np.empty(len(y), dtype=object)
for f in sorted(feat.fold.unique()):
    tr, te = feat.fold.values != f, feat.fold.values == f
    clf = HistGradientBoostingClassifier(class_weight='balanced', random_state=0).fit(X[tr], y[tr])
    pred[te] = clf.predict(X[te])
rec_b = {c: float(((pred == c) & (y == c)).sum() / (y == c).sum()) for c in cls}
out['morph_baseline_features'] = dict(acc=float((pred == y).mean()), balanced_acc=float(np.mean(list(rec_b.values()))), recall=rec_b)
print(json.dumps({k: v for k, v in out.items() if k.startswith('morph')}, indent=1))
json.dump(out, open(ROOT / 'data' / 'seg_morph_extra.json', 'w'), indent=1)
