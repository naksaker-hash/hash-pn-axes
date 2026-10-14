"""Build the YOLO-seg dataset from cutouts and quality-filtered pseudo-label masks.

Image channels: Halpha, Halpha minus scaled r (or Halpha for SHS), r (or Halpha).
Classes: 0 R, 1 E, 2 B, 3 S (HASH mainClass; I and A are too few and are left out).
Split: blocks of 10 deg in Galactic longitude, assigned to train/val/test (70/15/15)
so neighbouring fields never cross the split.
"""
import pathlib, shutil, json
import numpy as np, pandas as pd, cv2
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from ultralytics.data.converter import merge_multi_segment

ROOT = pathlib.Path(__file__).resolve().parents[1]
DS = ROOT / 'data' / 'yolo_seg'
CLASSES = ['R', 'E', 'B', 'S']
rng = np.random.default_rng(0)


def stretch(x):
    x = np.nan_to_num(x)
    _, med, std = sigma_clipped_stats(x, sigma=3)
    hi = np.percentile(x, 99.8)
    y = np.arcsinh(np.clip((x - med) / max(hi - med, 1e-6), -0.05, 1) * 10) / np.arcsinh(10)
    return (np.clip(y, 0, 1) * 255).astype(np.uint8)


def image(fn, k):
    h = fits.open(fn)
    ha = h[0].data.astype(float)
    if 'R' in h and np.isfinite(k):
        r = h['R'].data.astype(float)
        _, hm, _ = sigma_clipped_stats(ha)
        _, rm, _ = sigma_clipped_stats(r)
        sub = (ha - hm) - k * (r - rm)
    else:
        r, sub = ha, ha
    # flip so north is up in the PNG (FITS row 0 is the bottom)
    return np.flipud(np.dstack([stretch(ha), stretch(sub), stretch(r)]))


def polygons(mask):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < 6:
            continue
        c = cv2.approxPolyDP(c, 0.7, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(c)
    return out


def quality(t):
    ratio = t.maj_mask / t.MajDiam
    return ((t.frac_edge == 0) & (t.peak_snr > 10) & ratio.between(0.7, 1.5)
            & (t.frac_in_search > 0.5) & t.mainClass.isin(CLASSES))


if __name__ == '__main__':
    t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
    t['good'] = quality(t)
    # assign shuffled 10 deg longitude blocks by object count (all covered PNe): 70/15/15
    cov = pd.read_csv(ROOT / 'data' / 'hash_coverage.csv')
    cnt = (cov[cov.anyHa].Glon // 10).astype(int).value_counts().reindex(range(36), fill_value=0)
    blocks = np.arange(36)
    rng.shuffle(blocks)
    cum = np.cumsum(cnt[blocks].values) / cnt.sum()
    split = {b: ('train' if c0 < 0.70 else 'val' if c0 < 0.85 else 'test')
             for b, c0 in zip(blocks, np.r_[0, cum[:-1]])}
    t['split'] = [split[int(l // 10) % 36] for l in t.Glon]
    if DS.exists():
        shutil.rmtree(DS)
    for s in ['train', 'val', 'test']:
        (DS / 'images' / s).mkdir(parents=True)
        (DS / 'labels' / s).mkdir(parents=True)
    for _, r in t.iterrows():
        stem = r.file.replace('.fits', '')
        img = image(ROOT / 'data' / 'cutouts' / r.file, r.kcont)
        cv2.imwrite(str(DS / 'images' / r.split / f'{stem}.png'), img[:, :, ::-1])
        lines = []
        if r.good:
            m = np.load(ROOT / 'data' / 'masks' / f'{stem}.npz')['mask']
            n = m.shape[0]
            ps = polygons(m)
            if ps:
                # one PN = one instance: join fragments (e.g. separate lobes) with zero-width bridges
                p = (np.concatenate(merge_multi_segment([q.reshape(-1).tolist() for q in ps]))
                     if len(ps) > 1 else ps[0])
                xy = ' '.join(f'{x / n:.5f} {1 - y / n:.5f}' for x, y in p)  # FITS y is up
                lines.append(f'{CLASSES.index(r.mainClass)} {xy}')
        # objects without a trusted mask get no label file entry and are kept out of training
        if lines:
            (DS / 'labels' / r.split / f'{stem}.txt').write_text('\n'.join(lines) + '\n')
        elif r.split == 'train' or r.split == 'val':
            (DS / 'images' / r.split / f'{stem}.png').unlink()
    (DS / 'data.yaml').write_text(
        f"path: {DS}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names: {json.dumps(dict(enumerate(CLASSES)))}\n")
    t.to_csv(ROOT / 'data' / 'pseudolabels.csv', index=False)
    print(t.groupby(['split', 'good']).size().unstack(fill_value=0))
    print(t[t.good].groupby(['split', 'mainClass']).size().unstack(fill_value=0))
