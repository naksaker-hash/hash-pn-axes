"""Segmentation summary over YOLO runs (test longitudes) for Section 3.1. Output: data/seg_summary.json"""
import json, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
runs = ['yolo26n-seg_s0', 'yolo26n-seg_s1', 'yolo26n-seg_s2']
keys = dict(detected=lambda r: r['detected'], iou=lambda r: r['iou_good_median'], ratio=lambda r: r['ratio_dl_hash_median_all'],
            iqr_lo=lambda r: r['ratio_dl_hash_iqr_all'][0], iqr_hi=lambda r: r['ratio_dl_hash_iqr_all'][1],
            failed_med=lambda r: r['ratio_dl_hash_iqr_classical_failed'][1], class_acc=lambda r: r['class_acc'],
            pa_median=lambda r: r['pa_dl']['median'])
out = {'n_test': json.load(open(ROOT / 'data' / f'eval_{runs[0]}.json'))['test']['n']}
for k, f in keys.items():
    v = [f(json.load(open(ROOT / 'data' / f'eval_{r}.json'))['test']) for r in runs]
    out[k] = dict(mean=float(np.mean(v)), min=float(min(v)), max=float(max(v)))
s = json.load(open(ROOT / 'data' / 'eval_yolo26s-seg_s0.json'))['test']
out['yolo26s'] = {k: f(s) for k, f in keys.items()}
c = json.load(open(ROOT / 'data' / f'eval_{runs[0]}.json'))['test']
out['classical_iqr'] = c['ratio_cl_hash_iqr_all']
out['classical_pa'] = c['pa_classical']
print(json.dumps(out, indent=1))
json.dump(out, open(ROOT / 'data' / 'seg_summary.json', 'w'), indent=1)
