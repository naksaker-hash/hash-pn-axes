"""Size and precision of the present axes against the earlier orientation studies (Table 3 and the comparison
statements of the abstract, Introduction and Section 4.2). Output: data/literature_comparison.json"""
import json, pathlib
import pandas as pd
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'
cat = pd.read_csv(D / 'catalogue_pa.csv'); n = int(cat.flag_reliable.sum())
lit = {'Corradi et al. 1998': 209, 'Weidmann & Diaz 2008': 440, 'Rees & Zijlstra 2013': 130, 'Ritter & Parker 2020': 766, 'Tan et al. 2023b': 136}
r9 = json.load(open(D / 'rev9.json'))['misc']['visual_same_objects']
out = dict(n_axes=n, literature=lit, ratio={k: n / v for k, v in lit.items()},
           visual_scatter=dict(disc_same_objects=r9['W_vs_C'], network_disc=[r9['net_vs_W'], r9['net_vs_C']], ratio_disc=r9['net_vs_W'] / r9['W_vs_C'], bulge=13.0))
json.dump(out, open(D / 'literature_comparison.json', 'w'), indent=1); print(json.dumps(out, indent=1))
