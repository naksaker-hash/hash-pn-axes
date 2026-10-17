"""Classification network trained on all 2332 R, E, B and S PNe and applied to the HASH irregular (I) and
asymmetric (A) PNe, which were never used in training, for the ERBIAS confusion figure.
Output: data/morph_IA.csv (class probabilities of the I and A PNe)
"""
import sys, pathlib, importlib
import numpy as np, pandas as pd, torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
mc = importlib.import_module('09_morph_classifier')
pa = mc.pa
t = pd.read_csv(ROOT / 'data' / 'pseudolabels.csv')
tr = t[t.mainClass.isin(mc.CLASSES)].reset_index(drop=True)
ia = t[t.mainClass.isin(['I', 'A'])].reset_index(drop=True)
x = pa.load_images(tr); y = torch.tensor([mc.CLASSES.index(c) for c in tr.mainClass])
m = mc.train(x, y)
P = mc.predict(m, pa.load_images(ia))
for i, c in enumerate(mc.CLASSES):
    ia[f'p_{c}'] = P[:, i]
ia['cnn_class'] = [mc.CLASSES[i] for i in P.argmax(1)]
ia[['idPNMain', 'Name', 'mainClass', 'MajDiam', 'survey'] + [f'p_{c}' for c in mc.CLASSES] + ['cnn_class']].to_csv(ROOT / 'data' / 'morph_IA.csv', index=False)
print(pd.crosstab(ia.mainClass, ia.cnn_class))
