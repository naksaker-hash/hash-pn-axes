"""Gains of the seed averaged network over the low isophote and flux weighted moments, per class, with the block
bootstrap of 38_revision7_analyses.py (rng 71, 4000 draws). Also the share of catalogue class PNe above the size cut passing R_TTA >= 0.8, labelled and unlabelled.
Output: data/lowiso_gain.json"""
import json, pathlib
import numpy as np, pandas as pd
D=str(pathlib.Path(__file__).resolve().parents[1] / 'data') + '/'
out={}
wrap=lambda d:(np.asarray(d,float)+90)%180-90
dpa=lambda a,b:np.abs(wrap(np.asarray(a,float)-np.asarray(b,float)))
cv=pd.read_csv(D+'pa_cnn_cv.csv'); ids=cv.idPNMain.values
al=[cv.set_index('idPNMain').reindex(ids)]+[pd.read_csv(D+f'pa_cnn_cv_s{s}.csv').set_index('idPNMain').reindex(ids) for s in (1,2,3,4)]
c5=cv[['idPNMain','EPA','mainClass','Glon']].copy()
c5['d_net5']=np.mean([dpa(s.cnn_pa.values,cv.EPA.values) for s in al],0)
bp=pd.read_csv(D+'baselines_pa.csv')
c5=c5.merge(bp[['idPNMain','pa_low','pa_flux']],on='idPNMain',how='left')
blk=(c5.Glon//10).astype(int).values.copy(); blk[blk==35]=0
rng=np.random.default_rng(71)
for cl in ['B','E','all']:
  s=c5 if cl=='all' else c5[c5.mainClass==cl]; ii=s.index.values; bb=blk[ii]
  groups=[ii[bb==q] for q in np.unique(bb)]
  for col in ['pa_low','pa_flux']:
    dm=dpa(c5[col],c5.EPA); dn=c5.d_net5.values; g=[]
    for _ in range(4000):
      p=np.concatenate([groups[q] for q in rng.integers(0,len(groups),len(groups))]); g.append(np.nanmedian(dm[p])-np.median(dn[p]))
    out[f'{cl} {col}']=dict(gain=float(np.nanmedian(dm[ii])-np.median(dn[ii])),ci95=[float(v) for v in np.percentile(g,[2.5,97.5])])
    print(cl,col,out[f'{cl} {col}'])

CUT = float(__import__('os').environ.get('CUT', '2'))
px = pd.read_csv(D+'pa_final.csv'); big = px[(px.MajDiam >= CUT) & px.mainClass.isin(list('BEIA'))]
out['pass08_labelled'] = float((big[big.EPA.notna()].R_tta >= 0.8).mean()); out['pass08_unlabelled'] = float((big[big.EPA.isna()].R_tta >= 0.8).mean())
print('pass', out['pass08_labelled'], out['pass08_unlabelled'])
json.dump(out, open(D+'lowiso_gain.json','w'), indent=1)
