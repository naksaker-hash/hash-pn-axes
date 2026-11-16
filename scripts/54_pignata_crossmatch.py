import csv,collections,sys
# Cross match of the Gemini GMOS classes of Pignata, Weidmann & Mast 2026 (MNRAS 553, stag1980, Table 3)
# with the HASH sample, the classification network (morph_cv.csv) and the axis catalogue.
D='/Users/nazimaksaker/Downloads/HASH_DL_morph/data/'
P={'R':"003.3+66.1 053.1-02.1 058.9+09.0 065.9+02.4 070.9+02.4 075.0-07.2 078.9+00.7 084.0+09.5 089.1+08.7 093.3-00.9 110.6-12.9 121.6+03.5 137.7+03.3 144.3-15.5 160.7-02.9 167.0-00.9 224.3-03.4 248.5+10.5 257.8-05.4 277.7-03.5",
'E':"011.4-07.3 016.0-07.6 016.7-07.3 017.9-04.8 025.4-09.2 027.6+16.9 032.9+07.8 038.4+03.6 040.3+03.4 040.8-09.7 041.2-00.6 048.9+04.3 051.9-02.5 069.7+00.0 078.6+05.2 086.8+01.2 112.5+03.7 132.8+02.0 139.0+03.2 175.6+11.4 178.3-02.5 184.8+04.4 197.3+02.7 212.2-04.7 251.1+00.7 260.1+00.2 275.9-01.0 283.8-04.2 290.2+02.0 315.4+05.2 336.1+04.1 353.3-08.3 355.1+04.7",
'B':"005.1+02.0 065.2-05.6 069.2+03.8 069.6-03.9 084.6-07.9 103.7+00.4 104.4-01.6 204.8-03.5 222.9-01.1 242.6-04.4 290.1-00.4 291.7+03.7 331.4+00.5 133.2-00.9 228.5-11.4 250.4-01.3",
'I':"033.8+01.5 036.0-01.4 043.3+03.5 122.9+00.3 151.0-00.4 156.4+01.1 188.6+04.4 197.0+05.8 280.5+01.8 284.4+07.8 302.8-04.7 307.8-19.7",
'L':"034.0+02.2 195.4-04.0 252.5-01.6",'M':"043.3+10.4 044.9+00.0"}
pg={g:c for c,s in P.items() for g in s.split()}
print('Pignata total',len(pg))
H={r['PNG']:r for r in csv.DictReader(open(D+'hash_master.csv'))}
C={r['PNG']:r for r in csv.DictReader(open(D+'catalogue_pa.csv'))}
idmap={r['idPNMain']:r['PNG'] for r in H.values()}
M={}
for r in csv.DictReader(open(D+'morph_cv.csv')):
    g=idmap.get(r['idPNMain']); 
    if g: M[g]=r
inH=[g for g in pg if g in H]; print('in hash_master (True PNe)',len(inH),'missing',[g for g in pg if g not in H])
print('in catalogue_pa (has image)',sum(g in C for g in pg))
sel=[g for g in pg if g in C and C[g]['flag_reliable']=='1']; print('selected axes',len(sel))
print('in classifier cv',sum(g in M for g in pg))
# HASH class vs Pignata
t=collections.Counter((pg[g],H[g]['mainClass']) for g in inH); print('Pignata vs HASH',sorted(t.items()))
cm=collections.Counter((pg[g],M[g]['cnn_class']) for g in pg if g in M); print('Pignata vs CNN',sorted(cm.items()))
for c in 'REB':
    gs=[g for g in pg if g in M and pg[g]==c and M[g]['mainClass'] in 'REB']
    a=sum(M[g]['mainClass']==c for g in gs); b=sum(M[g]['cnn_class']==c for g in gs)
    print(c,'n',len(gs),'HASH agree',a,'CNN agree',b)
# bipolar prob for Pignata B vs E
import statistics as st
for c in 'BE':
    v=[float(M[g]['p_B']) for g in pg if g in M and pg[g]==c]; print(c,'median p_B',round(st.median(v),2),len(v))
bb=[g for g in pg if g in M and pg[g] in 'BLM']; ee=[g for g in pg if g in M and pg[g] in 'ER']
from itertools import product
auc=sum((float(M[x]['p_B'])>float(M[y]['p_B']))+0.5*(M[x]['p_B']==M[y]['p_B']) for x,y in product(bb,ee))/(len(bb)*len(ee)); print('AUC B/L/M vs E/R',round(auc,2),len(bb),len(ee))
for g in sel: print(g,C[g]['Name'],pg[g],C[g]['HASH_class'],C[g]['PA_cnn'],C[g]['R_TTA'],C[g]['HASH_PA'],C[g]['survey'])
