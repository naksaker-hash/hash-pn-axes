"""Statistics of the samples after the size cut (env CUT, default 2 arcsec; author request 2026-10-17). CPU only.
Outputs data/cut4.json (name kept for traceability; it holds the results of the current cut) and paper_mlst/tab_align.tex (Table S3). Covers the counts of the catalogue and of the alignment
samples, the tests of Section 3.4 and supplementary Section S4, the error strata of Section S3 and the checks that the
earlier scripts made with the cut of 6 arcsec."""
import json, pathlib, re
import numpy as np, pandas as pd
from scipy import stats
CUT = float(__import__('os').environ.get('CUT', '2'))
ROOT = pathlib.Path(__file__).resolve().parents[1]; D = ROOT / 'data'; rng = np.random.default_rng(121)
wrap = lambda a: (a + 90) % 180 - 90; dpa = lambda a, b: np.abs(wrap(a - b))
key = lambda s: re.sub(r'[^a-z0-9]', '', str(s).lower())


def circ(pa, mu=90.0):
    pa = np.asarray(pd.Series(pa).dropna(), float); n = len(pa)
    if n < 2:
        return dict(n=n)
    z = np.exp(2j * np.radians(pa)).mean(); R = float(abs(z)); V = float(np.mean(np.cos(np.radians(2 * (pa - mu)))))
    return dict(n=n, mean=float(np.degrees(np.angle(z)) / 2 % 180), R=R, V=V, V_p=float(stats.norm.sf(V * np.sqrt(2 * n))),
                V_p2=float(2 * stats.norm.sf(abs(V) * np.sqrt(2 * n))), rayleigh_p=float(np.exp(np.sqrt(1 + 4 * n + 4 * (n ** 2 - (n * R) ** 2)) - (1 + 2 * n))),
                R_bias_corrected=float(np.sqrt(max((n * R ** 2 - 1) / (n - 1), 0))))


x = pd.read_csv(D / 'pa_final.csv'); cv = pd.read_csv(D / 'pa_cnn_cv.csv')
base = (x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
B = base & (x.mainClass == 'B'); E = base & (x.mainClass == 'E') & (x.q_mask < 0.85)
ob = x[B & ~x.bulge_sky]; obb = x[B & x.bulge_sky & x.morph_class.isna()]
out = dict(cut=CUT)
# catalogue
cat = x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & x.mainClass.isin(['B', 'E', 'I', 'A'])]
new = cat[cat.EPA.isna()]
vis = set()
rz = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'rees_zijlstra2013_bulge.csv'); co = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv')
w = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'weidmann_diaz2008.csv')
vkeys = set(rz.PNG) | set(co.PNG_approx_from_lb.dropna()) | set(w.PNG_from_name.dropna()) | set(w.PNG_approx_from_lb.dropna())
co['nkey'] = co.Name.map(key); w['nkey'] = w.Name.map(key)
nkeys = set(co.nkey) | set(w.nkey)
inv = new[~(new.png.isin(vkeys) | new.Name.map(key).isin(nkeys))]
s1 = pd.read_csv(D / 'pa_cnn_all_s1.csv').set_index('idPNMain'); s2 = pd.read_csv(D / 'pa_cnn_all_s2.csv').set_index('idPNMain')
a0 = new.set_index('idPNMain').pa
d1 = dpa(a0, s1.reindex(a0.index).cnn_pa); d2 = dpa(a0, s2.reindex(a0.index).cnn_pa); dd = np.concatenate([d1.dropna(), d2.dropna(), dpa(s1.reindex(a0.index).cnn_pa, s2.reindex(a0.index).cnn_pa).dropna()])
out['catalogue'] = dict(n=len(cat), n_E=int((cat.mainClass == 'E').sum()), n_B=int((cat.mainClass == 'B').sum()), n_IPHAS_VPHAS=int(cat.survey.isin(['IPHAS', 'VPHAS']).sum()),
                        n_new=len(new), n_new_no_visual=len(inv), n_new_no_visual_IV=int(inv.survey.isin(['IPHAS', 'VPHAS']).sum()),
                        seed_scatter_median=float(np.median(dd)), seed_scatter_gt20=float(np.mean(dd > 20)), seed_scatter_gt45=float(np.mean(dd > 45)))
# main samples
out['outside_B'] = circ(ob.gpa); out['outside_B_abs_b_q98'] = float(ob.Glat.abs().quantile(0.98)); out['outside_B_frac_within_5p1'] = float((ob.Glat.abs() <= 5.1).mean())
out['outside_E'] = circ(x[E & ~x.bulge_sky].gpa); out['outside_E_SHS'] = circ(x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & (x.survey == 'SHS') & (x.mainClass == 'E') & (x.q_mask < 0.85) & ~x.bulge_sky].gpa)
out['bulge_other_B'] = circ(obb.gpa); out['bulge_all_B'] = circ(x[B & x.bulge_sky].gpa); out['bulge_E'] = circ(x[E & x.bulge_sky].gpa)
out['bulge_all_B_n_rz'] = int((x[B & x.bulge_sky].morph_class.notna()).sum())
xr = x.merge(rz[['PNG', 'GPA_deg', 'morph_class']].rename(columns={'PNG': 'png', 'morph_class': 'rz_class'}), on='png')
rzb = xr[xr.rz_class == 'bipolar']; rzs = rzb[(rzb.MajDiam >= CUT) & (rzb.R_tta >= 0.8) & (rzb.survey != 'SHS')]
out['rz'] = dict(n_bipolar_all=int((rz.morph_class == 'bipolar').sum()), visual_all=circ(rz[rz.morph_class == 'bipolar'].GPA_deg), n_in_hash=len(rzb),
                 n_small=int((rzb.MajDiam < CUT).sum()), n_low_R=int(((rzb.MajDiam >= CUT) & (rzb.R_tta < 0.8)).sum()), n_shs=int(((rzb.MajDiam >= CUT) & (rzb.R_tta >= 0.8) & (rzb.survey == 'SHS')).sum()),
                 selected_visual=circ(rzs.GPA_deg), selected_network=circ(rzs.gpa), median_diff=float(np.median(dpa(rzs.gpa, rzs.GPA_deg))),
                 polar_visual_all=circ(rz[rz.morph_class == 'polar'].GPA_deg))
Ec = lambda pool: float(np.mean(np.cos(np.radians(2 * pool))))
mres = (cv.mainClass == 'B') & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8) & (cv.survey != 'SHS'); cvres = wrap(cv[mres].cnn_pa - cv[mres].EPA).dropna().values
err_rz = wrap(xr[xr.rz_class.notna()].gpa - xr[xr.rz_class.notna()].GPA_deg).dropna().values
n_ = len(rzs); Rv = out['rz']['selected_visual']['R']
out['rz']['expected_R'] = [float(np.sqrt((Rv * e) ** 2 + (1 - (Rv * e) ** 2) / n_)) for e in (Ec(cvres), Ec(err_rz))]
out['pool'] = dict(n=len(cvres), Ecos2=Ec(cvres), gt60=float(np.mean(np.abs(cvres) > 60)))
tan = pd.read_csv(D / 'tan_comparison.csv').dropna(subset=['gpa', 'GPA_deg']); ts = tan[(tan.MajDiam >= CUT) & (tan.R_tta >= 0.8) & (tan.survey != 'SHS')]
out['tan'] = dict(n=len(tan), n_selected=len(ts), median_all=float(np.median(dpa(tan.gpa, tan.GPA_deg))), median_selected=float(np.median(dpa(ts.gpa, ts.GPA_deg))),
                  net_all=circ(tan.gpa, 107.5), net_selected=circ(ts.gpa, 107.5), vis_selected=circ(ts.GPA_deg, 107.5))
# binaries
bj = pd.read_csv(D / 'binaries_clean.csv'); bt = bj[bj.in_alignment_test]
out['binaries'] = dict(outside=circ(bt[~bt.bulge_sky].gpa), bulge=circ(bt[bt.bulge_sky].gpa), n_candidates=int(bt.candidate.sum()),
                       outside_B=circ(bt[~bt.bulge_sky & (bt.mainClass == 'B')].gpa), bulge_B=circ(bt[bt.bulge_sky & (bt.mainClass == 'B')].gpa))
# subsamples outside and in the bulge region (Table S2 and Section S4)
comp = x.Component
sub = {'outside P_B >= 0.5': ob[ob.p_B >= 0.5].gpa, 'outside P_B >= 0.7': ob[ob.p_B >= 0.7].gpa, 'outside P_B >= 0.9': ob[ob.p_B >= 0.9].gpa,
       'outside disc': ob[ob.Component.isin(['Thin Disk', 'Thick Disk'])].gpa, 'outside thin': ob[ob.Component == 'Thin Disk'].gpa, 'outside thick': ob[ob.Component == 'Thick Disk'].gpa,
       'outside R >= 0.9': ob[ob.R_tta >= 0.9].gpa, 'outside R >= 0.95': ob[ob.R_tta >= 0.95].gpa, 'outside IPHAS': ob[ob.survey == 'IPHAS'].gpa, 'outside VPHAS': ob[ob.survey == 'VPHAS'].gpa,
       'outside |l| < 30': ob[((ob.Glon + 180) % 360 - 180).abs() < 30].gpa,
       'bulge other below 35': obb[obb.MajDiam < 35].gpa, 'bulge other bulge component': obb[obb.Component == 'Bulge'].gpa, 'bulge other beyond 5 kpc': obb[obb.rgeo_kpc > 5].gpa,
       'bulge other 6 to 10 kpc': obb[(obb.rgeo_kpc > 6) & (obb.rgeo_kpc < 10)].gpa, 'bulge P_B >= 0.7': x[B & x.bulge_sky & (x.p_B >= 0.7)].gpa}
out['subsamples'] = {k: circ(v) for k, v in sub.items()}
nm = pd.read_csv(D / 'pa_cnn_all_nomash.csv').set_index('idPNMain'); gn = ((nm.reindex(ob.idPNMain).cnn_pa + ob.set_index('idPNMain').psi) % 180).dropna()
Rn = nm.reindex(ob.idPNMain).cnn_R if 'cnn_R' in nm else None
keep = (Rn >= 0.8).values if Rn is not None else np.ones(len(gn), bool)
out['subsamples']['outside nomash'] = circ(((nm.reindex(ob.idPNMain).cnn_pa + ob.set_index('idPNMain').psi) % 180)[keep])
# bulge other bipolars against the selected RZ bipolars
s_rz = rzs; d_o = obb.AngDiam_pc.dropna(); d_r = s_rz.AngDiam_pc.dropna()
Vo = lambda g: float(np.mean(np.cos(np.radians(2 * (np.asarray(g) - 90)))))
allg = np.r_[rzs.gpa.values, obb.gpa.values]; nr = len(rzs); obs = Vo(rzs.gpa) - Vo(obb.gpa); perm = []
for _ in range(20000):
    p = rng.permutation(allg); perm.append(Vo(p[:nr]) - Vo(p[nr:]))
out['bulge_groups'] = dict(perm_p=float(np.mean(np.array(perm) >= obs)), diam_pc_other=float(d_o.median()), diam_pc_rz=float(d_r.median()),
                           mw_p=float(stats.mannwhitneyu(d_o, d_r).pvalue), absb_other=float(obb.Glat.abs().median()), absb_rz=float(s_rz.Glat.abs().median()),
                           beyond5_other=float((obb.rgeo_kpc > 5).mean()), beyond5_rz=float((s_rz.rgeo_kpc > 5).mean()),
                           rot_range=[float((x[B & x.bulge_sky].psi % 180).min()), float((x[B & x.bulge_sky].psi % 180).max())],
                           stellar_vphas_bulge=circ(x[(x.mainClass == 'S') & (x.survey == 'VPHAS') & x.bulge_sky & x.MajDiam.notna()].gpa))
# independence of neighbours and block permutation outside the bulge region
from astropy.coordinates import SkyCoord
import astropy.units as u
c = SkyCoord(ob.Glon.values * u.deg, ob.Glat.values * u.deg, frame='galactic'); i1, i2, sep, _ = c.search_around_sky(c, 1 * u.deg)
k = i1 < i2; cc = np.cos(np.radians(2 * (ob.gpa.values[i1[k]] - ob.gpa.values[i2[k]])))
blk = (ob.Glon // 10).astype(int).values; V0 = Vo(ob.gpa); ps = []
for _ in range(20000):
    rot = {b_: rng.uniform(0, 180) for b_ in np.unique(blk)}
    ps.append(Vo(ob.gpa.values + np.array([rot[b_] for b_ in blk])))
out['independence'] = dict(n_pairs=int(k.sum()), mean_cos=float(cc.mean()), sem=float(cc.std(ddof=1) / np.sqrt(k.sum())), n_blocks=int(len(np.unique(blk))), perm_p=float(np.mean(np.array(ps) >= V0)))
# Section S3 strata
eq = x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & x.mainClass.isin(['B', 'E', 'I', 'A'])].pa.values
lab = x[(x.MajDiam >= CUT) & x.mainClass.isin(['B', 'E', 'I', 'A'])]
out['strata'] = dict(n_eq=len(eq), fourfold_R=float(abs(np.exp(4j * np.radians(eq)).mean())), pass_labelled=float((lab[lab.trained].R_tta >= 0.8).mean()),
                     pass_unlabelled=float((lab[~lab.trained].R_tta >= 0.8).mean()))
cvc = cv[cv.mainClass.isin(['B', 'E', 'I', 'A']) & (cv.MajDiam >= CUT) & (cv.cnn_R >= 0.8)]; dcv = dpa(cvc.cnn_pa, cvc.EPA)
out['cv_selected'] = dict(n=len(cvc), median=float(dcv.median()), gt60=float((dcv > 60).mean()))
cv46 = cv[(cv.MajDiam >= 4) & (cv.MajDiam < 6)]; out['cv_4to6'] = dict(n=len(cv46), median=float(dpa(cv46.cnn_pa, cv46.EPA).median()))
json.dump(out, open(D / 'cut4.json', 'w'), indent=1, default=float)
exec(open(ROOT / 'scripts' / '49b_small_checks.py').read())

# Table S3
rows = [('\\citet{rees2013} bipolar, visual, all', out['rz']['visual_all'], False), ('\\citet{rees2013} bipolar, visual, selected$^a$', out['rz']['selected_visual'], True),
        ('\\citet{rees2013} bipolar, network, selected$^a$', out['rz']['selected_network'], True), ('\\citet{rees2013} polar, visual, all', out['rz']['polar_visual_all'], False),
        ('Bulge region bipolar, all', out['bulge_all_B'], False), ('Bulge region bipolar, not in \\citet{rees2013}', out['bulge_other_B'], False),
        ('Bulge region bipolar, $P_{\\mathrm{B}} \\geq 0.7$', out['subsamples']['bulge P_B >= 0.7'], False), ('Bulge region elliptical', out['bulge_E'], False),
        ('Outside bulge region, bipolar', out['outside_B'], False), ('Outside bulge region, bipolar, $P_{\\mathrm{B}} \\geq 0.7$', out['subsamples']['outside P_B >= 0.7'], False),
        ('Outside bulge region, bipolar, disc component', out['subsamples']['outside disc'], False), ('Outside bulge region, elliptical', out['outside_E'], False),
        ('Outside bulge region, elliptical, SHS only', out['outside_E_SHS'], False), ('Known close binaries, outside bulge region', out['binaries']['outside'], False),
        ('\\quad bipolar only, outside bulge region', out['binaries']['outside_B'], False), ('Known close binaries, bulge region', out['binaries']['bulge'], False),
        ('\\quad bipolar only, bulge region', out['binaries']['bulge_B'], False)]
fp = lambda p: '$<0.001$' if p < 0.001 else ('$>0.999$' if p > 0.999 else (f'{p:.2g}' if p < 0.1 else f'{p:.2f}'))
lines = []
for nm_, r, nop in rows:
    if nop:
        lines.append(f"{nm_} & {r['n']} & {r['mean']:.0f} & {r['R']:.2f} & -- & -- & -- \\\\")
    else:
        lines.append(f"{nm_} & {r['n']} & {r['mean']:.0f} & {r['R']:.2f} & {fp(r['rayleigh_p'])} & {fp(r['V_p'])} & {fp(r['V_p2'])} \\\\")
old = open(ROOT / 'paper_mlst' / 'tab_align.tex').read()
t0 = old.index('\\begin{tabular}'); h1 = old.index('\\hline\n', t0); h2 = old.index('\\hline\n', h1 + 6)
head = old[:h2 + len('\\hline\n')]  # keep the caption, the top rule, the column headings and the rule below them
tail = old[old.rindex('\\hline'):]
(ROOT / 'paper_mlst' / 'tab_align.tex').write_text(head + '\n'.join(lines) + '\n' + tail)
print(json.dumps({k: v for k, v in out.items() if k not in ('subsamples',)}, indent=0, default=float)[:6000]); print(json.dumps(out['subsamples'], default=float)[:3000])
