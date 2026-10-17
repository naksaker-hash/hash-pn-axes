"""Checks of the small PNe and of the size cut, run from 49_cut_stats.py (sixth deep review). Output: data/cut_checks.json
a. validation below the smallest training size of 4 arcsec: HASH labelled PNe of 2 to 4 arcsec that were never trained
   on, and unseen PNe of Rees & Zijlstra of 2 to 4 arcsec, in the catalogue classes, with the 95 per cent upper bound
   on the gross failure rate
b. error strata of the selected unseen external PNe including 2 to 6 arcsec, and the R_TTA and survey checks of S3
c. robustness of the plane limit to the cut and to the errors of the smallest PNe
d. the network axis of M 3-42 replaced by the visual axes of Tan et al. and of Rees & Zijlstra in the bulge binary test
e. compact bipolars of Rees & Zijlstra and compact HASH bipolars outside their sample
f. the permutation test of the selected bipolars of Rees & Zijlstra against the other bulge bipolars at several cuts and
   at matched size"""
from scipy import stats as _st
ck = {}
cls = ['B', 'E', 'I', 'A']
def fails(d):
    d = np.asarray(d); return dict(n=int(len(d)), median=float(np.median(d)) if len(d) else None, gt60=int((d > 60).sum()), gt20=int((d > 20).sum()))
# a
lab_small = x[(x.MajDiam >= 2) & (x.MajDiam < 4) & x.EPA.notna() & ~x.trained & x.mainClass.isin(cls) & (x.R_tta >= 0.8)]
rzc = xr[xr.rz_class.notna() & xr.EPA.isna()]
rz_small = rzc[(rzc.MajDiam >= 2) & (rzc.MajDiam < 4) & rzc.mainClass.isin(cls) & (rzc.R_tta >= 0.8) & (rzc.survey != 'SHS')]
d1 = dpa(lab_small.pa, lab_small.EPA).values; d2 = dpa(rz_small.gpa, rz_small.GPA_deg).values
both = np.r_[d1, d2]; nb = len(both); k = int((both > 60).sum())
ub = float(_st.beta.ppf(0.95, k + 1, nb - k)) if nb else None  # one sided 95 per cent Clopper Pearson bound
ck['small'] = dict(hash_labelled=fails(d1), hash_labelled_names=lab_small.Name.tolist(), rz_unseen=fails(d2), combined=fails(both), gross_rate_upper95=ub,
                   n_catalogue_2to4=int(((x.MajDiam >= 2) & (x.MajDiam < 4) & (x.R_tta >= 0.8) & x.mainClass.isin(cls)).sum()),
                   n_outside_B_2to4=int(ob.MajDiam.lt(4).sum()), n_outside_B_2to6=int(ob.MajDiam.lt(6).sum()))
# b
m_ = pd.read_csv(D / 'hash_master.csv', dtype=str).drop_duplicates('idPNMain'); m_['idPNMain'] = m_.idPNMain.astype(int)
co_ = pd.read_csv(ROOT / 'refs' / 'pa_tables' / 'corradi1998.csv'); co_['nkey'] = co_.Name.map(key)
h_ = x[['idPNMain', 'Name', 'png', 'mainClass', 'MajDiam', 'survey', 'pa', 'R_tta', 'trained', 'EPA']].copy(); h_['nkey'] = h_.Name.map(key)
e_rz = h_.merge(rz[['PNG', 'EPA_deg']], left_on='png', right_on='PNG')
b1 = h_.merge(co_[['nkey', 'EPA_deg']], on='nkey'); b2 = h_.merge(co_[['PNG_approx_from_lb', 'EPA_deg']], left_on='png', right_on='PNG_approx_from_lb')
e_co = pd.concat([b1, b2[~b2.idPNMain.isin(b1.idPNMain)]]).drop_duplicates('idPNMain')
ext = pd.concat([e_co, e_rz]); ext = ext[ext.mainClass.isin(cls) & (ext.MajDiam >= CUT) & (ext.R_tta >= 0.8) & ext.EPA.isna()]
ext = ext.assign(d=dpa(ext.pa, ext.EPA_deg))
ck['external_strata'] = dict(n=len(ext), median=float(ext.d.median()), gt60=float((ext.d > 60).mean()),
                             **{f'{a}-{b}': fails(ext[(ext.MajDiam >= a) & (ext.MajDiam < b)].d) for a, b in [(2, 6), (6, 10), (10, 20), (20, 1e9)]},
                             **{f'class {c}': fails(ext[ext.mainClass == c].d) for c in ('B', 'E')}, **{f'survey {s_}': fails(ext[ext.survey == s_].d) for s_ in ('IPHAS', 'VPHAS', 'SHS')})
sel_all = x[(x.MajDiam >= CUT) & (x.R_tta >= 0.8) & x.mainClass.isin(cls)]
ck['s3'] = dict(frac_above095_labelled=float((sel_all[sel_all.trained].R_tta >= 0.95).mean()), frac_above095_unlabelled=float((sel_all[~sel_all.trained].R_tta >= 0.95).mean()),
                shs_selected_all=circ(sel_all[sel_all.survey == 'SHS'].pa), vphas_selected_all=circ(sel_all[sel_all.survey == 'VPHAS'].pa))
# c
def lim_mixed(pa_obs, small_mask, pool, small_uniform, ns=10000):
    nn = len(pa_obs); s2 = np.sqrt(-2 * np.log(0.83)); Vo_ = np.mean(np.cos(np.radians(2 * (pa_obs - 90))))
    def P(f):
        al = rng.random((ns, nn)) < f
        pa = np.where(al, 90 + np.degrees(rng.normal(0, s2, (ns, nn))) / 2, rng.uniform(0, 180, (ns, nn)))
        e = rng.choice(pool, (ns, nn))
        if small_uniform:
            e = np.where(small_mask[None, :], rng.uniform(-90, 90, (ns, nn)), e)
        return float(np.mean(np.mean(np.cos(np.radians(2 * (pa + e - 90))), 1) <= Vo_))
    fg = np.round(np.arange(0.08, 0.22, 0.01), 2); Pv = np.array([P(f) for f in fg]); i = np.flatnonzero(Pv < 0.05)[0]
    return float(fg[i - 1] + (Pv[i - 1] - 0.05) * (fg[i] - fg[i - 1]) / (Pv[i - 1] - Pv[i]))
small = ob.MajDiam.values < 4
ob4 = ob[ob.MajDiam >= 4]
ck['robust'] = dict(limit_cut2=lim_mixed(ob.gpa.values, small, cvres, False), limit_cut2_uniform_small=lim_mixed(ob.gpa.values, small, cvres, True),
                    limit_cut4=lim_mixed(ob4.gpa.values, np.zeros(len(ob4), bool), cvres, False), outside_B_cut4=circ(ob4.gpa))
# d
def p_all_(pa_o, pool, ns=400_000):
    nn = len(pa_o); s2 = np.sqrt(-2 * np.log(0.83))
    pa = 90 + np.degrees(rng.normal(0, s2, (ns, nn))) / 2 + rng.choice(pool, (ns, nn))
    return float(np.mean(np.mean(np.cos(np.radians(2 * (pa - 90))), 1) <= np.mean(np.cos(np.radians(2 * (pa_o - 90))))))
bs_ = bj[bj.in_alignment_test & bj.bulge_sky].copy()
d_ = {}
for lab_, vis_ in [('network', None), ('Tan visual', 92.5), ('RZ13 visual', 83.0)]:
    g_ = bs_.gpa.values.copy()
    if vis_ is not None:
        g_[bs_.Name.values == 'M 3-42'] = vis_
    m20 = bs_.MajDiam.values <= 20
    d_[lab_] = dict(all=[p_all_(g_, cvres), p_all_(g_, err_rz)], le20=[p_all_(g_[m20], cvres), p_all_(g_[m20], err_rz)])
ck['m342'] = d_
# e
rzsel = rzs.copy()
c26 = rzsel[rzsel.MajDiam < 6]; c6 = rzsel[rzsel.MajDiam >= 6]
oc = obb[obb.MajDiam < 10]
ck['compact_rz'] = dict(rz_2to6=dict(n=len(c26), visual=circ(c26.GPA_deg), network=circ(c26.gpa)), rz_ge6=dict(n=len(c6), visual=circ(c6.GPA_deg), network=circ(c6.gpa)),
                        hash_other_compact=circ(oc.gpa), n_hash_other_compact=len(oc), rz_selected_hash_classes=rzsel.mainClass.value_counts().to_dict())
# f
def perm(gr, go, n=20000):
    Vo2 = lambda g: float(np.mean(np.cos(np.radians(2 * (np.asarray(g) - 90)))))
    allg = np.r_[gr, go]; nr = len(gr); obs_ = Vo2(gr) - Vo2(go); ps_ = []
    for _ in range(n):
        p_ = rng.permutation(allg); ps_.append(Vo2(p_[:nr]) - Vo2(p_[nr:]))
    ps_ = np.array(ps_); return dict(n_rz=nr, n_other=len(go), one_sided=float(np.mean(ps_ >= obs_)), two_sided=float(np.mean(np.abs(ps_) >= abs(obs_))))
pf = {}
for c_ in (6, 4, 2):
    bc = (x.MajDiam >= c_) & (x.R_tta >= 0.8) & (x.survey != 'SHS')
    r_ = xr[(xr.rz_class == 'bipolar') & (xr.MajDiam >= c_) & (xr.R_tta >= 0.8) & (xr.survey != 'SHS')]
    o_ = x[bc & (x.mainClass == 'B') & x.bulge_sky & x.morph_class.isna()]
    pf[f'cut {c_}'] = perm(r_.gpa.values, o_.gpa.values)
pf['matched below 10'] = perm(rzsel[rzsel.MajDiam < 10].gpa.values, obb[obb.MajDiam < 10].gpa.values)
pf['matched 6 to 35'] = perm(rzsel[(rzsel.MajDiam >= 6) & (rzsel.MajDiam < 35)].gpa.values, obb[(obb.MajDiam >= 6) & (obb.MajDiam < 35)].gpa.values)
pf['rz HASH bipolars only'] = perm(rzsel[rzsel.mainClass == 'B'].gpa.values, obb.gpa.values)
ck['permutation'] = pf
json.dump(ck, open(D / 'cut_checks.json', 'w'), indent=1, default=float)
print(json.dumps(ck, default=float, indent=0)[:5000])
