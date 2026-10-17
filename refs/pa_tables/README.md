# Literature PA tables for Galactic PNe (independent validation set)

Fetched 2026-10-13. Raw downloads were kept in `raw/`, which is not distributed because it holds copies of the published papers; the CSVs here are parsed from them by
`build_pa_tables.py` (run with `python3 -I build_pa_tables.py <this dir>`).

Conventions used in all files
* EPA = position angle in the equatorial frame, measured from (equatorial) North through East, 0 to 180 deg.
* GPA = the same axis measured from Galactic North through Galactic East: GPA = EPA + psi, where psi is
  the angle between equatorial and Galactic north at the object (Corradi et al. 1998 formula, J2000).
  GPA = 90 deg means the axis lies parallel to the Galactic plane.
* All PAs are axial (180 deg ambiguity).
* `*_recomputed_deg` columns are mine: GPA recomputed from the published EPA and (l, b) with the psi
  formula. They are a consistency check, not published data. Agreement is better than 1 deg
  (median 0.3 to 0.5 deg) except for the few rows noted below.
* `PNG_approx_from_lb` is built from the published l, b (rounded to 0.1 deg), so it can be off by 0.1 deg
  from the official PN G designation. Match on name or coordinates where possible.

---

## 1. `rees_zijlstra2013_bulge.csv` (130 rows)

Source: Rees & Zijlstra 2013, MNRAS 435, 975, "Alignment of the angular momentum vectors of PNe in the
Galactic Bulge", doi:10.1093/mnras/stt1300. Not in VizieR (J/MNRAS/435/975 does not exist). Parsed from the
LaTeX of Table 1 ("The orientations of the PNe") in the arXiv source, arXiv:1307.5711v1
(`raw/arxiv_1307.5711/x/AngleTableToCheck.tex`). This is the arXiv version; differences from the
printed journal table were not checked.

Columns: `PNG` (PN G name; no common names in the table), `morph_class` (bipolar 44 / polar 55 /
non-polar 31, their own classes), `EPA_deg`, `GPA_deg`, `err_deg` (same uncertainty applies to EPA and GPA;
2 deg for polar and non-polar, larger, measured values for bipolars), `telescope` (NTT H-alpha or HST),
`GPA_recomputed_deg`, `GPA_minus_recomputed_deg`.

Frame: both EPA (column "PA", N through E) and GPA (from Galactic N through E) are given.
Axis: the polar / symmetry (outflow) axis. Bipolars: direction of the lobes and of the low-intensity
interior of the central shell. Polar class (no lobes): direction of central cavity / low-intensity
regions indicating mass-loss direction. Non-polar: major axis at the 10 per cent isophote. One special case:
PN G359.8+05.2 (bipolar remnant) was taken perpendicular to its high-intensity structure.
Measurement: by eye with a ruler tool (HEASARC fv) on high-resolution H-alpha images, range [0, 180).
Sample: Galactic bulge PNe only. Check: all 130 GPAs reproduce from EPA within 3.1 deg.

## 2. `tan2023_bulge_binaries.csv` (16 rows, 14 usable GPAs)

Source: Tan, Parker, Zijlstra & Ritter 2023, ApJL 951, L44, doi:10.3847/2041-8213/acdbcd
(arXiv:2307.07140). Taken from the IOP machine-readable version of Table 1
(`raw/tan2023_t1_ascii.txt`, from the article's suppdata link), cross-checked against the arXiv LaTeX.

IMPORTANT: the full set of 136 (126 with measurable GPA) bulge PNe is NOT published. The paper
(both the IOP version and the arXiv source) only tabulates the short-period binary subsample; the only
other table (`raw/tan2023_t2_ascii.txt`) is statistics. The text states that EPAs for the full bulge sample
"will soon be available in Ritter et al. (in prep)". No such paper was found on arXiv as of this date.
Their Paper I (Tan et al. 2023, MNRAS, arXiv:2211.06029, bulge PN morphologies) was checked and contains
no PA column. The 112 non-binary GPAs would have to be requested from the authors.

Columns: `PNG`, `Name`, `subsample` (a = 6 confirmed short-period binary CSPNe plus M 1-34; b = 9 high-adf
PNe), `period_d_or_adf` (orbital period in days with reference number, or adf), `GPA_deg`, `morph_HASH`
(HASH ERBIAS+sparm code), `RAJ2000`, `DEJ2000`, `EPA_backcomputed_deg` (mine: GPA minus psi, so not
published), `note`. M 1-34 is a suspected binary that Tan+23 excluded from their analysis; H 2-42 has no GPA
(round/annular).

Frame: Galactic (GPA, from Galactic N through E, 0 to 180); EPA is not tabulated, only the back-computed one here.
Axis: the "long symmetry" / principal axis. Ellipticals: ellipse major axis. Bipolars: geometric lobe
direction plus low-density interior structures. Measurement: visual on VLT FORS2 and HST images,
rounded to 0.5 deg, random error about 2 to 5 deg (three-person comparison on 50 PNe).

## 3. `weidmann_diaz2008.csv` (459 rows, 444 with PA)

Source: Weidmann & Diaz 2008, PASP 120, 380, "The spatial orientation of PNe within the Milky Way",
2008PASP..120..380W. VizieR J/PASP/120/380/table1 (`raw/weidmann2008_table1.dat`, `raw/weidmann2008_ReadMe.txt`
from CDS FTP; SIMBAD names and coordinates from the VizieR ASU TSV, `raw/weidmann2008_table1.tsv`).

Columns: `Name` (as in the catalogue: common name, or Gxxx.x+yy.y for MASH objects), `PNG_from_name`
(when the name itself is a PN G designation), `PNG_approx_from_lb`, `GLON`, `GLAT`, `EPA_deg`, `GPA_deg`,
`source` (where the image or PA came from: DSS2 189, MASH 174, CAM98 54 = PAs adopted from Corradi et al.
1998, HST 42), `SimbadName`, `RAJ2000_simbad`, `DEJ2000_simbad` (added by CDS, not in the paper; use these to
cross-match to HASH), `GPA_recomputed_deg`, `GPA_minus_recomputed_deg`.

Frame: both EPA (equatorial, N through E) and GPA (Galactic) are given.
Axis: the projected (apparent) major axis of elongated PNe (ReadMe: "Position angle of the projected
major axis"); they follow Corradi+98 in taking the main symmetry axis. Measurement: on DSS2 red,
MASH H-alpha and HST F502N/F656N images (or adopted from CAM98).
Caveats: 15 HST rows have no EPA/GPA (blank in the catalogue). Four rows have published GPA inconsistent
with their EPA and position: NGC 6369 (32 deg off), NGC 6445 (82 deg off; also EPA 149 vs Corradi 72),
DeHt 3 (5.5 deg), WeSb 3 (3.5 deg). Treat these as suspect. Paper says 440 PNe; the catalogue has 459 records.

## 4. `corradi1998.csv` (209 rows)

Source: Corradi, Aznar & Mampaso 1998, MNRAS 297, 617, "Orientation of PNe within the Galaxy",
doi:10.1046/j.1365-8711.1998.01532.x. Not in VizieR (J/MNRAS/297/617 does not exist) and not on arXiv.
Parsed from the text layer of the ADS full-text PDF (`raw/corradi1998/1998MNRAS.297..617C.pdf`,
`raw/corradi1998/corradi1998.txt`), Tables 1 (bipolar) and 2 (elliptical). The text layer was clean
(the minus sign is encoded as a special glyph and was converted); class counts match the paper exactly
(B1 64, B2 35, E1 55, E2 55; Table 3 counts plus the stated |b| > 20 rejections) and every recomputed
angle agrees with the published one within 0.5 deg, so the parse is reliable.

Columns: `Name` (common name only, as printed), `PNG_approx_from_lb`, `GLON`, `GLAT`, `class` (B1, B2, E1,
E2: Corradi & Schwarz 1995 types, 1/2 = classification quality), `EPA_deg`, `thetaG_folded_0_90_deg`
(published "vG", which they folded into 0 to 90 deg, i.e. the complement of the inclination to the
Galactic equator; 90 = parallel to the plane), `GPA_0_180_recomputed_deg` (mine, unfolded GPA from EPA),
`fold(recomputed)_minus_thetaG_deg`.

Frame: EPA equatorial (N through E); the published Galactic angle is folded to 0 to 90, so use the
recomputed 0 to 180 column for comparison with other GPAs.
Axis: the longer of the two symmetry axes, i.e. the polar outflow axis; point-symmetric bright
substructures ignored. Measurement: visual on narrow-band (H-alpha or [N II]) CCD images from the
literature, repeated by two authors; errors about 5 deg for B1, about 10 deg for B2 and E.
Caveats: bulge PNe were excluded by design (disc sample). M 2-55 appears in both E1 (PA 38) and E2 (PA 43)
in the printed tables; both rows are kept. Name spellings are as printed (e.g. "PN G321.6+0.2.2", "TH 2-B").

---

Overlaps worth noting for cross-checks: the 54 Weidmann CAM98 rows duplicate Corradi values. RZ13 say
they share 19 PNe with Weidmann; a naive match on `PNG_approx_from_lb` finds only 7 because of 0.1 deg
rounding, so match on coordinates. All 14 Tan+23 GPAs are also in RZ13 (exact PN G match). Most agree to
within about 10 deg (M 2-19: 165 vs 165), but H 2-13 differs by 56 deg (Tan 91, RZ13 147, where RZ13 call it
non-polar) and H 2-36 by 12 deg: independent visual PAs of compact bulge PNe can disagree a lot.
