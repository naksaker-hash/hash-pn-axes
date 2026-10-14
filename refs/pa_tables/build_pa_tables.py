"""Parse downloaded PA tables (raw/) into clean CSVs. Reads data only."""
import csv, math, re, sys
from pathlib import Path

OUT = Path(sys.argv[1])
RAW = OUT / "raw"


def psi(l, b):
    """Corradi+98 angle between equatorial and Galactic north (deg, J2000)."""
    lr = math.radians(l - 32.9)
    br = math.radians(b)
    num = math.cos(lr)
    den = math.cos(br) / math.tan(math.radians(62.9)) - math.sin(br) * math.sin(lr)
    return math.degrees(math.atan2(num, den))


def gpa_from_epa(epa, l, b):
    return (epa + psi(l, b)) % 180.0


def fold90(a):
    a = a % 180.0
    return 180.0 - a if a > 90 else a


def png(l, b):
    return f"{l:05.1f}{'+' if b >= 0 else '-'}{abs(b):04.1f}"


def gal_from_png(s):
    m = re.match(r"(\d{3}\.\d)([+-]\d{2}\.\d)", s)
    return float(m.group(1)), float(m.group(2))


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(name, len(rows))


# ---------------- Rees & Zijlstra 2013 ----------------
tex = (RAW / "arxiv_1307.5711/x/AngleTableToCheck.tex").read_text()
rows = []
for line in tex.splitlines():
    if not re.match(r"^\d{3}\.\d", line):
        continue
    line = line.replace(r"\,--\,", "-").replace(r"\textit{HST}", "HST").rstrip(" \\")
    f = [x.strip() for x in line.split("&")]
    assert len(f) == 12, line
    for k in (0, 6):
        g, morph, pa, gpa, err, tel = f[k:k + 6]
        l, b = gal_from_png(g)
        gpa_calc = gpa_from_epa(float(pa), l, b)
        d = (float(gpa) - gpa_calc + 90) % 180 - 90
        rows.append(["PN G" + g, morph, int(pa), int(gpa), int(err), tel,
                     round(gpa_calc, 1), round(d, 1)])
rows.sort(key=lambda r: r[0])
write("rees_zijlstra2013_bulge.csv",
      ["PNG", "morph_class", "EPA_deg", "GPA_deg", "err_deg", "telescope",
       "GPA_recomputed_deg", "GPA_minus_recomputed_deg"], rows)

# ---------------- Tan et al. 2023 ----------------
txt = (RAW / "tan2023_t1_ascii.txt").read_text()
rows, part = [], None
for line in txt.splitlines():
    if line.startswith("(a)"):
        part = "a_confirmed_SP_binary"
    elif line.startswith("(b)"):
        part = "b_high_adf"
    f = line.split("\t")
    if not re.match(r"^\d{3}\.\d", f[0]):
        continue
    g = f[0].replace(" ", "")
    name = f[1].replace("^bull", "").strip()
    gpa = None if f[3].strip() == "cdots" else float(f[3])
    l, b = gal_from_png(g)
    epa_back = None if gpa is None else round((gpa - psi(l, b)) % 180.0, 1)
    note = ""
    if "^bull" in f[1]:
        note = "suspected SP binary; excluded from Tan+23 analysis"
    if gpa is None:
        note = "round/annular; no GPA"
    rows.append(["PN G" + g, name, part, f[2].strip(), gpa, f[4].strip(),
                 f[5].strip(), f[6].strip(), epa_back, note])
write("tan2023_bulge_binaries.csv",
      ["PNG", "Name", "subsample", "period_d_or_adf", "GPA_deg", "morph_HASH",
       "RAJ2000", "DEJ2000", "EPA_backcomputed_deg", "note"], rows)

# ---------------- Weidmann & Diaz 2008 ----------------
simbad = {}
for line in (RAW / "weidmann2008_table1.tsv").read_text().splitlines():
    if line.startswith("#") or not line.strip():
        continue
    f = line.split("\t")
    if len(f) < 7 or f[0].strip() in ("Name", "") or f[0].startswith("---"):
        continue
    simbad.setdefault(f[0].strip(), []).append((f[4].strip(), f[5].strip(), f[6].strip()))
rows = []
for line in (RAW / "weidmann2008_table1.dat").read_text().splitlines():
    name = line[0:19].strip()
    l = float(line[20:25]); b = float(line[26:32])
    epa = line[34:37].strip(); gpa = line[38:41].strip(); src = line[42:47].strip()
    epa = int(epa) if epa else None
    gpa = int(gpa) if gpa else None
    s = simbad.get(name, [("", "", "")])
    sname, ra, de = s.pop(0) if s else ("", "", "")
    png_cat = "PN " + name if re.match(r"^G\d{3}\.\d[+-]\d{2}\.\d", name) else ""
    calc = d = None
    if epa is not None:
        calc = round(gpa_from_epa(epa, l, b), 1)
        if gpa is not None:
            d = round((gpa - calc + 90) % 180 - 90, 1)
    rows.append([name, png_cat, "PN G" + png(l, b), l, b, epa, gpa, src, sname, ra, de, calc, d])
write("weidmann_diaz2008.csv",
      ["Name", "PNG_from_name", "PNG_approx_from_lb", "GLON", "GLAT", "EPA_deg", "GPA_deg",
       "source", "SimbadName", "RAJ2000_simbad", "DEJ2000_simbad", "GPA_recomputed_deg",
       "GPA_minus_recomputed_deg"], rows)

# ---------------- Corradi, Aznar & Mampaso 1998 ----------------
lines = (RAW / "corradi1998/corradi1998.txt").read_text().splitlines()
pat = re.compile(r"(\S+(?: \S+)*)\s+(\d{1,3}\.\d)\s+(¹?\d{1,2}\.\d)\s+(\d{1,3})\s+(\d{1,2})(?=\s|$)")
rows = []
for i, line in enumerate(lines, start=1):
    if 180 <= i <= 249:
        tab = 1
    elif 253 <= i <= 329:
        tab, cls = 2, "E1"
    elif 330 <= i <= 420:
        tab, cls = 2, "E2"
    else:
        continue
    for m in pat.finditer(line):
        if tab == 1:
            cls = "B1" if m.start() < 60 else "B2"
        name = m.group(1)
        l = float(m.group(2)); b = float(m.group(3).replace("¹", "-"))
        pa = int(m.group(4)); vg = int(m.group(5))
        calc = gpa_from_epa(pa, l, b)
        rows.append([name, "PN G" + png(l, b), l, b, cls, pa, vg, round(calc, 1),
                     round(fold90(calc) - vg, 1)])
write("corradi1998.csv",
      ["Name", "PNG_approx_from_lb", "GLON", "GLAT", "class", "EPA_deg", "thetaG_folded_0_90_deg",
       "GPA_0_180_recomputed_deg", "fold(recomputed)_minus_thetaG_deg"], rows)
from collections import Counter
print(Counter(r[4] for r in rows))
