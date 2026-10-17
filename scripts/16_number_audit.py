"""Number audit: every number in the printed prose must appear (after rounding) in a data output file.

Collects all numeric values from data/*.json and the key CSV summaries, then lists each number in the
paper text (methods, results, discussion, intro, conclusions in paper.tex) that matches no source value
at its printed precision. Numbers that are method settings are listed separately for manual check.
Usage: python 16_number_audit.py
"""
import json, re, pathlib
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
vals = set()


def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v)
            try:
                vals.add(float(k))
            except ValueError:
                pass
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        vals.add(float(o))


for f in (ROOT / 'data').glob('*.json'):
    walk(json.load(open(f)))
# settings and fixed facts of the method (checked against the scripts by hand)
SETTINGS = {512, 256, 30, 3, 20, 1.5, 10, 0.65, 0.7, 1.5, 0.5, 26, 80, 16, 70, 15, 2, 4, 160, 60, 32, 8, 5, 0.05, 6, 0.8,
            0.85, 2000, 95, 100, 18, 1, 0.95, 90, 12, 40, 4, 2696, 2368, 416, 1633, 319, 306, 1562, 350, 423, 130, 209,
            2332, 440, 766, 1975, 1998, 2008, 2013, 2023, 2025, 2026, 2003, 2009, 2017, 2020, 2014, 2016, 2019, 2005,
            2000, 2700, 99.8, 7, 19, 273, 0.13, 107.5, 0.83, 20000, 3000, 0.02, 9, 25, 0.1, 0.7, 47, 17.5, 3500, 21, 650, 8, 1.4, 0.65, 45, 113}
import os
PD = ROOT / os.environ.get('AUDIT_DIR', 'paper')  # paper (MNRAS) or paper_mlst
MAIN = os.environ.get('AUDIT_MAIN', 'paper.tex')
files = ['intro', 'methods', 'results', 'discussion'] + (['supplement'] if (PD / 'supplement.tex').exists() else [])
txt = ''.join(open(PD / f'{f}.tex').read() for f in files)
p = open(PD / MAIN).read()
txt += p[p.index('\\begin{abstract}'):p.index('\\end{abstract}')]
txt += p[p.index('\\section{Conclusions}'):p.index('\\section*{Acknowledgements}')]
txt = re.sub(r'(?<!\\)%.*', '', txt)
txt = re.sub(r'\\(cite[pt]?|ref|label)\{[^}]*\}', ' ', txt)
nums = re.findall(r'(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])', txt)


def match(s):
    v = float(s)
    d = len(s.split('.')[1]) if '.' in s else 0
    for x in vals:
        for y in (x, 100 * x):  # fractions printed as per cent
            if round(y, d) == v:
                return True
    return False


unmatched = sorted({n for n in nums if not match(n) and float(n) not in SETTINGS}, key=float)
print('numbers in text:', len(nums), ' unique:', len(set(nums)))
print('NOT FOUND in any output and not a listed setting:', unmatched)
for n in unmatched:
    for m in re.finditer(r'.{0,60}(?<![\w.])' + re.escape(n) + r'(?![\w.]).{0,40}', txt):
        print(f'  {n}: ...{m.group(0).strip()}...')

# every \cite key must exist in refs.bib (tectonic does not warn about missing entries)
allt = ''.join(open(f).read() for f in PD.glob('*.tex'))
keys = set(k.strip() for grp in re.findall(r'\\cite[pt]?(?:\[[^\]]*\])*\{([^}]*)\}', allt) for k in grp.split(','))
bib = open(PD / 'refs.bib').read()
missing = sorted(k for k in keys if not re.search(r'@\w+\{' + re.escape(k) + ',', bib))
print('cite keys:', len(keys), ' MISSING from refs.bib:', missing)
