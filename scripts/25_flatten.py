"""Flatten the manuscript for submission: inline every \\input and the compiled bibliography (.bbl),
copy figures and class files into paper/submission/, build both versions and compare page counts and text.
Usage: python 25_flatten.py
"""
import re, shutil, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / 'paper'
OUT = P / 'submission'
if OUT.exists():
    shutil.rmtree(OUT)
(OUT / 'figures').mkdir(parents=True)

# build the source version with intermediates to obtain the .bbl
subprocess.run(['tectonic', '--keep-intermediates', 'paper.tex'], cwd=P, check=True, capture_output=True)
bbl = (P / 'paper.bbl').read_text()


def inline(text):
    return re.sub(r'\\input\{([\w\-]+)\}', lambda m: inline((P / f'{m.group(1)}.tex').read_text()), text)


s = inline((P / 'paper.tex').read_text())
s = re.sub(r'\\bibliographystyle\{mnras\}\s*\\bibliography\{refs\}', lambda m: bbl, s)
s = s.replace('\\newcommand{\\pend}[1]{\\textcolor{red}{[PENDING: #1]}}\n', '')
assert '\\input{' not in s and '\\bibliography{' not in s and '\\pend{' not in s
(OUT / 'paper.tex').write_text(s)
for f in (P / 'figures').glob('*.pdf'):
    if re.search(re.escape('figures/' + f.name), s):
        shutil.copy(f, OUT / 'figures' / f.name)
for f in ['mnras.cls', 'mnras.bst']:
    shutil.copy(P / f, OUT / f)
subprocess.run(['tectonic', 'paper.tex'], cwd=OUT, check=True, capture_output=True)


def info(pdf):
    pages = int(re.search(r'Pages:\s+(\d+)', subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout).group(1))
    txt = subprocess.run(['pdftotext', str(pdf), '-'], capture_output=True, text=True).stdout
    return pages, txt


p1, t1 = info(P / 'paper.pdf'); p2, t2 = info(OUT / 'paper.pdf')
fonts = subprocess.run(['pdffonts', str(OUT / 'paper.pdf')], capture_output=True, text=True).stdout
not_emb = [l for l in fonts.splitlines()[2:] if len(l.split()) >= 7 and l.split()[-5] == 'no']  # emb column
print('pages source', p1, 'flattened', p2, 'text identical', t1 == t2)
print('figures copied', len(list((OUT / 'figures').glob('*.pdf'))), 'fonts not embedded', len(not_emb))
for l in not_emb[:5]:
    print('  ', l)
