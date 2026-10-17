"""Flatten the MLST manuscript and supplement into paper_mlst/submission (inputs and .bbl inlined, used figures copied,
both built there). The cover letter in submission/ is kept."""
import re, shutil, subprocess, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]; P = ROOT / 'paper_mlst'; OUT = P / 'submission'
(OUT / 'figures').mkdir(parents=True, exist_ok=True)
for f in (OUT / 'figures').glob('*'):
    f.unlink()
inline = lambda t: re.sub(r'\\input\{([\w\-]+)\}', lambda m: inline((P / f'{m.group(1)}.tex').read_text()), t)
used = set()
for name in ('main', 'supplement'):
    subprocess.run(['tectonic', '--keep-intermediates', f'{name}.tex'], cwd=P, check=True, capture_output=True)
    bbl = (P / f'{name}.bbl').read_text()
    s = inline((P / f'{name}.tex').read_text())
    s = re.sub(r'\\bibliographystyle\{mnras\}\s*\\bibliography\{refs\}', lambda m: bbl, s)
    assert '\\input{' not in s and '\\bibliography{' not in s
    (OUT / f'{name}.tex').write_text(s)
    used |= set(re.findall(r'figures/([\w\-]+\.pdf)', s))
for f in used:
    shutil.copy(P / 'figures' / f, OUT / 'figures' / f)
shutil.copy(P / 'mnras.bst', OUT / 'mnras.bst') if (P / 'mnras.bst').exists() else None
for name in ('main', 'supplement'):
    subprocess.run(['tectonic', f'{name}.tex'], cwd=OUT, check=True, capture_output=True)
print(sorted(used))
