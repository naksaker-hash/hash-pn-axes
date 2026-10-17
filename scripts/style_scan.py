"""Scan printed prose of a .tex file for the author's style rules: no hyphen, colon or semicolon in prose, no
we/our/us, -ise spelling, no 'Table/Figure X <verb>' openings. Math, comments, commands, labels, citations,
the author block and the bibliography are skipped. Usage: python style_scan.py file.tex"""
import re, sys
s = open(sys.argv[1]).read()
s = s[s.index(r"\begin{abstract}"):s.index(r"\bibliographystyle")]
s = re.sub(r"(?<!\\)%.*", "", s)                                   # comments
s = re.sub(r"\\begin\{keywords\}.*?\\end\{keywords\}", " ", s, flags=re.S)
s = re.sub(r"\$[^$]*\$", " MATH ", s)                               # inline math
s = re.sub(r"\\(cite[pt]?|ref|label|citet|citep|url|includegraphics|begin|end|section\*?|subsection)\*?(\[[^\]]*\])?\{[^}]*\}", " ", s)
s = re.sub(r"\\[a-zA-Z]+", " ", s)                                  # remaining commands
bad = []
for n, line in enumerate(s.splitlines(), 1):
    for m in re.finditer(r"(?<=[A-Za-z])-(?=[A-Za-z])|(?<!\d):|;|\bwe\b|\bour\b|\bus\b|\b(?!siz)\w+iz(e|es|ed|ing|ation)\b", line, re.I):
        bad.append((n, m.group(0), line.strip()[:110]))
    for m in re.finditer(r"(Table|Fig\.?|Figure)\s*~?\s*\{?\}?\s+(shows|lists|gives|summari[sz]es|presents|compares)", line):
        bad.append((n, m.group(0), line.strip()[:110]))
for b in bad:
    print(b)
print(len(bad), "issues")
