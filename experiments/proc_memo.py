"""Build tex/proc_memo.html: a short memo on the PROC studies (PROC, PROC-REC, PROC-LAYOUT) for collaborators, set like a
LaTeX article. Template experiments/proc_memo.html; formulas are typeset with KaTeX and the figures tex/figures/proc-single.png
and proc-layout.png inlined as data URIs, so the memo is one self-contained file (it can be mailed as is). The page is an
OverLyX article and is linked into reports/ for the 127.0.0.1:8095 server. With --fragment OUT it also writes the page without
its document wrapper (doctype, html, head, body tags), the form the claude.ai Artifact tool publishes.
Run: python experiments/proc_memo.py [--fragment OUT]"""

import argparse
import base64
import re
from pathlib import Path

from report_katex import stylesheet, typeset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tex/proc_memo.html'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--fragment', type=Path); a = ap.parse_args()
    html = typeset((ROOT / 'experiments/proc_memo.html').read_text())
    html = re.sub(r'\{\{IMG:([\w-]+)\}\}', lambda m: 'data:image/png;base64,' + base64.b64encode((ROOT / 'tex/figures' / f'{m.group(1)}.png').read_bytes()).decode(), html)
    html = html.replace('/*__KATEX_CSS__*/', stylesheet())
    assert '{{' not in html and '\\(' not in html and '\\[' not in html
    OUT.write_text(html); link = ROOT / 'reports' / OUT.name
    if not link.is_symlink(): link.symlink_to(OUT)
    print(OUT); print(f'http://127.0.0.1:8095/{OUT.name}'); print(f'{len(html.encode()) / 1e6:.2f} MB')
    if a.fragment:
        frag = re.sub(r'<!doctype html>\s*|</?html[^>]*>\s*|</?head>\s*|</?body>\s*|<meta charset="utf-8">\s*|<meta name="viewport"[^>]*>\s*', '', html)
        a.fragment.write_text(frag); print(a.fragment)


if __name__ == '__main__': main()
