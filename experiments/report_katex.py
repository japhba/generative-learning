"""Typeset \\(...\\) and \\[...\\] in an HTML string with KaTeX at build time, and inline its stylesheet and fonts.

The output needs no network access: the rendered markup, katex.min.css and its woff2 fonts all live in the page.
Rendering errors raise (throwOnError), so a broken formula fails the build instead of shipping raw TeX.
"""

import base64
import json
import os
from pathlib import Path
import re
import subprocess

KATEX = Path('/var/tmp/jbauer/report-tools/node_modules/katex')  # npm install katex@0.16.11
MATH = re.compile(r'\\\((.+?)\\\)|\\\[(.+?)\\\]', re.S)
RENDER = "const k=require(process.env.KATEX);let s='';process.stdin.on('data',d=>s+=d).on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s).map(([t,d])=>k.renderToString(t,{displayMode:d,throwOnError:true,output:'htmlAndMathml'})))))"


def typeset(html):
    found = [(m.group(1) or m.group(2), m.group(2) is not None) for m in MATH.finditer(html)]
    rendered = iter(json.loads(subprocess.run(['node', '-e', RENDER], input=json.dumps(found), capture_output=True, text=True, check=True, env=os.environ | {'KATEX': str(KATEX)}).stdout))
    return MATH.sub(lambda m: next(rendered), html)


def stylesheet():
    css = (KATEX / 'dist/katex.min.css').read_text()
    def font(m): return 'src:url(data:font/woff2;base64,' + base64.b64encode((KATEX / 'dist/fonts' / m.group(1)).read_bytes()).decode() + ') format("woff2")'
    css, count = re.subn(r'src:url\(fonts/([^)]+\.woff2)\) format\("woff2"\)[^;}]*', font, css)
    assert count > 15 and 'url(fonts/' not in css
    return css
