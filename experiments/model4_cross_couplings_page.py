"""Build tex/model_4_cross_couplings.html (OverLyX article) from experiments/model4_cross_couplings_page.html: fill in the numbers from
/nfs/gatsbystor/jbauer/generative-learning/model4_cross_couplings/numbers.json (model4_cross_couplings_figures.py), typeset every
\\(...\\) and \\[...\\] with KaTeX at build time (scripts are left alone) and inline KaTeX's stylesheet. Figures are referenced from
tex/figures/. The page is reachable at http://127.0.0.1:8095/practice-docs/model_4_cross_couplings.html (reports/practice-docs -> tex/).
Usage: python model4_cross_couplings_page.py"""

from datetime import datetime
import json
from pathlib import Path
import re

from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset

ROOT = Path(__file__).resolve().parents[1]; NUMS = Path('/nfs/gatsbystor/jbauer/generative-learning/model4_cross_couplings/numbers.json'); OUT = ROOT / 'tex/model_4_cross_couplings.html'


def sci(x):                                                                                                       # 0.0023 -> 0.23%, 5.1e-14 -> \(5.1\cdot10^{-14}\)
    if x >= 0.995: return '1'
    if x >= 0.01: return f'{x:.2g}'.ljust(4, '0') if x >= 0.1 else f'{x:.2g}'
    m, e = f'{x:.1e}'.split('e'); return rf'\({m}\cdot10^{{{int(e)}}}\)'


def pct(x): return f'{100 * x:.2g}%' if x >= 0.001 else sci(x)


def main():
    v = json.loads(NUMS.read_text()); r2 = v['rho2']; assert (v['v'], v['m']) == (16, 4)
    rows = []
    for i, (st, hf, pv, fv) in enumerate(zip(r2['states'], r2['halves_fit'], r2['plain_valid'], r2['fitted_valid'])):
        l = i + 1; rows.append(f'<tr><td>{l}</td><td>{st:.1f}</td><td>{4 ** l}</td><td>{sci(hf)}</td><td>{sci(pv)}</td><td>{sci(fv)}</td></tr>'
                               .replace(f'>{st:.1f}<', f'>{st:.1f}<' if st < 10 else f'>{st:.0f}<'))
    nums = {'count_rows': '\n'.join(rows), 'fit4': pct(r2['fitted_valid'][3]), 'fit6': sci(r2['fitted_valid'][5]), 'samples': f"{v['samples']:,}",
            'built': datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z')}
    html = (ROOT / 'experiments/model4_cross_couplings_page.html').read_text()
    for key in sorted(set(re.findall(r'\{\{(\w+)\}\}', html))): html = html.replace('{{' + key + '}}', nums[key])
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    scripts = re.findall(r'<script>.*?</script>', body, re.S); body = re.sub(r'<script>.*?</script>', '<!--SCRIPT-->', body, flags=re.S)
    body = typeset(body)
    for sc in scripts: body = body.replace('<!--SCRIPT-->', sc, 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + body + '</main>' + tail
    assert '\\(' not in re.sub(r'<annotation.*?</annotation>', '', body, flags=re.S).replace(''.join(scripts), '')
    OUT.write_text(html); print(OUT); print('http://127.0.0.1:8095/practice-docs/model_4_cross_couplings.html'); print(f'{len(html) / 1e6:.2f} MB')


if __name__ == '__main__': main()
