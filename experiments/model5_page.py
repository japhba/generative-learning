"""Build tex/model_5_lateral_verdict.html (OverLyX article) from experiments/model5_page.html: fill in the numbers from numbers.json
(model5_figures.py) and own_benefit_numbers.json next to it (model5_own_benefit_figures.py), typeset every \\(...\\) and \\[...\\] with KaTeX at build time (scripts are left alone) and inline KaTeX's stylesheet.
Figures are referenced from tex/figures/. Served at http://127.0.0.1:8095/practice-docs/model_5_lateral_verdict.html (reports/practice-docs -> tex/).
Usage: python model5_page.py <numbers.json> <out.html>"""

from datetime import datetime
import json
from pathlib import Path
import re
import sys

import numpy as np
from zoneinfo import ZoneInfo

from report_katex import stylesheet, typeset

ROOT = Path(__file__).resolve().parents[1]


def f(x): return f'{x:.2f}' if x >= 0.095 else (f'{x:.3f}' if x >= 0.001 else f'{x:.5f}'.rstrip('0'))                             # a loss in nats


def pct(x): return f'{100 * x:.0f}%' if x >= 0.095 else f'{100 * x:.1f}%'


def main():
    v = json.loads(Path(sys.argv[1]).read_text()); a, e, s, g, d, h = (v[k] for k in ('ablation', 'efficiency', 'source', 'gradient', 'depth', 'heatmap'))
    rows = []
    for n in ('1024', '2048', '4096', '8192', '16384', '32768'):
        r = e[n]; rows.append(f"<tr><td class=\"num\">{int(n):,}</td><td class=\"num\">{pct(1 - np.exp(-int(n) / 1024))}</td><td class=\"num\">{f(r['table'])}</td><td class=\"num\">{f(r['practice0.8'])}</td>"
                              f"<td class=\"num\">{f(r['recall']) if r['recall'] < 20 else '&gt; 20'}</td><td class=\"num\">{f(r['bottleneck'])}</td><td class=\"num\">{r['locked']}</td></tr>")
    gk = lambda arm, th, key: [g[f'{st}|{th}|{arm}'][key] for st in ('noise0.1', 'noise1.0', 'random')]
    best = np.array(h['kl']).min(0); i13, i22 = h['total'].index(8192), len(h['total']) - 1
    nums = dict(worlds=a['worlds'], start=f(a['start']), full=f(a['full']), table=f(a['table']), recall=f(a['recall']), locked=a['full_whole_worlds'], locked_min=pct(a['full_whole_min']), full_new=pct(a['full_new']),
                gen=f(a['gen']), no_veto=f(a['no_veto']), no_spread=f(a['no_spread']), no_spread_new=pct(a['no_spread_new']), private=f(a['private']), private_new=pct(a['private_new']), valid=pct(a['full_valid']),
                eff_rows='\n'.join(rows), locked_eff=e['4096']['locked'], x4096=f"{e['4096']['table'] / e['4096']['practice0.8']:.1f}", x16384=f"{e['16384']['table'] / e['16384']['practice0.8']:.1f}", p4096=f(e['4096']['practice0.8']), t4096=f(e['4096']['table']), p8192=f(e['8192']['practice0.8']), t8192=f(e['8192']['table']),
                p2048=f(e['2048']['practice0.8']), t2048=f(e['2048']['table']), p32768=f(e['32768']['practice0.8']), t32768=f(e['32768']['table']), r4096=f(e['4096']['recall']), r8192=f(e['8192']['recall']),
                own=f(s['own']), twin=f(s['twin']), twin_kl=f(s['twin_kl']), table_end=f(s['table_end']), own_wins=s['own_wins'], cross=f"{s['cross']:,}",
                teacher_end=f(s['teacher']), table_wins=h['table_wins'], totals=len(h['total']), h13=f(best[i13]), ht13=f(h['table'][i13]), h22=f(best[i22]), ht22=f(h['table'][i22]), h22share=f"{h['best'][i22]:g}",
                gw=f"{f(min(gk('wake', 0, 'kl')))} to {f(max(gk('wake', 0, 'kl')))}", gp=f"{f(min(gk('practice', 0, 'kl')))} to {f(max(gk('practice', 0, 'kl')))}", gvw=f"{pct(min(gk('wake', 0, 'valid')))} to {pct(max(gk('wake', 0, 'valid')))}",
                gvp=f"{pct(min(gk('practice', 0, 'valid')))} to {pct(max(gk('practice', 0, 'valid')))}", glock=sum(g[k]['locked'] for k in g if '|0|' in k), gel_t=pct(g['random|1|practice']['el_teacher']), gel_o=pct(g['random|1|practice']['el_own']),
                gth_w=f(g['random|1|wake']['kl']), gth_p=f(g['random|1|practice']['kl']), d_flat=f(d['16384']['flat']), d_first=f(d['16384']['first']), d_between=f(d['16384']['between']), d_flat2=f(d['131072']['flat']),
                d_first2=f(d['131072']['first']), d_between2=f(d['131072']['between']), built=datetime.now(ZoneInfo('Europe/London')).strftime('%d %B %Y, %H:%M %Z'))
    o = json.loads(Path(sys.argv[1]).with_name('own_benefit_numbers.json').read_text()); ob, od = o['block'], o['depth']; mix = np.array(od['mix']); bm = mix.min(0); tn = np.array(sorted(int(n) for n in ob)); tl = np.array([ob[str(n)]['table'] for n in tn])
    cell = lambda x: f'<td class="num">{f(x) if x < 20 else "&gt; 20"}</td>'; i2 = lambda t: od['total'].index(2 ** t)
    nums |= dict(ob_rows='\n'.join(f'<tr><td class="num">{int(n):,}</td>' + ''.join(cell(ob[n][k]) for k in ('table', 'note', 'eta1', 'found', 'oracle')) + '</tr>' for n in ('1024', '2048', '4096', '8192', '16384', '32768')),
                 ob_drows='\n'.join(f'<tr><td class="num">\\(2^{{{int(np.log2(t))}}}\\)</td>{cell(od["flat"][i])}{cell(bm[i])}<td class="num">{od["best_own_share"][i]:g}</td>{cell(od["clustered"][i])}</tr>' for i, t in enumerate(od['total'])),
                 ob_ratio=f"{np.exp(np.interp(-np.log(ob['2048']['found']), -np.log(tl), np.log(tn))) / 2048:.0f}", ob_own_below=f"{o['speed']['below_table']:,}", ob_dworlds=od['worlds'], ob_dc15=f(od['clustered'][i2(15)]), ob_df15=f(od['flat'][i2(15)]),
                 **{f'ob_{k}{n}': f(ob[str(n)][key]) for k, key in (('t', 'table'), ('p', 'eta1'), ('f', 'found'), ('o', 'oracle')) for n in (1024, 2048, 4096)}, **{f'ob_d{k}{t}': f(v[i2(t)]) for k, v in (('m', bm), ('f', od['flat'])) for t in (19, 20, 21)})
    assert all(ob[n]['eta1'] < ob[n]['table'] for n in ('2048', '4096', '8192', '16384')) and ob['1024']['eta1'] > ob['1024']['table'] and all(w == od['worlds'] for w in od['mix_wins_worlds'][i2(18):]) and all(bm[:i2(18)] > np.array(od['flat'][:i2(18)]))
    assert all(np.array(od['clustered']) <= bm) and (mix[:5, i2(19):] < np.array(od['flat'][i2(19):])).all() and 0.25 <= min(od['best_own_share'][i2(18):]) and max(od['best_own_share'][i2(18):]) <= 0.9                                    # claims made in section 6
    assert nums['glock'] == 0 and h['table_wins'] == len(h['total']) and e['16384']['table'] > e['16384']['practice0.8'] and e['32768']['table'] < e['32768']['practice0.8']   # claims made in the text
    html = (ROOT / 'experiments/model5_page.html').read_text()
    for key in sorted(set(re.findall(r'\{\{(\w+)\}\}', html))): html = html.replace('{{' + key + '}}', str(nums[key]))
    head, rest = html.split('<main>', 1); body, tail = rest.split('</main>', 1)
    scripts = re.findall(r'<script>.*?</script>', body, re.S); body = re.sub(r'<script>.*?</script>', '<!--SCRIPT-->', body, flags=re.S); body = typeset(body)
    for sc in scripts: body = body.replace('<!--SCRIPT-->', sc, 1)
    html = head.replace('/*__KATEX_CSS__*/', stylesheet()) + '<main>' + body + '</main>' + tail
    assert '\\(' not in re.sub(r'<annotation.*?</annotation>', '', body, flags=re.S).replace(''.join(scripts), '')
    Path(sys.argv[2]).write_text(html); print(sys.argv[2]); print(f'{len(html) / 1e6:.2f} MB')


if __name__ == '__main__': main()
