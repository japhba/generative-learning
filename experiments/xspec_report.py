"""One-page HTML report for the cross-task reader-source specificity search (reports/task-specificity-search.html)."""
import html
import json
from pathlib import Path

from xspec_search import DIRECTORY,MODELS,SOURCES,read
from task_search_prompts import answer_prompt

REPORTS=Path(__file__).resolve().parents[1]/'reports'
OUT=REPORTS/'task-specificity-search.html'
NAMES={'qwen':'Qwen','gemma':'Gemma','teacher':'Teacher','initial':'Initial only'}
STYLE="""
:root{color-scheme:light;--blue:#216897;--orange:#a95209;--ink:#172b3c;--muted:#536575;--line:#d9e1e6;--good:#1d6b3a;--bad:#9b2c2c}
*{box-sizing:border-box}body{margin:0;background:#edf1f4;color:var(--ink);font:14px/1.4 system-ui,sans-serif}
main{max-width:1060px;margin:24px auto;padding:28px 34px;background:white;border:1px solid var(--line)}
header{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:12px}h1{font-size:26px;line-height:1.15;margin:0 0 7px}h2{font-size:17px;margin:20px 0 6px;border-top:1px solid var(--line);padding-top:12px}h3{font-size:14px;margin:12px 0 4px}p{margin:5px 0}
.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}.lead{font-size:15px}
code,pre{font-family:ui-monospace,monospace;font-size:12px}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;margin:3px 0;line-height:1.35;overflow-wrap:anywhere}
table{border-collapse:collapse;width:100%;margin:8px 0}th,td{padding:5px 8px;border:1px solid var(--line);vertical-align:top;text-align:left}th{background:#f4f6f8}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}.own{font-weight:700}.small{font-size:12px;color:var(--muted)}.hit{background:#fff6ed}.sel{background:#fde9d6}
.ok{color:var(--good)}.no{color:var(--bad)}figure{margin:8px 0}img{width:100%;height:auto;display:block}figcaption{font-size:12px;color:var(--muted);margin-top:2px}
details{margin:6px 0;border:1px solid var(--line);padding:4px 10px}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}.ex td{font-size:12px}.ex td pre{max-height:180px;overflow-y:auto}
footer{margin-top:14px;font-size:11px;color:var(--muted)}a{color:#216897}
@media screen and (max-width:650px){main{margin:0;padding:16px}th,td{padding:4px}h1{font-size:22px}}
"""


def esc(x): return html.escape(str(x))
def pct(x): return f'{100*x:.1f}%'
def pp(x): return f'{100*x:+.1f}'
def ci(v): return f'{pp(v[0])} <span class="small">[{pp(v[1])}, {pp(v[2])}]</span>' if len(v)==3 else f'{pp(v[0])} <span class="small">[{pp(v[0])}, {pp(v[1])}]</span>'
def label(setting): return esc(setting.replace('__default','').replace('__',' · ').replace('· k',' · K='))


def screen_section(screen,plan):
    s=screen['settings'];rows=sorted(s,key=lambda k:-s[k]['H']);gen=screen['generation_summary']
    out=[f"<h2 id='screen'>Screen: {len(s)} settings, 4 episodes × 8 tests, N = 16</h2>",
         "<p>Each point is one task setting. The upper-right quadrant is the pattern we are looking for: <strong>each reader does better with its own rehearsal answers than with the other model's</strong> (same questions, no feedback). Screen numbers only select candidates; with 32 tests per cell, single points are noisy (one test = 3.1 pp).</p>",
         "<figure><img src='xspec-screen-preferences.png' alt='Screen own-source preferences'><figcaption>Jittered by ±0.6 pp for visibility. Ringed: both preferences ≥ 2/32 (orange); advanced to confirmation (red, labelled).</figcaption></figure>",
         f"<p><strong>{len(screen['qualifying'])} settings</strong> met the frozen criterion (both preferences ≥ 6.25 pp); advanced (≤ 1 per family, top 6 by H): {', '.join(label(x) for x in screen['selected']) or 'none'}.</p>",
         "<details><summary>All screened settings, sorted by mean own-source preference H</summary><div class='scroll'><table><thead><tr><th>Setting</th><th class='num'>Calib. Q / G</th>"
         "<th class='num'>Q: init</th><th class='num'>Q: Q-src</th><th class='num'>Q: G-src</th><th class='num'>Q: teach</th><th class='num'>G: init</th><th class='num'>G: G-src</th><th class='num'>G: Q-src</th><th class='num'>G: teach</th>"
         "<th class='num'>P<sub>Q</sub></th><th class='num'>P<sub>G</sub></th><th class='num'>H</th><th class='num'>Rehearsal correct Q / G</th><th class='num'>Mean chars Q / G</th></tr></thead><tbody>"]
    for k in rows:
        v=s[k];a=v['accuracy'];c=plan['calibration'][k];cls='sel' if k in screen['selected'] else 'hit' if k in screen['qualifying'] else ''
        out.append(f"<tr class='{cls}'><td>{label(k)}</td><td class='num'>{pct(c['qwen'])} / {pct(c['gemma'])}</td>"
                   f"<td class='num'>{pct(a['qwen']['initial'])}</td><td class='num own'>{pct(a['qwen']['qwen'])}</td><td class='num'>{pct(a['qwen']['gemma'])}</td><td class='num'>{pct(a['qwen']['teacher'])}</td>"
                   f"<td class='num'>{pct(a['gemma']['initial'])}</td><td class='num own'>{pct(a['gemma']['gemma'])}</td><td class='num'>{pct(a['gemma']['qwen'])}</td><td class='num'>{pct(a['gemma']['teacher'])}</td>"
                   f"<td class='num'>{pp(v['P_Q'])}</td><td class='num'>{pp(v['P_G'])}</td><td class='num'><strong>{pp(v['H'])}</strong></td>"
                   f"<td class='num'>{pct(gen[k]['qwen']['correct'])} / {pct(gen[k]['gemma']['correct'])}</td><td class='num'>{gen[k]['qwen']['chars']:.0f} / {gen[k]['gemma']['chars']:.0f}</td></tr>")
    out.append("</tbody></table></div><p class='small'>Q = Qwen reads, G = Gemma reads; “Q-src” = rehearsal answers written by Qwen. Own-source cells in bold. P<sub>Q</sub> = Q:Q-src − Q:G-src, P<sub>G</sub> = G:G-src − G:Q-src, H = mean. Calibration: N = 0, 8 tests, separate world.</p></details>")
    return '\n'.join(out)


def confirmation_section(conf,design):
    s=conf['settings'];settings=[x['id'] for x in design['settings']];gen=conf['generation_summary']
    mutual=[k for k in settings if s[k]['mutual']]
    out=[f"<h2 id='confirmation'>Fresh confirmation: 16 episodes × 12 tests per setting</h2>",
         "<p>New questions, new rehearsal sets, new generations; nothing from the screen is reused. Primary endpoint: H at N = 16 (the screened count), one-sided exact sign-flip test over episodes, Holm-corrected across the confirmed settings. A setting counts as a <em>mutual</em> own-source advantage only if Holm p &lt; 0.05 <em>and</em> both readers' preferences have 95% intervals above zero.</p>",
         "<table><thead><tr><th>Setting / reader</th><th class='num'>Initial only<br>N = 0</th><th class='num'>Qwen source<br>N = 16</th><th class='num'>Gemma source<br>N = 16</th><th class='num'>Teacher source<br>N = 16</th></tr></thead><tbody>"]
    for k in settings:
        for rd in MODELS:
            c=s[k]['curves'][rd];cells=[f"<td class='num'>{pct(c['initial'][0])}</td>"]+[f"<td class='num{' own' if src==rd else ''}'>{pct(c[src]['16'][0])}</td>" for src in SOURCES]
            out.append(f"<tr><td>{label(k)}<br><span class='small'>{NAMES[rd]} reads</span></td>{''.join(cells)}</tr>")
    out.append("</tbody></table><p class='small'>192 test questions per cell. Bold = reader's own source.</p>")
    out.append("<table><thead><tr><th>Setting</th><th class='num'>Qwen: own − Gemma src</th><th class='num'>Gemma: own − Qwen src</th><th class='num'>Mean preference H</th><th class='num'>Holm p</th><th>Mutual?</th></tr></thead><tbody>")
    for k in settings:
        v=s[k];out.append(f"<tr><td>{label(k)}</td><td class='num'>{ci([v['P_Q'],*v['P_Q_ci']])}</td><td class='num'>{ci([v['P_G'],*v['P_G_ci']])}</td><td class='num'>{ci([v['H'],*v['H_ci']])}</td><td class='num'>{v['holm_p']:.3g}</td><td class='{'ok' if v['mutual'] else 'no'}'>{'yes' if v['mutual'] else 'no'}</td></tr>")
    out.append("</tbody></table><p class='small'>Percentage points; brackets are 95% paired episode-bootstrap intervals (20,000 resamples).</p>")
    out.append("<figure><img src='xspec-confirmation-preferences.png' alt='Confirmation preferences'></figure>")
    out.append("<table><thead><tr><th>Setting</th><th class='num'>Qwen own − teacher</th><th class='num'>Qwen own − initial</th><th class='num'>Gemma own − teacher</th><th class='num'>Gemma own − initial</th></tr></thead><tbody>")
    for k in settings:
        v=s[k];out.append(f"<tr><td>{label(k)}</td>"+''.join(f"<td class='num'>{ci(v[f'{rd}_own_minus_{x}'])}</td>" for rd in MODELS for x in ['teacher','initial'])+"</tr>")
    out.append("</tbody></table>")
    out.append("<h3>Learning curves (all N)</h3><figure><img src='xspec-confirmation-curves.png' alt='Learning curves'><figcaption>Shaded: 95% episode-bootstrap intervals. N = 0 is the shared initial-only baseline. "+(f"N = 32 dropped for context: {esc(conf['dropped_counts'])}." if conf['dropped_counts'] else '')+"</figcaption></figure>")
    out.append("<h3>Rehearsal-source diagnostics</h3><table><thead><tr><th>Setting</th><th class='num'>Qwen answers correct</th><th class='num'>Gemma answers correct</th><th class='num'>Qwen / Gemma truncated</th><th class='num'>Qwen / Gemma mean chars</th></tr></thead><tbody>")
    for k in settings:
        g=gen[k];out.append(f"<tr><td>{label(k)}</td><td class='num'>{pct(g['qwen']['correct'])}</td><td class='num'>{pct(g['gemma']['correct'])}</td><td class='num'>{pct(g['qwen']['truncated'])} / {pct(g['gemma']['truncated'])}</td><td class='num'>{g['qwen']['chars']:.0f} / {g['gemma']['chars']:.0f}</td></tr>")
    out.append("</tbody></table><p class='small'>Checked after inference only; never used to filter what readers saw.</p>")
    return '\n'.join(out),mutual


def examples(phase,setting,count=8):
    directory=DIRECTORY/phase;scores=read(directory/'scores.json');name=f'{setting}__00';world=read(directory/'worlds'/f'{name}.json')
    gen={(g['model'],g['index']):g for g in scores['generations'] if g['setting']==setting and g['episode']==0}
    ev={(r['reader'],r['source'],r['n'],r['index']):r for r in scores['evaluations'] if r['setting']==setting and r['episode']==0}
    mark=lambda ok:f"<span class='{'ok' if ok else 'no'}'>{'✓' if ok else '✗'}</span>"
    n=16
    out=[f"<details><summary>{label(setting)}: episode 0, first {count} rehearsals and first {count} tests (complete text)</summary>",
         "<p class='small'>Rehearsal answers in original order; all were replayed. ✓/✗ added for this report only. “…” marks the 256-token output cap.</p>",
         "<div class='scroll'><table class='ex'><thead><tr><th style='width:34%'>Rehearsal question</th><th>Teacher</th><th>Qwen generation</th><th>Gemma generation</th></tr></thead><tbody>"]
    for i,entry in enumerate(world['practice'][:count]):
        cells=[f"<td><pre>{esc(entry['answer'])}</pre></td>"]
        for m in MODELS:
            g=read(directory/'generations'/m/f'{name}__{i:02}.json');cells.append(f"<td><pre>{esc(g['answer'])}{' …' if g['call']['status']!='ok' else ''}</pre>{mark(gen[m,i]['correct'])}</td>")
        out.append(f"<tr><td><pre>{esc(entry['question'])}</pre></td>{''.join(cells)}</tr>")
    out.append("</tbody></table></div>")
    for rd in MODELS:
        out.append(f"<h3>{NAMES[rd]} reads (N = {n})</h3><div class='scroll'><table class='ex'><thead><tr><th style='width:26%'>Test question</th><th>Truth</th><th>Initial only</th><th>Qwen source</th><th>Gemma source</th><th>Teacher source</th></tr></thead><tbody>")
        for i,entry in enumerate(world['tests'][:count]):
            cells=[]
            for src,nn in [('initial',0)]+[(x,n) for x in SOURCES]:
                r=read(directory/'evaluations'/rd/f'{name}__{src}__n{nn}__{i:02}.json');cells.append(f"<td{' class=own' if src==rd else ''}><pre>{esc(r['answer'])}{' …' if r['call']['status']!='ok' else ''}</pre>{mark(ev[rd,src,nn,i]['correct'])}</td>")
            out.append(f"<tr><td><pre>{esc(entry['question'])}</pre></td><td><pre>{esc(entry['answer'])}</pre></td>{''.join(cells)}</tr>")
        out.append("</tbody></table></div>")
    prompt=answer_prompt(world,[{'question':e['question'],'answer':read(directory/'generations'/'qwen'/f'{name}__{i:02}.json')['answer']} for i,e in enumerate(world['practice'][:n])],world['tests'][0])
    out.append(f"<details><summary>Exact prompts</summary><p><strong>Offline generation prompt</strong> (rehearsal 0):</p><pre>{esc(answer_prompt(world,[],world['practice'][0]))}</pre><p><strong>Final prompt with Qwen source</strong>, test 0, N = {n}; other sources change only the answer strings:</p><pre>{esc(prompt)}</pre></details></details>")
    return '\n'.join(out)


def main():
    calib=read(DIRECTORY/'calibration/summary.json');plan=read(DIRECTORY/'screen_plan.json');screen=read(DIRECTORY/'screen/analysis.json');sdesign=read(DIRECTORY/'screen/design.json')
    has_conf=(DIRECTORY/'confirmation/analysis.json').exists()
    cost=sum(json.loads(p.read_text())['response']['usage']['cost'] for p in (DIRECTORY/'api').glob('*/*.json') if len(p.stem)==64)
    body=[];mutual=[]
    if has_conf:
        conf=read(DIRECTORY/'confirmation/analysis.json');cdesign=read(DIRECTORY/'confirmation/design.json');section,mutual=confirmation_section(conf,cdesign)
        settings=[x['id'] for x in cdesign['settings']]
        body.append(section)
        if (DIRECTORY/'headline.json').exists(): body.append(read(DIRECTORY/'headline.json')['notes'])
        body.append("<h2 id='examples'>What the readers actually saw and answered</h2>"+''.join(examples('confirmation',k) for k in settings))
    body.append(screen_section(screen,plan))
    if not has_conf and screen['selected']: body.append("<h2>Screen examples</h2>"+''.join(examples('screen',k) for k in screen['selected']))
    lead=read(DIRECTORY/'headline.json')['html'] if (DIRECTORY/'headline.json').exists() else ''
    design=f"""<h2 id='design'>Design</h2>
<p>Same inference as the 12 September comparison: pinned local Qwen3.5-9B and Gemma 4 26B A4B via OpenRouter (BF16; DekaLLM for calibration and screen; in the confirmation each episode is pinned to one of DekaLLM / NextBit / Novita for throughput, so all own-versus-other comparisons stay within one provider), thinking off, one problem per call, 256-token cap. Each model answers every rehearsal question offline (T = 0.7, conditioned only on the K initial teacher examples); every saved answer, right or wrong, truncated or not, is replayed verbatim. Readers evaluate at T = 0 with N rehearsal examples from one source. No author labels, no feedback, no filtering.</p>
<p><strong>Funnel.</strong> All {len(calib['accuracy'])} length-feasible reasoning_gym settings (spec × K ∈ {{0, 4}}) were calibrated at N = 0 → {len(calib['eligible'])} not at both-floor/both-ceiling → best variant per family × K, capped at 80 by baseline spread → {len(sdesign['settings'])} screened after {len(sdesign['exclusions'])} construction failures → {len(screen['selected'])} met the screen rule (≤ 6 allowed) and were confirmed. Selection used only baselines (screen) or screen outcomes (confirmation); confirmations are fresh.</p>
<p class='small'>Protocol: <code>experiments/XSPEC_PROTOCOL.md</code> (frozen before inference). OpenRouter spend for this study: ${cost:.2f}.</p>"""
    page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Own versus other rehearsal</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">Qwen3.5-9B × Gemma 4 · frozen weights · cross-task search · 5 October 2026</div><h1>Own versus other model's rehearsal, across tasks</h1>
<p>{"<a href='#confirmation'>Confirmation</a> · <a href='#examples'>Examples</a> · " if has_conf else ''}<a href='#screen'>Screen</a> · <a href='#design'>Design</a> · <a href='task-search-report.html'>Earlier report</a></p>
<div class="lead">{lead}</div></header>
{''.join(body)}{design}
<footer>Data: results/task_search/xspec (= /var/tmp/jbauer/generative_task_search_20260910/xspec). Code: experiments/xspec_*.py.</footer></main></body></html>"""
    OUT.write_text(page);print(OUT)


if __name__=='__main__': main()
