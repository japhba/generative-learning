"""Build the comparison and observed examples from the completed confirmation."""
import base64
import json
from html import escape
from math import prod
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'experiments'))
from task_search_prompts import answer_prompt

root=PROJECT/'results/task_search'; out=PROJECT/'reports'
name='product_decomposition_k3_d3__default__1093732622.json'
world=json.loads((root/'decomposition_confirmation/worlds'/name).read_text())
evaluation=json.loads((root/'decomposition_confirmation/evaluations'/name).read_text())
own=[r['example'] for r in evaluation['generations']['self_fixed']]
for label,examples,query in [('teacher',world['practice'],world['tests'][0]),('own',own,world['tests'][0]),('generation',[],world['practice'][0])]:
    (out/f'task-search-{label}-prompt.txt').write_text(answer_prompt(world,examples,query))
result=next(r for r in json.loads((root/'final/analysis.json').read_text())['primary'] if r['task']=='product_decomposition_k3_d3')
assert result['own']==142/512 and result['teacher']==16/512 and result['zero']==43/512
assert world['practice'][0]['answer']=='[619, 148, 94]' and own[0]['answer']=='[88, 112, 887]'
answers={c['arm']:c['predictions'][0]['answer'] for c in evaluation['conditions'] if c['n']==32}
assert answers=={'teacher':'[101, 139, 526]','self_fixed':'[1, 1, 7469462]'}
def answer_cell(answer,target):
    product=prod(json.loads(answer)); status='Correct' if product==target else 'Incorrect'
    return f'<td><code>{escape(answer)}</code><br><span class="small">{status} · product {product:,}</span></td>'

practice_rows=[]; test_rows=[]
conditions={c['arm']:c for c in evaluation['conditions'] if c['n']==32}
for i in range(10):
    item=world['practice'][i]; target=item['metadata']['target']
    practice_rows.append(f'<tr><td>#{i+1} · <code>Q({target})</code></td>'+answer_cell(item['answer'],target)+answer_cell(own[i]['answer'],target)+'</tr>')
    item=world['tests'][i]; target=item['metadata']['target']
    responses=[conditions[arm]['predictions'][i] for arm in ['teacher','self_fixed']]
    assert all((prod(json.loads(r['answer']))==target)==r['correct'] for r in responses)
    test_rows.append(f'<tr><td>#{i+1} · <code>Q({target})</code></td>'+''.join(answer_cell(r['answer'],target) for r in responses)+'</tr>')
plot=base64.b64encode((out/'task-search-product-decomposition.png').read_bytes()).decode()
page='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Unjudged rehearsal — a matched comparison</title>
<style>
:root{color-scheme:light;--blue:#216897;--orange:#a95209;--ink:#172b3c;--muted:#536575;--line:#d9e1e6}
*{box-sizing:border-box}body{margin:0;background:#edf1f4;color:var(--ink);font:14px/1.4 system-ui,sans-serif}
main{max-width:1000px;margin:24px auto;padding:28px 34px;background:white;border:1px solid var(--line)}
header{border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:12px}h1{font-size:26px;line-height:1.15;margin:0 0 7px}h2{font-size:15px;margin:12px 0 6px}p{margin:5px 0}.eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:6px}.lead{font-size:15px}
code,pre{font-family:ui-monospace,monospace;font-size:12px}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;margin:5px 0;line-height:1.4}.question{background:#f3f6f8;padding:8px 11px;border-left:3px solid var(--ink)}
table{border-collapse:collapse;width:100%;table-layout:fixed;margin:8px 0}th,td{padding:7px 10px;border:1px solid var(--line);vertical-align:top;text-align:left}th:first-child,td:first-child{width:26%;font-weight:600}th{background:#f4f6f8}.teacher{color:var(--blue)}.own{color:var(--orange)}.teacher-bg{background:#f2f8fc}.own-bg{background:#fff6ed}.small{font-size:12px;color:var(--muted)}.columns{display:grid;grid-template-columns:1fr 1fr;gap:12px}.prompt{padding:8px 11px;border:1px solid var(--line)}.results{display:grid;grid-template-columns:30% 1fr;gap:18px;align-items:center}figure{margin:0}img{width:100%;height:auto;display:block}figcaption{font-size:11px;color:var(--muted);margin-top:2px}.score{font-size:22px;font-weight:700}.summary td:first-child{width:42%}.summary{margin:4px 0}.summary td{padding:5px 8px}.note{border-top:1px solid var(--line);padding-top:8px;margin-top:9px}footer{margin-top:9px;font-size:11px;color:var(--muted)}a{color:#216897;text-underline-offset:2px}
@page{size:A4 portrait;margin:8mm}
@media print{body{background:white;font-size:8.5pt;line-height:1.2}main{width:100%;max-width:none;margin:0;padding:0;border:0}h1{font-size:18pt}h2{font-size:11pt}.lead{font-size:10pt}code,pre,.small{font-size:8pt}pre{line-height:1.25}th,td{padding:2px 6px}.score{font-size:17pt}header{padding-bottom:8px;margin-bottom:8px}h2{margin:4px 0 3px}footer,figcaption{font-size:7pt}a{color:inherit}tr,figure{break-inside:avoid}thead{display:table-header-group}p{margin:2px 0}table{margin:4px 0}.question,.prompt{padding:5px 8px}.note{margin-top:5px;padding-top:5px}footer{margin-top:5px}.summary td:first-child,.summary th:first-child{width:34%}details:not([open])>:not(summary){display:none!important}}
@media screen and (max-width:650px){main{margin:0;padding:18px}.results{grid-template-columns:1fr}.columns{gap:6px}th,td{padding:6px}code,pre{font-size:10px}h1{font-size:23px}}
</style></head><body><main>
<header><div class="eyebrow">Qwen3.5-9B × Gemma 4 · frozen weights · 12 September 2026</div>
<h1>Teacher versus own rehearsal</h1>
<p><a href="#specificity">Reader–source specificity</a> · <a href="#witness">Earlier task search</a> · <a href="#constrained">Earlier factorization heatmaps</a></p>
<p class="lead">Latest: both models read both saved rehearsal sets, with matched questions and no feedback.</p></header>

<h2>Task and matched setup</h2>
<p class="question"><code>Q(T) = Return a JSON array of exactly 3 positive integers whose product is T. The order of the integers does not matter.</code></p>
<p>The teacher samples three independent integers uniformly from 1–999 and uses their product as T. Any valid decomposition is accepted, including <code>[1, 1, T]</code>; this shortcut is never suggested. Both conditions use the same four initial examples, rehearsal questions, order, and test questions.</p>
<p><strong class="teacher">Teacher context:</strong> sampler’s valid answers. <strong class="own">Own context:</strong> answers generated offline from the four initial examples, then inserted verbatim. Both are fresh calls to the same frozen model: no weight update or memory of authorship, and no filtering, correction, or feedback.</p>

<h2>1 · Ten actual rehearsal examples supplied as context</h2>
<p class="small">First 10 rehearsals in episode 1093732622, in original order. Correctness was checked afterwards and never shown to the model.</p>
<table aria-label="Actual rehearsal examples"><thead><tr><th>Assigned question</th><th class="teacher teacher-bg">Teacher answer</th><th class="own own-bg">Own generation — retained verbatim</th></tr></thead><tbody>__PRACTICE_ROWS__</tbody></table>

<h2>2 · Final prompts: only the additional answer strings differ</h2>
<p class="small">Shared template below; bracketed blocks abbreviate repeated pairs. Each additional pair is formatted as <code>Problem: Q(T)</code> followed by <code>Answer: [saved answer]</code>, using the respective column above.</p>
<pre class="question">Problem family: product_decomposition_k3_d3.
Examples: [same four Problem / Answer pairs]
Additional examples: [N teacher pairs OR N saved own pairs]
Solve this new problem. Return only the answer in the format requested by the problem, without working or explanations.
Problem: Q(T_test)
Answer:</pre>
<p class="small">One user message; no custom system message. Offline generation uses this template without additional examples (temperature 0.7, top-p 0.9), with no tests or previous own answers. Evaluation: temperature 0, top-p 1. Thinking disabled; maximum output 1,024 tokens in both stages.</p>

<h2 class="evaluation-heading">3 · Ten actual held-out evaluations: same questions, N = 32</h2>
<p class="small">First 10 tests in the same episode, in original order, with all 32 rehearsal pairs. These illustrate behavior, not its frequency.</p>
<table aria-label="Paired evaluation responses"><thead><tr><th>Held-out question</th><th class="teacher teacher-bg">Qwen with teacher context</th><th class="own own-bg">Qwen with own context</th></tr></thead><tbody>__TEST_ROWS__</tbody></table>

<h2>Aggregate evaluation and learning curve</h2>
<div class="results"><figure><img src="data:image/png;base64,__PLOT__" alt="Accuracy rises from 8.4 to 27.7 percent with own examples, while teacher examples decline to 3.1 percent."><figcaption>Pointwise 95% episode-bootstrap intervals.</figcaption></figure><div>
<p><strong>512 fresh tests:</strong> 32 episodes × 16 matched questions. Exact arithmetic checks three positive integers and their product. N = 0, 1, 2, 4, 8, 16, 32; prespecified endpoint: <strong>N = 32</strong>.</p>
<p><strong class="teacher">Teacher: 16/512 (3.1%)</strong><br><strong class="own">Own: 142/512 (27.7%)</strong><br>Initial four only: 43/512 (8.4%).</p>
<p>Own − teacher: <strong>+24.6 points</strong> (95% CI 17.2–32.8). Own − initial: <strong>+19.3 points</strong> (13.1–26.0). Joint Holm <strong>p = 0.00015</strong> across all 15 confirmation attempts; this is the sole confirmed positive.</p>
</div></div>
<p class="note"><strong>Interpretation.</strong> Only 65/1,024 own rehearsals were correct (6.3%), but identity decompositions account for 114/142 correct final answers (80.3%). This suggests selection of an easy valid strategy. The teacher is not optimized for instruction; token and compute budgets are not matched. The N = 2–32 error fit gives β = 0.070 (own) versus −0.009 (teacher), but does not establish an asymptotic power law or improved general factoring ability.</p>
<footer>Historical factorization prompts, episode 1093732622: <a href="task-search-teacher-prompt.txt">teacher evaluation</a> · <a href="task-search-own-prompt.txt">own evaluation</a> · <a href="task-search-generation-prompt.txt">first own generation</a>. Model: Qwen3.5-9B BF16/vLLM; no training. Source: <code>decomposition_confirmation</code> and <code>final/analysis.json</code>.</footer>
</main></body></html>'''
grid=json.loads((root/'initial_count_grid/analysis.json').read_text())
grid_cells={(r['initial_count'],r['arm'],r['rehearsal_count']):r for r in grid['cells']}
heatmap_section=(out/'initial-count-heatmap-section.html').read_text()
reading=f'<p><strong>Reading the grid.</strong> With no initial examples, accuracy is {grid_cells[(0,"self_fixed",0)]["accuracy"]:.1%} without rehearsal, {grid_cells[(0,"self_fixed",32)]["accuracy"]:.1%} with 32 own examples, and {grid_cells[(0,"teacher",32)]["accuracy"]:.1%} with 32 teacher examples. With 16 initial teacher pairs, the no-rehearsal baseline falls to {grid_cells[(16,"self_fixed",0)]["accuracy"]:.1%}. Teacher demonstrations therefore strongly reduce performance on this task; own rehearsal also improves the strong no-example baseline. This pattern is consistent with changing which existing solution strategy the model uses.</p>'
truncated_generations=sum(r['samples']-r['statuses']['ok'] for r in grid['generation_quality'])
truncated_finals=grid['verification']['call_statuses']['truncated']-truncated_generations
limits=f'<p class="small">Output-limit handling follows the original protocol: {truncated_generations}/3,072 rehearsal calls were truncated; their question slots remained with empty answer strings. {truncated_finals} final calls were truncated and scored incorrect. All raw completions are preserved. Completed own answers are used without correctness filtering.</p>'
heatmap_section=heatmap_section.replace('all saved answers are used unchanged','every rehearsal slot is retained')
heatmap_section=heatmap_section.replace('<details>',limits+'<details>')
heatmap_section=heatmap_section.replace('<div class="columns">',reading+'<div class="columns">',1)
heatmap_section=heatmap_section.replace('Initial context × rehearsal: matched accuracy heatmaps','Original task: two 1s allowed')
constrained_section=(out/'nontrivial-heatmap-section.html').read_text()
constrained=json.loads((root/'nontrivial_initial_count_grid/analysis.json').read_text())
cc={(r['initial_count'],r['arm'],r['rehearsal_count']):r for r in constrained['cells']}
constrained_reading=f'<p class="note"><strong>Reading the constrained result.</strong> The instruction does not ensure compliance: with K = 0 and N = 0, {round(256*cc[0,"self_fixed",0]["two_ones_fraction"])}/256 answers still contain two or more 1s. All are scored incorrect. With K = 4 and N = 32, accuracy is {cc[4,"self_fixed",32]["accuracy"]:.1%} for own context, {cc[4,"teacher",32]["accuracy"]:.1%} for teacher context, and {cc[4,"self_fixed",0]["accuracy"]:.1%} without rehearsal. Own accuracy at K = 4 is higher with one rehearsal ({cc[4,"self_fixed",1]["accuracy"]:.1%}) than with 32. The remaining source advantage therefore does not give a clean monotonic scaling curve. At K = 0, N = 32, all {cc[0,"self_fixed",32]["correct"]} correct own-context answers use the still-permitted single-1 strategy.</p>'
constrained_section=constrained_section.replace('</h2>','</h2>'+constrained_reading,1)
witness_section=(out/'witness-search-section.html').read_text()
specificity_section=(out/'specificity-section.html').read_text()
page=page.replace('</header>','</header>'+specificity_section+'<details><summary>Previous constructive task search and strategy-state toy</summary>'+witness_section+'</details><details><summary>Earlier factorization experiments, examples, and initial-count heatmaps</summary>')
page=page.replace('<footer>',heatmap_section+constrained_section+'<footer>').replace('</footer>','</footer></details>')
(out/'task-search-report.html').write_text(page.replace('__PLOT__',plot).replace('__PRACTICE_ROWS__',''.join(practice_rows)).replace('__TEST_ROWS__',''.join(test_rows)))
print(out/'task-search-report.html')
