"""Report for the two follow-ups to the cross-task search: native traces (NTRACE_PROTOCOL.md) and laid-out reasoning (LAYOUT_PROTOCOL.md)."""
from collections import defaultdict
import glob
import html
import json
from pathlib import Path

import numpy as np

import layout_search as LS
from layout_analyze import SHORT as LSHORT,bootstrap,sign_flip
from layout_tasks import correct
from ntrace_analyze import SHORT as NSHORT
from ntrace_search import DIRECTORY as NDIR,NAMES as NNAMES

REPORTS=Path(__file__).resolve().parents[1]/'reports';OUT=REPORTS/'own-reasoning-report.html'
STYLE=(Path(__file__).parent/'xspec_report.py').read_text().split('STYLE="""')[1].split('"""')[0]


def esc(x): return html.escape(str(x))
def pp(x): return f'{100*x:+.1f}'
def ci(m,lo,hi): return f'{pp(m)} <span class="small">[{pp(lo)}, {pp(hi)}]</span>'


def spend(directory): return sum(json.load(open(p))['response']['usage']['cost'] for p in glob.glob(str(directory/'api/*/*.json')) if len(Path(p).stem)==64)


def ntrace_section():
    a=json.loads((NDIR/'matrix/analysis.json').read_text());r=a['correct'];l=a['lenient'];pooled=np.array(r['pooled'])*100;init=np.array(r['initial'])*100
    rows=''.join(f"<tr><td>{NSHORT[m]}</td><td class='num'>{init[i]:.0f}</td>"+''.join(f"<td class='num{' own' if j==i else ''}'>{pooled[i,j]:.0f}</td>" for j in range(len(NNAMES)))+'</tr>' for i,m in enumerate(NNAMES))
    fam=''.join(f"<tr><td>{NSHORT[m]}</td><td class='num'>{ci(*v['own_minus_family'])}</td><td class='num'>{ci(*v['family_minus_cross'])}</td></tr>" for m,v in r['family'].items())
    gens=a['generations'];tasks=sorted({k.split('|')[0] for k in gens})
    gen=''.join(f"<tr><td>{NSHORT[m]}</td>"+''.join(f"<td class='num'>{100*gens[f'{t}|{m}']['correct']:.0f}%</td>" for t in tasks)+f"<td class='num'>{np.mean([gens[f'{t}|{m}']['chars'] for t in tasks]):.0f}</td></tr>" for m in NNAMES)
    return f"""<h2 id='ntrace'>1 · Native traces: free-form worked solutions, readers also reason at test</h2>
<p>Seven models (two Qwen, two Llama, Gemma 4, Phi-4, Mistral-Nemo) each wrote free-form worked solutions to 8 practice questions with <strong>no teacher examples</strong>, so every trace is in the model's own idiom. Every reader then solved 12 test questions per episode, reasoning step by step, after 8 traces from one source (7 sources, same questions). 3 tasks chosen by baseline only (calendar arithmetic, number sequences, time intervals) × 6 episodes.</p>
<p class='lead'><strong>Own traces help least.</strong> Own − other accuracy S = {ci(r['S'],*r['S_ci'])} pp (one-sided p for S &gt; 0: {r['p_one_sided']:.3f}); format-lenient scoring {pp(l['S'])} pp. By task: {', '.join(f'{t} {pp(v[0])}' for t,v in r['S_by_task'].items())}.</p>
<figure><img src='ntrace-matrix.png' alt='7x7 matrix'><figcaption>Left: accuracy after 8 traces, pooled over tasks (diagonal outlined). Right: double-centred reader × source interaction; negative diagonal = own traces worse than expected from reader and source main effects.</figcaption></figure>
<div class='scroll'><table><thead><tr><th>Reader \\ source</th><th class='num'>none</th>{''.join(f"<th class='num'>{NSHORT[m]}</th>" for m in NNAMES)}</tr></thead><tbody>{rows}</tbody></table></div>
<p>Own traces mostly leave accuracy at the no-example level, while foreign traces add 5–12 points for weaker readers. For Llama-3.1-8B, Llama-3.2-3B, Phi-4 and Nemo the gap is answered-but-wrong questions (not format); for Qwen3-30B partly longer, truncated answers. A model's own unjudged trace is what it would have produced anyway: it carries no new information, and its errors are the reader's own misconceptions, so they are reinforced.</p>
<h3>Family gradient (prediction: own &gt; same family &gt; other family)</h3><table><thead><tr><th>Reader</th><th class='num'>own − same-family source</th><th class='num'>same-family − cross-family</th></tr></thead><tbody>{fam}</tbody></table>
<details><summary>Practice-trace correctness and length</summary><table><thead><tr><th>Generator</th>{''.join(f"<th class='num'>{t}</th>" for t in tasks)}<th class='num'>mean chars</th></tr></thead><tbody>{gen}</tbody></table></details>"""


def layout_section():
    a=json.loads((LS.DIRECTORY/'analysis.json').read_text());design=LS.read(LS.DIRECTORY/'design.json');worlds=list(design['worlds']);N=LS.NAMES;M=len(N)
    acc={};comp=defaultdict(list)
    for n in worlds:
        w=LS.read(LS.DIRECTORY/'worlds'/f'{n}.json')
        for r in N:
            for s in LS.SOURCES:
                res=[LS.read(LS.evaluation_path(r,n,s,i)) for i in range(len(w['tests']))];acc[n,r,s]=np.mean([correct(LS.extract(x['text']),w['tests'][i]) for i,x in enumerate(res)])
                kind=s if s in ('initial','teacher') else ('own' if s.split(':')[0]==r else 'other')+'-'+s.split(':')[1]
                comp[r,kind]+= [x['status']=='ok' and bool(LS.LINE.match(x['text'].strip())) for x in res]
    def contrib(n):
        D=np.array([[acc[n,r,f'{s}:full']-acc[n,r,f'{s}:ans'] for s in N] for r in N]);return [D[i,i]-np.mean([D[i,j] for j in range(M) if j!=i]) for i in range(M)]
    C={n:contrib(n) for n in worlds};clean=[N.index(m) for m in ['qwen35_9b','qwen3_30b','llama31_8b','nemo_12b']]
    Lc=[np.mean([C[n][i] for i in clean]) for n in worlds];Lci=bootstrap(Lc);kinds=['initial','teacher','own-ans','other-ans','own-full','other-full']
    comp_rows=''.join(f"<tr><td>{LSHORT[r]}</td>"+''.join(f"<td class='num'>{100*np.mean(comp[r,k]):.0f}%</td>" for k in kinds)+'</tr>' for r in N)
    per=''.join(f"<tr><td>{LSHORT[r]}</td>"+''.join(f"<td class='num'>{pp(np.mean([C[n][i] for n in worlds if n.startswith(f)]))}</td>" for f in ['cipher','operator','units'])+f"<td class='num'><strong>{pp(np.mean([C[n][i] for n in worlds]))}</strong></td></tr>" for i,r in enumerate(N))
    stat=''.join(f"<tr><td>{label}</td><td class='num'>{ci(a[k]['mean'],*a[k]['ci'])}</td><td class='num'>{a[k]['p_one_sided']:.4f}</td><td>{', '.join(f'{f} {pp(v[0])}' for f,v in a[k]['by_family'].items())}</td></tr>" for k,label in [('L','L: own laid-out value beyond answers, vs foreign'),('V','V: own full − own answers-only'),('S','S: own full − foreign full')])
    trace=LS.read(LS.generation_path('gemma4_26b','operator__05',7))['text']
    return f"""<h2 id='layout'>2 · Laid-out reasoning: hidden rules, answer-only tests</h2>
<p>Your hypothesis: learning needs one's reasoning <em>laid out in front of oneself</em>. The native-traces design could not test it, because readers reasoned again at test time. Here every episode has a <strong>hidden rule</strong> shown by 4 examples (mystery operator a◇b = p·a+q·b+r; Caesar cipher; alien unit conversions with two hidden ratios). Six models reason out loud on 8 practice problems. At test, readers must <strong>answer directly, no working</strong>, after one source's practice either <strong>in full</strong> or <strong>reduced to its final answers</strong>. 18 episodes × 12 tests.</p>
<table><thead><tr><th>Frozen statistic</th><th class='num'>estimate (pp)</th><th class='num'>one-sided p</th><th>by family</th></tr></thead><tbody>{stat}</tbody></table>
<p class='lead'><strong>Laying out reasoning matters a lot (V), and the primary self-specific contrast is positive: L = {pp(a['L']['mean'])} pp, p = {a['L']['p_one_sided']:.4f}. But it does not survive scrutiny as evidence for legibility.</strong></p>
<figure><img src='layout-value.png' alt='full minus answers-only'><figcaption>Full reasoning − answers-only accuracy for each reader (row) and practice source (column); diagonal outlined. Colour scales differ per panel.</figcaption></figure>
<h3>Where L comes from</h3><table><thead><tr><th>Reader</th><th class='num'>cipher</th><th class='num'>operator</th><th class='num'>units</th><th class='num'>all</th></tr></thead><tbody>{per}</tbody></table>
<ol><li><strong>Compliance.</strong> Gemma and Phi-4 often ignore “no working”, start reasoning and hit the 32-token cap. Gemma complies far more after its <em>own</em> full traces, which by itself creates an own-source advantage:
<table><thead><tr><th>Reader</th>{''.join(f"<th class='num'>{k}</th>" for k in kinds)}</tr></thead><tbody>{comp_rows}</tbody></table><p class='small'>Share of outputs that are a bare “Answer:” line.</p></li>
<li><strong>Ceiling × answer quality (units).</strong> Llama and Nemo produce mostly wrong practice answers (44%, 17% correct), so their answers-only arm is very low, while <em>any</em> full trace reveals the ratios. Their large own (full − answers) gap is matched by Nemo reading Llama's poor traces (+61 pp).</li>
<li><strong>Strong source × strong reader (operator).</strong> Gemma alone often derives the rule in its traces, but runs out of tokens before the final answer; reading its own full traces it reaches 21% versus 0% with every other source. Gemma is also the reader best at applying a rule without working, so the diagonal is the one cell where a good source meets a capable reader. That multiplicative structure produces a positive interaction without any self-specific legibility.</li></ol>
<p>Restricted to the four readers that comply fully (both Qwens, Llama-3.1-8B, Nemo), L = {ci(np.mean(Lc),*Lci)} pp (p = {sign_flip(Lc):.3f}), entirely from units; own full traces are not better than foreign full traces for these readers.</p>
<details><summary>Example: Gemma's own operator trace (episode 5, rule 3a+3b−6), complete text</summary><pre>{esc(trace)}</pre></details>"""


def matched_section():
    import mlayout_search as A
    from task_search_api import ROOT
    a1=json.loads((A.DIRECTORY/'analysis.json').read_text());a2=json.loads((ROOT.resolve()/'mlayout2/analysis.json').read_text());p3=ROOT.resolve()/'mlayout3/analysis.json'
    names3=['qwen35_9b','qwen3_30b','gemma4_26b'];short={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','gemma4_26b':'Gemma-4-26B','phi4':'Phi-4'}
    def table(pooled,names,ref):
        head=''.join(f"<th class='num'>{short[m]}</th>" for m in names)
        rows=''.join(f"<tr><td>{short[r]}</td>"+''.join(f"<td class='num{' own' if i==j else ''}'>{100*pooled[i][j]:.0f}</td>" for j in range(len(names)))+f"<td class='num'>{100*ref[r]['initial']:.0f}</td><td class='num'>{100*ref[r]['answers']:.0f}</td></tr>" for i,r in enumerate(names))
        return f"<table><thead><tr><th>Reader \\ source of correct solutions</th>{head}<th class='num'>none</th><th class='num'>answers only</th></tr></thead><tbody>{rows}</tbody></table>"
    op1=a1['by_family_full']['operator'];names1=['qwen35_9b','qwen3_30b','gemma4_26b','phi4']
    html_=f"""<h2 id='matched'>3 · Correctness-matched: own vs foreign solutions of the same correctness</h2>
<p>Every source's practice consists of <strong>correct worked solutions to the same questions</strong> (several samples per question; the first qualifying one per model is used; only questions every model solved are kept). The test reply is <strong>prefilled with “Answer:”</strong>, so no reader can reason or ignore the format. No model's text is rewritten. Models are served from pinned weights on gpu_lowp (vLLM 0.28).</p>
<h3>3a · Answer-correct matching, 4 models (operator + units)</h3>
<p>S = {ci(a1['S']['mean'],*a1['S']['ci'])} pp, one-sided p = {a1['S']['p_one_sided']:.3f}; operator {pp(a1['S']['by_family']['operator'][0])} pp, units {pp(a1['S']['by_family']['units'][0])} pp (units saturated at 98–100%). Phi-4's “correct” operator solutions often reach the answer by nonsense arithmetic, so answer-matching did not equalise reasoning quality.</p>
{table(op1,names1,a1['reference']['operator'])}<p class='small'>Operator, {sum(1 for n in a1['included'] if n.startswith('operator'))} episodes.</p>
<h3>3b · Process-correct matching, 3 models (operator, 16 fresh episodes)</h3>
<p>A solution qualifies only if it reaches the correct answer <em>and</em> explicitly computes both partial products p·x and q·y. S = {ci(a2['S']['mean'],*a2['S']['ci'])} pp, one-sided p = {a2['S']['p_one_sided']:.3f}; odds-scale own effect {a2['logit_own']['gamma']:+.2f} [{a2['logit_own']['ci'][0]:+.2f}, {a2['logit_own']['ci'][1]:+.2f}] log-odds. Per reader: {', '.join(f"{short[r]} {pp(v[0])} [{pp(v[1])}, {pp(v[2])}]" for r,v in a2['per_reader_own'].items())}.</p>
{table(a2['pooled_full'],names3,a2['reference'])}<p class='small'>{len(a2['included'])} of 16 episodes had ≥ 3 questions solved process-correctly by all three models.</p>"""
    if p3.exists():
        a3=json.loads(p3.read_text());g=a3['primary_gemma_own']
        html_+=f"""<h3>3c · Confirmation of Gemma's own preference, process-correct and length-matched (16 fresh episodes)</h3>
<p>Pre-registered after 3a/3b singled out Gemma 4 (pooled post hoc: +9.4 pp [+3.3, +15.4]). Each model contributes the qualifying solution closest to the question's median length. <strong>Primary, Gemma own − other two: {ci(g['mean'],*g['ci'])} pp, one-sided p = {g['p_one_sided']:.3f}.</strong> S = {ci(a3['S']['mean'],*a3['S']['ci'])} pp; per reader: {', '.join(f"{short[r]} {pp(v[0])}" for r,v in a3['per_reader_own'].items())}. Mean solution length after matching: {', '.join(f"{short[m]} {v:.0f}" for m,v in a3['solution_chars'].items())} characters.</p>
{table(a3['pooled_full'],names3,a3['reference'])}<p class='small'>{len(a3['included'])} of 16 episodes included.</p>"""
    return html_


def further_section():
    from task_search_api import ROOT
    R=ROOT.resolve();short={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','gemma4_26b':'Gemma-4-26B','nemotron3_30b':'Nemotron-3-Nano','qwen3_30b_hybrid':'Qwen3-30B-A3B (hybrid)'}
    def matrix(pooled,names,ref=None):
        head=''.join(f"<th class='num'>{short[m]}</th>" for m in names)+("<th class='num'>none</th><th class='num'>answers only</th>" if ref else '')
        rows=''.join(f"<tr><td>{short[r]}</td>"+''.join(f"<td class='num{' own' if i==j else ''}'>{100*pooled[i][j]:.0f}</td>" for j in range(len(names)))+(f"<td class='num'>{100*ref[r]['initial']:.0f}</td><td class='num'>{100*ref[r]['answers']:.0f}</td>" if ref else '')+'</tr>' for i,r in enumerate(names))
        return f"<div class='scroll'><table><thead><tr><th>Reader \\ source</th>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
    t1=json.loads((R/'mthink/analysis.json').read_text());n1=['qwen35_9b','gemma4_26b','nemotron3_30b']
    html_=f"""<h2 id='further'>5 · Searching further</h2>
<h3>5a · Thinking traces as practice (closest to “legible only to oneself”)</h3>
<p>Practice = each model's <em>private thinking</em> (thinking mode on), matched as above: identical questions, process-correct, length-matched (≈ 7–8k characters each); test answer-only with “Answer:” prefill. Qwen3.5-9B, Gemma 4, Nemotron-3-Nano; 16 episodes.</p>
<p><strong>Pre-registered S = {ci(t1['S']['mean'],*t1['S']['ci'])} pp, one-sided p = {t1['S']['p_one_sided']:.4f}</strong>; logit {t1['logit_own']['gamma']:+.2f} [{t1['logit_own']['ci'][0]:+.2f}, {t1['logit_own']['ci'][1]:+.2f}]. Per reader: {', '.join(f"{short[r]} {pp(v[0])} [{pp(v[1])}, {pp(v[2])}]" for r,v in t1['per_reader_own'].items())}. Only Gemma answers directly well; the other two sit at 10–17%.</p>{matrix(t1['pooled_full'],n1,t1['reference'])}"""
    p2=R/'mthink2/analysis.json'
    if p2.exists():
        t2=json.loads(p2.read_text());n2=['qwen35_9b','gemma4_26b','nemotron3_30b','qwen3_30b_hybrid']
        html_+=f"""<h3>5b · Replication with a fourth thinking model (16 fresh episodes)</h3>
<p>Adds Qwen3-30B-A3B (hybrid thinking), a second reader that answers directly reasonably well. <strong>Pre-registered S = {ci(t2['S']['mean'],*t2['S']['ci'])} pp, one-sided p = {t2['S']['p_one_sided']:.4f}</strong>; secondary Gemma own − others {ci(t2['gemma_own']['mean'],*t2['gemma_own']['ci'])} pp (p = {t2['gemma_own']['p_one_sided']:.4f}); Qwen-pair H {pp(t2['qwen_pair_H'][0])} pp; logit {t2['logit_own']['gamma']:+.2f} [{t2['logit_own']['ci'][0]:+.2f}, {t2['logit_own']['ci'][1]:+.2f}]. Per reader: {', '.join(f"{short[r]} {pp(v[0])}" for r,v in t2['per_reader_own'].items())}.</p>{matrix(t2['pooled_full'],n2,t2['reference'])}"""
    c=json.loads((R/'mcot/analysis.json').read_text())
    html_+=f"""<h3>5c · Reasoning allowed at test, matched practice</h3>
<p>With the matched practice of 3c and step-by-step reasoning at test, every reader reaches 100% after a single correct solution from any source (N = 1 and N = 6): a ceiling, so no own-source effect can show. (An answer-extraction bug — “Final Answer:\\n\\nAnswer: 7” parsed as “Answer: 7” — first produced a spurious +18 pp for Qwen3-30B; fixed everywhere, it changed 140 of 4,860 scores here, 7 of 12,096 in the native-traces study and 125 of 53,248 practice checks.)</p>"""
    a=json.loads((R/'msft/analysis.json').read_text())['operator_cot'];n3=['qwen35_9b','qwen3_30b','gemma4_26b']
    rows=''.join(f"<tr><td>{short[st]}</td><td class='num'>{100*a['accuracy'][st]['base']:.0f}</td>"+''.join(f"<td class='num{' own' if st==src else ''}'>{100*a['accuracy'][st][src]:.0f}</td>" for src in n3)+f"<td class='num'>{ci(*a['own_minus_foreign'][st])}</td></tr>" for st in n3)
    html_+=f"""<h3>5d · Learning in the weights: LoRA on own vs foreign matched solutions</h3>
<p>Each model fine-tuned (LoRA, 2 epochs) on 557 process-correct, length-matched solutions to identical questions from each source; tested on fresh operator problems with step-by-step reasoning. <strong>Own − foreign, mean over students: {ci(*a['mean_over_students'])} pp.</strong> Gemma's solutions are the best training data for every student; own data starts at 0.10–0.15 loss per token and barely moves (nothing new to learn), foreign data at 0.32–1.09. Retention (GSM-symbolic) stays at 98–99% for all.</p>
<div class='scroll'><table><thead><tr><th>Student \\ training data</th><th class='num'>base</th>{''.join(f"<th class='num'>{short[m]}</th>" for m in n3)}<th class='num'>own − foreign</th></tr></thead><tbody>{rows}</tbody></table></div>
<figure><img src='msft-learning-curves.png' alt='learning curves'><figcaption>Training loss per solution token for every student × data source (own source thick).</figcaption></figure>"""
    return html_


def finetune_section():
    from task_search_api import ROOT
    R=ROOT.resolve();a=json.loads((R/'small/analysis.json').read_text());r=json.loads((R/'render/analysis.json').read_text());st={'qwen3_1_7b':'Qwen3-1.7B','llama32_3b':'Llama-3.2-3B','smollm3_3b':'SmolLM3-3B','phi4_mini':'Phi-4-mini'}
    sp=a['finetune_pooled']['mean']['peers'];rp=r['finetune']['pooled']['peers']['mean'];icl=a['icl']['mean']['peers']
    per=lambda d:', '.join(f"{st[k]} {pp(v[0])}" for k,v in d.items())
    return f"""<h2 id='finetune'>7 · Small students, fine-tuned: the one own advantage that holds up</h2>
<p>Four small students from four families (Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B, Phi-4-mini), LoRA-fine-tuned on GSM8K solutions with identical problems and correctness for every source, evaluated on 500 GSM8K test problems; two training seeds per run.</p>
<ul><li><strong>Own generated solutions vs peers' (1,718 problems): {ci(*sp)} pp</strong>, pooled over seeds (seed 0 {pp(a['finetune']['mean']['peers'][0])}, seed 1 {pp(a['finetune_seed1']['mean']['peers'][0])}). Per student: {per(a['finetune_pooled']['per_reader']['peers'])}. A large model's data (Qwen3-30B) is still better than own data ({pp(a['finetune_pooled']['mean']['large'][0])} pp own − large).</li>
<li><strong>Same content, own wording</strong> (each model rewrote the same reference solutions in its own words, every step and number kept): own rewording vs peers' rewordings <strong>{ci(*rp)} pp</strong> pooled over seeds, with the same per-student pattern ({per(r['finetune']['pooled']['peers']['per'])}). Rewordings of any model beat the terse original human solutions by about 9 pp.</li>
<li>In context, the same solutions give no own advantage ({ci(*icl)} pp). Re-scoring with the numeric scorer (last number on the answer line) leaves both fine-tuning effects unchanged (+1.6 pp each).</li></ul>
<p>The effect is small and carried by two of the four students (Qwen3-1.7B, SmolLM3), with Phi-4-mini reversed in both studies, so it reads as "some models learn better from text in their own style" rather than a general law.</p>"""


def edge_section():
    from task_search_api import ROOT
    e=json.loads((ROOT.resolve()/'edge/main2/analysis.json').read_text());s=lambda x:ci(x['mean'],*x['ci'])
    html_=f"""<h2 id='edge'>8 · At the edge of capability, matched in number correct</h2>
<p>Hypothesis: a model learns more easily from its own attempts, right or wrong, when the task is at the edge of what it can do. Design: hidden-rule “alien units” problems; four similar small models each make 6 attempts per practice problem; a practice set shows <strong>2 right + 2 wrong unchecked attempts</strong> with identical problems and right/wrong pattern for every source; the reader then solves new problems step by step. S = mean(own) − mean(peers) cancels “this source helps everyone”. 120 episodes, 65 with a matched mixed set.</p>
<ul><li><strong>Mixed practice: S = {s(e['primary_S_mixed'])} pp</strong> (pre-registered one-sided test for an own advantage p = {e['primary_S_mixed']['p_one_sided']:.2f}). Without the much weaker Llama-3.2-3B: {s(e['without_llama']['mixed'])} pp, every pair negative.</li>
<li>All-correct practice: {s(e['S_correct'])} pp; without Llama {s(e['without_llama']['correct'])} pp (a tight null).</li>
<li>Marking the wrong attempts as wrong changes nothing ({s(e['label']['S_minus_unlabelled'])} pp). Practice itself helps: {s(e['practice_effect']['mixed'])} pp (mixed), {s(e['practice_effect']['correct'])} pp (all correct).</li>
<li>A first run was void: exact-match scoring mis-read formats such as “Answer: **Answer: 30**”, so the right/wrong pattern was not really matched and readers that copied a format were marked wrong (protocol amendment 4).</li></ul>
<figure><img src='edge-of-capability.png' alt='edge of capability'><figcaption>Example episode, own − peers by practice type, and the reader × source matrix.</figcaption></figure>"""
    a2=ROOT.resolve()/'edge2/analysis.json'
    if a2.exists():
        f=json.loads(a2.read_text());n={'deepseek_flash':'DeepSeek-V4.1-Flash','gemma4_31b':'Gemma-4-31B','llama4_maverick':'Llama-4-Maverick'}
        html_+=f"""<h3>8b · Flash-tier models (OpenRouter)</h3>
<p>DeepSeek-V4.1-Flash, Gemma-4-31B and Llama-4-Maverick, thinking off so every attempt is fully visible, on the mystery-operator task (where all three are at the edge for hard rules). At this tier models fail by running out of tokens rather than by finishing wrong, so an unsuccessful attempt includes an unfinished one. {f['episodes_complete']} tested episodes with a matched 2 successful + 2 unsuccessful set.</p>
<ul><li><strong>S = {s(f['S_mixed'])} pp</strong> (two-sided p = {f['S_mixed']['p_two_sided']:.3f}). Per reader: {', '.join(f"{n[k]} {pp(v['mean'])}" for k,v in f['own_effect'].items())}. Pairwise: {', '.join(f"{' vs '.join(n[x].split('-')[0] for x in k.split('|'))} {pp(v['mean'])}" for k,v in f['pairwise_H'].items())}.</li>
<li>Practice effect {s(f['practice_effect'])} pp.</li></ul>
<figure><img src='edge2-flash.png' alt='flash tier'><figcaption>Reader × source accuracy and own − peers contrasts for the flash-tier panel.</figcaption></figure>"""
    return html_


def lang_section():
    from task_search_api import ROOT
    a=json.loads((ROOT.resolve()/'lang_main/analysis.json').read_text());s=lambda x:ci(x['mean'],*x['ci'])
    up=lambda k:' / '.join(pp(a[f'uplift{k}'][r]['mean']) for r in ['own','same language','other language'])
    return f"""<h2 id='lang'>10 · Language-dominant models: comprehensibility without obfuscation</h2>
<p>Idea (user): an own advantage should appear when foreign examples are correct but less comprehensible to the reader; foreign examples should still help, only less efficiently, and ideally the reader recognises them yet applies them worse. Panel: two models each dominant in English (Llama-3.1-8B, OLMo-3-7B), Japanese (Llama-3.1-Swallow-8B, Gemma-2-Llama-Swallow-9B) and Chinese (GLM-4-9B, InternLM3-8B). Every prompt is in the model's language and every model reasons in it (100% compliance). Problems are language-neutral alien-unit conversions; practice is 1, 2 or 4 correct worked solutions on identical problems for every source. A separate probe asks the reader to state the hidden rule. {a['usable_episodes']} of {a['prepared']} episodes usable; contrasts are on the double-centred reader × source matrix (reader skill and general source quality removed).</p>
<div class='scroll'><table><thead><tr><th></th><th class='num'>N = 1</th><th class='num'>N = 2</th><th class='num'>N = 4</th></tr></thead><tbody>
<tr><td><strong>Same-language peer − other language</strong> (pre-registered primary at N = 1)</td><td class='num'>{s(a['C1'])}</td><td class='num'>{s(a['C2'])}</td><td class='num'>{s(a['C4'])}</td></tr>
<tr><td>Own − same-language peer</td><td class='num'>{s(a['self1'])}</td><td class='num'>{s(a['self2'])}</td><td class='num'>{s(a['self4'])}</td></tr>
<tr><td>Recognition: same-language peer − other language</td><td class='num'>{s(a['recognition_C1'])}</td><td class='num'>{s(a['recognition_C2'])}</td><td class='num'>{s(a['recognition_C4'])}</td></tr>
<tr><td>Uplift over no practice: own / same language / other language</td><td class='num'>{up(1)}</td><td class='num'>{up(2)}</td><td class='num'>{up(4)}</td></tr></tbody></table></div>
<p><strong>Language matters, self does not.</strong> A solution written by a model of the reader's language helps it about 4–5 pp more than that source's general quality predicts, and the gap does not close with more examples. The reader's own solutions help <em>less</em> than a same-language peer's. Other-language solutions are recognised less often and, at N ≥ 2, applied worse even when the rule was recognised (+4–6 pp on the interaction scale; not at N = 1). The effect is carried by the English and Japanese readers; the Chinese readers show none.</p>
<figure><img src='lang-confirmatory.png' alt='language study'><figcaption>A: uplift by source relation and number of solutions. B: language and self contrasts. C: accuracy after one solution (blocks = same language).</figcaption></figure>"""


def apply_section():
    from task_search_api import ROOT
    a=json.loads((ROOT.resolve()/'lang_apply/analysis.json').read_text());s=lambda x:ci(x['mean'],*x['ci'])
    rows=''.join(f"<tr><td>{label}</td>"+''.join(f"<td class='num'>{s(a[f'{key}_N{k}'])}</td>" for k in (1,2,4))+"</tr>" for label,key in [('Standard (LANG design)','C_std'),('Reader states the rule first','C_self'),('… and stated it correctly','C_self_said_correct'),('<strong>Correct rule injected as the reader’s own words</strong>','C_inj')])
    return f"""<h2 id='apply'>11 · Recognised but applied worse? Holding recognition fixed</h2>
<p>After the practice, the reader is asked for the hidden rule in the same conversation; the assistant turn is then either the <em>correct</em> rule (recognition 100% for every source) or the reader's own answer, followed by the test problem. Episodes: the {a['episodes']['lang_main']} LANG episodes plus {a['episodes'].get('lang_main2',0)} fresh ones. Same-language peer − other language, interaction scale:</p>
<div class='scroll'><table><thead><tr><th></th><th class='num'>N = 1</th><th class='num'>N = 2</th><th class='num'>N = 4</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><strong>Pre-registered primary (rule injected, mean over N): {s(a['PRIMARY_C_inj_mean_over_N'])} pp, p = {a['PRIMARY_C_inj_mean_over_N']['p_two_sided']:.2f}.</strong> On the fresh episodes the original single-solution language effect did not replicate ({s(a['replication_C_std_N1_setB'])} pp), while the recognition gap did (+11 to +16 pp). Once the rule is known, demonstrations in any language add about the same (+9 to +21 pp over the rule alone), and the reader's own demonstrations stay less useful than a same-language peer's (−2 to −4 pp). The data favour “understood less, hence applied less” over “recognised but applied worse”.</p>
<figure><img src='lang-apply.png' alt='recognition held fixed'><figcaption>A: language contrast in the standard, self-stated and rule-injected conditions. B: what the demonstrations add once the rule is known.</figcaption></figure>"""


def proc_explorer():
    """Compact interactive explorer of PROC example rows at N = 1 (one worked example), embedded as inline JSON."""
    import proc_search as PS
    from proc_analyze import SHORT as PSHORT,relation as prel
    from proc_tasks import describe
    from mlayout_search import final_answer
    d=PS.DIRECTORY;done=json.loads((d/'complete.json').read_text());matched=json.loads((d/'matched.json').read_text());read=lambda q:json.loads(q.read_text())
    worlds={n:read(d/'worlds'/f'{n}.json') for n in done['usable']};names=PS.NAMES;c=lambda s:f'src:{s}:1'
    rec=lambda r,n,s:sum(PS.chosen(read(PS.probe_path(r,n,c(s),j))['text'])==worlds[n]['probes'][j]['answer'] for j in range(worlds[n]['length']))
    dep=lambda r,n,s:sum(PS.correct(read(PS.eval_path(r,n,c(s),i))['text'],worlds[n]['tests'][i]) for i in range(PS.TESTS))
    avg={r:{x:{'dep':float(np.mean([dep(r,n,s)/PS.TESTS for n in worlds for s in names if prel(r,s)==x])),'rec':float(np.mean([rec(r,n,s)/worlds[n]['length'] for n in worlds for s in names if prel(r,s)==x]))} for x in ['own','same language','other language']} for r in names}
    chosen=[n for n in worlds if n.startswith('L3')][:4]+[n for n in worlds if n.startswith('L4')][:4];episodes=[]
    for n in chosen:
        w=worlds[n];q=matched[n][0];test=w['tests'][0]
        cells={r:{s:(lambda t:{'text':t,'ok':bool(PS.correct(t,test)),'got':final_answer(t),'rec':int(rec(r,n,s)),'dep':int(dep(r,n,s))})(read(PS.eval_path(r,n,c(s),0))['text']) for s in names} for r in names}
        episodes.append({'id':n,'L':w['length'],'steps':[describe(x,'en').split(' (')[0].rstrip('.') for x in w['procedure']],'demo_input':w['pool'][q['index']]['input'],'demo_values':w['pool'][q['index']]['values'],
                         'test':test,'demos':{s:read(PS.gen_path(s,n,q['index']))['samples'][q['sample'][s]]['text'] for s in names},'cells':cells})
    default=next(((e['id'],r) for e in episodes for r in names for peer in [next(s for s in names if prel(r,s)=='same language')]
                  if e['cells'][r][r]['ok'] and not e['cells'][r][peer]['ok'] and e['cells'][r][r]['rec']==e['L'] and e['cells'][r][peer]['rec']==e['L']),(episodes[0]['id'],names[0]))
    data={'episodes':episodes,'models':[{'id':m,'name':PSHORT[m].split(' (')[0],'lang':{'en':'English','ja':'Japanese','zh':'Chinese'}[PS.LANGUAGE[m]]} for m in names],'avg':avg,'default':{'episode':default[0],'reader':default[1]},'tests':PS.TESTS}
    blob=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
    return """<style>
#px{border:1px solid var(--line);padding:8px 10px;margin:8px 0;font-size:12px}#px .bar{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;margin-bottom:6px}
#px select{font:inherit;padding:1px 4px}#px .src{display:flex;gap:4px;flex-wrap:wrap}#px .src button{font:inherit;font-size:11px;padding:2px 7px;border:1px solid var(--line);background:white;cursor:pointer;border-radius:10px}
#px .src button.on{color:white}#px .r-own{--c:#2a78d6}#px .r-same{--c:#1baf7a}#px .r-other{--c:#eb6834}#px .src button{border-color:var(--c);color:var(--c)}#px .src button.on{background:var(--c);color:white}
#px .proc{color:var(--muted);margin:2px 0 6px}#px .proc b{color:var(--ink)}#px .panes{display:grid;grid-template-columns:1fr 1fr;gap:8px}#px .pane{border:1px solid var(--line);min-width:0}
#px .pane h4{margin:0;padding:3px 6px;font-size:11px;background:#f4f6f8;border-bottom:1px solid var(--line);font-weight:600}#px pre{max-height:215px;overflow-y:auto;margin:0;padding:4px 6px;font-size:11px}
#px .stats{display:flex;flex-wrap:wrap;gap:4px 14px;margin-top:6px;color:var(--muted)}#px .stats b{color:var(--ink)}#px .tag{font-weight:600;color:var(--c)}
@media (max-width:650px){#px .panes{grid-template-columns:1fr}}
</style>
<div id='px'><div class='bar'><label>Episode <select id='px-ep'></select></label><label>Reader <select id='px-rd'></select></label><span class='small'>Source of the one worked example (<span class='tag r-own'>own</span> · <span class='tag r-same'>same-language peer</span> · <span class='tag r-other'>other language</span>):</span><span class='src' id='px-src'></span></div>
<div class='proc' id='px-proc'></div>
<div class='panes'><div class='pane'><h4 id='px-dh'></h4><pre id='px-demo'></pre></div><div class='pane'><h4 id='px-th'></h4><pre id='px-try'></pre></div></div>
<div class='stats' id='px-stats'></div></div>
<script type='application/json' id='px-data'>"""+blob+"""</script>
<script>
(()=>{const D=JSON.parse(document.getElementById('px-data').textContent),$=id=>document.getElementById(id),M=Object.fromEntries(D.models.map(m=>[m.id,m]));
const rel=(r,s)=>r===s?'own':M[r].lang===M[s].lang?'same':'other',label={own:'own',same:'same-language peer',other:'other language'},pct=x=>Math.round(100*x)+'%';
let ep=D.default.episode,rd=D.default.reader,src=D.default.reader;
for(const e of D.episodes){const o=new Option(`${e.id.replace('__',' · ')} (${e.L} steps)`,e.id);$('px-ep').add(o)}for(const m of D.models)$('px-rd').add(new Option(`${m.name} (${m.lang})`,m.id));
$('px-ep').onchange=e=>{ep=e.target.value;draw()};$('px-rd').onchange=e=>{rd=e.target.value;src=rd;draw()};
function draw(){const E=D.episodes.find(e=>e.id===ep),C=E.cells[rd][src],R=rel(rd,src);$('px-ep').value=ep;$('px-rd').value=rd;
 $('px-src').replaceChildren(...D.models.map(m=>{const b=document.createElement('button'),r=rel(rd,m.id);b.className='r-'+r+(m.id===src?' on':'');b.textContent=`${m.name} · ${r==='own'?'own':m.lang}`;b.title=label[r];b.onclick=()=>{src=m.id;draw()};return b}));
 $('px-proc').innerHTML='Hidden procedure (never shown to the reader): '+E.steps.map((s,i)=>`<b>${i+1}.</b> ${s}`).join(' → ');
 $('px-dh').innerHTML=`Worked example shown: ${E.demo_input} → ${E.demo_values.at(-1)} · written by <span class="tag r-${R}">${M[src].name} (${label[R]})</span>`;$('px-demo').textContent=E.demos[src].replace(/\\n\\s*\\n/g,'\\n');
 $('px-th').innerHTML=`${M[rd].name} applies it to ${E.test.input} (true ${E.test.answer}): <span class="${C.ok?'ok':'no'}">${C.ok?'✓ correct':'✗ got '+(C.got||'no answer')}</span>`;$('px-try').textContent=C.text.replace(/\\n\\s*\\n/g,'\\n');
 const A=D.avg[rd];$('px-stats').innerHTML=`<span>This cell: recognised <b>${C.rec}/${E.L}</b> steps · deployed <b>${C.dep}/${D.tests}</b> new numbers</span><span>${M[rd].name} over all episodes (1 example): `+['own','same language','other language'].map(x=>`${x==='own'?'own':x==='same language'?'peer':'other'} <b>${pct(A[x].dep)}</b> deployed / ${pct(A[x].rec)} recognised`).join(' · ')+'</span>'}
draw()})();
</script>"""


def proc_section():
    from task_search_api import ROOT
    a=json.loads((ROOT.resolve()/'proc_main/analysis.json').read_text());s=lambda x:ci(x['mean'],*x['ci'])
    rows=''.join(f"<tr><td>{label}</td>"+''.join(f"<td class='num'>{s(a[key.format(k=k)])}</td>" for k in (1,2,4))+"</tr>" for label,key in [('<strong>Own − same-language peer, deployment</strong>','self_deployment_N{k}'),('Own − same-language peer, recognition','self_recognition_N{k}'),('Own − peer, deployment, every step recognised','self_deployment_given_full_recognition_N{k}'),('Same-language peer − other language, deployment','language_deployment_N{k}'),('Same-language peer − other language, recognition','language_recognition_N{k}')])
    return f"""<h2 id='proc'>12 · Recognising vs deploying a demonstrated procedure: the first own advantage</h2>
<p>The earlier tasks could not separate recognition from deployment: “recognition” meant stating a rule (itself generative) and applying a known rule was trivial. Here writers are <em>told</em> a hidden 3–4-step number procedure (e.g. swap first and last digits → subtract 563 → sort digits ascending → multiply by 9) and write worked executions in their language; readers see only 1–4 worked examples and either <em>recognise</em> the steps (4-way questions, “which describes step 2?”) or <em>deploy</em> the procedure on new numbers. Panel: OLMo-3-7B and Phi-4-mini (English), Gemma-2-Llama-Swallow-9B and ABEJA-Qwen2.5-7B (Japanese), GLM-4-9B and InternLM3-8B (Chinese). {a['usable']} of {a['prepared']} episodes usable; interaction scale.</p>
<div class='scroll'><table><thead><tr><th></th><th class='num'>N = 1</th><th class='num'>N = 2</th><th class='num'>N = 4</th></tr></thead><tbody>{rows}</tbody></table></div>
<p><strong>Pre-registered: own − same-language peer in deployment {s(a['P1_self_deployment'])} pp (Holm p = {a['holm'][0]:.3f}); dissociation (deployment − recognition) {s(a['P2_dissociation'])} pp (Holm p = {a['holm'][1]:.3f}).</strong> Readers deploy a procedure better from their own worked executions than from a same-language peer's — about 10 pp when they recognised every step — while recognising them no better (slightly worse after one example). The advantage grows with procedure length and is not verbatim copying (4-gram overlap of test solutions with the demonstrations is equal). Language mainly affects recognition after a single example. Caveats: one task family, ~8B models, a panel different from §10, and a modest dissociation carried by N = 1.</p>
<figure><img src='proc-confirmatory.png' alt='procedure study'><figcaption>A: own − same-language peer for recognition and deployment. B: same-language peer − other language.</figcaption></figure>
<h3>More recognition data (73 episodes)</h3>
<p>Each reader × source cell got 3 × L step questions plus 4 verification items (number chains, half with one wrong step); 39 fresh episodes were added. The deployment own effect replicates on the fresh episodes (+5.5 pp [+3.1, +7.8]). Verification is at chance for every reader, so it carries no information. On step identification, four readers recognise every step from their own <em>and</em> their peer's example (97–100%; own − peer 0.0 pp), yet deploy the procedure <strong>+7.6 pp [+5.2, +10.0] better from their own</strong> (OLMo +15, GLM +13, Phi-4-mini +6, InternLM −3; pre-registered before the fresh episodes were analysed). The earlier sign-changing recognition effect came only from the two weak Japanese recognisers, in opposite directions.</p>
<figure><img src='proc-single.png' alt='recognition vs deployment'><figcaption>Each reader's advantage from its own worked example over a same-language peer's: in recognising the steps (x) vs in deploying the procedure (y). Four readers sit at x ≈ 0 (they recognise every step from either source) yet mostly above y = 0; the diamond pools them.</figcaption></figure>
<h3>Example explorer (one worked example, first new number)</h3>"""+proc_explorer()


def main():
    cost={'xspec':spend(NDIR.parent/'xspec'),'ntrace':spend(NDIR),'layout':spend(LS.DIRECTORY)}
    body=ntrace_section()+layout_section()+matched_section()+further_section()+"""<h2 id='open'>6 · What remained open after section 5</h2>
<p>Two limits recurred: readers that cannot answer directly sit at a floor, and readers allowed to reason hit a ceiling. Sections 7–8 address this with small students (fine-tuning) and with tasks placed at the edge of each panel's capability.</p>"""+finetune_section()+edge_section()+lang_section()+apply_section()+proc_section()+"""<h2 id='now'>13 · Where this leaves the idea</h2>
<p>Reading one's own worked solutions in context gives no advantage when the hard part is <em>working out a rule</em>: with correct practice own and peer solutions are equal, with mistakes own attempts help less (§8), and language affects only how often the rule is recognised (§10–11). The one in-context own advantage appears when the hard part is <em>executing a demonstrated multi-step procedure</em>: models deploy it better from their own worked executions than from a same-language peer's, without recognising them better (§12, pre-registered). In the weights, small students fine-tuned on their own wording learn about 1.5 pp more (§7). Next: longer procedures and a second procedural family, and fine-tuning on own vs peer executions.</p>"""
    page=f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Own versus foreign reasoning</title><style>{STYLE}</style></head><body><main>
<header><div class="eyebrow">Follow-ups to the cross-task search · updated 6 October 2026</div><h1>Does a model learn better from its own laid-out reasoning?</h1>
<p><a href='#ntrace'>Native traces</a> · <a href='#layout'>Laid-out reasoning</a> · <a href='#matched'>Correctness-matched</a> · <a href='#further'>Further search</a> · <a href='#finetune'>Small students</a> · <a href='#edge'>Edge of capability</a> · <a href='#lang'>Language</a> · <a href='#apply'>Recognition vs application</a> · <a href='#proc'>Procedures</a> · <a href='task-specificity-search.html'>Cross-task search</a></p>
<div class='lead'><p><strong>No robust own-source advantage anywhere.</strong> Once own and foreign practice are matched (same questions, process-correct, similar length), no model learns more from its own text — not from polished worked solutions (pre-registered confirmation: Gemma +1.1 pp [−4.4, +7.5]), not from its private thinking traces (first cohort +3.2 pp, pre-registered replication +0.7 pp [−1.1, +2.5]), and not when the same solutions are used as fine-tuning data (−0.4 pp [−4.6, +3.7]). Every positive exploratory cohort was singled out by Gemma 4 and failed fresh confirmation. Own text is measurably more <em>familiar</em> (fine-tuning loss 0.10–0.15 vs 0.3–1.1 per token), but familiarity did not turn into better learning. When readers may reason at test and practice is unmatched, own traces help <em>least</em>. <strong>Update (6 Oct):</strong> at the edge of capability with the number correct matched, own attempts help <em>less</em> than peers' when they include mistakes (§8); the one replicated own advantage is small and in fine-tuning of small students (+1.5 pp, §7). <strong>Update (7 Oct):</strong> comprehensibility matters at the level of language: a same-language peer's worked solution helps more than an other-language one (+4.1 pp, pre-registered), but the reader's own solutions help less than a same-language peer's (§10). Holding recognition fixed removes the language effect: other-language solutions are understood less, not applied worse (§11). <strong>But for deploying a demonstrated multi-step procedure, models do better from their own worked examples than from a same-language peer's (+5.3 pp, pre-registered), without recognising them better (§12).</strong></p></div></header>
{body}
<footer>Protocols: experiments/NTRACE_PROTOCOL.md, experiments/LAYOUT_PROTOCOL.md (all amendments listed). Data: results/task_search/ntrace and /layout. OpenRouter spend: cross-task ${cost['xspec']:.2f}, native traces ${cost['ntrace']:.2f}, laid-out ${cost['layout']:.2f}, total ${sum(cost.values()):.2f}.</footer></main></body></html>"""
    OUT.write_text(page);print(OUT)


if __name__=='__main__': main()
