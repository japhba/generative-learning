"""Build the complete source-only search report and compact paired learning curves."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/matched_source_icl'
NAMES={'qwen_flash':'Qwen3.5 Flash','gemini31':'Gemini 3.1 Flash Lite'}


def read(phase): return json.loads((DATA/phase/'analysis.json').read_text())
def row(cell,source,n): return next(r for r in cell['rows'] if r['source']==source and r['budget']==n)
def contrast(cell,control,n): return next(r for r in cell['comparisons'] if r['control']==control and r['budget']==n)
def interval(values,scale=100): return f'[{scale*values[0]:+.2f}, {scale*values[1]:+.2f}]'
def label(cell): return cell['family'].replace('_',' ')+f" / {cell['level']}"


def effect_table(cells,reference,n):
    text='| Learner | Task / setting | Self | Teacher | Difference, percentage points [95% CI] | Holm p |\n|---|---|---:|---:|---|---:|\n'
    for cell in cells:
        model=cell['model']; other=reference[model]; delta=contrast(cell,other,n)
        text+=f"| {NAMES[model]} | {label(cell)} | {100*row(cell,model,n)['accuracy']:.2f}% | {100*row(cell,other,n)['accuracy']:.2f}% | {100*delta['accuracy_difference']:+.2f} {interval(delta['ci95'])} | {delta['holm_p']:.3f} |\n"
    return text


def slope_table(cells,reference,window):
    text='| Learner | Task / setting | Self beta | Teacher beta | Difference [95% CI] |\n|---|---|---:|---:|---|\n'
    for cell in cells:
        model=cell['model']; other=reference[model]
        fits={r['source']:r for r in cell['fits'] if r['window']==window}
        if 'generated_minus_'+other not in fits: continue
        delta=fits['generated_minus_'+other]
        text+=f"| {NAMES[model]} | {label(cell)} | {fits[model]['beta']:.3f} | {fits[other]['beta']:.3f} | {delta['beta']:+.3f} {interval(delta['ci95'],1)} |\n"
    return text


def effect_plot(cells,references,title,destination):
    fig,axes=plt.subplots(1,2,sharey=True,layout='constrained')
    for ax,model in zip(axes,('qwen_flash','gemini31'),strict=True):
        selected=[c for c in cells if c['model']==model]
        for cell in selected:
            contrasts=sorted([c for c in cell['comparisons'] if c['control']==references[model]],key=lambda r:r['budget'])
            ns=[0]+[r['budget'] for r in contrasts]; means=[0]+[100*r['accuracy_difference'] for r in contrasts]
            line,=ax.plot(ns,means,marker='o',label=cell['family'].replace('_',' '))
            ax.fill_between(ns,[0]+[100*r['ci95'][0] for r in contrasts],[0]+[100*r['ci95'][1] for r in contrasts],alpha=.12,color=line.get_color())
        ax.axhline(0,color='gray',linestyle=':'); ax.set_xscale('symlog',base=2,linthresh=1); ax.set_xticks(ns,labels=ns)
        ax.set(xlabel='Additional examples N',title=NAMES[model]); ax.grid(alpha=.2)
    axes[0].set_ylabel('Self minus teacher accuracy (points)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2)
    fig.suptitle(title); fig.savefig(destination); plt.close(fig)


def accuracy_plot(cells,references,titles,teacher_label,destination):
    fig,axes=plt.subplots(1,2,sharey=True,layout='constrained')
    for ax,cell,title in zip(axes,cells,titles,strict=True):
        zero=row(cell,'zero',0); model=cell['model']
        for source,name in ((model,'Self-generated'),(references[model],teacher_label)):
            points=[zero]+sorted([r for r in cell['rows'] if r['source']==source],key=lambda r:r['budget'])
            ns=[r['budget'] for r in points]
            line,=ax.plot(ns,[r['accuracy'] for r in points],marker='o',label=name)
            ax.fill_between(ns,[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.12,color=line.get_color())
        ax.set_xscale('symlog',base=2,linthresh=1); ax.set_xticks(ns,labels=ns)
        ax.set(xlabel='Additional examples N',title=title,ylim=(0,1)); ax.grid(alpha=.2)
    axes[0].set_ylabel('Held-out accuracy')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncols=2)
    fig.savefig(destination); plt.close(fig)


def main():
    screen=read('screen'); mainq=read('confirmation'); maing=read('confirmation_gemini')
    tracescreen=read('trace_screen'); trace=read('trace_confirmation')
    bbhscreen=read('bbh_screen'); bbh=read('bbh_confirmation'); bbhreplay=read('bbh_label_replay'); bbhrefs={'qwen_flash':'oracle'}
    maincells=mainq['cells']+maing['cells']; mainrefs={'qwen_flash':'oracle','gemini31':'oracle'}
    tracerefs=trace['reference_by_model']; tracecells=trace['cells']
    effect_plot(maincells,mainrefs,'Fresh confirmation: answer source only',ROOT/'reports/matched-source-confirmation.png')
    effect_plot(tracecells,tracerefs,'Fresh confirmation: full solution source only',ROOT/'reports/matched-source-traces.png')
    accuracy_plot([c for c in maincells if c['family']=='shortest_paths'],mainrefs,['Shortest paths\n'+NAMES[model] for model in ('qwen_flash','gemini31')],
                  'Exact teacher',ROOT/'reports/matched-source-learning.png')
    accuracy_plot(bbh['cells'],bbhrefs,['Formal fallacies','Translation error detection'],'Published teacher',ROOT/'reports/matched-source-bbh.png')
    accuracy_plot(bbhreplay['cells'],bbhrefs,['Formal fallacies','Translation error detection'],'Published teacher',ROOT/'reports/matched-source-bbh-labels.png')
    screen_table='| Task / setting | Initial examples only | Self, N=16 | Exact teacher, N=16 | Difference (points) |\n|---|---:|---:|---:|---:|\n'
    for cell in screen['cells']:
        model=cell['model']; delta=contrast(cell,'oracle',16)
        screen_table+=f"| {label(cell)} | {100*row(cell,'zero',0)['accuracy']:.2f}% | {100*row(cell,model,16)['accuracy']:.2f}% | {100*row(cell,'oracle',16)['accuracy']:.2f}% | {100*delta['accuracy_difference']:+.2f} |\n"
    trace_table='| Task | Qwen self − Gemini teacher, N=8 | Gemini self − Qwen teacher, N=8 |\n|---|---:|---:|\n'
    families=sorted({c['family'] for c in tracescreen['cells']})
    for family in families:
        cells={c['model']:c for c in tracescreen['cells'] if c['family']==family}
        trace_table+=f"| {family.replace('_',' ')} | {100*contrast(cells['qwen_flash'],'gemini31',8)['accuracy_difference']:+.2f} | {100*contrast(cells['gemini31'],'qwen_flash',8)['accuracy_difference']:+.2f} |\n"
    bbh_table='| BBH task | Initial examples only | Self, N=8 | Published teacher, N=8 | Difference (points) |\n|---|---:|---:|---:|---:|\n'
    for cell in bbhscreen['cells']:
        bbh_table+=f"| {cell['family'].replace('_',' ')} | {100*row(cell,'zero',0)['accuracy']:.2f}% | {100*row(cell,'qwen_flash',8)['accuracy']:.2f}% | {100*row(cell,'oracle',8)['accuracy']:.2f}% | {100*contrast(cell,'oracle',8)['accuracy_difference']:+.2f} |\n"
    diagnostic_table='| Learner / generator | Task | Correct final rehearsal answers | Unparsed generations | Final unparsed query answers across budgets/sources |\n|---|---|---:|---:|---:|\n'
    work_table='| Learner | Task | Source | N | Mean final-inference tokens | Mean sample-generation tokens |\n|---|---|---|---:|---:|---:|\n'
    for cell in tracecells:
        g=cell['generation']
        diagnostic_table+=f"| {NAMES[cell['model']]} | {label(cell)} | {g['correct']}/{g['count']} | {g['unparsed']} | {sum(r['unparsed'] for r in cell['rows'])} |\n"
        for source in (cell['model'],tracerefs[cell['model']]):
            value=row(cell,source,8)
            work_table+=f"| {NAMES[cell['model']]} | {label(cell)} | {NAMES[source]} | 8 | {value['mean_evaluation_tokens']:.0f} | {value['mean_generation_tokens']:.0f} |\n"
    spending=json.loads((DATA/'api/spending.json').read_text())
    positive=[c for c in tracecells if c['model']=='qwen_flash' and contrast(c,'gemini31',8)['accuracy_difference']>0 and contrast(c,'gemini31',8)['holm_p']<.05]
    status='A primary Qwen full-solution contrast survives the selected-task multiplicity correction; its quality, format, and work controls need to determine its interpretation.' if positive else 'Neither selected primary Qwen full-solution contrast establishes a generation advantage after the selected-task multiplicity correction.'
    report=f'''# Strict source-only rehearsal search

The original ranked-curriculum result did not meet the user's intended constraint.
This new search fixes the rehearsal questions literally and varies only the source
of their answers or complete solutions. The answer-only experiment found no
confirmed generation advantage. {status}

No curriculum, question distribution, initial examples, learner checkpoint, final
prompt template, decoding parameters, or final output allowance changes between
paired source arms. All questions and their order are shared. No self-generated
sample is judged for inclusion, repaired, filtered, or rerolled. See the
[prospective protocol and operational amendments](MATCHED_SOURCE_PROTOCOL.md).

## Exact-teacher answer-only comparison

The broad screen covers 16 synthetic task families with two settings each, four
worlds per setting, four initial correct examples, and 24 disjoint test queries.
Qwen3.5 Flash solves each of 16 rehearsal questions independently from the initial
examples, with no test access or previous generated answers. The final model gets
either its own answer strings or the exact answers to the same questions. N=4
and N=16 prefixes are paired with a shared N=0 initial-example baseline.

Six of 32 exploratory differences at N=16 were positive, nine zero, and 17
negative. Three positive cells were near floor. The three positive cells with
usable headroom were frozen for fresh confirmation: Boolean expressions setting 1,
permutation programs setting 2, and shortest paths setting 1. Selection and the
screen's original analysis hash are saved in `selection.json`.

Confirmation uses Qwen on 32 new worlds per cell and Gemini on 16 of those same
worlds, 24 held-out queries per world, and N=0,1,2,4,8,16,32. The primary endpoint
is N=16. Differences are paired within world. Intervals use 6,000 whole-world
bootstrap resamples. Paired two-sided sign-flip tests are Holm-adjusted over the
three selected cells within each learner. Bootstrap intervals are pointwise;
their boundary behavior can differ from the discrete randomization test.

{effect_table(maincells,mainrefs,16)}

![Paired answer-source curves](../reports/matched-source-confirmation.png)

All three exploratory Qwen gains reversed sign in fresh confirmation. No primary
N=16 source advantage survived correction in either learner. This does not prove
equivalence or impossibility. The final confirmation had no missing parsed query
answers and no unparsed generated rehearsal answers. Qwen produced 2,760/3,072
correct rehearsal answers; Gemini produced 1,480/1,536. None were corrected.

The prescribed finite-range error fit is `log(error)=a-beta*log(N)` over
N=4,8,16,32. Negative beta means worsening error, not a useful scaling law.
No pseudocount or fitted error floor is imposed.

{slope_table(maincells,mainrefs,[4,32])}

Gemini's shortest-path accuracy improves with N under both sources, while the
exact teacher improves at least as quickly in this fit. Thus the experiment
includes ordinary learning curves as well as null/floor cases; it does not
support the predicted steeper generated-sample curve.

![Ordinary learning curves on shortest paths](../reports/matched-source-learning.png)

## Full worked solutions from matched sources

Answer-only samples can only differ from an exact teacher when a generated answer
is wrong or malformed. A second screen therefore retains the complete worked
solution from either the learner or the other LLM. Both sources get precisely the
same question, initial examples, generation instructions, and 1,024-token limit.
Both final arms get N full worked solutions under an identical template, with
no source names. Final inference remains answer-only with a 512-token allowance.
This richer representation is provided to both source arms; it is not compared
against a teacher receiving less detailed examples as a source-effect claim.

The exploratory full-solution screen includes every family at setting 1, two
new worlds per family, 12 test queries per world, and N=0,2,8. The primary
direction is Qwen learning from its own versus Gemini's solutions. The reciprocal
Gemini comparison is secondary. Its complete exploratory results are below.

{trace_table}

The two largest primary Qwen differences were hidden affine inference and short
permutation programs, both +8.33 points at N=8. Every rehearsal final answer from
both sources was correct in those two small screen cells; intermediate reasoning
was not judged. Fresh confirmation was frozen before its outcomes on 24 worlds
per task, 12 test queries, both source models, and N=0,1,2,4,8,16. The primary
endpoint remains Qwen at N=8; Holm correction covers the two selected cells.
`trace_selection.json` preserves the selection and original analysis hash.

{effect_table(tracecells,tracerefs,8)}

![Paired full-solution source curves](../reports/matched-source-traces.png)

The secondary Gemini hidden-affine contrast is positive (+11.46 points, Holm
p=0.023), but both learners prefer Gemini's solutions on that task: Qwen's
own-minus-Gemini difference is −6.25 points. Gemini's rehearsal final answers
are correct in 321/384 cases (83.6%), versus Qwen's 198/384 (51.6%). This pattern
is consistent with a stronger solution source; it does not isolate a special
benefit of generating one's own samples. Source quality is a measured consequence,
not an exclusion criterion. Neither task confirms a primary Qwen advantage.

One Qwen hidden-affine self-source evaluation at N=8 returned repetitive gibberish
until its 512-token limit, leaving six query answers missing. Its full request,
response, and truths are saved in spot_checks.json. This is an actual failed
response, not a parser rejecting valid answers. Even counting all six as correct
would leave Qwen's self-minus-Gemini difference negative (−4.17 points). No output
was repaired or rerun because of correctness or format.

The primary finite-range fit uses N=2,4,8,16; the 1–8 window is retained as a
sensitivity analysis in the machine-readable output. There is no asymptotic
exponent claim from such short ranges, especially for curved or flat trajectories.

{slope_table(tracecells,tracerefs,[2,16])}

{diagnostic_table}

These quality diagnostics are recorded after generating and evaluating the entire
unfiltered cohort. They grade final rehearsal answers only, not every reasoning
step. They never choose examples, supply corrections, or alter the curriculum.
The exact-teacher and matched model-teacher answer-only references for these same
worlds remain in `trace_confirmation_inputs`; they are not substituted for a
full-solution teacher in the primary comparison.

## BBH classification tasks

The final screen covers all 23 classification configurations used in Self-ICL,
including language, logic, causal judgment, navigation, and tracking tasks.
Each uses two random splits, four initial examples, 12 test questions, N=0,2,8,
and the same Qwen answer-only treatment. These are supplied questions with either
self-generated or published teacher answers, not a literal reproduction of the
Self-ICL question-generation procedure. See the [original BBH release](https://github.com/suzgunmirac/BIG-Bench-Hard)
and [Self-ICL implementation](https://github.com/ntunlplab/Self-ICL).

An audit before paid calls identified nine records to remove identically from
both source pools: three malformed option targets, four records in two pairs
of conflicting duplicate questions, and two extra copies of consistent duplicates.
The original release contains these issues too. Every raw record is retained;
`bbh_inputs/validation.json` and `duplicates.json` document them. All 5,502 eligible
unique questions remain available. This data-integrity step uses published corpus
records, never generated sample correctness. Published labels are not proof-checked
oracles and may themselves contain other errors or subjective judgments.

The screen has four positive contrasts, 17 ties (within floating-point precision),
and two negative contrasts. Formal fallacies and translation error detection,
both +8.33 points, were selected before any confirmation responses. The selected
tasks use the other half of their datasets for confirmation, with eight random
splits, 12 test queries, and N=0,1,2,4,8,16. The initial plan of 16 splits/task was
reduced before outcomes because its maximum-token charge bound exceeded the
remaining shared budget. Both selected tasks and every budget were retained.

{effect_table(bbh['cells'],bbhrefs,8)}

![BBH confirmation learning curves](../reports/matched-source-bbh.png)

The primary endpoint remains N=8 with Holm adjustment over the two selected tasks.
Whole-split bootstrap intervals are conditional on a fixed finite dataset: split
draws can reuse records within their phase, although no question occurs in both
the screen and confirmation pools. Eight splits give limited precision. The
descriptive error-slope window remains N=2,4,8,16:

{slope_table(bbh['cells'],bbhrefs,[2,16])}

Neither primary BBH endpoint establishes an advantage. Translation detection's
positive relative beta reflects a worsening teacher curve (beta −0.073) against
a nearly flat generated curve (beta 0.007), not a useful improving generated
power law. Its paired beta interval is pointwise and unadjusted for multiple
tasks/windows. Formal fallacies have a favorable 1–8 sensitivity slope but an
unfavorable primary 2–16 slope; the finite-window estimate is not stable.

Full-response review found seven formal-fallacy generations ending in INVALID
without an ANSWER: prefix. The strict parser had inserted UNPARSED in those
slots. A separate post hoc replay accepts an exact final-line task label when
the prefix is absent, using only the response and displayed options. All raw
generations, questions, test sets, and teacher/zero rows are reused. Incorrect
INVALID labels remain incorrect; no correctness feedback is supplied. This
corrects a format-extraction limitation, not a model answer. The screen had no
such missing labels. The original confirmation above remains preserved.

{effect_table(bbhreplay['cells'],bbhrefs,8)}

![BBH label-extraction sensitivity curves](../reports/matched-source-bbh-labels.png)

These replay results use the same data and are not an independent confirmation.
Both N=8 contrasts remain unchanged. Formal-fallacy rehearsal correctness becomes
120/128 rather than 115/128, with no missing labels. Its replay beta difference
is +0.009, interval [−0.005, +0.018], so the parser correction does not establish
a scaling advantage. Translation-error results are identical to the original.
Its separate audit checks every changed label and verifies that all other
sample content, source arms, and inference settings remain fixed.

All 23 screening configurations, including null results and ceiling/floor cases:

{bbh_table}

## Work, scope, and interpretation

Both sources have the same generation instruction and maximum output allowance,
but may naturally produce different wording and length. Those are consequences
of the source, not independently assigned interventions. Actual token work is
recorded, and a source difference alone cannot establish an intrinsic authorship
effect separate from the resulting content. At N=8 in full-solution confirmation:

{work_table}

Generation tokens include both input and output. Final inference tokens include
all test batches for that world. They are API work counters, not hardware FLOPs.
Exact-teacher acquisition cost is omitted in the answer-only reference. All
sources use the same final allowance within a paired comparison; equal N does
not imply equal total tokens or equal compute.

This study uses frozen-model in-context learning. Generations happen in separate
calls, and only their recorded sample content enters final inference. An identical
final prompt has the same prediction distribution regardless of who authored its
text. The request cache deliberately reuses identical requests. Equality for
identical content is a design identity, not independent replication or a learned
memory of the act of generation.

All question strings are disjoint across initial support, rehearsal, and test
within a world. They are sampled once by a common task generator without source-
specific ranking or selection. The study tests generated solutions to fixed
questions; it does not test a learner proposing its own question distribution.

The initial tasks are algorithmic and synthetic, followed by the BBH corpus screen.
Fifteen of 32 initial synthetic settings
had at most 5% answer-only baseline accuracy. Such floor cases limit what their
null contrasts can establish. The complete screen remains reported rather than
being replaced with a favorable subset. These results do not exclude benefits
on other tasks, with other inference settings, or with weight updates.
No SFT, RL, active curriculum, rejection sampling, or correctness feedback was used.

## Full answer-only screen

These are exploratory selection data, not confirmations. All 32 settings remain:

{screen_table}

Individual sample-count curves for every cell are retained under each cohort's
`plots` directory. The secondary full-solution screen and confirmation likewise
retain both learner directions and every budget, including unfavorable outcomes.

## Integrity, cost, and reproduction

Eleven offline tests cover exact task semantics, disjoint sampled questions,
target isolation, retention of wrong samples, fixed answer slots, and identical
full-solution templates, plus BBH option normalization, disjoint phase pools,
and retention of unprefixed labels without correctness-based repair.
The first interface check found that inserting an
unfinished response wholesale could put reasoning into only one answer-only arm.
Before the main screen, that fallback was replaced with an UNPARSED answer in
the original sample slot. Full raw responses remain saved. The original interface
cohort and its source snapshot are preserved; it is not pooled with the screen.
Full-solution conditions retain raw responses from both sources by design.

The final audit checks all planned worlds and conditions; regenerated task data;
frozen source/request hashes; disjoint question partitions; generation blind to
test queries and teacher answers; identical final instructions and limits;
unchanged source samples; exact scoring; and spending reconciliation. Full-solution
audits additionally compare every solution to its archived input-cohort response.
`verification.json` and per-trace-cohort `verification.json` preserve the checks.
BBH cohorts additionally audit the frozen original corpus, label normalization,
question deduplication, and disjoint screen/confirmation pools.

All new phases together record **${spending['paid_usd']}** in settled charges plus
**${spending['unresolved_reserved_usd']}** conservatively reserved for unresolved
attempts: **${spending['committed_usd']}** committed against the persistent $3 cap.
Earlier Self-ICL and inverse-compilation studies have separate, unchanged ledgers.

Qwen uses `qwen/qwen3.5-flash-02-23` through Alibaba at $0.065 input / $0.26 output
per million tokens. Gemini uses `google/gemini-3.1-flash-lite` through Google AI
Studio at $0.25 / $1.50. These were pinned price ceilings, with temperature zero,
top-p one, hidden reasoning disabled, and provider fallbacks disabled. Endpoint
snapshots are retained. Official references: [Qwen pricing](https://openrouter.ai/qwen/qwen3.5-flash-02-23/pricing)
and [Gemini pricing](https://openrouter.ai/google/gemini-3.1-flash-lite).

The environment is the existing uv environment outside NFS, loading `~/.env`
through python-dotenv. Main commands from the project root:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_source.py --phase screen --seed-base 300000
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_source.py --phase confirmation --selection results/matched_source_icl/selection.json --seed-base 600000 --worlds 32 --budgets 0 1 2 4 8 16 32
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_source.py --phase confirmation_gemini --selection results/matched_source_icl/selection.json --models gemini31 --seed-base 600000 --worlds 16 --budgets 0 1 2 4 8 16 32
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_source.py --phase trace_inputs --models qwen_flash gemini31 --levels 1 --worlds 2 --seed-base 900000 --budgets 0 2 8 --tests 12 --cross-teacher
"$VENV_LOCAL/generative-learning/bin/python" experiments/evaluate_matched_traces.py results/matched_source_icl/trace_inputs --phase trace_screen
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_source.py --phase trace_confirmation_inputs --selection results/matched_source_icl/trace_selection.json --models qwen_flash gemini31 --worlds 24 --seed-base 1200000 --budgets 0 1 2 4 8 16 --tests 12 --cross-teacher
"$VENV_LOCAL/generative-learning/bin/python" experiments/evaluate_matched_traces.py results/matched_source_icl/trace_confirmation_inputs --phase trace_confirmation
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_bbh.py --phase bbh_screen
"$VENV_LOCAL/generative-learning/bin/python" experiments/run_matched_bbh.py --phase bbh_confirmation --pool confirmation --selection results/matched_source_icl/bbh_selection.json --seed-base 1800000 --worlds 8 --budgets 0 1 2 4 8 16 --concurrency 8
"$VENV_LOCAL/generative-learning/bin/python" experiments/replay_matched_bbh_labels.py results/matched_source_icl/bbh_confirmation --phase bbh_label_replay
```

Offline analysis uses `analyze_matched_source.py PHASE --plot`; for full-solution
cohorts add `--reference other_model --primary-budget 8`. For their answer-only
input cohorts use `--primary-budget 8`. Analysis and audits make no paid calls.
For BBH use `--primary-budget 8 --teacher-label 'Published teacher labels'`.
Rebuild this report with `build_matched_source_report.py`. Completed inference
cases and successful requests are reused on resume; do not separate the cache
from its spending ledger or overwrite a frozen manifest with changed settings.

Data live at `results/matched_source_icl`, linked to local fast storage. Its
`archive.json` records the verified persistent CEPH copy. The archive includes
all raw requests/responses, the retained interface run, all selection and
confirmation cohorts, code, environment lockfile, audit results, and PNG curves.
'''
    (ROOT/'experiments/MATCHED_SOURCE_RESULTS.md').write_text(report)
    print(ROOT/'experiments/MATCHED_SOURCE_RESULTS.md')


if __name__=='__main__': main()
