"""Report every selected confirmation, with paired uncertainty and learning curves."""

from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_task_search import holm,paired_interval,sign_flip_p
from analyze_number_format_task_search import slopes
from task_search_api import ROOT

NAMES={'graph_color':'Graph coloring','maze':'Maze distance','number_format':'Number format','count_primes':'Prime count','palindrome_partitioning':'Palindrome partitions','path_star':'Graph paths','self_reference':'Self reference','count_bits':'Bit count','coin_flip':'Coin probability','color_cube_rotation':'Cube rotation'}
NAMES['vertex_colors_v6_c6_p40']='Color allocation'
for colors in [3,6,12]:
    for density in [15,40,70]: NAMES[f'graph_palette_v6_c{colors}_p{density}']=f'Online: {colors} colors, {density}% edges'
for operation in ['sum','product']:
    for arity in [2,3]:
        for digits in [3,6,12]: NAMES[f'{operation}_decomposition_k{arity}_d{digits}']=f'{operation.capitalize()} of {arity} integers'


def load_phase(root,phase,audited):
    directory=root/phase; preflight=json.loads((directory/'preflight.json').read_text()); names=[r['world'] for r in preflight if r['status']=='ready']
    assert sorted(names)==sorted(p.name for p in (directory/'evaluations').glob('*.json'))
    raw=[json.loads((directory/'evaluations'/name).read_text()) for name in names]
    if not audited: return raw
    audit={(r['spec']['id'],r['seed']):r for r in json.loads((directory/'constraint_audit.json').read_text())['evaluations']}
    # Only coin probability was prospectively amended. Other stricter syntax checks are sensitivity analyses.
    return [audit[(r['spec']['id'],r['seed'])] if r['spec']['task']=='coin_flip' else r for r in raw]


def aggregate(evaluations,arm,phase,endpoint=8):
    tasks=defaultdict(list); rows=[]; curves=[]
    for evaluation in evaluations: tasks[evaluation['spec']['id']].append(evaluation)
    for spec,episodes in sorted(tasks.items()):
        cells=defaultdict(dict); unavailable=[]; task=episodes[0]['spec']['task']; query_counts={}
        for episode in episodes:
            for condition in episode['conditions']:
                if condition['status']=='ok': cells[(condition['arm'],condition['n'])][episode['seed']]=condition['accuracy']; query_counts[episode['seed']]=len(condition['predictions'])
                else: unavailable.append({'seed':episode['seed'],'arm':condition['arm'],'n':condition['n'],'status':condition['status']})
        own,teacher,zero=[cells[key] for key in [(arm,endpoint),('teacher',endpoint),('zero',0)]]; seeds=sorted(own.keys()&teacher.keys()&zero.keys()); assert len(seeds)>0
        differences=[own[s]-teacher[s] for s in seeds]; gains=[own[s]-zero[s] for s in seeds]
        row={'phase':phase,'task':task,'spec':spec,'arm':arm,'n':endpoint,'episodes':len(seeds),'queries':sum(query_counts[s] for s in seeds),'own':float(np.mean([own[s] for s in seeds])),
             'teacher':float(np.mean([teacher[s] for s in seeds])),'zero':float(np.mean([zero[s] for s in seeds])),'delta_teacher':float(np.mean(differences)),'delta_zero':float(np.mean(gains)),
             'ci95_teacher':paired_interval(differences),'ci95_zero':paired_interval(gains),'p_teacher':sign_flip_p(differences),'p_zero':sign_flip_p(gains),'unavailable':unavailable,
             'episode_differences':dict(zip(seeds,differences)),'episode_gains':dict(zip(seeds,gains))}
        row['p_joint']=max(row['p_teacher'],row['p_zero']); rows.append(row)
        for (source,n),values in sorted(cells.items()):
            points=[values[s] for s in seeds if s in values]
            curves.append({'phase':phase,'task':task,'spec':spec,'arm':source,'n':n,'episodes':len(points),'accuracy':float(np.mean(points)),'ci95':paired_interval(points)})
    return rows,curves


def forest(rows,path,title):
    rows=sorted(rows,key=lambda r:r['task']); fig,ax=plt.subplots(); y=np.arange(len(rows))
    for offset,key,label in [(-.1,'teacher','Own − teacher'),(.1,'zero','Own − initial-only')]:
        values=np.array([r['delta_'+key] for r in rows])*100; intervals=np.array([r['ci95_'+key] for r in rows])*100
        ax.errorbar(values,y+offset,xerr=np.array([values-intervals[:,0],intervals[:,1]-values]),fmt='o',label=label)
    ax.axvline(0,color='gray'); ax.set(yticks=y,yticklabels=[NAMES[r['task']] for r in rows],xlabel='Accuracy difference (percentage points)',title=title); ax.legend(loc='lower right'); fig.tight_layout(); fig.savefig(path); plt.close(fig)


def table(rows):
    lines=['| Task | N | Queries | Initial only | Teacher | Own | Own − teacher, points [95% CI] | Own − initial, points | Joint Holm p |','|---|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in sorted(rows,key=lambda r:r['task']):
        low,high=r['ci95_teacher']; lines.append(f"| {NAMES[r['task']]} | {r['n']} | {r['queries']} | {r['zero']:.1%} | {r['teacher']:.1%} | {r['own']:.1%} | {100*r['delta_teacher']:+.2f} [{100*low:+.2f}, {100*high:+.2f}] | {100*r['delta_zero']:+.2f} | {r['p_joint_holm']:.4f} |")
    return '\n'.join(lines)


def main():
    root=ROOT.resolve(); output=root/'final'; plots=output/'plots'; plots.mkdir(parents=True,exist_ok=True); project=Path(__file__).resolve().parents[1]
    primary,curves=aggregate(load_phase(root,'number_format_confirmation',False),'self_fixed','number_format_confirmation')
    more,points=aggregate(load_phase(root,'primary_confirmation',True),'self_fixed','primary_confirmation'); primary+=more; curves+=points
    secondary,points=aggregate(load_phase(root,'secondary_confirmation',True),'self_free','secondary_confirmation'); curves+=points
    assert len(primary)==len(secondary)==6
    frame_plan=json.loads((root/'frame_confirmation_plan.json').read_text()); online_plan=json.loads((root/'palette_online_confirmation_plan.json').read_text()); decomposition_plan=json.loads((root/'decomposition_confirmation_plan.json').read_text())
    assert (root/'frame_confirmation/verification.json').exists() and (root/'palette_online_confirmation_complete.json').exists()
    extension_phases=[('frame_confirmation',frame_plan)]
    if online_plan['tasks']: extension_phases.append(('palette_online_confirmation',online_plan))
    allocation_phases=list(extension_phases); assert (root/'decomposition_complete.json').exists()
    if decomposition_plan['tasks']: extension_phases.append(('decomposition_confirmation',decomposition_plan))
    for phase,plan in extension_phases:
        rows,points=aggregate(load_phase(root,phase,False),'self_fixed',phase,plan['primary_endpoint']); assert len(rows)==len(plan['tasks']); primary+=rows; curves+=points
    for rows in [primary,secondary]:
        for row,p in zip(rows,holm([r['p_joint'] for r in rows])): row['p_joint_holm']=p
        for contrast in ['teacher','zero']:
            for row,p in zip(rows,holm([r['p_'+contrast] for r in rows])): row['p_'+contrast+'_holm']=p
    for row,p in zip(primary+secondary,holm([r['p_joint'] for r in primary+secondary])): row['p_joint_holm_all_attempts']=p
    positive=lambda row:row['delta_teacher']>0 and row['delta_zero']>0 and row['p_joint_holm']<.05 and not row['unavailable']
    sensitivity=[]
    for phase,arm in [('primary_confirmation','self_fixed'),('secondary_confirmation','self_free')]:
        rows,_=aggregate(json.loads((root/phase/'constraint_audit.json').read_text())['evaluations'],arm,phase)
        sensitivity+=rows
    for phase,plan in allocation_phases:
        rows,_=aggregate(json.loads((root/phase/'constraint_audit.json').read_text())['evaluations'],'self_fixed',phase,plan['primary_endpoint']); sensitivity+=rows
    report={'primary':primary,'secondary':secondary,'curves':curves,'confirmed_primary':[r['task'] for r in primary if positive(r)],'confirmed_secondary':[r['task'] for r in secondary if positive(r)],
            'strict_output_constraint_sensitivity':sensitivity,
            'plan_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['confirmation_plan.json','frame_confirmation_plan.json','palette_online_confirmation_plan.json','decomposition_plan.json','decomposition_confirmation_plan.json']},
            'intervals':f'20,000 paired episode bootstrap draws; pointwise 95% intervals. One-sided episode sign-flip p-values; max of both contrasts for conjunction, then Holm over all {len(primary)} primary attempts and separately all six secondary attempts. Correction across all {len(primary)+len(secondary)} attempts also retained.'}
    frame=json.loads((root/'frame_screen/analysis.json').read_text()); online=json.loads((root/'palette_online_screen/analysis.json').read_text()); sequential=json.loads((root/'palette_online_screen/online_analysis.json').read_text()); strategy=json.loads((root/'palette_injective_control/analysis.json').read_text())
    report['shared_wording_screen']=frame; report['online_screen']=online; report['online_paired_diagnostic']=sequential; report['valid_teacher_strategy_control']=strategy
    decomposition=json.loads((root/'decomposition_screen/analysis.json').read_text()); report['decomposition_screen']=decomposition; report['decomposition_strategies']=json.loads((root/'decomposition_screen/strategy_diagnostics.json').read_text())['summary']
    report['decomposition_factor_diagnostics']={phase:json.loads((root/phase/'factor_representation_audit.json').read_text())['summary'] for phase in ['decomposition_screen']+(['decomposition_confirmation'] if decomposition_plan['tasks'] else [])}
    scaling=[]
    for phase in ['number_format_confirmation']+[p for p,_ in extension_phases]:
        grouped=defaultdict(list)
        for result in load_phase(root,phase,False): grouped[result['spec']['id']].append(result)
        for spec,episodes in grouped.items():
            cells=defaultdict(dict)
            for episode in episodes:
                for condition in episode['conditions']:
                    if condition['status']=='ok': cells[(condition['arm'],condition['n'])][episode['seed']]=condition['accuracy']
            required=[(arm,n) for arm in ['teacher','self_fixed'] for n in [2,4,8,16,32]]
            if any(len(cells[key])!=len(episodes) for key in required):
                scaling.append({'phase':phase,'spec':spec,'status':'unavailable: context-censored slope window'}); continue
            if any(np.mean(list(cells[key].values()))==1 for key in required):
                scaling.append({'phase':phase,'spec':spec,'status':'unavailable: zero empirical error in slope window; no pseudocount applied'}); continue
            estimate={'phase':phase,'spec':spec,'task':episodes[0]['spec']['task'],'status':'ok',**slopes(cells)}; scaling.append(estimate)
            fig,ax=plt.subplots(); ns=np.array(estimate['sample_counts'])
            for arm,label in [('teacher','Teacher'),('self_fixed','Own answers')]:
                errors=np.array([1-np.mean(list(cells[(arm,int(n))].values())) for n in ns]); beta=estimate['estimates'][arm]['beta']
                line,=ax.plot(ns,errors,marker='o',label=f'{label}, beta={beta:.3f}'); intercept=np.mean(np.log(errors)+beta*np.log(ns)); ax.plot(ns,np.exp(intercept)*ns**(-beta),linestyle='--',color=line.get_color())
            ax.set_xscale('log',base=2); ax.set_xticks(ns,labels=[str(n) for n in ns]); ax.set(yscale='log',xlabel='Additional examples',ylabel='Held-out error',title=NAMES[estimate['task']]+': finite-window fit'); ax.legend(); fig.tight_layout(); fig.savefig(plots/(estimate['task']+'_error_scaling.png')); plt.close(fig)
    report['finite_window_scaling']=scaling
    rehearsal_quality=[]
    for phase in ['number_format_confirmation','primary_confirmation']+[p for p,_ in extension_phases]:
        groups=defaultdict(Counter)
        for episode in json.loads((root/phase/'diagnostics.json').read_text())['episodes']:
            for sample in episode['generations']:
                assert sample['arm']=='self_fixed'; groups[episode['spec']]['samples']+=1; groups[episode['spec']]['correct']+=sample['correct']; groups[episode['spec']]['truncated']+=sample['output_status']=='truncated'
        for spec,counts in groups.items(): rehearsal_quality.append({'phase':phase,'spec':spec,**dict(counts),'correct_fraction':counts['correct']/counts['samples']})
    report['primary_rehearsal_quality']=rehearsal_quality
    (output/'analysis.json').write_text(json.dumps(report,indent=2))
    fields=['phase','task','arm','n','episodes','queries','zero','teacher','own','delta_teacher','delta_zero','ci95_teacher','ci95_zero','p_teacher','p_zero','p_teacher_holm','p_zero_holm','p_joint_holm','p_joint_holm_all_attempts']
    with (output/'confirmation_table.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader(); writer.writerows({k:r[k] for k in fields} for r in primary+secondary)
    forest(primary,plots/'primary_confirmation.png','Same rehearsal questions: fresh confirmation')
    forest(secondary,plots/'secondary_confirmation.png','Generated complete examples: fresh confirmation')
    for row in primary+secondary:
        selected=[r for r in curves if r['phase']==row['phase'] and r['spec']==row['spec']]; zero=[r for r in selected if r['arm']=='zero']; assert len(zero)==1
        fig,ax=plt.subplots()
        for arm,label in [('teacher','Teacher examples'),(row['arm'],'Own examples')]:
            points=zero+sorted([r for r in selected if r['arm']==arm],key=lambda r:r['n']); ns=[r['n'] for r in points]
            ax.plot(ns,[r['accuracy'] for r in points],marker='o',label=label); ax.fill_between(ns,[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.15)
        ax.set(xlabel='Additional examples',ylabel='Held-out accuracy',ylim=(0,1),title=NAMES[row['task']]); ax.legend(); fig.tight_layout(); fig.savefig(plots/(row['task']+'_'+row['arm']+'.png')); plt.close(fig)
    for name in ['primary_confirmation','secondary_confirmation']: shutil.copyfile(plots/(name+'.png'),project/'reports'/('task-search-'+name.replace('_','-')+'.png'))
    finding=next(r for r in primary if r['task']=='product_decomposition_k3_d3')
    shutil.copyfile(plots/(finding['task']+'_self_fixed.png'),project/'reports/task-search-product-decomposition.png')
    factors=[r for r in report['decomposition_factor_diagnostics']['decomposition_confirmation'] if r['spec']==finding['spec']]
    factor_final={r['arm']:r for r in factors if r['kind']=='final' and r['n'] in [0,finding['n']]}
    factor_samples={r['arm']:r for r in factors if r['kind']=='sample'}
    identity_share=factor_final['self_fixed']['valid_identity']/finding['own']
    identity_net_share=(factor_final['self_fixed']['valid_identity']-factor_final['zero']['valid_identity'])/finding['delta_zero']
    number=json.loads((root/'number_format_analysis/analysis.json').read_text()); screen=json.loads((root/'screen/analysis.json').read_text()); palette=json.loads((root/'palette_screen/analysis.json').read_text())
    audit=[json.loads((root/p/'constraint_audit.json').read_text()) for p in ['screen','primary_confirmation','secondary_confirmation','palette_screen','frame_screen','palette_online_screen']+[p for p,_ in allocation_phases]]; changes=Counter(r['spec'] for a in audit for r in a['scoring_changes'])
    statuses=Counter(); generated=Counter(); overlaps=duplicates=0
    for phase in ['screen','palette_screen','frame_screen','palette_online_screen','decomposition_screen','number_format_confirmation','primary_confirmation','secondary_confirmation']+[p for p,_ in extension_phases]:
        for episode in json.loads((root/phase/'diagnostics.json').read_text())['episodes']:
            for sample in episode['generations']: generated[sample['output_status']]+=1; overlaps+=sample['matches_test_question']; duplicates+=sample['duplicates_previous_generation']
            for condition in episode['conditions']: statuses.update(condition['output_statuses'])
    low,high=finding['ci95_teacher']; low_zero,high_zero=finding['ci95_zero']
    lines=['# Local Qwen task search','',
           f"Unjudged own answers improve three-factor product decomposition on fresh data. With {finding['n']} additional examples, Qwen3.5-9B scores {finding['own']:.1%}, versus {finding['teacher']:.1%} with teacher answers and {finding['zero']:.1%} with the initial examples alone ({finding['queries']} held-out queries in {finding['episodes']} paired episodes). The gain over teacher is {100*finding['delta_teacher']:.1f} percentage points, 95% episode-bootstrap interval [{100*low:.1f}, {100*high:.1f}]; the gain over initial-only is {100*finding['delta_zero']:.1f} points [{100*low_zero:.1f}, {100*high_zero:.1f}]. The joint test survives Holm correction across all {len(primary)+len(secondary)} attempted confirmations (p={finding['p_joint_holm_all_attempts']:.5f}).",'',
           f"The task asks for exactly three positive integers whose product equals the given target T. The teacher draws three factors independently from 1 through 999 and supplies those valid factors. The learner may return any valid decomposition, including [1, 1, T]. In the own condition, only {factor_samples['self_fixed']['valid']:.1%} of rehearsal answers are correct. Valid identity answers account for {identity_share:.1%} of correct final predictions and {identity_net_share:.1%} of the net gain over initial-only. This descriptive breakdown suggests a shift in solution strategy; it does not establish causal mediation. The questions and their order are identical between source conditions, so greater question coverage cannot explain the difference.",'',
           '![Product decomposition learning curve](../reports/task-search-product-decomposition.png)','',
           f"Across the completed fresh confirmations, {len(report['confirmed_primary'])} fixed-question and {len(report['confirmed_secondary'])} complete-example comparisons pass the prespecified joint tests and correction within each comparison family. A positive effect must exceed both the teacher and initial-only conditions. The finite-range curves do not establish an asymptotic power law.",'',
           'Qwen3.5-9B ran locally in BF16 through vLLM on the RTX 4500 Ada 24 GB. No paid API calls, weight updates, generated-sample filtering, correction, or correctness feedback were used. The earlier matched-source study remains separate.','',
           'The final learner always receives four initial correct examples, the same final questions and native answer instructions, temperature zero, thinking disabled, and a 1,024-token output allowance. Rehearsal generations use temperature 0.7. Independent generation sees the initial examples only; the explicitly identified online experiment additionally sees previous unjudged own answers. Only example content changes between source conditions at final evaluation; source names do not appear in prompts. Equal sample counts do not equal token or compute budgets.','',
           '## Fresh fixed-question confirmations','',table(primary),'','![Primary confirmation](../reports/task-search-primary-confirmation.png)','',
           f"The additional rehearsal questions and their order are literally identical in this comparison. Number format used 32 new episodes of 16 tests and N=0,1,2,4,8,16,32, with primary endpoint N=8. The other five original tasks used 32 new episodes of 16 tests and N=0,2,8. The wording confirmation uses the seven-point curve with endpoint N=8; any online or arithmetic decomposition confirmations use the same seven-point curve with endpoint N=32. All {len(primary)} selected primary hypotheses remain in one multiplicity correction, including failed attempts.",'',
           '## Fresh complete-example confirmations','',table(secondary),'','![Complete-example confirmation](../reports/task-search-secondary-confirmation.png)','',
           'These six tasks each used 32 fresh episodes of eight tests and N=0,2,8. The model generated both questions and answers with a generic same-family request; there were no targeted curriculum hints or test access. This comparison allows a change in question distribution. It cannot isolate a fixed-question advantage or an intrinsic effect of authorship.','',
           '## Number-format scaling and notation control','',
           'The selecting screen showed own 15/16 versus teacher 13/16; the 512-query confirmation reversed that difference. At N=8, own accuracy was 84.6%, canonical teacher 88.5%, initial-only 85.2%, and a valid teacher copying the winning candidate in input notation 85.5%. At N=32, these three rehearsal sources scored 86.1%, 91.0%, and 81.4%, respectively. Changing the valid teacher representation can change the curve substantially.','',
           f"The prespecified finite-window fit log(error)=a−beta log(N), N=2,4,8,16,32, gives teacher beta {number['slopes']['estimates']['teacher']['beta']:.3f}, own beta {number['slopes']['estimates']['self_fixed']['beta']:.3f}, and own-minus-teacher beta {number['slopes']['beta_own_minus_teacher']['difference']:.3f} with 95% episode-bootstrap interval {number['slopes']['beta_own_minus_teacher']['ci95']}. The own curve does not show reliably decreasing error over this window. The stricter nearest-displayed-candidate scoring diagnostic changed zero predictions.",'',
           '![Number-format curve](../results/task_search/number_format_analysis/number_format_learning_curve.png)','',
           '## Broad search and focused strategy test','',
           'Baseline calibration examined 311 context-eligible settings across 99 families from 385 attempted procedural configurations. One setting per family was selected from initial-only accuracy, retaining floor and ceiling tasks. The source screen completed 194 episodes across 98 families: two episodes per setting where preflight succeeded, eight tests, N=0,2,8. Six fixed-question and six complete-example tasks exceeded both controls in this small exploratory screen, and every one was advanced to fresh confirmation. Polynomial multiplication lost both episodes to oracle/scorer disagreement; two other source-screen episodes were excluded before inference for context or oracle failure.','',
           'The focused graph-coloring extension varied the palette and edge density on six-vertex graphs, after a larger baseline-only grid. All nine compact settings received four episodes, eight tests, and N=0,2,8,32. None exceeded both controls at N=32. When six or twelve colors were available, the model spontaneously produced an injective, always-valid coloring in 23/768 rehearsal samples; only 424/768 of those rehearsals were valid overall. The extra strategy variety did not establish a generation benefit.','',
           '| Palette setting | Initial only | Teacher, N=32 | Own, N=32 | Own − teacher, points |','|---|---:|---:|---:|---:|']
    for r in sorted(palette['comparisons'],key=lambda r:r['spec']): lines.append(f"| {r['spec']} | {r['zero']:.1%} | {r['teacher']:.1%} | {r['own']:.1%} | {100*r['delta_teacher']:+.2f} |")
    lines+=['','## Shared task wording','',
            'The same six-entity allocation problem was expressed as meeting slots, storage bins, record codes, radio channels, or vertex colors. Wording was identical between source arms. Four baseline queries selected one geometry per wording without source comparisons; all five wordings then received four new episodes and eight tests, N=0,2,8. These are five wordings of one mathematical task, not five new task families. Underlying graphs are paired across wordings at a fixed geometry.','',
            '| Shared wording | Initial only | Teacher, N=8 | Own, N=8 |','|---|---:|---:|---:|']
    for r in sorted(frame['comparisons'],key=lambda r:r['spec']): lines.append(f"| {r['task']} | {r['zero']:.1%} | {r['teacher']:.1%} | {r['own']:.1%} |")
    lines+=['','Only the vertex-color wording exceeded both controls in the selecting screen. Its fresh 512-query confirmation appears in the primary table regardless of outcome, with all seven sample counts retained.','',
            '## Sequential unjudged rehearsal','',
            'The online screen repeats all 36 original palette episodes and the same assigned questions, but each own rehearsal answer can condition on previous unjudged own answers. No future question, final test, teacher rehearsal answer, correction, or strategy hint enters generation. Initial-only and teacher condition records match exactly between the independent and sequential screens. Every first generation request is identical. This changes generation input-token cost; the final source comparison still uses the same questions and decoding.','',
            '| Palette setting | Initial only | Teacher, N=32 | Independent own, N=32 | Sequential own, N=32 |','|---|---:|---:|---:|---:|']
    independent_by_spec={r['spec']:r for r in palette['comparisons']}
    for r in sorted(online['comparisons'],key=lambda r:r['spec']): lines.append(f"| {r['spec']} | {r['zero']:.1%} | {r['teacher']:.1%} | {independent_by_spec[r['spec']]['own']:.1%} | {r['own']:.1%} |")
    lines+=['',f"The frozen rule selected {len(online_plan['tasks'])} uncensored candidates exceeding both controls at N=32, up to a cap of two. Every selected candidate received 32 new episodes of 16 tests and the seven-point curve; every attempted confirmation remains in the primary table and correction.",'',
            '![Online screen learning curve](../results/task_search/palette_online_screen/online_learning_curve.png)','',
            '## Valid teacher strategy control','',
            'On the same 192 graphs from the six settings with at least as many colors as vertices, replacing the standard valid teacher answers with a canonical distinct-color assignment produced 76.0%, 88.5%, and 92.2% accuracy at N=2,8,32. At N=32 the standard teacher scored 60.4%, independent own answers 56.3%, and initial-only 59.4%. All 576 new evaluation prompts have exactly the same token counts and byte counts as their standard-teacher counterparts; all 768 constructed teacher examples are valid.','',
            'This diagnostic demonstrates sensitivity to the teacher solution representation. It does not demonstrate an advantage of unjudged generation. The exact distinct-color assignment appears in 71.9% of final control answers at N=32, versus 4.2% with own examples and 1.0% with the standard teacher. All graphs use IDs 0 through 5, so the assignment is a universally valid constant answer; transfer to new vertex labels or sizes has not been tested.','',
            '![Valid teacher strategy learning curve](../results/task_search/palette_injective_control/strategy_teacher_learning_curve.png)']
    lines+=['','## Arithmetic decomposition strategy test','',
            'Two additional task operations ask for a JSON array of two or three integers with a specified sum or product. Sum parts may be zero; product factors must be positive and may be one. The task statement gives no solution strategy. A sum teacher samples uniformly from all ordered weak compositions of a random target. A product teacher samples independent uniform positive factors and presents their product as the target. The sampler does not exclude zero parts or factors equal to one. All mathematically valid decompositions receive full credit under exact arithmetic.','',
            'The twelve baseline configurations vary operation, arity, and digit scale. The scale describes target digits for sums and teacher-factor digits for products. Four baseline queries select one magnitude per operation/arity, retaining all four settings even at floor or ceiling. Sums scored 100% in this small baseline calibration; two-factor products had useful headroom, while three-factor products were at floor. All source conditions still receive identical initial examples, assigned questions, final prompts and decoder settings. Own answers are independent, unjudged generations with no strategy hint or test access.','',
            '| Arithmetic setting | Initial only | Teacher, N=32 | Own, N=32 |','|---|---:|---:|---:|']
    for r in sorted(decomposition['comparisons'],key=lambda r:r['spec']): lines.append(f"| {r['spec']} | {r['zero']:.1%} | {r['teacher']:.1%} | {r['own']:.1%} |")
    lines+=['',f"The screen uses four fresh episodes of eight tests per setting and N=0,2,8,32. Its frozen rule selects {len(decomposition_plan['tasks'])} uncensored settings exceeding both controls, capped at two, for fresh 32-episode confirmations of 16 tests each. Every attempted confirmation is included in the primary table and correction, regardless of result. Identity-answer frequencies are measured only after full inference. These are two operations with several settings, not twelve independent families."]
    lines+=['','In the selecting screen, only 10/128 own two-factor answers and 3/128 own three-factor answers were correct. The positive point estimates therefore do not rely on highly accurate rehearsal labels. Most output strings are well-formed integer arrays. The post-hoc factor-size audit measures use of a factor at most 10, including incorrect answers: it is a description of solution representation, not a correctness filter or a causal intervention. In the two-factor screen, identity-answer frequency itself did not increase relative to initial-only; the broader shift toward small factors is a more plausible hypothesis to examine.','',
            'The teacher here supplies random valid decompositions, not a pedagogically optimized policy. The task permits identity solutions, so this experiment concerns selecting an easy valid solution rather than acquiring a general-purpose factoring algorithm. The problem family identifier includes arity and magnitude, and the initial examples demonstrate the teacher distribution. These limits matter when interpreting a generation advantage.']
    lines+=['','The fresh three-factor confirmation gives the following final-answer breakdown. All identity answers counted here are arithmetically valid; they are a subset of total accuracy. The audit occurs after inference and supplies no feedback.','',
            '| Context | Correct answers | Valid identity answers |','|---|---:|---:|']
    for arm,label in [('zero','Initial only'),('teacher','32 teacher examples'),('self_fixed','32 own examples')]:
        row=factor_final[arm]; lines.append(f"| {label} | {round(row['count']*row['valid'])}/{row['count']} ({row['valid']:.1%}) | {round(row['count']*row['valid_identity'])}/{row['count']} ({row['valid_identity']:.1%}) |")
    own_samples=factor_samples['self_fixed']
    lines+=['',f"Of {own_samples['count']} own rehearsal answers, {round(own_samples['count']*own_samples['valid'])} are correct and {round(own_samples['count']*own_samples['valid_identity'])} are valid identity decompositions. All teacher rehearsal answers are correct; none is an identity decomposition. No three-factor rehearsal answer was truncated.",'',
            'The two-factor confirmation exceeds teacher at N=32, but its advantage over initial-only has a confidence interval crossing zero. Its own curve peaks at N=8 and then declines. It fails the prespecified joint confirmation criterion; selecting the peak afterwards would change the test.']
    for r in primary:
        if r['phase']=='decomposition_confirmation' and r['task']!=finding['task']: lines+=['',f"![Decomposition learning curve](../results/task_search/final/plots/{r['task']}_self_fixed.png)"]
    lines+=['','## Finite-window slopes for all longer confirmations','',
            'The common descriptive window is N=2,4,8,16,32, fitting log(error)=intercept−beta log(N) without a pseudocount or error floor. This window was fixed before completion of the new longer confirmations. Intervals resample whole episodes and preserve both sources. These descriptive fits are not asymptotic laws or additional multiplicity-corrected discovery tests. A positive slope difference alone does not establish an absolute accuracy benefit.','',
            '| Task | Teacher beta | Own beta | Own − teacher beta [95% CI] |','|---|---:|---:|---|']
    for r in scaling:
        if r['status']=='ok':
            difference=r['beta_own_minus_teacher']; low,high=difference['ci95']; lines.append(f"| {NAMES[r['task']]} | {r['estimates']['teacher']['beta']:.3f} | {r['estimates']['self_fixed']['beta']:.3f} | {difference['difference']:+.3f} [{low:+.3f}, {high:+.3f}] |")
        else: lines.append(f"| {r['spec']} | — | — | {r['status']} |")
    lines+=['','## Rehearsal answer quality measured afterwards','',
            'These diagnostics are computed after full inference and never determine inclusion, repair, feedback, or reweighting. Every assigned rehearsal slot remains present. Differences in answer quality are descriptive and do not identify the mechanism of a source effect.','',
            '| Fresh confirmation | Generated answers | Correct fraction | Truncated answers |','|---|---:|---:|---:|']
    for r in rehearsal_quality: lines.append(f"| {r['spec']} | {r['samples']} | {r['correct_fraction']:.1%} | {r['truncated']} |")
    lines+=['','## Scoring, controls, and limits','',
            'The independent audit recomputed selected-task teacher solutions using prime enumeration, bit counts, BFS, palindrome enumeration, truth-count enumeration, combinatorial probability, graph constraints, and the cube benchmark convention. Coin-flip native prefix credit can incorrectly equal one; confirmation scoring requires absolute probability error at most 1e-4 against the exact fraction, as specified before fresh inference. Other primary metrics retain the native full-credit criterion. Graph colorings and palindrome partitions also undergo stricter output-constraint sensitivity checks, kept separately in the final JSON. All native outputs and scores are retained.','',
            f"Constraint-audit score changes across screen and confirmations: {dict(changes)}. Generated sample statuses across the reported source phases: {dict(generated)}. Final response statuses: {dict(statuses)}. Generated questions exactly matching a held-out question: {overlaps}; within-pool duplicate generated questions: {duplicates}. None was removed or repaired. Full raw outputs, including truncations and invalid JSON, are retained; incomplete samples occupy explicit empty-answer slots.",'',
            'The cube task does not explicitly specify the rotation axis when moving bottom to top; its grader assumes front/back swap with left/right retained. Its result therefore concerns that benchmark convention. These are answer-only, non-thinking, frozen-weight experiments with one checkpoint. Floor settings and the small exploratory screen limit sensitivity to small effects. New episode seeds do not imply globally unique question strings in finite task spaces. All 512 palindrome-confirmation answers have a valid compact serialization of at most 430 tokens, below the shared 1,024-token allowance.','',
            'These procedural episodes generally sample a known task family rather than introduce a new episode-specific hidden rule. That limits what they test about latent-model learning. The model retains no state beyond the supplied context: replaying the same demonstration text produces the same experimental condition. Any source effect here concerns the generated content and its conditioning, not an independent memory of authorship.','',
            f"Uncertainty clusters by episode; intervals are pointwise, while joint p-values account for testing both required contrasts and all {len(primary)} primary confirmation attempts. The machine-readable table also includes separate per-contrast corrections and Holm correction across all {len(primary)+len(secondary)} primary and secondary hypotheses. The sign-flip tests use within-episode source exchangeability under the null. Per-task learning curves remain in the final plots directory. Three sample counts in most original confirmations are insufficient to identify a power law.",'',
            'Exact request reconstruction verifies identical final instructions and decoding, the intended independent or sequential generation contexts, fixed-question matching, unchanged generated prefixes, source hashes, cached payload hashes, and raw answer retention. Ten fresh prefix-cache-salt replays produced ten exact full-response matches. Seven offline control tests cover sample retention, held-out isolation, alternative valid solutions, malformed scoring, and the probability-prefix defect. A separate offline check verifies that sequential rehearsal sees only previous own answers.','',
            '## Complete exploratory screen','',
            '| Task | Initial only | Teacher, N=8 | Own answers, N=8 | Own complete examples, N=8 |','|---|---:|---:|---:|---:|']
    screen_by_spec=defaultdict(dict)
    for r in screen['comparisons']: screen_by_spec[r['spec']][r['arm']]=r
    for spec,arms in sorted(screen_by_spec.items()):
        fixed,free=arms['self_fixed'],arms['self_free']; lines.append(f"| {spec} | {fixed['zero']:.1%} | {fixed['teacher']:.1%} | {fixed['own']:.1%} | {free['own']:.1%} |")
    lines+=['','## Reproduction and evidence','',
            'The prospective design and amendments are in TASK_SEARCH_PROTOCOL.md. The pinned checkpoint revision is c202236235762e1c871ad0ccb60c8ee5ba337b9a; dependencies and the private runtime setup are in task_search_runtime/README.md and its uv.lock. Re-run finish_task_search.py, finish_frame_confirmation_task_search.py, finish_online_confirmation_task_search.py, and finish_decomposition_task_search.py to resume their cached cohorts. The finalizer audits completed phases and regenerates this report. Raw evidence lives in results/task_search on local NVMe, with a persistent CEPH archive recorded in archive.json. No model weights are included in the evidence archive.','']
    (project/'experiments/TASK_SEARCH_RESULTS.md').write_text('\n'.join(lines))
    print(json.dumps({'primary':[{k:r[k] for k in ['task','zero','teacher','own','delta_teacher','delta_zero','p_joint_holm','p_joint_holm_all_attempts']} for r in primary],
                      'secondary':[{k:r[k] for k in ['task','zero','teacher','own','delta_teacher','delta_zero','p_joint_holm','p_joint_holm_all_attempts']} for r in secondary],
                      'confirmed_primary':report['confirmed_primary'],'confirmed_secondary':report['confirmed_secondary']},indent=2))


if __name__=='__main__': main()
