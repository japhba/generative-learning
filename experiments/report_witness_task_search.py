"""Build the new-task results and an explicitly hypothetical mechanistic comparison."""
import base64
from collections import Counter
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_task_search import holm,paired_interval
from task_search_api import ROOT
from task_search_prompts import answer_prompt
from witness_task_search import valid

LABELS={'bezout':'Bézout coefficients','congruences':'Simultaneous congruences','linear_allocation':'Bounded linear allocation','three_squares':'Three squares','rational_interval':'Rational interval','balanced_sat':'Balanced satisfiability','perfect_matching':'Perfect matching','hamiltonian_path':'Hamiltonian path','topological_order':'Topological order','equal_partition':'Equal-sum groups','supersequence':'Common supersequence','register_program':'Register program'}


def label(spec): return LABELS[spec.split('__')[0].removeprefix('witness_').rsplit('_l',1)[0]]


def encoded(path): return '<img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'" alt="'+escape(path.stem)+'">'


def pp_interval(value,interval): return f'{100*value:+.1f} [{100*interval[0]:+.1f}, {100*interval[1]:+.1f}]'


def answer_html(answer,call,cache):
    if call['status']=='truncated':
        raw=json.loads((cache/(call['key']+'.json')).read_text())['response']['choices'][0]['message']['content']
        return '<em>Output limit; empty answer slot under the protocol.</em><details><summary>Full raw completion</summary><pre>'+escape(raw)+'</pre></details>'
    return '<pre>'+escape(answer)+'</pre>'


def example_section(root,out,phase,spec,prefix):
    paths=sorted((root/phase/'evaluations').glob(spec+'__*.json'),key=lambda p:int(p.stem.rsplit('__',1)[1]));world=json.loads((root/phase/'worlds'/paths[0].name).read_text());result=json.loads(paths[0].read_text());generated=result['generations']['self_fixed'];cache=root/'api'/json.loads((root/phase/'inference_manifest.json').read_text())['engine_fingerprint']
    parts=['<details><summary>Ten actual rehearsals and ten paired evaluations: '+label(spec)+', K='+str(len(world['support']))+'</summary><p>First episode by numeric seed and original item order. These are illustrations, not estimates of outcome frequency. The final comparison uses N = 32; correctness annotations were added after inference.</p><table><thead><tr><th>Rehearsal question</th><th>Teacher answer</th><th>Own answer</th></tr></thead><tbody>']
    for entry,g in zip(world['practice'][:10],generated[:10]): parts.append('<tr><td>'+escape(entry['question'])+'</td><td><pre>'+escape(entry['answer'])+'</pre></td><td>'+answer_html(g['example']['answer'],g['call'],cache)+('correct' if valid(g['example']['answer'],entry) else 'incorrect')+'</td></tr>')
    parts.append('</tbody></table><table><thead><tr><th>Held-out question and one valid answer</th><th>Teacher context</th><th>Own context</th></tr></thead><tbody>');shown=0
    for path in paths:
        w=json.loads((root/phase/'worlds'/path.name).read_text());e=json.loads(path.read_text());ends={c['arm']:c for c in e['conditions'] if c['n']==32}
        for i,entry in enumerate(w['tests']):
            parts.append('<tr><td>'+escape(entry['question'])+'<pre>'+escape(entry['answer'])+'</pre></td>'+''.join('<td>'+answer_html(ends[arm]['predictions'][i]['answer'],ends[arm]['calls'][i],cache)+('correct' if ends[arm]['predictions'][i]['correct'] else 'incorrect')+'</td>' for arm in ['teacher','self_fixed'])+'</tr>');shown+=1
            if shown==10: break
        if shown==10: break
    assert shown==10
    parts.append('</tbody></table>')
    for name,examples,query in [('teacher',world['practice'],world['tests'][0]),('own',[g['example'] for g in generated],world['tests'][0]),('generation',[],world['practice'][0])]: (out/f'{prefix}-{name}-prompt.txt').write_text(answer_prompt(world,examples,query))
    parts.append(f'<p>Full prompts: <a href="{prefix}-generation-prompt.txt">generation</a> · <a href="{prefix}-teacher-prompt.txt">teacher evaluation</a> · <a href="{prefix}-own-prompt.txt">own evaluation</a>.</p></details>')
    return '\n'.join(parts)


def toy_section(out):
    return f'''<h3>A minimal toy: strategy cues in context</h3>
<div class="columns"><figure>{encoded(out/'task-search-strategy-toy.png')}</figure><div>
<p>Keep logits h over solution strategies. Generate each rehearsal style independently from softmax(h₀), then read saved answers with the same update in both conditions:</p>
<pre>h ← h + diag(w) one_hot(answer style)
held-out strategy probabilities = softmax(h)</pre>
<p>A rare useful cue can have a larger fixed weight w than a common difficult one. The teacher supplies correct solutions, but their style can elicit a strategy the limited learner executes poorly. Correctness never enters generation or the update.</p>
<p>The plotted expectations are exact binomial sums with illustrative, unfitted parameters. Equal cue weights remove the advantage; a teacher using the easier strategy matches or exceeds it. Identical replayed content produces identical predictions regardless of authorship.</p>
<p><strong>This demonstrates a possible mechanism, not an identified Qwen circuit or a power-law exponent.</strong></p>
</div></div>
<details><summary>Failure case, probabilistic interpretation, information bounds, and weight learning</summary>
<figure>{encoded(out/'task-search-strategy-toy-trap.png')}</figure>
<p>A third, salient invalid strategy produces early gains followed by decline. These expectations are exact multinomial sums. The dashed curve changes only the strategy’s validity for scoring; it does not simulate the actual instruction intervention, which also changed Qwen’s generations.</p>
<p>The update has an exact interpretation as Bayesian recognition under a cue likelihood. The asymmetry is that generator and reader assume different cue distributions. A coherent posterior-predictive null returns the prior on average and leaves this toy’s mean linear readout unchanged. Posterior averaging alone does not prohibit learning from real external observations.</p>
<p>Internally generated R adds no task information conditional on initial data D and weights W: I(θ;D,R|W) = I(θ;D|W). Additional computation can help a limited reader access existing knowledge. For SFT, sampling from exactly the fitted policy has zero expected score gradient at those parameters. Weight learning therefore needs an explicit asymmetry, such as separate generative and recognition maps in <a href="https://www.cs.utoronto.ca/~hinton/absps/ws.htm">wake–sleep</a>. A teacher supplying the identical joint distribution removes that source advantage.</p>
<p>Finite strategy mixtures do not imply different power-law exponents. A distribution of increasingly slow modes could produce a power law, but its small-rate behavior would need to be demonstrated.</p></details>
<p class="small"><a href="task-search-strategy-toy-notes.txt">Full toy assumptions, equations, controls, and falsification tests</a>.</p>'''


def main():
    from witness_format_sensitivity import analyze
    analyze()
    root=ROOT.resolve();project=Path(__file__).resolve().parents[1];out=project/'reports';complete=json.loads((root/'witness_complete.json').read_text());audit=json.loads((root/'witness_verification.json').read_text());screen=json.loads((root/'witness_screen/analysis.json').read_text());rows=sorted(screen['comparisons'],key=lambda r:r['spec']);assert len(rows)==12
    confirm=json.loads((root/'witness_confirmation/analysis.json').read_text()) if complete['selected_tasks'] else {'comparisons':[],'curves':[]};fresh=confirm['comparisons'];zero_done=json.loads((root/'witness_zero_complete.json').read_text());zero_screen=json.loads((root/'witness_zero_screen/analysis.json').read_text());zero_confirm=json.loads((root/'witness_zero_confirmation/analysis.json').read_text()) if zero_done['selected_tasks'] else {'comparisons':[],'curves':[]};zero_fresh=zero_confirm['comparisons'];combined=fresh+zero_fresh;prior=json.loads((root/'final/analysis.json').read_text())['primary'];joint=[max(r['p_teacher'],r['p_zero']) for r in combined];adjusted=holm(joint);all_adjusted=holm([r['p_joint'] for r in prior]+joint)[len(prior):]
    for r,p,ph,pa in zip(combined,joint,adjusted,all_adjusted): r.update(p_joint=p,p_joint_holm_new=ph,p_joint_holm_all_primary=pa,p_joint_bonferroni_33=min(1.,33*p))
    slopes=[]
    for row in fresh:
        phase=root/'witness_confirmation';files=sorted((phase/'evaluations').glob(row['spec']+'__*.json'));worlds=[json.loads(p.read_text()) for p in files];ns=[2,4,8,16,32];values={}
        for arm in ['teacher','self_fixed']:
            matrix=np.array([[next(c['accuracy'] for c in w['conditions'] if c['arm']==arm and c['n']==n) for n in ns] for w in worlds]);error=1-matrix.mean(axis=0)
            if np.all(error>0):
                beta=-float(np.polyfit(np.log(ns),np.log(error),1)[0]);values[arm]=beta
            else: values[arm]=None
        slopes.append({'spec':row['spec'],'window':ns,'beta':values,'interpretation':'Descriptive finite-window slopes only; no asymptotic claim or fitted floor.'})
    directory=root/'witness_analysis';directory.mkdir(exist_ok=True);summary={'screen':rows,'confirmation':fresh,'finite_window_slopes':slopes,'audit':audit,'confirmed_primary':[r['spec'] for r in fresh if r['p_joint_holm_all_primary']<.05 and r['delta_teacher']>0 and r['delta_zero']>0],'prior_primary_attempts':len(prior),'conservative_screen_budget':33,'zero_screen':zero_screen['comparisons'],'zero_confirmation':zero_fresh,'zero_curves':zero_confirm['curves'],'confirmed_zero_primary':[r['spec'] for r in zero_fresh if r['p_joint_holm_all_primary']<.05 and r['delta_teacher']>0 and r['delta_zero']>0]};(directory/'analysis.json').write_text(json.dumps(summary,indent=2))
    fig,ax=plt.subplots();ys=np.arange(len(rows))
    for field,name in [('zero','Initial only'),('teacher','Teacher N=32'),('own','Own N=32')]: ax.plot([r[field] for r in rows],ys,'o',label=name)
    ax.set(yticks=ys,yticklabels=[label(r['spec']) for r in rows],xlabel='Held-out accuracy',xlim=(0,1),title='Exploratory screen · 32 paired tests per task');ax.invert_yaxis();ax.legend();fig.tight_layout();path=out/'task-search-witness-screen.png';fig.savefig(path);plt.close(fig)
    confirmed=summary['confirmed_primary']+summary['confirmed_zero_primary'];headline=('New confirmed advantages: '+', '.join(label(s) for s in confirmed)+'.') if confirmed else 'No additional task confirmed an own-rehearsal advantage in this search.'
    parts=['<section class="heatmap-section" id="witness"><h2>New tasks: twelve constructive families</h2><p class="lead"><strong>'+headline+'</strong> We tested both zero and four initial examples. The toy below demonstrates a possible mechanism, with explicit assumptions; it is not a fit to these task outcomes.</p><table><thead><tr><th>Stage</th><th class="teacher">Teacher rehearsal</th><th class="own">Own rehearsal</th></tr></thead><tbody><tr><td>Initial context</td><td colspan="2">Same K = 0 or K = 4 exact problem–answer pairs</td></tr><tr><td>Additional questions</td><td colspan="2">Identical questions, order, and counts within each comparison</td></tr><tr><td>Additional answers</td><td>Valid solutions from construction or an exact algorithm</td><td>Independent offline Qwen generations, retained without correctness feedback</td></tr><tr><td>Final evaluation</td><td colspan="2">Fresh calls to the same frozen Qwen3.5-9B, same held-out questions, instructions, and decoder</td></tr></tbody></table><p>Only the additional answer strings change. Generation sees neither held-out tests nor previous own answers. Input coverage is therefore identical between sources. Counts are matched; token and total compute budgets are not.</p><h3>Four initial examples: exploratory screen</h3><p>Eight baseline queries selected one of two difficulties per family before source comparisons. Each selected setting then used four episodes × eight tests, four initial pairs, and N = 0, 2, 8, 32. Every task requires an input-dependent certificate; the factorization identity is unavailable.</p>', '<figure>'+encoded(path)+'</figure>', '<details><summary>All twelve exploratory results and their learning curves</summary><table><thead><tr><th>Task</th><th>Initial only</th><th>Teacher N=32</th><th>Own N=32</th><th>Paired tests</th></tr></thead><tbody>']
    parts.extend(f'<tr><td>{label(r["spec"])}</td><td>{r["zero"]:.1%}</td><td>{r["teacher"]:.1%}</td><td>{r["own"]:.1%}</td><td>{8*r["episodes"]}</td></tr>' for r in rows);parts.append('</tbody></table><div class="columns">')
    for r in rows:
        fig,ax=plt.subplots()
        for arm,name in [('teacher','Teacher'),('self_fixed','Own')]:
            curve=sorted([c for c in screen['curves'] if c['spec']==r['spec'] and c['arm']==arm],key=lambda c:c['n']);baseline=next(c for c in screen['curves'] if c['spec']==r['spec'] and c['arm']=='zero');curve=[baseline]+curve
            ax.plot([c['n'] for c in curve],[c['accuracy'] for c in curve],marker='o',label=name)
        ax.set(xlabel='Additional examples N',ylabel='Accuracy',ylim=(0,1),title=label(r['spec'])+' · exploratory');ax.legend();fig.tight_layout();p=directory/(r['spec']+'_curve.png');fig.savefig(p);plt.close(fig);parts.append('<figure>'+encoded(p)+'</figure>')
    parts.append('</div></details>')
    parts.append('<h3>Fresh confirmation</h3>')
    if fresh:
        parts.append('<p>The two strongest eligible screen candidates (or fewer if unavailable) were selected by their smaller gain over teacher and initial-only at N = 32. Each selected task used 16 newly seeded episodes × 16 tests, N = 0, 1, 2, 4, 8, 16, 32, and a fixed primary endpoint N = 32. These are new episodes, not a claim that every question is globally novel across the finite procedural distributions. Intervals resample paired episodes. The joint test requires improvement over both controls.</p><table><thead><tr><th>Task</th><th>Initial / Teacher / Own</th><th>Own − teacher, points [95% CI]</th><th>Own − initial, points [95% CI]</th><th>Joint Holm p, including prior local primary attempts</th></tr></thead><tbody>')
        for r in fresh: parts.append(f'<tr><td>{label(r["spec"])}</td><td>{r["zero"]:.1%} / {r["teacher"]:.1%} / {r["own"]:.1%}</td><td>{pp_interval(r["delta_teacher"],r["ci95_teacher"])}</td><td>{pp_interval(r["delta_zero"],r["ci95_zero"])}</td><td>{r["p_joint_holm_all_primary"]:.4g}</td></tr>')
        parts.append('</tbody></table><div class="columns">')
        for r in fresh:
            fig,ax=plt.subplots()
            for arm,name in [('teacher','Teacher'),('self_fixed','Own')]:
                curve=sorted([c for c in confirm['curves'] if c['spec']==r['spec'] and c['arm']==arm],key=lambda c:c['n']);baseline=next(c for c in confirm['curves'] if c['spec']==r['spec'] and c['arm']=='zero');curve=[baseline]+curve
                ax.plot([c['n'] for c in curve],[c['accuracy'] for c in curve],marker='o',label=name);ax.fill_between([c['n'] for c in curve],[c['ci95'][0] for c in curve],[c['ci95'][1] for c in curve],alpha=.15)
            ax.set(xscale='symlog',xlabel='Additional examples N',ylabel='Accuracy',ylim=(0,1),title=label(r['spec'])+' · fresh confirmation');ax.legend();fig.tight_layout();p=out/('task-search-witness-'+r['spec']+'.png');fig.savefig(p);plt.close(fig);parts.append('<figure>'+encoded(p)+'</figure>')
        parts.append('</div>')
        result_text='; '.join(label(s) for s in summary['confirmed_primary']) or 'none'
        parts.append('<p><strong>Confirmed source advantages under the prespecified joint test:</strong> '+result_text+'. All attempts, including failures, remain in the analysis. A favorable source gap by itself is insufficient; the own condition must also beat the initial-only baseline. These curves do not establish an asymptotic power law.</p>')
    else: parts.append('<p>No exploratory candidate exceeded both controls at the fixed N = 32 endpoint. The prespecified rule therefore selected no fresh confirmations. This search found no additional confirmed generation advantage.</p>')
    total=Counter()
    for a in audit: total.update(a['counts'])
    parts.append(f'<p class="small">Audit: every available world was regenerated and every saved request and score checked. {total["generation_truncated"]} generated answers reached the 1,024-token limit and retained empty answer slots; {total["evaluation_truncated"]} final calls were truncated and scored incorrect. {total["censored_cells"]} cells exceeded context capacity. Invalid outputs were retained in the experiment.</p>')
    chosen=fresh[0] if fresh else rows[0];phase='witness_confirmation' if fresh else 'witness_screen';parts.append(example_section(root,out,phase,chosen['spec'],'task-search-witness'))
    from report_witness_zero import zero_section
    parts.append(zero_section(root,out,summary,label,encoded))
    for r in zero_fresh: parts.append(example_section(root,out,'witness_zero_confirmation',r['spec'],'task-search-witness-zero'))
    sensitivity=json.loads((root/'witness_format_sensitivity/analysis.json').read_text());summary['posthoc_format_sensitivity']=sensitivity
    parts.append('<details><summary>Post-hoc answer-format sensitivity on every fresh confirmation</summary><p>Manual review found a correct path after unwanted explanation in one complete response. As a separate diagnostic, we extract the last syntactically complete JSON array from each completed response, using the same rule for both sources and the initial-only baseline. Extraction never tests correctness to choose an array. Truncated outputs remain incorrect. This was not the primary scorer or a new candidate-selection rule; model contexts are unchanged.</p><table><thead><tr><th>Task / initial pairs</th><th>Initial / Teacher / Own</th><th>Own − teacher, points [95% CI]</th><th>Own − initial, points [95% CI]</th></tr></thead><tbody>')
    for r in sensitivity: parts.append(f'<tr><td>{label(r["spec"])} / K={r["initial_examples"]}</td><td>{r["initial"]:.1%} / {r["teacher"]:.1%} / {r["own"]:.1%}</td><td>{pp_interval(r["delta_teacher"],r["ci95_teacher"])}</td><td>{pp_interval(r["delta_initial"],r["ci95_initial"])}</td></tr>')
    parts.append('</tbody></table></details>')
    parts.append('</details>')
    parts.append('<h3>Relation to Self-ICL</h3><p><a href="https://aclanthology.org/2023.emnlp-main.968/">Chen et al., Self-ICL (EMNLP 2023)</a> generates pseudo-inputs using the test question, labels them with the same frozen model, and inserts the pairs into the final prompt. Its main direct result is 50.81% → 53.93% averaged over 23 BBH tasks. It chiefly tests improvement over zero-shot prompting. The paper uses diversity instructions for generated inputs; its shot-count comparison is 0, 1, and 3, rather than a broad power-law fit. Random-label demonstrations perform worse than model-labeled ones but still improve over zero-shot in its aggregate analysis. Our stricter contrast fixes rehearsal inputs, tests zero and four initial pairs, withholds test inputs from generation, and compares against an exact teacher as well as initial-only. Therefore this is a related source-control experiment, not a literal reproduction or contradiction of the paper.</p>')
    parts.append(toy_section(out));parts.append('</section>')
    section='\n'.join(parts);overview='<h3>Fresh confirmations: 256 held-out questions each</h3><table><thead><tr><th>Task / initial pairs</th><th>Initial only</th><th>Teacher N=32</th><th>Own N=32</th></tr></thead><tbody>'
    for initial,attempts in [(4,fresh),(0,zero_fresh)]:
        for r in attempts: overview+=f'<tr><td>{label(r["spec"])} / K={initial}</td><td>{r["zero"]:.1%}</td><td>{r["teacher"]:.1%}</td><td>{r["own"]:.1%}</td></tr>'
    overview+='</tbody></table><p class="small">Candidates were selected in separate small screens, then tested on newly sampled episodes. The primary endpoint was fixed at N = 32; confirmation requires improvement over both controls with correction across attempts. Curves and paired uncertainty are below.</p><details><summary>Full task search: screens, learning curves, prompts, and 10+10 examples per candidate</summary>'
    section=section.replace('<h3>Four initial examples: exploratory screen</h3>',overview+'<h3>Four initial examples: exploratory screen</h3>',1)
    (directory/'analysis.json').write_text(json.dumps(summary,indent=2));(out/'task-search-strategy-toy-notes.txt').write_text((project/'experiments/STRATEGY_CONTEXT_TOY.md').read_text());(out/'witness-search-section.html').write_text(section);(out/'task-search-witness-results.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
