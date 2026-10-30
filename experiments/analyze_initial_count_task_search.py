"""Audit the complete grid and produce matched heatmaps and learning curves."""
from collections import Counter,defaultdict
import base64
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
from tqdm.auto import tqdm

from analyze_task_search import paired_interval
from decomposition_task_search import solution
from initial_count_task_search import INITIAL,PHASE,REHEARSAL
from task_search_api import ROOT,digest
from task_search_data import seed_for
from task_search_prompts import answer_prompt


def main():
    root=ROOT.resolve(); directory=root/PHASE; project=Path(__file__).resolve().parents[1]
    complete=json.loads((directory/'complete.json').read_text()); assert complete['completed_worlds']==96
    design=json.loads((directory/'design.json').read_text()); manifest=json.loads((directory/'inference_manifest.json').read_text())
    assert hashlib.sha256((directory/'design.json').read_bytes()).hexdigest()==manifest['design_sha256']
    records=json.loads((directory/'preflight.json').read_text()); cache=root/'api'/manifest['engine_fingerprint']
    assert {p.name for p in (directory/'evaluations').glob('*.json')}=={r['world'] for r in records}
    checked=set(); statuses=Counter(); cells=defaultdict(dict); generation_quality=defaultdict(list); source_identical=0

    def verify_call(call,prompt,seed,temperature):
        request={'model':'qwen35-9b','messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':1024,'seed':seed,'chat_template_kwargs':{'enable_thinking':False}}
        assert digest(request)==call['key']
        cached=json.loads((cache/f"{call['key']}.json").read_text()); assert cached['request']==request
        choice=cached['response']['choices'][0]
        assert call['value']['answer']==choice['message']['content'].strip()
        assert call['finish_reason']==choice['finish_reason']
        assert call['status']==('ok' if choice['finish_reason']=='stop' else 'truncated')
        checked.add(call['key']); statuses[call['status']]+=1

    for record in tqdm(records,desc='Audit all initial-context grid requests',mininterval=5):
        world_path=directory/'worlds'/record['world']; assert hashlib.sha256(world_path.read_bytes()).hexdigest()==record['world_sha256']
        world=json.loads(world_path.read_text()); result=json.loads((directory/'evaluations'/record['world']).read_text()); k=record['initial_count']; seed=world['seed']
        assert result['initial_count']==len(world['support'])==k and result['seed']==seed
        original_path=root/'decomposition_confirmation/worlds'/record['source_world']; assert hashlib.sha256(original_path.read_bytes()).hexdigest()==design['original_world_sha256'][record['source_world']]
        original=json.loads(original_path.read_text()); assert world['practice']==original['practice'] and world['tests']==original['tests']
        longest=json.loads((directory/'worlds'/('k16__'+record['source_world'])).read_text()); assert world['support']==longest['support'][:k]
        if k==4:
            old=json.loads((root/'decomposition_confirmation/evaluations'/record['source_world']).read_text())
            assert world==original and result['conditions']==old['conditions'] and result['generations']==old['generations']; source_identical+=1
        assert all(solution(e['answer'],e)[0] for e in world['support']+world['practice']+world['tests'])
        generated=result['generations']['self_fixed']; assert len(generated)==32
        for i,g in enumerate(generated):
            entry=world['practice'][i]; assert g['index']==i and g['example']['question']==entry['question']
            verify_call(g['call'],answer_prompt(world,[],entry),seed_for(world['spec'],f'{seed}|fixed',i),.7)
            assert g['example']['answer']==(g['call']['value']['answer'] if g['call']['status']=='ok' else '')
            valid,identity=solution(g['example']['answer'],entry)
            generation_quality[k].append({'seed':seed,'index':i,'valid':valid,'identity':identity,'status':g['call']['status']})
        expected={('zero',0)}|{(arm,n) for arm in ['teacher','self_fixed'] for n in REHEARSAL if n>0}
        assert {(c['arm'],c['n']) for c in result['conditions']}==expected
        for condition in result['conditions']:
            arm,n=condition['arm'],condition['n']; assert condition['status']=='ok'
            examples=[] if n==0 else world['practice'][:n] if arm=='teacher' else [g['example'] for g in generated[:n]]
            assert len(condition['predictions'])==len(condition['calls'])==16
            correct=[]; identity=[]
            for i,(entry,prediction,call) in enumerate(zip(world['tests'],condition['predictions'],condition['calls'])):
                verify_call(call,answer_prompt(world,examples,entry),seed+i,0.)
                assert prediction['answer']==(call['value']['answer'] if call['status']=='ok' else '')
                valid,is_identity=solution(prediction['answer'],entry); assert valid==prediction['correct'] and prediction['id']==entry['id'] and prediction['api_key']==call['key']
                correct.append(valid); identity.append(is_identity)
            assert np.mean(correct)==condition['accuracy']
            cells[(k,arm,n)][seed]={'accuracy':float(np.mean(correct)),'identity':float(np.mean(identity)),'correct':sum(correct),'tests':len(correct)}
    rows=[]; differences=[]
    for k in INITIAL:
        for n in REHEARSAL:
            for arm in ['teacher','self_fixed']:
                data=cells[(k,'zero',0)] if n==0 else cells[(k,arm,n)]; assert len(data)==16
                values=[r['accuracy'] for _,r in sorted(data.items())]; total=sum(r['tests'] for r in data.values())
                rows.append({'initial_count':k,'rehearsal_count':n,'arm':arm,'episodes':len(data),'queries':total,'correct':sum(r['correct'] for r in data.values()),'accuracy':float(np.mean(values)),'ci95':paired_interval(values),'identity_accuracy':float(np.mean([r['identity'] for r in data.values()]))})
            own=cells[(k,'zero',0)] if n==0 else cells[(k,'self_fixed',n)]; teacher=cells[(k,'zero',0)] if n==0 else cells[(k,'teacher',n)]; baseline=cells[(k,'zero',0)]
            assert own.keys()==teacher.keys()==baseline.keys()
            delta=[own[s]['accuracy']-teacher[s]['accuracy'] for s in sorted(own)]; gain=[own[s]['accuracy']-baseline[s]['accuracy'] for s in sorted(own)]
            differences.append({'initial_count':k,'rehearsal_count':n,'own_minus_teacher':float(np.mean(delta)),'ci95_teacher':paired_interval(delta),'own_minus_initial':float(np.mean(gain)),'ci95_initial':paired_interval(gain)})
    quality=[{'initial_count':k,'samples':len(values),'correct':sum(v['valid'] for v in values),'correct_fraction':float(np.mean([v['valid'] for v in values])),'valid_identity':sum(v['identity'] for v in values),'statuses':dict(Counter(v['status'] for v in values))} for k,values in sorted(generation_quality.items())]
    verification={'worlds':len(records),'unique_requests':len(checked),'call_statuses':dict(statuses),'original_k4_episodes_exactly_reused':source_identical,'matched_questions_and_nested_support':True,'all_raw_requests_and_scores_verified':True}
    report={'design':design,'cells':rows,'paired_differences':differences,'generation_quality':quality,'verification':verification,'uncertainty':'Pointwise 95% intervals from 20,000 bootstrap resamples of the 16 paired episodes; descriptive follow-up, no new multiplicity-corrected significance claim.'}
    (directory/'analysis.json').write_text(json.dumps(report,indent=2)); (directory/'verification.json').write_text(json.dumps(verification,indent=2))
    (directory/'request_keys.json').write_text(json.dumps(sorted(checked),indent=2))
    with (directory/'cells.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    plots=directory/'plots'; plots.mkdir(exist_ok=True); lookup={(r['initial_count'],r['arm'],r['rehearsal_count']):r for r in rows}
    for arm,label,filename in [('teacher','Teacher rehearsal','task-search-teacher-heatmap.png'),('self_fixed','Own rehearsal','task-search-own-heatmap.png')]:
        matrix=np.array([[lookup[(k,arm,n)]['accuracy'] for n in REHEARSAL] for k in INITIAL]); fig,ax=plt.subplots()
        heat=ax.imshow(matrix,origin='lower',aspect='auto',vmin=0,vmax=1,cmap='viridis',interpolation='nearest')
        ax.set(xticks=np.arange(len(REHEARSAL)),xticklabels=REHEARSAL,yticks=np.arange(len(INITIAL)),yticklabels=INITIAL,xlabel='Additional rehearsal examples (N)',ylabel='Initial problem–answer pairs (K)',title=label+' · 256 paired test questions per cell')
        for i,k in enumerate(INITIAL):
            for j,n in enumerate(REHEARSAL): ax.text(j,i,f'{100*matrix[i,j]:.1f}%',ha='center',va='center',color='white' if matrix[i,j]<.5 else 'black')
        colorbar=fig.colorbar(heat,ax=ax); colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1)); colorbar.set_label('Held-out accuracy')
        fig.tight_layout(); fig.savefig(plots/filename); fig.savefig(project/'reports'/filename); plt.close(fig)
    for k in INITIAL:
        fig,ax=plt.subplots()
        for arm,label in [('teacher','Teacher rehearsal'),('self_fixed','Own rehearsal')]:
            selected=[lookup[(k,arm,n)] for n in REHEARSAL]
            ax.plot(REHEARSAL,[r['accuracy'] for r in selected],marker='o',label=label); ax.fill_between(REHEARSAL,[r['ci95'][0] for r in selected],[r['ci95'][1] for r in selected],alpha=.15)
        ax.set(xlabel='Additional rehearsal examples',ylabel='Held-out accuracy',ylim=(0,1),title=f'{k} initial examples · 16 paired episodes'); ax.legend(); fig.tight_layout(); fig.savefig(plots/f'initial_{k}_learning_curve.png'); plt.close(fig)
    section=['<section class="heatmap-section"><h2>Initial context × rehearsal: matched accuracy heatmaps</h2>',
             '<p>Both panels use the same 0–100% color scale. Horizontal axis: additional rehearsal pairs N; vertical axis: initial teacher pairs K. Counts are discrete settings, not a linear axis. Every cell uses the same 256 tests (16 episodes × 16 questions).</p><div class="columns">']
    for name in ['task-search-teacher-heatmap.png','task-search-own-heatmap.png']:
        encoded=base64.b64encode((project/'reports'/name).read_bytes()).decode(); section.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{name}"></figure>')
    section+=['</div><p class="small">Initial contexts are nested prefixes. Rehearsal questions and held-out tests stay fixed at every cell. Own answers are generated separately for each K, seeing only those initial pairs and their assigned question; all saved answers are used unchanged. At N=0 both panels share the same initial-only evaluation. K=0 supplies the task instruction with no initial examples.</p>',
              '<p class="small">Descriptive follow-up on the first 16 original confirmation episodes by numeric seed, fixed before this grid was run. The K=4 row exactly reuses their cached predictions; this subset has 256 tests, whereas the main confirmation above uses all 512. Pointwise paired-episode bootstrap intervals and raw counts are available in the <a href="task-search-heatmap-cells.csv">cell table</a>; this is not a new independent confirmation.</p>',
              '<details><summary>Show learning curves for all six initial-example counts</summary><div class="columns">']
    for k in INITIAL:
        encoded=base64.b64encode((plots/f'initial_{k}_learning_curve.png').read_bytes()).decode(); section.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="Learning curves with {k} initial examples"></figure>')
    section.append('</div></details></section>')
    (project/'reports/initial-count-heatmap-section.html').write_text('\n'.join(section)); (project/'reports/task-search-heatmap-cells.csv').write_bytes((directory/'cells.csv').read_bytes())
    print(json.dumps({'verification':verification,'n32_cells':[r for r in rows if r['rehearsal_count']==32],'generation_quality':quality},indent=2))


if __name__=='__main__': main()
