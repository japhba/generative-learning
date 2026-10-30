"""Complete task tables, episode-paired uncertainty, and sample learning curves."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from task_search_api import ROOT

LABELS={'zero':'Initial examples only','teacher':'Teacher answers','self_fixed':'Own answers, same questions','self_free':'Own questions and answers'}


def paired_interval(values,seed=7169):
    values=np.asarray(values,dtype=float); rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),size=(20000,len(values)))].mean(axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def sign_flip_p(values,seed=1729):
    values=np.asarray(values,dtype=float); count=len(values); observed=float(values.mean())
    if count<=18:
        integers=np.arange(2**count,dtype=np.uint32); signs=2*((integers[:,None]>>np.arange(count))&1).astype(float)-1
        return float(np.mean((signs*values).mean(axis=1)>=observed-1e-12))
    rng=np.random.default_rng(seed); draws=rng.choice([-1.,1.],size=(100000,count))
    return float((1+np.sum((draws*values).mean(axis=1)>=observed-1e-12))/100001)


def holm(values):
    order=np.argsort(values); result=np.ones(len(values)); previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,values[index]*(len(values)-rank))); result[index]=previous
    return result.tolist()


def calibration(directory,output):
    summary=json.loads((directory/'summary.json').read_text()); pre=json.loads((directory/'preflight.json').read_text())
    ready={r['spec']:r for r in pre if r['status']=='ready'}; assert len(summary)==len(ready),'Do not select before calibration finishes.'
    families=defaultdict(list)
    for row in summary: families[row['task']].append(row)
    chosen=[]; decisions=[]
    for task,rows in sorted(families.items()):
        best=min(rows,key=lambda r:(abs(r['conditions'][0]['accuracy']-.5),ready[r['spec']]['max_teacher_prompt_tokens'],r['spec']))
        spec=json.loads((directory/'specs'/f"{best['spec']}.json").read_text()); chosen.append(spec)
        decisions.append({'task':task,'selected':best['spec'],'calibration_accuracy':best['conditions'][0]['accuracy'],'candidates':[{ 'spec':r['spec'],'accuracy':r['conditions'][0]['accuracy']} for r in rows]})
    record={'purpose':'Broad exploratory screen; one difficulty per family; selection uses initial-example baseline only. Floor and ceiling families are retained to keep the search broad.',
            'calibration_phase':directory.name,'calibration_summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),'specifications':chosen,'decisions':decisions}
    output.write_text(json.dumps(record,indent=2))
    fig,ax=plt.subplots(); ax.hist([r['calibration_accuracy'] for r in decisions],bins=np.linspace(0,1,6)); ax.set(xlabel='Baseline accuracy at selected difficulty',ylabel='Task families',title=f'{len(chosen)} task families selected using baseline only'); fig.tight_layout(); fig.savefig(directory/'baseline_calibration.png'); plt.close(fig)
    print(json.dumps({'selected_families':len(chosen),'selected_baseline_counts':{str(a):sum(r['calibration_accuracy']==a for r in decisions) for a in sorted({r['calibration_accuracy'] for r in decisions})}},indent=2))


def analyze(directory,selected_output=None,limit=6,endpoint=None):
    files=sorted((directory/'evaluations').glob('*.json')); results=[json.loads(p.read_text()) for p in files]; families=defaultdict(list)
    for result in results: families[result['spec']['id']].append(result)
    rows=[]; curves=[]
    for spec,worlds in sorted(families.items()):
        cells=defaultdict(list); missing=[]
        for world in worlds:
            for condition in world['conditions']:
                if condition['status']=='ok': cells[(condition['arm'],condition['n'])].append((world['seed'],condition['accuracy']))
                else: missing.append({'seed':world['seed'],'arm':condition['arm'],'n':condition['n'],'status':condition['status']})
        for (arm,n),values in sorted(cells.items()): curves.append({'spec':spec,'task':worlds[0]['spec']['task'],'arm':arm,'n':n,'accuracy':float(np.mean([a for _,a in values])),'episodes':len(values),'ci95':paired_interval([a for _,a in values])})
        ns=sorted({n for arm,n in cells if arm=='teacher'})
        if not ns: continue
        n=max(ns) if endpoint is None else endpoint; assert n in ns
        teachers=dict(cells[('teacher',n)]); zeros=dict(cells[('zero',0)])
        for arm in ['self_fixed','self_free']:
            own=dict(cells[(arm,n)]); seeds=sorted(own.keys()&teachers.keys()&zeros.keys())
            if not seeds: continue
            delta=[own[s]-teachers[s] for s in seeds]; gain=[own[s]-zeros[s] for s in seeds]
            rows.append({'task':worlds[0]['spec']['task'],'spec':spec,'arm':arm,'n':n,'episodes':len(seeds),'teacher':float(np.mean([teachers[s] for s in seeds])),'own':float(np.mean([own[s] for s in seeds])),'zero':float(np.mean([zeros[s] for s in seeds])),
                         'delta_teacher':float(np.mean(delta)),'delta_zero':float(np.mean(gain)),'ci95_teacher':paired_interval(delta),'ci95_zero':paired_interval(gain),
                         'p_teacher':sign_flip_p(delta),'p_zero':sign_flip_p(gain),'episode_delta_teacher':delta,'episode_delta_zero':gain,'missing':missing})
    for arm in ['self_fixed','self_free']:
        arm_rows=[r for r in rows if r['arm']==arm]
        for row,p in zip(arm_rows,holm([max(r['p_teacher'],r['p_zero']) for r in arm_rows])): row['p_joint_holm']=p
    rows.sort(key=lambda r:(r['arm'],-min(r['delta_teacher'],r['delta_zero']),r['spec']))
    (directory/'analysis.json').write_text(json.dumps({'worlds':len(results),'tasks':len(families),'comparisons':rows,'curves':curves},indent=2))
    plot_dir=directory/'plots'; plot_dir.mkdir(exist_ok=True)
    fig,ax=plt.subplots()
    for arm,marker in [('self_fixed','o'),('self_free','x')]:
        arm_rows=[r for r in rows if r['arm']==arm]; ax.scatter([r['delta_teacher']*100 for r in arm_rows],[r['delta_zero']*100 for r in arm_rows],label=LABELS[arm],marker=marker,alpha=.6)
    ax.axhline(0,color='gray'); ax.axvline(0,color='gray'); ax.set(xlabel='Own − teacher accuracy (percentage points)',ylabel='Own − initial-only accuracy (percentage points)',title=f'{len(families)} task families: {directory.name}'); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/'task_search_overview.png'); plt.close(fig)
    # Individual curves stay legible using matplotlib's default figure and element sizes.
    for spec in sorted(families):
        fig,ax=plt.subplots(); selected=[r for r in curves if r['spec']==spec]; zero=[r for r in selected if r['arm']=='zero']; assert len(zero)==1
        for arm in ['teacher','self_fixed','self_free']:
            points=zero+sorted([r for r in selected if r['arm']==arm],key=lambda r:r['n'])
            if len(points)==1: continue
            ax.plot([r['n'] for r in points],[r['accuracy'] for r in points],marker='o',label=LABELS[arm]); ax.fill_between([r['n'] for r in points],[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.15)
        ax.set(xlabel='Additional examples',ylabel='Held-out accuracy',ylim=(0,1),title=spec.replace('_',' ')); ax.legend(); fig.tight_layout(); fig.savefig(plot_dir/(spec+'.png')); plt.close(fig)
    if selected_output is not None:
        # Positive on both contrasts, ranked within the primary fixed-question arm.
        candidates=[r for r in rows if r['arm']=='self_fixed' and r['delta_teacher']>0 and r['delta_zero']>0]
        chosen=candidates[:limit]; spec_map={r['spec']['id']:r['spec'] for r in results}
        selected_output.write_text(json.dumps({'purpose':'Fresh confirmation of exploratory fixed-question candidates; no inference claim from the selecting screen.','source_phase':directory.name,'analysis_sha256':hashlib.sha256((directory/'analysis.json').read_bytes()).hexdigest(),'specifications':[spec_map[r['spec']] for r in chosen],'screen_comparisons':chosen},indent=2))
    print(json.dumps({'worlds':len(results),'tasks':len(families),'ranked_comparisons':rows},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); parser.add_argument('--calibration-selection',type=Path); parser.add_argument('--confirmation-selection',type=Path); parser.add_argument('--limit',type=int,default=6); parser.add_argument('--endpoint',type=int); args=parser.parse_args(); directory=ROOT.resolve()/args.phase
    if args.calibration_selection: calibration(directory,args.calibration_selection)
    else: analyze(directory,args.confirmation_selection,args.limit,args.endpoint)


if __name__=='__main__': main()
