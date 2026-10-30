"""Paired number-format confirmation, notation control, and finite-window slopes."""

from collections import defaultdict
from decimal import Decimal,InvalidOperation
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_task_search import paired_interval,sign_flip_p
from task_search_api import ROOT


def selection_correct(answer,entry):
    try: value=Decimal(answer.strip().replace(',',''))
    except InvalidOperation: return False
    if not value.is_finite(): return False
    candidates=[Decimal(x.replace(',','')) for x in entry['metadata']['formatted_candidates']]
    target={'largest':max,'smallest':min}[entry['metadata']['size']](candidates)
    return abs(value-target)<Decimal('.01') and abs(value-target)==min(abs(value-c) for c in candidates)


def compare(cells,arm,reference,n,reference_n):
    own=cells[(arm,n)]; other=cells[(reference,reference_n)]; assert own.keys()==other.keys()
    differences=[own[s]-other[s] for s in sorted(own)]
    return {'arm':arm,'reference':reference,'n':n,'reference_n':reference_n,'episodes':len(own),
            'difference':float(np.mean(differences)),'ci95':paired_interval(differences),'one_sided_p':sign_flip_p(differences)}


def slopes(cells):
    ns=[2,4,8,16,32]; x=np.log(ns); centered=x-x.mean(); seeds=sorted(cells[('teacher',2)])
    rng=np.random.default_rng(1768); indices=rng.integers(0,len(seeds),size=(20000,len(seeds))); estimates={}; draws={}
    for arm in ['teacher','self_fixed']:
        scores=np.array([[cells[(arm,n)][s] for n in ns] for s in seeds]); errors=1-scores.mean(axis=0)
        assert np.all(errors>0)
        beta=float(-np.log(errors)@centered/(centered@centered)); bootstrap_errors=1-scores[indices].mean(axis=1)
        finite=np.all(bootstrap_errors>0,axis=1); bootstrap=-np.log(bootstrap_errors[finite])@centered/(centered@centered)
        estimates[arm]={'beta':beta,'ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'finite_bootstrap_draws':int(finite.sum())}; draws[arm]=(finite,bootstrap_errors)
    joint=draws['teacher'][0]&draws['self_fixed'][0]; assert joint.any(),'No paired bootstrap draw has positive error at every point in both sources.'
    difference=(-np.log(draws['self_fixed'][1][joint])+np.log(draws['teacher'][1][joint]))@centered/(centered@centered)
    return {'definition':'Descriptive log(error)=intercept-beta*log(N), no error floor or pseudocount; not evidence for an asymptotic power law.',
            'sample_counts':ns,'estimates':estimates,'beta_own_minus_teacher':{'difference':estimates['self_fixed']['beta']-estimates['teacher']['beta'],'ci95':np.quantile(difference,[.025,.975]).tolist(),'joint_finite_bootstrap_draws':int(joint.sum())}}


def main():
    root=ROOT.resolve(); original=root/'number_format_confirmation'; control=root/'number_format_copy_control'; output=root/'number_format_analysis'; output.mkdir(exist_ok=True)
    source=json.loads((original/'preflight.json').read_text()); names=[r['world'] for r in source if r['status']=='ready']; assert len(names)==32
    cells=defaultdict(dict); diagnostic=defaultdict(dict); disagreements=[]; examples=[]
    for name in names:
        world=json.loads((original/'worlds'/name).read_text()); main=json.loads((original/'evaluations'/name).read_text()); copied=json.loads((control/'evaluations'/name).read_text())
        assert main['seed']==copied['seed']==world['seed']
        for condition in main['conditions']+copied['conditions']:
            assert condition['status']=='ok'; key=(condition['arm'],condition['n']); seed=world['seed']; cells[key][seed]=condition['accuracy']; checked=[]
            for prediction,entry in zip(condition['predictions'],world['tests']):
                correct=selection_correct(prediction['answer'],entry); checked.append(correct)
                if correct!=prediction['correct']:
                    disagreements.append({'seed':seed,'arm':condition['arm'],'n':condition['n'],'question':entry['question'],'teacher_answer':entry['answer'],'prediction':prediction,'selection_correct':correct})
            diagnostic[key][seed]=float(np.mean(checked))
        for generated,entry,copied_entry in zip(main['generations']['self_fixed'],world['practice'],copied['examples']):
            answer=generated['example']['answer']; examples.append({'seed':world['seed'],'answer_tokens':generated['call']['usage']['completion_tokens'],
                'own_equals_canonical_teacher':answer==entry['answer'],'own_equals_copy_teacher':answer==copied_entry['answer'],'teacher_strings_differ':entry['answer']!=copied_entry['answer']})
    curves=[]
    for (arm,n),values in sorted(cells.items()):
        accuracy=list(values.values()); curves.append({'arm':arm,'n':n,'episodes':len(values),'accuracy':float(np.mean(accuracy)),'ci95':paired_interval(accuracy),
                                                     'strict_selection_accuracy':float(np.mean(list(diagnostic[(arm,n)].values())))})
    comparisons=[compare(cells,arm,ref,n,0 if ref=='zero' else n) for n in [2,8,32] for arm,ref in [('self_fixed','teacher'),('self_fixed','teacher_copy'),('teacher_copy','teacher'),('self_fixed','zero')]]
    record={'source_manifest_sha256':{p.name:hashlib.sha256((p/'inference_manifest.json').read_bytes()).hexdigest() for p in [original,control]},'curves':curves,'comparisons':comparisons,'slopes':slopes(cells),
            'strict_selection_metric':'In addition to absolute error <.01, prediction must be at least as close to the winning displayed candidate as to any other displayed candidate. Secondary diagnostic only.',
            'strict_selection_disagreements':disagreements,'generated_answer_notation':{key:float(np.mean([r[key] for r in examples])) for key in ['own_equals_canonical_teacher','own_equals_copy_teacher','teacher_strings_differ']}}
    (output/'analysis.json').write_text(json.dumps(record,indent=2))
    labels={'teacher':'Teacher, canonical notation','self_fixed':'Own answers, same questions','teacher_copy':'Teacher, input notation'}; zero=[r for r in curves if r['arm']=='zero']; assert len(zero)==1
    fig,ax=plt.subplots()
    for arm,label in labels.items():
        points=zero+sorted([r for r in curves if r['arm']==arm],key=lambda r:r['n']); ns=[r['n'] for r in points]
        ax.plot(ns,[r['accuracy'] for r in points],marker='o',label=label); ax.fill_between(ns,[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.15)
    ax.set(xlabel='Additional examples',ylabel='Held-out accuracy',ylim=(.7,1),title='Number format: 512 fresh questions'); ax.legend(); fig.tight_layout(); fig.savefig(output/'number_format_learning_curve.png'); plt.close(fig)
    fig,ax=plt.subplots(); window=record['slopes']['sample_counts']
    for arm,label in [('teacher','Teacher'),('self_fixed','Own answers')]:
        points=sorted([r for r in curves if r['arm']==arm and r['n'] in window],key=lambda r:r['n']); ns=np.array([r['n'] for r in points]); errors=1-np.array([r['accuracy'] for r in points]); beta=record['slopes']['estimates'][arm]['beta']
        line,=ax.plot(ns,errors,marker='o',label=f'{label}, beta={beta:.3f}'); ax.fill_between(ns,[1-r['ci95'][1] for r in points],[1-r['ci95'][0] for r in points],alpha=.15)
        intercept=np.mean(np.log(errors)+beta*np.log(ns)); ax.plot(ns,np.exp(intercept)*ns**(-beta),linestyle='--',color=line.get_color())
    ax.set(xscale='log',yscale='log',xlabel='Additional examples',ylabel='Held-out error',title='Number format: finite-window error slopes'); ax.legend(); fig.tight_layout(); fig.savefig(output/'number_format_error_scaling.png'); plt.close(fig)
    print(json.dumps({'curves':curves,'comparisons_at_primary_endpoint':[r for r in comparisons if r['n']==8],'slopes':record['slopes'],'strict_selection_disagreements':len(disagreements),'generated_answer_notation':record['generated_answer_notation']},indent=2))


if __name__=='__main__': main()
