"""Compare independent and sequential rehearsal after the complete paired screen."""

from collections import defaultdict
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_task_search import paired_interval
from task_search_api import ROOT


def main():
    root=ROOT.resolve(); independent=root/'palette_screen'; online=root/'palette_online_screen'; manifest=json.loads((online/'inference_manifest.json').read_text()); names=manifest['ready_worlds']
    assert sorted(names)==sorted(p.name for p in (online/'evaluations').glob('*.json')) and (online/'verification.json').exists()
    cells=defaultdict(dict); strategies=defaultdict(list); quality=defaultdict(list)
    for directory,label in [(independent,'independent'),(online,'sequential')]:
        audits={(r['spec'],r['seed']):r for r in json.loads((directory/'strategy_diagnostics.json').read_text())['episodes']}
        for name in names:
            result=json.loads((directory/'evaluations'/name).read_text()); key=(result['spec']['id'],result['seed']); task=result['spec']['task']
            if label=='sequential':
                original=json.loads((independent/'evaluations'/name).read_text()); reference={(c['arm'],c['n']):c for c in original['conditions']}
                for condition in result['conditions']:
                    if condition['arm'] in {'teacher','zero'}: assert condition==reference[(condition['arm'],condition['n'])]
                assert result['generations']['self_fixed'][0]['call']==original['generations']['self_fixed'][0]['call']
            for condition in result['conditions']:
                if condition['status']!='ok': continue
                arm=label if condition['arm']=='self_fixed' else condition['arm']; cells[(task,arm,condition['n'])][key]=condition['accuracy']
            for sample in audits[key]['samples']:
                if sample['arm']=='self_fixed': quality[(label,sample['index']+1)].append(sample['valid']); strategies[(label,sample['index']+1)].append(sample['injective_valid'])
    tasks=sorted({task for task,_,_ in cells}); comparisons=[]; curves=[]
    for n in [2,8,32]:
        for task in tasks:
            seq=cells[(task,'sequential',n)]; iid=cells[(task,'independent',n)]; keys=sorted(seq.keys()&iid.keys())
            delta=[seq[k]-iid[k] for k in keys]; comparisons.append({'task':task,'n':n,'episodes':len(keys),'difference':float(np.mean(delta)),'ci95':paired_interval(delta)})
    # The pooled plot is descriptive across the entire preselected nine-setting screen.
    for arm in ['zero','teacher','independent','sequential']:
        for n in ([0] if arm=='zero' else [2,8,32]):
            values=[value for task in tasks for value in cells[(task,arm,n)].values()]
            curves.append({'arm':arm,'n':n,'episodes':len(values),'accuracy':float(np.mean(values)),'ci95':paired_interval(values)})
    output={'purpose':'Descriptive matched online-versus-independent diagnostic on all nine prespecified palette settings, after the full screen; fresh candidate confirmation is separate.',
            'curves':curves,'comparisons':comparisons,
            'generation_dynamics':[{'mode':label,'step':step,'samples':len(values),'correct_fraction':float(np.mean(values)),'injective_valid_fraction':float(np.mean(strategies[(label,step)]))} for (label,step),values in sorted(quality.items())],
            'verification':'All zero and teacher condition records match exactly between independent and sequential phases; every first generation call is identical.',
            'limitations':'Question distribution, initial examples and evaluation requests are paired. Sequential generation consumes more input tokens. Task-pooled intervals cluster by episode and are descriptive; no broad task-population inference or power law is claimed.'}
    (online/'online_analysis.json').write_text(json.dumps(output,indent=2)); zero=[r for r in curves if r['arm']=='zero']; fig,ax=plt.subplots()
    for arm,label in [('teacher','Teacher answers'),('independent','Independent own answers'),('sequential','Sequential own answers')]:
        points=zero+sorted([r for r in curves if r['arm']==arm],key=lambda r:r['n']); ns=[r['n'] for r in points]
        ax.plot(ns,[r['accuracy'] for r in points],marker='o',label=label); ax.fill_between(ns,[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.15)
    ax.set(xlabel='Additional examples',ylabel='Held-out accuracy',ylim=(0,1),title='Online rehearsal: all nine palette settings'); ax.legend(); fig.tight_layout(); fig.savefig(online/'online_learning_curve.png'); plt.close(fig)
    fig,ax=plt.subplots()
    for label in ['independent','sequential']:
        points=[r for r in output['generation_dynamics'] if r['mode']==label]; ax.plot([r['step'] for r in points],[r['correct_fraction'] for r in points],label=label.capitalize())
    ax.set(xlabel='Rehearsal step',ylabel='Fraction of valid own answers',ylim=(0,1),title='Unjudged rehearsal: correctness measured afterwards'); ax.legend(); fig.tight_layout(); fig.savefig(online/'generation_quality_curve.png'); plt.close(fig)
    print(json.dumps({'curves':curves,'comparisons_at_32':[r for r in comparisons if r['n']==32],'verification':output['verification']},indent=2))


if __name__=='__main__': main()
