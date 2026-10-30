"""Paired learning curves and raw-request checks for the valid strategy teacher."""

from collections import defaultdict
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from analyze_task_search import paired_interval,sign_flip_p
from audit_task_search_constraints import answer_valid
from palette_teacher_control import injective_answer
from task_search_api import ROOT,digest
from task_search_prompts import answer_prompt


def main():
    root=ROOT.resolve(); directory=root/'palette_injective_control'; manifest=json.loads((directory/'inference_manifest.json').read_text()); original=root/manifest['source_phase']; api=root/'api'/manifest['engine_fingerprint']
    assert sorted(manifest['ready_worlds'])==sorted(p.name for p in (directory/'evaluations').glob('*.json'))
    assert hashlib.sha256((original/'inference_manifest.json').read_bytes()).hexdigest()==manifest['source_inference_manifest_sha256']
    for name,expected in manifest['source_sha256'].items(): assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
    cells=defaultdict(dict); by_task=defaultdict(lambda:defaultdict(dict)); strategies=defaultdict(list); requests=0; examples_checked=0; equal_prompt_tokens=0; equal_prompt_bytes=0
    for name in manifest['ready_worlds']:
        control=json.loads((directory/'evaluations'/name).read_text()); baseline=json.loads((original/'evaluations'/name).read_text()); world=json.loads((original/'worlds'/name).read_text()); key=(world['spec']['id'],world['seed'])
        expected=[{**entry,'answer':injective_answer(entry)} for entry in world['practice']]; assert control['examples']==expected and control['seed']==world['seed']
        teacher_conditions={c['n']:c for c in baseline['conditions'] if c['arm']=='teacher'}
        for example in expected: assert answer_valid('graph_color',example['answer'],example); examples_checked+=1
        for condition in baseline['conditions']+control['conditions']:
            assert condition['status']=='ok'; arm,n=condition['arm'],condition['n']; cells[(arm,n)][key]=condition['accuracy']; by_task[key[0]][(arm,n)][key]=condition['accuracy']
            for i,(query,call,prediction) in enumerate(zip(world['tests'],condition['calls'],condition['predictions'])):
                if arm=='teacher_injective':
                    request={'model':'qwen35-9b','messages':[{'role':'user','content':answer_prompt(world,expected[:n],query)}],'temperature':0.,'top_p':1.,'max_tokens':1024,'seed':world['seed']+i,'chat_template_kwargs':{'enable_thinking':False}}
                    assert digest(request)==call['key']; raw=json.loads((api/(call['key']+'.json')).read_text()); assert raw['request']==request
                    assert prediction['answer']==(raw['response']['choices'][0]['message']['content'].strip() if call['status']=='ok' else ''); requests+=1
                    equal_prompt_tokens+=call['usage']['prompt_tokens']==teacher_conditions[n]['calls'][i]['usage']['prompt_tokens']
                    equal_prompt_bytes+=len(request['messages'][0]['content'].encode())==len(answer_prompt(world,world['practice'][:n],query).encode())
                valid=answer_valid('graph_color',prediction['answer'],query)
                mapping=json.loads(prediction['answer']) if valid else None
                strategies[(arm,n)].append({'valid':valid,'injective':valid and len(set(mapping.values()))==len(mapping),'exact_universal_assignment':valid and mapping==json.loads(injective_answer(query))})
    def curves_for(mapping):
        return [{'arm':arm,'n':n,'episodes':len(values),'accuracy':float(np.mean(list(values.values()))),'ci95':paired_interval(list(values.values()))} for (arm,n),values in sorted(mapping.items())]
    curves=curves_for(cells); comparisons=[]
    for n in [2,8,32]:
        own=cells[('teacher_injective',n)]
        for reference in ['teacher','self_fixed','zero']:
            other=cells[(reference,0 if reference=='zero' else n)]; assert own.keys()==other.keys(); differences=[own[key]-other[key] for key in sorted(own)]
            comparisons.append({'n':n,'reference':reference,'difference':float(np.mean(differences)),'ci95':paired_interval(differences),'p_one_sided':sign_flip_p(differences)})
    strategy_rows=[{'arm':arm,'n':n,'predictions':len(items),**{metric:float(np.mean([r[metric] for r in items])) for metric in ['valid','injective','exact_universal_assignment']}} for (arm,n),items in sorted(strategies.items())]
    record={'purpose':'Exploratory strategy-teaching diagnostic on the original matched graphs; this is not a generated-source advantage.','curves':curves,'comparisons':comparisons,'strategies':strategy_rows,'per_task':{task:curves_for(mapping) for task,mapping in by_task.items()},
            'verification':{'episodes':len(manifest['ready_worlds']),'constructed_valid_teacher_examples':examples_checked,'exact_requests_checked':requests,'same_actual_prompt_tokens_as_standard_teacher':equal_prompt_tokens,'same_prompt_bytes_as_standard_teacher':equal_prompt_bytes},
            'limitation':'All graphs use the same six vertex IDs. The canonical injective assignment is a universal fixed answer; this control does not establish transfer to new vertex names or sizes.'}
    (directory/'analysis.json').write_text(json.dumps(record,indent=2)); labels={'teacher':'Standard valid teacher','self_fixed':'Own answers, same graphs','teacher_injective':'Valid distinct-color teacher'}
    fig,ax=plt.subplots(); zero=[r for r in curves if r['arm']=='zero']; assert len(zero)==1
    for arm,label in labels.items():
        points=zero+sorted([r for r in curves if r['arm']==arm],key=lambda r:r['n']); ns=[r['n'] for r in points]
        ax.plot(ns,[r['accuracy'] for r in points],marker='o',label=label); ax.fill_between(ns,[r['ci95'][0] for r in points],[r['ci95'][1] for r in points],alpha=.15)
    ax.set(xlabel='Additional examples',ylabel='Held-out accuracy',ylim=(0,1),title='Same 192 graphs: solution representation'); ax.legend(); fig.tight_layout(); fig.savefig(directory/'strategy_teacher_learning_curve.png'); plt.close(fig)
    print(json.dumps({'curves':curves,'comparisons_at_32':[r for r in comparisons if r['n']==32],'strategies_at_32':[r for r in strategy_rows if r['n']==32],'verification':record['verification']},indent=2))


if __name__=='__main__': main()
