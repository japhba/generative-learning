"""Post-inference factor-size diagnostics, including structurally useful wrong answers."""

import argparse
from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from decomposition_task_search import solution
from task_search_api import ROOT


def features(text,entry):
    valid,identity=solution(text,entry); record={'valid':valid,'valid_identity':identity,'positive_integer_array':False,'contains_one':False,'attempted_identity':False,'small_factor':False,'log10_smallest_factor':None}
    try: values=json.loads(text,parse_float=Fraction)
    except json.JSONDecodeError: return record
    if not isinstance(values,list) or len(values)!=entry['metadata']['arity']: return record
    if any(not (type(v) is int or isinstance(v,Fraction) and v.denominator==1) for v in values) or any(v<1 for v in values): return record
    values=[int(v) for v in values]; record.update(positive_integer_array=True,contains_one=1 in values,attempted_identity=sum(v!=1 for v in values)<=1,small_factor=min(values)<=10,log10_smallest_factor=math.log10(min(values))); return record


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--phases',nargs='+',required=True); args=parser.parse_args(); root=ROOT.resolve()
    for phase in args.phases:
        directory=root/phase; pre=json.loads((directory/'preflight.json').read_text()); names=[r['world'] for r in pre if r['status']=='ready']; assert sorted(names)==sorted(p.name for p in (directory/'evaluations').glob('*.json')) and (directory/'verification.json').exists()
        groups=defaultdict(list); records=[]
        for name in tqdm(names,desc='Factor representation audit: '+phase,mininterval=10):
            world=json.loads((directory/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text())
            if not world['spec']['task'].startswith('product_'): continue
            spec=world['spec']['id']
            for arm,entries in [('initial',world['support']),('teacher',world['practice'])]:
                for i,entry in enumerate(entries):
                    row={'spec':spec,'seed':world['seed'],'kind':'sample','arm':arm,'n':None,'index':i,**features(entry['answer'],entry)}; records.append(row); groups[(spec,'sample',arm,None)].append(row)
            for sample in result['generations']['self_fixed']:
                row={'spec':spec,'seed':world['seed'],'kind':'sample','arm':'self_fixed','n':None,'index':sample['index'],**features(sample['example']['answer'],world['practice'][sample['index']])}; records.append(row); groups[(spec,'sample','self_fixed',None)].append(row)
            for condition in result['conditions']:
                if condition['status']!='ok': continue
                for entry,prediction in zip(world['tests'],condition['predictions']):
                    row={'spec':spec,'seed':world['seed'],'kind':'final','arm':condition['arm'],'n':condition['n'],'index':entry['id'],**features(prediction['answer'],entry)}; assert row['valid']==prediction['correct']; records.append(row); groups[(spec,'final',condition['arm'],condition['n'])].append(row)
        summary=[]
        for (spec,kind,arm,n),rows in sorted(groups.items(),key=lambda pair:str(pair[0])):
            logs=[r['log10_smallest_factor'] for r in rows if r['positive_integer_array']]
            summary.append({'spec':spec,'kind':kind,'arm':arm,'n':n,'count':len(rows),**{key:float(np.mean([r[key] for r in rows])) for key in ['valid','valid_identity','positive_integer_array','contains_one','attempted_identity','small_factor']},'mean_log10_smallest_factor_among_arrays':float(np.mean(logs)) if logs else None})
        record={'phase':phase,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing':'Descriptive audit after complete inference; never used to select or change any experience. Small factor means a positive integer factor at most10, whether or not the supplied factorization is correct.','summary':summary,'records':records}
        (directory/'factor_representation_audit.json').write_text(json.dumps(record,indent=2))
        for spec in sorted({r['spec'] for r in summary}):
            fig,ax=plt.subplots(); zero=[r for r in summary if r['spec']==spec and r['kind']=='final' and r['arm']=='zero']; assert len(zero)==1
            for arm,label in [('teacher','Teacher examples'),('self_fixed','Own examples')]:
                points=zero+sorted([r for r in summary if r['spec']==spec and r['kind']=='final' and r['arm']==arm],key=lambda r:r['n']); ax.plot([r['n'] for r in points],[r['small_factor'] for r in points],marker='o',label=label)
            ax.set(xlabel='Additional examples',ylabel='Final answers with a factor at most 10',ylim=(0,1),title='Factor choice: '+spec.split('_decomposition_')[1].replace('__default','')); ax.legend(); fig.tight_layout(); fig.savefig(directory/'plots'/(spec+'_factor_choice.png')); plt.close(fig)
        print(json.dumps({'phase':phase,'sample_summary':[r for r in summary if r['kind']=='sample'],'final_at32':[r for r in summary if r['kind']=='final' and r['n']==32]},indent=2))


if __name__=='__main__': main()
