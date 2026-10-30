"""After-inference graph-strategy diagnostics, never used to filter experience."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from strategy_task_search import register
from task_search_api import ROOT
from task_search_data import dataset
from task_search_scoring import score_answer


def describe(data,task,text,entry):
    valid=score_answer(data,task,text,entry)>=1-1e-9; puzzle=entry['metadata']['puzzle']; vertices=puzzle['vertices']
    try: coloring=json.loads(text)
    except json.JSONDecodeError: return {'valid':valid,'complete_map':False,'injective':False,'injective_valid':False,'colors_used':None}
    if not isinstance(coloring,dict) or any(str(v) not in coloring for v in vertices): return {'valid':valid,'complete_map':False,'injective':False,'injective_valid':False,'colors_used':None}
    values=[coloring[str(v)] for v in vertices]
    if any(type(value) not in (int,float,str,bool) for value in values): return {'valid':valid,'complete_map':False,'injective':False,'injective_valid':False,'colors_used':None}
    used=len(set(values)); conflicts=sum(coloring[str(a)]==coloring[str(b)] for a,b in puzzle['edges'])
    return {'valid':valid,'complete_map':True,'injective':used==len(vertices),'injective_valid':valid and used==len(vertices),'colors_used':used,'conflicting_edges':conflicts,'extra_keys':sorted(set(coloring)-{str(v) for v in vertices})}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); args=parser.parse_args(); register(); directory=ROOT.resolve()/args.phase
    pre=json.loads((directory/'preflight.json').read_text()); files=sorted((directory/'evaluations').glob('*.json')); assert len(files)==sum(r['status']=='ready' for r in pre),'Complete inference before inspecting generated strategies.'
    records=[]; groups=defaultdict(list)
    for path in tqdm(files,desc='Graph strategy audit',mininterval=10):
        result=json.loads(path.read_text()); world=json.loads((directory/'worlds'/path.name).read_text()); data=dataset(world); task=world['spec']['task']; samples=[]; finals=[]
        for arm,items in [('initial',world['support']),('teacher',world['practice'])]:
            for i,entry in enumerate(items):
                record={'arm':arm,'index':i,**describe(data,task,entry['answer'],entry)}; samples.append(record); groups[(task,'sample',arm,None)].append(record)
        for item in result['generations']['self_fixed']:
            entry=world['practice'][item['index']]; assert entry['question']==item['example']['question']
            record={'arm':'self_fixed','index':item['index'],'status':item['call']['status'],**describe(data,task,item['example']['answer'],entry)}; samples.append(record); groups[(task,'sample','self_fixed',None)].append(record)
        for condition in result['conditions']:
            if condition['status']!='ok': continue
            for prediction,entry in zip(condition['predictions'],world['tests']):
                record={'arm':condition['arm'],'n':condition['n'],'id':entry['id'],**describe(data,task,prediction['answer'],entry)}
                assert record['valid']==prediction['correct']; finals.append(record); groups[(task,'final',condition['arm'],condition['n'])].append(record)
        records.append({'spec':world['spec']['id'],'seed':world['seed'],'samples':samples,'finals':finals})
    summaries=[]
    for (task,kind,arm,n),items in sorted(groups.items(),key=lambda pair:str(pair[0])):
        used=[r['colors_used'] for r in items if r['colors_used'] is not None]
        summaries.append({'task':task,'kind':kind,'arm':arm,'n':n,'count':len(items),'valid_fraction':float(np.mean([r['valid'] for r in items])),
                          'injective_valid_fraction':float(np.mean([r['injective_valid'] for r in items])),'mean_colors_used_when_complete':float(np.mean(used)) if used else None})
    output={'timing':'Computed only after every phase evaluation completed; never used for sample inclusion or feedback.','audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':summaries,'episodes':records}
    (directory/'strategy_diagnostics.json').write_text(json.dumps(output,indent=2)); print(json.dumps({'episodes':len(records),'summary':summaries},indent=2))


if __name__=='__main__': main()
