"""Source-matched arithmetic decompositions with multiple valid solution strategies."""

import argparse
from collections import defaultdict
from dataclasses import dataclass,make_dataclass
from fractions import Fraction
import hashlib
import json
from math import prod
from pathlib import Path
from random import Random
import sys

import numpy as np
from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import register_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_api import ROOT
from task_search_data import make_world,seed_for,specification
from task_search_prompts import answer_prompt


@dataclass
class DecompositionConfig:
    operation:str='sum'; arity:int=2; digits:int=3; seed:int|None=None; size:int=1000
    def validate(self): assert self.operation in {'sum','product'} and self.arity in {2,3} and self.digits in {3,6,12}


def solution(text,entry):
    try: values=json.loads(text,parse_float=Fraction)
    except json.JSONDecodeError: return False,False
    meta=entry['metadata']
    if not isinstance(values,list) or len(values)!=meta['arity']: return False,False
    if any(not (type(v) is int or isinstance(v,Fraction) and v.denominator==1) for v in values): return False,False
    minimum=0 if meta['operation']=='sum' else 1
    valid=all(v>=minimum for v in values) and {'sum':sum,'product':prod}[meta['operation']](values)==meta['target']
    identity=valid and sum(v!=minimum for v in values)<=1
    return bool(valid),bool(identity)


class DecompositionDataset(ProceduralDataset):
    def __init__(self,config): super().__init__(config=config,seed=config.seed,size=config.size)
    def __getitem__(self,index):
        rng=Random(self.seed+index); config=self.config
        if config.operation=='sum':
            target=rng.randint(10**(config.digits-1),10**config.digits-1); cuts=[-1,*sorted(rng.sample(range(target+config.arity-1),config.arity-1)),target+config.arity-1]
            values=[b-a-1 for a,b in zip(cuts,cuts[1:])]; domain='nonnegative'; policy='Uniform over all ordered weak compositions of the target; zero parts are allowed.'
        else:
            values=[rng.randint(1,10**config.digits-1) for _ in range(config.arity)]; target=prod(values); domain='positive'; policy='Independent uniform positive factors generate the target and a valid answer; factors equal to one are allowed.'
        question=f'Return a JSON array of exactly {config.arity} {domain} integers whose {config.operation} is {target}. The order of the integers does not matter.'
        return {'question':question,'answer':json.dumps(values),'metadata':{'operation':config.operation,'arity':config.arity,'target':target,'teacher_policy':policy}}
    def score_answer(self,answer,entry): return float(solution(answer,entry)[0])


def register():
    names=[]
    for operation in ['sum','product']:
        for arity in [2,3]:
            for digits in [3,6,12]:
                name=f'{operation}_decomposition_k{arity}_d{digits}'; config=make_dataclass(name+'Config',[('operation',str,operation),('arity',int,arity),('digits',int,digits)],bases=(DecompositionConfig,))
                register_dataset(name,DecompositionDataset,config); names.append(name)
    return names


def prepare(names):
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True); parser.add_argument('--selection',type=Path); parser.add_argument('--worlds',type=int,default=1); parser.add_argument('--practice',type=int,default=32); parser.add_argument('--tests',type=int,default=4); args=parser.parse_args()
    root=ROOT.resolve(); directory=root/args.phase; (directory/'worlds').mkdir(parents=True,exist_ok=True); (directory/'specs').mkdir(exist_ok=True)
    specs=[specification(name,'default') for name in names] if args.selection is None else json.loads(args.selection.read_text())['specifications']
    manifest={'phase':args.phase,'worlds_per_spec':args.worlds,'practice_count':args.practice,'test_count':args.tests,'support_count':4,'specifications':specs,'max_teacher_prompt_tokens':14000,
              'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['decomposition_task_search.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']},
              'design':'Identical sum/product questions and all final settings. The procedural teacher supplies random valid decompositions. Own answers are unjudged independent generations from the initial four examples. The problem statement gives no solution strategy.'}
    path=directory/'data_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(json.loads((root/'runtime/model.json').read_text())['path']); records=[]
    for spec in tqdm(specs,desc='Prepare decomposition tasks',mininterval=5):
        (directory/'specs'/f"{spec['id']}.json").write_text(json.dumps(spec,indent=2))
        for index in range(args.worlds):
            seed=seed_for(spec,args.phase,index); world=make_world(spec,seed,args.practice,args.tests); name=spec['id']+'__'+str(seed)+'.json'; (directory/'worlds'/name).write_text(json.dumps(world,indent=2))
            maximum=max(len(tokenizer.encode(answer_prompt(world,world['practice'],q))) for q in world['tests']); records.append({'spec':spec['id'],'seed':seed,'status':'ready' if maximum<=14000 else 'context_too_long','world':name,'max_teacher_prompt_tokens':maximum})
    (directory/'preflight.json').write_text(json.dumps(records,indent=2)); print(json.dumps({'settings':len(specs),'episodes':len(records),'ready':sum(r['status']=='ready' for r in records)}))


def select():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); directory=ROOT.resolve()/args.phase
    summary=json.loads((directory/'summary.json').read_text()); pre=json.loads((directory/'preflight.json').read_text()); ready={r['spec']:r for r in pre if r['status']=='ready'}; assert len(summary)==len(ready)
    groups=defaultdict(list)
    for row in summary: groups[row['task'].rsplit('_d',1)[0]].append(row)
    chosen=[]; decisions=[]
    for family,rows in sorted(groups.items()):
        best=min(rows,key=lambda r:(abs(r['conditions'][0]['accuracy']-.5),ready[r['spec']]['max_teacher_prompt_tokens'],r['spec']))
        chosen.append(json.loads((directory/'specs'/f"{best['spec']}.json").read_text())); decisions.append({'family':family,'selected':best['spec'],'baseline_accuracy':best['conditions'][0]['accuracy'],'all_candidates':rows})
    args.output.write_text(json.dumps({'purpose':'One magnitude per operation/arity, chosen using initial-only accuracy. All four operation/arity settings are retained, including ceilings and floors.','source_summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),'specifications':chosen,'decisions':decisions},indent=2)); print(json.dumps(decisions,indent=2))


def audit():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True); args=parser.parse_args(); directory=ROOT.resolve()/args.phase; pre=json.loads((directory/'preflight.json').read_text()); names=[r['world'] for r in pre if r['status']=='ready']
    assert sorted(names)==sorted(p.name for p in (directory/'evaluations').glob('*.json')); groups=defaultdict(list); episodes=[]
    for name in tqdm(names,desc='Decomposition strategy audit',mininterval=10):
        world=json.loads((directory/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text()); samples=[]; finals=[]
        for arm,entries in [('initial',world['support']),('teacher',world['practice'])]:
            for i,entry in enumerate(entries):
                valid,identity=solution(entry['answer'],entry); assert valid; row={'arm':arm,'index':i,'valid':valid,'identity':identity}; samples.append(row); groups[(world['spec']['id'],'sample',arm,None)].append(row)
        for entry in world['tests']: assert solution(entry['answer'],entry)[0]
        for generated in result['generations']['self_fixed']:
            entry=world['practice'][generated['index']]; assert generated['example']['question']==entry['question']; valid,identity=solution(generated['example']['answer'],entry)
            row={'arm':'self_fixed','index':generated['index'],'valid':valid,'identity':identity}; samples.append(row); groups[(world['spec']['id'],'sample','self_fixed',None)].append(row)
        for condition in result['conditions']:
            if condition['status']!='ok': continue
            for entry,prediction in zip(world['tests'],condition['predictions']):
                valid,identity=solution(prediction['answer'],entry); assert valid==prediction['correct']; row={'arm':condition['arm'],'n':condition['n'],'id':entry['id'],'valid':valid,'identity':identity}; finals.append(row); groups[(world['spec']['id'],'final',condition['arm'],condition['n'])].append(row)
        episodes.append({'spec':world['spec']['id'],'seed':world['seed'],'samples':samples,'finals':finals})
    summary=[{'spec':spec,'kind':kind,'arm':arm,'n':n,'count':len(rows),'valid_fraction':float(np.mean([r['valid'] for r in rows])),'identity_fraction':float(np.mean([r['identity'] for r in rows]))} for (spec,kind,arm,n),rows in sorted(groups.items(),key=lambda p:str(p[0]))]
    record={'timing':'After complete inference; never used to filter, repair, label, or reweight own examples.','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'summary':summary,'episodes':episodes}
    (directory/'strategy_diagnostics.json').write_text(json.dumps(record,indent=2)); print(json.dumps(summary,indent=2))


def main():
    names=register(); mode=sys.argv.pop(1)
    if mode=='prepare': prepare(names)
    elif mode=='select': select()
    elif mode=='strategy_audit': audit()
    else:
        assert mode in {'run','audit'}
        if mode=='run':
            from run_task_search import main as run
            phase=sys.argv[sys.argv.index('--phase')+1]; source=ROOT.resolve()/phase/'extension_source'; source.mkdir(exist_ok=True); (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes()); run()
        else:
            from audit_task_search import main as sample_audit
            sample_audit()


if __name__=='__main__': main()
