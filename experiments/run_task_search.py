"""Run a fixed-model task calibration or paired source comparison."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from task_search_api import LocalModel,ROOT,schema
from task_search_data import dataset,free_prompt,seed_for
from task_search_scoring import score_answer
from task_search_prompts import answer_prompt


async def evaluate(model,world,examples,arm,n):
    data=dataset(world); queries=world['tests']
    calls=await asyncio.gather(*(model.call(answer_prompt(world,examples,query),None,world['seed']+i,max_tokens=1024) for i,query in enumerate(queries)))
    records=[]
    for entry,call in zip(queries,calls):
        if call['status']=='context_overflow': return {'arm':arm,'n':n,'status':'context_overflow','calls':calls}
        answer=call['value']['answer'] if call['status']=='ok' else ''
        reward=score_answer(data,world['spec']['task'],answer,entry)
        records.append({'id':entry['id'],'answer':answer,'reward':reward,'correct':reward>=1-1e-9,'api_key':call['key'],'output_status':call['status']})
    return {'arm':arm,'n':n,'status':'ok','accuracy':float(np.mean([r['correct'] for r in records])),'mean_reward':float(np.mean([r['reward'] for r in records])),'predictions':records,'calls':calls}


async def run_world(model,world,config):
    arms=config['arms']; ns=config['sample_counts']; count=max(ns); outputs=[]; generations={}
    # Generation sees only the initial examples and its assigned rehearsal question, never tests.
    jobs=[]; keys=[]
    for arm in arms:
        if arm=='self_fixed':
            for i,item in enumerate(world['practice'][:count]):
                jobs.append(model.call(answer_prompt(world,[],item),None,seed_for(world['spec'],f"{world['seed']}|fixed",i),temperature=.7,max_tokens=1024)); keys.append((arm,i,item['question']))
        elif arm=='self_free':
            for i in range(count):
                jobs.append(model.call(free_prompt(world),schema(['question','answer']),seed_for(world['spec'],f"{world['seed']}|free",i),temperature=.7,max_tokens=1024)); keys.append((arm,i,None))
    calls=await asyncio.gather(*jobs)
    for (arm,i,question),call in zip(keys,calls):
        if arm not in generations: generations[arm]=[]
        fields={'answer'} if arm=='self_fixed' else {'question','answer'}
        valid=call['status']=='ok' and isinstance(call['value'],dict) and set(call['value'])==fields and all(isinstance(v,str) for v in call['value'].values())
        example={'question':question if arm=='self_fixed' else call['value']['question'] if valid else '[Generation did not produce a complete example.]','answer':call['value']['answer'] if valid else ''}
        generations[arm].append({'index':i,'example':example,'call':call})
    conditions=[('zero',0,[])] if 0 in ns else []
    for arm in arms:
        for n in ns:
            if n==0: continue
            examples=world['practice'][:n] if arm=='teacher' else [g['example'] for g in generations[arm][:n]]
            assert len(examples)==n
            conditions.append((arm,n,examples))
    outputs=await asyncio.gather(*(evaluate(model,world,examples,arm,n) for arm,n,examples in conditions))
    return {'spec':world['spec'],'seed':world['seed'],'generations':generations,'conditions':outputs}


async def main_async(args):
    directory=ROOT.resolve()/args.phase; results=directory/'evaluations'; results.mkdir(exist_ok=True); model=LocalModel(args.concurrency)
    preflight=json.loads((directory/'preflight.json').read_text()); ready=[r for r in preflight if r['status']=='ready']
    config={'phase':args.phase,'engine_fingerprint':model.fingerprint,'arms':args.arms,'sample_counts':args.ns,'questions_per_response':1,'generation_temperature':.7,'evaluation_temperature':0.,'enable_thinking':False,'max_output_tokens':1024,
            'ready_worlds':[r['world'] for r in ready],'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']}}
    manifest=directory/'inference_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==config,'Inference source/config changed. Preserve this phase and start a new one.'
    else: manifest.write_text(json.dumps(config,indent=2))
    snapshot=directory/'source'; snapshot.mkdir(exist_ok=True)
    for name in config['source_sha256']: (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    queue=asyncio.Semaphore(args.world_concurrency)
    async def one(record):
        path=results/record['world']
        if path.exists(): return json.loads(path.read_text())
        async with queue:
            world=json.loads((directory/'worlds'/record['world']).read_text()); result=await run_world(model,world,config)
            path.write_text(json.dumps(result,indent=2)); return result
    summary=[]
    for future in tqdm(asyncio.as_completed([one(r) for r in ready]),total=len(ready),desc=args.phase,mininterval=10):
        result=await future; summary.append({'task':result['spec']['task'],'spec':result['spec']['id'],'seed':result['seed'],'conditions':[{k:v for k,v in condition.items() if k not in {'predictions','calls'}} for condition in result['conditions']]})
        (directory/'summary.json').write_text(json.dumps(sorted(summary,key=lambda r:(r['spec'],r['seed'])),indent=2))
    await model.close()
    print(json.dumps({'worlds':len(summary),'engine':model.fingerprint},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); parser.add_argument('--arms',nargs='*',default=['teacher','self_fixed','self_free'])
    parser.add_argument('--ns',nargs='+',type=int,default=[0,2,8]); parser.add_argument('--concurrency',type=int,default=16); parser.add_argument('--world-concurrency',type=int,default=4)
    args=parser.parse_args(); assert set(args.arms)<={'teacher','self_fixed','self_free'}; assert args.ns==sorted(set(args.ns)); asyncio.run(main_async(args))


if __name__=='__main__': main()
