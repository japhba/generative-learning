"""Target-blind forward rehearsal and paired inverse-inference curves."""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from tqdm.auto import tqdm

from inverse_icl_api import ChatRouter, MODELS, gather, save_json, work
from inverse_icl_task import audit_generated, evaluation_prompt, make_world, predictions, rehearsal_prompt, table_text

ROOT=Path(__file__).resolve().parents[1]
SYSTEM='Follow the specified arithmetic and output format carefully. Work only with the information in the prompt.'


def messages(prompt): return [{'role':'system','content':SYSTEM},{'role':'user','content':prompt}]


async def run_world(api,directory,world,model,budgets,conditions,evaluation_tokens,longer,recognizer):
    path=directory/'cases'/f'{model}__d{world["dimension"]}__{world["seed"]}.json'
    if path.exists(): return json.loads(path.read_text())
    rows=[]; generated=[]; generation_calls=[]; allowance=0

    async def evaluate(condition,n,text,preparation,limit):
        async def one(test):
            query_world={**world,'tests':[test]}
            result=await api.complete(model,messages(evaluation_prompt(query_world,text,reasoning=recognizer=='reasoning' or condition=='long_direct')),limit)
            return {'response':result['text'],'work':work(result),'scores':predictions(query_world,result['text'])}
        results=await gather(*(one(test) for test in world['tests']))
        aggregate={key:sum(result['work'][key] for result in results) for key in ('prompt_tokens','completion_tokens','cost_usd','seconds')}
        aggregate['finish_reason']='length' if any(result['work']['finish_reason']=='length' for result in results) else 'stop'
        aggregate['call_ids']=[result['work']['call_id'] for result in results]
        rows.append({'condition':condition,'budget':n,'scores':[score for result in results for score in result['scores']],
                     'response':'\n\n'.join(result['response'] for result in results),'evaluations':results,
                     'evaluation':aggregate,'preparation':list(preparation),'examples':text,'output_limit':limit})

    await evaluate('zero',0,'',[],evaluation_tokens)
    previous=0
    for n in budgets:
        if n==0: continue
        assert n<=len(world['structured'])
        subsets=[row['subset'] for row in world['structured'][previous:n]]
        limit=128+24*len(subsets)
        response=await api.complete(model,messages(rehearsal_prompt(world,subsets)),limit)
        generated.append({'budget':n,'subsets':subsets,'response':response['text'],'call':work(response)})
        generation_calls.append(work(response)); allowance+=limit
        text='\n'.join(record['response'] for record in generated)
        examples={'generated_parts':text,'teacher_whole':table_text(world['full'][:n]),
                  'teacher_parts':table_text(world['structured'][:n]),'teacher_random_parts':table_text(world['random_parts'][:n])}
        calls=[evaluate(condition,n,examples[condition],generation_calls if condition=='generated_parts' else [],evaluation_tokens) for condition in conditions]
        if longer: calls.append(evaluate('long_direct',n,'',[],evaluation_tokens+math.ceil(allowance/len(world['tests']))))
        await gather(*calls)
        previous=n
    result={'world':world,'model':model,'rows':rows,'generated':generated}
    # This diagnostic is computed after every response; it cannot affect requests.
    result['generation_audits']=[audit_generated(world,row['subsets'],row['response']) for row in generated]
    save_json(path,result)
    return result


async def run(args):
    root=args.output.resolve(); directory=root/args.phase
    (directory/'cases').mkdir(parents=True,exist_ok=True)
    config={'models':{name:MODELS[name] for name in args.models},'dimensions':args.dimensions,'world_count':args.worlds,'seed':args.seed,
            'queries':args.queries,'budgets':args.budgets,'conditions':args.conditions,'evaluation_tokens':args.evaluation_tokens,'long_direct':args.long_direct,
            'recognizer':args.recognizer,'evaluation_query_mode':'individual'}
    source_names=['inverse_icl_task.py','inverse_icl_api.py','run_inverse_icl.py']
    manifest={'config':config,'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat(),
              'source_sha256':{name:hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in source_names}}
    if (directory/'manifest.json').exists():
        previous=json.loads((directory/'manifest.json').read_text())
        assert config==previous['config'] and manifest['source_sha256']==previous['source_sha256'],'Frozen protocol changed.'
    else:
        save_json(directory/'manifest.json',manifest)
        (directory/'source').mkdir(exist_ok=True)
        for name in source_names: shutil.copy2(ROOT/'experiments'/name,directory/'source'/name)
    worlds=[make_world(args.seed+i+dimension*1000,dimension,args.queries) for dimension in args.dimensions for i in range(args.worlds)]
    save_json(directory/'worlds.json',worlds)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],args.concurrency)
    progress=tqdm(total=len(worlds)*len(args.models),desc=args.phase,unit='world',mininterval=5)
    semaphore=asyncio.Semaphore(args.world_concurrency)
    async def one(world,model):
        async with semaphore:
            value=await run_world(api,directory,world,model,[n for n in args.budgets if n<=2**(world['dimension']//2+1)],args.conditions,args.evaluation_tokens,args.long_direct,args.recognizer)
            progress.update()
            return value
    try:
        values=await gather(*(one(world,model) for world in worlds for model in args.models))
        save_json(directory/'results.json',values)
    finally:
        progress.close(); await api.close(); print(json.dumps(api.spending()),flush=True)


def main():
    load_dotenv(Path.home()/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/inverse_icl')
    parser.add_argument('--phase',required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=['qwen_flash','gemini31'])
    parser.add_argument('--dimensions',type=int,nargs='+',default=[8,10,12])
    parser.add_argument('--worlds',type=int,default=2)
    parser.add_argument('--seed',type=int,default=101000)
    parser.add_argument('--queries',type=int,default=8)
    parser.add_argument('--budgets',type=int,nargs='+',default=[0,8,32])
    parser.add_argument('--conditions',nargs='+',default=['generated_parts','teacher_whole','teacher_parts','teacher_random_parts'])
    parser.add_argument('--evaluation-tokens',type=int,default=1536)
    parser.add_argument('--recognizer',choices=['fast','reasoning'],default='reasoning')
    parser.add_argument('--long-direct',action='store_true')
    parser.add_argument('--concurrency',type=int,default=8)
    parser.add_argument('--world-concurrency',type=int,default=2)
    args=parser.parse_args()
    assert args.budgets==sorted(set(args.budgets)) and args.budgets[0]==0
    asyncio.run(run(args))


if __name__=='__main__': main()
