"""Compare unfiltered learner answers with teacher answers on exactly matched questions."""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from inverse_icl_api import ChatRouter,MODELS,gather,save_json,work
from matched_source_tasks import FAMILIES,canonical,extract_answer,generation_prompt,inference_prompt,make_world,messages,score

ROOT=Path(__file__).resolve().parents[1]


async def run_world(api,directory,world,config):
    path=directory/'cases'/f"{world['family']}__{world['level']}__{world['seed']}.json"
    if path.exists(): return json.loads(path.read_text())
    generations={}; rows=[]
    async def forward(model,item):
        record=await api.complete(model,messages(generation_prompt(world,item)),config['generation_tokens'])
        answer=extract_answer(record['text'])
        return {'question':item['question'],'answer':answer,'used_answer':answer if answer is not None else 'UNPARSED',
                'response':record['text'],'call':work(record)}
    async def prepare(model):
        generations[model]=await gather(*(forward(model,item) for item in world['practice']))
    await gather(*(prepare(model) for model in config['models']))
    batches=[world['tests'][i:i+config['batch_size']] for i in range(0,len(world['tests']),config['batch_size'])]
    async def evaluate(model,source,n):
        if source=='zero': answers=[]; preparation=[]
        elif source=='oracle': answers=[item['answer'] for item in world['practice'][:n]]; preparation=[]
        else: answers=[item['used_answer'] for item in generations[source][:n]]; preparation=[item['call'] for item in generations[source][:n]]
        async def one(queries):
            record=await api.complete(model,messages(inference_prompt(world,answers,queries)),config['inference_tokens'])
            return {'response':record['text'],'call':work(record),'scores':score(queries,record['text'])}
        evaluations=await gather(*(one(batch) for batch in batches))
        scores=[item for evaluation in evaluations for item in evaluation['scores']]
        rows.append({'model':model,'source':source,'budget':n,'answers':answers,'preparation':preparation,'evaluations':evaluations,
                     'scores':scores,'accuracy':float(np.mean([item['correct'] for item in scores]))})
    calls=[]
    for model in config['models']:
        calls.append(evaluate(model,'zero',0))
        sources=['oracle',model]+([other for other in config['models'] if other!=model] if config['cross_teacher'] else [])
        for n in config['budgets']:
            if n: calls.extend(evaluate(model,source,n) for source in sources)
    await gather(*calls)
    # Oracle answers enter self-generation records only after every evaluation is complete.
    for model,generation in generations.items():
        for item,truth in zip(generation,world['practice'],strict=True): item['correct']=item['answer']==canonical(truth['answer'])
    result={'world':world,'generations':generations,'rows':rows}
    save_json(path,result); return result


async def run(args):
    root=args.output.resolve(); directory=root/args.phase; (directory/'cases').mkdir(parents=True,exist_ok=True)
    cells=[(family,level) for family in args.families for level in args.levels]
    if args.selection: cells=[(cell['family'],cell['level']) for cell in json.loads(args.selection.read_text())['cells']]
    config={'models':{model:MODELS[model] for model in args.models},'cells':[list(cell) for cell in cells],
            'seed_base':args.seed_base,'worlds_per_cell':args.worlds,'test_count':args.tests,'support_count':4,'budgets':args.budgets,
            'generation_tokens':args.generation_tokens,'inference_tokens':args.inference_tokens,'batch_size':args.batch_size,
            'cross_teacher':args.cross_teacher,'sample_source':'Identical questions, ordering, initial examples, and prompt template; only answer source varies.'}
    sources=['matched_source_tasks.py','run_matched_source.py','inverse_icl_api.py','self_icl_api.py']
    hashes={name:hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in sources}
    manifest={'config':config,'source_sha256':hashes,'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    if (directory/'manifest.json').exists():
        previous=json.loads((directory/'manifest.json').read_text())
        assert previous['config']==config and previous['source_sha256']==hashes,'Frozen source-only protocol changed.'
    else:
        save_json(directory/'manifest.json',manifest); (directory/'source').mkdir(exist_ok=True)
        for name in sources: shutil.copy2(ROOT/'experiments'/name,directory/'source'/name)
    worlds=[make_world(family,level,args.seed_base+1000*FAMILIES.index(family)+100*level+i,max(args.budgets),args.tests) for family,level in cells for i in range(args.worlds)]
    save_json(directory/'worlds.json',worlds)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],args.concurrency)
    semaphore=asyncio.Semaphore(args.world_concurrency); progress=tqdm(total=len(worlds),desc=args.phase,unit='world',mininterval=5)
    async def one(world):
        async with semaphore:
            value=await run_world(api,directory,world,config); progress.update(); return value
    try:
        values=await gather(*(one(world) for world in worlds)); save_json(directory/'results.json',values)
    finally:
        progress.close(); await api.close(); print(json.dumps(api.spending()),flush=True)


def main():
    load_dotenv(Path.home()/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/matched_source_icl'); parser.add_argument('--phase',required=True)
    parser.add_argument('--families',nargs='+',choices=FAMILIES,default=list(FAMILIES)); parser.add_argument('--levels',nargs='+',type=int,default=[1,2])
    parser.add_argument('--selection',type=Path); parser.add_argument('--models',nargs='+',choices=MODELS,default=['qwen_flash'])
    parser.add_argument('--seed-base',type=int,default=300000); parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--budgets',nargs='+',type=int,default=[0,4,16]); parser.add_argument('--tests',type=int,default=24)
    parser.add_argument('--generation-tokens',type=int,default=1024); parser.add_argument('--inference-tokens',type=int,default=512)
    parser.add_argument('--batch-size',type=int,default=6); parser.add_argument('--cross-teacher',action='store_true')
    parser.add_argument('--concurrency',type=int,default=32); parser.add_argument('--world-concurrency',type=int,default=8)
    args=parser.parse_args()
    assert args.budgets==sorted(set(args.budgets)) and args.budgets[0]==0 and args.budgets[-1]<=64
    asyncio.run(run(args))


if __name__=='__main__': main()
