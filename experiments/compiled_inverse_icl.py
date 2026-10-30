"""Unjudged forward generation for amortized modular inverse queries with a known Zipf prior."""

import argparse
import asyncio
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from inverse_icl_api import ChatRouter,MODELS,gather,save_json,work

ROOT=Path(__file__).resolve().parents[1]
MODULUS=257
SIZE=256
EXPONENT=1.5
FORWARD_TOKENS=1024
STRATA=[(1,4,4),(5,8,4),(9,16,4),(17,32,4),(33,64,4),(65,128,4),(129,256,8)]


def make_world(seed):
    rng=np.random.default_rng(seed)
    a=int(rng.integers(17,240)); b=int(rng.integers(0,MODULUS))
    labels=np.arange(1,SIZE+1)
    probabilities=labels.astype(float)**-EXPONENT; probabilities/=probabilities.sum()
    teacher=rng.choice(labels,size=128,p=probabilities).tolist()
    queries=[]
    for lower,upper,count in STRATA:
        for label in rng.choice(np.arange(lower,upper+1),size=count,replace=False):
            queries.append({'label':int(label),'code':(a*int(label)+b)%MODULUS,'weight':float(probabilities[label-1]*(upper-lower+1)/count)})
    rng.shuffle(queries)
    for index,item in enumerate(queries): item['id']=index+1
    return {'seed':seed,'a':a,'b':b,'modulus':MODULUS,'size':SIZE,'exponent':EXPONENT,'teacher_labels':teacher,'queries':queries,'probabilities':probabilities.tolist()}


def header(world):
    return (f"Labels z are integers 1 through {world['size']}. Their codes are x = ({world['a']} * z + {world['b']}) mod {world['modulus']}, "
            f"where mod means the nonnegative remainder. This map is one-to-one on the allowed labels.\n"
            'Your task is to invert the code: given x, return its label z. Labels are queried with probability proportional to z^(-1.5), so smaller labels are more frequent.\n')


def generation_prompt(world,label):
    return (f"Calculate the nonnegative remainder of ({world['a']} * {label} + {world['b']}) divided by {world['modulus']}. "
            'Show a short calculation, then end with CODE: followed by the remainder as an integer. Do not invert any code.')


def decode_prompt(world,examples,queries,reasoning=False):
    reference='\nForward examples:\n'+examples+'\n' if examples else ''
    instruction='Reason briefly, then give the answers.' if reasoning else 'Return only the answer lines. No reasoning or explanation.'
    return header(world)+reference+'\n'+instruction+' Use id: integer on each line; for example 12: 7 means the answer to query 12 is label 7.\n'+'\n'.join(f"{row['id']}: code {row['code']}" for row in queries)


def messages(prompt): return [{'role':'system','content':'Follow the arithmetic definition and requested answer format.'},{'role':'user','content':prompt}]


def parse_code(text):
    matches=re.findall(r'CODE\s*:\s*(-?\d+)',text,re.I)
    return int(matches[-1]) if matches else None


def table(world,labels): return '\n'.join(f"Code {(world['a']*label+world['b'])%world['modulus']} -> label {label}" for label in labels)


def score(queries,text):
    clean=re.sub(r'\*\*|`','',text)
    pattern=r'^\s*(?:ANSWER\s+)?(\d+)\s*:\s*(?:code\s+\d+\s*->\s*)?(?:(?:label|answer)\s*:?\s*)?(-?\d+)\s*[.!]?\s*$'
    answers={int(index):int(label) for index,label in re.findall(pattern,clean,re.I|re.M)}
    return [{**row,'prediction':answers[row['id']] if row['id'] in answers else None,'correct':row['id'] in answers and answers[row['id']]==row['label']} for row in queries]


async def run_world(api,directory,world,model,budgets,batch_size,long_budgets,iid_generated):
    path=directory/'cases'/f'{model}__{world["seed"]}.json'
    if path.exists(): return json.loads(path.read_text())
    generation=[]; iid_generation=[]; rows=[]; preparation=[]; iid_preparation=[]
    batches=[world['queries'][i:i+batch_size] for i in range(0,len(world['queries']),batch_size)]

    async def evaluate(condition,n,examples,prep,limit,reasoning=False):
        async def one(queries):
            result=await api.complete(model,messages(decode_prompt(world,examples,queries,reasoning)),limit)
            return {'response':result['text'],'call':work(result),'scores':score(queries,result['text'])}
        values=await gather(*(one(batch) for batch in batches))
        records=[row for value in values for row in value['scores']]
        rows.append({'condition':condition,'budget':n,'evaluations':values,'preparation':list(prep),'examples':examples,
                     'risk':sum(row['weight']*(not row['correct']) for row in records),'scores':records})

    await evaluate('zero',0,'',[],512)
    previous=0
    for n in budgets:
        if n==0: continue
        async def forward(label):
            result=await api.complete(model,messages(generation_prompt(world,label)),FORWARD_TOKENS)
            return {'label':label,'code':parse_code(result['text']),'response':result['text'],'call':work(result)}
        generated=await gather(*(forward(label) for label in range(previous+1,n+1)))
        generation.extend(generated); preparation.extend(item['call'] for item in generated)
        examples='\n'.join(f"Code {item['code']} -> label {item['label']}" if item['code'] is not None else f"Forward computation for label {item['label']}: {item['response']}" for item in generation)
        calls=[evaluate('generated',n,examples,preparation,512),evaluate('teacher_iid',n,table(world,world['teacher_labels'][:n]),[],512),
               evaluate('teacher_ranked',n,table(world,range(1,n+1)),[],512)]
        if iid_generated:
            additions=await gather(*(forward(label) for label in world['teacher_labels'][previous:n]))
            iid_generation.extend(additions); iid_preparation.extend(item['call'] for item in additions)
            iid_examples='\n'.join(f"Code {item['code']} -> label {item['label']}" if item['code'] is not None else f"Forward computation for label {item['label']}: {item['response']}" for item in iid_generation)
            calls.append(evaluate('generated_iid',n,iid_examples,iid_preparation,512))
        if n in long_budgets:
            extra_tokens=sum(item['prompt_tokens']+item['completion_tokens'] for item in preparation)
            calls.append(evaluate('long_direct',n,'',[],512+math.ceil(extra_tokens/len(batches)),True))
        await gather(*calls)
        previous=n
    # Ground truth enters only the final diagnostic, never forward generation or its selection.
    for item in generation+iid_generation:
        item['correct_code']=(world['a']*item['label']+world['b'])%world['modulus']
        item['correct']=item['code']==item['correct_code']
    result={'world':world,'model':model,'rows':rows,'generation':generation,'iid_generation':iid_generation}
    save_json(path,result)
    return result


async def run(args):
    root=args.output.resolve(); directory=root/args.phase
    (directory/'cases').mkdir(parents=True,exist_ok=True)
    config={'models':{model:MODELS[model] for model in args.models},'seeds':list(range(args.seed,args.seed+args.worlds)),
            'budgets':args.budgets,'batch_size':args.batch_size,'long_budgets':args.long_budgets,'iid_generated':args.iid_generated,
            'forward_tokens':FORWARD_TOKENS,'modulus':MODULUS,'size':SIZE,'exponent':EXPONENT,'strata':[list(row) for row in STRATA]}
    sources=['compiled_inverse_icl.py','inverse_icl_api.py']
    hashes={name:hashlib.sha256((ROOT/'experiments'/name).read_bytes()).hexdigest() for name in sources}
    manifest={'config':config,'source_sha256':hashes,'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    if (directory/'manifest.json').exists():
        previous=json.loads((directory/'manifest.json').read_text())
        assert previous['config']==config and previous['source_sha256']==hashes,'Frozen protocol changed.'
    else:
        save_json(directory/'manifest.json',manifest); (directory/'source').mkdir(exist_ok=True)
        for name in sources: shutil.copy2(ROOT/'experiments'/name,directory/'source'/name)
    worlds=[make_world(seed) for seed in config['seeds']]
    save_json(directory/'worlds.json',worlds)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],args.concurrency)
    semaphore=asyncio.Semaphore(args.world_concurrency)
    progress=tqdm(total=len(worlds)*len(args.models),desc=args.phase,unit='world',mininterval=5)
    async def one(world,model):
        async with semaphore:
            value=await run_world(api,directory,world,model,args.budgets,args.batch_size,args.long_budgets,args.iid_generated)
            progress.update(); return value
    try:
        result=await gather(*(one(world,model) for world in worlds for model in args.models))
        save_json(directory/'results.json',result)
    finally:
        progress.close(); await api.close(); print(json.dumps(api.spending()),flush=True)


def main():
    load_dotenv(Path.home()/'.env')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/inverse_icl')
    parser.add_argument('--phase',required=True)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=['qwen_flash'])
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--seed',type=int,default=201000)
    parser.add_argument('--budgets',type=int,nargs='+',default=[0,8,32,64])
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--long-budgets',type=int,nargs='*',default=[])
    parser.add_argument('--iid-generated',action='store_true')
    parser.add_argument('--concurrency',type=int,default=16)
    parser.add_argument('--world-concurrency',type=int,default=2)
    args=parser.parse_args()
    assert args.budgets==sorted(set(args.budgets)) and args.budgets[0]==0 and args.budgets[-1]<=128
    asyncio.run(run(args))


if __name__=='__main__': main()
