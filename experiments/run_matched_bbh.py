"""Source-only rehearsal on the 23 original Self-ICL BBH classification tasks."""

import argparse
import asyncio
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from inverse_icl_api import ChatRouter,MODELS,gather,save_json,work
from matched_source_tasks import extract_answer,generation_prompt,inference_prompt,messages

ROOT=Path(__file__).resolve().parents[1]


def normalized(text): return re.sub(r'\s+','',text).strip('`*_()[]{}.,:;! \"\'').upper()


def answer_label(question,text):
    value=normalized(text)
    options=re.findall(r'^\s*\(([A-Z])\)\s*(.+?)\s*$',question,re.M)
    if options:
        if value in {letter for letter,_ in options}: return value
        matched=[letter for letter,content in options if value==normalized(content)]
        return matched[0] if len(matched)==1 else None
    return value if value in {'TRUE','FALSE','YES','NO','VALID','INVALID'} else None


def score_bbh(queries,text):
    pairs=re.findall(r'^\s*(\d+)\s*:\s*(?:ANSWER\s*:\s*)?(.+?)\s*$',text.replace('**','').replace('`',''),re.I|re.M)
    values={}
    for index,value in pairs: values.setdefault(int(index),[]).append(value)
    result=[]
    for query in queries:
        candidates={answer_label(query['question'],value) for value in values[query['id']]} if query['id'] in values else set()
        prediction=next(iter(candidates)) if len(candidates)==1 else None
        result.append({**query,'prediction':prediction,'correct':prediction is not None and prediction==query['answer']})
    return result


def eligible_indices(task):
    groups=defaultdict(list)
    for i,item in enumerate(task['examples']): groups[item['input']].append((i,answer_label(item['input'],item['target'])))
    return [items[0][0] for items in groups.values() if None not in {label for _,label in items} and len({label for _,label in items})==1]


def make_world(tasks,name,seed,pool,practice_count,test_count):
    task=tasks[name]; index=sorted(tasks).index(name)
    eligible=eligible_indices(task)
    order=np.random.default_rng(43210+index).permutation(eligible); boundary=len(order)//2
    available=order[:boundary] if pool=='screen' else order[boundary:]
    selected=np.random.default_rng(seed).choice(available,size=4+practice_count+test_count,replace=False).tolist()
    items=[]
    for idx in selected:
        original=task['examples'][idx]; answer=answer_label(original['input'],original['target'])
        assert answer is not None,(name,idx,original)
        items.append({'question':original['input'],'answer':answer,'dataset_index':idx})
    assert len(items)==len({item['question'] for item in items})
    description=task['description']+' If lettered options are provided, give only the uppercase option letter without parentheses. Otherwise give the Boolean or validity label requested by the task. Do not give option text in place of a letter.'
    return {'family':name,'level':1,'seed':seed,'pool':pool,'description':description,'support':items[:4],
            'practice':items[4:4+practice_count],'tests':[{**item,'id':i+1} for i,item in enumerate(items[4+practice_count:])]}


async def run_world(api,directory,world,config):
    path=directory/'cases'/f"{world['family']}__1__{world['seed']}.json"
    if path.exists(): return json.loads(path.read_text())
    model='qwen_flash'
    async def forward(item):
        record=await api.complete(model,messages(generation_prompt(world,item)),config['generation_tokens'])
        extracted=extract_answer(record['text']); answer=answer_label(item['question'],extracted) if extracted is not None else None
        return {'question':item['question'],'answer':answer,'used_answer':answer if answer is not None else 'UNPARSED','response':record['text'],'call':work(record)}
    generation=await gather(*(forward(item) for item in world['practice'])); rows=[]
    batches=[world['tests'][i:i+config['batch_size']] for i in range(0,len(world['tests']),config['batch_size'])]
    async def evaluate(source,n):
        answers=[] if source=='zero' else ([item['answer'] for item in world['practice'][:n]] if source=='oracle' else [item['used_answer'] for item in generation[:n]])
        async def batch(queries):
            record=await api.complete(model,messages(inference_prompt(world,answers,queries)),config['inference_tokens'])
            return {'response':record['text'],'call':work(record),'scores':score_bbh(queries,record['text'])}
        evaluations=await gather(*(batch(queries) for queries in batches)); scores=[item for e in evaluations for item in e['scores']]
        rows.append({'model':model,'source':source,'budget':n,'answers':answers,'preparation':[g['call'] for g in generation[:n]] if source==model else [],
                     'evaluations':evaluations,'scores':scores,'accuracy':float(np.mean([s['correct'] for s in scores]))})
    await gather(evaluate('zero',0),*(evaluate(source,n) for source in ('oracle',model) for n in config['budgets'] if n))
    for item,truth in zip(generation,world['practice'],strict=True): item['correct']=item['answer']==truth['answer']
    result={'world':world,'generations':{model:generation},'rows':rows}; save_json(path,result); return result


async def run(args):
    root=args.output.resolve(); directory=root/args.phase; (directory/'cases').mkdir(parents=True,exist_ok=True)
    tasks=json.loads((root/'bbh_inputs/original_tasks.json').read_text()); names=sorted(tasks)
    if args.selection: names=[cell['family'] for cell in json.loads(args.selection.read_text())['cells']]
    config={'models':{'qwen_flash':MODELS['qwen_flash']},'cells':[[name,1] for name in names],'worlds_per_cell':args.worlds,'seed_base':args.seed_base,
            'budgets':args.budgets,'test_count':args.tests,'support_count':4,'generation_tokens':1024,'inference_tokens':512,'batch_size':6,'cross_teacher':False,'pool':args.pool,
            'reference_kind':'Published BBH labels, with exact option-text-to-letter normalization.',
            'data_integrity':'Before sampling either source, exclude invalid option targets and conflicting duplicate questions; keep one copy of consistent duplicates. Preserve and report every excluded corpus record.'}
    sources=['run_matched_bbh.py','matched_source_tasks.py','inverse_icl_api.py','self_icl_api.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in sources}
    manifest={'config':config,'source_sha256':hashes,'data_sha256':hashlib.sha256((root/'bbh_inputs/original_tasks.json').read_bytes()).hexdigest(),'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    if (directory/'manifest.json').exists():
        previous=json.loads((directory/'manifest.json').read_text()); assert previous['config']==config and previous['source_sha256']==hashes and previous['data_sha256']==manifest['data_sha256']
    else:
        save_json(directory/'manifest.json',manifest); (directory/'source').mkdir(exist_ok=True)
        for name in sources: shutil.copy2(Path(__file__).parent/name,directory/'source'/name)
    worlds=[make_world(tasks,name,args.seed_base+1000*sorted(tasks).index(name)+i,args.pool,max(args.budgets),args.tests) for name in names for i in range(args.worlds)]
    save_json(directory/'worlds.json',worlds)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],args.concurrency)
    semaphore=asyncio.Semaphore(8); progress=tqdm(total=len(worlds),desc=args.phase,unit='world',mininterval=5)
    async def one(world):
        async with semaphore:
            result=await run_world(api,directory,world,config); progress.update(); return result
    try:
        results=await gather(*(one(world) for world in worlds)); save_json(directory/'results.json',results)
    finally:
        progress.close(); await api.close(); print(json.dumps(api.spending()),flush=True)


def audit(directory):
    root=directory.resolve().parent; manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']
    tasks=json.loads((root/'bbh_inputs/original_tasks.json').read_text()); checked=0
    assert hashlib.sha256((root/'bbh_inputs/original_tasks.json').read_bytes()).hexdigest()==manifest['data_sha256']
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    cases=sorted((directory/'cases').glob('*.json')); assert len(cases)==len(config['cells'])*config['worlds_per_cell']
    expected={(name,config['seed_base']+1000*sorted(tasks).index(name)+i) for name,_ in config['cells'] for i in range(config['worlds_per_cell'])}
    observed=set()
    for path in tqdm(cases,desc='Audit BBH',unit='world'):
        case=json.loads(path.read_text()); world=case['world']; observed.add((world['family'],world['seed']))
        assert world==make_world(tasks,world['family'],world['seed'],config['pool'],max(config['budgets']),config['test_count'])
        generation=case['generations']['qwen_flash']; assert len(generation)==max(config['budgets'])
        for item,query in zip(generation,world['practice'],strict=True):
            record=json.loads((root/'api'/f"{item['call']['call_id']}.json").read_text()); request=record['request']
            assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==item['call']['call_id']
            assert request['messages']==messages(generation_prompt(world,query)) and request['max_tokens']==1024
            assert record['text']==item['response'] and item['question']==query['question']
            extracted=extract_answer(item['response']); answer=answer_label(query['question'],extracted) if extracted is not None else None
            assert item['answer']==answer and item['used_answer']==(answer if answer is not None else 'UNPARSED') and item['correct']==(answer==query['answer'])
        expected_rows={('zero',0)}|{(source,n) for source in ('oracle','qwen_flash') for n in config['budgets'] if n}
        assert len(case['rows'])==len(expected_rows) and {(r['source'],r['budget']) for r in case['rows']}==expected_rows
        for row in case['rows']:
            n=row['budget']; source=row['source']; answers=[] if source=='zero' else ([i['answer'] for i in world['practice'][:n]] if source=='oracle' else [i['used_answer'] for i in generation[:n]])
            assert row['answers']==answers; scores=[]
            for evaluation in row['evaluations']:
                record=json.loads((root/'api'/f"{evaluation['call']['call_id']}.json").read_text()); request=record['request']; queries=evaluation['scores']
                assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==evaluation['call']['call_id']
                assert request['model']==MODELS['qwen_flash']['id'] and request['temperature']==0 and request['reasoning']=={'enabled':False}
                assert request['max_tokens']==512 and request['messages']==messages(inference_prompt(world,answers,queries))
                assert record['text']==evaluation['response'] and score_bbh(queries,record['text'])==queries
                checked+=1; scores.extend(queries)
            assert scores==row['scores'] and row['accuracy']==float(np.mean([s['correct'] for s in scores]))
            assert [{key:item[key] for key in ('question','answer','dataset_index','id')} for item in scores]==world['tests']
    assert observed==expected
    result={'worlds':len(cases),'evaluation_requests_checked':checked,'checks':'Frozen source/data hashes; regenerated disjoint question partitions; separate screen/confirmation dataset halves; no teacher/test labels in generation; source-only answer slots; exact option-text normalization; unchanged unfiltered samples and final scoring.'}
    save_json(directory/'verification.json',result); print(json.dumps(result,indent=2))


def main():
    load_dotenv(Path.home()/'.env'); parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'results/matched_source_icl'); parser.add_argument('--phase',required=True)
    parser.add_argument('--pool',choices=['screen','confirmation'],default='screen'); parser.add_argument('--seed-base',type=int,default=1500000)
    parser.add_argument('--worlds',type=int,default=2); parser.add_argument('--tests',type=int,default=12); parser.add_argument('--budgets',nargs='+',type=int,default=[0,2,8])
    parser.add_argument('--selection',type=Path); parser.add_argument('--concurrency',type=int,default=32); parser.add_argument('--audit',action='store_true'); args=parser.parse_args()
    if args.audit: audit(args.output/args.phase)
    else: asyncio.run(run(args))


if __name__=='__main__': main()
