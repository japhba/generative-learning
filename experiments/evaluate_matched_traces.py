"""Replay full learner or teacher solutions on matched rehearsal questions, without judging."""

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

from inverse_icl_api import ChatRouter,gather,save_json,work
from matched_source_tasks import initial_context,messages,score


def trace_prompt(world,solutions,queries):
    examples='\n\nAdditional worked examples:\n'+'\n'.join(f"Input: {item['question']}\nSolution:\n{solution}" for item,solution in zip(world['practice'][:len(solutions)],solutions,strict=True)) if solutions else ''
    return initial_context(world)+examples+'\n\nSolve the following independent queries. Return only one line per query in the format id: answer. No reasoning or explanations.\n'+'\n'.join(f"{item['id']}: {item['question']}" for item in queries)


async def run(args):
    root=args.input.resolve().parent; directory=root/args.phase; (directory/'cases').mkdir(parents=True,exist_ok=True)
    original=json.loads((args.input/'manifest.json').read_text()); config={**original['config'],'representation':'full_solution','input_phase':args.input.name,'cross_teacher':True}
    paths=sorted((args.input/'cases').glob('*.json'))
    assert len(paths)==len(config['cells'])*config['worlds_per_cell'] and len(config['models'])==2
    inputs={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    source_names=['evaluate_matched_traces.py','matched_source_tasks.py','inverse_icl_api.py','self_icl_api.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in source_names}
    manifest={'config':config,'input_case_sha256':inputs,'source_sha256':hashes,'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    if (directory/'manifest.json').exists():
        previous=json.loads((directory/'manifest.json').read_text())
        assert previous['config']==config and previous['source_sha256']==hashes and previous['input_case_sha256']==inputs
    else:
        save_json(directory/'manifest.json',manifest); (directory/'source').mkdir(exist_ok=True)
        for name in source_names: shutil.copy2(Path(__file__).parent/name,directory/'source'/name)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],args.concurrency)
    semaphore=asyncio.Semaphore(args.world_concurrency); progress=tqdm(total=len(paths),desc=args.phase,unit='world',mininterval=5)
    async def one(path):
        async with semaphore:
            destination=directory/'cases'/path.name
            if destination.exists(): progress.update(); return json.loads(destination.read_text())
            case=json.loads(path.read_text()); world=case['world']; rows=[row for row in case['rows'] if row['source']=='zero']
            batches=[world['tests'][i:i+config['batch_size']] for i in range(0,len(world['tests']),config['batch_size'])]
            async def evaluate(model,source,n):
                generation=case['generations'][source][:n]; solutions=[item['response'] for item in generation]
                async def batch(queries):
                    record=await api.complete(model,messages(trace_prompt(world,solutions,queries)),config['inference_tokens'])
                    return {'response':record['text'],'call':work(record),'scores':score(queries,record['text'])}
                evaluations=await gather(*(batch(queries) for queries in batches)); scores=[item for e in evaluations for item in e['scores']]
                rows.append({'model':model,'source':source,'budget':n,'answers':solutions,'preparation':[item['call'] for item in generation],
                             'evaluations':evaluations,'scores':scores,'accuracy':float(np.mean([item['correct'] for item in scores]))})
            await gather(*(evaluate(model,source,n) for model in config['models'] for source in config['models'] for n in config['budgets'] if n))
            result={'world':world,'generations':case['generations'],'rows':rows}; save_json(destination,result); progress.update(); return result
    try:
        results=await gather(*(one(path) for path in paths)); save_json(directory/'results.json',results)
    finally:
        progress.close(); await api.close(); print(json.dumps(api.spending()),flush=True)


def audit(directory):
    root=directory.resolve().parent; manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']; checked=0
    for filename,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/filename).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()==digest
    cases=sorted((directory/'cases').glob('*.json')); assert len(cases)==len(manifest['input_case_sha256'])
    for path in tqdm(cases,desc='Audit full solutions',unit='world'):
        original=root/config['input_phase']/'cases'/path.name
        assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['input_case_sha256'][path.name]
        case=json.loads(path.read_text()); original_case=json.loads(original.read_text()); world=case['world']
        assert world==original_case['world'] and case['generations']==original_case['generations']
        expected={(m,'zero',0) for m in config['models']}|{(m,s,n) for m in config['models'] for s in config['models'] for n in config['budgets'] if n}
        assert len(case['rows'])==len(expected) and {(r['model'],r['source'],r['budget']) for r in case['rows']}==expected
        for row in case['rows']:
            if row['source']=='zero':
                assert row==next(r for r in original_case['rows'] if r['source']=='zero' and r['model']==row['model']); continue
            solutions=[item['response'] for item in case['generations'][row['source']][:row['budget']]]
            assert row['answers']==solutions
            scores=[]
            for evaluation in row['evaluations']:
                record=json.loads((root/'api'/f"{evaluation['call']['call_id']}.json").read_text()); request=record['request']
                assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==evaluation['call']['call_id']
                assert request['messages']==messages(trace_prompt(world,solutions,evaluation['scores']))
                assert request['model']==config['models'][row['model']]['id'] and request['max_tokens']==config['inference_tokens']
                assert request['temperature']==0 and request['reasoning']=={'enabled':False}
                assert record['text']==evaluation['response'] and score(evaluation['scores'],record['text'])==evaluation['scores']
                checked+=1; scores.extend(evaluation['scores'])
            assert scores==row['scores'] and row['accuracy']==float(np.mean([item['correct'] for item in scores]))
            assert [{key:item[key] for key in ('question','answer','id')} for item in scores]==world['tests']
    result={'worlds':len(cases),'evaluation_requests_checked':checked,'checks':'Frozen source and input hashes; complete paired source arms; identical worlds and question ordering; original full solutions retained without filtering, repair, or source labels; identical final instructions and allowance; exact scores.'}
    save_json(directory/'verification.json',result); print(json.dumps(result,indent=2))


def main():
    load_dotenv(Path.home()/'.env')
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('input',type=Path); parser.add_argument('--phase'); parser.add_argument('--audit',action='store_true')
    parser.add_argument('--concurrency',type=int,default=32); parser.add_argument('--world-concurrency',type=int,default=8); args=parser.parse_args()
    if args.audit: audit(args.input)
    else:
        assert args.phase; asyncio.run(run(args))


if __name__=='__main__': main()
