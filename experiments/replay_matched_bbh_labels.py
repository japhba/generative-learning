"""Gold-blind parser sensitivity: retain bare final BBH labels as generated answers."""

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
from matched_source_tasks import extract_answer,inference_prompt,messages
from run_matched_bbh import answer_label,score_bbh


def generated_label(question,text):
    marked=extract_answer(text)
    return answer_label(question,marked if marked is not None else text.strip().split('\n')[-1])


async def run(args):
    root=args.input.resolve().parent; directory=root/args.phase; (directory/'cases').mkdir(parents=True,exist_ok=True)
    original=json.loads((args.input/'manifest.json').read_text()); config={**original['config'],'input_phase':args.input.name,'parser_sensitivity':'Accept an exact bare last-line option/validity label when ANSWER prefix is absent; no gold consulted.'}
    paths=sorted((args.input/'cases').glob('*.json')); assert len(paths)==len(config['cells'])*config['worlds_per_cell']
    inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    names=['replay_matched_bbh_labels.py','run_matched_bbh.py','matched_source_tasks.py','inverse_icl_api.py','self_icl_api.py']
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}
    manifest={'config':config,'input_case_sha256':inputs,'source_sha256':hashes,'created_at':datetime.now(ZoneInfo('Europe/London')).isoformat()}
    assert not (directory/'manifest.json').exists()
    save_json(directory/'manifest.json',manifest); (directory/'source').mkdir()
    for name in names: shutil.copy2(Path(__file__).parent/name,directory/'source'/name)
    api=ChatRouter(root/'api',os.environ['OPENROUTER_API_KEY'],8); progress=tqdm(total=len(paths),desc=args.phase,unit='world')
    async def one(path):
        case=json.loads(path.read_text()); world=case['world']; generation=[]
        for item in case['generations']['qwen_flash']:
            answer=generated_label(item['question'],item['response'])
            generation.append({**item,'answer':answer,'used_answer':answer if answer is not None else 'UNPARSED'})
        rows=[r for r in case['rows'] if r['source']!='qwen_flash']
        async def evaluate(n):
            answers=[g['used_answer'] for g in generation[:n]]
            async def batch(queries):
                record=await api.complete('qwen_flash',messages(inference_prompt(world,answers,queries)),512)
                return {'response':record['text'],'call':work(record),'scores':score_bbh(queries,record['text'])}
            evaluations=await gather(*(batch(world['tests'][i:i+6]) for i in range(0,len(world['tests']),6)))
            scores=[s for e in evaluations for s in e['scores']]
            rows.append({'model':'qwen_flash','source':'qwen_flash','budget':n,'answers':answers,'preparation':[g['call'] for g in generation[:n]],
                         'evaluations':evaluations,'scores':scores,'accuracy':float(np.mean([s['correct'] for s in scores]))})
        await gather(*(evaluate(n) for n in config['budgets'] if n))
        for item,query in zip(generation,world['practice'],strict=True): item['correct']=item['answer']==query['answer']
        result={'world':world,'generations':{'qwen_flash':generation},'rows':rows}; save_json(directory/'cases'/path.name,result); progress.update(); return result
    try: save_json(directory/'results.json',await gather(*(one(path) for path in paths)))
    finally: progress.close(); await api.close(); print(json.dumps(api.spending()))


def audit(directory):
    root=directory.resolve().parent; manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']; changed=0; checked=0
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==digest
    paths=sorted((directory/'cases').glob('*.json')); assert {p.name for p in paths}==set(manifest['input_case_sha256'])
    for path in paths:
        original=root/config['input_phase']/'cases'/path.name; assert hashlib.sha256(original.read_bytes()).hexdigest()==manifest['input_case_sha256'][path.name]
        case=json.loads(path.read_text()); before=json.loads(original.read_text()); world=case['world']; assert world==before['world']
        for item,old,query in zip(case['generations']['qwen_flash'],before['generations']['qwen_flash'],world['practice'],strict=True):
            answer=generated_label(query['question'],old['response']); expected={**old,'answer':answer,'used_answer':answer if answer is not None else 'UNPARSED','correct':answer==query['answer']}
            assert item==expected; changed+=item['used_answer']!=old['used_answer']
        assert len(case['rows'])==len(before['rows']) and {(r['source'],r['budget']) for r in case['rows']}=={(r['source'],r['budget']) for r in before['rows']}
        for row in case['rows']:
            if row['source']!='qwen_flash':
                assert row==next(r for r in before['rows'] if r['source']==row['source'] and r['budget']==row['budget']); continue
            answers=[g['used_answer'] for g in case['generations']['qwen_flash'][:row['budget']]]; assert row['answers']==answers; scores=[]
            for evaluation in row['evaluations']:
                record=json.loads((root/'api'/f"{evaluation['call']['call_id']}.json").read_text()); request=record['request']; queries=evaluation['scores']
                assert hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()==evaluation['call']['call_id']
                assert request['messages']==messages(inference_prompt(world,answers,queries)) and request['model']==config['models']['qwen_flash']['id']
                assert request['temperature']==0 and request['reasoning']=={'enabled':False} and request['max_tokens']==512
                assert record['text']==evaluation['response'] and score_bbh(queries,record['text'])==queries
                scores.extend(queries); checked+=1
            assert scores==row['scores'] and row['accuracy']==float(np.mean([s['correct'] for s in scores]))
            assert [{key:s[key] for key in ('question','answer','dataset_index','id')} for s in scores]==world['tests']
    result={'worlds':len(paths),'changed_rehearsal_labels':changed,'evaluation_requests_checked':checked,'checks':'Frozen sources and input cases; same raw generations, questions, and teacher/zero rows; gold-blind bare-label parsing only; incorrect labels retained; unchanged inference templates and limits; exact scoring.'}
    save_json(directory/'verification.json',result); print(json.dumps(result,indent=2))


def main():
    load_dotenv(Path.home()/'.env'); parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('input',type=Path); parser.add_argument('--phase'); parser.add_argument('--audit',action='store_true'); args=parser.parse_args()
    if args.audit: audit(args.input)
    else: assert args.phase; asyncio.run(run(args))


if __name__=='__main__': main()
