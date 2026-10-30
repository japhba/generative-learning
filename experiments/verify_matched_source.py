"""Audit source-only interventions against cached requests and regenerated task worlds."""

import argparse
from decimal import Decimal
from functools import cache
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from matched_source_tasks import FAMILIES,canonical,extract_answer,generation_prompt,inference_prompt,make_world,messages,score
from self_icl_api import save_json


def verify(root,phases):
    @cache
    def response(call):
        record=json.loads((root/'api'/f'{call}.json').read_text())
        assert hashlib.sha256(json.dumps(record['request'],sort_keys=True).encode()).hexdigest()==call
        assert record['request']['temperature']==0 and record['request']['top_p']==1
        assert record['request']['reasoning']=={'enabled':False}
        return record
    summaries={}
    for phase in phases:
        directory=root/phase; manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']
        for filename,digest in manifest['source_sha256'].items():
            assert hashlib.sha256((directory/'source'/filename).read_bytes()).hexdigest()==digest
            assert hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()==digest
        cases=[json.loads(p.read_text()) for p in sorted((directory/'cases').glob('*.json'))]
        expected={(family,level,config['seed_base']+1000*FAMILIES.index(family)+100*level+i) for family,level in config['cells'] for i in range(config['worlds_per_cell'])}
        assert len(cases)==len(expected)
        assert {(c['world']['family'],c['world']['level'],c['world']['seed']) for c in cases}==expected
        generation_calls=0; evaluation_calls=0; identical=0; unparsed=0
        for case in tqdm(cases,desc=phase,unit='world'):
            world=case['world']; nmax=max(config['budgets'])
            assert world==make_world(world['family'],world['level'],world['seed'],nmax,config['test_count'],config['support_count'])
            items=world['support']+world['practice']+world['tests']
            assert len(items)==len({item['question'] for item in items})
            assert set(case['generations'])==set(config['models'])
            for model,generation in case['generations'].items():
                assert len(generation)==nmax
                for item,query in zip(generation,world['practice'],strict=True):
                    record=response(item['call']['call_id']); request=record['request']; generation_calls+=1
                    assert request['model']==config['models'][model]['id']
                    assert request['messages']==messages(generation_prompt(world,query))
                    assert request['max_tokens']==config['generation_tokens']
                    assert item['question']==query['question'] and record['text']==item['response']
                    assert item['answer']==extract_answer(item['response'])
                    assert item['used_answer']==(item['answer'] if item['answer'] is not None else 'UNPARSED')
                    assert item['correct']==(item['answer']==canonical(query['answer']))
                    unparsed+=item['answer'] is None
            expected_rows=set()
            for model in config['models']:
                expected_rows.add((model,'zero',0))
                sources=['oracle',model]+([other for other in config['models'] if other!=model] if config['cross_teacher'] else [])
                expected_rows.update((model,source,n) for source in sources for n in config['budgets'] if n)
            assert len(case['rows'])==len(expected_rows) and {(r['model'],r['source'],r['budget']) for r in case['rows']}==expected_rows
            for row in case['rows']:
                model=row['model']; source=row['source']; n=row['budget']
                if source=='zero': answers=[]
                elif source=='oracle': answers=[item['answer'] for item in world['practice'][:n]]
                else: answers=[item['used_answer'] for item in case['generations'][source][:n]]
                assert row['answers']==answers and len(answers)==n
                scores=[]
                for evaluation in row['evaluations']:
                    record=response(evaluation['call']['call_id']); request=record['request']; evaluation_calls+=1
                    queries=[{key:item[key] for key in ('question','answer','id')} for item in evaluation['scores']]
                    assert request['model']==config['models'][model]['id']
                    assert request['messages']==messages(inference_prompt(world,answers,queries))
                    assert request['max_tokens']==config['inference_tokens']
                    assert record['text']==evaluation['response']
                    assert score(queries,record['text'])==evaluation['scores']
                    scores.extend(evaluation['scores'])
                assert scores==row['scores']
                assert [{key:item[key] for key in ('question','answer','id')} for item in scores]==world['tests']
                assert row['accuracy']==float(np.mean([item['correct'] for item in scores]))
                if source==model: identical+=answers==[item['answer'] for item in world['practice'][:n]]
        summaries[phase]={'worlds':len(cases),'model_worlds':len(cases)*len(config['models']),'generation_requests_checked':generation_calls,
                          'evaluation_requests_checked':evaluation_calls,'self_contexts_identical_to_oracle':identical,'unparsed_generation_slots_retained':unparsed}
    reserved={}; settled={}
    for line in (root/'api/ledger.jsonl').read_text().splitlines():
        event=json.loads(line); target=reserved if event['kind']=='reserve' else settled
        assert event['id'] not in target; target[event['id']]=Decimal(event['usd'])
    paid=sum(settled.values(),Decimal(0)); uncertain=sum((value for key,value in reserved.items() if key not in settled),Decimal(0))
    spending=json.loads((root/'api/spending.json').read_text())
    assert paid==Decimal(spending['paid_usd']) and uncertain==Decimal(spending['unresolved_reserved_usd'])
    assert paid+uncertain<=Decimal(spending['cap_usd'])
    assert all(Decimal(0)<=value<=reserved[key] for key,value in settled.items())
    result={'cohorts':summaries,'unique_api_records_checked':response.cache_info().currsize,'spending':spending,
            'checks':'Frozen current and archived sources; all planned worlds, conditions, budgets and models; independent world reconstruction; disjoint input partitions; generation blind to teacher and test targets; same final prompt template and inference allowance; only answer slots differ across source arms; unparsed samples retained without raw-rationale injection; exact scoring and ledger reconciliation.'}
    save_json(root/'verification.json',result); return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('root',type=Path); parser.add_argument('phases',nargs='+'); args=parser.parse_args()
    print(json.dumps(verify(args.root,args.phases),indent=2))


if __name__=='__main__': main()
