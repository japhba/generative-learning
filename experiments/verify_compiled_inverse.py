"""Offline audit of final modular inverse cohorts and their cached API requests."""

import argparse
from decimal import Decimal
from functools import cache
import hashlib
import json
import math
from pathlib import Path

from tqdm.auto import tqdm

from compiled_inverse_icl import FORWARD_TOKENS,decode_prompt,generation_prompt,make_world,messages,parse_code,score,table
from self_icl_api import save_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('phases',nargs='+')
    args=parser.parse_args()
    root=args.root

    @cache
    def response(call_id):
        record=json.loads((root/'api'/f'{call_id}.json').read_text())
        assert hashlib.sha256(json.dumps(record['request'],sort_keys=True).encode()).hexdigest()==call_id
        assert record['request']['temperature']==0 and record['request']['reasoning']=={'enabled':False}
        return record

    summaries={}
    for phase in args.phases:
        directory=root/phase
        manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']
        for filename,digest in manifest['source_sha256'].items(): assert hashlib.sha256((directory/'source'/filename).read_bytes()).hexdigest()==digest
        for filename,digest in manifest['source_sha256'].items(): assert hashlib.sha256((Path(__file__).parent/filename).read_bytes()).hexdigest()==digest
        cases=[json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
        expected={(model,seed) for model in config['models'] for seed in config['seeds']}
        assert len(cases)==len(expected) and {(case['model'],case['world']['seed']) for case in cases}==expected
        checked=0; same_ranked=0; same_iid=0; generated_codes=0; correct_codes=0; overages=0
        for case in tqdm(cases,desc=phase,unit='encoder'):
            world=case['world']
            assert world==make_world(world['seed'])
            expected_rows={('zero',0)}|{(condition,n) for n in config['budgets'] if n>0 for condition in ('generated','teacher_iid','teacher_ranked','generated_iid')}|{('long_direct',n) for n in config['long_budgets']}
            assert len(case['rows'])==len(expected_rows) and {(row['condition'],row['budget']) for row in case['rows']}==expected_rows
            assert [item['label'] for item in case['generation']]==list(range(1,max(config['budgets'])+1))
            assert [item['label'] for item in case['iid_generation']]==world['teacher_labels'][:max(config['budgets'])]
            for item in case['generation']+case['iid_generation']:
                record=response(item['call']['call_id'])
                assert record['request']['messages']==messages(generation_prompt(world,item['label']))
                assert record['request']['max_tokens']==FORWARD_TOKENS
                assert item['response']==record['text'] and item['code']==parse_code(record['text'])
                assert item['correct_code']==(world['a']*item['label']+world['b'])%world['modulus']
                assert item['correct']==(item['code']==item['correct_code'])
            generated_codes+=len(case['generation']); correct_codes+=sum(item['correct'] for item in case['generation'])
            for row in case['rows']:
                condition=row['condition']; n=row['budget']
                if condition in ('zero','long_direct'): expected_examples=''
                elif condition=='teacher_iid': expected_examples=table(world,world['teacher_labels'][:n])
                elif condition=='teacher_ranked': expected_examples=table(world,range(1,n+1))
                else:
                    history=case['generation'][:n] if condition=='generated' else case['iid_generation'][:n]
                    expected_examples='\n'.join(f"Code {item['code']} -> label {item['label']}" if item['code'] is not None else f"Forward computation for label {item['label']}: {item['response']}" for item in history)
                assert row['examples']==expected_examples
                records=[]
                for evaluation in row['evaluations']:
                    record=response(evaluation['call']['call_id'])
                    queries=evaluation['scores']
                    assert record['request']['messages']==messages(decode_prompt(world,expected_examples,queries,condition=='long_direct'))
                    expected_limit=512
                    if condition=='long_direct':
                        tokens=sum(item['call']['prompt_tokens']+item['call']['completion_tokens'] for item in case['generation'][:n])
                        expected_limit+=math.ceil(tokens/len(row['evaluations']))
                    assert record['request']['max_tokens']==expected_limit
                    assert record['text']==evaluation['response']
                    assert score(queries,record['text'])==queries
                    assert record['request']['model']==config['models'][case['model']]['id']
                    records.extend(queries); checked+=1
                    overages+=record['reported_output_tokens_over_allowance']>0
                assert records==row['scores']
                assert [{key:item[key] for key in ('label','code','weight','id')} for item in records]==world['queries']
                assert abs(row['risk']-sum(item['weight']*(not item['correct']) for item in records))<1e-12
                if condition=='generated': same_ranked+=row['examples']==table(world,range(1,n+1))
                if condition=='generated_iid': same_iid+=row['examples']==table(world,world['teacher_labels'][:n])
        summaries[phase]={'cases':len(cases),'evaluation_requests_checked':checked,'ranked_generated_codes':generated_codes,'correct_ranked_codes':correct_codes,
                          'ranked_contexts_identical_to_teacher':same_ranked,'iid_contexts_identical_to_teacher':same_iid,'reported_token_overages':overages}

    reserved,settled={},{}
    for line in (root/'api/ledger.jsonl').read_text().splitlines():
        event=json.loads(line); values=reserved if event['kind']=='reserve' else settled
        assert event['id'] not in values
        values[event['id']]=Decimal(event['usd'])
    paid=sum(settled.values(),Decimal(0)); uncertain=sum((amount for key,amount in reserved.items() if key not in settled),Decimal(0))
    spending=json.loads((root/'api/spending.json').read_text())
    assert paid==Decimal(spending['paid_usd']) and uncertain==Decimal(spending['unresolved_reserved_usd'])
    assert paid+uncertain<=Decimal(spending['cap_usd'])
    assert all(Decimal(0)<=amount<=reserved[key] for key,amount in settled.items())
    result={'cohorts':summaries,'unique_checked_api_records':response.cache_info().currsize,'spending':spending,
            'checks':'Complete planned cohorts; frozen source hashes; exact request hashes; target-blind forward requests; unchanged erroneous codes; control contents; exact numeric scoring; reconstructed token allowances; spending reconciliation.'}
    save_json(root/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
