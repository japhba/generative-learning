"""Verify final cohorts, source hashes, cached prompts, data isolation, and spending offline."""

from collections import Counter
from decimal import Decimal
from functools import cache
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from tqdm.auto import tqdm

from self_icl import generation_prompt, prediction_prompt
from self_icl_api import save_json


def main():
    root=Path(sys.argv[1])
    tasks=json.loads((root/'inputs/tasks.json').read_text())
    spec=importlib.util.spec_from_file_location('upstream_self_icl_prompt',root/'inputs/upstream_prompt.py')
    upstream=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=upstream
    spec.loader.exec_module(upstream)
    query='Held-out question'; description='Task description'; shots=[('Practice question','Practice answer')]
    for k in (1,3,16):
        prompt=upstream.StreamPrompt(description,query,k,[upstream.Shot(*shot) for shot in shots])
        assert generation_prompt(description,query,k)==prompt.gen_demo_inputs(diversity=True)
        for cot in (False,True): assert prediction_prompt(description,query,shots,cot)==prompt.gen_prediction(cot=cot)

    @cache
    def response(call_id): return json.loads((root/'api'/f'{call_id}.json').read_text())

    verified={}
    for phase in ('historical','scaling','scaling_cot'):
        directory=root/phase
        manifest=json.loads((directory/'manifest.json').read_text())
        assert hashlib.sha256((root/'inputs/tasks.json').read_bytes()).hexdigest()==manifest['inputs_sha256']
        for name,digest in manifest['source_sha256'].items():
            assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==digest
        expected={(model,task,index) for model in manifest['models'] for task in manifest['tasks'] for index in manifest['case_indices'][task]}
        cases=[json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
        assert len(cases)==len(expected)
        assert {(case['model'],case['task'],case['index']) for case in cases}==expected
        counts=Counter()
        for case in tqdm(cases,desc=phase,unit='case'):
            task=tasks[case['task']]
            assert case['query']==task['examples'][case['index']]['input']
            assert case['target']==task['examples'][case['index']]['target']
            teachers={task['examples'][index]['input'] for index in task['permutation'][:32]}
            teachers.update(demo['Q'] for demo in task['official_demos'])
            assert case['query'] not in teachers
            for generation in case['generations']:
                record=response(generation['generation']['call_id'])
                assert record['request']['prompt']==generation_prompt(task['description'],case['query'],generation['budget'])
                assert record['text']==generation['response']
            for row in case['rows']:
                counts['rows']+=1
                if row['pipeline_failure']:
                    assert not row['correct'] and row['evaluation'] is None and not row['demonstrations']
                    counts['count_failures']+=1
                    continue
                cot=row['mode']=='cot'
                prefix=' (' if not cot and case['target'].startswith('(') else ''
                evaluated=response(row['evaluation']['call_id'])
                assert evaluated['request']['prompt']==prediction_prompt(task['description'],case['query'],row['demonstrations'],cot)+prefix
                assert evaluated['text']==row['response']
                if row['condition']=='self':
                    assert len(row['demonstrations'])==row['budget']
                    for (question,answer),call in zip(row['demonstrations'],row['preparation'][1:],strict=True):
                        labelled=response(call['call_id'])
                        assert labelled['request']['prompt']==prediction_prompt(task['description'],question,[],cot)+prefix
                        assert answer==((prefix.strip() if prefix else '')+labelled['text']).strip()
                        counts['independent_label_checks']+=1
            counts['cases']+=1
        verified[phase]=dict(counts)

    reserved,settled={},{}
    for line in (root/'api/ledger.jsonl').read_text().splitlines():
        event=json.loads(line)
        destination=reserved if event['event']=='reserve' else settled
        assert event['id'] not in destination
        destination[event['id']]=Decimal(event['usd'])
    assert all(Decimal('0')<=amount<=reserved[key] for key,amount in settled.items())
    paid=sum(settled.values(),Decimal('0'))
    unresolved=sum((amount for key,amount in reserved.items() if key not in settled),Decimal('0'))
    spending=json.loads((root/'api/spending.json').read_text())
    assert paid==Decimal(spending['paid_usd'])
    assert unresolved==Decimal(spending['unresolved_reserved_usd'])
    assert paid+unresolved<=Decimal(spending['cap_usd'])
    result={'cohorts':verified,'unique_checked_api_records':response.cache_info().currsize,'spending':spending,
            'checks':'Released prompt identity; source/data hashes; complete planned case sets; real/test text isolation; cached generation, independent label and final prompts; retained pipeline failures; reconciled spending ledger.'}
    save_json(root/'verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
