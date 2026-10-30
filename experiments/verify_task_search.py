"""Verify raw request provenance and the source-only intervention after inference."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from task_search_api import ROOT,digest,schema
from task_search_data import free_prompt,seed_for
from task_search_prompts import answer_prompt


def verify_copy_control(root,phase):
    from number_format_copy_control import copied_answer
    directory=root/phase; manifest=json.loads((directory/'inference_manifest.json').read_text()); source=root/manifest['source_phase']; api=root/'api'/manifest['engine_fingerprint']
    assert hashlib.sha256((source/'inference_manifest.json').read_bytes()).hexdigest()==manifest['source_inference_manifest_sha256']
    for name,expected in manifest['code_sha256'].items(): assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
    names=[r['world'] for r in json.loads((source/'preflight.json').read_text()) if r['status']=='ready']; assert sorted(names)==sorted(p.name for p in (directory/'evaluations').glob('*.json')); requests=0
    for name in tqdm(names,desc='Verify teacher notation control',mininterval=10):
        world=json.loads((source/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text()); expected=[{**entry,'answer':copied_answer(entry)} for entry in world['practice']]
        assert result['examples']==expected and result['seed']==world['seed']; assert {c['n'] for c in result['conditions']}==set(manifest['sample_counts'])
        for condition in result['conditions']:
            assert condition['status']=='ok' and condition['arm']=='teacher_copy'
            for i,(query,call,prediction) in enumerate(zip(world['tests'],condition['calls'],condition['predictions'])):
                request={'model':'qwen35-9b','messages':[{'role':'user','content':answer_prompt(world,expected[:condition['n']],query)}],'temperature':0.,'top_p':1.,'max_tokens':1024,'seed':world['seed']+i,'chat_template_kwargs':{'enable_thinking':False}}
                assert digest(request)==call['key']; raw=json.loads((api/(call['key']+'.json')).read_text()); assert raw['request']==request
                assert prediction['answer']==raw['response']['choices'][0]['message']['content'].strip(); requests+=1
    summary={'phase':phase,'episodes':len(names),'verified_requests':requests,'valid_teacher_notation_change_only':True}; (directory/'verification.json').write_text(json.dumps(summary,indent=2)); return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phases',nargs='+',required=True); args=parser.parse_args(); root=ROOT.resolve(); summaries=[]
    for phase in args.phases:
        if phase=='number_format_copy_control': summaries.append(verify_copy_control(root,phase)); continue
        directory=root/phase; manifest=json.loads((directory/'inference_manifest.json').read_text()); api=root/'api'/manifest['engine_fingerprint']
        assert digest(json.loads((api/'engine.json').read_text()))==manifest['engine_fingerprint']
        for name,expected in manifest['source_sha256'].items(): assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
        preflight=json.loads((directory/'preflight.json').read_text()); names=[r['world'] for r in preflight if r['status']=='ready']
        assert sorted(names)==sorted(p.name for p in (directory/'evaluations').glob('*.json')); assert names==manifest['ready_worlds']
        counts=Counter(); keys=set()
        def verify_call(call,prompt,seed,temperature,output_schema=None):
            request={'model':'qwen35-9b','messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':1024,'seed':seed,'chat_template_kwargs':{'enable_thinking':False}}
            if output_schema is not None: request['response_format']={'type':'json_schema','json_schema':{'name':'result','strict':True,'schema':output_schema}}
            assert digest(request)==call['key']; counts[call['status']]+=1
            if call['status']=='context_overflow': return None
            raw=json.loads((api/(call['key']+'.json')).read_text()); assert raw['request']==request; assert raw['response']['usage']==call['usage']; keys.add(call['key'])
            return raw['response']['choices'][0]
        for name in tqdm(names,desc='Verify '+phase,mininterval=10):
            world=json.loads((directory/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text()); assert result['seed']==world['seed'] and result['spec']==world['spec']
            questions=[' '.join(x['question'].split()) for x in world['support']+world['practice']+world['tests']]; assert len(questions)==len(set(questions))
            for arm,generations in result['generations'].items():
                assert len(generations)==max(manifest['sample_counts'])
                for i,generation in enumerate(generations):
                    assert generation['index']==i
                    if arm=='self_fixed':
                        prompt=answer_prompt(world,[],world['practice'][i]); output_schema=None; seed=seed_for(world['spec'],f"{world['seed']}|fixed",i)
                        assert generation['example']['question']==world['practice'][i]['question']
                    else: prompt=free_prompt(world); output_schema=schema(['question','answer']); seed=seed_for(world['spec'],f"{world['seed']}|free",i)
                    choice=verify_call(generation['call'],prompt,seed,.7,output_schema)
                    if generation['call']['status']=='ok':
                        value={'answer':choice['message']['content'].strip()} if arm=='self_fixed' else json.loads(choice['message']['content'])
                        for field,content in value.items(): assert generation['example'][field]==content
                    else: assert generation['example']['answer']==''
            expected={('zero',0)}|{(arm,n) for arm in manifest['arms'] for n in manifest['sample_counts'] if n}
            assert {(c['arm'],c['n']) for c in result['conditions']}==expected
            for condition in result['conditions']:
                arm,n=condition['arm'],condition['n']
                examples=[] if arm=='zero' else world['practice'][:n] if arm=='teacher' else [g['example'] for g in result['generations'][arm][:n]]
                assert len(examples)==n and len(condition['calls'])==len(world['tests'])
                for i,(query,call) in enumerate(zip(world['tests'],condition['calls'])):
                    choice=verify_call(call,answer_prompt(world,examples,query),world['seed']+i,0.)
                    if condition['status']=='ok':
                        prediction=condition['predictions'][i]; assert prediction['id']==query['id'] and prediction['api_key']==call['key']
                        assert prediction['answer']==(choice['message']['content'].strip() if call['status']=='ok' else '')
            counts['episodes']+=1
        summary={'phase':phase,'counts':dict(counts),'unique_requests':len(keys),'manifest_sha256':hashlib.sha256((directory/'inference_manifest.json').read_bytes()).hexdigest()}
        (directory/'verification.json').write_text(json.dumps(summary,indent=2)); summaries.append(summary)
    print(json.dumps(summaries,indent=2))


if __name__=='__main__': main()
