"""Sequential unjudged rehearsal on the same assigned questions and final prompts."""

import argparse
import asyncio
import hashlib
import importlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from run_task_search import evaluate
from task_search_api import ROOT,LocalModel,digest
from task_search_data import seed_for
from task_search_prompts import answer_prompt


def atomic_json(path,value):
    temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(value,indent=2)); temporary.replace(path)


async def run_world(model,world,directory,name,ns):
    path=directory/'generation_pools'/name; samples=json.loads(path.read_text()) if path.exists() else []; assert len(samples)<=max(ns)
    for i in range(len(samples),max(ns)):
        question=world['practice'][i]; previous=[s['example'] for s in samples]
        call=await model.call(answer_prompt(world,previous,question),None,seed_for(world['spec'],f"{world['seed']}|fixed",i),temperature=.7,max_tokens=1024)
        example={'question':question['question'],'answer':call['value']['answer'] if call['status']=='ok' else ''}
        samples.append({'index':i,'conditioning_sample_count':i,'example':example,'call':call}); atomic_json(path,samples)
    jobs=[evaluate(model,world,[],'zero',0)]
    for n in ns:
        if n==0: continue
        jobs.append(evaluate(model,world,world['practice'][:n],'teacher',n))
        jobs.append(evaluate(model,world,[s['example'] for s in samples[:n]],'self_fixed',n))
    conditions=await asyncio.gather(*jobs)
    return {'spec':world['spec'],'seed':world['seed'],'generations':{'self_fixed':samples},'conditions':conditions}


async def run(args,registry):
    root=ROOT.resolve(); original=root/args.source_phase; directory=root/args.phase; directory.mkdir(exist_ok=True)
    for name in ['evaluations','generation_pools','source']: (directory/name).mkdir(exist_ok=True)
    for name in ['worlds','specs']:
        path=directory/name
        if path.exists(): assert path.resolve()==(original/name).resolve()
        else: path.symlink_to(original/name,target_is_directory=True)
    preflight=json.loads((original/'preflight.json').read_text()); ready=[r for r in preflight if r['status']=='ready']; model=LocalModel(args.concurrency)
    names=['online_task_search.py','run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py',Path(registry.__file__).name]
    manifest={'phase':args.phase,'source_phase':args.source_phase,'registry':args.registry,'engine_fingerprint':model.fingerprint,'arms':['teacher','self_fixed'],'sample_counts':args.ns,
              'generation_conditioning':'Initial examples plus all earlier unjudged own answers, then the next assigned rehearsal question. No future question, test, teacher rehearsal answer, or correctness signal.',
              'generation_seed_rule':'Same per-question random seeds as independent generation; the first generation has an identical cached request.','generation_temperature':.7,'evaluation_temperature':0.,'enable_thinking':False,'max_output_tokens':1024,
              'ready_worlds':[r['world'] for r in ready],'source_preflight_sha256':hashlib.sha256((original/'preflight.json').read_bytes()).hexdigest(),'source_data_manifest_sha256':hashlib.sha256((original/'data_manifest.json').read_bytes()).hexdigest(),
              'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}}
    path=directory/'inference_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: atomic_json(path,manifest)
    (directory/'preflight.json').write_bytes((original/'preflight.json').read_bytes())
    for name in names: (directory/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    semaphore=asyncio.Semaphore(args.world_concurrency)
    async def one(row):
        path=directory/'evaluations'/row['world']
        if path.exists(): return json.loads(path.read_text())
        async with semaphore:
            world=json.loads((directory/'worlds'/row['world']).read_text()); assert len(world['practice'])>=max(args.ns)
            result=await run_world(model,world,directory,row['world'],args.ns); atomic_json(path,result); return result
    summary=[]
    for future in tqdm(asyncio.as_completed([one(row) for row in ready]),total=len(ready),desc=args.phase,mininterval=10):
        result=await future; summary.append({'task':result['spec']['task'],'spec':result['spec']['id'],'seed':result['seed'],'conditions':[{k:v for k,v in c.items() if k not in {'predictions','calls'}} for c in result['conditions']]})
        atomic_json(directory/'summary.json',sorted(summary,key=lambda r:(r['spec'],r['seed'])))
    await model.close(); print(json.dumps({'episodes':len(summary)},indent=2))


def verify(args):
    root=ROOT.resolve(); directory=root/args.phase; manifest=json.loads((directory/'inference_manifest.json').read_text()); original=root/manifest['source_phase']; api=root/'api'/manifest['engine_fingerprint']; requests=0
    assert hashlib.sha256((original/'preflight.json').read_bytes()).hexdigest()==manifest['source_preflight_sha256']
    assert hashlib.sha256((original/'data_manifest.json').read_bytes()).hexdigest()==manifest['source_data_manifest_sha256']
    for name,expected in manifest['source_sha256'].items(): assert hashlib.sha256((directory/'source'/name).read_bytes()).hexdigest()==expected
    assert sorted(manifest['ready_worlds'])==sorted(p.name for p in (directory/'evaluations').glob('*.json'))
    def check(call,prompt,seed,temperature):
        nonlocal requests
        request={'model':'qwen35-9b','messages':[{'role':'user','content':prompt}],'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':1024,'seed':seed,'chat_template_kwargs':{'enable_thinking':False}}
        assert digest(request)==call['key']; requests+=1
        if call['status']=='context_overflow': return ''
        raw=json.loads((api/(call['key']+'.json')).read_text()); assert raw['request']==request
        return raw['response']['choices'][0]['message']['content'].strip() if call['status']=='ok' else ''
    for name in tqdm(manifest['ready_worlds'],desc='Verify sequential rehearsal',mininterval=10):
        world=json.loads((directory/'worlds'/name).read_text()); result=json.loads((directory/'evaluations'/name).read_text()); samples=result['generations']['self_fixed']; assert len(samples)==max(manifest['sample_counts'])
        for i,sample in enumerate(samples):
            assert sample['index']==sample['conditioning_sample_count']==i; assert sample['example']['question']==world['practice'][i]['question']
            answer=check(sample['call'],answer_prompt(world,[s['example'] for s in samples[:i]],world['practice'][i]),seed_for(world['spec'],f"{world['seed']}|fixed",i),.7); assert sample['example']['answer']==answer
        for condition in result['conditions']:
            arm,n=condition['arm'],condition['n']; examples=[] if arm=='zero' else world['practice'][:n] if arm=='teacher' else [s['example'] for s in samples[:n]]
            for i,(query,call) in enumerate(zip(world['tests'],condition['calls'])):
                answer=check(call,answer_prompt(world,examples,query),world['seed']+i,0.)
                if condition['status']=='ok': assert condition['predictions'][i]['answer']==answer
    record={'episodes':len(manifest['ready_worlds']),'reconstructed_requests':requests,'only_past_own_rehearsal_in_generation_context':True,'teacher_and_final_questions_unchanged':True}; atomic_json(directory/'verification.json',record); print(json.dumps(record,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('mode',choices=['run','verify']); parser.add_argument('--source-phase',default='palette_screen'); parser.add_argument('--phase',default='palette_online_screen'); parser.add_argument('--registry',choices=['palette','frame'],default='palette'); parser.add_argument('--ns',nargs='+',type=int,default=[0,2,8,32]); parser.add_argument('--concurrency',type=int,default=4); parser.add_argument('--world-concurrency',type=int,default=4); args=parser.parse_args()
    registry=importlib.import_module({'palette':'strategy_task_search','frame':'framed_task_search'}[args.registry]); registry.register(); assert args.ns==sorted(set(args.ns)) and args.ns[0]==0
    if args.mode=='run': asyncio.run(run(args,registry))
    else: verify(args)


if __name__=='__main__': main()
