"""Fresh, source-only 2-reader by 3-source factorial ICL experiment."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
from random import Random
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

import decomposition_task_search
from nontrivial_task_runner import RULE,solution
from specificity_api import DIRECTORY,GEMMA,Gemma
from task_search_api import LocalModel,digest
from task_search_data import make_world,seed_for,specification
from task_search_prompts import answer_prompt
import witness_task_search

MODELS=['qwen','gemma']
SOURCES=MODELS+['teacher']
COUNTS=[0,2,8,32]
FILES=['SPECIFICITY_PROTOCOL.md','specificity_api.py','specificity_experiment.py','task_search_api.py','task_search_data.py','task_search_prompts.py','task_search_scoring.py','decomposition_task_search.py','nontrivial_task_runner.py','witness_task_search.py']


def now(): return datetime.now(ZoneInfo('Europe/London')).isoformat()
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}
def correct(answer,entry): return solution(answer,entry)[0] if 'operation' in entry['metadata'] else witness_task_search.valid(answer,entry)
def register(): decomposition_task_search.register();witness_task_search.register()


def prepare():
    register();DIRECTORY.mkdir(exist_ok=True)
    for folder in ['worlds','source','generations','evaluations']: (DIRECTORY/folder).mkdir(exist_ok=True)
    design={'time_uk':now(),'phase':'specificity_20260912','models':MODELS,'gemma':GEMMA,'settings':['factor_k0','factor_k4','bezout_k0'],'episodes':16,'tests_per_episode':12,'rehearsal_counts':COUNTS,'max_tokens':256,'generation_temperature':.7,'evaluation_temperature':0.,'source_sha256':hashes()}
    assert not (DIRECTORY/'design.json').exists(),'Protocol already frozen; run the resumable experiment instead.'
    records=[]
    for family,task in [('factor','product_decomposition_k3_d3'),('bezout','witness_bezout_l0')]:
        spec=specification(task,'default')
        for index in tqdm(range(16),desc='Prepare '+family):
            seed=seed_for(spec,'specificity_20260912',index);original=make_world(spec,seed,32,12,4 if family=='factor' else 0)
            if family=='factor':
                original={**original,**{key:[{**entry,'question':entry['question']+' '+RULE} for entry in original[key]] for key in ['support','practice','tests']}}
            assert all(correct(e['answer'],e) for key in ['support','practice','tests'] for e in original[key])
            for k in ([0,4] if family=='factor' else [0]):
                setting=f'{family}_k{k}';world={**original,'support':original['support'][:k],'setting':setting,'episode':index}
                name=f'{setting}__{index:02}.json';write(DIRECTORY/'worlds'/name,world)
                records.append({'name':name,'setting':setting,'episode':index,'sha256':digest(world)})
    design['worlds']=records;write(DIRECTORY/'design.json',design)
    for name in FILES: (DIRECTORY/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'settings':3,'worlds':48,'planned_generations':3072,'planned_evaluations':11520,'protocol_sha256':design['source_sha256']['SPECIFICITY_PROTOCOL.md']}))


async def run(stage,reader):
    design=json.loads((DIRECTORY/'design.json').read_text());assert design['source_sha256']==hashes()
    model=LocalModel(16) if reader=='qwen' else Gemma(16)
    inference_path=DIRECTORY/f'{reader}_inference.json'
    inference={'engine_fingerprint':model.fingerprint,'engine':model.engine,'design_sha256':digest(design),'max_tokens':256}
    if inference_path.exists(): assert json.loads(inference_path.read_text())==inference
    else: write(inference_path,inference)
    worlds={r['name']:json.loads((DIRECTORY/'worlds'/r['name']).read_text()) for r in design['worlds']}
    assert all(digest(worlds[r['name']])==r['sha256'] for r in design['worlds'])
    directory=DIRECTORY/('generations' if stage=='generate' else 'evaluations')/reader;directory.mkdir(exist_ok=True)
    jobs=[]
    if stage=='generate':
        for name,world in worlds.items():
            for index,entry in enumerate(world['practice']): jobs.append({'world':name,'index':index,'source':reader,'n':1})
    else:
        for name,world in worlds.items():
            for source,n in [('initial',0)]+[(s,n) for n in COUNTS[1:] for s in SOURCES]:
                for index in range(len(world['tests'])): jobs.append({'world':name,'index':index,'source':source,'n':n})
        for generator in MODELS: assert (DIRECTORY/f'generate_{generator}_complete.json').exists()
    Random(210921 if stage=='generate' else 210922).shuffle(jobs)
    semaphore=asyncio.Semaphore(32)
    async def one(job):
        world=worlds[job['world']];name=Path(job['world']).stem;index=job['index'];source=job['source'];n=job['n']
        path=directory/(f'{name}__{index:02}.json' if stage=='generate' else f'{name}__{source}__n{n}__{index:02}.json')
        if path.exists(): return
        async with semaphore:
            if stage=='generate':
                entry=world['practice'][index];prompt=answer_prompt(world,[],entry)
                seed=seed_for(world['spec'],f"{world['seed']}|specificity|generation|k{len(world['support'])}",index);temperature=.7
            else:
                entry=world['tests'][index]
                if source=='initial': examples=[]
                elif source=='teacher': examples=world['practice'][:n]
                else:
                    examples=[{'question':item['question'],'answer':json.loads((DIRECTORY/'generations'/source/f'{name}__{i:02}.json').read_text())['answer']} for i,item in enumerate(world['practice'][:n])]
                prompt=answer_prompt(world,examples,entry);seed=world['seed']+index;temperature=0.
            call=await model.call(prompt,None,seed,temperature=temperature,max_tokens=256)
            assert call['status']!='context_overflow',(job,call)
            answer=call['value']['answer'];record={**job,'reader':reader,'answer':answer,'call':call}
            # Validity is intentionally computed only in the post-inference analyzer.
            write(path,record)
    completed=0
    for future in tqdm(asyncio.as_completed([one(j) for j in jobs]),total=len(jobs),desc=f'{stage} {reader}',mininterval=10):
        await future;completed+=1
        if completed%32==0: write(DIRECTORY/f'{stage}_{reader}_progress.json',{'completed':completed,'total':len(jobs),'time_uk':now()})
    await model.close();write(DIRECTORY/f'{stage}_{reader}_complete.json',{'completed':completed,'time_uk':now()})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','generate','evaluate']);parser.add_argument('--reader',choices=MODELS);args=parser.parse_args()
    if args.stage=='prepare': prepare()
    else: asyncio.run(run(args.stage,args.reader))


if __name__=='__main__': main()
