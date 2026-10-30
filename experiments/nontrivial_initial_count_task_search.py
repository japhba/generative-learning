"""Matched initial-context/rehearsal grid; all requests and episodes are resumable."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

from nontrivial_task_runner import solution,run_world,RULE
from task_search_api import LocalModel,ROOT
from task_search_prompts import answer_prompt

PHASE='nontrivial_initial_count_grid'
INITIAL=[0,1,2,4,8,16]
REHEARSAL=[0,1,2,4,8,16,32]
SOURCES=['nontrivial_initial_count_task_search.py','nontrivial_task_runner.py','decomposition_task_search.py','run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']


def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCES}


def prepare():
    root=ROOT.resolve(); directory=root/PHASE; (directory/'worlds').mkdir(parents=True,exist_ok=True)
    originals=sorted((root/'initial_count_grid/worlds').glob('*.json'))
    assert len(originals)==96
    design={'phase':PHASE,'initial_counts':INITIAL,'rehearsal_counts':REHEARSAL,'episodes':16,'tests_per_episode':16,
            'rule':RULE,'selection':'Same complete 16-episode grid as initial_count_grid, fixed before new inference.',
            'design':'Append the identical constraint to every support, rehearsal, and test question. All targets, teacher answers, seeds, source-arm formatting, and decoders remain unchanged. Regenerate own rehearsal answers independently for each K, with no correctness feedback, filtering, repairs, or weight updates.',
            'interpretation':'Descriptive matched intervention on the previously selected task and episodes, not an independent confirmation. Every own generation and final prediction is requested anew under the explicit constraint.',
            'original_world_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},'source_sha256':hashes()}
    manifest=directory/'design.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==design
    else: manifest.write_text(json.dumps(design,indent=2))
    model=LocalModel(); records=[]
    for source in tqdm(originals,desc='Prepare constrained grid'):
        original=json.loads(source.read_text()); k=len(original['support'])
        world={**original,**{key:[{**entry,'question':entry['question']+' '+RULE} for entry in original[key]] for key in ['support','practice','tests']}}
        assert all(solution(e['answer'],e)[0] for e in world['support']+world['practice']+world['tests']), 'An unchanged teacher answer violates the new rule.'
        name=source.name; path=directory/'worlds'/name
        if path.exists(): assert json.loads(path.read_text())==world
        else: path.write_text(json.dumps(world,indent=2))
        maximum=0
        for query in world['tests']:
            rendered=model.tokenizer.apply_chat_template([{'role':'user','content':answer_prompt(world,world['practice'],query)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            maximum=max(maximum,len(model.tokenizer.encode(rendered,add_special_tokens=False)))
        assert maximum+1024<=16384
        records.append({'world':name,'initial_count':k,'seed':world['seed'],'source_world':source.name.split('__',1)[1],'world_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'maximum_teacher_prompt_tokens':maximum})
    (directory/'preflight.json').write_text(json.dumps(records,indent=2)); asyncio.run(model.close())
    print(json.dumps({'worlds':len(records),'test_queries_per_cell':256,'initial_counts':INITIAL,'rehearsal_counts':REHEARSAL}))


async def run(concurrency):
    directory=ROOT.resolve()/PHASE; design=json.loads((directory/'design.json').read_text()); assert design['source_sha256']==hashes()
    records=json.loads((directory/'preflight.json').read_text()); model=LocalModel(concurrency)
    config={'phase':PHASE,'engine_fingerprint':model.fingerprint,'arms':['teacher','self_fixed'],'sample_counts':REHEARSAL,'generation_temperature':.7,'evaluation_temperature':0.,'enable_thinking':False,'max_output_tokens':1024,'source_sha256':hashes(),'design_sha256':hashlib.sha256((directory/'design.json').read_bytes()).hexdigest()}
    manifest=directory/'inference_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==config
    else: manifest.write_text(json.dumps(config,indent=2))
    snapshot=directory/'source'; snapshot.mkdir(exist_ok=True)
    for name in SOURCES: (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    output=directory/'evaluations'; output.mkdir(exist_ok=True); queue=asyncio.Semaphore(16)
    async def one(record):
        path=output/record['world']
        if path.exists(): return record['world']
        async with queue:
            world_path=directory/'worlds'/record['world']; assert hashlib.sha256(world_path.read_bytes()).hexdigest()==record['world_sha256']
            world=json.loads(world_path.read_text()); result=await run_world(model,world,config)
            assert all(c['status']=='ok' for c in result['conditions']), 'A grid cell exceeds context capacity; retain all evidence and redesign before interpretation.'
            result={'initial_count':record['initial_count'],'source_world':record['source_world'],**result}
            temporary=path.with_suffix('.tmp'); temporary.write_text(json.dumps(result,indent=2)); temporary.replace(path)
            return record['world']
    completed=[]
    for future in tqdm(asyncio.as_completed([one(r) for r in records]),total=len(records),desc='Constrained initial-context grid',mininterval=10):
        completed.append(await future)
        (directory/'progress.json').write_text(json.dumps({'completed_worlds':len(completed),'total_worlds':len(records),'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))
    await model.close(); (directory/'complete.json').write_text(json.dumps({'completed_worlds':len(completed),'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','run']); parser.add_argument('--concurrency',type=int,default=16); args=parser.parse_args()
    if args.mode=='prepare': prepare()
    else: asyncio.run(run(args.concurrency))


if __name__=='__main__': main()
