"""Matched initial-context/rehearsal grid; all requests and episodes are resumable."""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

from decomposition_task_search import register,solution
from run_task_search import run_world
from task_search_api import LocalModel,ROOT
from task_search_data import dataset
from task_search_prompts import answer_prompt

PHASE='initial_count_grid'
INITIAL=[0,1,2,4,8,16]
REHEARSAL=[0,1,2,4,8,16,32]
SOURCES=['initial_count_task_search.py','decomposition_task_search.py','run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']


def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCES}


def prepare():
    root=ROOT.resolve(); directory=root/PHASE; (directory/'worlds').mkdir(parents=True,exist_ok=True)
    originals=sorted((root/'decomposition_confirmation/worlds').glob('product_decomposition_k3_d3__default__*.json'),key=lambda p:int(p.stem.rsplit('__',1)[1]))[:16]
    assert len(originals)==16
    design={'phase':PHASE,'initial_counts':INITIAL,'rehearsal_counts':REHEARSAL,'episodes':16,'tests_per_episode':16,
            'selection':'First 16 original three-factor confirmation episodes sorted by numeric seed; fixed before new inference. No outcome-based selection.',
            'design':'Same tests and rehearsal questions at every cell. Initial examples are nested prefixes, beginning with the original four. Additional initial pairs use unused subsequent dataset indices. Own answers are regenerated independently for each initial count; only that initial prefix and the assigned question are visible. No feedback, filtering, correction, or weight updates.',
            'interpretation':'Descriptive follow-up on an already selected task and a subset of the original confirmation cohort, not a new independent confirmation. The K=4 row reuses identical cached requests.',
            'original_world_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in originals},'source_sha256':hashes()}
    manifest=directory/'design.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==design
    else: manifest.write_text(json.dumps(design,indent=2))
    model=LocalModel(); records=[]
    for source in tqdm(originals,desc='Prepare matched initial-context grid'):
        original=json.loads(source.read_text()); data=dataset(original); supports=list(original['support']); all_entries=original['support']+original['practice']+original['tests']
        seen={e['question'] for e in all_entries}; start=1+max(e['dataset_index'] for e in all_entries)
        for index in range(start,1000):
            entry=data[index]
            if entry['question'] in seen: continue
            assert solution(entry['answer'],entry)[0]
            supports.append({**entry,'dataset_index':index}); seen.add(entry['question'])
            if len(supports)==max(INITIAL): break
        assert len(supports)==max(INITIAL)
        for k in INITIAL:
            world={**original,'support':supports[:k]}
            if k==4: assert world==original
            assert len({e['question'] for e in world['support']+world['practice']+world['tests']})==k+32+16
            assert all(solution(e['answer'],e)[0] for e in world['support']+world['practice']+world['tests'])
            name=f'k{k}__{source.name}'; path=directory/'worlds'/name
            if path.exists(): assert json.loads(path.read_text())==world
            else: path.write_text(json.dumps(world,indent=2))
            maximum=0
            for query in world['tests']:
                rendered=model.tokenizer.apply_chat_template([{'role':'user','content':answer_prompt(world,world['practice'],query)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                maximum=max(maximum,len(model.tokenizer.encode(rendered,add_special_tokens=False)))
            assert maximum+1024<=16384
            records.append({'world':name,'initial_count':k,'seed':world['seed'],'source_world':source.name,'world_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'maximum_teacher_prompt_tokens':maximum})
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
    for future in tqdm(asyncio.as_completed([one(r) for r in records]),total=len(records),desc='Initial-context grid',mininterval=10):
        completed.append(await future)
        (directory/'progress.json').write_text(json.dumps({'completed_worlds':len(completed),'total_worlds':len(records),'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))
    await model.close(); (directory/'complete.json').write_text(json.dumps({'completed_worlds':len(completed),'time_uk':datetime.now(ZoneInfo('Europe/London')).isoformat()},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('mode',choices=['prepare','run']); parser.add_argument('--concurrency',type=int,default=16); args=parser.parse_args(); register()
    if args.mode=='prepare': prepare()
    else: asyncio.run(run(args.concurrency))


if __name__=='__main__': main()
