"""A valid injective teacher on exactly the completed palette-screen questions."""

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from strategy_task_search import register
from task_search_api import ROOT,LocalModel
from run_task_search import evaluate


def injective_answer(entry):
    puzzle=entry['metadata']['puzzle']; vertices=puzzle['vertices']; colors=puzzle['color_options']
    assert len(set(colors))>=len(vertices)
    return json.dumps(dict(zip(map(str,vertices),colors)),sort_keys=True)


async def main_async(args):
    root=ROOT.resolve(); original=root/args.source_phase; directory=root/args.phase; directory.mkdir(exist_ok=True); (directory/'evaluations').mkdir(exist_ok=True)
    preflight=json.loads((original/'preflight.json').read_text()); worlds=[]
    assert len(list((original/'evaluations').glob('*.json')))==sum(r['status']=='ready' for r in preflight)
    for row in preflight:
        if row['status']!='ready': continue
        world=json.loads((original/'worlds'/row['world']).read_text()); puzzle=world['practice'][0]['metadata']['puzzle']
        if len(puzzle['color_options'])>=len(puzzle['vertices']): worlds.append((row['world'],world))
    model=LocalModel(args.concurrency)
    manifest={'source_phase':args.source_phase,'phase':args.phase,'engine_fingerprint':model.fingerprint,'sample_counts':args.ns,'ready_worlds':[name for name,_ in worlds],
              'source_inference_manifest_sha256':hashlib.sha256((original/'inference_manifest.json').read_bytes()).hexdigest(),
              'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['palette_teacher_control.py','strategy_task_search.py','run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']},
              'intervention':'Only additional teacher answer strings change. Each vertex receives a distinct allowed color, which is valid by construction. The original four examples, rehearsal questions, tests, prompt, and decoding are unchanged. No generated sample is read, selected, or repaired.',
              'selection':'All completed six-vertex screen settings whose palette permits an injective coloring. Structural selection, without filtering by source effect.'}
    path=directory/'inference_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    snapshot=directory/'source'; snapshot.mkdir(exist_ok=True)
    for name in manifest['source_sha256']: (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    semaphore=asyncio.Semaphore(args.world_concurrency)
    async def one(name,world):
        path=directory/'evaluations'/name
        if path.exists(): return json.loads(path.read_text())
        async with semaphore:
            examples=[{**item,'answer':injective_answer(item)} for item in world['practice']]
            conditions=await asyncio.gather(*(evaluate(model,world,examples[:n],'teacher_injective',n) for n in args.ns))
            record={'spec':world['spec'],'seed':world['seed'],'examples':examples,'conditions':conditions}; path.write_text(json.dumps(record,indent=2)); return record
    summary=[]
    for future in tqdm(asyncio.as_completed([one(name,world) for name,world in worlds]),total=len(worlds),desc='Injective teacher control',mininterval=10):
        result=await future; summary.append({'spec':result['spec']['id'],'seed':result['seed'],'conditions':[{k:v for k,v in c.items() if k not in {'calls','predictions'}} for c in result['conditions']]})
        (directory/'summary.json').write_text(json.dumps(sorted(summary,key=lambda r:(r['spec'],r['seed'])),indent=2))
    await model.close(); print(json.dumps({'episodes':len(summary)},indent=2))


def main():
    register(); parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--source-phase',default='palette_screen'); parser.add_argument('--phase',default='palette_injective_control'); parser.add_argument('--ns',nargs='+',type=int,default=[2,8,32]); parser.add_argument('--concurrency',type=int,default=4); parser.add_argument('--world-concurrency',type=int,default=2); args=parser.parse_args(); asyncio.run(main_async(args))


if __name__=='__main__': main()
