"""Valid teacher answers copied in input notation, with all learner settings matched."""

import argparse
import asyncio
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm

from task_search_api import ROOT,LocalModel
from run_task_search import evaluate


def copied_answer(entry):
    candidates=entry['metadata']['formatted_candidates']; choose={'largest':max,'smallest':min}[entry['metadata']['size']]
    answer=choose(candidates,key=lambda value:Decimal(value.replace(',','')))
    assert abs(float(answer.replace(',',''))-entry['metadata']['solution'])<1e-2
    return answer


async def main_async(args):
    root=ROOT.resolve(); original=root/args.source_phase; directory=root/args.phase; directory.mkdir(exist_ok=True); (directory/'evaluations').mkdir(exist_ok=True)
    records=json.loads((original/'preflight.json').read_text()); ready=[r for r in records if r['status']=='ready']; model=LocalModel(args.concurrency)
    manifest={'source_phase':args.source_phase,'phase':args.phase,'engine_fingerprint':model.fingerprint,'sample_counts':args.ns,'primary_endpoint':8,
              'source_inference_manifest_sha256':hashlib.sha256((original/'inference_manifest.json').read_bytes()).hexdigest(),
              'code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['number_format_copy_control.py','run_task_search.py','task_search_api.py','task_search_prompts.py']},
              'intervention':'Only the additional teacher answer strings change: copy the winning candidate in its displayed notation. The four initial examples, assigned questions, tests, and final prompt/decoding settings are unchanged. No model-generated sample is read or repaired.'}
    path=directory/'inference_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    snapshot=directory/'source'; snapshot.mkdir(exist_ok=True)
    for name in manifest['code_sha256']: (snapshot/name).write_bytes((Path(__file__).parent/name).read_bytes())
    semaphore=asyncio.Semaphore(args.world_concurrency)
    async def one(record):
        output=directory/'evaluations'/record['world']
        if output.exists(): return json.loads(output.read_text())
        async with semaphore:
            world=json.loads((original/'worlds'/record['world']).read_text()); assert world['spec']['task']=='number_format'
            examples=[{**item,'answer':copied_answer(item)} for item in world['practice']]
            conditions=await asyncio.gather(*(evaluate(model,world,examples[:n],'teacher_copy',n) for n in args.ns))
            result={'spec':world['spec'],'seed':world['seed'],'examples':examples,'conditions':conditions}; output.write_text(json.dumps(result,indent=2)); return result
    summary=[]
    for future in tqdm(asyncio.as_completed([one(record) for record in ready]),total=len(ready),desc='Teacher notation control',mininterval=10):
        result=await future; summary.append({'spec':result['spec']['id'],'seed':result['seed'],'conditions':[{k:v for k,v in c.items() if k not in {'calls','predictions'}} for c in result['conditions']]})
        (directory/'summary.json').write_text(json.dumps(sorted(summary,key=lambda x:x['seed']),indent=2))
    await model.close(); print(json.dumps({'episodes':len(summary)},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--source-phase',default='number_format_confirmation'); parser.add_argument('--phase',default='number_format_copy_control'); parser.add_argument('--ns',nargs='+',type=int,default=[2,8,32]); parser.add_argument('--concurrency',type=int,default=8); parser.add_argument('--world-concurrency',type=int,default=2); args=parser.parse_args(); asyncio.run(main_async(args))


if __name__=='__main__': main()
