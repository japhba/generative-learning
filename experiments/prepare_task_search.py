"""Preflight all procedural settings with explicit, recorded generation failures."""

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_data import EXCLUSIONS,catalog,final_prompt,seed_for

ROOT=Path(__file__).resolve().parents[1]/'results/task_search'


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',default='calibration'); parser.add_argument('--selection',type=Path)
    parser.add_argument('--worlds',type=int,default=1); parser.add_argument('--practice',type=int,default=8); parser.add_argument('--tests',type=int,default=8)
    parser.add_argument('--workers',type=int,default=6); parser.add_argument('--timeout',type=int,default=25); args=parser.parse_args()
    root=ROOT.resolve(); directory=root/args.phase; (directory/'worlds').mkdir(parents=True,exist_ok=True); (directory/'specs').mkdir(exist_ok=True); (directory/'preflight_logs').mkdir(exist_ok=True)
    specs=catalog() if args.selection is None else json.loads(args.selection.read_text())['specifications']
    model=json.loads((root/'runtime/model.json').read_text()); tokenizer=AutoTokenizer.from_pretrained(model['path'])
    for spec in specs: (directory/'specs'/f"{spec['id']}.json").write_text(json.dumps(spec,indent=2))
    config={'phase':args.phase,'worlds_per_spec':args.worlds,'practice_count':args.practice,'test_count':args.tests,'support_count':4,'max_teacher_prompt_tokens':11000,
            'generation_timeout_seconds':args.timeout,'reasoning_gym_version':importlib.metadata.version('reasoning-gym'),'specifications':specs,'excluded_task_families':EXCLUSIONS,
            'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['task_search_data.py','task_search_scoring.py']}}
    manifest=directory/'data_manifest.json'
    if manifest.exists(): assert json.loads(manifest.read_text())==config
    else: manifest.write_text(json.dumps(config,indent=2))
    environment={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def one(spec,index):
        seed=seed_for(spec,args.phase,index); key=spec['id']+'__'+str(seed); path=directory/'worlds'/f'{key}.json'; log=directory/'preflight_logs'/f'{key}.log'
        command=[sys.executable,str(Path(__file__).parent/'task_search_data.py'),'--spec',str(directory/'specs'/f"{spec['id']}.json"),'--output',str(path),'--seed',str(seed),'--practice',str(args.practice),'--tests',str(args.tests)]
        if path.exists(): return {'spec':spec['id'],'seed':seed,'status':'generated','world':path.name}
        with log.open('w') as stream:
            try: result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=environment,timeout=args.timeout)
            except subprocess.TimeoutExpired: return {'spec':spec['id'],'seed':seed,'status':'generation_timeout','log':log.name}
        return {'spec':spec['id'],'seed':seed,'status':'generated' if result.returncode==0 else 'generation_error','world':path.name,'log':log.name}
    records=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(one,spec,index) for spec in specs for index in range(args.worlds)]
        for future in tqdm(as_completed(futures),total=len(futures),desc='Task preflight',unit='world',mininterval=5):
            record=future.result()
            if record['status']=='generated':
                world=json.loads((directory/'worlds'/record['world']).read_text()); prompts=[final_prompt(world,world['practice'],world['tests'][i:i+4]) for i in range(0,len(world['tests']),4)]
                tokens=[len(tokenizer.encode(prompt)) for prompt in prompts]; record['max_teacher_prompt_tokens']=max(tokens)
                record['status']='ready' if max(tokens)<=11000 else 'context_too_long'
            records.append(record)
            (directory/'preflight.json').write_text(json.dumps(sorted(records,key=lambda r:(r['spec'],r['seed'])),indent=2))
    counts={status:sum(r['status']==status for r in records) for status in sorted({r['status'] for r in records})}
    print(json.dumps({'specifications':len(specs),'worlds':len(records),'statuses':counts},indent=2))


if __name__=='__main__': main()
