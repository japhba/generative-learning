"""Freeze disjoint constructive-task worlds and check every teacher prompt budget."""
import argparse
import hashlib
import json
from pathlib import Path

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_api import ROOT
from task_search_data import make_world,seed_for,specification
from task_search_prompts import answer_prompt

SOURCES=['witness_task_search.py','prepare_witness_zero_task_search.py','witness_zero_task_search.py','run_task_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']


def main(names):
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True);parser.add_argument('--selection',type=Path);parser.add_argument('--worlds',type=int,default=1);parser.add_argument('--practice',type=int,default=32);parser.add_argument('--tests',type=int,default=8);args=parser.parse_args()
    root=ROOT.resolve();directory=root/args.phase;directory.mkdir(exist_ok=True)
    for folder in ['worlds','specs','source']: (directory/folder).mkdir(exist_ok=True)
    specs=[specification(name,'default') for name in names] if args.selection is None else json.loads(args.selection.read_text())['specifications']
    manifest={'phase':args.phase,'worlds_per_spec':args.worlds,'practice_count':args.practice,'test_count':args.tests,'support_count':0,'specifications':specs,'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in SOURCES},'design':'Same questions and source-blind prompts in both arms; independent unjudged own answers; exact acceptance of alternative certificates; teacher solutions generated without inspecting model output.'}
    path=directory/'data_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    for name in SOURCES: (directory/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    tokenizer=AutoTokenizer.from_pretrained(json.loads((root/'runtime/model.json').read_text())['path']);records=[]
    for spec in tqdm(specs,desc='Prepare witness tasks',mininterval=5):
        (directory/'specs'/f"{spec['id']}.json").write_text(json.dumps(spec,indent=2))
        for i in range(args.worlds):
            seed=seed_for(spec,args.phase,i);world=make_world(spec,seed,args.practice,args.tests,support_count=0);name=spec['id']+'__'+str(seed)+'.json';p=directory/'worlds'/name
            if p.exists(): assert json.loads(p.read_text())==world
            else: p.write_text(json.dumps(world,indent=2))
            maximum=0
            for query in world['tests']:
                prompt=tokenizer.apply_chat_template([{'role':'user','content':answer_prompt(world,world['practice'],query)}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                maximum=max(maximum,len(tokenizer.encode(prompt,add_special_tokens=False)))
            records.append({'spec':spec['id'],'seed':seed,'status':'ready' if maximum+1024<=16384 else 'context_too_long','world':name,'max_teacher_prompt_tokens':maximum,'world_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (directory/'preflight.json').write_text(json.dumps(records,indent=2));print(json.dumps({'settings':len(specs),'episodes':len(records),'ready':sum(r['status']=='ready' for r in records),'maximum_teacher_prompt_tokens':max(r['max_teacher_prompt_tokens'] for r in records)}))
