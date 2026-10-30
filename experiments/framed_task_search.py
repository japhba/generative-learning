"""Shared wordings of the same allocation constraints, with source-only contrasts."""

import argparse
from collections import defaultdict
from dataclasses import make_dataclass
import hashlib
import json
from pathlib import Path
import sys

from reasoning_gym.factory import DATASETS,register_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_api import ROOT
from task_search_data import make_world,seed_for,specification
from task_search_prompts import answer_prompt

FRAMES={'meeting_slots':('meeting','meetings','time slot','time slots'),
        'storage_bins':('item','items','storage bin','storage bins'),
        'record_codes':('record','records','numeric code','numeric codes'),
        'radio_channels':('transmitter','transmitters','radio channel','radio channels'),
        'vertex_colors':('vertex','vertices','color','colors')}
GraphDataset,GraphConfig=DATASETS['graph_color']


class FramedDataset(GraphDataset):
    def __getitem__(self,index):
        entry=super().__getitem__(index); puzzle=entry['metadata']['puzzle']; entity,entities,resource,resources=FRAMES[self.frame]
        entry['question']=(f'Allocate one {resource} to each {entity}. For every listed pair, its two {entities} must receive different {resources}. These are the only constraints.\n\n'
                           f'{entities.capitalize()}: {puzzle["vertices"]}\nPairs: {puzzle["edges"]}\nAllowed {resources}: {puzzle["color_options"]}\n\n'
                           f'Return a JSON object mapping {entity} IDs to {resource} numbers.')
        entry['answer']=json.dumps(entry['metadata']['possible_answer'],sort_keys=True); entry['metadata']['frame']=self.frame; return entry


def register():
    names=[]
    for frame in FRAMES:
        for colors in [6,12]:
            for probability in [.4,.7]:
                name=f'{frame}_v6_c{colors}_p{int(100*probability)}'; config=make_dataclass(name+'Config',[('min_num_vertices',int,6),('max_num_vertices',int,6),('num_colors',int,colors),('edge_probability',float,probability)],bases=(GraphConfig,))
                cls=type(name+'Dataset',(FramedDataset,),{'frame':frame}); register_dataset(name,cls,config); names.append(name)
    return names


def prepare(names):
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True); parser.add_argument('--selection',type=Path); parser.add_argument('--worlds',type=int,default=1); parser.add_argument('--practice',type=int,default=8); parser.add_argument('--tests',type=int,default=4); args=parser.parse_args()
    root=ROOT.resolve(); directory=root/args.phase; (directory/'worlds').mkdir(parents=True,exist_ok=True); (directory/'specs').mkdir(exist_ok=True)
    specs=[specification(name,'default') for name in names] if args.selection is None else json.loads(args.selection.read_text())['specifications']
    manifest={'phase':args.phase,'worlds_per_spec':args.worlds,'practice_count':args.practice,'test_count':args.tests,'support_count':4,'specifications':specs,'max_teacher_prompt_tokens':14000,
              'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['framed_task_search.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']},
              'design':'One allocation problem under five shared wordings. Every source uses the same wording, initial examples, questions, and decoding. Palette >= vertices permits an injective valid solution, without a source-specific hint. Teacher uses the same standard greedy oracle.',
              'pairing':'Episode seeds depend on geometry and phase, not wording, so the underlying graphs are paired across frames at the same geometry.'}
    path=directory/'data_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(json.loads((root/'runtime/model.json').read_text())['path']); records=[]
    for spec in tqdm(specs,desc='Prepare shared task wordings',mininterval=5):
        (directory/'specs'/f"{spec['id']}.json").write_text(json.dumps(spec,indent=2)); geometry=spec['task'].split('_v6_')[1]
        for index in range(args.worlds):
            seed=seed_for({'id':'allocation_v6_'+geometry},args.phase,index); world=make_world(spec,seed,args.practice,args.tests); name=spec['id']+'__'+str(seed)+'.json'
            (directory/'worlds'/name).write_text(json.dumps(world,indent=2)); maximum=max(len(tokenizer.encode(answer_prompt(world,world['practice'],query))) for query in world['tests'])
            records.append({'spec':spec['id'],'seed':seed,'status':'ready' if maximum<=14000 else 'context_too_long','world':name,'max_teacher_prompt_tokens':maximum})
    (directory/'preflight.json').write_text(json.dumps(records,indent=2)); print(json.dumps({'settings':len(specs),'episodes':len(records),'ready':sum(r['status']=='ready' for r in records)},indent=2))


def select():
    parser=argparse.ArgumentParser(); parser.add_argument('--phase',required=True); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); directory=ROOT.resolve()/args.phase
    summary=json.loads((directory/'summary.json').read_text()); pre=json.loads((directory/'preflight.json').read_text()); ready={r['spec']:r for r in pre if r['status']=='ready'}; assert len(summary)==len(ready)
    frames=defaultdict(list)
    for row in summary: frames[row['task'].split('_v6_')[0]].append(row)
    chosen=[]; decisions=[]
    for frame,rows in sorted(frames.items()):
        best=min(rows,key=lambda r:(abs(r['conditions'][0]['accuracy']-.5),ready[r['spec']]['max_teacher_prompt_tokens'],r['spec']))
        chosen.append(json.loads((directory/'specs'/f"{best['spec']}.json").read_text())); decisions.append({'frame':frame,'selected':best['spec'],'baseline_accuracy':best['conditions'][0]['accuracy'],'candidates':rows})
    args.output.write_text(json.dumps({'purpose':'One geometry per shared wording, selected by baseline headroom only; all five wordings are retained. No source outcome has been observed.','source_summary_sha256':hashlib.sha256((directory/'summary.json').read_bytes()).hexdigest(),'specifications':chosen,'decisions':decisions},indent=2)); print(json.dumps(decisions,indent=2))


def main():
    names=register(); mode=sys.argv.pop(1)
    if mode=='prepare': prepare(names)
    elif mode=='select': select()
    else:
        assert mode in {'run','audit','strategy_audit'}
        if mode=='run':
            from run_task_search import main as run
            phase=sys.argv[sys.argv.index('--phase')+1]; source=ROOT.resolve()/phase/'extension_source'; source.mkdir(exist_ok=True); (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes()); run()
        elif mode=='audit':
            from audit_task_search import main as audit
            audit()
        else:
            from audit_palette_strategy import main as audit
            audit()


if __name__=='__main__': main()
