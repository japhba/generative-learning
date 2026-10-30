"""Graph-coloring task grid with several valid strategies under identical constraints."""

import argparse
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

GraphDataset,GraphConfig=DATASETS['graph_color']


class PaletteDataset(GraphDataset):
    def __getitem__(self,index):
        entry=super().__getitem__(index); entry['answer']=json.dumps(entry['metadata']['possible_answer'],sort_keys=True); return entry


def register():
    names=[]
    for vertices in [6,10,14,18]:
        for colors in [vertices//2,vertices,2*vertices]:
            for probability in [.15,.4,.7]:
                name=f'graph_palette_v{vertices}_c{colors}_p{int(100*probability)}'
                config=make_dataclass(name+'Config',[('min_num_vertices',int,vertices),('max_num_vertices',int,vertices),('num_colors',int,colors),('edge_probability',float,probability)],bases=(GraphConfig,))
                register_dataset(name,PaletteDataset,config); names.append(name)
    return names


def prepare(names):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); parser.add_argument('--selection',type=Path); parser.add_argument('--worlds',type=int,default=1); parser.add_argument('--practice',type=int,default=8); parser.add_argument('--tests',type=int,default=4); args=parser.parse_args()
    root=ROOT.resolve(); directory=root/args.phase; (directory/'worlds').mkdir(parents=True,exist_ok=True); (directory/'specs').mkdir(exist_ok=True)
    specs=[specification(name,'default') for name in names] if args.selection is None else json.loads(args.selection.read_text())['specifications']
    manifest={'phase':args.phase,'worlds_per_spec':args.worlds,'practice_count':args.practice,'test_count':args.tests,'support_count':4,'max_teacher_prompt_tokens':14000,'specifications':specs,
              'source_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['strategy_task_search.py','task_search_data.py','task_search_scoring.py','task_search_prompts.py']},
              'hypothesis':'A palette with at least one color per vertex admits a trivial injective coloring, while the canonical teacher uses greedy color reuse. No strategy hint or extra feedback is provided to either source.'}
    path=directory/'data_manifest.json'
    if path.exists(): assert json.loads(path.read_text())==manifest
    else: path.write_text(json.dumps(manifest,indent=2))
    tokenizer=AutoTokenizer.from_pretrained(json.loads((root/'runtime/model.json').read_text())['path']); records=[]
    for spec in tqdm(specs,desc='Strategy task grid',mininterval=5):
        (directory/'specs'/f"{spec['id']}.json").write_text(json.dumps(spec,indent=2))
        for index in range(args.worlds):
            seed=seed_for(spec,args.phase,index); world=make_world(spec,seed,args.practice,args.tests); name=spec['id']+'__'+str(seed)+'.json'
            (directory/'worlds'/name).write_text(json.dumps(world,indent=2)); maximum=max(len(tokenizer.encode(answer_prompt(world,world['practice'],query))) for query in world['tests'])
            records.append({'spec':spec['id'],'seed':seed,'status':'ready' if maximum<=14000 else 'context_too_long','world':name,'max_teacher_prompt_tokens':maximum})
    (directory/'preflight.json').write_text(json.dumps(records,indent=2)); print(json.dumps({'settings':len(specs),'episodes':len(records),'ready':sum(r['status']=='ready' for r in records)},indent=2))


def main():
    names=register(); mode=sys.argv.pop(1); assert mode in {'prepare','run'}
    if mode=='prepare': prepare(names)
    else:
        # The same inference implementation is used for both the broad and focused searches.
        from run_task_search import main as run
        phase=sys.argv[sys.argv.index('--phase')+1]; source=ROOT.resolve()/phase/'extension_source'; source.mkdir(exist_ok=True)
        (source/Path(__file__).name).write_bytes(Path(__file__).read_bytes()); run()


if __name__=='__main__': main()
