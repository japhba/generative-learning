"""Remove recognition pretraining while keeping the sampled initial caches."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_sampled as s
import practice_drawing_library as m


def run(job):
    source,output,index,route=job
    cat=m.Catalogue(); world=np.load(source/f'run_{index}/world.npz')
    saved=np.load(source/f'run_{index}/initial.npz')
    initial={key:saved[key] for key in saved.files}
    initial['caches']=list(np.split(saved['caches'],np.cumsum(saved['cache_sizes'])[:-1]))
    initial['recognition']=np.zeros_like(saved['recognition'])
    seed=int(world['seed'])
    ids=np.searchsorted(cat.images,world['observed'])
    probe=m.draw(world['teacher_a'],world['teacher_b'],np.random.default_rng(seed+340000),512)
    def evaluate(a,b,geometry,recognition,rng):
        _,p=cat.probabilities(a,b)
        lp=m.recognize(geometry,recognition,cat.canvas[probe],rng,cat.actions[probe])[2]
        valid=m.recognize(geometry,recognition,cat.canvas[probe],rng)[1].mean()
        return -np.log(p[ids]).mean(),np.sum(world['teacher_p']*np.log(world['teacher_p']/p)),-lp.mean(),valid
    result={}
    for condition in ['sleep','prior','frozen_proposal']:
        fitted=s.train(world['observed'],world['initial_a'],world['initial_b'],initial,seed,condition,evaluate,s.CONFIG['max_steps'],cache_dreams=route=='both')
        np.savez_compressed(output/f'{condition}_{index}.npz',**fitted)
        result[condition]=dict(final=fitted['records'][-1].tolist(),status=fitted['status'])
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=8)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--route',choices=['both','proposal'],required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    shutil.copy2(s.__file__,args.output/'learner_source.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        retrospective=True,source=str(args.source),worlds=args.worlds,route=args.route,
        question='Does continuous neural recognition learning matter when recognition does not start pretrained?',
        change='Set all initial proposal weights to zero; keep identical sampled initial caches, initial generator, data, and likelihood certificate.',
        controls='Current-generator sleep, initial-generator sleep, and frozen zero proposal. Effective cache posterior adapts in every condition.',
        scope='Same worlds as the preceding test. A causal follow-up, not an independent replication.',
        routing='Both means dreams may directly enter caches; proposal means dreams train the inverse network but only its conditional proposals can enter caches.'),indent=2)+'\n')
    rows={}
    with ProcessPoolExecutor(args.workers) as executor:
        futures={executor.submit(run,(args.source,args.output,i,args.route)):i for i in range(args.worlds)}
        for future in tqdm(as_completed(futures),total=args.worlds,desc='Untrained recognition'):
            i=futures[future]; rows[i]=future.result()
            (args.output/'summary.json').write_text(json.dumps({str(k):v for k,v in sorted(rows.items())},indent=2)+'\n')
            print(json.dumps(dict(world=i,conditions=rows[i])),flush=True)


if __name__=='__main__': main()
