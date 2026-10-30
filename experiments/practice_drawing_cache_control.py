"""Retrospective control: adapt latent weights but discover no new programs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_sampled as s
import practice_drawing_library as m


def train(cat,world,initial):
    caches=list(np.split(initial['caches'],np.cumsum(initial['cache_sizes'])[:-1]))
    mass=initial['mass']
    a,b=world['initial_a'].copy(),world['initial_b'].copy()
    geometry=s.Geometry(); recognition=initial['recognition']
    rng=np.random.default_rng(int(world['seed'])+330001)
    ids=np.searchsorted(cat.images,world['observed'])
    probe=m.draw(world['teacher_a'],world['teacher_b'],np.random.default_rng(int(world['seed'])+340000),512)
    records,parameters=[],[]
    start=time.perf_counter(); status='maximum updates'
    for step in range(s.CONFIG['max_steps']+1):
        codes,weights,bound=s.posterior(caches,mass,a,b)
        if step % s.CONFIG['eval_every']==0:
            _,p=cat.probabilities(a,b)
            ll=np.log(p[ids]).mean()
            risk=np.sum(world['teacher_p']*np.log(world['teacher_p']/p))
            lp=m.recognize(geometry,recognition,cat.canvas[probe],rng,cat.actions[probe])[2]
            valid=m.recognize(geometry,recognition,cat.canvas[probe],rng)[1].mean()
            generated=m.draw(a,b,rng,512); actions=s.decode(generated)
            dream_ce=-m.recognize(geometry,recognition,m.render(actions),rng,actions)[2].mean()
            records.append([step,-ll,risk,-bound,bound-float(initial['upper_ll']),-lp.mean(),valid,dream_ce,time.perf_counter()-start])
            parameters.append(np.r_[a.ravel(),b.ravel()])
            assert ll>=bound-1e-10
            if step>=s.CONFIG['min_steps']:
                recent=np.array(records[-s.CONFIG['window']:])
                slopes=[np.polyfit(np.log(recent[:,0]),np.log(recent[:,j]),1)[0] for j in [3,7]]
                if max(abs(x) for x in slopes)<s.CONFIG['slope']:
                    status='stationary variational bound and dream recognition'; break
        if step==s.CONFIG['max_steps']: break
        ac,bc=s.expected_counts(codes,weights,a,b)
        a,b=s.mstep(ac,bc)
        assert s.posterior(caches,mass,a,b)[2]>=bound-1e-10
    return dict(records=np.array(records),weights=np.array(parameters),a=a,b=b,status=status)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=8)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    protocol=dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),source=str(args.source),
        retrospective=True,trigger='Several sleep and frozen-proposal outcomes are almost identical.',
        control='Keep the sampled initial program caches fixed. Continue updating motif responsibilities and generator-dependent posterior weights, followed by exact categorical M steps. No new latent program is proposed or discovered.',
        distinction='This freezes the set of explanations, not the full recognition distribution. Generated probes are evaluation-only.')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cat=m.Catalogue(); rows=[]
    for i in tqdm(range(args.worlds),desc='Fixed-cache control'):
        world=np.load(args.source/f'run_{i}/world.npz')
        initial=np.load(args.source/f'run_{i}/initial.npz')
        run=train(cat,world,initial)
        np.savez_compressed(args.output/f'fixed_cache_{i}.npz',**run)
        rows.append(dict(world=i,final=run['records'][-1].tolist(),status=run['status']))
        (args.output/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(rows[-1]),flush=True)


if __name__=='__main__': main()
