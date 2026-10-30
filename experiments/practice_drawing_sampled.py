"""Learn drawing causes without enumerating an initial posterior.

Training retains sampled programs only. A one-time binomial upper confidence
bound on initial likelihood turns a later variational bound into a certificate.
The exhaustive catalogue is isolated inside the evaluation callback.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
from practice_drawing_fixed_posterior import mstep, complete_q

CONFIG = dict(initial_dreams=1048576, initial_batch=4096, batch=128, rate=.15, trials=16,
              cache_size=16, bootstrap_rounds=100, bootstrap_trials=256,
              max_steps=1600, eval_every=20, min_steps=200, window=10, slope=.003, delta=.001)


def decode(codes): return ((codes[:,None]//m.POWERS) % 4).astype(np.uint8)


class Geometry:
    def __init__(self):
        dy,dx=np.divmod(np.arange(64),8)
        dx,dy=dx-3,dy-3
        self.indices=np.empty((64,4,64),dtype=np.uint64)
        for position in range(64):
            y,x=divmod(position,8)
            for direction in range(4):
                xx=(x+dx*m.DX[direction]-dy*m.DY[direction]) % 8
                yy=(y+dx*m.DY[direction]+dy*m.DX[direction]) % 8
                self.indices[position,direction]=yy*8+xx


def initial_parameters(observed, noise, mixing):
    ink=((observed[:,None] >> np.arange(64,dtype=np.uint64)) & np.uint64(1)).sum(axis=-1)
    forward=(ink.mean()-1)/9
    base=np.array([forward,(1-forward)/3,(1-forward)/3,(1-forward)/3])
    assert np.all(base>=0)
    return mixing.copy(),m.row_normalize(base[None,None,:]*noise)


def world(cat,seed):
    teacher_rng=np.random.default_rng(seed)
    prototypes=teacher_rng.choice(np.flatnonzero((m.MICRO==0).any(axis=-1)),3,replace=False)
    teacher_b=.94*np.eye(4)[m.MICRO[prototypes]]+.06/4
    teacher_a=.9*teacher_rng.dirichlet(np.full(3,.6),size=3)+.1/3
    _,teacher_p=cat.probabilities(teacher_a,teacher_b)
    observed=cat.images[teacher_rng.choice(len(cat.images),128,p=teacher_p)]
    rng=np.random.default_rng(seed+310000)
    noise=rng.lognormal(0,.35,(3,3,4))
    mixing=m.row_normalize(rng.dirichlet(np.full(3,6.),size=3))
    a,b=initial_parameters(observed,noise,mixing)
    return dict(seed=seed,observed=observed,teacher_a=teacher_a,teacher_b=teacher_b,
                teacher_p=teacher_p,initial_a=a,initial_b=b,noise=noise)


def program_probability(codes,a,b):
    actions=decode(codes).reshape(-1,3,3)
    probability=np.ones(len(codes))
    for c in range(3):
        component=np.prod(b[np.arange(3)[None,:,None],np.arange(3)[None,None,:],actions[:,None,c,:]],axis=-1)
        probability*=component @ a[c]
    return probability


def expected_counts(codes,weights,a,b):
    actions=decode(codes).reshape(-1,3,3)
    acount,bcount=np.zeros((3,3)),np.zeros((3,3,4))
    for c in range(3):
        component=np.prod(b[np.arange(3)[None,:,None],np.arange(3)[None,None,:],actions[:,None,c,:]],axis=-1)
        responsibility=component*a[c]
        responsibility/=responsibility.sum(axis=-1,keepdims=True)
        weighted=responsibility*weights[:,None]
        acount[c]=weighted.sum(axis=0)
        for t in range(3):
            for action in range(4): bcount[:,t,action]+=weighted[actions[:,c,t]==action].sum(axis=0)
    assert np.max(abs(acount.sum(axis=-1)-1))<1e-10
    assert abs(bcount.sum()-9)<1e-10
    return acount,bcount


def upper_binomial(frequency,n,alpha):
    """Invert the Bernoulli KL Chernoff lower-tail bound."""
    assert np.all((frequency>=0) & (frequency<1))
    entropy=(1-frequency)*np.log1p(-frequency)
    positive=frequency>0
    entropy[positive]+=frequency[positive]*np.log(frequency[positive])
    lo,hi=frequency.copy(),np.ones_like(frequency)
    threshold=np.log(1/alpha)/n
    for _ in range(60):
        mid=(lo+hi)/2
        divergence=entropy-frequency*np.log(mid)-(1-frequency)*np.log1p(-mid)
        below=divergence<threshold
        lo=np.where(below,mid,lo); hi=np.where(below,hi,mid)
    return hi


def merge(caches,targets,codes,bitmaps,a,b):
    for i,target in enumerate(targets):
        combined=np.union1d(caches[i],codes[bitmaps==target])
        probability=program_probability(combined,a,b)
        caches[i]=combined[np.argsort(probability,kind='stable')[-CONFIG['cache_size']:]]


def posterior(caches,mass,a,b):
    codes=np.concatenate(caches)
    sizes=np.array([len(c) for c in caches])
    assert np.all(sizes>0)
    probability=program_probability(codes,a,b)
    total=np.add.reduceat(probability,np.r_[0,np.cumsum(sizes)[:-1]])
    weights=probability*np.repeat(mass/total,sizes)
    bound=mass @ np.log(total)
    return codes,weights,float(bound)


def sleep_update(geometry,recognition,codes,rng):
    actions=decode(codes); bitmaps=m.render(actions)
    gradient=m.recognize(geometry,recognition,bitmaps,rng,actions,update=True)[3]
    recognition+=CONFIG['rate']*gradient
    return bitmaps


def initialize(observed,a,b,seed,learn_recognition):
    geometry=Geometry()
    targets,frequency=np.unique(observed,return_counts=True)
    mass=frequency/len(observed)
    rng=np.random.default_rng(seed+320000)
    caches=[np.array([],dtype=np.int64) for _ in targets]
    recognition=np.zeros((9,4,129))
    hits=np.zeros(len(targets),dtype=int)
    probe=m.draw(a,b,np.random.default_rng(seed+320001),512)
    probe_actions=decode(probe); probe_images=m.render(probe_actions)
    history=[]
    for offset in tqdm(range(0,CONFIG['initial_dreams'],CONFIG['initial_batch']),desc='Initial generated experience',leave=False):
        codes=m.draw(a,b,rng,CONFIG['initial_batch'])
        bitmaps=m.render(decode(codes))
        hits+=np.array([(bitmaps==target).sum() for target in targets])
        merge(caches,targets,codes,bitmaps,a,b)
        if learn_recognition: sleep_update(geometry,recognition,codes[:CONFIG['batch']],rng)
        if offset % (CONFIG['initial_batch']*16)==0:
            lp=m.recognize(geometry,recognition,probe_images,rng,probe_actions)[2]
            history.append([offset+CONFIG['initial_batch'],-lp.mean(),sum(len(c)>0 for c in caches)])
    upper=upper_binomial(hits/CONFIG['initial_dreams'],CONFIG['initial_dreams'],CONFIG['delta']/len(targets))
    upper_ll=float(mass @ np.log(upper))
    for attempt in range(CONFIG['bootstrap_rounds']):
        missing=np.array([len(c)==0 for c in caches])
        if not missing.any(): break
        questions=np.repeat(targets[missing],CONFIG['bootstrap_trials'])
        codes,valid,_,_=m.recognize(geometry,recognition,questions,rng)
        merge(caches,targets,codes[valid],questions[valid],a,b)
        if learn_recognition: sleep_update(geometry,recognition,m.draw(a,b,rng,CONFIG['batch']),rng)
    assert all(len(c)>0 for c in caches), 'Sampled initialization failed to find a valid cause for every image'
    return dict(targets=targets,mass=mass,caches=caches,recognition=recognition,
                upper_ll=upper_ll,upper_probabilities=upper,hits=hits,history=np.array(history),bootstrap_rounds=attempt)


def train(observed,initial_a,initial_b,initial,seed,condition,evaluate,steps,cache_dreams):
    geometry=Geometry()
    rng=np.random.default_rng(seed+330000)
    eval_rng=np.random.default_rng(seed+330001)
    targets,mass=initial['targets'],initial['mass']
    caches=[c.copy() for c in initial['caches']]
    recognition=initial['recognition'].copy()
    a,b=initial_a.copy(),initial_b.copy()
    codes,weights,bound=posterior(caches,mass,a,b)
    fixed_counts=expected_counts(codes,weights,a,b)
    records,parameters=[],[]
    start=time.perf_counter()
    status='maximum updates'
    for step in tqdm(range(steps+1),desc=condition,leave=False):
        if step % CONFIG['eval_every']==0 or step==steps:
            nll,risk,recognition_ce,valid=evaluate(a,b,geometry,recognition,eval_rng)
            generated=m.draw(a,b,eval_rng,512); actions=decode(generated)
            dream_ce=-m.recognize(geometry,recognition,m.render(actions),eval_rng,actions)[2].mean()
            assert -nll>=bound-1e-9
            records.append([step,nll,risk,-bound,bound-initial['upper_ll'],recognition_ce,valid,dream_ce,time.perf_counter()-start])
            parameters.append(np.r_[a.ravel(),b.ravel()])
            assert np.isfinite(records[-1]).all()
            if step>=CONFIG['min_steps']:
                recent=np.array(records[-CONFIG['window']:])
                slopes=[np.polyfit(np.log(recent[:,0]),np.log(recent[:,j]),1)[0] for j in [3,7]]
                if max(abs(s) for s in slopes)<CONFIG['slope']:
                    status='stationary variational bound and dream recognition'; break
        if step==steps: break
        old_bound=bound
        if condition!='fixed_posterior':
            if condition!='fixed_cache':
                if condition=='wake': dreams=rng.choice(codes,CONFIG['batch'],p=weights)
                elif condition=='prior': dreams=m.draw(initial_a,initial_b,rng,CONFIG['batch'])
                else: dreams=m.draw(a,b,rng,CONFIG['batch'])
                if condition!='frozen_proposal': sleep_update(geometry,recognition,dreams,rng)
                if cache_dreams: merge(caches,targets,dreams,m.render(decode(dreams)),a,b)
                questions=np.repeat(targets,CONFIG['trials'])
                proposals,valid,_,_=m.recognize(geometry,recognition,questions,rng)
                merge(caches,targets,proposals[valid],questions[valid],a,b)
            codes,weights,bound=posterior(caches,mass,a,b)
            assert bound>=old_bound-1e-10
            acount,bcount=expected_counts(codes,weights,a,b)
        else: acount,bcount=fixed_counts
        next_a,next_b=mstep(acount,bcount)
        gain=complete_q(next_a,next_b,acount,bcount)-complete_q(a,b,acount,bcount)
        assert gain>=-1e-10
        a,b=next_a,next_b
        if condition=='fixed_posterior': bound+=gain
        else: codes,weights,bound=posterior(caches,mass,a,b)
        assert bound>=old_bound+gain-1e-9
    return dict(records=np.array(records),weights=np.array(parameters),a=a,b=b,recognition=recognition,status=status,cache_dreams=cache_dreams,
                caches=np.concatenate(caches),cache_sizes=np.array([len(c) for c in caches]))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20103120)
    parser.add_argument('--steps',type=int,default=CONFIG['max_steps'])
    parser.add_argument('--route',choices=['both','proposal'],default='both')
    parser.add_argument('--recognition-initialization',choices=['generated','zero'],default='generated')
    parser.add_argument('--conditions',nargs='+',default=['sleep','prior','wake','frozen_proposal','fixed_posterior'],
                        choices=['sleep','prior','wake','frozen_proposal','fixed_posterior','fixed_cache'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    protocol=dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),config=CONFIG,seed=args.seed,steps=args.steps,
        scope='Exploratory sampled-initialization experiment; original draws are bitmaps only.',
        initialization='Generator from ink-count frequency and independent random perturbations. Valid caches from generated examples, without program enumeration.',
        recognition_initialization=args.recognition_initialization,route=args.route,
        certificate='Initial likelihood upper bound from independent initial-generator samples, simultaneous binomial KL upper bounds at failure probability .001 per world. Later ELBO minus this bound certifies improvement for all times on the same event.',
        stopping='Variational bound and generated-cause recognition cross entropy only.',
        evaluation='An isolated callback uses enumeration and teacher probabilities for diagnostics, never to accept updates.',
        conditions=args.conditions)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    cat=m.Catalogue()
    data=world(cat,args.seed)
    np.savez_compressed(args.output/'world.npz',**data)
    initial=initialize(data['observed'],data['initial_a'],data['initial_b'],args.seed,learn_recognition=args.recognition_initialization=='generated')
    np.savez_compressed(args.output/'initial.npz',**{k:v for k,v in initial.items() if k!='caches'},
                        caches=np.concatenate(initial['caches']),cache_sizes=np.array([len(c) for c in initial['caches']]))
    observed_ids=np.searchsorted(cat.images,data['observed'])
    probe=m.draw(data['teacher_a'],data['teacher_b'],np.random.default_rng(args.seed+340000),512)
    def evaluate(a,b,geometry,recognition,rng):
        _,p=cat.probabilities(a,b)
        lp=m.recognize(geometry,recognition,cat.canvas[probe],rng,cat.actions[probe])[2]
        valid=m.recognize(geometry,recognition,cat.canvas[probe],rng)[1].mean()
        return -np.log(p[observed_ids]).mean(),np.sum(data['teacher_p']*np.log(data['teacher_p']/p)),-lp.mean(),valid
    summary=dict(seed=args.seed,upper_ll=initial['upper_ll'],conditions={})
    for condition in args.conditions:
        run=train(data['observed'],data['initial_a'],data['initial_b'],initial,args.seed,condition,evaluate,args.steps,cache_dreams=args.route=='both')
        np.savez_compressed(args.output/f'{condition}.npz',**run)
        summary['conditions'][condition]=dict(final=run['records'][-1].tolist(),status=run['status'])
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps(dict(condition=condition,**summary['conditions'][condition])),flush=True)


if __name__=='__main__': main()
