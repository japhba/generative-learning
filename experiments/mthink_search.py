"""Arm 1: own versus foreign THINKING traces as practice (the reasoning a model does for itself), matched for correctness and length.

Same pipeline as mlayout_search.py (answer-only tests with an "Answer:" prefill), operator family only, three models that
regularly find the rule. A practice solution qualifies only if it finishes with the correct answer AND explicitly
computes both partial products p*x and q*y; for each pool question where every model has such solutions, each model
contributes the qualifying solution whose length is closest to the median length of all qualifying solutions for that
question (ties: lower sample index), up to six questions.
"""
import argparse
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
from random import Random
import re

import numpy as np
from tqdm.auto import tqdm

from mlayout_api import Served
from mlayout_search import direct_answer,evaluation_messages,final_answer,generation_messages,now,read,write
from mlayout_tasks import correct,item
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'mthink'
NAMES=['qwen35_9b','gemma4_26b','nemotron3_30b']
EPISODES=16;SUPPORT=4;POOL=20;TESTS=12;PRACTICE=3;MIN_PRACTICE=3;SAMPLES=8;MAX_REASONING_CHARS=30000
GEN_TOKENS=16384;ANSWER_TOKENS=12;GEN_TEMPERATURE=.8
CONDITIONS=['initial','answers']+[f'think:{m}' for m in NAMES]
FILES=['MTHINK_PROTOCOL.md','mlayout_api.py','mlayout_search.py','mlayout_tasks.py','mthink_search.py']


def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}


def make_world(seed):
    rng=Random(seed);p,q=rng.choice([(p,q) for p in (2,3,4,5) for q in (2,3,4) if p!=q]);spec={'symbol':rng.choice('◇⊛⊙★⊕⊗'),'p':p,'q':q};seen=set();items=[]
    while len(items)<SUPPORT+POOL+TESTS:
        question,answer=item('operator',spec,rng)
        if question not in seen: seen.add(question);items.append({'question':question,'answer':answer})
    return {'family':'operator','seed':seed,'rule':spec,'support':items[:SUPPORT],'pool':items[SUPPORT:SUPPORT+POOL],'tests':items[SUPPORT+POOL:]}


def process_correct(sample,entry,rule):
    if sample['status']!='ok' or len(sample['reasoning'])>MAX_REASONING_CHARS or not correct(final_answer(sample['text']),entry): return False
    a,b=map(int,re.findall(r'\d+',entry['question'])[:2]);numbers=set(re.findall(r'\d+',(sample['reasoning']+' '+sample['text']).replace(',','')))
    return str(rule['p']*a) in numbers and str(rule['q']*b) in numbers


def generation_path(m,n,i): return DIRECTORY/'generations'/m/f'{n}__{i:02}.json'
def evaluation_path(r,n,c,i): return DIRECTORY/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"


def prepare():
    assert not (DIRECTORY/'design.json').exists();worlds={}
    for e in range(EPISODES):
        seed=int(hashlib.sha256(f'mthink_20261005|operator|{e}'.encode()).hexdigest()[:8],16);worlds[f'operator__{e:02}']={**make_world(seed),'episode':e}
    for n,w in worlds.items(): write(DIRECTORY/'worlds'/f'{n}.json',w)
    write(DIRECTORY/'design.json',{'time_uk':now(),'models':NAMES,'conditions':CONDITIONS,'episodes':EPISODES,'pool':POOL,'samples':SAMPLES,'practice':PRACTICE,'min_practice':MIN_PRACTICE,
                                   'generation_tokens':GEN_TOKENS,'answer_tokens':ANSWER_TOKENS,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    for name in FILES: (DIRECTORY/'source'/name).parent.mkdir(parents=True,exist_ok=True);(DIRECTORY/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'worlds':len(worlds)}))


def match(worlds):
    out={}
    for n,w in worlds.items():
        chosen=[]
        for i,e in enumerate(w['pool']):
            good={m:[(k,len(s['reasoning'])) for k,s in enumerate(read(generation_path(m,n,i))['samples']) if process_correct(s,e,w['rule'])] for m in NAMES}
            if all(good.values()):
                target=float(np.median([length for v in good.values() for _,length in v]))
                chosen.append({'index':i,'target_chars':target,'sample':{m:min(v,key=lambda x:(abs(x[1]-target),x[0]))[0] for m,v in good.items()}})
            if len(chosen)==PRACTICE: break
        out[n]={'questions':chosen,'included':len(chosen)>=MIN_PRACTICE}
    write(DIRECTORY/'matched.json',out);return out


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes()
    worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    models={m:Served(m) for m in NAMES};write(DIRECTORY/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})
    async def generate(m,n,i):
        if generation_path(m,n,i).exists(): return
        w=worlds[n];samples=await models[m].call(generation_messages(w,w['pool'][i]['question']),w['seed']+1000+i,GEN_TEMPERATURE,GEN_TOKENS,n=SAMPLES,thinking=True)
        write(generation_path(m,n,i),{'model':m,'world':n,'index':i,'samples':samples})
    jobs=[(m,n,i) for n in worlds for i in range(POOL) for m in NAMES];Random(1).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc='generate',mininterval=15): await f
    matched=match(worlds)
    def practice(n,c):
        w=worlds[n];qs=matched[n]['questions']
        if c=='initial': return []
        if c=='answers': return [{'question':w['pool'][q['index']]['question'],'solution':f"Answer: {w['pool'][q['index']]['answer']}"} for q in qs]
        m=c.split(':')[1];chosen=[read(generation_path(m,n,q['index']))['samples'][q['sample'][m]] for q in qs]
        return [{'question':w['pool'][q['index']]['question'],'solution':f"Reasoning: {s['reasoning'].strip()}\nAnswer: {final_answer(s['text'])}"} for q,s in zip(qs,chosen)]
    async def evaluate(r,n,c,i):
        if evaluation_path(r,n,c,i).exists(): return
        w=worlds[n];out=(await models[r].call(evaluation_messages(w,practice(n,c),w['tests'][i]['question']),w['seed']+i,0.,ANSWER_TOKENS,prefill=True))[0]
        write(evaluation_path(r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**out})
    groups=[(r,n,c) for n in worlds if matched[n]['included'] for c in CONDITIONS for r in NAMES];Random(2).shuffle(groups)
    queues={m:deque(g for g in groups if g[0]==m) for m in NAMES};bar=tqdm(total=len(groups)*TESTS,desc='evaluate',mininterval=15)
    async def worker(q):
        while q:
            r,n,c=q.popleft();head=asyncio.create_task(evaluate(r,n,c,0))
            if not evaluation_path(r,n,c,0).exists(): await asyncio.sleep(1)
            await asyncio.gather(head,*(evaluate(r,n,c,i) for i in range(1,TESTS)));bar.update(TESTS)
    await asyncio.gather(*(worker(queues[m]) for m in NAMES for _ in range(6)))
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'included_episodes':sum(v['included'] for v in matched.values())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())
