"""Arm 2: own versus foreign matched practice when readers may reason at test.

Reuses the mlayout3 confirmation's practice (identical questions, process-correct, length-matched solutions from three
models) and tests with step-by-step reasoning allowed (2048 tokens, final "Answer:" line), after the first 1 or all 6
practice solutions from one source. No new practice is generated.
"""
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
from random import Random

from tqdm.auto import tqdm

import mlayout3_search as M3
from mlayout_api import Served
from mlayout_search import final_answer,header,now,read,write
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'mcot'
NAMES=M3.NAMES;COUNTS=[1,6];TESTS=12;TOKENS=2048
CONDITIONS=['initial']+[f'{kind}:{m}:{n}' for n in COUNTS for m in NAMES for kind in ['full']]+[f'answers:{n}' for n in COUNTS]
FILES=['mcot_search.py','mlayout_api.py','mlayout3_search.py','mlayout_search.py','mlayout_tasks.py']
INSTRUCTION='Solve this new problem. Reason step by step, then finish with a final line "Answer: <answer>".'


def path_of(r,n,c,i): return DIRECTORY/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"


async def main():
    hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
    if (DIRECTORY/'design.json').exists(): assert read(DIRECTORY/'design.json')['source_sha256']==hashes,'Frozen sources changed.'
    else: write(DIRECTORY/'design.json',{'time_uk':now(),'practice_from':'mlayout3','conditions':CONDITIONS,'tokens':TOKENS,'source_sha256':hashes})
    matched=read(M3.DIRECTORY/'matched.json');worlds={n:read(M3.DIRECTORY/'worlds'/f'{n}.json') for n in matched if matched[n]['included']}
    models={m:Served(m) for m in NAMES};write(DIRECTORY/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})
    def practice(n,c):
        w=worlds[n];qs=matched[n]['questions']
        if c=='initial': return []
        kind,*rest=c.split(':');count=int(rest[-1]);qs=qs[:count]
        if kind=='answers': return [{'question':w['pool'][q['index']]['question'],'solution':f"Answer: {w['pool'][q['index']]['answer']}"} for q in qs]
        m=rest[0];return [{'question':w['pool'][q['index']]['question'],'solution':read(M3.generation_path(m,n,q['index']))['samples'][q['sample'][m]]['text']} for q in qs]
    def messages(n,c,i):
        shown='\n\nWorked practice problems:\n'+'\n\n'.join(f"Problem: {p['question']}\nSolution: {p['solution']}" for p in practice(n,c)) if c!='initial' else ''
        return [{'role':'user','content':header(worlds[n])+shown+f"\n\n{INSTRUCTION}\nProblem: {worlds[n]['tests'][i]['question']}"}]
    async def evaluate(r,n,c,i):
        if path_of(r,n,c,i).exists(): return
        out=(await models[r].call(messages(n,c,i),worlds[n]['seed']+i,0.,TOKENS))[0];write(path_of(r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**out})
    groups=[(r,n,c) for n in worlds for c in CONDITIONS for r in NAMES];Random(3).shuffle(groups);queues={m:deque(g for g in groups if g[0]==m) for m in NAMES}
    bar=tqdm(total=len(groups)*TESTS,desc='mcot',mininterval=15)
    async def worker(q):
        while q:
            r,n,c=q.popleft();head=asyncio.create_task(evaluate(r,n,c,0))
            if not path_of(r,n,c,0).exists(): await asyncio.sleep(1)
            await asyncio.gather(head,*(evaluate(r,n,c,i) for i in range(1,TESTS)));bar.update(TESTS)
    await asyncio.gather(*(worker(queues[m]) for m in NAMES for _ in range(8)))
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'episodes':len(worlds)})


if __name__=='__main__': asyncio.run(main())
