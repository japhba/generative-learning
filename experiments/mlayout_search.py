"""Correctness-matched laid-out reasoning: does a model learn more from its own correct worked solutions than from
other models' equally correct solutions to the same questions?

Every episode has a hidden rule shown by K=4 examples. Each model samples sixteen worked solutions per pool question. For
each pool question (in order) where every model has a finished, correct solution, the first such solution per model is
kept, up to six questions; own and foreign practice are therefore all correct and cover identical questions. At test,
readers answer directly: the reply is prefilled with "Answer:", so no reader can reason or ignore the format.
"""
import argparse
import asyncio
from collections import defaultdict,deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
from random import Random
import re
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

from mlayout_api import DIRECTORY,Served
from mlayout_tasks import FAMILIES,correct,world as make_world
from task_search_api import digest

NAMES=['qwen35_9b','qwen3_30b','gemma4_26b','phi4']
FAMILY={'qwen35_9b':'qwen','qwen3_30b':'qwen','gemma4_26b':'google','phi4':'microsoft'}
EPISODES=8;SUPPORT=4;POOL=16;TESTS=12;PRACTICE=6;MIN_PRACTICE=3;SAMPLES=16
GEN_TOKENS=2048;ANSWER_TOKENS=12;GEN_TEMPERATURE=.8
CONDITIONS=['initial','answers']+[f'full:{m}' for m in NAMES]
FILES=['MLAYOUT_PROTOCOL.md','mlayout_api.py','mlayout_search.py','mlayout_tasks.py']
TITLES={'operator':'mystery operator','units':'alien units'}
LINE=re.compile(r'^[ \t>#*_-]*(?:final[ \t]+)?answer[ \t]*[*_]*[ \t]*[:：][ \t]*(.*?)[ \t]*$',re.I|re.M)  # no newline crossing: '### Final Answer:\n\nAnswer: 7' -> '7'


def now(): return datetime.now(ZoneInfo('Europe/London')).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def read(path): return json.loads(path.read_text())
def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}
def final_answer(text):
    found=[a.replace('**','').replace('`','').strip().rstrip('.').strip() for a in LINE.findall(text)];found=[a for a in found if a];return found[-1] if found else ''
def direct_answer(text): return (text.strip().splitlines() or [''])[0].replace('**','').replace('`','').strip().rstrip('.').strip()


def header(world):
    return f"Problem family: {TITLES[world['family']]}. Every problem follows the same hidden rule.\nExamples:\n"+'\n\n'.join(f"Problem: {e['question']}\nAnswer: {e['answer']}" for e in world['support'])


def generation_messages(world,question):
    return [{'role':'user','content':header(world)+f'\n\nSolve this new problem. Work out the hidden rule from the examples and reason step by step in your own words, then finish with a final line "Answer: <answer>".\nProblem: {question}\nSolution:'}]


def evaluation_messages(world,practice,question):
    shown='\n\nWorked practice problems:\n'+'\n\n'.join(f"Problem: {p['question']}\nSolution: {p['solution']}" for p in practice) if practice else ''
    return [{'role':'user','content':header(world)+shown+f'\n\nSolve this new problem. Do not show any working: reply with only the final answer.\nProblem: {question}'},{'role':'assistant','content':'Answer:'}]


def prepare():
    assert not (DIRECTORY/'design.json').exists(),'Already frozen.';worlds={}
    for family in FAMILIES:
        for episode in range(EPISODES):
            seed=int(hashlib.sha256(f'mlayout_20261005|{family}|{episode}'.encode()).hexdigest()[:8],16)
            worlds[f'{family}__{episode:02}']={**make_world(family,seed,SUPPORT,POOL,TESTS),'episode':episode}
    for name,w in worlds.items(): write(DIRECTORY/'worlds'/f'{name}.json',w)
    write(DIRECTORY/'design.json',{'time_uk':now(),'models':NAMES,'conditions':CONDITIONS,'episodes':EPISODES,'support':SUPPORT,'pool':POOL,'tests':TESTS,'practice':PRACTICE,
                                   'min_practice':MIN_PRACTICE,'samples':SAMPLES,'generation_tokens':GEN_TOKENS,'answer_tokens':ANSWER_TOKENS,'generation_temperature':GEN_TEMPERATURE,
                                   'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    for name in FILES: (DIRECTORY/'source'/name).parent.mkdir(parents=True,exist_ok=True);(DIRECTORY/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'worlds':len(worlds)}))


def generation_path(model,name,index): return DIRECTORY/'generations'/model/f'{name}__{index:02}.json'
def evaluation_path(reader,name,condition,index): return DIRECTORY/'evaluations'/reader/f"{name}__{condition.replace(':','-')}__{index:02}.json"


def match(worlds):
    """Frozen selection: pool questions (in order) where every model has a finished correct sample; first correct sample per model."""
    out={}
    for name,w in worlds.items():
        chosen=[]
        for i,entry in enumerate(w['pool']):
            picks={}
            for m in NAMES:
                samples=read(generation_path(m,name,i))['samples'];good=[k for k,s in enumerate(samples) if s['status']=='ok' and correct(final_answer(s['text']),entry)]
                if good: picks[m]=good[0]
            if len(picks)==len(NAMES): chosen.append({'index':i,'sample':picks})
            if len(chosen)==PRACTICE: break
        out[name]={'questions':chosen,'included':len(chosen)>=MIN_PRACTICE}
    write(DIRECTORY/'matched.json',out);return out


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    models={m:Served(m) for m in NAMES};write(DIRECTORY/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})

    async def generate(m,name,i):
        path=generation_path(m,name,i)
        if path.exists(): return
        w=worlds[name];samples=await models[m].call(generation_messages(w,w['pool'][i]['question']),w['seed']+1000+i,GEN_TEMPERATURE,GEN_TOKENS,n=SAMPLES)
        write(path,{'model':m,'world':name,'index':i,'samples':samples})
    jobs=[(m,n,i) for n in worlds for i in range(POOL) for m in NAMES];Random(1).shuffle(jobs)
    for future in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc='generate',mininterval=15): await future
    matched=match(worlds)

    def practice(name,condition):
        w=worlds[name];qs=matched[name]['questions']
        if condition=='initial': return []
        if condition=='answers': return [{'question':w['pool'][q['index']]['question'],'solution':f"Answer: {w['pool'][q['index']]['answer']}"} for q in qs]
        m=condition.split(':')[1]
        return [{'question':w['pool'][q['index']]['question'],'solution':read(generation_path(m,name,q['index']))['samples'][q['sample'][m]]['text']} for q in qs]

    async def evaluate(r,name,c,i):
        path=evaluation_path(r,name,c,i)
        if path.exists(): return
        w=worlds[name];out=(await models[r].call(evaluation_messages(w,practice(name,c),w['tests'][i]['question']),w['seed']+i,0.,ANSWER_TOKENS,prefill=True))[0]
        write(path,{'reader':r,'world':name,'condition':c,'index':i,**out})
    groups=[(r,n,c) for n in worlds if matched[n]['included'] for c in CONDITIONS for r in NAMES];Random(2).shuffle(groups)
    queues={m:deque(g for g in groups if g[0]==m) for m in NAMES};bar=tqdm(total=len(groups)*TESTS,desc='evaluate',mininterval=15)
    async def worker(q):
        while q:
            r,n,c=q.popleft();head=asyncio.create_task(evaluate(r,n,c,0))
            if not evaluation_path(r,n,c,0).exists(): await asyncio.sleep(1)  # first request fills the prefix cache
            await asyncio.gather(head,*(evaluate(r,n,c,i) for i in range(1,TESTS)));bar.update(TESTS)
    await asyncio.gather(*(worker(queues[m]) for m in NAMES for _ in range(6)))
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'included_episodes':sum(v['included'] for v in matched.values())})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())


if __name__=='__main__': main()
