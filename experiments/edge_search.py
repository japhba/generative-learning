"""Edge of capability: do small models learn more from their own unchecked attempts than from peers' attempts with the
same number of correct ones? (EDGE_PROTOCOL.md)

Each episode has a hidden rule shown by K=4 solved examples. Each of four similar-strength small models makes six
step-by-step attempts at each of twelve pool problems. Practice sets have identical problems and an identical
correct/incorrect pattern for every source (mixed: 2 correct + 2 incorrect; all-correct: 4). At test the reader sees
the examples, one source's practice attempts (or none) and solves a new problem step by step (greedy).
"""
import argparse
import asyncio
import hashlib
import json
import math
from pathlib import Path
from random import Random

import re

import numpy as np
from tqdm.auto import tqdm

import layout_tasks as LT
import mlayout_tasks as MT
from mlayout_api import Served
from mlayout_search import final_answer,now,read,write
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'edge'
NAMES=['qwen3_1_7b','llama32_3b','smollm3_3b','phi4_mini']
FAMILIES=['operator','offset','units','cipher']
TITLES={'operator':'mystery operator','offset':'mystery operator','units':'alien units','cipher':'secret code'}
SUPPORT=4;POOL=12;TESTS=12;SAMPLES=6;SET=4;CALIBRATION_EPISODES=12
GEN_TOKENS=1024;GEN_TEMPERATURE=.8;CONCURRENCY=96
INSTRUCTION='Solve this new problem. Work out the hidden rule from the examples and reason step by step in your own words, then finish with a final line "Answer: <answer>".'
FILES=['EDGE_PROTOCOL.md','edge_search.py','layout_tasks.py','mlayout_tasks.py','mlayout_api.py','mlayout_search.py']


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def strict_correct(text,entry): return MT.correct(final_answer(text),entry)  # frozen scorer, void (amendment 4)
def correct(text,entry):  # amendment 4: last number on the final answer line
    found=re.findall(r'-?\d+(?:\.\d+)?',final_answer(text).replace(',',''));return bool(found) and abs(float(found[-1])-float(entry['answer']))<1e-6


def make_world(family,seed):
    if family in ('operator','units'): w=MT.world(family,seed,SUPPORT,POOL,TESTS)
    else: w=LT.world('operator' if family=='offset' else family,seed,SUPPORT,POOL,TESTS);w['pool']=w.pop('practice')
    return {**w,'family':family}


def header(world):
    return f"Problem family: {TITLES[world['family']]}. Every problem follows the same hidden rule.\nExamples:\n"+'\n\n'.join(f"Problem: {e['question']}\nAnswer: {e['answer']}" for e in world['support'])


def messages(world,question,attempts=()):
    shown='\n\nEarlier attempts at practice problems (unchecked; some may be wrong):\n'+'\n\n'.join(f"Problem: {a['question']}\nAttempt: {a['text']}" for a in attempts) if attempts else ''
    return [{'role':'user','content':header(world)+shown+f'\n\n{INSTRUCTION}\nProblem: {question}'}]


def gen_path(d,m,n,i): return d/'generations'/m/f'{n}__{i:02}.json'
def eval_path(d,r,n,c,i): return d/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"


def prepare_main2():
    """Amendment 4: same episodes and attempts as main, re-matched and re-tested with the numeric scorer."""
    d=DIRECTORY/'main2';assert not (d/'design.json').exists(),'Already frozen.';d.mkdir(parents=True);main=read(DIRECTORY/'main'/'design.json')
    for sub in ['worlds','generations']: (d/sub).symlink_to(DIRECTORY/'main'/sub)
    write(d/'design.json',{**main,'time_uk':now(),'from':'main (amendment 4)','source_sha256':hashes()})
    for f in FILES: (d/'source'/f).parent.mkdir(parents=True,exist_ok=True);(d/'source'/f).write_bytes((Path(__file__).parent/f).read_bytes())
    print('main2 frozen',len(main['worlds']))


def prepare(phase):
    if phase=='main2': return prepare_main2()
    d=DIRECTORY/phase;assert not (d/'design.json').exists(),'Already frozen.'
    plan={f:CALIBRATION_EPISODES for f in FAMILIES} if phase=='calibration' else read(DIRECTORY/'main_plan.json')['episodes']
    worlds={f'{f}__{e:03}':{**make_world(f,int(hashlib.sha256(f'edge_{phase}_20261006|{f}|{e}'.encode()).hexdigest()[:8],16)),'episode':e} for f,k in plan.items() for e in range(k)}
    for n,w in worlds.items(): write(d/'worlds'/f'{n}.json',w)
    write(d/'design.json',{'time_uk':now(),'models':NAMES,'plan':plan,'support':SUPPORT,'pool':POOL,'tests':TESTS,'samples':SAMPLES,'set':SET,'gen_tokens':GEN_TOKENS,
                           'gen_temperature':GEN_TEMPERATURE,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    for f in FILES: (d/'source'/f).parent.mkdir(parents=True,exist_ok=True);(d/'source'/f).write_bytes((Path(__file__).parent/f).read_bytes())
    print(json.dumps({'phase':phase,'worlds':len(worlds),'plan':plan}))


def match(w,gens):
    """Frozen selection: identical problems and correct/incorrect pattern for every source; per slot the qualifying attempt closest to the median length."""
    def eligible(m,i,want): return [k for k,s in enumerate(gens[m][i]) if s['status']=='ok' and final_answer(s['text']) and correct(s['text'],w['pool'][i])==want]
    C=[i for i in range(POOL) if all(eligible(m,i,True) for m in NAMES)];I=[i for i in range(POOL) if all(eligible(m,i,False) for m in NAMES)]
    def pick(slots):
        out=[]
        for i,want in slots:
            options={m:[(k,len(gens[m][i][k]['text'])) for k in eligible(m,i,want)] for m in NAMES};target=float(np.median([l for v in options.values() for _,l in v]))
            out.append({'index':i,'correct':want,'sample':{m:min(v,key=lambda x:(abs(x[1]-target),x[0]))[0] for m,v in options.items()}})
        order=list(range(len(out)));Random(w['seed']).shuffle(order);return [out[j] for j in order]
    wrong=I[:SET//2];right=[i for i in C if i not in wrong][:SET//2]
    if len(right)<SET//2 or len(wrong)<SET//2: right=C[:SET//2];wrong=[i for i in I if i not in right][:SET//2]
    mixed=pick([(i,True) for i in right]+[(i,False) for i in wrong]) if len(right)==SET//2 and len(wrong)==SET//2 else None
    return {'C':C,'I':I,'mixed':mixed,'correct':pick([(i,True) for i in C[:SET]]) if len(C)>=SET else None}


def conditions(m): return ['initial']+[f'{arm}:{s}' for arm in ['mixed','correct'] if m[arm] for s in NAMES]


def practice(d,w,n,m,c):
    if c=='initial': return []
    arm,s=c.split(':');return [{'question':w['pool'][x['index']]['question'],'text':read(gen_path(d,s,n,x['index']))['samples'][x['sample'][s]]['text']} for x in m[arm]]


async def run(phase):
    d=DIRECTORY/phase;design=read(d/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    worlds={n:read(d/'worlds'/f'{n}.json') for n in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    models={m:Served(m,CONCURRENCY) for m in NAMES};write(d/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})

    async def generate(m,n,i):
        if gen_path(d,m,n,i).exists(): return
        w=worlds[n];samples=await models[m].call(messages(w,w['pool'][i]['question']),w['seed']+1000+i,GEN_TEMPERATURE,GEN_TOKENS,n=SAMPLES)
        write(gen_path(d,m,n,i),{'model':m,'world':n,'index':i,'samples':samples})
    jobs=[(m,n,i) for n in worlds for i in range(POOL) for m in NAMES]  # world-major order keeps each support prefix hot
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc=f'{phase} attempts',mininterval=15): await f
    matched={n:match(w,{m:[read(gen_path(d,m,n,i))['samples'] for i in range(POOL)] for m in NAMES}) for n,w in worlds.items()};write(d/'matched.json',matched)
    print('mixed sets',sum(bool(v['mixed']) for v in matched.values()),'all-correct sets',sum(bool(v['correct']) for v in matched.values()),'of',len(worlds),flush=True)

    async def evaluate(r,n,c,i):
        if eval_path(d,r,n,c,i).exists(): return
        w=worlds[n];out=(await models[r].call(messages(w,w['tests'][i]['question'],practice(d,w,n,matched[n],c)),w['seed']+i,0.,GEN_TOKENS))[0]
        write(eval_path(d,r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**out})
    jobs=[(r,n,c,i) for n in worlds for c in conditions(matched[n]) for r in NAMES for i in range(TESTS)]
    for f in tqdm(asyncio.as_completed([evaluate(*j) for j in jobs]),total=len(jobs),desc=f'{phase} tests',mininterval=15): await f
    for s in models.values(): await s.close()
    write(d/'complete.json',{'time_uk':now(),'mixed':sum(bool(v['mixed']) for v in matched.values()),'correct':sum(bool(v['correct']) for v in matched.values())})


def calibrate():
    """Frozen selection rule (EDGE_PROTOCOL.md): uses yields and no-practice accuracy only."""
    d=DIRECTORY/'calibration';assert (d/'complete.json').exists();design=read(d/'design.json');matched=read(d/'matched.json');summary={}
    for f in FAMILIES:
        names=[n for n in design['worlds'] if n.startswith(f+'__')];worlds={n:read(d/'worlds'/f'{n}.json') for n in names}
        initial={m:float(np.mean([correct(read(eval_path(d,m,n,'initial',i))['text'],worlds[n]['tests'][i]) for n in names for i in range(TESTS)])) for m in NAMES}
        attempts={m:float(np.mean([correct(s['text'],worlds[n]['pool'][i]) for n in names for i in range(POOL) for s in read(gen_path(d,m,n,i))['samples']])) for m in NAMES}
        summary[f]={'mixed_yield':float(np.mean([bool(matched[n]['mixed']) for n in names])),'correct_yield':float(np.mean([bool(matched[n]['correct']) for n in names])),'initial':initial,'attempts':attempts}
        summary[f]['eligible']=summary[f]['mixed_yield']>=.25 and all(.05<=v<=.9 for v in initial.values())
    chosen=sorted([f for f in FAMILIES if summary[f]['eligible']],key=lambda f:-summary[f]['mixed_yield'])[:2]
    plan={'rule':'EDGE_PROTOCOL.md phase 1','time_uk':now(),'families':chosen,'episodes':{f:min(150,math.ceil((48 if len(chosen)>1 else 120)/summary[f]['mixed_yield'])) for f in chosen},'summary':summary}  # amendment 1: a single family gets ~120 included episodes
    write(d/'summary.json',summary);print(json.dumps(summary,indent=1));print('chosen',chosen,plan['episodes'])
    assert chosen,'No family meets the frozen rule: redesign needed.'
    write(DIRECTORY/'main_plan.json',plan)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','calibrate']);parser.add_argument('phase',nargs='?',choices=['calibration','main','main2']);args=parser.parse_args()
    if args.stage=='prepare': prepare(args.phase)
    elif args.stage=='run': asyncio.run(run(args.phase))
    else: calibrate()
