"""EDGE2: the edge-of-capability own-vs-peers design with four flash-tier models on OpenRouter (EDGE2_PROTOCOL.md)."""
import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

from dotenv import load_dotenv
import httpx
import numpy as np
from random import Random
from tqdm.auto import tqdm

import edge_search as ES
import mlayout_tasks as MT
from mlayout_search import final_answer,now,read,write
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'edge2'
MODELS={'deepseek_flash':{'model':'deepseek/deepseek-v4.1-flash','provider':'deepinfra/fp8','name':'DeepInfra','reasoning':{'enabled':False}},
        'gemma4_31b':{'model':'google/gemma-4-31b-it','provider':'crusoe/bf16','name':'Crusoe'},
        'llama4_maverick':{'model':'meta-llama/llama-4-maverick','provider':'novita/fp8','name':'Novita'}}  # amendment 1: Qwen3.8-Flash and Mistral-Small dropped
CALIBRATION_MODELS=['deepseek_flash','qwen_flash','mistral_small','gemma4_31b']  # the frozen calibration panel
NAMES=list(MODELS);FAMILIES=['operator','offset','units','cipher']
PHASES={'calibration':{'episodes':3,'pool':8,'tests':0},'main':{'episodes':36,'pool':8,'tests':8},'main_hard':{'episodes':36,'pool':8,'tests':8,'family':'operator_hard'}}  # amendments 1-2
SUPPORT=4;SAMPLES=3;SET=4;TOKENS=1024;TEMPERATURE=.8;BUDGET=4.35;CONCURRENCY=16  # amendment 1 (calibration used 4 samples)
FILES=['EDGE2_PROTOCOL.md','edge2_search.py','edge_search.py','layout_tasks.py','mlayout_tasks.py']
RETRY={408,500,502,503,504,520,522,524,529};SPENT=[]


UNITS=['vorp','blim','tazz','quon','frell','zink','glorp','mib','drax','plon']


def units4(seed,support=4,pool=12,tests=12):
    """Four units a > b > c > d with hidden ratios; the examples show only a→b, a→c, a→d and b→d, so b→c, c→d etc. must be
    inferred by division; questions go in both directions (reverse ones have integer answers)."""
    rng=Random(seed);u=rng.sample(UNITS,4);r=[rng.randint(3,12) for _ in range(3)];size=lambda i,j:int(np.prod(r[i:j]))  # units of u[j] in one u[i]
    def item(i,j,reverse):
        k=rng.randint(2,19)
        return (f'How many {u[i]}s make {k*size(i,j)} {u[j]}s?',str(k)) if reverse else (f'How many {u[j]}s are in {k} {u[i]}s?',str(k*size(i,j)))
    shown=[item(0,1,False),item(0,2,False),item(0,3,False),item(1,3,False)];seen={q for q,_ in shown};rest=[]
    pairs=[(i,j) for i in range(4) for j in range(i+1,4)]
    while len(rest)<pool+tests:
        i,j=rng.choice(pairs);q,a=item(i,j,rng.random()<.3)
        if q not in seen: seen.add(q);rest.append({'question':q,'answer':a})
    return {'family':'units4','seed':seed,'rule':{'units':u,'ratios':r},'support':[{'question':q,'answer':a} for q,a in shown],'pool':rest[:pool],'tests':rest[pool:]}


def operator_hard(seed,support=4,pool=12,tests=12):
    """Amendment 2: the EDGE operator family (a◇b = p·a + q·b, operands 11–49) restricted to p ∈ {4,5}, q ∈ {2,3,4}, p ≠ q."""
    rng=Random(seed);p,q=rng.choice([(p,q) for p in (4,5) for q in (2,3,4) if p!=q]);spec={'symbol':rng.choice('◇⊛⊙★⊕⊗'),'p':p,'q':q};seen=set();items=[]
    while len(items)<support+pool+tests:
        question,answer=MT.item('operator',spec,rng)
        if question not in seen: seen.add(question);items.append({'question':question,'answer':answer})
    return {'family':'operator_hard','seed':seed,'rule':spec,'support':items[:support],'pool':items[support:support+pool],'tests':items[support+pool:]}


def make_world(family,seed): return units4(seed) if family=='units4' else operator_hard(seed) if family=='operator_hard' else ES.make_world(family,seed)


TITLES=ES.TITLES|{'units4':'alien units','operator_hard':'mystery operator'}
def header(world): return f"Problem family: {TITLES[world['family']]}. Every problem follows the same hidden rule.\nExamples:\n"+'\n\n'.join(f"Problem: {e['question']}\nAnswer: {e['answer']}" for e in world['support'])
def messages(world,question,attempts=()):
    shown='\n\nEarlier attempts at practice problems (unchecked; some may be wrong):\n'+'\n\n'.join(f"Problem: {a['question']}\nAttempt: {a['text']}" for a in attempts) if attempts else ''
    return [{'role':'user','content':header(world)+shown+f'\n\n{ES.INSTRUCTION}\nProblem: {question}'}]


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def correct(text,entry):
    line=final_answer(text).replace(',','')
    if re.fullmatch(r'-?\d+',entry['answer']): found=re.findall(r'-?\d+(?:\.\d+)?',line);return bool(found) and abs(float(found[-1])-float(entry['answer']))<1e-6
    found=re.findall(r'[A-Za-z]+',line);return bool(found) and found[-1].upper()==entry['answer'].upper()


class Remote:
    def __init__(self,key):
        load_dotenv(Path.home()/'.env');c=MODELS[key];self.name=c['name']
        self.config={'model':c['model'],'provider':{'only':[c['provider']],'allow_fallbacks':False,'require_parameters':True}}|({'reasoning':c['reasoning']} if 'reasoning' in c else {})
        self.fingerprint=digest(self.config);self.directory=DIRECTORY/'api'/self.fingerprint;self.directory.mkdir(parents=True,exist_ok=True);(self.directory/'engine.json').write_text(json.dumps(self.config,indent=2))
        self.client=httpx.AsyncClient(timeout=300,headers={'Authorization':'Bearer '+os.environ['OPENROUTER_API_KEY']});self.semaphore=asyncio.Semaphore(CONCURRENCY);self.locks={}
        if not SPENT: SPENT.append(sum(json.loads(p.read_text())['response']['usage']['cost'] for p in (DIRECTORY/'api').glob('*/*.json') if len(p.stem)==64))

    async def close(self): await self.client.aclose()

    async def call(self,msgs,seed,temperature):
        request={**self.config,'messages':msgs,'temperature':temperature,'top_p':.9 if temperature else 1.,'max_tokens':TOKENS,'seed':seed};key=digest(request);path=self.directory/f'{key}.json'
        self.locks.setdefault(key,asyncio.Lock())
        async with self.locks[key]:
            if path.exists(): record=json.loads(path.read_text())
            else:
                async with self.semaphore:
                    assert SPENT[0]<BUDGET,f'EDGE2 budget ceiling ${BUDGET} reached; cached work is retained.'
                    attempt=limited=0
                    while True:
                        started=time.time()
                        try: response=await self.client.post('https://openrouter.ai/api/v1/chat/completions',json=request)
                        except (httpx.TimeoutException,httpx.TransportError) as error:
                            attempt+=1;(self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'error':repr(error),'time':started}))
                            if attempt==10: raise
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.status_code==429:  # shared upstream pools throttle: patient jittered retries
                            limited+=1;(self.directory/f'{key}.failure-r{limited}.json').write_text(json.dumps({'status':429,'body':response.text,'time':started}))
                            if limited==40: response.raise_for_status()
                            await asyncio.sleep(min(2**limited,60)*(.5+random.random()));continue
                        if response.status_code in RETRY:
                            attempt+=1;(self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'status':response.status_code,'body':response.text,'time':started}))
                            if attempt==10: response.raise_for_status()
                            await asyncio.sleep(min(2**attempt,30));continue
                        if response.is_error: (self.directory/f'{key}.error.json').write_text(json.dumps({'request':request,'status':response.status_code,'body':response.text},indent=2))
                        response.raise_for_status();body=response.json()
                        if 'choices' not in body:
                            attempt+=1;(self.directory/f'{key}.failure-{attempt}.json').write_text(json.dumps({'status':200,'body':body,'time':started}))
                            if attempt==10: raise RuntimeError(body)
                            await asyncio.sleep(min(2**attempt,30));continue
                        record={'request':request,'response':body,'started_unix':started,'elapsed_seconds':time.time()-started}
                        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(record,indent=2));temporary.replace(path);SPENT[0]+=body['usage']['cost'];break
        response=record['response'];choice=response['choices'][0];message=choice['message']
        assert response['provider']==self.name,(response['provider'],self.name);assert not message.get('reasoning'),'hidden reasoning must be off'
        return {'key':key,'status':'ok' if choice['finish_reason']=='stop' and message['content'] else 'truncated','text':(message['content'] or '').strip(),'usage':response['usage']}


def gen_path(d,m,n,i,k): return d/'generations'/m/f'{n}__{i:02}__{k}.json'
def eval_path(d,r,n,c,i): return d/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"


def prepare(phase):
    d=DIRECTORY/phase;assert not (d/'design.json').exists(),'Already frozen.';P=PHASES[phase]
    families=FAMILIES if phase=='calibration' else [P['family']] if 'family' in P else [read(DIRECTORY/'main_plan.json')['family']]
    worlds={}
    for f in families:
        for e in range(P['episodes']):
            seed=int(hashlib.sha256(f'edge2_{phase}_20261006|{f}|{e}'.encode()).hexdigest()[:8],16);w=make_world(f,seed)  # EDGE generators sized for 12+12 items
            worlds[f'{f}__{e:03}']={**w,'pool':w['pool'][:P['pool']],'tests':w['tests'][:P['tests']],'episode':e}
    for n,w in worlds.items(): write(d/'worlds'/f'{n}.json',w)
    write(d/'design.json',{'time_uk':now(),'models':MODELS,'phase':P,'samples':SAMPLES,'tokens':TOKENS,'temperature':TEMPERATURE,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    print(json.dumps({'phase':phase,'worlds':len(worlds)}))


def match(w,gens):
    """As EDGE: identical problems and right/wrong pattern for every source; per slot the qualifying attempt closest to the median length."""
    pool=len(w['pool'])
    def eligible(m,i,want): return [k for k,s in enumerate(gens[m][i]) if ((s['status']=='ok' and correct(s['text'],w['pool'][i])) if want else (s['text'] and not correct(s['text'],w['pool'][i])))]  # amendment 1: unfinished attempts count as unsuccessful
    C=[i for i in range(pool) if all(eligible(m,i,True) for m in NAMES)];I=[i for i in range(pool) if all(eligible(m,i,False) for m in NAMES)]
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


def conditions(m): return ['initial']+[f'mixed:{s}' for s in NAMES]  # amendment 1: no all-correct arm


async def run(phase):
    d=DIRECTORY/phase;design=read(d/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    worlds={n:read(d/'worlds'/f'{n}.json') for n in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    models={m:Remote(m) for m in NAMES};write(d/'inference.json',{m:{'fingerprint':r.fingerprint,'config':r.config} for m,r in models.items()})
    async def generate(m,n,i,k):
        if gen_path(d,m,n,i,k).exists(): return
        w=worlds[n];write(gen_path(d,m,n,i,k),{'model':m,'world':n,'index':i,'sample':k,**await models[m].call(messages(w,w['pool'][i]['question']),w['seed']+1000+100*i+k,TEMPERATURE)})
    jobs=[(m,n,i,k) for n in worlds for i in range(len(worlds[n]['pool'])) for m in NAMES for k in range(SAMPLES)]
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc=f'{phase} attempts',mininterval=15): await f
    print('spent so far $%.3f'%SPENT[0],flush=True)
    matched={n:match(w,{m:[[read(gen_path(d,m,n,i,k)) for k in range(SAMPLES)] for i in range(len(w['pool']))] for m in NAMES}) for n,w in worlds.items()};write(d/'matched.json',matched)
    print('mixed sets',sum(bool(v['mixed']) for v in matched.values()),'all-correct sets',sum(bool(v['correct']) for v in matched.values()),'of',len(worlds),flush=True)
    if phase!='calibration':
        def practice(n,c):
            if c=='initial': return []
            arm,s=c.split(':');w=worlds[n];return [{'question':w['pool'][x['index']]['question'],'text':read(gen_path(d,s,n,x['index'],x['sample'][s]))['text']} for x in matched[n][arm]]
        async def evaluate(r,n,c,i):
            if eval_path(d,r,n,c,i).exists(): return
            w=worlds[n];write(eval_path(d,r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**await models[r].call(messages(w,w['tests'][i]['question'],practice(n,c)),w['seed']+i,0.)})
        for n in tqdm([n for n in worlds if matched[n]['mixed']],desc='episodes (tests)'):  # episode by episode, so a budget stop leaves complete episodes
            await asyncio.gather(*(evaluate(r,n,c,i) for c in conditions(matched[n]) for r in NAMES for i in range(len(worlds[n]['tests']))))
            write(d/'tested.json',{'episodes':sorted({p.name.split('__')[0]+'__'+p.name.split('__')[1] for p in (d/'evaluations').glob('*/*.json')}),'spent':SPENT[0],'time_uk':now()})
    for r in models.values(): await r.close()
    write(d/'complete.json',{'time_uk':now(),'spent':SPENT[0]});print('spent $%.3f'%SPENT[0])


def calibrate():
    d=DIRECTORY/'calibration';assert (d/'complete.json').exists();design=read(d/'design.json');matched=read(d/'matched.json');summary={}
    for f in FAMILIES:
        names=[n for n in design['worlds'] if n.startswith(f+'__')];worlds={n:read(d/'worlds'/f'{n}.json') for n in names}
        acc={m:float(np.mean([correct(read(gen_path(d,m,n,i,k))['text'],worlds[n]['pool'][i]) for n in names for i in range(len(worlds[n]['pool'])) for k in range(SAMPLES)])) for m in NAMES}
        trunc={m:float(np.mean([read(gen_path(d,m,n,i,k))['status']!='ok' for n in names for i in range(len(worlds[n]['pool'])) for k in range(SAMPLES)])) for m in NAMES}
        y=float(np.mean([bool(matched[n]['mixed']) for n in names]));summary[f]={'attempts':acc,'truncated':trunc,'mixed_yield':y,'correct_yield':float(np.mean([bool(matched[n]['correct']) for n in names])),'eligible':y>=.3 and all(.1<=v<=.9 for v in acc.values())}
    eligible=[f for f in FAMILIES if summary[f]['eligible']] or [f for f in FAMILIES if summary[f]['mixed_yield']>=.3]
    print(json.dumps(summary,indent=1));assert eligible,'No family reaches 30% mixed yield: redesign needed.'
    family=max(eligible,key=lambda f:summary[f]['mixed_yield']);write(DIRECTORY/'main_plan.json',{'rule':'EDGE2_PROTOCOL.md','family':family,'summary':summary,'time_uk':now()});print('chosen',family)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run','calibrate']);parser.add_argument('phase',nargs='?',choices=list(PHASES));args=parser.parse_args()
    if args.stage=='prepare': prepare(args.phase)
    elif args.stage=='run': asyncio.run(run(args.phase))
    else: calibrate()
