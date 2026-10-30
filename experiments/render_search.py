"""Same content, own versus foreign expression: every source re-expresses the SAME correct solution in its own words.

GSM8K arm: each model renders the human reference solution of each training problem (keep every step and number, end
with "Answer:"). A rendering is valid if its final answer is correct and every intermediate result of the reference
calculation appears. Problems where all six renderers have a valid rendering are kept; each source contributes the valid
rendering closest to the problem's median length. Outputs fine-tuning sets (six renderings + the raw human solution) and
an in-context arm for the small students.
Operator arm: a programmatic canonical solution (check the rule on the examples, then compute) is rendered by three
larger models; readers answer directly (prefilled "Answer:") after six practice items in one source's rendering.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
from random import Random
import re

import numpy as np
from tqdm.auto import tqdm

import mlayout2_search as M2
from mlayout_api import Served
from mlayout_search import direct_answer,evaluation_messages,final_answer,header,now,read,write
from mlayout_tasks import correct as op_correct
from small_search import correct as gsm_correct,messages as gsm_messages,number
from task_search_api import ROOT

DIRECTORY=ROOT.resolve()/'render';CEPH=Path('/ceph/scratch/jbauer/mlayout/render')
SMALL=['qwen3_1_7b','llama32_3b','smollm3_3b','phi4_mini'];LARGE=['gemma4_26b','qwen3_30b'];GSM_SOURCES=SMALL+LARGE
OP_MODELS=['qwen35_9b','qwen3_30b','gemma4_26b']
SAMPLES=4;TEMPERATURE=.8;GSM_TRAIN=2000;GSM_TEST=500;ICL_TEST=300;ICL_K=4;OP_EPISODES=16;OP_POOL=12;OP_PRACTICE=6;OP_TESTS=12
FILES=['RENDER_PROTOCOL.md','render_search.py','small_search.py','mlayout_api.py','mlayout_search.py','mlayout2_search.py']
RENDER='Here is a correct solution to a problem. Rewrite the solution in your own words, the way you would naturally explain it yourself. Keep every step and every number; do not add new steps or skip any. Finish with a final line "Answer: <answer>".\n\nProblem: {question}\n\nCorrect solution:\n{solution}\n\nYour own explanation:'


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def numbers(text): return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',text.replace(',','').replace('$',''))]
def gsm_valid(sample,row):
    if sample['status']!='ok' or not gsm_correct(sample['text'],row['answer']): return False
    found=numbers(sample['text']);return all(any(abs(f-v)<1e-6 for f in found) for v in (number(x) for x in row['intermediates']) if v is not None)
def canonical(w,question):
    p,q,sym=w['rule']['p'],w['rule']['q'],w['rule']['symbol'];a,b=map(int,re.findall(r'\d+',question)[:2])
    checks='\n'.join(f"{p}·{x} + {q}·{y} = {p*x} + {q*y} = {e['answer']} ✓" for e in w['support'] for x,y in [tuple(map(int,re.findall(r'\d+',e['question'])[:2]))])
    return f"Compare the examples with a rule of the form a {sym} b = p·a + q·b.\n{checks}\nSo the hidden rule is a {sym} b = {p}a + {q}b.\nFor {a} {sym} {b}: {p}·{a} = {p*a}, {q}·{b} = {q*b}, and {p*a} + {q*b} = {p*a+q*b}.\nAnswer: {p*a+q*b}"
def op_valid(sample,entry,w):
    if sample['status']!='ok' or not op_correct(final_answer(sample['text']),entry): return False
    a,b=map(int,re.findall(r'\d+',entry['question'])[:2]);found=set(re.findall(r'\d+',sample['text'].replace(',','')))
    return str(w['rule']['p']*a) in found and str(w['rule']['q']*b) in found
def pick(good):
    target=float(np.median([l for v in good.values() for _,l in v]));return {m:min(v,key=lambda x:(abs(x[1]-target),x[0]))[0] for m,v in good.items()}
def path(*parts): return DIRECTORY.joinpath(*parts[:-1])/f'{parts[-1]}.json'


def prepare():
    assert not (DIRECTORY/'design.json').exists();data=read(Path('/ceph/scratch/jbauer/mlayout/small/gsm8k_human.json'))
    train=data['train'][:];Random(20261006).shuffle(train);test=data['test'][:];Random(20261007).shuffle(test)
    worlds={f'operator__{e:02}':{**M2.make_world(int(hashlib.sha256(f'render_20261006|operator|{e}'.encode()).hexdigest()[:8],16)),'episode':e} for e in range(OP_EPISODES)}
    for n,w in worlds.items(): w['pool']=w['pool'][:OP_POOL];w['tests']=w['tests'][:OP_TESTS]
    write(DIRECTORY/'design.json',{'time_uk':now(),'gsm_train':train[:GSM_TRAIN],'gsm_test':test[:GSM_TEST],'op_worlds':worlds,'samples':SAMPLES,'temperature':TEMPERATURE,'source_sha256':hashes()})
    print('frozen')


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    models={m:Served(m,32) for m in sorted(set(GSM_SOURCES+OP_MODELS))};write(DIRECTORY/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})
    async def render(m,key,question,solution,seed):
        p=path('renderings',m,key)
        if p.exists(): return
        samples=await models[m].call([{'role':'user','content':RENDER.format(question=question,solution=solution)}],seed,TEMPERATURE,1024,n=SAMPLES);write(p,{'model':m,'key':key,'samples':samples})
    worlds=design['op_worlds'];jobs=[(m,r['id'],r['question'],r['human'],int(r['id'][5:])) for r in design['gsm_train'] for m in GSM_SOURCES]
    jobs+=[(m,f'{n}__{i}',e['question'],canonical(w,e['question']),w['seed']+i) for n,w in worlds.items() for i,e in enumerate(w['pool']) for m in OP_MODELS]
    Random(1).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([render(*j) for j in jobs]),total=len(jobs),desc='render',mininterval=15): await f
    # GSM8K matching and outputs
    chosen=[]
    for r in design['gsm_train']:
        good={m:[(k,len(s['text'])) for k,s in enumerate(read(path('renderings',m,r['id']))['samples']) if gsm_valid(s,r)] for m in GSM_SOURCES}
        if all(good.values()): chosen.append({'id':r['id'],'question':r['question'],'human':r['human'],'pick':pick(good)})
    write(DIRECTORY/'gsm_matched.json',chosen);CEPH.mkdir(parents=True,exist_ok=True)
    text=lambda s,c:c['human'] if s=='human' else read(path('renderings',s,c['id']))['samples'][c['pick'][s]]['text']
    for s in GSM_SOURCES+['human']: (CEPH/f'train_{s}.json').write_text(json.dumps([{'id':c['id'],'messages':gsm_messages(c['question'])+[{'role':'assistant','content':text(s,c)}]} for c in chosen]))
    (CEPH/'eval_sets.json').write_text(json.dumps({'gsm_test_cot':[{'id':q['id'],'answer':q['answer'],'messages':gsm_messages(q['question'])} for q in design['gsm_test']]}))
    print('gsm matched',len(chosen),flush=True)
    tests=design['gsm_test'][:ICL_TEST]
    async def icl(r,s,i):
        p=path('icl',r,f'{s}__{i}')
        if p.exists(): return
        examples=[{'question':c['question'],'solution':text(s,c)} for c in Random(1000+i).sample(chosen,ICL_K)] if s!='initial' else ()
        out=(await models[r].call(gsm_messages(tests[i]['question'],examples),i,0.,1024))[0];write(p,{'reader':r,'source':s,'index':i,'answer':tests[i]['answer'],**out})
    jobs=[(r,s,i) for r in SMALL for s in ['initial','human']+GSM_SOURCES for i in range(len(tests))];Random(2).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([icl(*j) for j in jobs]),total=len(jobs),desc='gsm icl',mininterval=15): await f
    # Operator matching and answer-only tests
    op={}
    for n,w in worlds.items():
        qs=[]
        for i,e in enumerate(w['pool']):
            good={m:[(k,len(s['text'])) for k,s in enumerate(read(path('renderings',m,f'{n}__{i}'))['samples']) if op_valid(s,e,w)] for m in OP_MODELS}
            if all(good.values()): qs.append({'index':i,'pick':pick(good)})
            if len(qs)==OP_PRACTICE: break
        op[n]=qs
    write(DIRECTORY/'op_matched.json',op)
    def practice(n,c):
        w=worlds[n];qs=op[n]
        if c=='initial': return []
        if c=='answers': return [{'question':w['pool'][q['index']]['question'],'solution':f"Answer: {w['pool'][q['index']]['answer']}"} for q in qs]
        if c=='canonical': return [{'question':w['pool'][q['index']]['question'],'solution':canonical(w,w['pool'][q['index']]['question'])} for q in qs]
        m=c.split(':')[1];return [{'question':w['pool'][q['index']]['question'],'solution':read(path('renderings',m,f"{n}__{q['index']}"))['samples'][q['pick'][m]]['text']} for q in qs]
    conditions=['initial','answers','canonical']+[f'render:{m}' for m in OP_MODELS]
    async def op_eval(r,n,c,i):
        p=path('op_eval',r,f"{n}__{c.replace(':','-')}__{i}")
        if p.exists(): return
        w=worlds[n];out=(await models[r].call(evaluation_messages(w,practice(n,c),w['tests'][i]['question']),w['seed']+i,0.,12,prefill=True))[0]
        write(p,{'reader':r,'world':n,'condition':c,'index':i,**out})
    jobs=[(r,n,c,i) for n in worlds if len(op[n])>=3 for c in conditions for i in range(OP_TESTS) for r in OP_MODELS];Random(3).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([op_eval(*j) for j in jobs]),total=len(jobs),desc='op eval',mininterval=15): await f
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'gsm_matched':len(chosen),'op_included':sum(len(v)>=3 for v in op.values())})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())
