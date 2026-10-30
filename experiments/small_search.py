"""Small students: do small models learn more from their own GSM8K solutions than from other models' equally correct ones?

Students Qwen3-1.7B, Llama-3.2-3B, SmolLM3-3B, Phi-4-mini; large sources Gemma 4 26B and Qwen3-30B-A3B. Each model
writes 4 zero-shot solutions per training problem. A problem is kept if every model has a correct, finished solution;
each model contributes the one closest to the problem's median length (identical questions and correctness for all six
sources). Outputs: fine-tuning sets (one per source) and an in-context arm (4 worked examples per test item, same
training problems for every source, step-by-step test).
"""
import argparse
import asyncio
from collections import deque
import hashlib
import json
from pathlib import Path
from random import Random

import numpy as np
from tqdm.auto import tqdm

from mlayout_api import Served
from mlayout_search import final_answer,now,read,write
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'small';CEPH=Path('/ceph/scratch/jbauer/mlayout/small')
STUDENTS=['qwen3_1_7b','llama32_3b','smollm3_3b','phi4_mini'];LARGE=['gemma4_26b','qwen3_30b'];SOURCES=STUDENTS+LARGE
TRAIN_Q=2000;SAMPLES=4;GEN_TOKENS=1024;TEST_N=500;ICL_TEST=300;ICL_K=4;GEN_TEMPERATURE=.8
INSTRUCTION='Solve the problem. Reason step by step, then finish with a final line "Answer: <number>".'
FILES=['SMALL_PROTOCOL.md','small_search.py','mlayout_api.py','mlayout_search.py']


def number(text):
    t=text.replace(',','').replace('$','').strip().rstrip('.')
    try: return float(t)
    except ValueError: return None
def correct(text,answer):
    value=number(final_answer(text));return value is not None and abs(value-float(answer))<1e-6
def messages(question,examples=()):
    shown='Worked examples:\n'+'\n\n'.join(f"Problem: {e['question']}\nSolution: {e['solution']}" for e in examples)+'\n\n' if examples else ''
    return [{'role':'user','content':f'{shown}{INSTRUCTION}\nProblem: {question}'}]
def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def gen_path(m,i): return DIRECTORY/'generations'/m/f'{i}.json'
def icl_path(r,s,i): return DIRECTORY/'icl'/r/f'{s}__{i}.json'


def prepare():
    assert not (DIRECTORY/'design.json').exists();data=read(CEPH/'gsm8k.json')
    train=data['train'][:];Random(20261006).shuffle(train);test=data['test'][:];Random(20261007).shuffle(test)
    design={'time_uk':now(),'students':STUDENTS,'large':LARGE,'train':train[:TRAIN_Q],'test':test[:TEST_N],'samples':SAMPLES,'gen_tokens':GEN_TOKENS,'icl_test':ICL_TEST,'icl_k':ICL_K,'source_sha256':hashes()}
    write(DIRECTORY/'design.json',design);print('frozen',len(design['train']),len(design['test']))


def match(design):
    chosen=[]
    for q in design['train']:
        good={m:[(k,len(s['text'])) for k,s in enumerate(read(gen_path(m,q['id']))['samples']) if s['status']=='ok' and correct(s['text'],q['answer'])] for m in SOURCES}
        if all(good.values()):
            target=float(np.median([l for v in good.values() for _,l in v]));chosen.append({'id':q['id'],'question':q['question'],'pick':{m:min(v,key=lambda x:(abs(x[1]-target),x[0]))[0] for m,v in good.items()}})
    write(DIRECTORY/'matched.json',chosen)
    CEPH.mkdir(parents=True,exist_ok=True)
    for m in SOURCES:
        rows=[{'id':c['id'],'messages':messages(c['question'])+[{'role':'assistant','content':read(gen_path(m,c['id']))['samples'][c['pick'][m]]['text']}]} for c in chosen]
        (CEPH/f'train_{m}.json').write_text(json.dumps(rows))
    (CEPH/'eval_sets.json').write_text(json.dumps({'gsm_test_cot':[{'id':q['id'],'answer':q['answer'],'messages':messages(q['question'])} for q in design['test']]}))
    print('matched',len(chosen),'of',len(design['train']));return chosen


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    models={m:Served(m,32) for m in SOURCES};write(DIRECTORY/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})
    async def generate(m,q):
        if gen_path(m,q['id']).exists(): return
        samples=await models[m].call(messages(q['question']),int(q['id'][5:])+1000,GEN_TEMPERATURE,GEN_TOKENS,n=SAMPLES);write(gen_path(m,q['id']),{'model':m,'id':q['id'],'samples':samples})
    jobs=[(m,q) for q in design['train'] for m in SOURCES];Random(1).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc='generate',mininterval=15): await f
    chosen=match(design);by_id={c['id']:c for c in chosen}
    tests=design['test'][:ICL_TEST]
    def examples(source,i):
        picks=Random(1000+i).sample(chosen,ICL_K)
        return [{'question':c['question'],'solution':read(gen_path(source,c['id']))['samples'][c['pick'][source]]['text']} for c in picks]
    async def evaluate(r,s,i):
        if icl_path(r,s,i).exists(): return
        q=tests[i];out=(await models[r].call(messages(q['question'],examples(s,i) if s!='initial' else ()),i,0.,GEN_TOKENS))[0]
        write(icl_path(r,s,i),{'reader':r,'source':s,'index':i,'answer':q['answer'],**out})
    jobs=[(r,s,i) for r in STUDENTS for s in ['initial']+SOURCES for i in range(len(tests))];Random(2).shuffle(jobs)
    for f in tqdm(asyncio.as_completed([evaluate(*j) for j in jobs]),total=len(jobs),desc='icl',mininterval=15): await f
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'matched':len(chosen)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())
