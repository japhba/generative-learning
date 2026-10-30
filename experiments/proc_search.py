"""PROC: recognition vs deployment of a demonstrated procedure, own vs same-language vs other-language demonstrations.

Writers are told a hidden multi-step number procedure (in their own language) and write worked executions of it. Readers
see only N in {1, 2, 4} worked examples from one source (identical inputs; every source's example correct and showing
every intermediate value), then either answer a 4-way question per step ("which describes step j?": recognition) or apply
the procedure to a new number (deployment). Also: input/output pairs only, and the procedure stated in the reader's
language without examples. Same six language-dominant models as LANG.
"""
import argparse
import asyncio
import hashlib
from pathlib import Path
import re

import numpy as np
from tqdm.auto import tqdm

import lang_search as LS
from mlayout_api import Served
from mlayout_search import now,read,write
from proc_tasks import describe,world as make_world
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'proc_main';SEED='proc_main_20261007'  # PROC_PROTOCOL.md; pilots in proc_pilot, proc_pilot2, proc_pilot3  # pilots 1-2 (LANG panel): 1/12 and 3/16 usable; Llama-3.1-8B and Swallow-8B execute the procedures correctly only 4-12% of the time
LANGUAGE={'olmo3_7b':'en','phi4_mini':'en','gswallow_9b':'ja','abeja_qwen7b':'ja','glm4_9b':'zh','internlm3_8b':'zh'};NAMES=list(LANGUAGE)  # weak writers replaced
LENGTHS=[3,4];EPISODES=20;POOL=12;TESTS=8;SAMPLES=16;COUNTS=[1,2,4];TOKENS=1024;TEMPERATURE=.8
FILES=['PROC_PROTOCOL.md','proc_search.py','proc_tasks.py','lang_search.py','mlayout_api.py']
TEXT={'en':{'given':'Here is a procedure that turns a number into another number:\n{steps}\n\nApply the procedure to the number {x}. Explain each step in your own words in English and give the result of every step, then finish with a final line "Answer: <number>".',
            'header':'Below are worked examples of a secret procedure that turns a number into another number.\n\n','example':'Number: {x}\nWorked solution: {text}','pair':'Number: {x}\nResult: {y}',
            'stated':'Here is a procedure that turns a number into another number:\n{steps}','test':'Apply the same procedure to the number {y}. Work step by step in English, then finish with a final line "Answer: <number>".',
            'probe':'Which of the following describes step {j} of the procedure used in these examples?\n{options}\nReply with only one line of the form "Answer: <letter>".'},
      'ja':{'given':'ある数を別の数に変える手順があります：\n{steps}\n\nこの手順を数 {x} に適用してください。各ステップを日本語で自分の言葉で説明し、各ステップの結果を示し、最後の行を「Answer: <数>」としてください。',
            'header':'以下は、ある数を別の数に変える秘密の手順の解答例です。\n\n','example':'数: {x}\n解答: {text}','pair':'数: {x}\n結果: {y}',
            'stated':'ある数を別の数に変える手順があります：\n{steps}','test':'同じ手順を数 {y} に適用してください。日本語で段階的に考え、最後の行を「Answer: <数>」としてください。',
            'probe':'これらの例で使われている手順のステップ {j} を説明しているのは次のどれですか？\n{options}\n「Answer: <記号>」の形の一行だけで答えてください。'},
      'zh':{'given':'下面是一个把一个数变成另一个数的步骤：\n{steps}\n\n请对数 {x} 应用这个步骤。用中文以你自己的话解释每一步，并给出每一步的结果，最后一行写“Answer: <数字>”。',
            'header':'下面是一些解答示例，展示了一个把一个数变成另一个数的秘密步骤。\n\n','example':'数: {x}\n解答: {text}','pair':'数: {x}\n结果: {y}',
            'stated':'下面是一个把一个数变成另一个数的步骤：\n{steps}','test':'请对数 {y} 应用同样的步骤。用中文逐步推理，最后一行写“Answer: <数字>”。',
            'probe':'下列哪一项描述了这些示例所用步骤中的第 {j} 步？\n{options}\n只用一行回答，格式为“Answer: <字母>”。'}}
CHOICE=re.compile(r'(?:answer|答え|答案)[ \t]*[*_]*[ \t]*[:：][ \t]*[*_(（]*[ \t]*([A-DＡ-Ｄ])',re.I)


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def steps(w,lang): return '\n'.join(f'{j+1}. {describe(s,lang)}' for j,s in enumerate(w['procedure']))
def shows_steps(text,item): return all(re.search(rf'(?<![\d,]){v}(?![\d,])',text.replace(',','').replace('，','')) for v in item['values'])
def correct(text,item): return LS.correct(text,item)
def chosen(text): found=CHOICE.findall(text);return found[-1].translate(str.maketrans('ＡＢＣＤ','ABCD')).upper() if found else ''


def prepare():
    assert not (DIRECTORY/'design.json').exists()
    worlds={f'L{L}__{e:02}':make_world(int(hashlib.sha256(f'{SEED}|{L}|{e}'.encode()).hexdigest()[:8],16),L,POOL,TESTS) for L in LENGTHS for e in range(EPISODES)}
    for n,w in worlds.items(): write(DIRECTORY/'worlds'/f'{n}.json',w)
    write(DIRECTORY/'design.json',{'time_uk':now(),'language':LANGUAGE,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()});print(len(worlds),'worlds')


def gen_path(m,n,i): return DIRECTORY/'generations'/m/f'{n}__{i:02}.json'
def eval_path(r,n,c,i): return DIRECTORY/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"
def probe_path(r,n,c,j): return DIRECTORY/'probes'/r/f"{n}__{c.replace(':','-')}__{j}.json"
def conditions(): return ['answers','stated']+[f'src:{s}:{k}' for k in COUNTS for s in NAMES]


def match(w,n):
    """Pool inputs (in order) where every source has a finished attempt with the right answer that shows every intermediate value; per source the one closest to its own median length."""
    chosen_=[]
    for i,item in enumerate(w['pool']):
        good={m:[(k,len(s['text'])) for k,s in enumerate(read(gen_path(m,n,i))['samples']) if s['status']=='ok' and correct(s['text'],item) and shows_steps(s['text'],item)] for m in NAMES}
        if all(good.values()): chosen_.append({'index':i,'sample':{m:min(v,key=lambda x:(abs(x[1]-np.median([l for _,l in v])),x[0]))[0] for m,v in good.items()}})
    return chosen_


def context(w,n,matched,c,lang):
    t=TEXT[lang]
    if c=='stated': return t['stated'].format(steps=steps(w,lang))
    if c=='answers': return t['header']+'\n\n'.join(t['pair'].format(x=w['pool'][q['index']]['input'],y=w['pool'][q['index']]['answer']) for q in matched[:max(COUNTS)])
    _,s,k=c.split(':');return t['header']+'\n\n'.join(t['example'].format(x=w['pool'][q['index']]['input'],text=read(gen_path(s,n,q['index']))['samples'][q['sample'][s]]['text']) for q in matched[:int(k)])


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes();worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']}
    models={m:Served(m,64) for m in NAMES};write(DIRECTORY/'inference.json',{m:{'engine':s.engine} for m,s in models.items()})
    async def generate(m,n,i):
        if gen_path(m,n,i).exists(): return
        w=worlds[n];t=TEXT[LANGUAGE[m]];msg=[{'role':'user','content':t['given'].format(steps=steps(w,LANGUAGE[m]),x=w['pool'][i]['input'])}]
        write(gen_path(m,n,i),{'model':m,'world':n,'index':i,'samples':await models[m].call(msg,w['seed']+1000+i,TEMPERATURE,TOKENS,n=SAMPLES)})
    jobs=[(m,n,i) for n in worlds for i in range(POOL) for m in NAMES]
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc='worked examples',mininterval=15): await f
    matched={n:match(w,n) for n,w in worlds.items()};write(DIRECTORY/'matched.json',matched);print({n:len(v) for n,v in matched.items()},flush=True)
    usable=[n for n in worlds if len(matched[n])>=max(COUNTS)]
    async def evaluate(r,n,c,i):
        if eval_path(r,n,c,i).exists(): return
        w=worlds[n];t=TEXT[LANGUAGE[r]];msg=[{'role':'user','content':context(w,n,matched[n],c,LANGUAGE[r])+'\n\n'+t['test'].format(y=w['tests'][i]['input'])}]
        write(eval_path(r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**(await models[r].call(msg,w['seed']+i,0.,TOKENS))[0]})
    async def interrogate(r,n,c,j):
        if probe_path(r,n,c,j).exists(): return
        w=worlds[n];t=TEXT[LANGUAGE[r]];p=w['probes'][j];opts='\n'.join(f"{'ABCD'[a]}. {describe(o,LANGUAGE[r])}" for a,o in enumerate(p['options']))
        msg=[{'role':'user','content':context(w,n,matched[n],c,LANGUAGE[r])+'\n\n'+t['probe'].format(j=j+1,options=opts)}]
        write(probe_path(r,n,c,j),{'reader':r,'world':n,'condition':c,'step':j,'answer':p['answer'],**(await models[r].call(msg,w['seed']+50+j,0.,256))[0]})
    jobs=[evaluate(r,n,c,i) for n in usable for c in conditions() for r in NAMES for i in range(TESTS)]+[interrogate(r,n,c,j) for n in usable for c in conditions() if c!='stated' for r in NAMES for j in range(worlds[n]['length'])]
    for f in tqdm(asyncio.as_completed(jobs),total=len(jobs),desc='tests+probes',mininterval=15): await f
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'usable':usable})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())
