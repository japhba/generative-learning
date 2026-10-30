"""Pilot: own vs same-language vs other-language worked practice, with a recognition probe.

Six ~8B instruction models dominant in three languages (English: Llama-3.1-8B, OLMo-3-7B; Japanese: Llama-3.1-Swallow-8B,
Gemma-2-Llama-Swallow-9B; Chinese: GLM-4-9B, InternLM3-8B). Every prompt is in the model's own language; problems are symbolic
(mystery operator, four alien units), so only the language of the reasoning differs between sources. Each model writes
4 step-by-step attempts per pool problem in its own language; practice sets are all correct, on identical problems for
every source. Readers solve new problems step by step in their own language after 0, answers-only, or N in {1, 4}
worked solutions from one source. A separate probe call with the same practice asks the reader to state the hidden rule.
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

import mlayout_tasks as MT
from edge2_search import units4
from mlayout_api import Served
from mlayout_search import now,read,write
from task_search_api import ROOT,digest

DIRECTORY=ROOT.resolve()/'lang_main';SEED='lang_main_20261007'  # lang_apply.py reassigns both for its fresh set; LANG_PROTOCOL.md; pilots in lang_pilot (operator unusable, probe asked adjacent ratios) and lang_pilot2
LANGUAGE={'llama31_8b':'en','olmo3_7b':'en','swallow_8b':'ja','gswallow_9b':'ja','glm4_9b':'zh','internlm3_8b':'zh'}
NAMES=list(LANGUAGE);FAMILIES=['units4'];EPISODES=40;POOL=12;TESTS=8;SAMPLES=8;COUNTS=[1,2,4];TOKENS=1024;TEMPERATURE=.8
TEXT={'en':{'title':{'operator':'mystery operator','units4':'alien units'},'header':'Problem family: {t}. Every problem follows the same hidden rule.\nExamples:\n','problem':'Problem','answer':'Answer','solution':'Solution','practice':'Worked practice problems:\n',
            'operator':'Compute {a} {s} {b}.','units':'How many {j}s are in {k} {i}s?',
            'instruction':'Solve this new problem. Work out the hidden rule from the examples and reason step by step in English, then finish with a final line "Answer: <number>".',
            'probe_operator':'Based on the examples and solutions above, what is the hidden rule? Reply with only one line of the form "Rule: a {s} b = p*a + q*b", with the numbers p and q filled in.',
            'probe_units4':'Based on the examples and solutions above, how many {u1}s, how many {u2}s and how many {u3}s are in one {u0}? Reply with only one line of the form "Ratios: x, y, z".'},
      'ja':{'title':{'operator':'謎の演算子','units4':'異星の単位'},'header':'問題の種類：{t}。すべての問題は同じ隠れた規則に従います。\n例：\n','problem':'問題','answer':'答え','solution':'解答','practice':'練習問題の解答例：\n',
            'operator':'{a} {s} {b} を計算してください。','units':'{k} {i} は何 {j} ですか？',
            'instruction':'この新しい問題を解いてください。例から隠れた規則を見つけ、日本語で段階的に推論し、最後の行を「Answer: <数>」としてください。',
            'probe_operator':'上の例と解答から、隠れた規則は何ですか？「Rule: a {s} b = p*a + q*b」の形で、p と q に数を入れた一行だけで答えてください。',
            'probe_units4':'上の例と解答から、1 {u0} は何 {u1}、何 {u2}、何 {u3} ですか？「Ratios: x, y, z」の形の一行だけで答えてください。'},
      'zh':{'title':{'operator':'神秘运算符','units4':'外星单位'},'header':'题型：{t}。所有题目都遵循同一个隐藏规则。\n例子：\n','problem':'题目','answer':'答案','solution':'解答','practice':'练习题的解答示例：\n',
            'operator':'计算 {a} {s} {b}。','units':'{k} 个 {i} 等于多少个 {j}？',
            'instruction':'请解答这道新题。从例子中找出隐藏规则，用中文逐步推理，最后一行写“Answer: <数字>”。',
            'probe_operator':'根据上面的例子和解答，隐藏规则是什么？只用一行回答，格式为“Rule: a {s} b = p*a + q*b”，并填入数字 p 和 q。',
            'probe_units4':'根据上面的例子和解答，1 个 {u0} 分别等于多少个 {u1}、多少个 {u2}、多少个 {u3}？只用一行回答，格式为“Ratios: x, y, z”。'}}
FILES=['LANG_PROTOCOL.md','lang_search.py','edge2_search.py','mlayout_tasks.py','mlayout_api.py','mlayout_search.py']
ANSWER=re.compile(r'^[ \t>#*_-]*(?:final[ \t]+)?(?:answer|答え|答案|最终答案)[ \t]*[*_]*[ \t]*[:：][ \t]*(.*?)[ \t]*$',re.I|re.M)


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def numbers(line): return [float(x) for x in re.findall(r'-?\d+(?:\.\d+)?',line.replace(',','').replace('，',''))]
def correct(text,entry):
    lines=[a for a in ANSWER.findall(text) if numbers(a)];return bool(lines) and abs(numbers(lines[-1])[-1]-float(entry['answer']))<1e-6
def probe_correct(text,world):
    """Pilot 2: cumulative factors from the largest unit (1 u0 = r0 u1 = r0*r1 u2 = r0*r1*r2 u3); numbers after '=' when the line has any."""
    line=[l for l in text.splitlines() if re.search(r'(rule|ratios)\s*[:：]',l,re.I)]
    if not line: return False
    body=re.split(r'[:：]',line[-1],1)[1];found=[numbers(x)[0] for x in body.split('=')[1:] if numbers(x)] if '=' in body else numbers(body)
    if world['family']=='operator': return found[-2:]==[world['rule']['p'],world['rule']['q']]
    r=world['rule']['ratios'];return found[:3]==[r[0],r[0]*r[1],r[0]*r[1]*r[2]]


def make_world(family,seed):
    """Language-neutral items: each item keeps its parameters so it can be rendered in any language."""
    rng=Random(seed)
    if family=='operator':
        p,q=rng.choice([(p,q) for p in (2,3,4,5) for q in (1,2,3,4) if p!=q]);rule={'symbol':rng.choice('◇⊛⊙★⊕⊗'),'p':p,'q':q};seen=set();items=[]
        while len(items)<4+POOL+TESTS:
            a,b=rng.randint(11,49),rng.randint(11,49)
            if (a,b) not in seen: seen.add((a,b));items.append({'kind':'operator','a':a,'b':b,'s':rule['symbol'],'answer':str(p*a+q*b)})
        return {'family':family,'seed':seed,'rule':rule,'support':items[:4],'pool':items[4:4+POOL],'tests':items[4+POOL:]}
    w=units4(seed,4,POOL,TESTS);parse=re.compile(r'How many (\w+)s (?:are in|make) (\d+) (\w+)s\?')
    def item(e):  # "How many X make k Y?" (reverse) asks the same as "How many X are in k Y?"
        j,k,i=parse.fullmatch(e['question']).groups();return {'kind':'units','j':j,'k':int(k),'i':i,'answer':e['answer']}
    return {**{k:w[k] for k in ['family','seed','rule']},'support':[item(e) for e in w['support']],'pool':[item(e) for e in w['pool']],'tests':[item(e) for e in w['tests']]}


def render(item,lang):
    t=TEXT[lang];return t['operator'].format(**item) if item['kind']=='operator' else t['units'].format(**item)


def header(world,lang):
    t=TEXT[lang];return t['header'].format(t=t['title'][world['family']])+'\n\n'.join(f"{t['problem']}: {render(e,lang)}\n{t['answer']}: {e['answer']}" for e in world['support'])


def context(world,lang,practice):
    t=TEXT[lang];shown='\n\n'+t['practice']+'\n\n'.join(f"{t['problem']}: {p['question']}\n{t['solution']}: {p['text']}" for p in practice) if practice else ''
    return header(world,lang)+shown


def ask(world,lang,practice,item): t=TEXT[lang];return [{'role':'user','content':context(world,lang,practice)+f"\n\n{t['instruction']}\n{t['problem']}: {render(item,lang)}"}]
def probe(world,lang,practice):
    t=TEXT[lang];u=world['rule'].get('units',['']*4);text=t['probe_'+world['family']].format(s=world['rule'].get('symbol',''),u0=u[0],u1=u[1],u2=u[2],u3=u[3])
    return [{'role':'user','content':context(world,lang,practice)+f'\n\n{text}'}]


def gen_path(m,n,i): return DIRECTORY/'generations'/m/f'{n}__{i:02}.json'
def eval_path(r,n,c,i): return DIRECTORY/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"
def probe_path(r,n,c): return DIRECTORY/'probes'/r/f"{n}__{c.replace(':','-')}.json"
def conditions(): return ['initial','answers']+[f'src:{s}:{k}' for k in COUNTS for s in NAMES]


def prepare():
    assert not (DIRECTORY/'design.json').exists();worlds={f'{f}__{e:02}':make_world(f,int(hashlib.sha256(f'{SEED}|{f}|{e}'.encode()).hexdigest()[:8],16)) for f in FAMILIES for e in range(EPISODES)}
    for n,w in worlds.items(): write(DIRECTORY/'worlds'/f'{n}.json',w)
    write(DIRECTORY/'design.json',{'time_uk':now(),'language':LANGUAGE,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()});print(len(worlds),'worlds')


def match(w,n):
    """Pool problems (in order) where every source has a finished correct attempt; per source the correct attempt closest to its own median length."""
    chosen=[]
    for i in range(POOL):
        good={m:[(k,len(s['text'])) for k,s in enumerate(read(gen_path(m,n,i))['samples']) if s['status']=='ok' and correct(s['text'],w['pool'][i])] for m in NAMES}
        if all(good.values()): chosen.append({'index':i,'sample':{m:min(v,key=lambda x:(abs(x[1]-np.median([l for _,l in v])),x[0]))[0] for m,v in good.items()}})
    return chosen


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes();worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']}
    models={m:Served(m,64) for m in NAMES};write(DIRECTORY/'inference.json',{m:{'engine':s.engine} for m,s in models.items()})
    async def generate(m,n,i):
        if gen_path(m,n,i).exists(): return
        w=worlds[n];write(gen_path(m,n,i),{'model':m,'world':n,'index':i,'samples':await models[m].call(ask(w,LANGUAGE[m],[],w['pool'][i]),w['seed']+1000+i,TEMPERATURE,TOKENS,n=SAMPLES)})
    jobs=[(m,n,i) for n in worlds for i in range(POOL) for m in NAMES]
    for f in tqdm(asyncio.as_completed([generate(*j) for j in jobs]),total=len(jobs),desc='attempts',mininterval=15): await f
    matched={n:match(w,n) for n,w in worlds.items()};write(DIRECTORY/'matched.json',matched);print({n:len(v) for n,v in matched.items()},flush=True)
    usable=[n for n in worlds if len(matched[n])>=max(COUNTS)]
    def practice(n,c,lang):
        w=worlds[n];qs=matched[n][:max(COUNTS)]
        if c=='initial': return []
        if c=='answers': return [{'question':render(w['pool'][q['index']],lang),'text':f"Answer: {w['pool'][q['index']]['answer']}"} for q in qs]
        _,s,k=c.split(':');return [{'question':render(w['pool'][q['index']],LANGUAGE[s]),'text':read(gen_path(s,n,q['index']))['samples'][q['sample'][s]]['text']} for q in qs[:int(k)]]
    async def evaluate(r,n,c,i):
        if eval_path(r,n,c,i).exists(): return
        w=worlds[n];write(eval_path(r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**(await models[r].call(ask(w,LANGUAGE[r],practice(n,c,LANGUAGE[r]),w['tests'][i]),w['seed']+i,0.,TOKENS))[0]})
    async def interrogate(r,n,c):
        if probe_path(r,n,c).exists(): return
        w=worlds[n];write(probe_path(r,n,c),{'reader':r,'world':n,'condition':c,**(await models[r].call(probe(w,LANGUAGE[r],practice(n,c,LANGUAGE[r])),w['seed'],0.,256))[0]})
    jobs=[evaluate(r,n,c,i) for n in usable for c in conditions() for r in NAMES for i in range(TESTS)]+[interrogate(r,n,c) for n in usable for c in conditions() for r in NAMES]
    for f in tqdm(asyncio.as_completed(jobs),total=len(jobs),desc='tests+probes',mininterval=15): await f
    for s in models.values(): await s.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'usable':usable})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())
