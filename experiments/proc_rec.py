"""PROC-REC (PROCREC_PROTOCOL.md): extra step-identification option sets and a verification probe for every reader x
source x N cell of the PROC confirmatory episodes (set A) and of 40 fresh episodes run through the full PROC pipeline (set B)."""
import asyncio
import hashlib
from pathlib import Path
from random import Random
import re

from tqdm.auto import tqdm

import proc_search as PS
from mlayout_api import Served
from mlayout_search import now,read,write
from proc_tasks import describe,make_input,options,run
from task_search_api import ROOT

OUT=ROOT.resolve()/'proc_rec';SETS={'proc_main':'proc_main_20261007','proc_main2':'proc_main2_20261007'};EXTRA=2;VERIFY=4
FILES=['PROCREC_PROTOCOL.md','proc_rec.py','proc_search.py','proc_tasks.py']
VTEXT={'en':{'q':'Below is how the same procedure might have been applied to a new number, showing the result after each step:\n{chain}\nIs every step consistent with the procedure used in the worked examples? Reply with only one line of the form "Answer: yes" or "Answer: no".'},
       'ja':{'q':'以下は、同じ手順を新しい数に適用した場合の各ステップの結果です：\n{chain}\nすべてのステップが解答例で使われた手順と一致していますか？「Answer: はい」または「Answer: いいえ」の一行だけで答えてください。'},
       'zh':{'q':'下面是把同样的步骤应用于一个新数时每一步的结果：\n{chain}\n每一步都与示例中使用的步骤一致吗？只用一行回答：“Answer: 是”或“Answer: 否”。'}}
YESNO=re.compile(r'(?:answer|答え|答案)[ \t]*[*_]*[ \t]*[:：][ \t]*[*_“"「]*[ \t]*(yes|no|はい|いいえ|是|否)',re.I)


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def said_yes(text):
    found=YESNO.findall(text);return None if not found else found[-1].lower() in ('yes','はい','是')


def items(w):
    """Extra items, seeded independently of the frozen episode: EXTRA more option sets per step; VERIFY chains on new inputs, half correct, half with one step replaced by a same-family alternative."""
    rng=Random(f"rec|{w['seed']}");sets=[[{'options':o,'answer':'ABCD'[a]} for o,a in (options(s,rng) for s in w['procedure'])] for _ in range(EXTRA)]
    used={x['input'] for x in w['pool']+w['tests']};chains=[]
    for v in range(VERIFY):
        ok=v%2==0
        while True:
            n=make_input(rng,w['procedure'])
            if n is None or n in used: continue
            steps=[dict(s) for s in w['procedure']]
            if not ok: j=rng.randrange(len(steps));steps[j]=rng.choice([o for o in options(steps[j],rng)[0] if o!=steps[j]])
            values=run(steps,n)
            if min(values)>0 and (ok or values!=run(w['procedure'],n)): break
        used.add(n);chains.append({'input':n,'values':values,'correct':ok})
    rng.shuffle(chains);return {'sets':sets,'chains':chains}


def probe_path(setname,r,n,c,kind): return OUT/setname/'probes'/r/f"{n}__{c.replace(':','-')}__{kind}.json"
def conditions(): return ['answers']+[f'src:{s}:{k}' for k in PS.COUNTS for s in PS.NAMES]


async def extra(setname):
    PS.DIRECTORY=ROOT.resolve()/setname;d=PS.DIRECTORY;done=read(d/'complete.json');matched=read(d/'matched.json');worlds={n:read(d/'worlds'/f'{n}.json') for n in done['usable']}
    extra_items={n:items(w) for n,w in worlds.items()};write(OUT/setname/'items.json',extra_items);models={m:Served(m,64) for m in PS.NAMES}
    async def ask(r,n,c,kind,text):
        if probe_path(setname,r,n,c,kind).exists(): return
        w=worlds[n];msg=[{'role':'user','content':PS.context(w,n,matched[n],c,PS.LANGUAGE[r])+'\n\n'+text}]
        write(probe_path(setname,r,n,c,kind),{'reader':r,'world':n,'condition':c,'kind':kind,**(await models[r].call(msg,w['seed']+70,0.,256))[0]})
    jobs=[]
    for n,w in worlds.items():
        for c in conditions():
            for r in PS.NAMES:
                lang=PS.LANGUAGE[r];t=PS.TEXT[lang]
                for a,opts in enumerate(extra_items[n]['sets']):
                    for j,p in enumerate(opts): jobs.append(ask(r,n,c,f'set{a}_step{j}',t['probe'].format(j=j+1,options='\n'.join(f"{'ABCD'[b]}. {describe(o,lang)}" for b,o in enumerate(p['options'])))))
                for v,ch in enumerate(extra_items[n]['chains']): jobs.append(ask(r,n,c,f'verify{v}',VTEXT[lang]['q'].format(chain=' → '.join(str(x) for x in [ch['input']]+ch['values']))))
    for f in tqdm(asyncio.as_completed(jobs),total=len(jobs),desc=f'{setname} extra probes',mininterval=30): await f
    for s in models.values(): await s.close()
    write(OUT/setname/'complete.json',{'time_uk':now(),'episodes':len(worlds)})


async def main():
    if (OUT/'design.json').exists(): assert read(OUT/'design.json')['source_sha256']==hashes(),'Frozen sources changed.'
    else: write(OUT/'design.json',{'time_uk':now(),'sets':SETS,'extra_sets':EXTRA,'verify':VERIFY,'source_sha256':hashes()})
    if not (OUT/'proc_main'/'complete.json').exists(): await extra('proc_main')
    PS.DIRECTORY=ROOT.resolve()/'proc_main2';PS.SEED=SETS['proc_main2']  # set B: the full PROC pipeline on fresh episodes
    if not (PS.DIRECTORY/'design.json').exists(): PS.prepare()
    if not (PS.DIRECTORY/'complete.json').exists(): await PS.run()
    if not (OUT/'proc_main2'/'complete.json').exists(): await extra('proc_main2')
    write(OUT/'complete.json',{'time_uk':now()})


if __name__=='__main__': asyncio.run(main())
