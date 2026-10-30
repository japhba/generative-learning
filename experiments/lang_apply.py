"""LANG-APPLY (LANGAPPLY_PROTOCOL.md): hold recognition fixed. After the practice and the rule probe, the assistant turn is
either the correct rule (inj) or the reader's own probe answer (self); then the reader solves the test problem."""
import asyncio
import hashlib
from pathlib import Path

from tqdm.auto import tqdm

import lang_search as LS
from mlayout_api import Served
from mlayout_search import now,read,write
from task_search_api import ROOT

OUT=ROOT.resolve()/'lang_apply';SETS={'lang_main':'lang_main_20261007','lang_main2':'lang_main2_20261007'}
FILES=['LANGAPPLY_PROTOCOL.md','lang_apply.py','lang_search.py']


def hashes(): return {f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
def rule_line(w): r=w['rule']['ratios'];return f'Ratios: {r[0]}, {r[0]*r[1]}, {r[0]*r[1]*r[2]}'
def eval_path(setname,r,n,c,i): return OUT/setname/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json"
def conditions(): return ['inj:none']+[f'{arm}:{s}:{k}' for arm in ['inj','self'] for k in LS.COUNTS for s in LS.NAMES]


def practice(d,w,matched,n,s,k):
    return [{'question':LS.render(w['pool'][q['index']],LS.LANGUAGE[s]),'text':read(d/'generations'/s/f"{n}__{q['index']:02}.json")['samples'][q['sample'][s]]['text']} for q in matched[n][:k]]


def conversation(d,w,matched,n,r,c,item):
    lang=LS.LANGUAGE[r];t=LS.TEXT[lang];arm,*rest=c.split(':')
    shown=[] if rest==['none'] else practice(d,w,matched,n,rest[0],int(rest[1]))
    said=rule_line(w) if arm=='inj' else read(d/'probes'/r/f"{n}__src-{rest[0]}-{rest[1]}.json")['text']
    return LS.probe(w,lang,shown)+[{'role':'assistant','content':said},{'role':'user','content':f"{t['instruction']}\n{t['problem']}: {LS.render(item,lang)}"}]


async def applied(setname):
    d=ROOT.resolve()/setname;done=read(d/'complete.json');matched=read(d/'matched.json');worlds={n:read(d/'worlds'/f'{n}.json') for n in done['usable']}
    models={m:Served(m,64) for m in LS.NAMES}
    async def evaluate(r,n,c,i):
        if eval_path(setname,r,n,c,i).exists(): return
        w=worlds[n];write(eval_path(setname,r,n,c,i),{'reader':r,'world':n,'condition':c,'index':i,**(await models[r].call(conversation(d,w,matched,n,r,c,w['tests'][i]),w['seed']+i,0.,LS.TOKENS))[0]})
    jobs=[evaluate(r,n,c,i) for n in worlds for c in conditions() for r in LS.NAMES for i in range(LS.TESTS)]
    for f in tqdm(asyncio.as_completed(jobs),total=len(jobs),desc=f'{setname} inj/self',mininterval=30): await f
    for s in models.values(): await s.close()
    write(OUT/setname/'complete.json',{'time_uk':now(),'episodes':len(worlds)})


async def main():
    if (OUT/'design.json').exists(): assert read(OUT/'design.json')['source_sha256']==hashes(),'Frozen sources changed.'
    else: write(OUT/'design.json',{'time_uk':now(),'sets':SETS,'conditions':conditions(),'source_sha256':hashes()})
    if not (OUT/'lang_main'/'complete.json').exists(): await applied('lang_main')
    LS.DIRECTORY=ROOT.resolve()/'lang_main2';LS.SEED=SETS['lang_main2']  # set B: the full LANG pipeline on fresh episodes
    if not (LS.DIRECTORY/'design.json').exists(): LS.prepare()
    if not (LS.DIRECTORY/'complete.json').exists(): await LS.run()
    if not (OUT/'lang_main2'/'complete.json').exists(): await applied('lang_main2')
    write(OUT/'complete.json',{'time_uk':now()})


if __name__=='__main__': asyncio.run(main())
