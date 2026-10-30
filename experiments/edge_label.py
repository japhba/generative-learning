"""Labelled arm of the edge study (EDGE_PROTOCOL.md amendment 3): the main phase's mixed practice sets, with each attempt
marked "Check: correct." or "Check: wrong." (true answers not shown). Same readers, test problems and decoding."""
import asyncio
import hashlib
from pathlib import Path

from tqdm.auto import tqdm

from edge_search import DIRECTORY,INSTRUCTION,NAMES,TESTS,eval_path,gen_path,header,now,read,write
from mlayout_api import Served

FILES=['EDGE_PROTOCOL.md','edge_label.py','edge_search.py']


def messages(world,question,attempts):
    shown='\n\nEarlier attempts at practice problems, each checked against the true answer:\n'+'\n\n'.join(f"Problem: {a['question']}\nAttempt: {a['text']}\nCheck: {'correct' if a['correct'] else 'wrong'}." for a in attempts)
    return [{'role':'user','content':header(world)+shown+f'\n\n{INSTRUCTION}\nProblem: {question}'}]


async def main():
    d=DIRECTORY/'main2';assert (d/'complete.json').exists();out=DIRECTORY/'label2';hashes={f:hashlib.sha256((Path(__file__).parent/f).read_bytes()).hexdigest() for f in FILES}
    # amendment 4: runs on the re-matched main2 sets
    if (out/'design.json').exists(): assert read(out/'design.json')['source_sha256']==hashes,'Frozen sources changed.'
    else: write(out/'design.json',{'time_uk':now(),'from':'main2','source_sha256':hashes})
    matched=read(d/'matched.json');worlds={n:read(d/'worlds'/f'{n}.json') for n in matched if matched[n]['mixed']}
    models={m:Served(m,96) for m in NAMES};write(out/'inference.json',{m:{'engine_fingerprint':s.fingerprint,'engine':s.engine} for m,s in models.items()})
    def attempts(n,s): w=worlds[n];return [{'question':w['pool'][x['index']]['question'],'text':read(gen_path(d,s,n,x['index']))['samples'][x['sample'][s]]['text'],'correct':x['correct']} for x in matched[n]['mixed']]
    async def evaluate(r,n,s,i):
        path=eval_path(out,r,n,f'label:{s}',i)
        if path.exists(): return
        w=worlds[n];result=(await models[r].call(messages(w,w['tests'][i]['question'],attempts(n,s)),w['seed']+i,0.,1024))[0]
        write(path,{'reader':r,'world':n,'condition':f'label:{s}','index':i,**result})
    jobs=[(r,n,s,i) for n in worlds for s in NAMES for r in NAMES for i in range(TESTS)]
    for f in tqdm(asyncio.as_completed([evaluate(*j) for j in jobs]),total=len(jobs),desc='labelled tests',mininterval=15): await f
    for s in models.values(): await s.close()
    write(out/'complete.json',{'time_uk':now(),'episodes':len(worlds)})


if __name__=='__main__': asyncio.run(main())
