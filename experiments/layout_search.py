"""Laid-out reasoning: is reasoning one has written out oneself what makes a hidden rule usable?

Every episode has a hidden rule shown by K=4 input/answer examples. During practice each model reasons out loud on N=8
problems (free-form, ending in "Answer:"). At test, readers must answer directly with no working, after seeing one
source's practice either in full or reduced to its answers. The contrast (own full - own answers) - (foreign full -
foreign answers) isolates the value of laying out one's own reasoning beyond the answers it produced.
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

from layout_api import DIRECTORY,SPENT,Remote
from layout_tasks import FAMILIES,world as make_world
from task_search_api import LocalModel,digest

MODELS={'qwen35_9b':{'local':True,'family':'qwen'},
        'qwen3_30b':{'model':'qwen/qwen3-30b-a3b-instruct-2507','providers':['nebius/fp8'],'family':'qwen'},
        'llama31_8b':{'model':'meta-llama/llama-3.1-8b-instruct','providers':['novita/fp8'],'family':'llama','concurrency':8},
        'gemma4_26b':{'model':'google/gemma-4-26b-a4b-it','providers':['dekallm/bf16','nextbit/bf16','novita/bf16'],'reasoning_flag':True,'family':'google'},
        'phi4':{'model':'microsoft/phi-4','providers':['deepinfra/bf16'],'family':'microsoft'},
        'nemo_12b':{'model':'mistralai/mistral-nemo','providers':['deepinfra/fp8'],'family':'mistral'}}
NAMES=list(MODELS)
SOURCES=['initial','teacher']+[f'{m}:{kind}' for m in NAMES for kind in ['full','ans']]
EPISODES=6;SUPPORT=4;PRACTICE=8;TESTS=12
GEN_TOKENS=1024;ANSWER_TOKENS=32;GEN_TEMPERATURE=.8
WORKERS={'qwen35_9b':3,'gemma4_26b':24};DEFAULT_WORKERS=12
FILES=['LAYOUT_PROTOCOL.md','layout_api.py','layout_search.py','layout_tasks.py','task_search_api.py']
TITLES={'operator':'mystery operator','cipher':'secret code','units':'alien units'}
LINE=re.compile(r'^[ \t>#*_-]*(?:final[ \t]+)?answer[ \t]*[*_]*[ \t]*[:：][ \t]*(.*?)[ \t]*$',re.I|re.M)  # no newline crossing: '### Final Answer:\n\nAnswer: 7' -> '7'


def now(): return datetime.now(ZoneInfo('Europe/London')).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def read(path): return json.loads(path.read_text())
def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}


def extract(text):
    found=[a for a in LINE.findall(text) if a.strip()]
    answer=found[-1] if found else (text.strip().splitlines() or [''])[0]
    return answer.replace('**','').replace('`','').strip().rstrip('.').strip()


def header(world):
    return f"Problem family: {TITLES[world['family']]}. Every problem follows the same hidden rule.\nExamples:\n"+'\n\n'.join(f"Problem: {e['question']}\nAnswer: {e['answer']}" for e in world['support'])


def generation_prompt(world,question):
    return header(world)+f'\n\nSolve this new problem. Work out the hidden rule from the examples and reason step by step in your own words, then finish with a final line "Answer: <answer>".\nProblem: {question}\nSolution:'


def evaluation_prompt(world,practice,question):
    shown='\n\nWorked practice problems:\n'+'\n\n'.join(f"Problem: {p['question']}\nSolution: {p['solution']}" for p in practice) if practice else ''
    return header(world)+shown+f'\n\nSolve this new problem. Do not show any working: reply with only the final line "Answer: <answer>".\nProblem: {question}'


def prepare():
    directory=DIRECTORY;assert not (directory/'design.json').exists(),'Already frozen.';worlds={}
    for family in FAMILIES:
        for episode in range(EPISODES):
            seed=int(hashlib.sha256(f'layout_20261005|{family}|{episode}'.encode()).hexdigest()[:8],16)
            worlds[f'{family}__{episode:02}']={**make_world(family,seed,SUPPORT,PRACTICE,TESTS),'episode':episode}
    for name,w in worlds.items(): write(directory/'worlds'/f'{name}.json',w)
    write(directory/'design.json',{'time_uk':now(),'models':MODELS,'sources':SOURCES,'episodes':EPISODES,'support':SUPPORT,'practice':PRACTICE,'tests':TESTS,'generation_tokens':GEN_TOKENS,
                                   'answer_tokens':ANSWER_TOKENS,'generation_temperature':GEN_TEMPERATURE,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    for name in FILES: (directory/'source'/name).parent.mkdir(parents=True,exist_ok=True);(directory/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'worlds':len(worlds),'evaluations':len(worlds)*TESTS*len(SOURCES)*len(NAMES)}))


def generation_path(model,name,index): return DIRECTORY/'generations'/model/f'{name}__{index:02}.json'
def evaluation_path(reader,name,source,index): return DIRECTORY/'evaluations'/reader/f"{name}__{source.replace(':','-')}__{index:02}.json"


async def run():
    design=read(DIRECTORY/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed.'
    worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    local=LocalModel(48);remotes={(m,p):Remote(c['model'],p,c.get('reasoning_flag',False),c.get('concurrency',32)) for m,c in MODELS.items() if not c.get('local') for p in c['providers']}
    write(DIRECTORY/'inference.json',{'qwen35_9b':{'engine_fingerprint':local.fingerprint,'engine':local.engine}}|{f'{m}|{p}':{'engine_fingerprint':r.fingerprint,'engine':r.engine} for (m,p),r in remotes.items()})

    async def call(model,world,text,seed,temperature,tokens):
        if MODELS[model].get('local'):
            r=await local.call(text,None,seed,temperature=temperature,max_tokens=tokens);assert r['status']!='context_overflow',r
            return {'key':r['key'],'status':r['status'],'text':r['value']['answer'],'usage':r['usage']}
        providers=MODELS[model]['providers'];return await remotes[model,providers[world['episode']%len(providers)]].call(text,seed,temperature,tokens)

    def practice(world,name,source):
        if source=='initial': return []
        if source=='teacher': return [{'question':p['question'],'solution':f"Answer: {p['answer']}"} for p in world['practice']]
        model,kind=source.split(':');texts=[read(generation_path(model,name,i))['text'] for i in range(len(world['practice']))]
        return [{'question':p['question'],'solution':t if kind=='full' else f'Answer: {extract(t)}'} for p,t in zip(world['practice'],texts)]

    def path_of(job): return generation_path(job['model'],job['world'],job['index']) if job['stage']=='generate' else evaluation_path(job['model'],job['world'],job['source'],job['index'])

    async def one(job):
        world=worlds[job['world']]
        if path_of(job).exists(): return
        if job['stage']=='generate':
            text=generation_prompt(world,world['practice'][job['index']]['question']);seed=world['seed']+1000+job['index']
            result=await call(job['model'],world,text,seed,GEN_TEMPERATURE,GEN_TOKENS)
        else:
            text=evaluation_prompt(world,practice(world,job['world'],job['source']),world['tests'][job['index']]['question'])
            result=await call(job['model'],world,text,world['seed']+job['index'],0.,ANSWER_TOKENS)
        write(path_of(job),{**job,**result})

    async def stage(label,jobs):
        groups=defaultdict(list)
        for j in jobs: groups[j['model'],j['world'],j.get('source')].append(j)
        order=sorted(groups);Random(digest(label)).shuffle(order);queues={m:deque(k for k in order if k[0]==m) for m in NAMES}
        bar=tqdm(total=len(jobs),desc=label,mininterval=15);done=[0]
        async def tick(job):
            await one(job);bar.update();done[0]+=1
            if done[0]%256==0: write(DIRECTORY/f'{label}_progress.json',{'completed':done[0],'total':len(jobs),'time_uk':now(),'cost_usd':SPENT[0]})
        async def worker(queue):
            while queue:  # first request fills the prefix cache; the rest follow after a short stagger
                first,*rest=groups[queue.popleft()];head=asyncio.create_task(tick(first))
                if not path_of(first).exists(): await asyncio.sleep(2)
                await asyncio.gather(head,*(tick(j) for j in rest))
        await asyncio.gather(*(worker(queues[m]) for m in NAMES for _ in range(WORKERS.get(m,DEFAULT_WORKERS))));bar.close()

    await stage('generate',[{'stage':'generate','model':m,'world':n,'index':i} for n,w in worlds.items() for i in range(len(w['practice'])) for m in NAMES])
    await stage('evaluate',[{'stage':'evaluate','model':r,'world':n,'source':s,'index':i} for n,w in worlds.items() for s in SOURCES for i in range(len(w['tests'])) for r in NAMES])
    for model in [local,*remotes.values()]: await model.close()
    write(DIRECTORY/'complete.json',{'time_uk':now(),'cost_usd':SPENT[0]})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);args=parser.parse_args()
    prepare() if args.stage=='prepare' else asyncio.run(run())


if __name__=='__main__': main()
