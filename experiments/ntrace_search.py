"""Native traces: do models learn in context better from their own free-form worked solutions than from other models'?

Seven readers/generators spanning families, training domains and sizes. Each model writes a free-form worked solution to
every rehearsal question with no teacher examples (K=0), so the rehearsal carries its own idiom; every reader then solves
test questions after N worked examples from one source. Phases: calibration (N=0, all candidate tasks; baseline-only task
choice) -> matrix (7 readers x 7 sources, fresh episodes). Correctness is computed only post-inference (ntrace_analyze.py).
"""
import argparse
import asyncio
from collections import defaultdict,deque
from datetime import datetime
import hashlib
import json
from pathlib import Path
from random import Random
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm

from ntrace_api import DIRECTORY,SPENT,Remote
from task_search_api import LocalModel,digest
from task_search_data import make_world,seed_for,specification

# Gemma alternates providers by episode (DekaLLM alone is rate-limited); every other model is pinned to one endpoint.
MODELS={'qwen35_9b':{'local':True,'family':'qwen'},
        'qwen3_30b':{'model':'qwen/qwen3-30b-a3b-instruct-2507','providers':['nebius/fp8'],'family':'qwen'},  # replaced Qwen3-Coder (Novita dropped 103/240 calibration texts)
        'llama31_8b':{'model':'meta-llama/llama-3.1-8b-instruct','providers':['novita/fp8'],'family':'llama','concurrency':8},
        'llama32_3b':{'model':'meta-llama/llama-3.2-3b-instruct','providers':['parasail/bf16'],'family':'llama','concurrency':8},
        'gemma4_26b':{'model':'google/gemma-4-26b-a4b-it','providers':['dekallm/bf16','nextbit/bf16','novita/bf16'],'reasoning_flag':True,'family':'google'},
        'phi4':{'model':'microsoft/phi-4','providers':['deepinfra/bf16'],'family':'microsoft'},
        'nemo_12b':{'model':'mistralai/mistral-nemo','providers':['deepinfra/fp8'],'family':'mistral'}}
NAMES=list(MODELS)
CANDIDATES=['gsm_symbolic','knights_knaves','calendar_arithmetic','time_intervals','family_relationships','letter_counting','aiw','basic_arithmetic',
            'decimal_chain_sum','simple_equations','products','number_sequence','count_primes','lcm','fraction_simplification','bitwise_arithmetic',
            'dice','base_conversion','leg_counting','circuit_logic']
MAX_TOKENS=1024;GEN_TEMPERATURE=.8
PHASES={'calibration':{'episodes':1,'practice':0,'tests':12,'counts':[0]},'matrix':{'episodes':6,'practice':8,'tests':12,'counts':[0,8]}}
WORKERS={'qwen35_9b':2,'gemma4_26b':24};DEFAULT_WORKERS=12  # few local prefix groups: long contexts must share the ~5 GB KV cache
FILES=['NTRACE_PROTOCOL.md','ntrace_api.py','ntrace_search.py','task_search_api.py','task_search_data.py','task_search_scoring.py']
INSTRUCTION='Solve the problem. Reason step by step in your own words, then finish with a final line of the form "Answer: <answer>", giving the answer in the format the problem requests.'


def now(): return datetime.now(ZoneInfo('Europe/London')).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def read(path): return json.loads(path.read_text())
def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}


def prompt(world,examples,question):
    shown='\n\nWorked examples:\n'+'\n\n'.join(f"Problem: {e['question']}\nSolution: {e['solution']}" for e in examples) if examples else ''
    return f"Problem family: {world['spec']['task']}.{shown}\n\n{INSTRUCTION}\nProblem: {question}\nSolution:"


def prepare(phase,tasks):
    directory=DIRECTORY/phase;assert not (directory/'design.json').exists(),f'{phase} already frozen.';config=PHASES[phase];worlds={}
    for task in tqdm(tasks,desc=f'Build {phase}'):
        spec=specification(task,'default')
        for episode in range(config['episodes']):
            world=make_world(spec,seed_for(spec,f'ntrace_{phase}_20261005',episode),config['practice'],config['tests'],0)
            worlds[f'{task}__{episode:02}']={**world,'episode':episode}
    for name,world in worlds.items(): write(directory/'worlds'/f'{name}.json',world)
    write(directory/'design.json',{'time_uk':now(),'phase':phase,**config,'tasks':tasks,'models':MODELS,'max_tokens':MAX_TOKENS,'generation_temperature':GEN_TEMPERATURE,
                                   'evaluation_temperature':0.,'instruction':INSTRUCTION,'worlds':{n:digest(w) for n,w in worlds.items()},'source_sha256':hashes()})
    for name in FILES: (directory/'source'/name).parent.mkdir(parents=True,exist_ok=True);(directory/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'phase':phase,'tasks':len(tasks),'worlds':len(worlds)}))


def generation_path(directory,model,name,index): return directory/'generations'/model/f'{name}__{index:02}.json'
def evaluation_path(directory,reader,name,source,n,index): return directory/'evaluations'/reader/f'{name}__{source}__n{n}__{index:02}.json'


async def run(phase):
    directory=DIRECTORY/phase;design=read(directory/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed; start a new phase.'
    worlds={name:read(directory/'worlds'/f'{name}.json') for name in design['worlds']};assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    local=LocalModel(48);remotes={(m,p):Remote(c['model'],p,c.get('reasoning_flag',False),c.get('concurrency',32)) for m,c in MODELS.items() if not c.get('local') for p in c['providers']}
    write(directory/'inference.json',{'qwen35_9b':{'engine_fingerprint':local.fingerprint,'engine':local.engine}}|{f'{m}|{p}':{'engine_fingerprint':r.fingerprint,'engine':r.engine} for (m,p),r in remotes.items()})

    async def call(model,world,text,seed,temperature):
        if MODELS[model].get('local'):
            result=await local.call(text,None,seed,temperature=temperature,max_tokens=MAX_TOKENS);assert result['status']!='context_overflow',result
            return {'key':result['key'],'status':result['status'],'text':result['value']['answer'],'usage':result['usage']}
        providers=MODELS[model]['providers'];return await remotes[model,providers[world['episode']%len(providers)]].call(text,seed,temperature,MAX_TOKENS)

    def examples(world,name,source,n):
        return [{'question':e['question'],'solution':read(generation_path(directory,source,name,i))['text']} for i,e in enumerate(world['practice'][:n])]

    async def one(job):
        model,name,index=job['model'],job['world'],job['index'];world=worlds[name]
        path=generation_path(directory,model,name,index) if job['stage']=='generate' else evaluation_path(directory,model,name,job['source'],job['n'],index)
        if path.exists(): return
        if job['stage']=='generate': text=prompt(world,[],world['practice'][index]['question']);seed=seed_for(world['spec'],f"{world['seed']}|ntrace|generation",index);temperature=GEN_TEMPERATURE
        else: text=prompt(world,examples(world,name,job['source'],job['n']),world['tests'][index]['question']);seed=world['seed']+index;temperature=0.
        result=await call(model,world,text,seed,temperature);write(path,{**job,**result})

    async def stage(label,jobs):
        # Group jobs sharing a prompt prefix; shuffle groups reproducibly; send each group's first request alone (prefix cache); per-model worker pools.
        groups=defaultdict(list)
        for j in jobs: groups[j['model'],j['world'],j.get('source'),j.get('n')].append(j)
        order=sorted(groups);Random(digest(label)).shuffle(order);queues={m:deque(k for k in order if k[0]==m) for m in NAMES}
        bar=tqdm(total=len(jobs),desc=f'{phase} {label}',mininterval=15);done=[0]
        async def tick(job):
            await one(job);bar.update();done[0]+=1
            if done[0]%256==0: write(directory/f'{label}_progress.json',{'completed':done[0],'total':len(jobs),'time_uk':now(),'cost_usd':SPENT[0]})
        def cached(job): return (generation_path(directory,job['model'],job['world'],job['index']) if job['stage']=='generate' else evaluation_path(directory,job['model'],job['world'],job['source'],job['n'],job['index'])).exists()
        async def worker(queue):
            while queue:  # the first request's prefill fills the prefix cache within ~1 s; the rest follow after a short stagger
                first,*rest=groups[queue.popleft()];head=asyncio.create_task(tick(first))
                if not cached(first): await asyncio.sleep(2)
                await asyncio.gather(head,*(tick(j) for j in rest))
        await asyncio.gather(*(worker(queues[m]) for m in NAMES for _ in range(WORKERS.get(m,DEFAULT_WORKERS))));bar.close()

    await stage('generate',[{'stage':'generate','model':m,'world':name,'index':i} for name,w in worlds.items() for i in range(len(w['practice'])) for m in NAMES])
    jobs=[{'stage':'evaluate','model':r,'world':name,'source':s,'n':n,'index':i} for name,w in worlds.items() for n in design['counts'] for s in (['initial'] if n==0 else NAMES)
          for i in range(len(w['tests'])) for r in NAMES]
    await stage('evaluate',jobs)
    for model in [local,*remotes.values()]: await model.close()
    write(directory/'complete.json',{'time_uk':now(),'evaluations':len(jobs),'cost_usd':SPENT[0]})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--phase',choices=list(PHASES),required=True);args=parser.parse_args()
    if args.stage=='prepare': prepare(args.phase,CANDIDATES if args.phase=='calibration' else read(DIRECTORY/'matrix_plan.json')['tasks'])
    else: asyncio.run(run(args.phase))


if __name__=='__main__': main()
