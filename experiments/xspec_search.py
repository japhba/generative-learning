"""Cross-task reader-source specificity search.

Qwen3.5-9B (pinned local) and Gemma 4 26B A4B (pinned OpenRouter endpoint) each answer the same rehearsal questions
offline; then each reader sees its own, the other model's, or teacher answers as in-context examples. Phases:
calibration (N=0 baselines for every catalog task) -> screen (4 episodes, N=16) -> confirmation (16 fresh episodes).
Validity is computed only post-inference (xspec_analyze.py); it never changes which samples are replayed.
"""
import argparse
import asyncio
from datetime import datetime
import hashlib
import json
from pathlib import Path
from collections import defaultdict,deque
from random import Random
import signal
from zoneinfo import ZoneInfo

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from task_search_api import LocalModel,ROOT,digest
from task_search_data import catalog,initial_context,make_world,seed_for
from task_search_prompts import answer_prompt
from xspec_api import DIRECTORY,Gemma

MODELS=['qwen','gemma']
SOURCES=MODELS+['teacher']
CONTEXT=16384;MAX_TOKENS=256
WORKERS={'qwen':8,'gemma':24}  # concurrent prefix groups per model
# 'optional' counts are dropped (and recorded) for a setting whose worst actual Qwen prompt would exceed the context.
# 'providers': Gemma BF16 endpoint per episode (episode mod len); all Gemma requests of an episode (generation and both sources) use one.
PHASES={'calibration':{'episodes':1,'practice':0,'tests':8,'counts':[0],'optional':[],'providers':['dekallm/bf16']},
        'screen':{'episodes':4,'practice':16,'tests':8,'counts':[0,16],'optional':[],'providers':['dekallm/bf16']},
        'confirmation':{'episodes':16,'practice':32,'tests':12,'counts':[0,2,8,16,32],'optional':[32],'providers':['dekallm/bf16','nextbit/bf16','novita/bf16']}}
FILES=['XSPEC_PROTOCOL.md','xspec_api.py','xspec_search.py','task_search_api.py','task_search_data.py','task_search_prompts.py','task_search_scoring.py','specificity_api.py']


def alarm(*_): raise TimeoutError('World construction exceeded 120 s.')
signal.signal(signal.SIGALRM,alarm)


def now(): return datetime.now(ZoneInfo('Europe/London')).isoformat()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def read(path): return json.loads(path.read_text())
def hashes(): return {name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in FILES}
def tokenizer(): return AutoTokenizer.from_pretrained(read(ROOT.resolve()/'runtime/model.json')['path'])
def worst_prompt(lengths,n): return lengths['initial']+n*(lengths['question']+MAX_TOKENS+16)+lengths['question']+80
def fits(lengths,n): return worst_prompt(lengths,n)+MAX_TOKENS<=CONTEXT
def length_eligible(lengths): return fits(lengths,16) and lengths['teacher_answer']<=200


def lengths_of(tok,world):
    count=lambda text:len(tok.encode(text,add_special_tokens=False))
    items=world['support_all']+world['practice']+world['tests']
    return {'initial':count(initial_context(world)),'question':max(count(e['question']) for e in items),'teacher_answer':max(count(e['answer']) for e in items)}


def build(phase,settings):
    """Fresh worlds for every setting; setting = (spec, K). K=0 and K=4 of one spec share questions."""
    config=PHASES[phase];tok=tokenizer();worlds={};exclusions={};lengths={};cache={}
    for setting in tqdm(settings,desc=f'Build {phase}'):
        spec=setting['spec']
        for episode in range(config['episodes']):
            seed=seed_for(spec,f'xspec_{phase}_20261005',episode)
            if (spec['id'],seed) not in cache:
                signal.alarm(120)  # some reasoning_gym generators loop forever for particular seeds (number_sequence level 3)
                try: cache[spec['id'],seed]=make_world(spec,seed,config['practice'],config['tests'],4)
                except Exception as error: cache[spec['id'],seed]=repr(error)  # catalog sweep: an unconstructible task is a recorded design exclusion
                finally: signal.alarm(0)
            world=cache[spec['id'],seed]
            if isinstance(world,str): exclusions[setting['id']]=world;break
            world={**world,'support_all':world['support'],'support':world['support'][:setting['k']],'setting':setting['id'],'k':setting['k'],'episode':episode}
            worlds[f"{setting['id']}__{episode:02}"]=world
            current=lengths_of(tok,world);previous=lengths.get(setting['id'],current)
            lengths[setting['id']]={key:max(current[key],previous[key]) for key in current}
        if setting['id'] in exclusions: worlds={k:v for k,v in worlds.items() if v['setting']!=setting['id']};lengths.pop(setting['id'],None)
    return worlds,exclusions,lengths


def prepare(phase,plan):
    directory=DIRECTORY/phase;assert not (directory/'design.json').exists(),f'{phase} already frozen.'
    worlds,exclusions,lengths=build(phase,plan['settings'])
    for name,world in worlds.items(): write(directory/'worlds'/f'{name}.json',world)
    design={'time_uk':now(),'phase':phase,**PHASES[phase],'models':MODELS,'max_tokens':MAX_TOKENS,'generation_temperature':.7,'evaluation_temperature':0.,
            'settings':[s for s in plan['settings'] if s['id'] not in exclusions],'exclusions':exclusions,'lengths':lengths,
            'worlds':{name:digest(world) for name,world in worlds.items()},'source_sha256':hashes(),'plan_sha256':digest(plan)}
    write(directory/'design.json',design)
    for name in FILES: (directory/'source'/name).parent.mkdir(parents=True,exist_ok=True);(directory/'source'/name).write_bytes((Path(__file__).parent/name).read_bytes())
    print(json.dumps({'phase':phase,'settings':len(design['settings']),'worlds':len(worlds),'exclusions':len(exclusions)}))


def generation_path(directory,model,name,index): return directory/'generations'/model/f'{name}__{index:02}.json'
def evaluation_path(directory,reader,name,source,n,index): return directory/'evaluations'/reader/f'{name}__{source}__n{n}__{index:02}.json'


def examples_for(directory,world,name,source,n):
    if source=='teacher': return world['practice'][:n]
    return [{'question':item['question'],'answer':read(generation_path(directory,source,name,i))['answer']} for i,item in enumerate(world['practice'][:n])]


async def run(phase):
    directory=DIRECTORY/phase;design=read(directory/'design.json');assert design['source_sha256']==hashes(),'Frozen sources changed; start a new phase.'
    worlds={name:read(directory/'worlds'/f'{name}.json') for name in design['worlds']}
    assert all(digest(worlds[n])==h for n,h in design['worlds'].items())
    if phase=='calibration': worlds={n:w for n,w in worlds.items() if length_eligible(design['lengths'][w['setting']])}  # protocol criteria 1-2
    providers=PHASES[phase]['providers'];qwen=LocalModel(16);gemmas={p:Gemma(32,p) for p in providers}
    model_for=lambda model,world:qwen if model=='qwen' else gemmas[providers[world['episode']%len(providers)]]
    write(directory/'inference.json',{'qwen':{'engine_fingerprint':qwen.fingerprint,'engine':qwen.engine},'gemma':{p:{'engine_fingerprint':g.fingerprint,'engine':g.engine} for p,g in gemmas.items()}})

    async def one(job):
        model,name,index=job['model'],job['world'],job['index'];world=worlds[name]
        path=generation_path(directory,model,name,index) if job['stage']=='generate' else evaluation_path(directory,model,name,job['source'],job['n'],index)
        if path.exists(): return
        if job['stage']=='generate':
            prompt=answer_prompt(world,[],world['practice'][index]);seed=seed_for(world['spec'],f"{world['seed']}|xspec|generation|k{world['k']}",index);temperature=.7
        else:
            prompt=answer_prompt(world,examples_for(directory,world,name,job['source'],job['n']),world['tests'][index]);seed=world['seed']+index;temperature=0.
        call=await model_for(model,world).call(prompt,None,seed,temperature=temperature,max_tokens=MAX_TOKENS)
        assert call['status']!='context_overflow',(job,call)
        write(path,{**job,'answer':call['value']['answer'],'call':call})

    async def stage(name,jobs):
        # Jobs sharing a prompt prefix (model, world, source, N) form one group; group order is shuffled reproducibly. Each group sends its
        # first request alone, then the rest, so vLLM's prefix cache is filled once. Each model has its own worker pool (scheduling only).
        groups=defaultdict(list)
        for j in jobs: groups[j['model'],j['world'],j.get('source'),j.get('n')].append(j)
        order=sorted(groups);Random(digest(name)).shuffle(order);queues={m:deque(k for k in order if k[0]==m) for m in MODELS}
        bar=tqdm(total=len(jobs),desc=f'{phase} {name}',mininterval=15);done=[0]
        async def tick(job):
            await one(job);bar.update();done[0]+=1
            if done[0]%256==0: write(directory/f'{name}_progress.json',{'completed':done[0],'total':len(jobs),'time_uk':now(),'gemma_cost_usd':gemmas[providers[0]].cost})
        async def worker(queue):
            while queue:
                first,*rest=groups[queue.popleft()];await tick(first);await asyncio.gather(*(tick(j) for j in rest))
        await asyncio.gather(*(worker(queues[m]) for m in MODELS for _ in range(WORKERS[m])));bar.close()

    await stage('generate',[{'stage':'generate','model':m,'world':name,'index':i} for name,w in worlds.items() for i in range(len(w['practice'])) for m in MODELS])
    jobs=[{'stage':'evaluate','model':r,'world':name,'source':s,'n':n,'index':i} for name,w in worlds.items() for n in design['counts'] for s in (['initial'] if n==0 else SOURCES)
          for i in range(len(w['tests'])) for r in MODELS]
    # Length-only design check before any evaluation (never uses correctness). Smaller N prompts are prefixes of the largest required one.
    tok=qwen.tokenizer;overflow=set();largest=max(n for n in design['counts'] if n not in design['optional'])
    for job in tqdm([j for j in jobs if j['model']=='qwen' and (j['n']==largest or j['n'] in design['optional'])],desc='Context check',mininterval=15):
        world=worlds[job['world']];prompt=answer_prompt(world,examples_for(directory,world,job['world'],job['source'],job['n']),world['tests'][job['index']])
        rendered=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        if len(tok.encode(rendered,add_special_tokens=False))+MAX_TOKENS>CONTEXT: overflow.add((world['setting'],job['n']))
    assert all(n in design['optional'] for _,n in overflow),f'Context design check failed: {sorted(overflow)}'
    write(directory/'dropped_counts.json',sorted(overflow));jobs=[j for j in jobs if (worlds[j['world']]['setting'],j['n']) not in overflow]
    await stage('evaluate',jobs)
    for model in [qwen,*gemmas.values()]: await model.close()
    write(directory/'complete.json',{'time_uk':now(),'generations':sum(len(w['practice']) for w in worlds.values())*2,'evaluations':len(jobs),'gemma_cost_usd':gemmas[providers[0]].cost})


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','run']);parser.add_argument('--phase',choices=list(PHASES),required=True);args=parser.parse_args()
    if args.stage=='prepare':
        if args.phase=='calibration': plan={'settings':[{'id':f"{spec['id']}__k{k}",'spec':spec,'k':k} for spec in catalog() for k in [0,4]]}
        else: plan=read(DIRECTORY/f'{args.phase}_plan.json')
        prepare(args.phase,plan)
    else: asyncio.run(run(args.phase))


if __name__=='__main__': main()
