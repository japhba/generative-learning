"""Constrained arithmetic intervention: unjudged rehearsal and exact final scoring."""
import asyncio
import json
import numpy as np
from decomposition_task_search import solution as unconstrained_solution
from task_search_data import seed_for
from task_search_prompts import answer_prompt

RULE='At most one of the three integers may equal 1. Answers containing two or more 1s are invalid.'


def solution(text,entry):
    valid,identity=unconstrained_solution(text,entry)
    return valid and not identity,identity


async def evaluate(model,world,examples,arm,n):
    queries=world['tests']
    calls=await asyncio.gather(*(model.call(answer_prompt(world,examples,query),None,world['seed']+i,max_tokens=1024) for i,query in enumerate(queries)))
    records=[]
    for entry,call in zip(queries,calls):
        if call['status']=='context_overflow': return {'arm':arm,'n':n,'status':'context_overflow','calls':calls}
        answer=call['value']['answer'] if call['status']=='ok' else ''
        reward=float(solution(answer,entry)[0])
        records.append({'id':entry['id'],'answer':answer,'reward':reward,'correct':reward>=1-1e-9,'api_key':call['key'],'output_status':call['status']})
    return {'arm':arm,'n':n,'status':'ok','accuracy':float(np.mean([r['correct'] for r in records])),'mean_reward':float(np.mean([r['reward'] for r in records])),'predictions':records,'calls':calls}


async def run_world(model,world,config):
    assert config['arms']==['teacher','self_fixed']
    arms=config['arms']; ns=config['sample_counts']; count=max(ns); outputs=[]; generations={}
    # Generation sees only the initial examples and its assigned rehearsal question, never tests.
    jobs=[]; keys=[]
    for arm in arms:
        if arm=='self_fixed':
            for i,item in enumerate(world['practice'][:count]):
                jobs.append(model.call(answer_prompt(world,[],item),None,seed_for(world['spec'],f"{world['seed']}|fixed",i),temperature=.7,max_tokens=1024)); keys.append((arm,i,item['question']))
    calls=await asyncio.gather(*jobs)
    for (arm,i,question),call in zip(keys,calls):
        if arm not in generations: generations[arm]=[]
        fields={'answer'}
        valid=call['status']=='ok' and isinstance(call['value'],dict) and set(call['value'])==fields and all(isinstance(v,str) for v in call['value'].values())
        example={'question':question,'answer':call['value']['answer'] if valid else ''}
        generations[arm].append({'index':i,'example':example,'call':call})
    conditions=[('zero',0,[])] if 0 in ns else []
    for arm in arms:
        for n in ns:
            if n==0: continue
            examples=world['practice'][:n] if arm=='teacher' else [g['example'] for g in generations[arm][:n]]
            assert len(examples)==n
            conditions.append((arm,n,examples))
    outputs=await asyncio.gather(*(evaluate(model,world,examples,arm,n) for arm,n,examples in conditions))
    return {'spec':world['spec'],'seed':world['seed'],'generations':generations,'conditions':outputs}


