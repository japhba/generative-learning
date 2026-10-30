"""Reproducible procedural task specifications and disjoint rehearsal/test worlds."""

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import reasoning_gym
from reasoning_gym.factory import CURRICULA,DATASETS
from task_search_scoring import score_answer,teacher_answer


EXCLUSIONS={'composite':'A mixture constructor rather than a standalone task.',
            'arc_agi':'A fixed external corpus; this search starts with procedural tasks.',
            'codeio':'Requires an additional external corpus download.'}


def parameters_for(name,variant):
    _,config=DATASETS[name]
    if variant=='default': parameters=dataclasses.asdict(config())
    else:
        curriculum=CURRICULA[name](); curriculum.set_global_level(int(variant))
        parameters=dataclasses.asdict(curriculum.generate_configuration())
    del parameters['seed']; del parameters['size']
    return parameters


def specification(name,variant):
    parameters=json.loads(json.dumps(parameters_for(name,variant),default=str))
    return {'task':name,'variant':variant,'id':name+'__'+variant,'parameters':parameters}


def catalog():
    result=[]
    for name in sorted(DATASETS):
        if name in EXCLUSIONS: continue
        seen=set()
        for variant in ['default']+(['1','3'] if name in CURRICULA else []):
            spec=specification(name,variant); identity=json.dumps(spec['parameters'],sort_keys=True)
            if identity in seen: continue
            seen.add(identity); result.append(spec)
    return result


def seed_for(spec,phase,index): return int(hashlib.sha256(f"{spec['id']}|{phase}|{index}".encode()).hexdigest()[:8],16)


def dataset(world): return reasoning_gym.create_dataset(world['spec']['task'],**parameters_for(world['spec']['task'],world['spec']['variant']),seed=world['seed'],size=1000)


def make_world(spec,seed,practice_count=8,test_count=8,support_count=4):
    assert spec==specification(spec['task'],spec['variant'])
    data=reasoning_gym.create_dataset(spec['task'],**parameters_for(spec['task'],spec['variant']),seed=seed,size=1000)
    items=[]; seen=set(); needed=support_count+practice_count+test_count
    for index in range(500):
        entry=data[index]
        entry['answer']=teacher_answer(spec['task'],entry)
        assert isinstance(entry['question'],str) and isinstance(entry['answer'],str),'Task does not provide text teacher answers.'
        question=' '.join(entry['question'].split())
        if question in seen: continue
        assert score_answer(data,spec['task'],entry['answer'],entry)>=1-1e-9,'Generator/scorer disagree on the teacher answer.'
        entry=json.loads(json.dumps(entry,default=str))
        assert score_answer(data,spec['task'],entry['answer'],entry)>=1-1e-9,'Scorer cannot round-trip its own JSON metadata.'
        seen.add(question); items.append({'question':entry['question'],'answer':entry['answer'],'dataset_index':index,'metadata':entry['metadata']})
        if len(items)==needed: break
    assert len(items)==needed,'Too few distinct questions at this difficulty for a disjoint episode.'
    return {'spec':spec,'seed':seed,'support':items[:support_count],'practice':items[support_count:support_count+practice_count],
            'tests':[{**item,'id':index+1} for index,item in enumerate(items[support_count+practice_count:])]}


def initial_context(world):
    return f"Problem family: {world['spec']['task']}.\nExamples:\n"+'\n\n'.join(f"Problem: {item['question']}\nAnswer: {item['answer']}" for item in world['support'])


def final_prompt(world,examples,queries):
    additional='\n\nAdditional examples:\n'+'\n\n'.join(f"Problem: {item['question']}\nAnswer: {item['answer']}" for item in examples) if examples else ''
    return initial_context(world)+additional+'\n\nSolve the following independent problems. Return a JSON object with an answers array containing one answer string per problem, in the same order. Give only answers, without working or explanations.\n'+'\n\n'.join(f"Problem {query['id']}: {query['question']}" for query in queries)


def fixed_prompt(world,question):
    return initial_context(world)+f'\n\nSolve this new problem: {question}\nReturn a JSON object with a short work string and the final answer string. Keep working under eight lines.'


def free_prompt(world):
    return initial_context(world)+'\n\nProduce one new example of the same problem family and approximately the same difficulty as these examples. Give a complete, self-contained problem and its answer. Do not copy an initial example. Return a JSON object with question and answer strings, without a rationale or study advice.'


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--catalog',type=Path); parser.add_argument('--spec',type=Path); parser.add_argument('--output',type=Path)
    parser.add_argument('--seed',type=int); parser.add_argument('--practice',type=int,default=8); parser.add_argument('--tests',type=int,default=8); args=parser.parse_args()
    if args.catalog:
        args.catalog.write_text(json.dumps({'specifications':catalog(),'exclusions':EXCLUSIONS},indent=2)); return
    world=make_world(json.loads(args.spec.read_text()),args.seed,args.practice,args.tests)
    args.output.write_text(json.dumps(world,indent=2)); print(json.dumps({'task':world['spec']['id'],'seed':world['seed'],'questions':len(world['support'])+len(world['practice'])+len(world['tests'])}))


if __name__=='__main__': main()
