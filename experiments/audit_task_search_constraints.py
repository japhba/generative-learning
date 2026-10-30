"""Independent checks of selected-task oracles and full-credit answer constraints."""

import argparse
from collections import Counter,defaultdict,deque
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re

from tqdm.auto import tqdm

from task_search_api import ROOT

TASKS={'graph_color','maze','count_primes','palindrome_partitioning','path_star','self_reference','count_bits','coin_flip','color_cube_rotation'}


@lru_cache
def prime(number): return number>=2 and all(number%divisor for divisor in range(2,math.isqrt(number)+1))


def partitions(string):
    if not string: return {()}
    return {(string[:i],)+tail for i in range(1,len(string)+1) if string[:i]==string[:i][::-1] for tail in partitions(string[i:])}


def shortest_path(adjacency,start,goal):
    queue=deque([(start,)]); seen={start}
    while queue:
        path=queue.popleft()
        if path[-1]==goal: return path
        for neighbor in adjacency[path[-1]]:
            if neighbor not in seen: seen.add(neighbor); queue.append(path+(neighbor,))
    raise AssertionError('Teacher question has no path.')


def check_oracle(task,entry):
    metadata=entry['metadata']; answer=entry['answer']
    if task=='graph_color': assert answer_valid(task,answer,entry); return
    if task=='count_bits': expected=str(metadata['number'].bit_count())
    elif task=='count_primes': expected=str(sum(prime(n) for n in range(metadata['start'],metadata['end']+1)))
    elif task=='coin_flip':
        n,k=metadata['num_trials'],metadata['k_heads']; numerator=math.comb(n,k) if metadata['problem_type']=='exact' else sum(math.comb(n,i) for i in range(k,n+1))
        assert metadata['rational']=={'numerator':numerator,'denominator':2**n}; assert answer_valid(task,answer,entry); return
    elif task=='palindrome_partitioning':
        assert {tuple(p) for p in json.loads(answer)}==partitions(metadata['string']); return
    elif task=='self_reference':
        patterns=[r'At least (\d+) of these 7 statements are true',r'At most (\d+) of these 7 statements are false',r'Exactly (\d+) of these 7 statements are true',r'Exactly (\d+) of these 7 statements are false']
        a,b,c,d=[int(re.search(pattern,entry['question'])[1]) for pattern in patterns]
        count=0
        for true in range(8):
            false=7-true; claims=[true>=a,false<=b,true==c,false==d,(true==c)!=(false==d),prime(true),false>1 and not prime(false)]
            count+=sum(claims)==true
        expected=str(count)
    elif task=='maze':
        grid=metadata['grid']; cells={(r,c) for r,row in enumerate(grid) for c,ch in enumerate(row) if ch!=metadata['wall']}
        start,goal=[next((r,c) for r,c in cells if grid[r][c]==metadata[name]) for name in ['start','goal']]
        adjacency={cell:[(cell[0]+dr,cell[1]+dc) for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)] if (cell[0]+dr,cell[1]+dc) in cells] for cell in cells}
        expected=str(len(shortest_path(adjacency,start,goal))-1)
    elif task=='path_star':
        text=entry['question'].rsplit('Solve the following task:',1)[1]; adjacency=defaultdict(list)
        for u,v in re.findall(r'\|(\d+) (\d+)',text): adjacency[u].append(v); adjacency[v].append(u)
        start,goal=re.search(r'/(\d+) (\d+) =',text).groups(); expected=' '.join(shortest_path(adjacency,start,goal))
    elif task=='color_cube_rotation':
        colors=metadata['initial_state'].copy(); cycles={'front':['top','front','bottom','back'],'back':['top','back','bottom','front'],'left':['top','left','bottom','right'],'right':['top','right','bottom','left']}
        for rotation in metadata['rotations']:
            old=colors.copy()
            if rotation=='bottom':
                for destination,source in [('top','bottom'),('bottom','top'),('front','back'),('back','front')]: colors[destination]=old[source]
            else:
                cycle=cycles[rotation]
                for destination,source in zip(cycle,cycle[1:]+cycle[:1]): colors[destination]=old[source]
        expected=colors[metadata['target_side']]
    else: raise ValueError(task)
    assert answer==expected,(task,entry,expected)


def answer_valid(task,answer,entry):
    if task=='coin_flip':
        try: value=Fraction(answer.replace(',',''))
        except (ValueError,ZeroDivisionError): return False
        rational=entry['metadata']['rational']; target=Fraction(rational['numerator'],rational['denominator'])
        return abs(value-target)<=Fraction(1,10000)
    if task=='graph_color':
        try: colors=json.loads(answer)
        except json.JSONDecodeError: return False
        puzzle=entry['metadata']['puzzle']; vertices={str(v) for v in puzzle['vertices']}
        return isinstance(colors,dict) and set(colors)==vertices and all(type(color) in (int,float) and color in puzzle['color_options'] for color in colors.values()) and all(colors[str(u)]!=colors[str(v)] for u,v in puzzle['edges'])
    if task=='palindrome_partitioning':
        try: parts=json.loads(answer)
        except json.JSONDecodeError: return False
        if not isinstance(parts,list) or any(not isinstance(part,list) or any(not isinstance(word,str) for word in part) for part in parts): return False
        return {tuple(p) for p in parts}==partitions(entry['metadata']['string'])
    # All other selected native full-credit scorers already require the oracle answer.
    return True


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); args=parser.parse_args(); directory=ROOT.resolve()/args.phase
    preflight=json.loads((directory/'preflight.json').read_text()); names=[r['world'] for r in preflight if r['status']=='ready']
    assert len(list((directory/'evaluations').glob('*.json')))==len(names),'No partial-phase outcome audit.'
    records=[]; changes=[]; checked=Counter()
    for name in tqdm(names,desc='Independent constraint audit',mininterval=10):
        result=json.loads((directory/'evaluations'/name).read_text()); task=result['spec']['task']
        if task not in TASKS: continue
        world=json.loads((directory/'worlds'/name).read_text()); conditions=[]
        for entry in world['support']+world['practice']+world['tests']: check_oracle(task,entry); checked[task]+=1
        for condition in result['conditions']:
            if condition['status']!='ok': conditions.append({k:v for k,v in condition.items() if k!='calls'}); continue
            predictions=[]
            for prediction,entry in zip(condition['predictions'],world['tests']):
                valid=answer_valid(task,prediction['answer'],entry)
                correct=valid if task=='coin_flip' else prediction['correct'] and valid
                predictions.append({**prediction,'native_correct':prediction['correct'],'correct':correct,'reward':float(correct)})
                if correct!=prediction['correct']: changes.append({'spec':world['spec']['id'],'seed':world['seed'],'arm':condition['arm'],'n':condition['n'],'question':entry['question'],'teacher_answer':entry['answer'],'prediction':prediction,'constraint_correct':correct})
            conditions.append({'arm':condition['arm'],'n':condition['n'],'status':'ok','accuracy':sum(p['correct'] for p in predictions)/len(predictions),'predictions':predictions})
        records.append({'spec':world['spec'],'seed':world['seed'],'conditions':conditions})
    report={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checked_teacher_examples':dict(checked),'scoring_changes':changes,'evaluations':records,
            'metric':'Native full credit intersected with graph/partition output constraints; coin-flip absolute probability tolerance replaces the erroneous prefix score.',
            'cube_limitation':'Oracle validation follows the library convention that bottom-to-top swaps front/back and retains left/right; that rotation axis is not explicit in the question.'}
    (directory/'constraint_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps({'checked_teacher_examples':dict(checked),'scoring_changes':len(changes),'changes_by_task':dict(Counter(r['spec'] for r in changes))},indent=2))


if __name__=='__main__': main()
