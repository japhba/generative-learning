"""Twelve constructive tasks with exact validation and no universal constant answer."""
import argparse
from dataclasses import dataclass,make_dataclass
from fractions import Fraction
from functools import lru_cache
import json
from math import gcd
from random import Random
import sys

from reasoning_gym.dataset import ProceduralDataset
from reasoning_gym.factory import register_dataset

FAMILIES=['bezout','congruences','linear_allocation','three_squares','rational_interval','balanced_sat','perfect_matching','hamiltonian_path','topological_order','equal_partition','supersequence','register_program']


def egcd(a,b):
    if b==0: return a,1,0
    g,x,y=egcd(b,a%b)
    return g,y,x-(a//b)*y


def apply_move(state,move,modulus):
    x,y=state
    if move=='A': return ((x+y)%modulus,y)
    if move=='B': return (x,(x+y)%modulus)
    assert move=='C'
    return (y,x)


def subsequence(short,long):
    iterator=iter(long)
    return all(any(c==value for value in iterator) for c in short)


def valid(answer,entry):
    try: v=json.loads(answer)
    except json.JSONDecodeError: return False
    m=entry['metadata']; family=m['family']
    if family=='congruences': return type(v) is int and m['lower']<=v<m['upper'] and all(v%mod==res for mod,res in m['constraints'])
    if not isinstance(v,list): return False
    if family=='supersequence': return all(type(c) is str and c in 'ABCD' and len(c)==1 for c in v) and len(v)<=m['limit'] and all(subsequence(s,v) for s in m['strings'])
    if family=='register_program':
        if any(type(c) is not str or c not in {'A','B','C'} for c in v) or len(v)>m['limit']: return False
        state=tuple(m['start'])
        for move in v: state=apply_move(state,move,m['modulus'])
        return state==tuple(m['target'])
    if any(type(x) is not int for x in v): return False
    if family=='bezout': return len(v)==2 and sum(a*x for a,x in zip(m['coefficients'],v))==1
    if family=='linear_allocation': return len(v)==2 and all(1<=x<=m['bound'] for x in v) and sum(a*x for a,x in zip(m['coefficients'],v))==m['target']
    if family=='three_squares': return len(v)==3 and all(1<=x<=m['bound'] for x in v) and sum(x*x for x in v)==m['target']
    if family=='rational_interval': return len(v)==2 and 1<=v[0]<=m['numerator_bound'] and 1<=v[1]<=m['denominator_bound'] and Fraction(m['lower'],1000)<Fraction(v[0],v[1])<Fraction(m['upper'],1000)
    if family=='balanced_sat': return len(v)==m['n'] and set(v)<={0,1} and sum(v)==m['n']//2 and all(any(v[abs(lit)-1]==int(lit>0) for lit in clause) for clause in m['clauses'])
    if family=='perfect_matching': return sorted(v)==list(range(1,m['n']+1)) and all(value in m['allowed'][i] for i,value in enumerate(v))
    if family in {'hamiltonian_path','topological_order'}:
        if sorted(v)!=list(range(1,m['n']+1)): return False
        edges={tuple(e) for e in m['edges']}
        if family=='hamiltonian_path': return all((a,b) in edges for a,b in zip(v,v[1:]))
        position={value:i for i,value in enumerate(v)}
        return all(position[a]<position[b] for a,b in edges)
    assert family=='equal_partition'
    return len(v)==len(m['numbers']) and set(v)=={1,2,3} and all(sum(value for value,group in zip(m['numbers'],v) if group==j)==m['capacity'] for j in [1,2,3])


def example(family,level,seed,index):
    rng=Random(seed+index); m={'family':family,'level':level}
    if family=='bezout':
        low,high=[(3,19),(20,99)][level]
        while True:
            a,b=rng.sample(range(low,high+1),2)
            if gcd(a,b)==1: break
        _,x,y=egcd(a,b); v=[x,y];m['coefficients']=[a,b]
        q=f'Return a JSON array [x,y] of integers satisfying {a}*x + {b}*y = 1. Negative integers are allowed.'
    elif family=='congruences':
        low,high=[(3,15),(13,41)][level]
        while True:
            a,b=rng.sample(range(low,high+1),2)
            if gcd(a,b)==1: break
        v=rng.randrange(1,5*a*b);m.update(lower=0,upper=5*a*b,constraints=[[a,v%a],[b,v%b]])
        q=f'Return one integer x, as a JSON number, such that 0 <= x < {5*a*b}, x mod {a} = {v%a}, and x mod {b} = {v%b}.'
    elif family=='linear_allocation':
        coefficient_bound,bound=[(15,40),(50,100)][level];a,b=rng.sample(range(3,coefficient_bound+1),2);v=[rng.randint(1,bound),rng.randint(1,bound)];target=a*v[0]+b*v[1]
        m.update(coefficients=[a,b],bound=bound,target=target)
        q=f'Return a JSON array [x,y] of integers such that 1 <= x,y <= {bound} and {a}*x + {b}*y = {target}.'
    elif family=='three_squares':
        bound=[15,40][level];v=[rng.randint(1,bound) for _ in range(3)];target=sum(x*x for x in v);m.update(bound=bound,target=target)
        q=f'Return a JSON array [x,y,z] of integers with 1 <= x,y,z <= {bound} and x*x + y*y + z*z = {target}. Any valid order is accepted.'
    elif family=='rational_interval':
        bound=[30,100][level];v=[rng.randint(1,3*bound),rng.randint(2,bound)];value=Fraction(*v);center=(1000*value).numerator//(1000*value).denominator;width=[4,1][level]
        m.update(numerator_bound=3*bound,denominator_bound=bound,lower=center-width,upper=center+width+1)
        q=f'Return a JSON array [p,q] of positive integers with p <= {3*bound}, q <= {bound}, and {m["lower"]}/1000 < p/q < {m["upper"]}/1000.'
    elif family=='balanced_sat':
        n=[6,10][level];v=[1]*(n//2)+[0]*(n//2);rng.shuffle(v);clauses=[]
        for _ in range(2*n):
            ids=rng.sample(range(1,n+1),3);clause=[j*rng.choice([-1,1]) for j in ids]
            if not any(v[abs(lit)-1]==int(lit>0) for lit in clause): clause[0]=-clause[0]
            clauses.append(clause)
        m.update(n=n,clauses=clauses)
        q=f'Return a JSON array of {n} bits for variables 1..{n}, with exactly {n//2} ones. Every listed clause must be true. A positive j means variable j; -j means its negation; each clause is an OR of its literals. Clauses: {json.dumps(clauses)}.'
    elif family=='perfect_matching':
        n=[5,8][level];v=list(range(1,n+1));rng.shuffle(v);allowed=[sorted({v[i]}|{j for j in range(1,n+1) if rng.random()<.35}) for i in range(n)];m.update(n=n,allowed=allowed)
        q=f'Assign jobs 1..{n} bijectively to workers 1..{n}. Every worker must get one allowed job and every job must be used once. Allowed job lists in worker order: {json.dumps(allowed)}. Return a JSON array of assigned job numbers in worker order.'
    elif family in {'hamiltonian_path','topological_order'}:
        n=([6,9] if family=='hamiltonian_path' else [7,11])[level];v=list(range(1,n+1));rng.shuffle(v)
        if family=='hamiltonian_path':
            edges=set(zip(v,v[1:]));edges|={(a,b) for a in v for b in v if a!=b and rng.random()<.15}
            q=f'Return a JSON array listing vertices 1..{n} exactly once, such that every consecutive pair is a directed edge. Directed edges: '
        else:
            edges={(a,b) for i,a in enumerate(v) for b in v[i+1:] if rng.random()<.4}
            q=f'Return a JSON array listing vertices 1..{n} exactly once, such that the source of every directed edge appears before its destination. Directed edges: '
        m.update(n=n,edges=sorted(edges));q+=json.dumps(m['edges'])+'.'
    elif family=='equal_partition':
        count,capacity=[(3,35),(4,70)][level];items=[]
        for group in [1,2,3]:
            cuts=[0,*sorted(rng.sample(range(1,capacity),count-1)),capacity]
            items.extend((b-a,group) for a,b in zip(cuts,cuts[1:]))
        rng.shuffle(items);numbers=[x for x,_ in items];v=[g for _,g in items];m.update(numbers=numbers,capacity=capacity)
        q=f'Assign every listed item to one of three groups so that each group sums to exactly {capacity}. Item values in order: {numbers}. Return a JSON array of group IDs (1, 2, or 3), one per item in this order.'
    elif family=='supersequence':
        length=[5,8][level];a=''.join(rng.choices('ABCD',k=length));b=''.join(rng.choices('ABCD',k=length))
        @lru_cache(None)
        def scs(i,j):
            if i==len(a): return b[j:]
            if j==len(b): return a[i:]
            if a[i]==b[j]: return a[i]+scs(i+1,j+1)
            return min([a[i]+scs(i+1,j),b[j]+scs(i,j+1)],key=lambda s:(len(s),s))
        text=scs(0,0);v=list(text);m.update(strings=[a,b],limit=len(v))
        q=f'Return a JSON array of letters from A,B,C,D, of length at most {len(v)}, containing both {a} and {b} as subsequences. A subsequence preserves order but need not be contiguous.'
    else:
        assert family=='register_program';modulus,limit=[(11,4),(23,6)][level]
        while True:
            start=(rng.randint(1,modulus-1),rng.randint(1,modulus-1));v=rng.choices(['A','B','C'],k=limit);state=start
            for move in v: state=apply_move(state,move,modulus)
            if state!=start: break
        m.update(start=start,target=state,modulus=modulus,limit=limit)
        q=f'Two registers start at (x,y)={start}; reach {state}. Moves: A sets x=(x+y) mod {modulus}; B sets y=(x+y) mod {modulus}; C swaps x and y. A and B leave the other register unchanged. Return a JSON array of at most {limit} move names (A, B, C) reaching the target.'
    entry={'question':q,'answer':json.dumps(v),'metadata':m};assert valid(entry['answer'],entry),(family,entry)
    return entry


@dataclass
class WitnessConfig:
    family:str='bezout';level:int=0;seed:int|None=None;size:int=1000
    def validate(self): assert self.family in FAMILIES and self.level in {0,1}


class WitnessDataset(ProceduralDataset):
    def __init__(self,config): super().__init__(config=config,seed=config.seed,size=config.size)
    def __getitem__(self,index): return example(self.config.family,self.config.level,self.seed,index)
    def score_answer(self,answer,entry): return float(valid(answer,entry))


def register():
    names=[]
    for family in FAMILIES:
        for level in [0,1]:
            name=f'witness_{family}_l{level}';config=make_dataclass(name+'Config',[('family',str,family),('level',int,level)],bases=(WitnessConfig,));register_dataset(name,WitnessDataset,config);names.append(name)
    return names


def main():
    names=register();mode=sys.argv.pop(1)
    if mode=='check':
        from tqdm.auto import tqdm
        for family in tqdm(FAMILIES,desc='Check constructive oracles'):
            for level in [0,1]:
                for i in range(200):
                    entry=json.loads(json.dumps(example(family,level,81137,i)))
                    assert valid(entry['answer'],entry)
                    for malformed in ['', '{}','null','true','[]','[true,false]','["incorrect"]']:
                        assert not valid(malformed,entry),(family,malformed,entry)
        print('4800 teacher certificates and 33600 malformed-answer checks passed.')
    elif mode=='prepare':
        from prepare_witness_task_search import main as prepare
        prepare(names)
    elif mode=='run':
        from run_task_search import main as run
        run()
    else:
        assert mode=='audit'
        from audit_task_search import main as audit
        audit()


if __name__=='__main__': main()
