"""Exact synthetic tasks with the same rehearsal questions in every source condition."""

from collections import deque
import itertools
import re

import numpy as np

FAMILIES=('affine_inverse','modular_arithmetic','hidden_affine','hidden_linear','boolean_circuit','permutation_program',
          'automaton','stack_program','string_edits','shortest_paths','bayes_urn','base_multiplication',
          'bracket_completion','alien_sorting','substitution_grammar','parity_code')
DIGITS='0123456789ABCDEF'


def base_text(value,base):
    if value==0: return '0'
    result=''
    while value: value,digit=divmod(value,base); result=DIGITS[digit]+result
    return result


def context(family,level,rng):
    state={'family':family,'level':level}
    if family in ('affine_inverse','modular_arithmetic','hidden_affine','hidden_linear'):
        modulus=257 if family=='affine_inverse' and level==2 else 97
        coefficients=rng.integers(2,10 if level==1 else modulus-1,size=4).tolist()
        state.update(modulus=modulus,coefficients=coefficients)
        a,b,c,d=coefficients
        if family=='affine_inverse': description=f'Labels z are integers 1..{modulus-1}. The code is x=({a}*z+{b}) mod {modulus}. Recover z from x.'
        elif family=='modular_arithmetic': description=f'Compute the nonnegative remainder modulo 97. Follow parentheses and ordinary multiplication before addition.'
        elif family=='hidden_affine': description='A hidden fixed rule maps x to y=(a*x+b) mod 97. Infer the rule from the examples; a and b are the same for all examples and queries.'
        else: description='A hidden fixed rule maps a pair (x1,x2) to y=(a*x1+b*x2+c) mod 97. Infer the coefficients from the examples; they stay fixed.'
    elif family=='boolean_circuit':
        description='Evaluate the fully parenthesized Boolean expression. AND, OR, XOR, and NOT have their usual Boolean meanings. Return T for true or F for false.'
    elif family=='permutation_program':
        size=5 if level==1 else 7
        permutations=[rng.permutation(size).tolist() for _ in range(3)]
        state.update(size=size,permutations=permutations)
        table='; '.join(f'{op}: '+', '.join(f'{i}->{j}' for i,j in enumerate(p)) for op,p in zip('ABC',permutations,strict=True))
        description='A program transforms an integer state. Apply the named operations from left to right. Return the final state. '+table
    elif family=='automaton':
        size=4 if level==1 else 6
        transitions=rng.integers(size,size=(size,3)).tolist(); state.update(size=size,transitions=transitions)
        table='; '.join(f'state {i}: '+', '.join(f'{symbol}->{target}' for symbol,target in zip('abc',row,strict=True)) for i,row in enumerate(transitions))
        description='Read the input string left to right, starting at the stated state. Use this transition table and return the final state. '+table
    elif family=='stack_program':
        description='Execute a stack program from left to right, initially with an empty stack. PUSH n appends n. ADD and MUL pop the top two numbers and push their sum or product modulo 97. DUP copies the top. SWAP exchanges the top two. Return the top number after the program.'
    elif family=='string_edits':
        description='Transform the uppercase string by applying operations left to right. REV reverses it; LEFT rotates it left by one character; RIGHT rotates it right by one character; NEXT replaces A by B, B by C, C by D, and D by A. Return only the final string.'
    elif family=='shortest_paths':
        size=12; adjacency=[set() for _ in range(size)]
        for i in range(size): adjacency[i].add((i+1)%size); adjacency[(i+1)%size].add(i)
        for i,j in itertools.combinations(range(size),2):
            if rng.random()<(.09 if level==1 else .18): adjacency[i].add(j); adjacency[j].add(i)
        state['adjacency']=[sorted(row) for row in adjacency]
        edges=', '.join(f'{i}-{j}' for i,row in enumerate(adjacency) for j in sorted(row) if i<j)
        description='Find the minimum number of edges connecting the stated nodes in this undirected, unweighted graph. Return an integer. Edges: '+edges
    elif family=='bayes_urn':
        description='A hidden urn is A or B. Given the stated prior and independent draws with replacement, return which urn has the larger posterior probability, A or B. Red/blue probabilities are complementary. Break an exact tie in favor of A.'
    elif family=='base_multiplication':
        base=7 if level==1 else 13; state['base']=base
        description=f'Multiply the two integers written in base {base}. Return the product in base {base}, using digits {DIGITS[:base]} and no leading zeros. A=10, B=11, C=12 where needed.'
    elif family=='bracket_completion':
        description='The given string is a valid prefix of a properly nested bracket expression using (), [], and {}. Append only the necessary closing brackets to close all remaining open brackets. Return that closing sequence, or NONE if already complete.'
    elif family=='alien_sorting':
        alphabet=''.join(rng.permutation(list('ABCDEFGH'))); state['alphabet']=alphabet
        description=f'Sort the given words in ascending lexicographic order using this alphabet from smallest to largest: {alphabet}. Return the words separated by commas, with no spaces.'
    elif family=='substitution_grammar':
        rules={letter:''.join(rng.choice(list('ABC'),size=2)) for letter in 'ABC'}; state['rules']=rules
        description='Apply the substitutions simultaneously to every character for the specified number of rounds. Return the final string. Rules: '+', '.join(f'{a}->{b}' for a,b in rules.items())
    elif family=='parity_code':
        width=8 if level==1 else 12
        matrix=rng.integers(2,size=(3,width)).tolist(); state.update(width=width,matrix=matrix)
        description='Given the binary input vector, return its three parity-check bits. Each output bit is the dot product of its row with the input, reduced modulo 2. Return three bits with no spaces. Rows: '+'; '.join(''.join(map(str,row)) for row in matrix)
    else: raise ValueError(family)
    state['description']=description
    return state


def sample(state,rng):
    family=state['family']; level=state['level']
    if family=='affine_inverse':
        modulus=state['modulus']; z=int(rng.integers(1,modulus)); a,b,_,_=state['coefficients']; x=(a*z+b)%modulus
        question=f'Code x={x}'; answer=str(z)
    elif family=='modular_arithmetic':
        v=rng.integers(2,50 if level==1 else 100,size=5).tolist(); a,b,c,d,e=v
        if level==1: question=f'({a}*{b}+{c}*{d}) mod 97'; answer=str((a*b+c*d)%97)
        else: question=f'(({a}*{b}+{c})*({d}+{e}*{a})) mod 97'; answer=str(((a*b+c)*(d+e*a))%97)
    elif family=='hidden_affine':
        x=int(rng.integers(97)); a,b,_,_=state['coefficients']; question=f'x={x}'; answer=str((a*x+b)%97)
    elif family=='hidden_linear':
        x,y=rng.integers(97,size=2).tolist(); a,b,c,_=state['coefficients']; question=f'x1={x}, x2={y}'; answer=str((a*x+b*y+c)%97)
    elif family=='boolean_circuit':
        def expression(depth):
            if depth==0:
                value=bool(rng.integers(2)); return ('T' if value else 'F'),value
            a,av=expression(depth-1); b,bv=expression(depth-1); op=str(rng.choice(['AND','OR','XOR']))
            value={'AND':av and bv,'OR':av or bv,'XOR':av!=bv}[op]; text=f'({a} {op} {b})'
            if rng.random()<.25: text=f'(NOT {text})'; value=not value
            return text,value
        question,value=expression(3 if level==1 else 4); answer='T' if value else 'F'
    elif family=='permutation_program':
        start=int(rng.integers(state['size'])); value=start
        ops=rng.integers(3,size=5 if level==1 else 10).tolist()
        for op in ops: value=state['permutations'][op][value]
        question=f'Start {start}; operations '+''.join('ABC'[i] for i in ops); answer=str(value)
    elif family=='automaton':
        start=int(rng.integers(state['size'])); value=start; symbols=rng.integers(3,size=8 if level==1 else 18).tolist()
        for symbol in symbols: value=state['transitions'][value][symbol]
        question=f'Start {start}; input '+''.join('abc'[i] for i in symbols); answer=str(value)
    elif family=='stack_program':
        stack=[]; commands=[]
        for _ in range(10 if level==1 else 20):
            options=['PUSH']+(['DUP'] if stack else [])+(['ADD','MUL','SWAP'] if len(stack)>=2 else [])
            op=str(rng.choice(options))
            if op=='PUSH': value=int(rng.integers(2,20)); stack.append(value); commands.append(f'PUSH {value}')
            else:
                commands.append(op)
                if op=='DUP': stack.append(stack[-1])
                elif op=='SWAP': stack[-1],stack[-2]=stack[-2],stack[-1]
                else:
                    a=stack.pop(); b=stack.pop(); stack.append((a+b if op=='ADD' else a*b)%97)
        question='; '.join(commands); answer=str(stack[-1])
    elif family=='string_edits':
        original=''.join(rng.choice(list('ABCD'),size=6 if level==1 else 10)); value=original
        ops=rng.choice(['REV','LEFT','RIGHT','NEXT'],size=4 if level==1 else 9).tolist()
        for op in ops:
            if op=='REV': value=value[::-1]
            elif op=='LEFT': value=value[1:]+value[0]
            elif op=='RIGHT': value=value[-1]+value[:-1]
            else: value=value.translate(str.maketrans('ABCD','BCDA'))
        question=f'String {original}; operations '+', '.join(ops); answer=value
    elif family=='shortest_paths':
        start,target=rng.choice(12,size=2,replace=False).tolist(); distances={start:0}; queue=deque([start])
        while target not in distances:
            node=queue.popleft()
            for neighbor in state['adjacency'][node]:
                if neighbor not in distances: distances[neighbor]=distances[node]+1; queue.append(neighbor)
        question=f'From {start} to {target}'; answer=str(distances[target])
    elif family=='bayes_urn':
        pa,pb=rng.choice(np.arange(1,10),size=2,replace=False).tolist(); prior=int(rng.integers(1,10))
        total=int(rng.integers(4,12 if level==1 else 35)); red=int(rng.integers(total+1)); blue=total-red
        wa=prior*pa**red*(10-pa)**blue; wb=(10-prior)*pb**red*(10-pb)**blue
        question=f'Prior P(A)={prior}/10; P(red|A)={pa}/10; P(red|B)={pb}/10; observed {red} red and {blue} blue.'; answer='A' if wa>=wb else 'B'
    elif family=='base_multiplication':
        base=state['base']; a,b=rng.integers(base,base**(2 if level==1 else 3),size=2).tolist()
        question=f'{base_text(a,base)} * {base_text(b,base)}'; answer=base_text(a*b,base)
    elif family=='bracket_completion':
        stack=[]; chars=[]; opening='([{'; closing=')]}'
        for _ in range(12 if level==1 else 24):
            if stack and rng.random()<.35: chars.append(closing[stack.pop()])
            else: kind=int(rng.integers(3)); stack.append(kind); chars.append(opening[kind])
        question=''.join(chars); answer=''.join(closing[i] for i in reversed(stack)) or 'NONE'
    elif family=='alien_sorting':
        words=[''.join(rng.choice(list('ABCDEFGH'),size=3)) for _ in range(4 if level==1 else 8)]
        order={letter:i for i,letter in enumerate(state['alphabet'])}
        question=','.join(words); answer=','.join(sorted(words,key=lambda word:tuple(order[c] for c in word)))
    elif family=='substitution_grammar':
        original=''.join(rng.choice(list('ABC'),size=5)); value=original; rounds=2 if level==1 else 3
        for _ in range(rounds): value=''.join(state['rules'][c] for c in value)
        question=f'String {original}; {rounds} rounds'; answer=value
    elif family=='parity_code':
        bits=rng.integers(2,size=state['width']); value=(np.asarray(state['matrix'])@bits)%2
        question=''.join(map(str,bits)); answer=''.join(map(str,value))
    else: raise ValueError(family)
    return {'question':question,'answer':answer}


def make_world(family,level,seed,practice_count,test_count,support_count=4):
    rng=np.random.default_rng(seed); state=context(family,level,rng)
    unique={}
    while len(unique)<support_count+practice_count+test_count:
        item=sample(state,rng); unique[item['question']]=item
    items=list(unique.values())
    return {'family':family,'level':level,'seed':seed,'description':state['description'],'oracle_state':state,
            'support':items[:support_count],'practice':items[support_count:support_count+practice_count],
            'tests':[{**item,'id':i+1} for i,item in enumerate(items[support_count+practice_count:])]}


def canonical(text): return re.sub(r'\s+','',text).strip('`\"\'').rstrip('.').upper()


def extract_answer(text):
    clean=text.replace('**','').replace('`','')
    matches=re.findall(r'^\s*ANSWER\s*:\s*(.+?)\s*$',clean,re.I|re.M)
    return canonical(matches[-1]) if matches else None


def score(queries,text):
    clean=text.replace('**','').replace('`','')
    pairs=re.findall(r'^\s*(\d+)\s*:\s*(?:ANSWER\s*:\s*)?(.+?)\s*$',clean,re.I|re.M)
    candidates={}
    for index,answer in pairs: candidates.setdefault(int(index),set()).add(canonical(answer))
    answers={index:next(iter(values)) for index,values in candidates.items() if len(values)==1}
    return [{**item,'prediction':answers[item['id']] if item['id'] in answers else None,
             'correct':item['id'] in answers and answers[item['id']]==canonical(item['answer'])} for item in queries]


def initial_context(world):
    return world['description']+'\n\nInitial examples:\n'+'\n'.join(f"Input: {item['question']}\nAnswer: {item['answer']}" for item in world['support'])


def generation_prompt(world,item):
    return initial_context(world)+f"\n\nInput: {item['question']}\nSolve this problem. Use at most eight short lines of working; do not repeat the examples or verify by multiple methods. End with exactly one line ANSWER: followed by the answer and no explanation on that line."


def inference_prompt(world,answers,queries):
    examples='\n\nAdditional examples:\n'+'\n'.join(f"Input: {item['question']}\nAnswer: {answer}" for item,answer in zip(world['practice'][:len(answers)],answers,strict=True)) if answers else ''
    return initial_context(world)+examples+'\n\nSolve the following independent queries. Return only one line per query in the format id: answer. No reasoning or explanations.\n'+'\n'.join(f"{item['id']}: {item['question']}" for item in queries)


def messages(prompt): return [{'role':'system','content':'Follow the task definition and requested answer format.'},{'role':'user','content':prompt}]
