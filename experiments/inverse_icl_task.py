"""Known forward subset-sum generator; hard inverse queries and compositional rehearsal."""

from collections import Counter
import itertools
import re

import numpy as np


def mask_text(mask): return ''.join(chr(65+i) for i,bit in enumerate(mask) if bit) or '-'


def decode_mask(text,dimension):
    clean=text.strip().upper()
    if clean=='-': return [0]*dimension
    if not re.fullmatch(r'[A-Z]+',clean) or len(set(clean))!=len(clean) or any(ord(letter)-65>=dimension for letter in clean): return None
    return [int(chr(65+i) in clean) for i in range(dimension)]


def make_world(seed,dimension,queries=8):
    assert dimension%2==0
    rng=np.random.default_rng(seed)
    weights=rng.choice(np.arange(101,999),size=dimension,replace=False)
    masks=np.asarray(list(itertools.product((0,1),repeat=dimension)))
    sums=masks@weights
    multiplicity=Counter(sums.tolist())
    half=dimension//2
    candidates=[i for i,(mask,total) in enumerate(zip(masks,sums,strict=True)) if multiplicity[int(total)]==1 and 0<int(mask[:half].sum())<half and 0<int(mask[half:].sum())<half]
    chosen=rng.choice(candidates,size=queries,replace=False).tolist()
    tests=[{'id':i+1,'total':int(sums[index]),'answer':mask_text(masks[index]),'mask':masks[index].tolist()} for i,index in enumerate(chosen)]
    teacher_pool=[i for i in range(len(masks)) if i not in chosen]
    teacher_indices=rng.choice(teacher_pool,size=128,replace=True).tolist()
    full=[{'subset':mask_text(masks[index]),'total':int(sums[index])} for index in teacher_indices]
    half_masks=list(itertools.product((0,1),repeat=half))
    order=rng.permutation(len(half_masks)).tolist()
    structured=[]
    for index in order:
        for side in (0,1):
            mask=[0]*dimension; mask[side*half:(side+1)*half]=half_masks[index]
            structured.append({'subset':mask_text(mask),'total':int(np.dot(mask,weights)),'side':side})
    random_parts=[structured[int(index)] for index in rng.integers(0,len(structured),size=128)]
    return {'seed':seed,'dimension':dimension,'weights':weights.tolist(),'tests':tests,'full':full,'structured':structured,'random_parts':random_parts}


def description(world):
    symbols=''.join(chr(65+i) for i in range(world['dimension']))
    mapping=', '.join(f'{letter}={weight}' for letter,weight in zip(symbols,world['weights'],strict=True))
    half=world['dimension']//2
    return ('A code is made by selecting a subset of symbols and adding their weights. Each symbol may occur at most once. '
            'A subset is written as its letters in alphabetical order; - means the empty subset, whose total is zero.\n'
            f'Weights: {mapping}.\n'
            f'For compositional calculation you may split the symbols into {symbols[:half]} and {symbols[half:]}. '
            'The total of a full subset equals the sum of its two partial-subset totals.\n'
            'The later task is to recover the selected symbols from a total. All test totals have exactly one valid subset.\n')


def rehearsal_prompt(world,subsets):
    return description(world)+'\nGenerate forward examples for these partial subsets by calculating their totals:\n'+', '.join(subsets)+'\nReturn exactly one line per requested subset, in the given order: SUBSET = TOTAL. No other text.'


def table_text(rows): return '\n'.join(f"{row['subset']} = {row['total']}" for row in rows)


def evaluation_prompt(world,examples,reasoning=False):
    examples_text='\nReference forward examples (subset = total):\n'+examples+'\n' if examples else ''
    queries='\n'.join(f"{item['id']}: {item['total']}" for item in world['tests'])
    instruction='Use concise calculations (aim for at most 120 words), then give the answer. ' if reasoning else 'Output only the answer lines, without reasoning, calculations, explanations, or code. '
    return (description(world)+examples_text+'\nDecode these totals. '+instruction+
            'Give one line per question: ANSWER id: SUBSET. Each answer must satisfy the sum exactly.\n'+queries)


def predictions(world,text):
    parsed={int(index):subset.upper() for index,subset in re.findall(r'^\s*(?:ANSWER\s+)?(\d+)\s*:\s*([A-Za-z]+|-)\b',text,re.I|re.M)}
    scores=[]
    for item in world['tests']:
        if item['id'] in parsed:
            subset=parsed[item['id']]; mask=decode_mask(subset,world['dimension'])
        else: subset=None; mask=None
        total=sum(bit*weight for bit,weight in zip(mask,world['weights'],strict=True)) if mask is not None else None
        scores.append({'id':item['id'],'prediction':subset,'target':item['answer'],'correct':total==item['total'],'parse_failure':mask is None})
    return scores


def audit_generated(world,subsets,text):
    rows=re.findall(r'^\s*([A-Za-z]+|-)\s*=\s*(\d+)\s*$',text,re.M)
    records=[]
    for subset,total in rows:
        mask=decode_mask(subset,world['dimension'])
        truth=sum(bit*weight for bit,weight in zip(mask,world['weights'],strict=True)) if mask is not None else None
        records.append({'subset':subset,'reported_total':int(total),'correct_total':truth,'correct':int(total)==truth})
    return {'expected_count':len(subsets),'parsed_count':len(rows),'requested_order':subsets==[subset for subset,total in rows],'correct_count':sum(row['correct'] for row in records),'records':records}
