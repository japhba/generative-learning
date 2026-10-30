"""Hidden-rule families for the correctness-matched laid-out-reasoning study. Rules are discoverable from four
examples, but applying them without working (multi-digit mental arithmetic) stays hard, so direct answers avoid a ceiling."""
from random import Random

FAMILIES=['operator','units']


def rule(family,rng):
    if family=='operator':
        p,q=rng.choice([(p,q) for p in (2,3,4,5) for q in (1,2,3,4) if p!=q])
        return {'symbol':rng.choice('◇⊛⊙★⊕⊗'),'p':p,'q':q}
    names=rng.sample(['vorp','blim','tazz','quon','frell','zink','glorp','mib'],3)
    return {'units':names,'ratios':[rng.randint(4,12),rng.randint(4,12)]}


def item(family,spec,rng,kind=None):
    if family=='operator':
        a,b=rng.randint(11,49),rng.randint(11,49)
        return f"Compute {a} {spec['symbol']} {b}.",str(spec['p']*a+spec['q']*b)
    (big,mid,small),(r1,r2)=spec['units'],spec['ratios'];k=rng.randint(3,19)
    kind=kind or rng.choice(['big-mid','mid-small','big-small'])
    if kind=='big-mid': return f'How many {mid}s are in {k} {big}s?',str(k*r1)
    if kind=='mid-small': return f'How many {small}s are in {k} {mid}s?',str(k*r2)
    return f'How many {small}s are in {k} {big}s?',str(k*r1*r2)


def world(family,seed,support,pool,tests):
    rng=Random(seed);spec=rule(family,rng);seen=set();items=[];kinds=['big-mid','mid-small','big-small','big-small']  # units support reveals both ratios
    while len(items)<support+pool+tests:
        question,answer=item(family,spec,rng,kinds[len(items)] if family=='units' and len(items)<support else None)
        if question not in seen: seen.add(question);items.append({'question':question,'answer':answer})
    return {'family':family,'seed':seed,'rule':spec,'support':items[:support],'pool':items[support:support+pool],'tests':items[support+pool:]}


def normalize(answer): return answer.strip().strip('.').replace(',','').replace(' ','').upper()
def correct(answer,entry): return bool(answer) and normalize(answer)==normalize(entry['answer'])
