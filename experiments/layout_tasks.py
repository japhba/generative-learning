"""Hidden-rule task families for the laid-out-reasoning study: every problem in an episode shares one latent rule,
shown only through a few input/answer examples, so working out the rule is what makes direct answering possible."""
from random import Random

WORDS=('apple bread chair dance eagle field grape house juice kite lemon money night ocean paper queen river stone table uncle voice water youth zebra '
       'beach candle dragon forest garden harbor island jacket kettle ladder magnet needle orange pencil rabbit saddle ticket velvet window yellow '
       'anchor basket camera desert engine flower guitar hammer insect jungle kitten letter marble number pepper rocket silver tunnel wallet winter').split()
FAMILIES=['operator','cipher','units']


def rule(family,rng):
    if family=='operator': return {'symbol':rng.choice('◇⊛⊙★⊕⊗'),'p':rng.choice([2,3,4,5]),'q':rng.choice([-3,-2,-1,1,2,3]),'r':rng.randint(-9,9)}
    if family=='cipher': return {'shift':rng.choice([k for k in range(3,24) if k!=13])}
    names=rng.sample(['vorp','blim','tazz','quon','frell','zink','glorp','mib'],3)
    return {'units':names,'ratios':[rng.randint(3,9),rng.randint(3,9)]}


def item(family,spec,rng,kind=None):
    if family=='operator':
        a,b=rng.randint(10,59),rng.randint(10,59)
        return f"Compute {a} {spec['symbol']} {b}.",str(spec['p']*a+spec['q']*b+spec['r'])
    if family=='cipher':
        word=rng.choice(WORDS);code=''.join(chr((ord(c)-97+spec['shift'])%26+65) for c in word)
        return f'Decode the secret word {code}.',word.upper()
    (big,mid,small),(r1,r2)=spec['units'],spec['ratios'];k=rng.randint(2,12)
    kind=kind or rng.choice(['big-mid','mid-small','big-small'])
    if kind=='big-mid': return f'How many {mid}s are in {k} {big}s?',str(k*r1)
    if kind=='mid-small': return f'How many {small}s are in {k} {mid}s?',str(k*r2)
    return f'How many {small}s are in {k} {big}s?',str(k*r1*r2)


def world(family,seed,support,practice,tests):
    rng=Random(seed);spec=rule(family,rng);seen=set();items=[];kinds=['big-mid','mid-small','big-small','big-small']  # units support reveals both ratios
    while len(items)<support+practice+tests:
        question,answer=item(family,spec,rng,kinds[len(items)] if family=='units' and len(items)<support else None)
        if question not in seen: seen.add(question);items.append({'question':question,'answer':answer})
    return {'family':family,'seed':seed,'rule':spec,'support':items[:support],'practice':items[support:support+practice],'tests':items[support+practice:]}


def normalize(answer): return answer.strip().strip('.').replace(',','').replace(' ','').upper()
def correct(answer,entry): return bool(answer) and normalize(answer)==normalize(entry['answer'])
