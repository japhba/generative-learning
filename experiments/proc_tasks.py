"""Hidden multi-step number procedures for the recognition-vs-deployment study (PROC). Each episode has a procedure of L
steps alternating digit permutations and arithmetic; it is stated only to the models that write the worked examples."""
from random import Random

PERM=['rev','sortdesc','sortasc','swapends'];ARITH=['add','sub','mul']
TEXT={'en':{'rev':'Reverse the order of its digits (leading zeros are dropped).','sortdesc':'Rearrange its digits from largest to smallest.','sortasc':'Rearrange its digits from smallest to largest (leading zeros are dropped).',
            'swapends':'Swap its first and last digits (leading zeros are dropped).','add':'Add {k}.','sub':'Subtract {k}.','mul':'Multiply by {k}.'},
      'ja':{'rev':'各桁の数字の並びを逆にする（先頭の0は取り除く）。','sortdesc':'各桁の数字を大きい順に並べ替える。','sortasc':'各桁の数字を小さい順に並べ替える（先頭の0は取り除く）。',
            'swapends':'最初の桁と最後の桁の数字を入れ替える（先頭の0は取り除く）。','add':'{k} を足す。','sub':'{k} を引く。','mul':'{k} を掛ける。'},
      'zh':{'rev':'把各位数字的顺序颠倒（去掉开头的0）。','sortdesc':'把各位数字按从大到小的顺序重新排列。','sortasc':'把各位数字按从小到大的顺序重新排列（去掉开头的0）。',
            'swapends':'交换第一位和最后一位数字（去掉开头的0）。','add':'加上 {k}。','sub':'减去 {k}。','mul':'乘以 {k}。'}}


def apply(step,n):
    op,k=step['op'],step.get('k');s=str(n)
    if op=='rev': return int(s[::-1])
    if op=='sortdesc': return int(''.join(sorted(s,reverse=True)))
    if op=='sortasc': return int(''.join(sorted(s)))
    if op=='swapends': return int(s[-1]+s[1:-1]+s[0]) if len(s)>1 else n
    return {'add':n+k,'sub':n-k,'mul':n*k}[op]


def run(procedure,n):
    values=[]
    for step in procedure: n=apply(step,n);values.append(n)
    return values


def describe(step,lang): return TEXT[lang][step['op']].format(k=step.get('k'))


def arith(rng,op=None):
    op=op or rng.choice(ARITH);return {'op':op,'k':rng.randint(3,9) if op=='mul' else rng.randint(101,899)}


def make_procedure(rng,length):
    """L = 3: perm, arith, perm; L = 4: perm, arith, perm, arith. The two permutations differ."""
    first=rng.choice(PERM);second=rng.choice([p for p in PERM if p!=first]);steps=[{'op':first},arith(rng),{'op':second}]
    return steps+[arith(rng)] if length==4 else steps


def unambiguous(procedure,n):
    """Every permutation step must give a different result for each of the four permutations, so demonstrations identify it."""
    for step in procedure:
        if step['op'] in PERM and len({apply({'op':p},n) for p in PERM})<len(PERM): return False
        n=apply(step,n)
    return True


def make_input(rng,procedure,tries=5000):
    for _ in range(tries):
        digits=rng.sample('123456789',4);n=int(''.join(digits))
        if digits!=sorted(digits) and digits!=sorted(digits,reverse=True) and min(run(procedure,n))>0 and unambiguous(procedure,n): return n
    return None  # e.g. sortasc then a large subtraction leaves 3-digit numbers, where rev and swapends coincide


def options(step,rng):
    """Four descriptions for a recognition question: the true step and three from the same family."""
    if step['op'] in PERM: others=[{'op':p} for p in PERM if p!=step['op']]
    else:
        others=[arith(rng,op) for op in ARITH if op!=step['op']]+[{'op':step['op'],'k':step['k']+rng.choice([-1,1])*rng.randint(1,3)*(1 if step['op']=='mul' else 37)}]
    choice=[step]+others[:3];order=list(range(4));rng.shuffle(order);return [choice[i] for i in order],order.index(0)


def world(seed,length,pool,tests):
    rng=Random(seed)
    while True:  # redraw procedures that admit too few unambiguous inputs
        procedure=make_procedure(rng,length);seen=set();inputs=[]
        while len(inputs)<pool+tests:
            n=make_input(rng,procedure)
            if n is None: break
            if n not in seen: seen.add(n);inputs.append(n)
        if len(inputs)==pool+tests: break
    items=[{'input':n,'values':run(procedure,n),'answer':str(run(procedure,n)[-1])} for n in inputs]
    probes=[options(step,rng) for step in procedure]
    return {'seed':seed,'length':length,'procedure':procedure,'pool':items[:pool],'tests':items[pool:],'probes':[{'options':o,'answer':'ABCD'[a]} for o,a in probes]}
