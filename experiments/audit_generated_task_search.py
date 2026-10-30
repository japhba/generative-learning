"""Post-inference question-distribution diagnostics for exactly parseable task forms."""

import argparse
from collections import Counter,defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from tqdm.auto import tqdm

from audit_task_search_constraints import prime
from task_search_api import ROOT

TASKS={'count_bits','coin_flip','self_reference'}


def describe(task,example):
    question=example['question']; answer=example['answer']; base={'question':question,'answer':answer,'answer_embedded_in_question':bool(re.search(r'\banswer\s*:',question,re.I))}
    if task=='count_bits':
        match=re.fullmatch(r'How many 1 bits are there in the binary representation of the number ([\d,]+)\?',question.strip(),re.I)
        if match is None: return {**base,'status':'unsupported_question_form'}
        n=int(match[1].replace(',','')); solution=str(n.bit_count())
        return {**base,'status':'parsed','identity':str(n),'solution':solution,'correct':answer.strip()==solution,'number':n,'binary_digits':n.bit_length(),'ones':n.bit_count(),'power_of_two':n>0 and n&(n-1)==0,'all_ones_binary':n>0 and n&(n+1)==0}
    if task=='coin_flip':
        match=re.fullmatch(r'What is the probability of getting (exactly|at least) (\d+) heads in (\d+) fair coin flips\?',question.strip(),re.I)
        if match is None: return {**base,'status':'unsupported_question_form'}
        mode=match[1].lower(); k,n=int(match[2]),int(match[3])
        if not 0<=k<=n: return {**base,'status':'outside_valid_parameter_range'}
        if n>1000: return {**base,'status':'outside_audit_parameter_scale'}
        numerator=math.comb(n,k) if mode=='exactly' else sum(math.comb(n,i) for i in range(k,n+1)); target=Fraction(numerator,2**n)
        try: correct=abs(Fraction(answer.replace(',',''))-target)<=Fraction(1,10000)
        except (ValueError,ZeroDivisionError): correct=False
        return {**base,'status':'parsed','identity':f'{mode}|{n}|{k}','solution':str(target),'correct':correct,'trials':n,'k_heads':k,'mode':mode}
    patterns=[r'At least (\d+) of these 7 statements are true',r'At most (\d+) of these 7 statements are false',r'Exactly (\d+) of these 7 statements are true',r'Exactly (\d+) of these 7 statements are false']
    matches=[re.search(pattern,question,re.I) for pattern in patterns]
    fixed=['Either Statement 3 or Statement 4 is true, but not both','The number of true statements is a prime number','The number of false statements is a composite number']
    numbered=re.findall(r'(?:^|\n)\s*[-*]?\s*Statement (\d+)\s*:',question,re.I)
    if numbered!=list('1234567') or any(match is None for match in matches) or any(statement.lower() not in question.lower() for statement in fixed): return {**base,'status':'unsupported_question_form'}
    a,b,c,d=[int(match[1]) for match in matches]; count=0
    for true in range(8):
        false=7-true; count+=sum([true>=a,false<=b,true==c,false==d,(true==c)!=(false==d),prime(true),false>1 and not prime(false)])==true
    return {**base,'status':'parsed','identity':f'{a}|{b}|{c}|{d}','solution':str(count),'correct':answer.strip()==str(count),'parameters':[a,b,c,d]}


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--phase',required=True); args=parser.parse_args(); directory=ROOT.resolve()/args.phase
    names=[r['world'] for r in json.loads((directory/'preflight.json').read_text()) if r['status']=='ready']; assert len(names)==len(list((directory/'evaluations').glob('*.json')))
    records=[]; grouped=defaultdict(list)
    for name in tqdm(names,desc='Question distribution audit',mininterval=10):
        result=json.loads((directory/'evaluations'/name).read_text()); task=result['spec']['task']
        if task not in TASKS: continue
        world=json.loads((directory/'worlds'/name).read_text()); samples={'teacher':world['practice'],'test_distribution':world['tests'],'self_free':[g['example'] for g in result['generations']['self_free']]}
        for source,examples in samples.items():
            seen=set()
            for i,example in enumerate(examples):
                record={'task':task,'seed':world['seed'],'source':source,'index':i,**describe(task,example)}
                if record['status']=='parsed': record['semantic_duplicate']=record['identity'] in seen; seen.add(record['identity'])
                records.append(record); grouped[(task,source)].append(record)
    summary=[]
    for (task,source),items in sorted(grouped.items()):
        parsed=[r for r in items if r['status']=='parsed']; assert parsed
        row={'task':task,'source':source,'count':len(items),'parseable':len(parsed),'statuses':dict(Counter(r['status'] for r in items)),'accuracy_among_parseable':float(np.mean([r['correct'] for r in parsed])),
             'semantic_duplicates_among_parseable':sum(r['semantic_duplicate'] for r in parsed),'answer_embedded_in_question':sum(r['answer_embedded_in_question'] for r in items)}
        if task=='count_bits': row.update({key:float(np.mean([r[key] for r in parsed])) for key in ['binary_digits','ones','power_of_two','all_ones_binary']})
        if task=='coin_flip': row['mean_trials']=float(np.mean([r['trials'] for r in parsed])); row['mode_counts']=dict(Counter(r['mode'] for r in parsed))
        if task=='self_reference': row['target_solution_counts']=dict(Counter(r['solution'] for r in parsed)); row['given_answer_counts']=dict(Counter(r['answer'] for r in parsed))
        summary.append(row)
    report={'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'timing':'After all phase inference. Never used to select, reject, rewrite, or label a generated sample for the learner.','scope':'Only canonical bit-count and coin-flip question forms (coin trials <=1000) and the seven numbered self-reference statements are scored; parse coverage is reported. These post-hoc associations do not establish a mechanism.','summary':summary,'samples':records}
    (directory/'question_distribution_audit.json').write_text(json.dumps(report,indent=2)); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
