"""Post-hoc diagnostics only: answer formatting, duplicated inputs, Boolean pseudo-labels."""

import argparse
import ast
from collections import defaultdict
import json
import math
from pathlib import Path
import re

import numpy as np

from self_icl import normalize
from self_icl_api import save_json


def relaxed_answer(text, primary, labels):
    if primary in labels: return primary
    clean = re.sub(r'\*\*|__|`', '', text)
    candidates = []
    for match in re.finditer(r'(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is|:)\s*',clean,re.I):
        candidates.append(clean[match.end():].lstrip(' .:\n'))
    for match in re.finditer(r'\\boxed\{([^{}]+)\}',clean): candidates.append(match.group(1))
    lines=[line.strip() for line in clean.splitlines() if line.strip()]
    if lines: candidates.append(lines[-1])
    for value in reversed(candidates):
        token=re.match(r'\s*\W*(\w+)',value)
        if token and normalize(token.group(1)) in labels: return normalize(token.group(1))
    return None


def option_text_answer(question, text, primary, labels):
    options=dict(re.findall(r'^\(([A-Z])\)\s+(.+)$',question,re.M))
    if not options: return relaxed_answer(text,primary,labels)
    if primary in options: return primary
    clean=re.sub(r'\*\*|__|`','',text)
    markers=list(re.finditer(r'(?:the\s+)?(?:(?:correct|final)\s+)?answer\s*(?:is|:)\s*',clean,re.I))
    if not markers: return relaxed_answer(text,primary,set(options))
    tail=clean[markers[-1].end():].strip(' .:\n')
    letter=re.match(r'(?:option\s+)?\(?([A-Z])\)?(?:\s|[.,:]|$)',tail)
    if letter and letter.group(1) in options: return letter.group(1)
    def canonical(value): return re.sub(r'\s+',' ',value).strip(' .!\"\'').casefold()
    candidate=canonical(tail)
    candidate=re.sub(r'^(?:that|option)\s+','',candidate)
    matches=[key for key,value in options.items() if candidate==canonical(value) or candidate.startswith(canonical(value)+'. ')]
    return matches[0] if len(matches)==1 else None


def boolean_truth(question):
    expression=re.sub(r'^Q:\s*','',question.strip())
    expression=re.sub(r'\s+is\s*[?.:]?\s*$','',expression)
    tree=ast.parse(expression.strip(),mode='eval')
    def evaluate(node):
        if isinstance(node,ast.Constant) and type(node.value) is bool: return node.value
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.Not): return not evaluate(node.operand)
        if isinstance(node,ast.BoolOp):
            values=[evaluate(item) for item in node.values]
            if isinstance(node.op,ast.And): return all(values)
            if isinstance(node.op,ast.Or): return any(values)
        raise ValueError('Generated question is outside the Boolean expression grammar.')
    return str(evaluate(tree.body)).upper()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    args=parser.parse_args()
    tasks=json.loads((args.phase.parent/'inputs/tasks.json').read_text())
    groups=defaultdict(lambda:defaultdict(int))
    observations=[]
    for path in sorted((args.phase/'cases').glob('*.json')):
        case=json.loads(path.read_text())
        labels={normalize(item['target']) for item in tasks[case['task']]['examples']}
        for row in case['rows']:
            key=f"{case['model']}/{row['mode']}/{row['condition']}/{row['budget']}"
            stats=groups[key]
            stats['cases']+=1; stats['primary_correct']+=row['correct']
            relaxed=relaxed_answer(row['response'],row['prediction'],labels)
            stats['relaxed_correct']+=relaxed==normalize(case['target'])
            content_answer=option_text_answer(case['query'],row['response'],row['prediction'],labels)
            stats['option_content_correct']+=content_answer==normalize(case['target'])
            stats['option_content_unparsed']+=content_answer is None
            stats['pipeline_failures']+=row['pipeline_failure']
            if row['mode']=='cot' and row['condition'] in ('zero','self'):
                observations.append({'task':case['task'],'index':case['index'],'model':case['model'],'condition':row['condition'],
                                     'budget':row['budget'],'target':case['target'],'primary_correct':row['correct'],
                                     'content_prediction':content_answer,'content_correct':content_answer==normalize(case['target'])})
            if row['condition']!='self': continue
            questions=[question for question,answer in row['demonstrations']]
            stats['demonstrations']+=len(questions)
            stats['unique_demonstrations']+=len(set(questions))
            stats['exact_query_copies']+=sum(question.strip()==case['query'].strip() for question in questions)
            if case['task']!='boolean_expressions': continue
            for question,answer in row['demonstrations']:
                record={'index':case['index'],'model':case['model'],'mode':row['mode'],'budget':row['budget'],'question':question,'response':answer}
                try: target=boolean_truth(question)
                except (SyntaxError,ValueError) as error:
                    record['status']='unscorable'; record['error']=str(error)
                    stats['boolean_unscorable']+=1
                else:
                    prediction=relaxed_answer(answer,None,{'TRUE','FALSE'})
                    record.update(status='scored',target=target,prediction=prediction,correct=prediction==target)
                    stats['boolean_scored']+=1; stats['boolean_correct']+=prediction==target
                observations.append(record)
    contrasts=[]
    evaluated=[record for record in observations if 'condition' in record]
    for model in sorted({record['model'] for record in evaluated}):
        model_rows=[record for record in evaluated if record['model']==model]
        tasks=sorted({record['task'] for record in model_rows})
        for budget in sorted({record['budget'] for record in model_rows if record['condition']=='self'}):
            arrays={}
            for condition,k in (('zero',0),('self',budget)):
                arrays[condition]=np.asarray([[record['content_correct'] for record in sorted(model_rows,key=lambda r:r['index']) if record['task']==task and record['condition']==condition and record['budget']==k] for task in tasks],dtype=float)
            delta=arrays['self']-arrays['zero']
            rng=np.random.default_rng(910410)
            draws=rng.integers(0,delta.shape[1],size=(4000,*delta.shape))
            boot=np.take_along_axis(delta[None,:,:],draws,axis=2).mean(axis=(1,2))
            zero_boot=np.take_along_axis(arrays['zero'][None,:,:],draws,axis=2).mean(axis=(1,2))
            self_boot=np.take_along_axis(arrays['self'][None,:,:],draws,axis=2).mean(axis=(1,2))
            wins=int((delta>0).sum()); losses=int((delta<0).sum()); d=wins+losses
            contrasts.append({'model':model,'budget':budget,'delta':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                              'zero_accuracy':float(arrays['zero'].mean()),'self_accuracy':float(arrays['self'].mean()),
                              'zero_ci95':np.quantile(zero_boot,[.025,.975]).tolist(),'self_ci95':np.quantile(self_boot,[.025,.975]).tolist(),
                              'wins':wins,'losses':losses,'mcnemar_two_sided_p':min(1.,2*sum(math.comb(d,i) for i in range(min(wins,losses)+1))/2**d)})
    save_json(args.phase/'audit.json',{'summary':dict(groups),'observations':observations,'option_text_cot_contrasts':contrasts,
                                     'scope':'Diagnostic scoring after all calls; no filtering or feedback to the learner. Exact option-text matching is a post-hoc sensitivity analysis, not semantic judging.'})
    for key,value in groups.items(): print(key,dict(value))


if __name__=='__main__': main()
