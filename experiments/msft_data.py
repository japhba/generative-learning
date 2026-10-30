"""Arm 3 data: own versus foreign fine-tuning sets with identical questions, process-correct and length-matched solutions.

Pools from the two process-matched cohorts (mlayout2, mlayout3). A question enters if every model has a process-correct
sample; each model contributes the one closest to the median length of all qualifying samples for that question.
"""
import json

import numpy as np

import mlayout2_search as M2
import mlayout3_search as M3
from mlayout_search import generation_messages,read,write
from task_search_api import ROOT

DIRECTORY=ROOT.resolve()/'msft';NAMES=M3.NAMES


def main():
    rows={m:[] for m in NAMES};stats={'questions':0,'considered':0}
    for mod in (M2,M3):
        for n in sorted(read(mod.DIRECTORY/'design.json')['worlds']):
            w=read(mod.DIRECTORY/'worlds'/f'{n}.json')
            for i,e in enumerate(w['pool']):
                if not all(mod.generation_path(m,n,i).exists() for m in NAMES): continue
                stats['considered']+=1;samples={m:read(mod.generation_path(m,n,i))['samples'] for m in NAMES}
                good={m:[(k,len(s['text'])) for k,s in enumerate(v) if M3.process_correct(s,e,w['rule'])] for m,v in samples.items()}
                if not all(good.values()): continue
                target=float(np.median([length for v in good.values() for _,length in v]));stats['questions']+=1
                for m,v in good.items():
                    k=min(v,key=lambda x:(abs(x[1]-target),x[0]))[0]
                    rows[m].append({'id':f'{mod.DIRECTORY.name}/{n}/{i}','messages':generation_messages(w,e['question'])+[{'role':'assistant','content':samples[m][k]['text']}]})
    for m,v in rows.items(): write(DIRECTORY/'data'/f'{m}.json',v)
    stats['mean_chars']={m:float(np.mean([len(r['messages'][-1]['content']) for r in v])) for m,v in rows.items()};write(DIRECTORY/'data/stats.json',stats);print(json.dumps(stats))


if __name__=='__main__': main()
