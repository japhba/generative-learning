"""Arm 3 statistics: accuracy after fine-tuning on own vs foreign matched solutions; learning curves (MSFT_PROTOCOL.md)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout_search import direct_answer,final_answer
from mlayout_tasks import normalize

ROOT=Path('/ceph/scratch/jbauer/mlayout/msft');REPORTS=Path(__file__).resolve().parents[1]/'reports';OUT=Path('/var/tmp/jbauer/generative_task_search_20260910/msft')
NAMES=['qwen35_9b','qwen3_30b','gemma4_26b'];SHORT={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','gemma4_26b':'Gemma-4-26B'}
SETS=['operator_cot','operator_answer_only','units_answer_only','gsm_cot']


def same(a,b):
    a,b=normalize(a),normalize(b)
    if a==b and a: return True
    try: return abs(float(a)-float(b))<1e-6
    except ValueError: return False


def scores(path):
    data=json.loads(path.read_text());return {s:np.array([same(direct_answer(x['text']) if s.endswith('answer_only') else final_answer(x['text']),x['answer']) for x in data[s]],float) for s in SETS}


def main():
    res={};out={}
    for st in NAMES:
        res[st,'base']=scores(ROOT/st/'eval_base.json')
        for src in NAMES:
            if (ROOT/st/f'from_{src}/eval.json').exists(): res[st,src]=scores(ROOT/st/f'from_{src}/eval.json')
    complete=[st for st in NAMES if all((st,src) in res for src in NAMES)];rng=np.random.default_rng(0)
    for s in SETS:
        table={st:{src:float(res[st,src][s].mean()) for src in ['base']+NAMES if (st,src) in res} for st in NAMES};contrasts={}
        for st in complete:
            d=res[st,st][s]-np.mean([res[st,o][s] for o in NAMES if o!=st],0);idx=rng.integers(0,len(d),(20000,len(d)))
            contrasts[st]=[float(d.mean()),*[float(x) for x in np.percentile(d[idx].mean(1),[2.5,97.5])]]
        pooled=None
        if complete:
            d=np.mean([res[st,st][s]-np.mean([res[st,o][s] for o in NAMES if o!=st],0) for st in complete],0);idx=rng.integers(0,len(d),(20000,len(d)))
            pooled=[float(d.mean()),*[float(x) for x in np.percentile(d[idx].mean(1),[2.5,97.5])]]
        out[s]={'accuracy':table,'own_minus_foreign':contrasts,'mean_over_students':pooled}
        print(f'\n{s}: rows student, columns training source (%)  [base first]')
        for st in NAMES: print('  %-14s base %3.0f | '%(SHORT[st],100*table[st]['base'])+' '.join(('[%3.0f]' if src==st else ' %3.0f ')%(100*table[st][src]) if src in table[st] else '  -  ' for src in NAMES)+('   own-foreign %+.1f [%+.1f, %+.1f]'%tuple(100*np.array(contrasts[st])) if st in contrasts else ''))
        if pooled: print('  mean over students: %+.1f pp [%+.1f, %+.1f]'%tuple(100*np.array(pooled)))
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'analysis.json').write_text(json.dumps(out,indent=1))
    figure,axes=plt.subplots(1,3,figsize=(15,4),sharey=False)
    for axis,st in zip(axes,NAMES):
        for src in NAMES:
            p=ROOT/st/f'from_{src}/loss.json'
            if p.exists():
                log=json.loads(p.read_text());axis.plot([x['step'] for x in log],[x['loss'] for x in log],label=f'{SHORT[src]} data'+(' (own)' if src==st else ''),lw=2.2 if src==st else 1.2)
        axis.set_title(f'{SHORT[st]} fine-tuning');axis.set_xlabel('optimizer step');axis.set_ylabel('training loss (per solution token)');axis.legend()
    figure.tight_layout();figure.savefig(REPORTS/'msft-learning-curves.png',dpi=140)


if __name__=='__main__': main()
