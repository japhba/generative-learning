"""Statistics for the same-content rendering study (RENDER_PROTOCOL.md): GSM8K fine-tuning (2 seeds), GSM8K in context, operator."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout_analyze import sign_flip
from mlayout_search import direct_answer,read
from mlayout_tasks import correct as op_correct
from render_search import CEPH,DIRECTORY,GSM_SOURCES,OP_MODELS,OP_TESTS,SMALL,path
from small_search import correct as gsm_correct

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen3_1_7b':'Qwen3-1.7B','llama32_3b':'Llama-3.2-3B','smollm3_3b':'SmolLM3-3B','phi4_mini':'Phi-4-mini','gemma4_26b':'Gemma 4 26B','qwen3_30b':'Qwen3-30B-A3B','human':'human (raw)','qwen35_9b':'Qwen3.5-9B'}
SRC=GSM_SOURCES+['human']


def boot(d,seed=0):
    d=np.asarray(d,float);idx=np.random.default_rng(seed).integers(0,len(d),(20000,len(d)));return [float(d.mean()),*[float(x) for x in np.percentile(d[idx].mean(1),[2.5,97.5])]]


def contrast(res,readers):
    out={}
    for kind,others in [('peers',lambda r:[s for s in SMALL if s!=r]),('large',lambda r:['gemma4_26b','qwen3_30b']),('human',lambda r:['human'])]:
        ok=[r for r in readers if all((r,s) in res for s in [r,*others(r)])]
        if ok: out[kind]={'mean':boot(np.mean([res[r,r]-np.mean([res[r,s] for s in others(r)],0) for r in ok],0)),'per':{r:boot(res[r,r]-np.mean([res[r,s] for s in others(r)],0)) for r in ok}}
    return out


def main():
    out={};design=read(DIRECTORY/'design.json')
    ft={0:{},1:{}}
    for st in SMALL:
        b=CEPH/'ft'/st/'eval_base.json'
        for seed in (0,1):
            if b.exists(): ft[seed][st,'base']=np.array([gsm_correct(x['text'],x['answer']) for x in json.loads(b.read_text())['gsm_test_cot']],float)
            for s in SRC:
                p=CEPH/'ft'/st/f'from_{s}_seed{seed}'/'eval.json'
                if p.exists(): ft[seed][st,s]=np.array([gsm_correct(x['text'],x['answer']) for x in json.loads(p.read_text())['gsm_test_cot']],float)
    pooled={k:(ft[0][k]+ft[1][k])/2 for k in ft[1] if k in ft[0]}
    out['finetune']={'seed0':contrast(ft[0],SMALL),'seed1':contrast(ft[1],SMALL),'pooled':contrast(pooled,SMALL),'accuracy':{f'{r}|{s}|seed{d}':float(v.mean()) for d in (0,1) for (r,s),v in ft[d].items()}}
    tests=design['gsm_test'][:300];icl={}
    for r in SMALL:
        for s in ['initial']+SRC:
            if all(path('icl',r,f'{s}__{i}').exists() for i in range(len(tests))): icl[r,'base' if s=='initial' else s]=np.array([gsm_correct(read(path('icl',r,f'{s}__{i}'))['text'],tests[i]['answer']) for i in range(len(tests))],float)
    out['icl']={'contrast':contrast(icl,SMALL),'accuracy':{f'{r}|{s}':float(v.mean()) for (r,s),v in icl.items()}}
    matched=read(DIRECTORY/'op_matched.json') if (DIRECTORY/'op_matched.json').exists() else {}
    worlds=design['op_worlds'];inc=[n for n,v in matched.items() if len(v)>=3];conds=['initial','answers','canonical']+[f'render:{m}' for m in OP_MODELS]
    if inc and all(path('op_eval',r,f"{n}__{c.replace(':','-')}__{i}").exists() for n in inc for r in OP_MODELS for c in conds for i in range(OP_TESTS)):
        acc=lambda n,r,c:np.mean([op_correct(direct_answer(read(path('op_eval',r,f"{n}__{c.replace(':','-')}__{i}"))['text']),worlds[n]['tests'][i]) for i in range(OP_TESTS)])
        F={n:np.array([[acc(n,r,f'render:{s}') for s in OP_MODELS] for r in OP_MODELS]) for n in inc};M=len(OP_MODELS)
        S=[np.mean([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for a in range(M)]) for n in inc]
        out['operator']={'episodes':len(inc),'S':boot(S),'p_one_sided':sign_flip(S),'pooled':np.mean([F[n] for n in inc],0).tolist(),
                         'per_reader':{r:boot([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for n in inc]) for a,r in enumerate(OP_MODELS)},
                         'reference':{r:{c:float(np.mean([acc(n,r,c) for n in inc])) for c in ['initial','answers','canonical']} for r in OP_MODELS}}
    (DIRECTORY/'analysis.json').write_text(json.dumps(out,indent=1))
    show=lambda c:', '.join(f"{k}: {100*v['mean'][0]:+.1f} [{100*v['mean'][1]:+.1f}, {100*v['mean'][2]:+.1f}] ("+' '.join(f"{SHORT[r]} {100*x[0]:+.1f}" for r,x in v['per'].items())+')' for k,v in c.items())
    for d in ('seed0','seed1','pooled'): print(f'FT {d}:',show(out['finetune'][d]))
    for d,res in [(0,ft[0]),(1,ft[1])]:
        print(f'\nFT seed {d} accuracy (%): rows student, columns source; base first');print('%-13s %5s '%('','base')+' '.join('%7s'%SHORT[s][:7] for s in SRC))
        for r in SMALL:
            if (r,'base') in res: print('%-13s %5.1f '%(SHORT[r],100*res[r,'base'].mean())+' '.join(('[%5.1f]' if s==r else ' %5.1f ')%(100*res[r,s].mean()) if (r,s) in res else '   -   ' for s in SRC))
    print('\nICL:',show(out['icl']['contrast']))
    if 'operator' in out: o=out['operator'];print('\nOperator: S = %+.1f pp [%+.1f, %+.1f], p = %.4f'%(100*o['S'][0],100*o['S'][1],100*o['S'][2],o['p_one_sided']),{SHORT[r]:round(100*v[0],1) for r,v in o['per_reader'].items()},'| reference',o['reference'])


if __name__=='__main__': main()
