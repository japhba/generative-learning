"""Frozen statistics for the refined, length-matched confirmation (MLAYOUT_PROTOCOL.md, confirmation)."""
from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout3_search import CONDITIONS,DIRECTORY,NAMES,TESTS,evaluation_path,generation_path,process_correct
from mlayout_analyze import SHORT,bootstrap,sign_flip
from mlayout_search import direct_answer,final_answer,read,write
from mlayout_tasks import correct

REPORTS=Path(__file__).resolve().parents[1]/'reports'


def logit_own(rows,M):
    X=[];y=[]
    for _,a,b,c in rows:
        x=np.zeros(M+(M-1)+1);x[a]=1
        if b>0: x[M+b-1]=1
        x[-1]=float(a==b);X.append(x);y.append(c)
    X=np.array(X);y=np.array(y);beta=np.zeros(X.shape[1])
    for _ in range(60):
        p=1/(1+np.exp(-X@beta));g=X.T@(y-p)-1e-3*beta;H=-(X.T*(p*(1-p)))@X-1e-3*np.eye(len(beta));beta-=np.linalg.solve(H,g)
    return float(beta[-1])


def main():
    assert (DIRECTORY/'complete.json').exists();matched=read(DIRECTORY/'matched.json');design=read(DIRECTORY/'design.json');M=len(NAMES)
    worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']};inc=[n for n in worlds if matched[n]['included']]
    res={(n,r,c,i):int(correct(direct_answer(read(evaluation_path(r,n,c,i))['text']),worlds[n]['tests'][i])) for n in inc for r in NAMES for c in CONDITIONS for i in range(TESTS)}
    acc=lambda n,r,c:np.mean([res[n,r,c,i] for i in range(TESTS)])
    F={n:np.array([[acc(n,r,f'full:{s}') for s in NAMES] for r in NAMES]) for n in inc}
    S=[np.mean([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for a in range(M)]) for n in inc]
    rows=[(n,a,b,res[n,r,f'full:{s}',i]) for n in inc for a,r in enumerate(NAMES) for b,s in enumerate(NAMES) for i in range(TESTS)]
    gamma=logit_own(rows,M);rng=np.random.default_rng(0);boot=[logit_own([x for p in rng.choice(inc,len(inc)) for x in rows if x[0]==p],M) for _ in range(2000)]
    pooled=np.mean([F[n] for n in inc],0);sib=(NAMES.index('qwen35_9b'),NAMES.index('qwen3_30b'));g=NAMES.index('gemma4_26b')
    rate=defaultdict(list);chars=defaultdict(list)
    for n,w in worlds.items():
        for m in NAMES:
            for i,e in enumerate(w['pool']): rate[m]+=[process_correct(s,e,w['rule']) for s in read(generation_path(m,n,i))['samples']]
        for q in matched[n]['questions']:
            for m in NAMES: chars[m].append(len(read(generation_path(m,n,q['index']))['samples'][q['sample'][m]]['text']))
    out={'included':inc,'matched_counts':{n:len(matched[n]['questions']) for n in worlds},'S':{'mean':float(np.mean(S)),'ci':bootstrap(S),'p_one_sided':sign_flip(S)},
         'logit_own':{'gamma':gamma,'ci':[float(x) for x in np.percentile(boot,[2.5,97.5])],'odds_ratio':float(np.exp(gamma))},
         'per_reader_own':{r:[float(np.mean([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for n in inc])),*bootstrap([F[n][a,a]-np.mean([F[n][a,b] for b in range(M) if b!=a]) for n in inc])] for a,r in enumerate(NAMES)},
         'pooled_full':pooled.tolist(),'reference':{r:{c:float(np.mean([acc(n,r,c) for n in inc])) for c in ['initial','answers']} for r in NAMES},
         'qwen_pair_H':[float(np.mean(h)),*bootstrap(h)] if (h:=[((F[n][sib[0],sib[0]]-F[n][sib[0],sib[1]])+(F[n][sib[1],sib[1]]-F[n][sib[1],sib[0]]))/2 for n in inc]) else None,
         'process_correct_rate':{m:float(np.mean(v)) for m,v in rate.items()},'solution_chars':{m:float(np.mean(v)) for m,v in chars.items()}}
    write(DIRECTORY/'analysis.json',out)
    g=NAMES.index('gemma4_26b');d=[F[n][g,g]-np.mean([F[n][g,b] for b in range(M) if b!=g]) for n in inc]
    out['primary_gemma_own']={'mean':float(np.mean(d)),'ci':bootstrap(d),'p_one_sided':sign_flip(d)};write(DIRECTORY/'analysis.json',out)
    print('PRIMARY Gemma own - other two = %+.1f pp [%+.1f, %+.1f], one-sided p = %.4f'%(100*np.mean(d),100*out['primary_gemma_own']['ci'][0],100*out['primary_gemma_own']['ci'][1],out['primary_gemma_own']['p_one_sided']))
    print('included %d/%d; matched counts %s'%(len(inc),len(worlds),out['matched_counts']))
    print('S = %+.1f pp [%+.1f, %+.1f], one-sided p = %.4f'%(100*out['S']['mean'],100*out['S']['ci'][0],100*out['S']['ci'][1],out['S']['p_one_sided']))
    print('logit own effect = %+.2f [%+.2f, %+.2f] log-odds (OR %.2f)'%(gamma,*out['logit_own']['ci'],out['logit_own']['odds_ratio']))
    print('per-reader own - foreign:',{SHORT[r]:[round(100*x,1) for x in v] for r,v in out['per_reader_own'].items()});print('Qwen-pair H (pp):',[round(100*x,1) for x in out['qwen_pair_H']])
    for a,r in enumerate(NAMES): print('%-14s '%SHORT[r]+' '.join(('[%3.0f]' if b==a else ' %3.0f ')%(100*pooled[a,b]) for b in range(M))+'  | none %3.0f answers %3.0f'%(100*out['reference'][r]['initial'],100*out['reference'][r]['answers']))
    print('process-correct sample rate',{SHORT[m]:round(100*v) for m,v in out['process_correct_rate'].items()},'chars',{SHORT[m]:round(v) for m,v in out['solution_chars'].items()})
    figure,axis=plt.subplots(figsize=(5.6,4.8));data=pooled*100;image=axis.imshow(data,cmap='viridis',vmin=0,vmax=100)
    for a in range(M):
        for b in range(M): axis.text(b,a,f'{data[a,b]:.0f}',ha='center',va='center',color='white' if data[a,b]<50 else 'black')
        axis.add_patch(plt.Rectangle((a-.5,a-.5),1,1,fill=False,edgecolor='r',lw=2))
    labels=[SHORT[m] for m in NAMES];axis.set_xticks(range(M));axis.set_xticklabels(labels,rotation=30,ha='right');axis.set_yticks(range(M));axis.set_yticklabels(labels)
    axis.set_title('Direct-answer accuracy (%) after 6 process-correct\nworked solutions from each source (operator)');axis.set_xlabel('source of the solutions');axis.set_ylabel('reader')
    figure.colorbar(image,ax=axis,fraction=.046);figure.tight_layout();figure.savefig(DIRECTORY/'matrix.png',dpi=140);figure.savefig(REPORTS/'mlayout3-matrix.png',dpi=140)


if __name__=='__main__': main()
