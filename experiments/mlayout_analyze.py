"""Frozen statistics for the correctness-matched laid-out-reasoning study (MLAYOUT_PROTOCOL.md)."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout_search import CONDITIONS,DIRECTORY,FAMILY,NAMES,TESTS,direct_answer,evaluation_path,final_answer,generation_path,read,write
from mlayout_tasks import correct

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','gemma4_26b':'Gemma-4-26B','phi4':'Phi-4'}


def sign_flip(values):
    values=np.asarray(values);n=len(values);signs=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
    return float(np.mean((signs*values).mean(1)>=values.mean()-1e-12))
def bootstrap(values,resamples=20000,seed=20261005):
    values=np.asarray(values);idx=np.random.default_rng(seed).integers(0,len(values),(resamples,len(values)));return [float(x) for x in np.percentile(values[idx].mean(1),[2.5,97.5])]


def main():
    assert (DIRECTORY/'complete.json').exists();matched=read(DIRECTORY/'matched.json');design=read(DIRECTORY/'design.json')
    worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in design['worlds']};included=[n for n in worlds if matched[n]['included']];M=len(NAMES)
    acc={(n,r,c):float(np.mean([correct(direct_answer(read(evaluation_path(r,n,c,i))['text']),worlds[n]['tests'][i]) for i in range(TESTS)])) for n in included for r in NAMES for c in CONDITIONS}
    F={n:np.array([[acc[n,r,f'full:{s}'] for s in NAMES] for r in NAMES]) for n in included}
    S=[np.mean([F[n][i,i]-np.mean([F[n][i,j] for j in range(M) if j!=i]) for i in range(M)]) for n in included]
    fam=lambda n:n.split('__')[0];families=sorted({fam(n) for n in included})
    pooled=np.mean([F[n] for n in included],0);dc=pooled-pooled.mean(1,keepdims=True)-pooled.mean(0,keepdims=True)+pooled.mean()
    out={'included':included,'excluded':[n for n in worlds if n not in included],'matched_counts':{n:len(matched[n]['questions']) for n in worlds},
         'S':{'mean':float(np.mean(S)),'ci':bootstrap(S),'p_one_sided':sign_flip(S),'by_family':{f:[float(np.mean([s for n,s in zip(included,S) if fam(n)==f])),*bootstrap([s for n,s in zip(included,S) if fam(n)==f])] for f in families}},
         'per_reader_own':{r:[float(np.mean([F[n][i,i]-np.mean([F[n][i,j] for j in range(M) if j!=i]) for n in included])),*bootstrap([F[n][i,i]-np.mean([F[n][i,j] for j in range(M) if j!=i]) for n in included])] for i,r in enumerate(NAMES)},
         'double_centred_own':dict(zip(NAMES,np.diag(dc).tolist())),'pooled_full':pooled.tolist(),
         'by_family_full':{f:np.mean([F[n] for n in included if fam(n)==f],0).tolist() for f in families},
         'reference':{f:{r:{c:float(np.mean([acc[n,r,c] for n in included if fam(n)==f])) for c in ['initial','answers']} for r in NAMES} for f in families},
         'pairwise_H':{f'{a}|{b}':float(((pooled[i,i]-pooled[i,j])+(pooled[j,j]-pooled[j,i]))/2) for i,a in enumerate(NAMES) for j,b in enumerate(NAMES) if i<j}}
    i,j=NAMES.index('qwen35_9b'),NAMES.index('qwen3_30b');cross=[NAMES.index(m) for m in NAMES if FAMILY[m]!='qwen']
    out['qwen_siblings']={r:{'own_minus_sibling':float(np.mean([F[n][a,a]-F[n][a,b] for n in included])),'sibling_minus_cross':float(np.mean([F[n][a,b]-F[n][a,cross].mean() for n in included]))} for r,a,b in [('qwen35_9b',i,j),('qwen3_30b',j,i)]}
    lengths=defaultdict(list);sample_acc=defaultdict(list)
    for n in worlds:
        for q in matched[n]['questions']:
            for m in NAMES: lengths[fam(n),m].append(len(read(generation_path(m,n,q['index']))['samples'][q['sample'][m]]['text']))
        for m in NAMES:
            for k,e in enumerate(worlds[n]['pool']): sample_acc[fam(n),m]+=[s['status']=='ok' and correct(final_answer(s['text']),e) for s in read(generation_path(m,n,k))['samples']]
    out['solution_chars']={f'{f}|{m}':float(np.mean(v)) for (f,m),v in lengths.items()};out['sample_accuracy']={f'{f}|{m}':float(np.mean(v)) for (f,m),v in sample_acc.items()}
    write(DIRECTORY/'analysis.json',out)
    print('included %d/%d episodes; matched counts %s'%(len(included),len(worlds),out['matched_counts']))
    print('S = %+.1f pp [%+.1f, %+.1f], one-sided p = %.4f'%(100*out['S']['mean'],100*out['S']['ci'][0],100*out['S']['ci'][1],out['S']['p_one_sided']),{f:round(100*v[0],1) for f,v in out['S']['by_family'].items()})
    print('per-reader own - foreign (pp):',{SHORT[r]:[round(100*x,1) for x in v] for r,v in out['per_reader_own'].items()})
    print('qwen siblings:',out['qwen_siblings'])
    for f in families:
        print(f'\n{f}: rows reader, columns source of correct full solutions (%); then none / answers-only')
        for a,r in enumerate(NAMES): print('%-14s '%SHORT[r]+' '.join(('[%3.0f]' if b==a else ' %3.0f ')%(100*out['by_family_full'][f][a][b]) for b in range(M))+'  | %3.0f %3.0f'%(100*out['reference'][f][r]['initial'],100*out['reference'][f][r]['answers']))
        print('solution chars:',{SHORT[m]:round(out['solution_chars'][f'{f}|{m}']) for m in NAMES},'sample accuracy:',{SHORT[m]:round(100*out['sample_accuracy'][f'{f}|{m}']) for m in NAMES})
    figure,axes=plt.subplots(1,len(families),figsize=(5.4*len(families),4.6),squeeze=False)
    for axis,f in zip(axes[0],families):
        data=np.array(out['by_family_full'][f])*100;image=axis.imshow(data,cmap='viridis',vmin=0,vmax=100)
        for a in range(M):
            for b in range(M): axis.text(b,a,f'{data[a,b]:.0f}',ha='center',va='center',color='white' if data[a,b]<50 else 'black')
            axis.add_patch(plt.Rectangle((a-.5,a-.5),1,1,fill=False,edgecolor='r',lw=2))
        labels=[SHORT[m] for m in NAMES];axis.set_xticks(range(M));axis.set_xticklabels(labels,rotation=35,ha='right');axis.set_yticks(range(M));axis.set_yticklabels(labels)
        axis.set_title(f'{f}: direct-answer accuracy (%) after\n6 correct worked solutions from each source');axis.set_xlabel('source of the correct solutions');axis.set_ylabel('reader');figure.colorbar(image,ax=axis,fraction=.046)
    figure.tight_layout();figure.savefig(DIRECTORY/'matrix.png',dpi=140);figure.savefig(REPORTS/'mlayout-matrix.png',dpi=140)


if __name__=='__main__': main()
