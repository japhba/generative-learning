"""Scoring and frozen statistics for the laid-out-reasoning study (LAYOUT_PROTOCOL.md)."""
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from layout_search import DIRECTORY,NAMES,SOURCES,evaluation_path,extract,generation_path,read,write
from layout_tasks import correct

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','llama31_8b':'Llama-3.1-8B','gemma4_26b':'Gemma-4-26B','phi4':'Phi-4','nemo_12b':'Mistral-Nemo'}


def sign_flip(values):
    values=np.asarray(values);n=len(values);signs=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
    return float(np.mean((signs*values).mean(1)>=values.mean()-1e-12))
def bootstrap(values,resamples=20000,seed=20261005):
    values=np.asarray(values);idx=np.random.default_rng(seed).integers(0,len(values),(resamples,len(values)));return [float(x) for x in np.percentile(values[idx].mean(1),[2.5,97.5])]


def main():
    design=read(DIRECTORY/'design.json');assert (DIRECTORY/'complete.json').exists();acc={};gen=defaultdict(list)
    for name in design['worlds']:
        w=read(DIRECTORY/'worlds'/f'{name}.json')
        for r in NAMES:
            for s in SOURCES: acc[name,r,s]=float(np.mean([correct(extract(read(evaluation_path(r,name,s,i))['text']),w['tests'][i]) for i in range(len(w['tests']))]))
        for m in NAMES:
            for i,p in enumerate(w['practice']): t=read(generation_path(m,name,i));gen[w['family'],m].append((correct(extract(t['text']),p),len(t['text']),t['status']!='ok'))
    worlds=list(design['worlds']);families=sorted({n.split('__')[0] for n in worlds})
    def per_episode(name):
        F=np.array([[acc[name,r,f'{s}:full'] for s in NAMES] for r in NAMES]);A=np.array([[acc[name,r,f'{s}:ans'] for s in NAMES] for r in NAMES]);D=F-A;M=len(NAMES)
        L=np.mean([D[i,i]-np.mean([D[i,j] for j in range(M) if j!=i]) for i in range(M)])
        return {'L':L,'V':np.mean(np.diag(D)),'S':np.trace(F)/M-(F.sum()-np.trace(F))/(M*(M-1)),'F':F,'A':A}
    E={n:per_episode(n) for n in worlds};out={}
    for key in ['L','V','S']:
        vals=[E[n][key] for n in worlds];out[key]={'mean':float(np.mean(vals)),'ci':bootstrap(vals),'p_one_sided':sign_flip(vals),
                                                   'by_family':{f:[float(np.mean([E[n][key] for n in worlds if n.startswith(f)])),*bootstrap([E[n][key] for n in worlds if n.startswith(f)])] for f in families}}
    out['full']={f:np.mean([E[n]['F'] for n in worlds if n.startswith(f)],0).tolist() for f in families}
    out['answers']={f:np.mean([E[n]['A'] for n in worlds if n.startswith(f)],0).tolist() for f in families}
    out['reference']={f:{r:{s:float(np.mean([acc[n,r,s] for n in worlds if n.startswith(f)])) for s in ['initial','teacher']} for r in NAMES} for f in families}
    out['practice']={f'{f}|{m}':{'correct':float(np.mean([x[0] for x in v])),'chars':float(np.mean([x[1] for x in v])),'truncated':float(np.mean([x[2] for x in v]))} for (f,m),v in gen.items()}
    out['own_value_by_reader']={r:float(np.mean([E[n]['F'][i,i]-E[n]['A'][i,i] for n in worlds])) for i,r in enumerate(NAMES)}
    write(DIRECTORY/'analysis.json',out)
    for key in ['L','V','S']: print(key,'%+.1f pp [%+.1f, %+.1f] p=%.4f'%(100*out[key]['mean'],100*out[key]['ci'][0],100*out[key]['ci'][1],out[key]['p_one_sided']),{f:round(100*v[0],1) for f,v in out[key]['by_family'].items()})
    for f in families:
        print(f'\n{f}: rows = reader; cells = full / answers-only (%); none, teacher at right')
        for i,r in enumerate(NAMES):
            print('%-14s '%SHORT[r]+' '.join(('[%3.0f/%3.0f]' if j==i else ' %3.0f/%3.0f ')%(100*out['full'][f][i][j],100*out['answers'][f][i][j]) for j in range(len(NAMES)))+'  | %3.0f %3.0f'%(100*out['reference'][f][r]['initial'],100*out['reference'][f][r]['teacher']))
        print('practice correct:',{SHORT[m]:round(100*out['practice'][f'{f}|{m}']['correct']) for m in NAMES})
    plot(out,families)


def plot(out,families):
    figure,axes=plt.subplots(1,len(families),figsize=(5.2*len(families),4.6),squeeze=False)
    for axis,f in zip(axes[0],families):
        D=(np.array(out['full'][f])-np.array(out['answers'][f]))*100;lim=np.abs(D).max()
        image=axis.imshow(D,cmap='RdBu_r',vmin=-lim,vmax=lim)
        for i in range(len(NAMES)):
            for j in range(len(NAMES)): axis.text(j,i,f'{D[i,j]:+.0f}',ha='center',va='center')
            axis.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='k',lw=2))
        labels=[SHORT[m] for m in NAMES];axis.set_xticks(range(len(NAMES)));axis.set_xticklabels(labels,rotation=40,ha='right');axis.set_yticks(range(len(NAMES)));axis.set_yticklabels(labels)
        axis.set_title(f'{f}: full reasoning − answers only (pp)');axis.set_xlabel('source of practice');axis.set_ylabel('reader');figure.colorbar(image,ax=axis,fraction=.046)
    figure.tight_layout();figure.savefig(DIRECTORY/'layout_value.png',dpi=140);figure.savefig(REPORTS/'layout-value.png',dpi=140)


if __name__=='__main__': main()
