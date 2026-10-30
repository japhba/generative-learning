"""Post-inference scoring, baseline-only task choice and matrix statistics for the native-traces study (NTRACE_PROTOCOL.md)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from audit_task_search_constraints import answer_valid
from ntrace_search import DIRECTORY,MODELS,NAMES,read,write
from task_search_data import dataset
from task_search_scoring import score_answer

REPORTS=Path(__file__).resolve().parents[1]/'reports'
LINE=re.compile(r'^[ \t>#*_-]*(?:final[ \t]+)?answer[ \t]*[*_]*[ \t]*[:：][ \t]*(.*?)[ \t]*$',re.I|re.M)  # no newline crossing: '### Final Answer:\n\nAnswer: 7' -> '7'
SHORT={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','llama31_8b':'Llama-3.1-8B','llama32_3b':'Llama-3.2-3B','gemma4_26b':'Gemma-4-26B','phi4':'Phi-4','nemo_12b':'Mistral-Nemo'}


def clean(answer):
    answer=re.sub(r'^```[a-zA-Z]*\s*|\s*```$','',answer.strip())
    answer=re.sub(r'\\boxed\{(.*)\}',r'\1',answer);answer=re.sub(r'^\$+(.*?)\$+$',r'\1',answer);answer=re.sub(r'^\\\((.*)\\\)$',r'\1',answer)
    answer=answer.replace('**','').replace('__','').strip().strip('`').strip()
    return answer[:-1].strip() if answer.endswith('.') else answer


def extract(text):
    found=[clean(a) for a in LINE.findall(text) if clean(a)]
    return found[-1] if found else ''


def correct(data,task,answer,entry): return bool(answer) and score_answer(data,task,answer,entry)>=1-1e-9 and answer_valid(task,answer,entry)


def judge(data,task,text,entry):
    """Primary (frozen cleanup) and format-lenient sensitivity (also accept the last number on the answer line)."""
    answer=extract(text);strict=correct(data,task,answer,entry);lenient=strict
    if not strict and answer:
        numbers=re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?',answer.replace(',',''))
        lenient=bool(numbers) and correct(data,task,numbers[-1],entry)
    return strict,lenient,answer


def score(phase):
    directory=DIRECTORY/phase;design=read(directory/'design.json');assert (directory/'complete.json').exists(),'Analyze only completed phases.';rows=[];gens=[]
    for name in tqdm(design['worlds'],desc=f'Score {phase}',mininterval=10):
        world=read(directory/'worlds'/f'{name}.json');data=dataset(world);task=world['spec']['task']
        for path in sorted((directory/'evaluations').glob(f'*/{name}__*.json')):
            r=read(path);s,l,a=judge(data,task,r['text'],world['tests'][r['index']])
            rows.append({'task':task,'episode':world['episode'],'reader':r['model'],'source':r['source'],'n':r['n'],'index':r['index'],'correct':s,'lenient':l,'extracted':a,'status':r['status'],'chars':len(r['text'])})
        for path in sorted((directory/'generations').glob(f'*/{name}__*.json')):
            r=read(path);s,l,a=judge(data,task,r['text'],world['practice'][r['index']])
            gens.append({'task':task,'episode':world['episode'],'model':r['model'],'index':r['index'],'correct':s,'lenient':l,'has_answer':bool(LINE.findall(r['text'])),'status':r['status'],'chars':len(r['text'])})
    write(directory/'scores.json',{'evaluations':rows,'generations':gens});return rows,gens


def calibration():
    rows,_=score('calibration');acc=defaultdict(list)
    for r in rows: acc[r['task'],r['reader']].append(r['correct'])
    tasks=sorted({r['task'] for r in rows});table={t:{m:float(np.mean(acc[t,m])) for m in NAMES} for t in tasks}
    eligible=[t for t in tasks if .2<=np.mean(list(table[t].values()))<=.8]
    chosen=sorted(eligible,key=lambda t:-np.mean([p*(1-p) for p in table[t].values()]))[:3]
    write(DIRECTORY/'calibration/summary.json',{'accuracy':table,'eligible':eligible,'chosen':chosen});write(DIRECTORY/'matrix_plan.json',{'rule':'NTRACE_PROTOCOL.md calibration','tasks':chosen})
    for t in tasks: print(f"{t:24s} mean {np.mean(list(table[t].values())):.2f} | "+' '.join(f"{SHORT[m][:9]} {table[t][m]:.2f}" for m in NAMES)+(' <- chosen' if t in chosen else ''))


def tensor(rows,key):
    """acc[(task,episode)] -> 7x7 matrix (reader x source) at N=8, plus initial vector."""
    cells=defaultdict(list)
    for r in rows: cells[r['task'],r['episode'],r['reader'],r['source'],r['n']].append(r[key])
    units=sorted({(r['task'],r['episode']) for r in rows});M=len(NAMES)
    mats={u:np.array([[np.mean(cells[u+(a,b,8)]) for b in NAMES] for a in NAMES]) for u in units}
    init={u:np.array([np.mean(cells[u+(a,'initial',0)]) for a in NAMES]) for u in units}
    return units,mats,init


def self_interaction(m): M=len(m);return float(np.trace(m)/M-(m.sum()-np.trace(m))/(M*(M-1)))
def double_centre(m): return m-m.mean(1,keepdims=True)-m.mean(0,keepdims=True)+m.mean()
def sign_flip(values):
    values=np.asarray(values);n=len(values);signs=1-2*((np.arange(2**n)[:,None]>>np.arange(n))&1)
    return float(np.mean((signs*values).mean(1)>=values.mean()-1e-12))
def bootstrap(values,resamples=20000,seed=20261005):
    values=np.asarray(values);idx=np.random.default_rng(seed).integers(0,len(values),(resamples,len(values)));return [float(x) for x in np.percentile(values[idx].mean(1),[2.5,97.5])]


def family_gradient(m):
    """For readers with a same-family partner: own - same-family source, and same-family - mean cross-family source."""
    out={}
    for i,r in enumerate(NAMES):
        mates=[j for j,s in enumerate(NAMES) if s!=r and MODELS[s]['family']==MODELS[r]['family']];others=[j for j,s in enumerate(NAMES) if MODELS[s]['family']!=MODELS[r]['family']]
        if mates: out[r]={'own_minus_family':float(m[i,i]-m[i,mates].mean()),'family_minus_cross':float(m[i,mates].mean()-m[i,others].mean())}
    return out


def matrix():
    rows,gens=score('matrix');result={}
    for key in ['correct','lenient']:
        units,mats,init=tensor(rows,key);S=[self_interaction(mats[u]) for u in units];tasks=sorted({u[0] for u in units})
        pooled=np.mean([mats[u] for u in units],0);fam=[family_gradient(mats[u]) for u in units]
        result[key]={'S':float(np.mean(S)),'S_ci':bootstrap(S),'p_one_sided':sign_flip(S),'S_by_task':{t:[float(np.mean([s for u,s in zip(units,S) if u[0]==t])),*bootstrap([s for u,s in zip(units,S) if u[0]==t])] for t in tasks},
                     'pooled':pooled.tolist(),'initial':np.mean([init[u] for u in units],0).tolist(),'double_centred_own':dict(zip(NAMES,np.diag(double_centre(pooled)).tolist())),
                     'by_task':{t:np.mean([mats[u] for u in units if u[0]==t],0).tolist() for t in tasks},
                     'family':{r:{k:[float(np.mean([f[r][k] for f in fam])),*bootstrap([f[r][k] for f in fam])] for k in fam[0][r]} for r in fam[0]},
                     'pairwise_H':{f'{a}|{b}':float(((pooled[i,i]-pooled[i,j])+(pooled[j,j]-pooled[j,i]))/2) for i,a in enumerate(NAMES) for j,b in enumerate(NAMES) if i<j}}
    g=defaultdict(list)
    for x in gens: g[x['task'],x['model']].append(x)
    result['generations']={f'{t}|{m}':{k:float(np.mean([x[k] for x in v])) for k in ['correct','lenient','has_answer','chars']}|{'truncated':float(np.mean([x['status']!='ok' for x in v]))} for (t,m),v in g.items()}
    write(DIRECTORY/'matrix/analysis.json',result)
    for key in ['correct','lenient']:
        r=result[key];print(key,'S=%+.2f pp [%+.2f, %+.2f] p=%.4f'%(100*r['S'],100*r['S_ci'][0],100*r['S_ci'][1],r['p_one_sided']),{t:round(100*v[0],1) for t,v in r['S_by_task'].items()})
    plots(result)


def plots(result):
    r=result['correct'];pooled=np.array(r['pooled'])*100;labels=[SHORT[m] for m in NAMES]
    figure,axes=plt.subplots(1,2,figsize=(13,5.6))
    for axis,data,title,cmap in [(axes[0],pooled,'Accuracy at N = 8 (%), pooled over tasks','viridis'),(axes[1],double_centre(pooled),'Reader × source interaction (double-centred, pp)','RdBu_r')]:
        lim=np.abs(data).max() if cmap=='RdBu_r' else None
        image=axis.imshow(data,cmap=cmap,vmin=-lim if lim else None,vmax=lim)
        for i in range(len(NAMES)):
            for j in range(len(NAMES)): axis.text(j,i,f'{data[i,j]:.0f}' if cmap=='viridis' else f'{data[i,j]:+.1f}',ha='center',va='center',color='white' if cmap=='viridis' and data[i,j]<data.mean() else 'black')
            axis.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='k',lw=2))
        axis.set_xticks(range(len(NAMES)));axis.set_xticklabels(labels,rotation=40,ha='right');axis.set_yticks(range(len(NAMES)));axis.set_yticklabels(labels)
        axis.set_xlabel('source of the 8 worked examples');axis.set_ylabel('reader');axis.set_title(title);figure.colorbar(image,ax=axis,fraction=.046)
    figure.tight_layout();figure.savefig(DIRECTORY/'matrix/matrix.png',dpi=140);figure.savefig(REPORTS/'ntrace-matrix.png',dpi=140)
    figure,axis=plt.subplots(figsize=(7,3.2));items=[('pooled',r['S'],*r['S_ci'])]+[(t,*v) for t,v in r['S_by_task'].items()]
    for k,(name,v,lo,hi) in enumerate(items): axis.errorbar(100*v,k,xerr=[[100*(v-lo)],[100*(hi-v)]],fmt='o',color='k' if k==0 else 'C0')
    axis.axvline(0,color='grey',lw=.8);axis.set_yticks(range(len(items)));axis.set_yticklabels([i[0] for i in items]);axis.invert_yaxis()
    axis.set_xlabel('own − other source accuracy, S (pp; 95% bootstrap over task-episodes)');figure.tight_layout()
    figure.savefig(DIRECTORY/'matrix/self_interaction.png',dpi=150,bbox_inches='tight');figure.savefig(REPORTS/'ntrace-self-interaction.png',dpi=150,bbox_inches='tight')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['calibration','matrix']);args=parser.parse_args()
    {'calibration':calibration,'matrix':matrix}[args.phase]()
