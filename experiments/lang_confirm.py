"""Frozen statistics for the LANG confirmatory phase (LANG_PROTOCOL.md): language contrast, self beyond language,
recognition, application given recognition, uplift by relation, per reader. Writes lang_main/analysis.json and a figure."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_analyze import sign_flip_two
from lang_analyze import SHORT,relation
from lang_search import COUNTS,DIRECTORY,LANGUAGE,NAMES,TESTS,correct,eval_path,probe_correct,probe_path,read,write

REPORTS=Path(__file__).resolve().parents[1]/'reports';M=len(NAMES);RELATIONS=['own','same language','other language']
COLORS={'own':'#2a78d6','same language':'#1baf7a','other language':'#eb6834'}
REL=np.array([[relation(r,s) for s in NAMES] for r in NAMES])


def boot(v,seed=20261007): v=np.asarray(v,float);i=np.random.default_rng(seed).integers(0,len(v),(20000,len(v)));return [float(x) for x in np.percentile(v[i].mean(1),[2.5,97.5])]
def summary(v): v=np.asarray(v,float);return {'mean':float(v.mean()),'ci':boot(v),'p_two_sided':sign_flip_two(v,100000),'n':len(v)}
def fmt(s): return '%+.1f pp [%+.1f, %+.1f] p=%.4f (n=%d)'%(100*s['mean'],100*s['ci'][0],100*s['ci'][1],s['p_two_sided'],s['n'])
def dc(A): return A-A.mean(1,keepdims=True)-A.mean(0,keepdims=True)+A.mean()


def main():
    done=read(DIRECTORY/'complete.json');eps=done['usable'];worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in eps}
    acc=lambda r,n,c:float(np.mean([correct(read(eval_path(r,n,c,i))['text'],worlds[n]['tests'][i]) for i in range(TESTS)]))
    rec=lambda r,n,c:float(probe_correct(read(probe_path(r,n,c))['text'],worlds[n]))
    A={(n,k):np.array([[acc(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps for k in COUNTS}
    P={(n,k):np.array([[rec(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps for k in COUNTS}
    base={n:np.array([acc(r,n,'initial') for r in NAMES]) for n in eps};answers={n:np.array([acc(r,n,'answers') for r in NAMES]) for n in eps}
    out={'usable_episodes':len(eps),'prepared':len(read(DIRECTORY/'design.json')['worlds'])}
    contrast=lambda D,a,b:float(D[REL==a].mean()-D[REL==b].mean())
    for k in COUNTS:
        out[f'C{k}']=summary([contrast(dc(A[n,k]),'same language','other language') for n in eps])
        out[f'self{k}']=summary([contrast(dc(A[n,k]),'own','same language') for n in eps])
        out[f'recognition_C{k}']=summary([contrast(dc(P[n,k]),'same language','other language') for n in eps])
        out[f'recognition_self{k}']=summary([contrast(dc(P[n,k]),'own','same language') for n in eps])
        out[f'uplift{k}']={rel:summary([float((A[n,k]-base[n][:,None])[REL==rel].mean()) for n in eps]) for rel in RELATIONS}
        out[f'recognised{k}']={rel:float(np.mean([P[n,k][REL==rel].mean() for n in eps])) for rel in RELATIONS}
        cond={rel:[float(A[n,k][(REL==rel)&(P[n,k]==1)].mean()) for n in eps if ((REL==rel)&(P[n,k]==1)).any()] for rel in RELATIONS}
        paired=[float(A[n,k][(REL=='same language')&(P[n,k]==1)].mean()-A[n,k][(REL=='other language')&(P[n,k]==1)].mean()) for n in eps if ((REL=='same language')&(P[n,k]==1)).any() and ((REL=='other language')&(P[n,k]==1)).any()]
        out[f'application_given_recognition{k}']={'by_relation':{rel:float(np.mean(v)) for rel,v in cond.items()},'same_minus_other':summary(paired)}
        out[f'per_reader_C{k}']={m:float(np.mean([dc(A[n,k])[r][REL[r]=='same language'].mean()-dc(A[n,k])[r][REL[r]=='other language'].mean() for n in eps])) for r,m in enumerate(NAMES)}
        out[f'matrix{k}']=np.mean([A[n,k] for n in eps],0).tolist()
    out['trend_C1_minus_C4']=summary([contrast(dc(A[n,1]),'same language','other language')-contrast(dc(A[n,4]),'same language','other language') for n in eps])
    out['none']=float(np.mean([base[n].mean() for n in eps]));out['answers_only']=float(np.mean([answers[n].mean() for n in eps]))
    write(DIRECTORY/'analysis.json',out)
    print('usable episodes',len(eps),'of',out['prepared'],'| no practice %.0f%%, answers only %.0f%%'%(100*out['none'],100*out['answers_only']))
    for k in COUNTS:
        print(f'N={k}: PRIMARY C' if k==1 else f'N={k}: C','(same-language peer - other language)',fmt(out[f'C{k}']),'| self beyond language',fmt(out[f'self{k}']))
        print('      recognition C',fmt(out[f'recognition_C{k}']),'| recognised',{x:round(100*v) for x,v in out[f'recognised{k}'].items()})
        print('      uplift over none',{x:'%+.1f'%(100*v['mean']) for x,v in out[f'uplift{k}'].items()},'| accuracy given recognised',{x:round(100*v) for x,v in out[f'application_given_recognition{k}']['by_relation'].items()},'same-other',fmt(out[f'application_given_recognition{k}']['same_minus_other']))
        print('      per reader C',{SHORT[m]:'%+.1f'%(100*v) for m,v in out[f'per_reader_C{k}'].items()})
    print('efficiency trend C1 - C4',fmt(out['trend_C1_minus_C4']))
    plot(out,A,base,eps)


def plot(out,A,base,eps):
    figure,axes=plt.subplots(1,3,figsize=(17,5))
    for rel in RELATIONS:
        m=[100*out[f'uplift{k}'][rel]['mean'] for k in COUNTS];lo=[100*out[f'uplift{k}'][rel]['ci'][0] for k in COUNTS];hi=[100*out[f'uplift{k}'][rel]['ci'][1] for k in COUNTS]
        axes[0].errorbar(COUNTS,m,yerr=[np.subtract(m,lo),np.subtract(hi,m)],marker='o',capsize=3,color=COLORS[rel],label=rel+(' (reader’s own)' if rel=='own' else ' model' if rel=='same language' else ' models'))
    axes[0].set_xticks(COUNTS);axes[0].set_xlabel('number of worked solutions shown (all correct)');axes[0].set_ylabel('uplift over no practice (pp test accuracy)');axes[0].set_title('A · Practice uplift by source relation',loc='left');axes[0].legend(frameon=False)
    for i,(key,label) in enumerate([('C','same-language peer − other language'),('self','own − same-language peer')]):
        m=[100*out[f'{key}{k}']['mean'] for k in COUNTS];ci=np.array([[100*x for x in out[f'{key}{k}']['ci']] for k in COUNTS])
        axes[1].errorbar(np.array(COUNTS)+(i-.5)*.08,m,yerr=[m-ci[:,0],ci[:,1]-m],marker='o',capsize=3,color=['#1baf7a','#2a78d6'][i],label=label)
    axes[1].axhline(0,color='#52514e',lw=.8);axes[1].set_xticks(COUNTS);axes[1].set_xlabel('number of worked solutions shown');axes[1].set_ylabel('interaction (pp, source and reader effects removed)');axes[1].set_title('B · Language and self contrasts',loc='left');axes[1].legend(frameon=False)
    B=100*np.array(out['matrix1']);image=axes[2].imshow(B,cmap='viridis',vmin=0,vmax=100);labels=[SHORT[m] for m in NAMES]
    for i in range(M):
        for j in range(M): axes[2].text(j,i,f'{B[i,j]:.0f}',ha='center',va='center',color='white' if B[i,j]<60 else 'black')
    for lo in [0,2,4]: axes[2].add_patch(plt.Rectangle((lo-.5,lo-.5),2,2,fill=False,edgecolor='k',lw=2))
    axes[2].set_xticks(range(M));axes[2].set_xticklabels(labels,rotation=35,ha='right');axes[2].set_yticks(range(M));axes[2].set_yticklabels(labels);axes[2].set_xlabel('source of the single worked solution');axes[2].set_ylabel('reader');axes[2].set_title('C · Test accuracy (%) after one solution',loc='left');figure.colorbar(image,ax=axes[2],fraction=.046)
    figure.tight_layout();figure.savefig(REPORTS/'lang-confirmatory.png',dpi=130)


if __name__=='__main__': main()
