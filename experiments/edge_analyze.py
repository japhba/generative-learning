"""Frozen statistics for the edge-of-capability study (EDGE_PROTOCOL.md); the calibration phase is exploratory."""
import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import re

from edge_search import DIRECTORY,NAMES,POOL,TESTS,conditions,correct,eval_path,gen_path,read,write
from mlayout_search import final_answer
from mlayout_tasks import normalize

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen3_1_7b':'Qwen3-1.7B','llama32_3b':'Llama-3.2-3B','smollm3_3b':'SmolLM3-3B','phi4_mini':'Phi-4-mini'}
COLORS=['#2a78d6','#eb6834','#1baf7a','#eda100']
M=len(NAMES)


def sign_flip(values,draws=200000,seed=20261006):
    values=np.asarray(values);signs=np.random.default_rng(seed).choice([-1.,1.],(draws,len(values)));return float((1+np.sum((signs*values).mean(1)>=values.mean()-1e-12))/(draws+1))
def bootstrap(values,resamples=20000,seed=20261006):
    values=np.asarray(values);idx=np.random.default_rng(seed).integers(0,len(values),(resamples,len(values)));return [float(x) for x in np.percentile(values[idx].mean(1),[2.5,97.5])]
def sign_flip_two(values,draws=200000,seed=20261006):
    values=np.asarray(values);signs=np.random.default_rng(seed).choice([-1.,1.],(draws,len(values)));return float((1+np.sum(np.abs((signs*values).mean(1))>=abs(values.mean())-1e-12))/(draws+1))
def summary(values): return {'mean':float(np.mean(values)),'ci':bootstrap(values),'p_one_sided':sign_flip(values),'p_two_sided':sign_flip_two(values),'n':len(values)}
def holm(p):
    p=np.asarray(p);order=np.argsort(p);adj=np.maximum.accumulate([(len(p)-k)*p[i] for k,i in enumerate(order)]);out=np.empty(len(p));out[order]=np.minimum(adj,1);return out
def S(A): return float(np.trace(A)/M-(A.sum()-np.trace(A))/(M*(M-1)))
def own_effects(A): return np.array([A[r,r]-np.mean([A[r,s] for s in range(M) if s!=r]) for r in range(M)])
def double_centre(A): return A-A.mean(1,keepdims=True)-A.mean(0,keepdims=True)+A.mean()


def quality_adjusted(A):
    """Q_r = own effect minus how much more the own source helps third-party readers than the foreign sources do."""
    h=lambda x,r:np.mean([A[t,x] for t in range(M) if t not in (r,x)])
    return np.array([(A[r,r]-np.mean([A[r,s] for s in range(M) if s!=r]))-(h(r,r)-np.mean([h(s,r) for s in range(M) if s!=r])) for r in range(M)])


def logit_S(A):
    L=np.log(((12*A+.5)/13)/(1-(12*A+.5)/13));return float(np.mean(np.diag(double_centre(L))))


QUESTION=re.compile(r'How many (\w+)s are in (\d+) (\w+)s\?')
def kind(w,question):
    target,k,unit=QUESTION.fullmatch(question).groups();big,mid,small=w['rule']['units'];return {(big,mid):'big-mid',(mid,small):'mid-small',(big,small):'big-small'}[unit,target],int(k)
def value(text):
    found=re.findall(r'-?\d+(?:\.\d+)?',final_answer(text).replace(',',''));return float(found[-1]) if found else None


def adoption(phase,E,mixed):
    """Amendment 2: share of test items (same conversion type as a wrong practice attempt) answered with k'·f, f = wrong answer / k."""
    d=DIRECTORY/phase;matched=read(d/'matched.json');rates={'own':[],'peers':[],'own_initial':[],'peers_initial':[]}
    for n in mixed:
        w=read(d/'worlds'/f'{n}.json');tests=[kind(w,t['question']) for t in w['tests']]
        factors={s:[(kind(w,w['pool'][x['index']]['question'])[0],v/kind(w,w['pool'][x['index']]['question'])[1]) for x in matched[n]['mixed'] if not x['correct'] for v in [value(read(gen_path(d,s,n,x['index']))['samples'][x['sample'][s]]['text'])] if v is not None] for s in NAMES}
        def rate(r,c,s):
            hits=[]
            for i,(t,k) in enumerate(tests):
                predicted=[k*f for kk,f in factors[s] if kk==t]
                if predicted: v=value(read(eval_path(d,NAMES[r],n,c,i))['text']);hits.append(v is not None and not correct(read(eval_path(d,NAMES[r],n,c,i))['text'],w['tests'][i]) and any(abs(v-p)<1e-6 for p in predicted))
            return np.mean(hits) if hits else np.nan
        for r in range(M):
            rates['own'].append(rate(r,f'mixed:{NAMES[r]}',NAMES[r]));rates['peers'].append(np.nanmean([rate(r,f'mixed:{NAMES[s]}',NAMES[s]) for s in range(M) if s!=r]))
            rates['own_initial'].append(rate(r,'initial',NAMES[r]));rates['peers_initial'].append(np.nanmean([rate(r,'initial',NAMES[s]) for s in range(M) if s!=r]))
    a={k:np.array(v,float) for k,v in rates.items()};ok=~np.isnan(a['own'])&~np.isnan(a['peers'])&~np.isnan(a['own_initial'])&~np.isnan(a['peers_initial'])
    return {k:float(np.mean(v[ok])) for k,v in a.items()}|{'own_minus_peers':summary(a['own'][ok]-a['peers'][ok]),'own_minus_peers_initial':summary(a['own_initial'][ok]-a['peers_initial'][ok]),'n_reader_episodes':int(ok.sum())}


def load(phase):
    d=DIRECTORY/phase;design=read(d/'design.json');assert (d/'complete.json').exists();matched=read(d/'matched.json');E={}
    for n in design['worlds']:
        w=read(d/'worlds'/f'{n}.json');acc={(r,c):float(np.mean([correct(read(eval_path(d,r,n,c,i))['text'],w['tests'][i]) for i in range(TESTS)])) for c in conditions(matched[n]) for r in NAMES}
        E[n]={'family':w['family'],'initial':np.array([acc[r,'initial'] for r in NAMES]),'attempts':np.array([np.mean([correct(s['text'],w['pool'][i]) for i in range(POOL) for s in read(gen_path(d,m,n,i))['samples']]) for m in NAMES])}
        for arm in ['mixed','correct']:
            if matched[n][arm]: E[n][arm]=np.array([[acc[r,f'{arm}:{s}'] for s in NAMES] for r in NAMES])
    return design,matched,E


def analyse(phase):
    design,matched,E=load(phase);mixed=[n for n in E if 'mixed' in E[n]];both=[n for n in mixed if 'correct' in E[n]];allc=[n for n in E if 'correct' in E[n]]
    families=sorted({E[n]['family'] for n in E});out={'episodes':len(E),'mixed_episodes':len(mixed),'correct_episodes':len(allc),'both':len(both)}
    out['primary_S_mixed']=summary([S(E[n]['mixed']) for n in mixed])
    out['S_mixed_by_family']={f:summary([S(E[n]['mixed']) for n in mixed if E[n]['family']==f]) for f in families if any(E[n]['family']==f for n in mixed)}
    out['S_correct']=summary([S(E[n]['correct']) for n in allc]) if allc else None
    out['S_mixed_minus_correct']=summary([S(E[n]['mixed'])-S(E[n]['correct']) for n in both]) if both else None
    out['quality_adjusted']=summary([quality_adjusted(E[n]['mixed']).mean() for n in mixed])
    out['logit_S']=summary([logit_S(E[n]['mixed']) for n in mixed])
    own=np.array([own_effects(E[n]['mixed']) for n in mixed]);base=np.array([E[n]['initial'] for n in mixed])
    per=[summary(own[:,r]) for r in range(M)];adjusted=holm([p['p_one_sided'] for p in per])
    out['per_reader']={m:{**p,'p_holm':float(h)} for m,p,h in zip(NAMES,per,adjusted)}
    d,b=own.ravel(),base.ravel();cuts=np.quantile(b,[1/3,2/3]);tert=np.digitize(b,cuts)
    X=np.column_stack([np.ones_like(b),b,b**2]);coef=np.linalg.lstsq(X,d,rcond=None)[0]
    rng=np.random.default_rng(1);boots=[]
    for _ in range(5000):
        k=rng.integers(0,len(mixed),len(mixed));dd=own[k].ravel();bb=base[k].ravel();boots.append(np.linalg.lstsq(np.column_stack([np.ones_like(bb),bb,bb**2]),dd,rcond=None)[0])
    out['dose_response']={'tertile_edges':cuts.tolist(),'by_tertile':[{'mean':float(d[tert==k].mean()),'n':int((tert==k).sum())} for k in range(3)],'quadratic':coef.tolist(),'quadratic_ci':np.percentile(boots,[2.5,97.5],axis=0).T.tolist()}
    out['practice_effect']={arm:summary([float((E[n][arm]-E[n]['initial'][:,None]).mean()) for n in E if arm in E[n]]) for arm in ['mixed','correct'] if any(arm in E[n] for n in E)}
    out['matrix_mixed']=np.mean([E[n]['mixed'] for n in mixed],0).tolist();out['matrix_correct']=np.mean([E[n]['correct'] for n in allc],0).tolist() if allc else None
    out['initial']={m:float(np.mean([E[n]['initial'][r] for n in E])) for r,m in enumerate(NAMES)};out['attempts']={m:float(np.mean([E[n]['attempts'][r] for n in E])) for r,m in enumerate(NAMES)}
    out['by_family_yield']={f:{'mixed':float(np.mean([bool(matched[n]['mixed']) for n in E if E[n]['family']==f])),'correct':float(np.mean([bool(matched[n]['correct']) for n in E if E[n]['family']==f]))} for f in families}
    out['adoption']=adoption(phase,E,mixed)
    trio=[NAMES.index(m) for m in ['qwen3_1_7b','smollm3_3b','phi4_mini']];sub=lambda A:A[np.ix_(trio,trio)];S3=lambda A:float(np.trace(sub(A))/3-(sub(A).sum()-np.trace(sub(A)))/6)
    H=lambda A,i,j:((A[i,i]-A[i,j])+(A[j,j]-A[j,i]))/2  # amendment 4: capability-matched checks (Llama-3.2-3B is far weaker than the rest)
    out['without_llama']={arm:summary([S3(E[n][arm]) for n in eps]) for arm,eps in [('mixed',mixed),('correct',allc)]}
    out['pairwise_H']={arm:{f'{NAMES[i]}|{NAMES[j]}':summary([H(E[n][arm],i,j) for n in eps]) for i in range(M) for j in range(i+1,M)} for arm,eps in [('mixed',mixed),('correct',allc)]}
    label={'main':'label','main2':'label2'}.get(phase)
    if label and (DIRECTORY/label/'complete.json').exists():
        L={n:np.array([[float(np.mean([correct(read(eval_path(DIRECTORY/label,r,n,f'label:{x}',i))['text'],read(DIRECTORY/phase/'worlds'/f'{n}.json')['tests'][i]) for i in range(TESTS)])) for x in NAMES] for r in NAMES]) for n in mixed}
        own_l=np.array([own_effects(L[n]) for n in mixed]);per_l=[summary(own_l[:,r]) for r in range(M)];adj_l=holm([p['p_one_sided'] for p in per_l])
        out['label']={'S':summary([S(L[n]) for n in mixed]),'S_minus_unlabelled':summary([S(L[n])-S(E[n]['mixed']) for n in mixed]),'matrix':np.mean([L[n] for n in mixed],0).tolist(),
                      'practice_effect':summary([float((L[n]-E[n]['initial'][:,None]).mean()) for n in mixed]),'labelled_minus_unlabelled':summary([float((L[n]-E[n]['mixed']).mean()) for n in mixed]),
                      'per_reader':{m:{**p,'p_holm':float(h)} for m,p,h in zip(NAMES,per_l,adj_l)},'without_llama':summary([S3(L[n]) for n in mixed]),'without_llama_minus_unlabelled':summary([S3(L[n])-S3(E[n]['mixed']) for n in mixed])}
    write(DIRECTORY/phase/'analysis.json',out);report(out);plot(out,E,mixed,allc,phase)


def fmt(s): return '%+.1f pp [%+.1f, %+.1f] p1=%.4f p2=%.4f (n=%d)'%(100*s['mean'],100*s['ci'][0],100*s['ci'][1],s['p_one_sided'],s['p_two_sided'],s['n'])


def report(out):
    print('episodes',out['episodes'],'mixed',out['mixed_episodes'],'all-correct',out['correct_episodes'])
    print('PRIMARY S (mixed)',fmt(out['primary_S_mixed']))
    for f,s in out['S_mixed_by_family'].items(): print('  ',f,fmt(s))
    for k in ['S_correct','S_mixed_minus_correct','quality_adjusted','logit_S']:
        if out[k]: print(k,fmt(out[k]))
    for m,p in out['per_reader'].items(): print('  own effect',SHORT[m],fmt(p),'holm %.3f'%p['p_holm'])
    for arm,v in out['without_llama'].items(): print('S without Llama',arm,fmt(v))
    for arm,v in out['pairwise_H'].items():
        for k,h in v.items(): print('  H',arm,'%-24s'%k,fmt(h))
    print('dose-response by tertile of no-practice accuracy',out['dose_response'])
    for arm,s in out['practice_effect'].items(): print('practice effect',arm,fmt(s))
    print('initial',{SHORT[m]:round(100*v) for m,v in out['initial'].items()},'attempts',{SHORT[m]:round(100*v) for m,v in out['attempts'].items()})
    a=out['adoption'];print('error adoption: own %.1f%% peers %.1f%% | same predictions without practice: own %.1f%% peers %.1f%%'%(100*a['own'],100*a['peers'],100*a['own_initial'],100*a['peers_initial']),'own-peers',fmt(a['own_minus_peers']),'baseline',fmt(a['own_minus_peers_initial']))
    if 'label' in out:
        l=out['label'];print('LABELLED S',fmt(l['S']),'| minus unlabelled',fmt(l['S_minus_unlabelled']),'| practice effect',fmt(l['practice_effect']),'| labelled - unlabelled',fmt(l['labelled_minus_unlabelled']))
        for m,p in l['per_reader'].items(): print('  labelled own effect',SHORT[m],fmt(p))
        print('  labelled S without Llama',fmt(l['without_llama']),'| minus unlabelled',fmt(l['without_llama_minus_unlabelled']))
    print('matrix (mixed, %; rows reader, cols source)');[print('%-13s'%SHORT[m],' '.join(('[%3.0f]' if j==i else ' %3.0f ')%(100*v) for j,v in enumerate(row))) for i,(m,row) in enumerate(zip(NAMES,out['matrix_mixed']))]


def plot(out,E,mixed,allc,phase):
    figure,axes=plt.subplots(1,3,figsize=(17,5.2));labels=[SHORT[m] for m in NAMES]
    A=np.array(out['matrix_mixed'])*100
    for axis,data,title,cmap in [(axes[0],A,'Mixed practice (2 right, 2 wrong): accuracy (%)','viridis'),(axes[1],double_centre(A),'Reader × source interaction (pp)','RdBu_r')]:
        lim=np.abs(data).max() if cmap=='RdBu_r' else None;image=axis.imshow(data,cmap=cmap,vmin=-lim if lim else None,vmax=lim)
        for i in range(M):
            for j in range(M): axis.text(j,i,f'{data[i,j]:.0f}' if cmap=='viridis' else f'{data[i,j]:+.1f}',ha='center',va='center',color='white' if cmap=='viridis' and data[i,j]<data.mean() else 'black')
            axis.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='k',lw=2))
        axis.set_xticks(range(M));axis.set_xticklabels(labels,rotation=30,ha='right');axis.set_yticks(range(M));axis.set_yticklabels(labels);axis.set_xlabel('source of the practice attempts');axis.set_ylabel('reader');axis.set_title(title,loc='left');figure.colorbar(image,ax=axis,fraction=.046)
    axis=axes[2];width=.18;bars=[('no practice',lambda n,r:E[n]['initial'][r],mixed),('own, mixed',lambda n,r:E[n]['mixed'][r,r],mixed),('peers, mixed',lambda n,r:np.mean([E[n]['mixed'][r,s] for s in range(M) if s!=r]),mixed)]
    if allc: bars+=[('own, all correct',lambda n,r:E[n]['correct'][r,r],allc),('peers, all correct',lambda n,r:np.mean([E[n]['correct'][r,s] for s in range(M) if s!=r]),allc)]
    colors=['#9a9893']+COLORS
    for k,(label,f,eps) in enumerate(bars):
        vals=[[f(n,r) for n in eps] for r in range(M)];means=[100*np.mean(v) for v in vals];cis=np.array([100*np.array(bootstrap(v)) for v in vals]).T;x=np.arange(M)+(k-(len(bars)-1)/2)*width
        axis.bar(x,means,width*.92,color=colors[k],label=label,yerr=[np.array(means)-cis[0],cis[1]-np.array(means)],capsize=2,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
    axis.set_xticks(range(M));axis.set_xticklabels(labels,rotation=30,ha='right');axis.set_ylabel('test accuracy (%)');axis.set_title('Practice helps? (episode means, 95% CI)',loc='left');axis.legend(frameon=False);axis.grid(axis='y',color='#e5e4df',zorder=0);axis.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(DIRECTORY/phase/'edge.png',dpi=140);figure.savefig(REPORTS/f'edge-{phase}.png',dpi=140)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['calibration','main','main2']);args=parser.parse_args();analyse(args.phase)
