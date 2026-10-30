"""Frozen statistics for LANG-APPLY (LANGAPPLY_PROTOCOL.md): the language contrast with recognition held fixed."""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import lang_search as LS
from edge_analyze import sign_flip_two
from lang_analyze import SHORT,relation
from lang_apply import OUT,eval_path
from mlayout_search import read,write
from task_search_api import ROOT

REPORTS=Path(__file__).resolve().parents[1]/'reports';NAMES=LS.NAMES;M=len(NAMES);K=LS.COUNTS
REL=np.array([[relation(r,s) for s in NAMES] for r in NAMES])


def boot(v,seed=20261007): v=np.asarray(v,float);i=np.random.default_rng(seed).integers(0,len(v),(20000,len(v)));return [float(x) for x in np.percentile(v[i].mean(1),[2.5,97.5])]
def summary(v): v=np.asarray(v,float);return {'mean':float(v.mean()),'ci':boot(v),'p_two_sided':sign_flip_two(v,100000),'n':len(v)}
def fmt(s): return '%+.1f pp [%+.1f, %+.1f] p=%.4f (n=%d)'%(100*s['mean'],100*s['ci'][0],100*s['ci'][1],s['p_two_sided'],s['n'])
def dc(A): return A-A.mean(1,keepdims=True)-A.mean(0,keepdims=True)+A.mean()
def C(A): D=dc(A);return float(D[REL=='same language'].mean()-D[REL=='other language'].mean())
def kind(w,t): u=w['rule']['units'];a,b=u.index(t['i']),u.index(t['j']);return 'reverse' if a>b else 'forward one-step' if b-a==1 else 'forward multi-step'


def load(setname):
    d=ROOT.resolve()/setname;eps=read(d/'complete.json')['usable'];worlds={n:read(d/'worlds'/f'{n}.json') for n in eps};E={}
    for n in eps:
        w=worlds[n];tests=w['tests']
        hit=lambda path,i:float(LS.correct(read(path)['text'],tests[i]))
        std=lambda r,c,i:hit(d/'evaluations'/r/f"{n}__{c.replace(':','-')}__{i:02}.json",i)
        app=lambda r,c,i:hit(eval_path(setname,r,n,c,i),i)
        e={'kinds':[kind(w,t) for t in tests],'none':np.array([[std(r,'initial',i) for i in range(LS.TESTS)] for r in NAMES]),'inj_none':np.array([[app(r,'inj:none',i) for i in range(LS.TESTS)] for r in NAMES])}
        for k in K:  # item-level arrays: reader x source x item
            e['std',k]=np.array([[[std(r,f'src:{s}:{k}',i) for i in range(LS.TESTS)] for s in NAMES] for r in NAMES])
            for arm in ['inj','self']: e[arm,k]=np.array([[[app(r,f'{arm}:{s}:{k}',i) for i in range(LS.TESTS)] for s in NAMES] for r in NAMES])
            e['said',k]=np.array([[float(LS.probe_correct(read(d/'probes'/r/f"{n}__src-{s}-{k}.json")['text'],w)) for s in NAMES] for r in NAMES])
        E[f'{setname}/{n}']=e
    return E


def main():
    E={};sets={}
    for s in ['lang_main','lang_main2']:
        if (OUT/s/'complete.json').exists(): part=load(s);E|=part;sets[s]=list(part)
    eps=list(E);out={'episodes':{s:len(v) for s,v in sets.items()}}
    out['PRIMARY_C_inj_mean_over_N']=summary([np.mean([C(E[n]['inj',k].mean(2)) for k in K]) for n in eps])
    for k in K:
        out[f'C_inj_N{k}']=summary([C(E[n]['inj',k].mean(2)) for n in eps]);out[f'C_std_N{k}']=summary([C(E[n]['std',k].mean(2)) for n in eps]);out[f'C_self_N{k}']=summary([C(E[n]['self',k].mean(2)) for n in eps])
        res=[]
        for n in eps:
            D=dc(E[n]['self',k].mean(2));said=E[n]['said',k]==1;a=(REL=='same language')&said;b=(REL=='other language')&said
            if a.any() and b.any(): res.append(float(D[a].mean()-D[b].mean()))
        out[f'C_self_said_correct_N{k}']=summary(res)
        out[f'self_said_correct_rate_N{k}']={x:float(np.mean([E[n]['said',k][REL==x].mean() for n in eps])) for x in ['own','same language','other language']}
        out[f'own_minus_peer_inj_N{k}']=summary([float(dc(E[n]['inj',k].mean(2))[REL=='own'].mean()-dc(E[n]['inj',k].mean(2))[REL=='same language'].mean()) for n in eps])
        out[f'demonstration_value_N{k}']={x:summary([float((E[n]['inj',k].mean(2)-E[n]['inj_none'].mean(1)[:,None])[REL==x].mean()) for n in eps]) for x in ['own','same language','other language']}
        for t in ['forward one-step','forward multi-step','reverse']:
            vals=[C(E[n]['inj',k][:,:,[i for i,x in enumerate(E[n]['kinds']) if x==t]].mean(2)) for n in eps if t in E[n]['kinds']]
            out[f'C_inj_N{k}_{t}']=summary(vals)
    for s,v in sets.items(): out[f'C_inj_mean_over_N_{s}']=summary([np.mean([C(E[n]['inj',k].mean(2)) for k in K]) for n in v])
    if 'lang_main2' in sets: out['replication_C_std_N1_setB']=summary([C(E[n]['std',1].mean(2)) for n in sets['lang_main2']])
    out['ceiling_inj_by_reader']={m:float(np.mean([E[n]['inj',k][r].mean() for n in eps for k in K])) for r,m in enumerate(NAMES)}
    out['inj_none_by_reader']={m:float(np.mean([E[n]['inj_none'][r].mean() for n in eps])) for r,m in enumerate(NAMES)}
    write(OUT/'analysis.json',out)
    print('episodes',out['episodes'])
    print('PRIMARY: C under rule injection, mean over N',fmt(out['PRIMARY_C_inj_mean_over_N']))
    for s in sets: print('   set',s,fmt(out[f'C_inj_mean_over_N_{s}']))
    if 'replication_C_std_N1_setB' in out: print('replication of LANG primary on fresh episodes (standard, N=1)',fmt(out['replication_C_std_N1_setB']))
    for k in K:
        print(f'N={k}: standard {fmt(out[f"C_std_N{k}"])} | self-stated {fmt(out[f"C_self_N{k}"])} | injected {fmt(out[f"C_inj_N{k}"])}')
        print(f'      self-stated, rule stated correctly: {fmt(out[f"C_self_said_correct_N{k}"])} | stated correctly',{x:round(100*v) for x,v in out[f'self_said_correct_rate_N{k}'].items()})
        print(f'      injected by item type:',{t:'%+.1f'%(100*out[f'C_inj_N{k}_{t}']['mean']) for t in ['forward one-step','forward multi-step','reverse']},'| own - peer (inj)',fmt(out[f'own_minus_peer_inj_N{k}']))
        print(f'      demonstration value over rule-only:',{x:'%+.1f'%(100*v['mean']) for x,v in out[f'demonstration_value_N{k}'].items()})
    print('accuracy under injection by reader',{SHORT[m]:round(100*v) for m,v in out['ceiling_inj_by_reader'].items()},'| rule only, no practice',{SHORT[m]:round(100*v) for m,v in out['inj_none_by_reader'].items()})
    plot(out)


def plot(out):
    figure,axes=plt.subplots(1,2,figsize=(14,5));colors={'std':'#9a9893','self':'#eda100','inj':'#2a78d6'};labels={'std':'standard (LANG)','self':'reader states the rule itself first','inj':'correct rule injected as the reader’s own words'}
    for j,arm in enumerate(['std','self','inj']):
        m=[100*out[f'C_{arm}_N{k}']['mean'] for k in K];ci=np.array([[100*x for x in out[f'C_{arm}_N{k}']['ci']] for k in K]);x=np.arange(len(K))+(j-1)*.25
        axes[0].bar(x,m,.24,color=colors[arm],label=labels[arm],yerr=[m-ci[:,0],ci[:,1]-m],capsize=3,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
    axes[0].axhline(0,color='#52514e',lw=.8);axes[0].set_xticks(range(len(K)));axes[0].set_xticklabels([f'N = {k}' for k in K]);axes[0].set_ylabel('same-language peer − other language (pp, interaction)')
    axes[0].set_title('A · Language contrast with recognition held fixed',loc='left');axes[0].legend(frameon=False);axes[0].grid(axis='y',color='#e5e4df',zorder=0);axes[0].spines[['top','right']].set_visible(False)
    for x,c in [('own','#2a78d6'),('same language','#1baf7a'),('other language','#eb6834')]:
        m=[100*out[f'demonstration_value_N{k}'][x]['mean'] for k in K];ci=np.array([[100*y for y in out[f'demonstration_value_N{k}'][x]['ci']] for k in K])
        axes[1].errorbar(K,m,yerr=[m-ci[:,0],ci[:,1]-m],marker='o',capsize=3,color=c,label=x)
    axes[1].axhline(0,color='#52514e',lw=.8);axes[1].set_xticks(K);axes[1].set_xlabel('number of worked solutions shown');axes[1].set_ylabel('accuracy gain over the rule alone (pp)')
    axes[1].set_title('B · Value of the demonstrations once the rule is known',loc='left');axes[1].legend(frameon=False);axes[1].spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(REPORTS/'lang-apply.png',dpi=130)


if __name__=='__main__': main()
