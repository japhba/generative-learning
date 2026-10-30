"""Frozen statistics for PROC-REC (PROCREC_PROTOCOL.md): own vs same-language peer on richer recognition measures."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import proc_search as PS
from mlayout_search import read,write
from proc_analyze import M,NAMES,REL,SHORT,dc,fmt,summary
from proc_rec import EXTRA,OUT,probe_path,said_yes
from task_search_api import ROOT

REPORTS=Path(__file__).resolve().parents[1]/'reports';K=PS.COUNTS;MEASURES=['step identification','verification','recognition composite','deployment']


def contrast(A,a,b): D=dc(A);return float(D[REL==a].mean()-D[REL==b].mean())


def load(setname):
    d=ROOT.resolve()/setname;eps=read(d/'complete.json')['usable'];worlds={n:read(d/'worlds'/f'{n}.json') for n in eps};items=read(OUT/setname/'items.json');E={};yes=[]
    for n in eps:
        w=worlds[n];L=w['length']
        for k in K:
            c=lambda s:f'src:{s}:{k}';step=np.zeros((M,M));ver=np.zeros((M,M));dep=np.zeros((M,M))
            for a,r in enumerate(NAMES):
                for b,s in enumerate(NAMES):
                    orig=[PS.chosen(read(d/'probes'/r/f"{n}__{c(s).replace(':','-')}__{j}.json")['text'])==w['probes'][j]['answer'] for j in range(L)]
                    more=[PS.chosen(read(probe_path(setname,r,n,c(s),f'set{x}_step{j}'))['text'])==items[n]['sets'][x][j]['answer'] for x in range(EXTRA) for j in range(L)]
                    said=[said_yes(read(probe_path(setname,r,n,c(s),f'verify{v}'))['text']) for v in range(len(items[n]['chains']))];yes+=said
                    step[a,b]=np.mean(orig+more);ver[a,b]=np.mean([y==ch['correct'] for y,ch in zip(said,items[n]['chains'])])
                    dep[a,b]=np.mean([PS.correct(read(d/'evaluations'/r/f"{n}__{c(s).replace(':','-')}__{i:02}.json")['text'],w['tests'][i]) for i in range(PS.TESTS)])
            E[f'{setname}/{n}',k]={'step identification':step,'verification':ver,'recognition composite':(step+ver)/2,'deployment':dep}
    return E,[f'{setname}/{n}' for n in eps],yes


def main():
    E={};sets={};yes=[]
    for s in ['proc_main','proc_main2']:
        if (OUT/s/'complete.json').exists(): e,eps,y=load(s);E|=e;sets[s]=eps;yes+=y
    eps=[n for v in sets.values() for n in v];out={'episodes':{s:len(v) for s,v in sets.items()}}
    mean_self=lambda m,ns:[np.mean([contrast(E[n,k][m],'own','same language') for k in K]) for n in ns]
    out['PRIMARY_self_recognition_composite']=summary(mean_self('recognition composite',eps))
    for m in MEASURES:
        out[f'self_{m}_meanN']=summary(mean_self(m,eps))
        for k in K: out[f'self_{m}_N{k}']=summary([contrast(E[n,k][m],'own','same language') for n in eps]);out[f'language_{m}_N{k}']=summary([contrast(E[n,k][m],'same language','other language') for n in eps])
        out[f'self_{m}_trend_N4_minus_N1']=summary([contrast(E[n,4][m],'own','same language')-contrast(E[n,1][m],'own','same language') for n in eps])
        out[f'level_{m}_by_reader']={x:float(np.mean([E[n,k][m][r].mean() for n in eps for k in K])) for r,x in enumerate(NAMES)}
    out['dissociation_deployment_minus_composite']=summary(np.subtract(mean_self('deployment',eps),mean_self('recognition composite',eps)))
    if 'proc_main2' in sets: out['replication_self_deployment_setB']=summary(mean_self('deployment',sets['proc_main2']))
    for s,v in sets.items(): out[f'self_composite_meanN_{s}']=summary(mean_self('recognition composite',v))
    out['verification_yes_rate']=float(np.mean([y is True for y in yes]));out['verification_unparsed']=float(np.mean([y is None for y in yes]))
    write(OUT/'analysis.json',out)
    print('episodes',out['episodes'])
    print('PRIMARY self (own - same-language peer) on recognition composite, mean over N:',fmt(out['PRIMARY_self_recognition_composite']))
    for s in sets: print('   set',s,fmt(out[f'self_composite_meanN_{s}']))
    if 'replication_self_deployment_setB' in out: print('replication: self in deployment on fresh set B',fmt(out['replication_self_deployment_setB']))
    print('dissociation (self deployment - self composite)',fmt(out['dissociation_deployment_minus_composite']))
    for m in MEASURES:
        print(f'{m:22s} mean over N {fmt(out[f"self_{m}_meanN"])} | trend N4-N1 {fmt(out[f"self_{m}_trend_N4_minus_N1"])}')
        print('   by N:',' | '.join(f'N={k} {100*out[f"self_{m}_N{k}"]["mean"]:+.1f} [{100*out[f"self_{m}_N{k}"]["ci"][0]:+.1f}, {100*out[f"self_{m}_N{k}"]["ci"][1]:+.1f}]' for k in K),'| language:',' '.join(f'{100*out[f"language_{m}_N{k}"]["mean"]:+.1f}' for k in K))
        print('   level by reader',{SHORT[x][:10]:round(100*v) for x,v in out[f'level_{m}_by_reader'].items()})
    print('verification: yes rate %.0f%% (balanced design: 50%%), unparsed %.1f%%'%(100*out['verification_yes_rate'],100*out['verification_unparsed']))
    plot(out)


def plot(out):
    figure,axis=plt.subplots(figsize=(10,4.6));colors={'step identification':'#eda100','verification':'#1baf7a','recognition composite':'#eb6834','deployment':'#2a78d6'}
    for j,m in enumerate(MEASURES):
        mm=np.array([100*out[f'self_{m}_N{k}']['mean'] for k in K]);ci=np.array([[100*x for x in out[f'self_{m}_N{k}']['ci']] for k in K]);x=np.arange(len(K))+(j-1.5)*.2
        axis.bar(x,mm,.19,color=colors[m],label=m,yerr=[mm-ci[:,0],ci[:,1]-mm],capsize=2,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
    axis.axhline(0,color='#52514e',lw=.8);axis.set_xticks(range(len(K)));axis.set_xticklabels([f'N = {k}' for k in K]);axis.set_ylabel('own − same-language peer (pp, interaction)')
    axis.set_title(f"Own − same-language peer: recognition measures vs deployment ({sum(out['episodes'].values())} episodes)",loc='left');axis.legend(frameon=False,ncol=2);axis.grid(axis='y',color='#e5e4df',zorder=0);axis.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(REPORTS/'proc-recognition.png',dpi=130)


if __name__=='__main__': main()
