"""Figure for PROC-REC amendment 1: own − same-language peer in recognition vs deployment, per reader and for the ceiling readers."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout_search import write
from proc_analyze import M,NAMES,REL,SHORT,summary
from proc_rec import OUT
from proc_rec_analyze import K,load

REPORTS=Path(__file__).resolve().parents[1]/'reports';CEIL=['olmo3_7b','phi4_mini','glm4_9b','internlm3_8b'];C={'step identification':'#eda100','deployment':'#2a78d6'}


def own_peer(A,r): return A[r,r]-A[r,[j for j in range(M) if REL[r,j]=='same language'][0]]


def main():
    E={};eps=[]
    for s in ['proc_main','proc_main2']: e,ep,_=load(s);E|=e;eps+=ep
    per={m:{x:summary([np.mean([own_peer(E[n,k][x],r) for k in K]) for n in eps]) for x in C} for r,m in enumerate(NAMES)}
    level={m:float(np.mean([E[n,k]['step identification'][r].mean() for n in eps for k in K])) for r,m in enumerate(NAMES)}
    ceil=[NAMES.index(m) for m in CEIL];byN={x:{k:summary([np.mean([own_peer(E[n,k][x],r) for r in ceil]) for n in eps]) for k in K} for x in C}
    write(OUT/'ceiling_analysis.json',{'per_reader':per,'step_identification_level':level,'ceiling_by_N':{x:{str(k):v for k,v in d.items()} for x,d in byN.items()},'episodes':len(eps)})
    figure,axes=plt.subplots(1,2,figsize=(14,4.8),gridspec_kw={'width_ratios':[1.25,1]})
    order=CEIL+[m for m in NAMES if m not in CEIL];y=np.arange(len(order))
    for j,x in enumerate(C):
        v=np.array([100*per[m][x]['mean'] for m in order]);ci=np.array([[100*c for c in per[m][x]['ci']] for m in order])
        axes[0].barh(y+(j-.5)*.38,v,.36,color=C[x],xerr=[v-ci[:,0],ci[:,1]-v],capsize=2,error_kw={'lw':1,'ecolor':'#52514e'},label=x,zorder=2)
    axes[0].set_yticks(y);axes[0].set_yticklabels([f"{SHORT[m]}\nrecognises {100*level[m]:.0f}% of steps" for m in order]);axes[0].invert_yaxis();axes[0].axvline(0,color='#52514e',lw=.8)
    axes[0].axhline(len(CEIL)-.5,color='#9a9893',lw=.8,ls='--');axes[0].set_xlabel('own − same-language peer (pp, raw, mean over N = 1, 2, 4)');axes[0].set_title('A · Per reader',loc='left');axes[0].legend(frameon=False,loc='lower left');axes[0].grid(axis='x',color='#e5e4df',zorder=0)
    for j,x in enumerate(C):
        v=np.array([100*byN[x][k]['mean'] for k in K]);ci=np.array([[100*c for c in byN[x][k]['ci']] for k in K]);xx=np.arange(len(K))+(j-.5)*.36
        axes[1].bar(xx,v,.34,color=C[x],yerr=[v-ci[:,0],ci[:,1]-v],capsize=3,error_kw={'lw':1,'ecolor':'#52514e'},label=x,zorder=2)
    axes[1].axhline(0,color='#52514e',lw=.8);axes[1].set_xticks(range(len(K)));axes[1].set_xticklabels([f'N = {k}' for k in K]);axes[1].set_ylabel('own − same-language peer (pp)')
    axes[1].set_title(f'B · The four readers who recognise every step ({len(eps)} episodes)',loc='left');axes[1].legend(frameon=False);axes[1].grid(axis='y',color='#e5e4df',zorder=0)
    for a in axes: a.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(REPORTS/'proc-recognition.png',dpi=130);print(REPORTS/'proc-recognition.png')


if __name__=='__main__': main()
