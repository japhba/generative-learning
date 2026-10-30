"""Single-panel summary of PROC + PROC-REC: per reader, own − same-language-peer advantage in recognising the steps (x)
vs in deploying the procedure (y), mean over N = 1, 2, 4, 73 episodes; plus the four readers who recognise every step, pooled."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from proc_analyze import M,NAMES,REL,SHORT,summary
from proc_rec_analyze import K,load
from proc_search import LANGUAGE

REPORTS=Path(__file__).resolve().parents[1]/'reports';CEIL=['olmo3_7b','phi4_mini','glm4_9b','internlm3_8b']
COLOR={'en':'#2a78d6','ja':'#eb6834','zh':'#1baf7a'};LANG={'en':'English','ja':'Japanese','zh':'Chinese'};MEASURES=['step identification','deployment']


def own_peer(A,r): return A[r,r]-A[r,[j for j in range(M) if REL[r,j]=='same language'][0]]


def main():
    E={};eps=[]
    for s in ['proc_main','proc_main2']: e,ep,_=load(s);E|=e;eps+=ep
    per={m:{x:summary([np.mean([own_peer(E[n,k][x],r) for k in K]) for n in eps]) for x in MEASURES} for r,m in enumerate(NAMES)}
    ceil=[NAMES.index(m) for m in CEIL];pooled={x:summary([np.mean([own_peer(E[n,k][x],r) for r in ceil for k in K]) for n in eps]) for x in MEASURES}
    level={m:np.mean([E[n,k]['step identification'][r].mean() for n in eps for k in K]) for r,m in enumerate(NAMES)}
    figure,axis=plt.subplots(figsize=(9,6))
    axis.axhline(0,color='#9a9893',lw=.8,zorder=0);axis.axvline(0,color='#9a9893',lw=.8,zorder=0)
    lim=40
    offsets={'olmo3_7b':(8,0),'glm4_9b':(8,0),'phi4_mini':(8,-4),'internlm3_8b':(8,0),'gswallow_9b':(0,14),'abeja_qwen7b':(0,-16)}
    for m in NAMES:
        x,y=(100*per[m][q]['mean'] for q in MEASURES);xe,ye=(np.abs(100*np.array(per[m][q]['ci'])-100*per[m][q]['mean'])[:,None] for q in MEASURES)
        axis.errorbar(x,y,xerr=xe,yerr=ye,fmt='o',color=COLOR[LANGUAGE[m]],ecolor=COLOR[LANGUAGE[m]],elinewidth=1,capsize=2,zorder=3)
        dx,dy=offsets[m];axis.annotate(f"{SHORT[m].split(' (')[0]}" + (f" ({100*level[m]:.0f}% of steps recognised)" if dx else f"\n{100*level[m]:.0f}% of steps recognised"),(x,y),xytext=(dx,dy),textcoords='offset points',ha='left' if dx else 'center',va='center' if dx else ('bottom' if dy>0 else 'top'),color='#3d3d3a')
    x,y=(100*pooled[q]['mean'] for q in MEASURES);ye=np.abs(100*np.array(pooled['deployment']['ci'])-y)[:,None]
    axis.errorbar(x,y,yerr=ye,fmt='D',color='#1d1d1b',elinewidth=1.2,capsize=3,zorder=4)
    axis.annotate(f'4 readers who recognise\nevery step, pooled:\ndeployment {y:+.1f} pp\n[{100*pooled["deployment"]["ci"][0]:+.1f}, {100*pooled["deployment"]["ci"][1]:+.1f}]\nrecognition {x:+.1f} pp',(x,y),xytext=(-10,0),textcoords='offset points',ha='right',va='center',color='#1d1d1b')
    for lang,c in COLOR.items(): axis.plot([],[],'o',color=c,label=f'{LANG[lang]}-dominant reader')
    axis.plot([],[],'D',color='#1d1d1b',label='ceiling readers pooled');axis.legend(frameon=False,loc='lower right')
    axis.set_xlim(-lim,lim);axis.set_ylim(-12,24)
    axis.set_xlabel('own − same-language peer in RECOGNISING the steps (pp)');axis.set_ylabel('own − same-language peer in DEPLOYING the procedure (pp)')
    axis.set_title(f'Recognised equally, deployed better from one’s own example\n{len(eps)} episodes · mean over 1, 2, 4 worked examples · 95% CI',loc='left')
    axis.spines[['top','right']].set_visible(False);figure.tight_layout();figure.savefig(REPORTS/'proc-single.png',dpi=140);print(REPORTS/'proc-single.png')


if __name__=='__main__': main()
