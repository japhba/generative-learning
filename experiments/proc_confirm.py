"""Frozen statistics for the PROC confirmatory phase (PROC_PROTOCOL.md)."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from proc_analyze import M,NAMES,REL,SHORT,boot,dc,summary,fmt
from proc_search import COUNTS,DIRECTORY,TESTS,chosen,correct,eval_path,gen_path,probe_path,read,write

REPORTS=Path(__file__).resolve().parents[1]/'reports'
PAIRS=[('olmo3_7b','phi4_mini'),('gswallow_9b','abeja_qwen7b'),('glm4_9b','internlm3_8b')]


def contrast(A,a,b): D=dc(A);return float(D[REL==a].mean()-D[REL==b].mean())
def holm(p): p=np.asarray(p);o=np.argsort(p);adj=np.maximum.accumulate([(len(p)-k)*p[i] for k,i in enumerate(o)]);out=np.empty(len(p));out[o]=np.minimum(adj,1);return out
def grams(text): t=''.join(text.split());return {t[i:i+4] for i in range(len(t)-3)}


def main():
    done=read(DIRECTORY/'complete.json');eps=done['usable'];worlds={n:read(DIRECTORY/'worlds'/f'{n}.json') for n in eps};matched=read(DIRECTORY/'matched.json')
    dep=lambda r,n,c:float(np.mean([correct(read(eval_path(r,n,c,i))['text'],worlds[n]['tests'][i]) for i in range(TESTS)]))
    rec=lambda r,n,c:float(np.mean([chosen(read(probe_path(r,n,c,j))['text'])==worlds[n]['probes'][j]['answer'] for j in range(worlds[n]['length'])]))
    full=lambda r,n,c:all(chosen(read(probe_path(r,n,c,j))['text'])==worlds[n]['probes'][j]['answer'] for j in range(worlds[n]['length']))
    Dp={(n,k):np.array([[dep(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps for k in COUNTS}
    Rc={(n,k):np.array([[rec(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]) for n in eps for k in COUNTS}
    out={'usable':len(eps),'prepared':len(read(DIRECTORY/'design.json')['worlds'])}
    self_dep=[np.mean([contrast(Dp[n,k],'own','same language') for k in COUNTS]) for n in eps];self_rec=[np.mean([contrast(Rc[n,k],'own','same language') for k in COUNTS]) for n in eps]
    out['P1_self_deployment']=summary(self_dep);out['P2_dissociation']=summary(np.subtract(self_dep,self_rec));out['self_recognition']=summary(self_rec)
    out['holm']=holm([out['P1_self_deployment']['p_two_sided'],out['P2_dissociation']['p_two_sided']]).tolist()
    for k in COUNTS:
        for name,X in [('deployment',Dp),('recognition',Rc)]:
            out[f'self_{name}_N{k}']=summary([contrast(X[n,k],'own','same language') for n in eps]);out[f'language_{name}_N{k}']=summary([contrast(X[n,k],'same language','other language') for n in eps])
            for L in [3,4]: out[f'self_{name}_N{k}_L{L}']=summary([contrast(X[n,k],'own','same language') for n in eps if worlds[n]['length']==L])
        out[f'self_deployment_per_reader_N{k}']={m:float(np.mean([dc(Dp[n,k])[r,r]-dc(Dp[n,k])[r,[j for j in range(M) if REL[r,j]=='same language'][0]] for n in eps])) for r,m in enumerate(NAMES)}
        out[f'pair_H_deployment_N{k}']={f'{a}|{b}':summary([((Dp[n,k][NAMES.index(a),NAMES.index(a)]-Dp[n,k][NAMES.index(a),NAMES.index(b)])+(Dp[n,k][NAMES.index(b),NAMES.index(b)]-Dp[n,k][NAMES.index(b),NAMES.index(a)]))/2 for n in eps]) for a,b in PAIRS}
        rows=[]
        for n in eps:  # deployment among cells where every step was recognised: own minus same-language peer (raw accuracy)
            F=np.array([[full(r,n,f'src:{s}:{k}') for s in NAMES] for r in NAMES]);own=np.diag(F);peer=F[REL=='same language']
            a=[Dp[n,k][r,r] for r in range(M) if F[r,r]];b=[Dp[n,k][r,j] for r in range(M) for j in range(M) if REL[r,j]=='same language' and F[r,j]]
            if a and b: rows.append(np.mean(a)-np.mean(b))
        out[f'self_deployment_given_full_recognition_N{k}']=summary(rows)
        out[f'levels_N{k}']={x:{'deployment':float(np.mean([Dp[n,k][REL==x].mean() for n in eps])),'recognition':float(np.mean([Rc[n,k][REL==x].mean() for n in eps]))} for x in ['own','same language','other language']}
        sim={'own':[],'same language':[]}
        for n in eps:  # exploratory: 4-gram overlap of the reader's test solutions with the shown demonstrations
            for r in range(M):
                for s in range(M):
                    if REL[r,s] not in sim: continue
                    demo=set().union(*[grams(read(gen_path(NAMES[s],n,q['index']))['samples'][q['sample'][NAMES[s]]]['text']) for q in matched[n][:k]])
                    sim[REL[r,s]].append(np.mean([len(grams(read(eval_path(NAMES[r],n,f'src:{NAMES[s]}:{k}',i))['text'])&demo)/max(1,len(grams(read(eval_path(NAMES[r],n,f'src:{NAMES[s]}:{k}',i))['text']))) for i in range(TESTS)]))
        out[f'imitation_N{k}']={x:float(np.mean(v)) for x,v in sim.items()}
    out['references']={c:{m:float(np.mean([dep(m,n,c) for n in eps])) for m in NAMES} for c in ['answers','stated']}
    write(DIRECTORY/'analysis.json',out)
    print('usable',out['usable'],'of',out['prepared'])
    print('PRIMARY 1 self (own - same-language peer), deployment, mean over N:',fmt(out['P1_self_deployment']),'Holm p=%.4f'%out['holm'][0])
    print('PRIMARY 2 dissociation (self deployment - self recognition):',fmt(out['P2_dissociation']),'Holm p=%.4f'%out['holm'][1],'| self recognition',fmt(out['self_recognition']))
    for k in COUNTS:
        print(f'N={k}: self deployment {fmt(out[f"self_deployment_N{k}"])} | self recognition {fmt(out[f"self_recognition_N{k}"])}')
        print(f'      language deployment {fmt(out[f"language_deployment_N{k}"])} | language recognition {fmt(out[f"language_recognition_N{k}"])}')
        print(f'      self deployment by L: L3 {fmt(out[f"self_deployment_N{k}_L3"])} | L4 {fmt(out[f"self_deployment_N{k}_L4"])}')
        print(f'      self deployment | every step recognised {fmt(out[f"self_deployment_given_full_recognition_N{k}"])}')
        print(f'      per reader',{SHORT[m][:12]:'%+.0f'%(100*v) for m,v in out[f'self_deployment_per_reader_N{k}'].items()},'| pairs',{p.split('|')[0][:6]:'%+.1f'%(100*v['mean']) for p,v in out[f'pair_H_deployment_N{k}'].items()})
        print(f'      levels',{x:{a:round(100*b) for a,b in v.items()} for x,v in out[f'levels_N{k}'].items()},'| 4-gram imitation own %.3f peer %.3f'%(out[f'imitation_N{k}']['own'],out[f'imitation_N{k}']['same language']))
    print('references',{c:{SHORT[m][:10]:round(100*v) for m,v in d.items()} for c,d in out['references'].items()})
    plot(out)


def plot(out):
    figure,axes=plt.subplots(1,2,figsize=(14,5));colors={'deployment':'#2a78d6','recognition':'#eda100'}
    for axis,key,title in [(axes[0],'self','A · Own − same-language peer'),(axes[1],'language','B · Same-language peer − other language')]:
        for j,name in enumerate(['recognition','deployment']):
            m=np.array([100*out[f'{key}_{name}_N{k}']['mean'] for k in COUNTS]);ci=np.array([[100*x for x in out[f'{key}_{name}_N{k}']['ci']] for k in COUNTS]);x=np.arange(len(COUNTS))+(j-.5)*.36
            axis.bar(x,m,.34,color=colors[name],label='recognition (which step was shown?)' if name=='recognition' else 'deployment (apply to a new number)',yerr=[m-ci[:,0],ci[:,1]-m],capsize=3,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
        axis.axhline(0,color='#52514e',lw=.8);axis.set_xticks(range(len(COUNTS)));axis.set_xticklabels([f'N = {k}' for k in COUNTS]);axis.set_ylabel('interaction (pp, reader and source effects removed)')
        axis.set_title(title,loc='left');axis.legend(frameon=False);axis.grid(axis='y',color='#e5e4df',zorder=0);axis.spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(REPORTS/'proc-confirmatory.png',dpi=130)


if __name__=='__main__': main()
