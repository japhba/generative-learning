"""Frozen statistics for EDGE2 (EDGE2_PROTOCOL.md, amendment 1): 3 × 3 reader × source matrix on complete episodes."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_analyze import bootstrap,sign_flip,sign_flip_two
from edge_search import strict_correct
from edge2_search import DIRECTORY,NAMES,conditions,correct,eval_path,gen_path,read,write

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'deepseek_flash':'DeepSeek-V4.1-Flash','gemma4_31b':'Gemma-4-31B','llama4_maverick':'Llama-4-Maverick'}
M=len(NAMES)


def summary(v): v=np.asarray(v);return {'mean':float(v.mean()),'ci':bootstrap(v),'p_two_sided':sign_flip_two(v,100000),'p_one_sided':sign_flip(v,100000),'n':len(v)}
def fmt(s): return '%+.1f pp [%+.1f, %+.1f] p2=%.4f p1=%.4f (n=%d)'%(100*s['mean'],100*s['ci'][0],100*s['ci'][1],s['p_two_sided'],s['p_one_sided'],s['n'])
def S(A): return float(np.trace(A)/M-(A.sum()-np.trace(A))/(M*(M-1)))


def main():
    E={};strict={};status={};complete=[];phase_of={};prepared=0;with_mixed=0;matched_all={}
    for phase in ['main','main_hard']:  # amendment 2: pooled primary, each phase also separately
        d=DIRECTORY/phase
        if not (d/'matched.json').exists(): continue
        design=read(d/'design.json');matched=read(d/'matched.json');prepared+=len(design['worlds']);with_mixed+=sum(bool(v['mixed']) for v in matched.values())
        for n in design['worlds']:
            if not (matched[n]['mixed'] and all(eval_path(d,r,n,c,i).exists() for r in NAMES for c in conditions(matched[n]) for i in range(8))): continue
            key=f'{phase}/{n}';complete.append(key);phase_of[key]=phase;matched_all[key]=(d,n,matched[n])
            w=read(d/'worlds'/f'{n}.json');ev=lambda r,c,i:read(eval_path(d,r,n,c,i))
            E[key]={c:np.array([np.mean([correct(ev(r,c,i)['text'],w['tests'][i]) for i in range(8)]) for r in NAMES]) for c in conditions(matched[n])}
            strict[key]={c:np.array([np.mean([strict_correct(ev(r,c,i)['text'],w['tests'][i]) for i in range(8)]) for r in NAMES]) for c in conditions(matched[n])}
            status[key]={c:np.array([np.mean([ev(r,c,i)['status']!='ok' for i in range(8)]) for r in NAMES]) for c in conditions(matched[n])}
    A={n:np.column_stack([E[n][f'mixed:{s}'] for s in NAMES]) for n in complete}  # rows reader, columns source
    out={'episodes_prepared':prepared,'episodes_with_mixed_set':with_mixed,'episodes_complete':len(complete),'by_phase':{ph:summary([S(A[n]) for n in complete if phase_of[n]==ph]) for ph in sorted(set(phase_of.values())) if sum(phase_of[n]==ph for n in complete)>1}}
    out['S_mixed']=summary([S(A[n]) for n in complete])
    out['own_effect']={m:summary([A[n][r,r]-np.mean([A[n][r,s] for s in range(M) if s!=r]) for n in complete]) for r,m in enumerate(NAMES)}
    out['pairwise_H']={f'{NAMES[i]}|{NAMES[j]}':summary([((A[n][i,i]-A[n][i,j])+(A[n][j,j]-A[n][j,i]))/2 for n in complete]) for i in range(M) for j in range(i+1,M)}
    out['practice_effect']=summary([float((A[n]-E[n]['initial'][:,None]).mean()) for n in complete])
    out['matrix']=np.mean([A[n] for n in complete],0).tolist();out['initial']=np.mean([E[n]['initial'] for n in complete],0).tolist()
    out['matrix_strict']=np.mean([np.column_stack([strict[n][f'mixed:{s}'] for s in NAMES]) for n in complete],0).tolist()
    out['matrix_truncated']=np.mean([np.column_stack([status[n][f'mixed:{s}'] for s in NAMES]) for n in complete],0).tolist()
    att={m:[] for m in NAMES};unfinished={m:[] for m in NAMES}
    for phase in ['main','main_hard']:
        d=DIRECTORY/phase
        if not (d/'matched.json').exists(): continue
        for n in read(d/'design.json')['worlds']:
            w=read(d/'worlds'/f'{n}.json')
            for m in NAMES:
                for i in range(len(w['pool'])):
                    for k in range(3): g=read(gen_path(d,m,n,i,k));att[m].append(correct(g['text'],w['pool'][i]));unfinished[m].append(g['status']!='ok')
    out['attempt_accuracy']={m:float(np.mean(v)) for m,v in att.items()};out['attempt_unfinished']={m:float(np.mean(v)) for m,v in unfinished.items()}
    slot_unfinished={m:float(np.mean([read(gen_path(matched_all[k][0],m,matched_all[k][1],x['index'],x['sample'][m]))['status']!='ok' for k in complete for x in matched_all[k][2]['mixed'] if not x['correct']])) for m in NAMES}
    out['unsuccessful_slots_unfinished']=slot_unfinished
    write(DIRECTORY/'analysis.json',out)
    print('episodes: prepared',out['episodes_prepared'],'with mixed set',out['episodes_with_mixed_set'],'tested',out['episodes_complete'])
    print('attempt accuracy',{SHORT[m]:round(100*v) for m,v in out['attempt_accuracy'].items()},'unfinished',{SHORT[m]:round(100*v) for m,v in out['attempt_unfinished'].items()},'| unsuccessful slots unfinished',{SHORT[m]:round(100*v) for m,v in slot_unfinished.items()})
    print('PRIMARY S(mixed), pooled',fmt(out['S_mixed']));[print('  phase',ph,fmt(v)) for ph,v in out['by_phase'].items()]
    for m,v in out['own_effect'].items(): print('  own effect',SHORT[m],fmt(v))
    for k,v in out['pairwise_H'].items(): print('  H',k,fmt(v))
    print('practice effect',fmt(out['practice_effect']))
    for name,key in [('numeric','matrix'),('strict format','matrix_strict'),('truncated','matrix_truncated')]:
        print(f'matrix ({name}, %; rows reader, cols source) | no practice');[print('%-20s'%SHORT[m],' '.join(('[%3.0f]' if j==i else ' %3.0f ')%(100*v) for j,v in enumerate(row)),'| %3.0f'%(100*out['initial'][i])) for i,(m,row) in enumerate(zip(NAMES,out[key]))]
    plot(out,A,complete)


def plot(out,A,complete):
    figure,axes=plt.subplots(1,2,figsize=(13,5));labels=[SHORT[m] for m in NAMES];B=100*np.array(out['matrix'])
    image=axes[0].imshow(B,cmap='viridis',vmin=0,vmax=100)
    for i in range(M):
        for j in range(M): axes[0].text(j,i,f'{B[i,j]:.0f}',ha='center',va='center',color='white' if B[i,j]<60 else 'black')
        axes[0].add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='k',lw=2))
    axes[0].set_xticks(range(M));axes[0].set_xticklabels(labels,rotation=15,ha='right');axes[0].set_yticks(range(M));axes[0].set_yticklabels([f'{l}\n(no practice {100*b:.0f}%)' for l,b in zip(labels,out['initial'])])
    axes[0].set_xlabel('whose attempts were shown (2 successful + 2 unsuccessful)');axes[0].set_ylabel('reader');axes[0].set_title(f'Test accuracy (%), operator task, {len(complete)} episodes',loc='left');figure.colorbar(image,ax=axes[0],fraction=.046)
    items=[('S (all pairs)',out['S_mixed'])]+[(' vs '.join(SHORT[x].split('-')[0] for x in k.split('|')),v) for k,v in out['pairwise_H'].items()]
    for k,(name,v) in enumerate(items): axes[1].errorbar(100*v['mean'],k,xerr=[[100*(v['mean']-v['ci'][0])],[100*(v['ci'][1]-v['mean'])]],fmt='o',color='#2a78d6' if k else '#1d1d1b',capsize=3)
    axes[1].axvline(0,color='#52514e',lw=.8);axes[1].set_yticks(range(len(items)));axes[1].set_yticklabels([i[0] for i in items]);axes[1].invert_yaxis()
    axes[1].set_xlabel('own − peers’ attempts (pp test accuracy, 95% CI)');axes[1].set_title('Own vs peers, matched in number successful',loc='left');axes[1].spines[['top','right']].set_visible(False)
    figure.tight_layout();figure.savefig(REPORTS/'edge2-flash.png',dpi=130)


if __name__=='__main__': main()
