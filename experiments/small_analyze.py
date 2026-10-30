"""Small-students statistics (SMALL_PROTOCOL.md): in-context and fine-tuning arms, own vs peer and large sources."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from mlayout_search import read
from small_search import CEPH,DIRECTORY,LARGE,SOURCES,STUDENTS,correct,icl_path

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen3_1_7b':'Qwen3-1.7B','llama32_3b':'Llama-3.2-3B','smollm3_3b':'SmolLM3-3B','phi4_mini':'Phi-4-mini','gemma4_26b':'Gemma 4 26B','qwen3_30b':'Qwen3-30B-A3B'}


def boot(d,seed=0):
    d=np.asarray(d,float);idx=np.random.default_rng(seed).integers(0,len(d),(20000,len(d)));return [float(d.mean()),*[float(x) for x in np.percentile(d[idx].mean(1),[2.5,97.5])]]


def contrasts(res,readers):
    out={'per_reader':{},'mean':{}}
    for kind,others in [('peers',lambda r:[s for s in STUDENTS if s!=r]),('large',lambda r:LARGE)]:
        ds=[res[r,r]-np.mean([res[r,s] for s in others(r)],0) for r in readers if all((r,s) in res for s in [r,*others(r)])]
        out['per_reader'][kind]={r:boot(res[r,r]-np.mean([res[r,s] for s in others(r)],0)) for r in readers if all((r,s) in res for s in [r,*others(r)])}
        out['mean'][kind]=boot(np.mean(ds,0)) if ds else None
    return out


def heat(axis,res,readers,title,base_key):
    data=np.array([[100*res[r,s].mean() if (r,s) in res else np.nan for s in SOURCES] for r in readers]);image=axis.imshow(data,cmap='viridis',vmin=np.nanmin(data)-5,vmax=np.nanmax(data)+2)
    for i,r in enumerate(readers):
        for j in range(len(SOURCES)):
            if not np.isnan(data[i,j]): axis.text(j,i,f'{data[i,j]:.0f}',ha='center',va='center',color='white' if data[i,j]<np.nanmean(data) else 'black')
        axis.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='r',lw=2))
    axis.set_xticks(range(len(SOURCES)));axis.set_xticklabels([SHORT[s] for s in SOURCES],rotation=35,ha='right');axis.set_yticks(range(len(readers)))
    axis.set_yticklabels([f"{SHORT[r]}\n({base_key} {100*res[r,'base'].mean():.0f})" for r in readers]);axis.axvline(3.5,color='white',lw=2);axis.set_title(title);axis.set_xlabel('source of the solutions');plt.colorbar(image,ax=axis,fraction=.046)


def main():
    out={}
    design=read(DIRECTORY/'design.json');tests=design['test'][:design['icl_test']];icl={}
    for r in STUDENTS:
        for s in ['initial']+SOURCES:
            if all(icl_path(r,s,i).exists() for i in range(len(tests))): icl[r,'base' if s=='initial' else s]=np.array([correct(read(icl_path(r,s,i))['text'],tests[i]['answer']) for i in range(len(tests))],float)
    out['icl']={'accuracy':{f'{r}|{s}':float(v.mean()) for (r,s),v in icl.items()},**contrasts(icl,STUDENTS)}
    ft={}
    for st in STUDENTS:
        base=CEPH/'ft'/st/'eval_base.json'
        if base.exists(): ft[st,'base']=np.array([correct(x['text'],x['answer']) for x in json.loads(base.read_text())['gsm_test_cot']],float)
        for src in SOURCES:
            p=CEPH/'ft'/st/f'from_{src}'/'eval.json'
            if p.exists(): ft[st,src]=np.array([correct(x['text'],x['answer']) for x in json.loads(p.read_text())['gsm_test_cot']],float)
    ft1={}
    for st in STUDENTS:
        if (st,'base') in ft: ft1[st,'base']=ft[st,'base']
        for src in SOURCES:
            p=CEPH/'ft'/st/f'from_{src}_seed1'/'eval.json'
            if p.exists(): ft1[st,src]=np.array([correct(x['text'],x['answer']) for x in json.loads(p.read_text())['gsm_test_cot']],float)
    if any(k[1]!='base' for k in ft1):
        pooled={k:(ft[k]+ft1[k])/2 for k in ft1 if k in ft}
        c0,c1,cp=contrasts(ft,STUDENTS),contrasts(ft1,STUDENTS),contrasts(pooled,STUDENTS)
        out['finetune_seed1']={'accuracy':{f'{r}|{s}':float(v.mean()) for (r,s),v in ft1.items()},**c1}
        out['finetune_pooled']={'accuracy':{f'{r}|{s}':float(v.mean()) for (r,s),v in pooled.items()},**cp,
                                'seed_change_own_minus_peers':{r:float(c1['per_reader']['peers'][r][0]-c0['per_reader']['peers'][r][0]) for r in c1['per_reader']['peers'] if r in c0['per_reader']['peers']}}
    loss={f'{st}|{src}':json.loads((CEPH/'ft'/st/f'from_{src}'/'loss.json').read_text())[0]['loss'] for st in STUDENTS for src in SOURCES if (CEPH/'ft'/st/f'from_{src}'/'loss.json').exists()}
    out['finetune']={'accuracy':{f'{r}|{s}':float(v.mean()) for (r,s),v in ft.items()},'initial_loss':loss,**contrasts(ft,STUDENTS)}
    (DIRECTORY/'analysis.json').write_text(json.dumps(out,indent=1))
    arms=[('icl',icl),('finetune',ft)]+([('finetune_seed1',ft1),('finetune_pooled',{k:(ft[k]+ft1[k])/2 for k in ft1 if k in ft})] if 'finetune_seed1' in out else [])
    for arm,res in arms:
        print(f'\n== {arm}: rows student, columns source (%)');print('%-14s %5s '%('','base')+' '.join('%8s'%SHORT[s][:8] for s in SOURCES))
        for r in STUDENTS:
            if (r,'base') in res: print('%-14s %5.1f '%(SHORT[r],100*res[r,'base'].mean())+' '.join(('[%5.1f] ' if s==r else ' %5.1f  ')%(100*res[r,s].mean()) if (r,s) in res else '    -    ' for s in SOURCES))
        c=out[arm]
        if 'seed_change_own_minus_peers' in c: print('  seed-to-seed change in own - peers (pp):',{SHORT[r]:round(100*v,1) for r,v in c['seed_change_own_minus_peers'].items()})
        for kind in ['peers','large']:
            if c['mean'][kind]: print(f"  own - {kind}: mean {100*c['mean'][kind][0]:+.1f} pp [{100*c['mean'][kind][1]:+.1f}, {100*c['mean'][kind][2]:+.1f}] | per reader",{SHORT[r]:round(100*v[0],1) for r,v in c['per_reader'][kind].items()})
    figure,axes=plt.subplots(1,2,figsize=(15,5))
    if icl: heat(axes[0],icl,[r for r in STUDENTS if (r,'base') in icl],'In context: GSM8K accuracy (%) with 4 worked examples','no examples')
    if ft: heat(axes[1],ft,[r for r in STUDENTS if (r,'base') in ft],'Fine-tuned (LoRA, 1,718 solutions): GSM8K accuracy (%)','base')
    figure.tight_layout();figure.savefig(REPORTS/'small-students.png',dpi=140)


if __name__=='__main__': main()
