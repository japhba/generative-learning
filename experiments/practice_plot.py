"""Does practice help? Accuracy with no practice vs answers-only vs own / others' worked practice, plus fine-tuning."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mlayout2_search as M2
import mlayout3_search as M3
import mthink2_search as T2
import mthink_search as T1
from mlayout_search import direct_answer,read
from mlayout_tasks import correct

REPORTS=Path(__file__).resolve().parents[1]/'reports'
SHORT={'qwen35_9b':'Qwen3.5-9B','qwen3_30b':'Qwen3-30B-A3B','gemma4_26b':'Gemma 4 26B','nemotron3_30b':'Nemotron-3-Nano','qwen3_30b_hybrid':'Qwen3-30B-A3B\n(hybrid)'}
COLORS=['#2a78d6','#eb6834','#1baf7a','#eda100']  # validated categorical slots 1-4 (adjacent CVD/normal-vision pass)
LABELS=['no practice','correct answers only','own worked practice','others’ worked practice']


def boot(values,seed=0):
    values=np.asarray(values,float);idx=np.random.default_rng(seed).integers(0,len(values),(20000,len(values)));return np.percentile(values[idx].mean(1),[2.5,97.5])


def icl(modules,kind):
    """Per reader: per-episode accuracy for none / answers / own / mean of others."""
    out={}
    for mod in modules:
        matched=read(mod.DIRECTORY/'matched.json')
        for n,v in matched.items():
            if not v['included']: continue
            w=read(mod.DIRECTORY/'worlds'/f'{n}.json');acc=lambda r,c:np.mean([correct(direct_answer(read(mod.evaluation_path(r,n,c,i))['text']),w['tests'][i]) for i in range(mod.TESTS)])
            for r in mod.NAMES:
                d=out.setdefault(r,{k:[] for k in range(4)})
                d[0].append(acc(r,'initial'));d[1].append(acc(r,'answers'));d[2].append(acc(r,f'{kind}:{r}'));d[3].append(np.mean([acc(r,f'{kind}:{s}') for s in mod.NAMES if s!=r]))
    return out


def finetune():
    from msft_analyze import scores,ROOT,NAMES
    out={}
    for st in NAMES:
        base=scores(ROOT/st/'eval_base.json')['operator_cot'];own=scores(ROOT/st/f'from_{st}/eval.json')['operator_cot']
        others=np.mean([scores(ROOT/st/f'from_{o}/eval.json')['operator_cot'] for o in NAMES if o!=st],0);out[st]={0:list(base),2:list(own),3:list(others)}
    return out


def panel(axis,data,title,ylabel,keys):
    readers=list(data);width=.8/len(keys)
    for j,k in enumerate(keys):
        means=[100*np.mean(data[r][k]) for r in readers];cis=np.array([100*boot(data[r][k]) for r in readers]).T;x=np.arange(len(readers))+(j-(len(keys)-1)/2)*width
        axis.bar(x,means,width*.92,color=COLORS[k],label=LABELS[k] if k!=0 or 'base' not in ylabel else 'base model',yerr=[np.array(means)-cis[0],cis[1]-np.array(means)],capsize=2,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
        for xi,m,hi in zip(x,means,cis[1]): axis.text(xi,hi+1,f'{m:.0f}',ha='center',va='bottom',color='#52514e')
    axis.set_xticks(range(len(readers)));axis.set_xticklabels([SHORT[r] for r in readers]);axis.set_ylabel(ylabel);axis.set_title(title,loc='left')
    axis.set_ylim(0,105);axis.grid(axis='y',color='#e5e4df',zorder=0);axis.spines[['top','right']].set_visible(False)


def example_panel(axis):
    """One real item (mlayout3 operator__00, test 1, Gemma 4 reading) with the examples, a practice excerpt and the answers."""
    n,i,reader='operator__00',1,'gemma4_26b';w=read(M3.DIRECTORY/'worlds'/f'{n}.json');q=read(M3.DIRECTORY/'matched.json')[n]['questions'][0]
    practice=w['pool'][q['index']];solution=read(M3.generation_path(reader,n,q['index']))['samples'][q['sample'][reader]]['text'].strip().splitlines()
    solution=[l for l in solution if l.strip()];excerpt=solution[:3]+['   …']+solution[-3:]
    test=w['tests'][i];ans={c:direct_answer(read(M3.evaluation_path(reader,n,c,i))['text']) for c in ['initial','answers',f'full:{reader}','full:qwen35_9b']}
    mark=lambda a:'✓' if correct(a,test) else '✗'
    lines=[f"Hidden rule (never shown to the model): a {w['rule']['symbol']} b = {w['rule']['p']}a + {w['rule']['q']}b",
           'The model sees 4 solved examples:  '+'   '.join(f"{e['question'][8:-1]} → {e['answer']}" for e in w['support']),
           f"Practice: 6 more problems, e.g. “{practice['question']}”, shown with a correct worked solution (here Gemma's own, excerpt):"]+[f'      {l[:110]}' for l in excerpt]+[
           f"Test (answer directly, no working): {test['question']}  correct = {test['answer']}",
           f"   Gemma 4's direct answer — no practice: {ans['initial']} {mark(ans['initial'])}     with correct answers only: {ans['answers']} {mark(ans['answers'])}",
           f"                             after own worked practice: {ans[f'full:{reader}']} {mark(ans[f'full:{reader}'])}     after Qwen3.5's worked practice: {ans['full:qwen35_9b']} {mark(ans['full:qwen35_9b'])}"]
    axis.axis('off');axis.set_title('Example task: the “mystery operator”',loc='left')
    axis.text(0,1,'\n'.join(lines),va='top',ha='left',family='monospace',transform=axis.transAxes)


def main():
    worked=icl([M2,M3],'full');thinking=icl([T1,T2],'think');tuned=finetune()
    figure=plt.figure(figsize=(12,15.5));grid=figure.add_gridspec(4,1,height_ratios=[.62,1,1,1]);example_panel(figure.add_subplot(grid[0]));axes=[figure.add_subplot(grid[k]) for k in (1,2,3)]
    panel(axes[0],worked,'A · In-context practice with correct worked solutions (operator, 28 episodes; test: answer directly)','test accuracy (%)',[0,1,2,3])
    panel(axes[1],thinking,'B · In-context practice with correct thinking traces (operator, 32 episodes; test: answer directly)','test accuracy (%)',[0,1,2,3])
    panel(axes[2],{k:v for k,v in tuned.items()},'C · Practice in the weights: LoRA on 557 correct solutions (fresh operator problems, step-by-step)','test accuracy (%)',[0,2,3])
    handles,labels=axes[0].get_legend_handles_labels();axes[0].legend(handles,labels,loc='upper left',ncol=2,frameon=False)
    h2,_=axes[2].get_legend_handles_labels();axes[2].legend(h2,['base model (no fine-tuning)','fine-tuned on own solutions','fine-tuned on others’ solutions'],loc='upper left',ncol=3,frameon=False)
    figure.tight_layout();figure.savefig(REPORTS/'practice-helps.png',dpi=150)
    summary={name:{SHORT[r].replace('\n',' '):{LABELS[k]:float(np.mean(v[k])) for k in v} for r,v in d.items()} for name,d in [('worked',worked),('thinking',thinking),('finetune',tuned)]}
    (REPORTS/'practice-helps.json').write_text(json.dumps(summary,indent=1));print(json.dumps(summary,indent=1))


if __name__=='__main__': main()
