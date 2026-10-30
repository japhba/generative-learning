"""Figure for the edge-of-capability study (main2 / label2): a worked example, own − peers by practice type, the reader × source matrix, and practice gains."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from edge_analyze import COLORS,DIRECTORY,M,NAMES,SHORT,TESTS,bootstrap,correct,eval_path,gen_path,load,read
from mlayout_search import final_answer

REPORTS=Path(__file__).resolve().parents[1]/'reports'
TRIO=[NAMES.index(m) for m in ['qwen3_1_7b','smollm3_3b','phi4_mini']]


def S(A,k):
    A=A[np.ix_(k,k)];n=len(k);return float(np.trace(A)/n-(A.sum()-np.trace(A))/(n*(n-1)))


def excerpt(text,lines=3):
    rows=[l for l in text.strip().splitlines() if l.strip()];return ['      '+l[:105] for l in rows[-lines:]]


def example_panel(axis):
    """First main2 episode where Phi-4-mini answers a test right after SmolLM3's mixed attempts but wrong after its own."""
    d=DIRECTORY/'main2';matched=read(d/'matched.json');r,p='phi4_mini','smollm3_3b'
    for n in sorted(matched):
        if not matched[n]['mixed']: continue
        w=read(d/'worlds'/f'{n}.json');hits=[i for i in range(TESTS) if correct(read(eval_path(d,r,n,f'mixed:{p}',i))['text'],w['tests'][i]) and not correct(read(eval_path(d,r,n,f'mixed:{r}',i))['text'],w['tests'][i])]
        if hits: i=hits[0];break
    slots=matched[n]['mixed'];wrong=[x for x in slots if not x['correct']][0];test=w['tests'][i]
    own=final_answer(read(eval_path(d,r,n,f'mixed:{r}',i))['text']);peer=final_answer(read(eval_path(d,r,n,f'mixed:{p}',i))['text']);none=final_answer(read(eval_path(d,r,n,'initial',i))['text'])
    big,mid,small=w['rule']['units'];r1,r2=w['rule']['ratios']
    lines=[f'Hidden rule (never stated): 1 {big} = {r1} {mid}s, 1 {mid} = {r2} {small}s.   Shown: 4 solved examples, e.g. “{w["support"][0]["question"]}” → {w["support"][0]["answer"]}',
           'Practice: 4 earlier attempts by ONE model (unchecked), always 2 right + 2 wrong, same problems and same right/wrong pattern for every source:',
           '   '+'   '.join(f"{w['pool'][x['index']]['question']} {'✓' if x['correct'] else '✗'}" for x in slots),
           f"e.g. the ✗ slot “{w['pool'][wrong['index']]['question']}” (true {w['pool'][wrong['index']]['answer']}), end of Phi-4-mini's own attempt:"]+excerpt(read(gen_path(d,r,n,wrong['index']))['samples'][wrong['sample'][r]]['text'])+[
           f"   …and of SmolLM3's attempt at the same problem:"]+excerpt(read(gen_path(d,p,n,wrong['index']))['samples'][wrong['sample'][p]]['text'])+[
           f"Test (reason step by step): {test['question']}  true = {test['answer']}",
           f"   Phi-4-mini — no practice: {none}    after its own attempts: {own}    after SmolLM3's attempts: {peer}"]
    axis.axis('off');axis.set_title(f'Example episode ({n}, alien units)',loc='left');axis.text(0,1,'\n'.join(lines),va='top',ha='left',family='monospace',transform=axis.transAxes)


def main():
    _,matched,E=load('main2');mixed=[n for n in E if 'mixed' in E[n]];allc=[n for n in E if 'correct' in E[n]]
    L={n:np.array([[float(np.mean([correct(read(eval_path(DIRECTORY/'label2',r,n,f'label:{x}',i))['text'],read(DIRECTORY/'main2'/'worlds'/f'{n}.json')['tests'][i]) for i in range(TESTS)])) for x in NAMES] for r in NAMES]) for n in mixed}
    arms=[('4 right',[E[n]['correct'] for n in allc]),('2 right + 2 wrong',[E[n]['mixed'] for n in mixed]),('2 + 2, wrong ones\nmarked “wrong”',[L[n] for n in mixed])]
    figure=plt.figure(figsize=(16,12));grid=figure.add_gridspec(2,2,height_ratios=[.42,1]);example_panel(figure.add_subplot(grid[0,:]))
    axis=figure.add_subplot(grid[1,0]);width=.38;summary={}
    for k,(group,idx) in enumerate([('all four models',list(range(M))),('three similar models\n(without Llama-3.2-3B)',TRIO)]):
        vals=[[100*S(A,idx) for A in mats] for _,mats in arms];means=[np.mean(v) for v in vals];cis=np.array([bootstrap(v) for v in vals]).T;x=np.arange(len(arms))+(k-.5)*width
        axis.bar(x,means,width*.92,color=COLORS[k],label=group,yerr=[np.array(means)-cis[0],cis[1]-np.array(means)],capsize=3,error_kw={'lw':1,'ecolor':'#52514e'},zorder=2)
        for xi,m,lo,hi in zip(x,means,*cis): axis.text(xi,(lo if m<0 else hi)+(-.25 if m<0 else .25),f'{m:+.1f}',ha='center',va='top' if m<0 else 'bottom',color='#52514e')
        floor=min(cis[0].min(),floor) if k else cis[0].min()
        summary[group]={a:[float(m),*map(float,c)] for (a,_),m,c in zip(arms,means,cis.T)}
    axis.set_ylim(floor-1.2,axis.get_ylim()[1]);axis.axhline(0,color='#52514e',lw=.8);axis.set_xticks(range(len(arms)));axis.set_xticklabels([a for a,_ in arms]);axis.set_ylabel('own − peers’ practice (pp test accuracy)')
    axis.set_title(f'A · Own vs peers’ attempts, matched in number correct\n(S = mean diagonal − mean off-diagonal; {len(mixed)} / {len(allc)} episodes, 95% CI)',loc='left');axis.legend(frameon=False,loc='lower left');axis.grid(axis='y',color='#e5e4df',zorder=0);axis.spines[['top','right']].set_visible(False)
    axis=figure.add_subplot(grid[1,1]);A=100*np.mean([E[n]['mixed'] for n in mixed],0);image=axis.imshow(A,cmap='viridis',vmin=0,vmax=100)
    for i in range(M):
        for j in range(M): axis.text(j,i,f'{A[i,j]:.0f}',ha='center',va='center',color='white' if A[i,j]<60 else 'black')
        axis.add_patch(plt.Rectangle((i-.5,i-.5),1,1,fill=False,edgecolor='k',lw=2))
    base=100*np.mean([E[n]['initial'] for n in mixed],0);labels=[SHORT[m] for m in NAMES]
    axis.set_xticks(range(M));axis.set_xticklabels(labels,rotation=20,ha='right');axis.set_yticks(range(M));axis.set_yticklabels([f'{l}\n(no practice {b:.0f}%)' for l,b in zip(labels,base)])
    axis.set_xlabel('whose attempts were shown (2 right + 2 wrong)');axis.set_ylabel('reader');axis.set_title('B · Test accuracy (%) by reader × source',loc='left');figure.colorbar(image,ax=axis,fraction=.046)
    figure.tight_layout();figure.savefig(REPORTS/'edge-of-capability.png',dpi=130);(REPORTS/'edge-of-capability.json').write_text(json.dumps({'S':summary,'matrix_mixed':A.tolist(),'initial':base.tolist()},indent=1))
    print(json.dumps(summary,indent=1))


if __name__=='__main__': main()
