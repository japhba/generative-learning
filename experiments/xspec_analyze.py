"""Post-inference scoring, frozen selections and statistics for the cross-task specificity search (XSPEC_PROTOCOL.md)."""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from audit_task_search_constraints import answer_valid
from task_search_data import dataset
from task_search_scoring import score_answer
from xspec_search import DIRECTORY,MODELS,SOURCES,fits,length_eligible,read,write

REPORTS=Path(__file__).resolve().parents[1]/'reports'
CAP=80;SELECT=6


def correct(data,world,entry,record):
    if record['call']['status']!='ok': return False
    answer=record['answer'];task=world['spec']['task']
    return score_answer(data,task,answer,entry)>=1-1e-9 and answer_valid(task,answer,entry)


def score(phase):
    """Per-record correctness for every evaluation and generation of a completed phase; cached alongside the phase."""
    directory=DIRECTORY/phase;design=read(directory/'design.json');assert (directory/'complete.json').exists(),'Analyze only completed phases.'
    worlds={name:read(directory/'worlds'/f'{name}.json') for name in design['worlds']};rows=[];generations=[]
    for name,world in tqdm(worlds.items(),desc=f'Score {phase}',mininterval=10):
        files=sorted((directory/'evaluations').glob(f'*/{name}__*.json'))
        if not files: continue
        data=dataset(world)
        for path in files:
            record=read(path);entry=world['tests'][record['index']]
            rows.append({'setting':world['setting'],'episode':world['episode'],'reader':record['model'],'source':record['source'],'n':record['n'],'index':record['index'],
                         'correct':correct(data,world,entry,record),'status':record['call']['status'],'chars':len(record['answer'])})
        for path in sorted((directory/'generations').glob(f'*/{name}__*.json')):
            record=read(path);entry=world['practice'][record['index']]
            generations.append({'setting':world['setting'],'episode':world['episode'],'model':record['model'],'index':record['index'],
                                'correct':correct(data,world,entry,record),'status':record['call']['status'],'chars':len(record['answer'])})
    write(directory/'scores.json',{'evaluations':rows,'generations':generations});return rows,generations


def table(rows):
    """accuracy[setting][reader][source][n] -> per-episode accuracies (equal tests per episode)."""
    cells=defaultdict(list)
    for r in rows: cells[r['setting'],r['reader'],r['source'],r['n'],r['episode']].append(r['correct'])
    result=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
    for (setting,reader,source,n,episode),values in sorted(cells.items()): result[setting][reader][source].setdefault(n,{})[episode]=float(np.mean(values))
    return {s:{r:{src:{n:[v[e] for e in sorted(v)] for n,v in by_n.items()} for src,by_n in by_src.items()} for r,by_src in by_r.items()} for s,by_r in result.items()}


def preferences(cell,n):
    """Episode-level own-source preferences P_Q, P_G and H at count n."""
    q=np.array(cell['qwen']['qwen'][n])-np.array(cell['qwen']['gemma'][n]);g=np.array(cell['gemma']['gemma'][n])-np.array(cell['gemma']['qwen'][n])
    return q,g,(q+g)/2


def bootstrap(values,resamples=20000,seed=20261005):
    values=np.asarray(values);index=np.random.default_rng(seed).integers(0,len(values),(resamples,len(values)))
    return [float(x) for x in np.percentile(values[index].mean(1),[2.5,97.5])]


def sign_flip(values):
    values=np.asarray(values);signs=np.array(list(itertools.product([1,-1],repeat=len(values))))
    return float(np.mean((signs*values).mean(1)>=values.mean()-1e-12))


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.empty(len(pvalues));running=0.
    for rank,i in enumerate(order): running=max(running,min(1.,(len(pvalues)-rank)*pvalues[i]));adjusted[i]=running
    return [float(x) for x in adjusted]


def calibration():
    rows,_=score('calibration');design=read(DIRECTORY/'calibration/design.json')
    accuracy=defaultdict(dict)
    for r in rows: accuracy[r['setting']].setdefault(r['reader'],[]).append(r['correct'])
    summary={s:{m:float(np.mean(v[m])) for m in MODELS} for s,v in accuracy.items()}
    eligible={}
    for setting in design['settings']:
        s=setting['id']
        if s not in summary or not length_eligible(design['lengths'][s]): continue
        a=summary[s]
        if all(a[m]<=1/8 for m in MODELS) or all(a[m]>=7/8 for m in MODELS): continue
        eligible[s]={**setting,'calibration':a,'spread':float(np.mean([a[m]*(1-a[m]) for m in MODELS]))}
    best={}
    for s,v in eligible.items():
        key=(v['spec']['task'],v['k'])
        if key not in best or v['spread']>best[key]['spread']: best[key]=v
    chosen=sorted(best.values(),key=lambda v:-v['spread'])[:CAP]
    plan={'rule':'XSPEC_PROTOCOL.md phase 1','settings':[{'id':v['id'],'spec':v['spec'],'k':v['k']} for v in chosen],'calibration':{v['id']:v['calibration'] for v in chosen}}
    write(DIRECTORY/'screen_plan.json',plan);write(DIRECTORY/'calibration/summary.json',{'accuracy':summary,'eligible':sorted(eligible),'chosen':[v['id'] for v in chosen]})
    print(json.dumps({'calibrated':len(summary),'eligible':len(eligible),'families_k':len(best),'chosen':len(chosen)}))


def screen():
    rows,generations=score('screen');design=read(DIRECTORY/'screen/design.json');cells=table(rows);result={}
    for s,cell in cells.items():
        q,g,h=preferences(cell,16)
        result[s]={'P_Q':float(q.mean()),'P_G':float(g.mean()),'H':float(h.mean()),
                   'accuracy':{r:{src:float(np.mean(cell[r][src][16])) for src in SOURCES} | {'initial':float(np.mean(cell[r]['initial'][0]))} for r in MODELS}}
    qualifying=[s for s,v in result.items() if v['P_Q']>=2/32-1e-9 and v['P_G']>=2/32-1e-9]
    best={}
    for s in qualifying:
        family=s.split('__')[0];key=(result[s]['H'],min(result[s]['P_Q'],result[s]['P_G']))
        if family not in best or key>best[family][0]: best[family]=(key,s)
    selected=[s for _,s in sorted(best.values(),reverse=True)][:SELECT]
    specs={x['id']:x for x in design['settings']}
    write(DIRECTORY/'screen/analysis.json',{'settings':result,'qualifying':qualifying,'selected':selected,'generation_summary':generation_summary(generations)})
    if selected: write(DIRECTORY/'confirmation_plan.json',{'rule':'XSPEC_PROTOCOL.md phase 2','settings':[specs[s] for s in selected]})
    print(json.dumps({'screened':len(result),'qualifying':qualifying,'selected':selected}))
    scatter(result,qualifying,selected)


def generation_summary(generations):
    out=defaultdict(dict);cells=defaultdict(list)
    for g in generations: cells[g['setting'],g['model']].append(g)
    for (s,m),v in cells.items(): out[s][m]={'correct':float(np.mean([x['correct'] for x in v])),'truncated':float(np.mean([x['status']!='ok' for x in v])),'chars':float(np.mean([x['chars'] for x in v])),'count':len(v)}
    return out


def scatter(result,qualifying,selected):
    figure,axis=plt.subplots(figsize=(6.4,6.4))
    x=np.array([v['P_Q'] for v in result.values()])*100;y=np.array([v['P_G'] for v in result.values()])*100
    rng=np.random.default_rng(0);jitter=lambda a:a+rng.uniform(-.6,.6,len(a))
    axis.axhline(0,color='grey',lw=.8);axis.axvline(0,color='grey',lw=.8)
    axis.scatter(jitter(x),jitter(y),alpha=.5,label=f'screened settings ({len(result)})')
    for s in qualifying:
        v=result[s];axis.scatter(v['P_Q']*100,v['P_G']*100,facecolors='none',edgecolors='C3' if s in selected else 'C1',s=90)
        if s in selected: axis.annotate(s.replace('__default','').replace('__',' '),(v['P_Q']*100,v['P_G']*100),xytext=(14,max(y)+4-3.5*selected.index(s)),textcoords='data',color='C3',arrowprops={'arrowstyle':'-','color':'C3','lw':.6})
    for s,v in result.items():  # label the large one-sided transfers for context
        if max(abs(v['P_Q']),abs(v['P_G']))>=.2:
            right=v['P_Q']>.25;axis.annotate(s.replace('__default','').replace('__',' '),(v['P_Q']*100,v['P_G']*100),textcoords='offset points',xytext=(-6 if right else 6,4),ha='right' if right else 'left',color='grey')
    axis.set_ylim(min(y)-5,max(y)+8)
    axis.set_xlabel('Qwen own − Gemma source (pp)');axis.set_ylabel('Gemma own − Qwen source (pp)');axis.set_title('Screen, N=16: own-source preference of each reader')
    axis.legend(loc='lower left');figure.tight_layout();figure.savefig(DIRECTORY/'screen/preferences.png',dpi=150)
    figure.savefig(REPORTS/'xspec-screen-preferences.png',dpi=150)


def confirmation():
    rows,generations=score('confirmation');design=read(DIRECTORY/'confirmation/design.json');cells=table(rows);settings=[s['id'] for s in design['settings']]
    dropped=read(DIRECTORY/'confirmation/dropped_counts.json');result={}
    for s in settings:
        cell=cells[s];q,g,h=preferences(cell,16);counts=sorted(cell['qwen']['qwen'])
        r={'P_Q':float(q.mean()),'P_G':float(g.mean()),'H':float(h.mean()),'P_Q_ci':bootstrap(q),'P_G_ci':bootstrap(g),'H_ci':bootstrap(h),'p_one_sided':sign_flip(h),'counts':counts,
           'curves':{rd:{src:{str(n):[float(np.mean(cell[rd][src][n])),*bootstrap(cell[rd][src][n])] for n in counts} for src in SOURCES} | {'initial':[float(np.mean(cell[rd]['initial'][0])),*bootstrap(cell[rd]['initial'][0])]} for rd in MODELS}}
        for rd,other in [('qwen','gemma'),('gemma','qwen')]:
            own=np.array(cell[rd][rd][16]);r[f'{rd}_own_minus_teacher']=[float((own-np.array(cell[rd]['teacher'][16])).mean()),*bootstrap(own-np.array(cell[rd]['teacher'][16]))]
            r[f'{rd}_own_minus_initial']=[float((own-np.array(cell[rd]['initial'][0])).mean()),*bootstrap(own-np.array(cell[rd]['initial'][0]))]
        result[s]=r
    adjusted=holm([result[s]['p_one_sided'] for s in settings])
    for s,p in zip(settings,adjusted): result[s]['holm_p']=p;result[s]['mutual']=bool(p<.05 and result[s]['P_Q_ci'][0]>0 and result[s]['P_G_ci'][0]>0)
    write(DIRECTORY/'confirmation/analysis.json',{'settings':result,'dropped_counts':dropped,'generation_summary':generation_summary(generations)})
    print(json.dumps({s:{k:result[s][k] for k in ['P_Q','P_G','H','H_ci','p_one_sided','holm_p','mutual']} for s in settings},indent=1))
    curves(result,settings);forest(result,settings)


def curves(result,settings):
    figure,axes=plt.subplots(len(settings),2,figsize=(10,3.2*len(settings)),squeeze=False)
    colors={'qwen':'C0','gemma':'C1','teacher':'C2'}
    for row,s in enumerate(settings):
        for col,rd in enumerate(MODELS):
            axis=axes[row,col];c=result[s]['curves'][rd]
            for src in SOURCES:
                ns=[int(n) for n in c[src]];points=[c['initial']]+[c[src][str(n)] for n in ns]  # N=0 is the shared initial-only baseline
                m,lo,hi=(np.array([x[i] for x in points])*100 for i in range(3))
                xs=[.7]+ns;label=f'{src} source'+(' (own)' if src==rd else '')
                axis.plot(xs,m,'o-',color=colors[src],label=label,lw=2.2 if src==rd else 1.2);axis.fill_between(xs,lo,hi,color=colors[src],alpha=.12)
            axis.set_xscale('log');axis.set_xticks([.7,2,8,16,32]);axis.set_xticklabels(['0','2','8','16','32'])
            axis.set_title(f"{s.replace('__',' · ')} — {rd.capitalize()} reads");axis.set_xlabel('rehearsal examples N');axis.set_ylabel('test accuracy (%)')
            if row==0 and col==0: axis.legend()
    figure.tight_layout();figure.savefig(DIRECTORY/'confirmation/curves.png',dpi=130)
    figure.savefig(REPORTS/'xspec-confirmation-curves.png',dpi=130)


def forest(result,settings):
    figure,axis=plt.subplots(figsize=(7.5,1.6+.9*len(settings)))
    for i,s in enumerate(settings):
        for j,(key,label,color) in enumerate([('P_Q','Qwen own − Gemma src','C0'),('P_G','Gemma own − Qwen src','C1'),('H','mean preference H','k')]):
            y=i-.25+.25*j;v=result[s][key]*100;lo,hi=np.array(result[s][key+'_ci'])*100
            axis.errorbar(v,y,xerr=[[v-lo],[hi-v]],fmt='o',color=color,label=label if i==0 else None)
    axis.axvline(0,color='grey',lw=.8);axis.set_yticks(range(len(settings)));axis.set_yticklabels([f"{s.replace('__',' · ')}\nHolm p={result[s]['holm_p']:.3g}" for s in settings])
    axis.invert_yaxis();axis.set_xlabel('percentage points at N=16 (95% episode bootstrap)');axis.legend(loc='upper center',bbox_to_anchor=(.5,-.22),ncol=3);figure.tight_layout()
    figure.savefig(DIRECTORY/'confirmation/preferences.png',dpi=150,bbox_inches='tight')
    figure.savefig(REPORTS/'xspec-confirmation-preferences.png',dpi=150,bbox_inches='tight')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['calibration','screen','confirmation']);args=parser.parse_args()
    {'calibration':calibration,'screen':screen,'confirmation':confirmation}[args.phase]()
