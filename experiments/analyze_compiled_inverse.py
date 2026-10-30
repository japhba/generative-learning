"""Weighted inverse risks, paired encoder bootstrap, and finite-window scaling diagnostics."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from compiled_inverse_icl import score
from self_icl_api import save_json

ROOT=Path(__file__).resolve().parents[1]
LABELS={'generated':'Generated, ranked','teacher_iid':'Teacher, IID','teacher_ranked':'Teacher, same ranks','generated_iid':'Generated, IID','long_direct':'Direct reasoning'}
NAMES={'qwen_flash':'Qwen3.5 Flash','qwen36':'Qwen3.6 35B A3B','gemini31':'Gemini 3.1 Flash Lite'}


def theory(probabilities,budgets):
    p=np.asarray(probabilities)
    return {'ranked':[float(p[n:].sum()) for n in budgets],'iid':[float((p*(1-p)**n).sum()) for n in budgets]}


def slope(ns,risks): return -np.polyfit(np.log(ns),np.log(risks),1)[0]


def analyze(directory,rescore=False):
    manifest=json.loads((directory/'manifest.json').read_text())
    config=manifest['config']
    cases=[json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
    assert len(cases)==len(config['seeds'])*len(config['models']),'Incomplete cohort.'
    groups=defaultdict(list)
    for case in cases:
        if rescore:
            for row in case['rows']:
                values=[item for evaluation in row['evaluations'] for item in score(evaluation['scores'],evaluation['response'])]
                row['risk']=sum(item['weight']*(not item['correct']) for item in values); row['scores']=values
        groups[case['model']].append(case)
    result={'manifest':manifest,'rescore':rescore,'models':{},'uncertainty':'6000 paired bootstrap resamples of complete random encoders, including their curricula and stratified test samples.'}
    for model,group in groups.items():
        group.sort(key=lambda case:case['world']['seed'])
        draws=np.random.default_rng(201910).integers(0,len(group),size=(6000,len(group)))
        keys=sorted({(row['condition'],row['budget']) for row in group[0]['rows']},key=lambda x:(x[1],x[0]))
        arrays={key:np.asarray([next(row['risk'] for row in case['rows'] if (row['condition'],row['budget'])==key) for case in group]) for key in keys}
        means={key:float(array.mean()) for key,array in arrays.items()}
        boots={key:array[draws].mean(axis=1) for key,array in arrays.items()}
        rows=[]
        for key in keys:
            values=[next(row for row in case['rows'] if (row['condition'],row['budget'])==key) for case in group]
            tokens=[]; completion=[]; limit_hits=0
            for row in values:
                calls=row['preparation']+[evaluation['call'] for evaluation in row['evaluations']]
                tokens.append(sum(call['prompt_tokens']+call['completion_tokens'] for call in calls))
                completion.append(sum(call['completion_tokens'] for call in calls))
                limit_hits+=sum(evaluation['call']['finish_reason']=='length' for evaluation in row['evaluations'])
            rows.append({'condition':key[0],'budget':key[1],'risk':means[key],'ci95':np.quantile(boots[key],[.025,.975]).tolist(),
                         'mean_total_tokens':float(np.mean(tokens)),'mean_completion_tokens':float(np.mean(completion)),
                         'parse_failures':sum(item['prediction'] is None for row in values for item in row['scores']),'evaluation_limit_hits':limit_hits})
        comparisons=[]
        for condition,n in keys:
            if condition!='generated': continue
            for control in ('teacher_iid','teacher_ranked','generated_iid','long_direct','zero'):
                other=(control,0 if control=='zero' else n)
                if other not in arrays: continue
                delta=arrays[(condition,n)]-arrays[other]
                comparisons.append({'budget':n,'control':control,'risk_difference':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()})
        fits=[]
        for lower,upper in ((4,32),(8,64),(16,128)):
            ns=np.asarray([n for n in config['budgets'] if lower<=n<=upper])
            if len(ns)<3: continue
            exponents={}
            for condition in ('generated','teacher_iid','teacher_ranked','generated_iid'):
                selected=[(condition,int(n)) for n in ns]
                if not all(key in arrays for key in selected): continue
                y=np.asarray([means[key] for key in selected]); samples=np.stack([boots[key] for key in selected],axis=1)
                if not np.all(y>0) or not np.all(samples>0): continue
                x=np.log(ns); centered=x-x.mean()
                sampled=-(np.log(samples)@centered)/(centered@centered)
                exponents[condition]=sampled
                fits.append({'window':[lower,upper],'condition':condition,'beta':float(slope(ns,y)),
                             'ci95':np.quantile(sampled,[.025,.975]).tolist(),'local_betas':(-np.diff(np.log(y))/np.diff(np.log(ns))).tolist()})
            if 'generated' in exponents and 'teacher_iid' in exponents:
                difference=exponents['generated']-exponents['teacher_iid']
                fits.append({'window':[lower,upper],'condition':'generated_minus_teacher_iid',
                             'beta':float(slope(ns,np.asarray([means[('generated',int(n))] for n in ns]))-slope(ns,np.asarray([means[('teacher_iid',int(n))] for n in ns]))),
                             'ci95':np.quantile(difference,[.025,.975]).tolist()})
        diagnostics=[]
        p=np.asarray(group[0]['world']['probabilities'])
        for n in config['budgets']:
            if n==0: continue
            generations=[item for case in group for item in case['generation'] if item['label']<=n]
            correct_mass=[]; hit_errors=[]; hit_mass=[]
            for case in group:
                correct={item['label'] for item in case['generation'] if item['label']<=n and item['correct']}
                correct_mass.append(float(sum(p[label-1] for label in correct)))
                row=next(row for row in case['rows'] if row['condition']=='generated' and row['budget']==n)
                hit_errors.append(sum(item['weight']*(not item['correct']) for item in row['scores'] if item['label'] in correct))
                hit_mass.append(sum(item['weight'] for item in row['scores'] if item['label'] in correct))
            diagnostics.append({'budget':n,'correct_codes':sum(item['correct'] for item in generations),'requested_codes':len(generations),
                                'unparsed_codes':sum(item['code'] is None for item in generations),'generation_limit_hits':sum(item['call']['finish_reason']=='length' for item in generations),
                                'exact_correct_coverage':float(np.mean(correct_mass)),'covered_query_error':float(np.sum(hit_errors)/np.sum(hit_mass)) if np.sum(hit_mass)>0 else None})
        result['models'][model]={'worlds':len(group),'queries_per_world':len(group[0]['world']['queries']),'rows':rows,'comparisons':comparisons,
                                'finite_window_fits':fits,'diagnostics':diagnostics,'cache_theory':theory(p,config['budgets'])}
    save_json(directory/('rescored_analysis.json' if rescore else 'analysis.json'),result)
    return result


def plot(result,directory):
    budgets=result['manifest']['config']['budgets']
    for model,value in result['models'].items():
        baseline=next(row for row in value['rows'] if row['condition']=='zero')
        fig,ax=plt.subplots(layout='constrained')
        for condition in ('generated','teacher_iid','teacher_ranked','generated_iid','long_direct'):
            rows=sorted([row for row in value['rows'] if row['condition']==condition],key=lambda row:row['budget'])
            if not rows: continue
            if condition!='long_direct': rows=[baseline]+rows
            line,=ax.plot([r['budget'] for r in rows],[r['risk'] for r in rows],marker='o',label=LABELS[condition],linestyle='none' if condition=='long_direct' else ('-' if condition=='generated' else '--'))
            if condition in ('generated','teacher_iid'): ax.fill_between([r['budget'] for r in rows],[r['ci95'][0] for r in rows],[r['ci95'][1] for r in rows],alpha=.12,color=line.get_color())
        ax.set_xscale('symlog',base=2,linthresh=1); ax.set_xticks(budgets,labels=budgets)
        ax.set(xlabel='Rehearsal budget N',ylabel='Error under the query distribution',title=f'{NAMES[model]}: {value["worlds"]} encoders')
        ax.grid(alpha=.2); ax.legend()
        fig.savefig(directory/f'compiled-inverse-{model}.png'); plt.close(fig)

        fig,axes=plt.subplots(1,2,layout='constrained')
        for condition in ('generated','teacher_iid','teacher_ranked','generated_iid'):
            rows=sorted([row for row in value['rows'] if row['condition']==condition and row['budget']>0],key=lambda row:row['budget'])
            if not rows: continue
            axes[0].plot([r['budget'] for r in rows],[r['risk'] for r in rows],marker='o',label=LABELS[condition])
            axes[1].plot([r['mean_total_tokens'] for r in rows],[r['risk'] for r in rows],marker='o',label=LABELS[condition])
        ns=[n for n in budgets if n>0]
        for kind in ('ranked','iid'):
            values=[v for n,v in zip(budgets,value['cache_theory'][kind],strict=True) if n>0]
            axes[0].plot(ns,values,linestyle=':' if kind=='ranked' else '--',color='gray',label='Ideal cache: '+kind)
        direct=[row for row in value['rows'] if row['condition']=='long_direct']
        if direct: axes[1].scatter([r['mean_total_tokens'] for r in direct],[r['risk'] for r in direct],color='C4',label=LABELS['long_direct'])
        for ax in axes: ax.set_xscale('log',base=2); ax.set_yscale('log'); ax.grid(alpha=.2)
        axes[0].set_xticks(ns,labels=ns); axes[0].set(xlabel='Rehearsal examples',ylabel='Error')
        axes[1].set(xlabel='Mean total API tokens')
        axes[1].set_xscale('log')
        axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        entries={label:handle for ax in axes for handle,label in zip(*ax.get_legend_handles_labels(),strict=True)}
        fig.legend(entries.values(),entries.keys(),loc='outside lower center',ncols=2)
        fig.suptitle('Finite-range sample scaling and inference work')
        fig.savefig(directory/f'compiled-inverse-{model}-scaling.png'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    parser.add_argument('--rescore',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result=analyze(args.phase,args.rescore)
    if args.plot: plot(result,ROOT/'reports')
    for model,value in result['models'].items():
        print(model,value['worlds'],'worlds')
        for row in value['rows']: print(row['condition'],row['budget'],'risk',round(row['risk'],5),'unparsed',row['parse_failures'])
        for row in value['finite_window_fits']: print(json.dumps(row))
        for row in value['diagnostics']: print(json.dumps(row))


if __name__=='__main__': main()
