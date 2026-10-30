"""Paired source contrasts, complete search records, and fresh-cohort learning curves."""

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from self_icl_api import save_json

ROOT=Path(__file__).resolve().parents[1]
NAMES={'qwen_flash':'Qwen3.5 Flash','gemini31':'Gemini 3.1 Flash Lite','qwen36':'Qwen3.6 35B A3B'}


def signflip(values):
    values=np.asarray(values); n=len(values)
    if n<=16:
        signs=np.asarray(list(itertools.product((-1,1),repeat=n)))
        return float(np.mean(np.abs(signs@values/n)>=abs(values.mean())-1e-12))
    signs=np.random.default_rng(78231).choice([-1,1],size=(30000,n))
    return float((1+np.sum(np.abs(signs@values/n)>=abs(values.mean())-1e-12))/(len(signs)+1))


def analyze(directory,reference='oracle',primary_budget=16):
    manifest=json.loads((directory/'manifest.json').read_text()); config=manifest['config']
    cases=[json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
    references={model:('oracle' if reference=='oracle' else next(other for other in config['models'] if other!=model)) for model in config['models']}
    assert len(cases)==len(config['cells'])*config['worlds_per_cell'],'Incomplete planned cohort.'
    groups=defaultdict(list)
    for case in cases: groups[(case['world']['family'],case['world']['level'])].append(case)
    assert set(groups)=={tuple(cell) for cell in config['cells']}
    summaries=[]
    for (family,level),group in sorted(groups.items()):
        group.sort(key=lambda c:c['world']['seed']); nworlds=len(group)
        draws=np.random.default_rng(601291).integers(nworlds,size=(6000,nworlds))
        for model in config['models']:
            keys=sorted({(row['source'],row['budget']) for row in group[0]['rows'] if row['model']==model},key=lambda item:(item[1],item[0]))
            arrays={key:np.asarray([next(row['accuracy'] for row in case['rows'] if row['model']==model and (row['source'],row['budget'])==key) for case in group]) for key in keys}
            rows=[]; comparisons=[]; fits=[]
            for source,budget in keys:
                values=arrays[(source,budget)]; records=[next(row for row in case['rows'] if row['model']==model and (row['source'],row['budget'])==(source,budget)) for case in group]
                evaluation_tokens=[sum(e['call']['prompt_tokens']+e['call']['completion_tokens'] for e in row['evaluations']) for row in records]
                generation_tokens=[sum(c['prompt_tokens']+c['completion_tokens'] for c in row['preparation']) for row in records]
                rows.append({'source':source,'budget':budget,'accuracy':float(values.mean()),'ci95':np.quantile(values[draws].mean(axis=1),[.025,.975]).tolist(),
                             'mean_evaluation_tokens':float(np.mean(evaluation_tokens)),'mean_generation_tokens':float(np.mean(generation_tokens)),
                             'mean_total_tokens':float(np.mean(np.asarray(evaluation_tokens)+generation_tokens)),
                             'unparsed':sum(item['prediction'] is None for row in records for item in row['scores']),
                             'evaluation_limit_hits':sum(e['call']['finish_reason']=='length' for row in records for e in row['evaluations'])})
            for n in config['budgets']:
                if not n: continue
                for control in ('oracle','zero')+tuple(other for other in config['models'] if other!=model):
                    key=(control,0 if control=='zero' else n)
                    if key not in arrays: continue
                    delta=arrays[(model,n)]-arrays[key]
                    comparisons.append({'budget':n,'control':control,'accuracy_difference':float(delta.mean()),
                                        'ci95':np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),'signflip_p':signflip(delta)})
            for lower,upper in ((1,8),(2,16),(4,32)):
                ns=np.asarray([n for n in config['budgets'] if lower<=n<=upper]); estimated={}; sampled={}
                if len(ns)<3 or lower not in ns or upper not in ns: continue
                x=np.log(ns); centered=x-x.mean()
                for source in (references[model],model):
                    errors=np.stack([1-arrays[(source,int(n))] for n in ns],axis=1)
                    means=errors.mean(axis=0); boot=errors[draws].mean(axis=1)
                    if np.any(means<=0) or np.any(boot<=0):
                        fits.append({'source':source,'window':[lower,upper],'status':'Zero empirical/bootstrap error; no floor or pseudocount imposed.'}); continue
                    estimated[source]=float(-(np.log(means)@centered)/(centered@centered))
                    sampled[source]=-(np.log(boot)@centered)/(centered@centered)
                    fits.append({'source':source,'window':[lower,upper],'beta':estimated[source],
                                 'ci95':np.quantile(sampled[source],[.025,.975]).tolist(),'local_betas':(-np.diff(np.log(means))/np.diff(np.log(ns))).tolist()})
                if len(estimated)==2:
                    fits.append({'source':'generated_minus_'+references[model],'window':[lower,upper],'beta':estimated[model]-estimated[references[model]],
                                 'ci95':np.quantile(sampled[model]-sampled[references[model]],[.025,.975]).tolist()})
            generation=[item for case in group for item in case['generations'][model]]
            identical={n:sum(next(row for row in case['rows'] if row['model']==model and row['source']==model and row['budget']==n)['answers']==next(row for row in case['rows'] if row['model']==model and row['source']==references[model] and row['budget']==n)['answers'] for case in group) for n in config['budgets'] if n}
            summaries.append({'family':family,'level':level,'model':model,'worlds':nworlds,'queries_per_world':config['test_count'],'rows':rows,'comparisons':comparisons,'fits':fits,
                              'generation':{'count':len(generation),'correct':sum(item['correct'] for item in generation),'unparsed':sum(item['answer'] is None for item in generation),
                                            'limit_hits':sum(item['call']['finish_reason']=='length' for item in generation),'identical_reference_contexts_by_budget':identical}})
    # Holm adjustment across task cells, separately for each learner and the fixed N=16 endpoint.
    for model in config['models']:
        primary=[next(c for c in value['comparisons'] if c['budget']==primary_budget and c['control']==references[model]) for value in summaries if value['model']==model] if primary_budget in config['budgets'] else []
        ordered=sorted(primary,key=lambda c:c['signflip_p']); running=0.
        for rank,row in enumerate(ordered): running=max(running,min(1.,(len(ordered)-rank)*row['signflip_p'])); row['holm_p']=running
    result={'manifest':manifest,'cells':summaries,'reference_by_model':references,'primary_budget':primary_budget,
            'uncertainty':f'6000 bootstrap resamples of complete paired worlds; pointwise intervals. Paired two-sided sign-flip tests, Holm-adjusted across task cells at N={primary_budget} within learner.'}
    save_json(directory/'analysis.json',result); return result


def plot(result,directory,teacher_label='Exact teacher'):
    directory.mkdir(exist_ok=True); config=result['manifest']['config']; budgets=config['budgets']
    for model in config['models']:
        cells=[cell for cell in result['cells'] if cell['model']==model]
        for cell in cells:
            zero=next(row for row in cell['rows'] if row['source']=='zero')
            fig,ax=plt.subplots(layout='constrained')
            for source in [model]+sorted({row['source'] for row in cell['rows']} - {model,'zero'}):
                rows=[zero]+sorted([row for row in cell['rows'] if row['source']==source],key=lambda row:row['budget'])
                label='Self-generated' if source==model else (teacher_label if source=='oracle' else NAMES[source]+' teacher')
                line,=ax.plot([r['budget'] for r in rows],[r['accuracy'] for r in rows],marker='o',label=label)
                ax.fill_between([r['budget'] for r in rows],[r['ci95'][0] for r in rows],[r['ci95'][1] for r in rows],alpha=.12,color=line.get_color())
            ax.set_xscale('symlog',base=2,linthresh=1); ax.set_xticks(budgets,labels=budgets)
            ax.set(xlabel='Additional examples N',ylabel='Held-out accuracy',ylim=(0,1),title=f"{cell['family'].replace('_',' ')} / setting {cell['level']}\n{NAMES[model]}, {cell['worlds']} worlds")
            ax.grid(alpha=.2); ax.legend(); fig.savefig(directory/f"{model}-{cell['family']}-{cell['level']}.png"); plt.close(fig)
        primary_budget=result['primary_budget']; reference=result['reference_by_model'][model]
        if primary_budget in budgets:
            families=sorted({cell['family'] for cell in cells}); levels=sorted({cell['level'] for cell in cells})
            matrix=np.full((len(families),len(levels)),np.nan)
            for cell in cells:
                contrast=next(row for row in cell['comparisons'] if row['budget']==primary_budget and row['control']==reference)
                matrix[families.index(cell['family']),levels.index(cell['level'])]=100*contrast['accuracy_difference']
            limit=max(float(np.nanmax(np.abs(matrix))),1)
            fig,ax=plt.subplots(layout='constrained'); picture=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
            ax.set_xticks(range(len(levels)),labels=[f'Setting {level}' for level in levels]); ax.set_yticks(range(len(families)),labels=[f.replace('_',' ') for f in families])
            for i,j in np.ndindex(matrix.shape):
                if np.isfinite(matrix[i,j]): ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',color='white' if abs(matrix[i,j])>.55*limit else 'black')
            title=teacher_label.lower() if reference=='oracle' else NAMES[reference]+' teacher'
            ax.set_title(NAMES[model]+f': generated minus {title} at N={primary_budget}')
            fig.colorbar(picture,ax=ax,label='Accuracy difference (percentage points)')
            fig.savefig(directory/f'{model}-source-contrasts.png'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('phase',type=Path); parser.add_argument('--plot',action='store_true')
    parser.add_argument('--reference',choices=['oracle','other_model'],default='oracle'); parser.add_argument('--primary-budget',type=int,default=16)
    parser.add_argument('--teacher-label',default='Exact teacher'); args=parser.parse_args()
    result=analyze(args.phase,args.reference,args.primary_budget)
    if args.plot: plot(result,args.phase/'plots',args.teacher_label)
    for cell in result['cells']:
        own=next(row for row in cell['rows'] if row['source']==cell['model'] and row['budget']==max(result['manifest']['config']['budgets']))
        teacher=next(row for row in cell['rows'] if row['source']==result['reference_by_model'][cell['model']] and row['budget']==own['budget'])
        print(cell['model'],cell['family'],cell['level'],'self',round(own['accuracy'],4),'teacher',round(teacher['accuracy'],4),'generation',cell['generation'])


if __name__=='__main__': main()
