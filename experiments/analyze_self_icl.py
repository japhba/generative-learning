"""Paired, task-stratified analysis of the Self-ICL experiments."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from self_icl_api import save_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'gpt35': 'GPT-3.5 Instruct', 'qwen': 'Qwen3 30B A3B', 'mistral': 'Mistral Small 3.2', 'gemini': 'Gemini 2.5 Flash Lite'}
SHORT = {'boolean_expressions': 'Boolean expressions', 'navigate': 'Navigation', 'logical_deduction_seven_objects': 'Deduction (7 objects)', 'tracking_shuffled_objects_seven_objects': 'Shuffling (7 objects)'}


def ci(samples): return np.quantile(samples, [.025, .975]).tolist()


def summarize(directory):
    manifest = json.loads((directory/'manifest.json').read_text())
    cases = [json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
    assert len(cases) == len(manifest['models'])*len(manifest['tasks'])*manifest['count_per_task'], 'Cohort incomplete; do not analyze as confirmation.'
    grouped = defaultdict(list)
    for case in cases: grouped[(case['model'],case['task'])].append(case)
    output = {'manifest': manifest, 'models': {}}
    bootstrap = {}
    for model in manifest['models']:
        task_cases = [sorted(grouped[(model, task)],key=lambda c:c['index']) for task in manifest['tasks']]
        keys = sorted({(r['mode'],r['condition'],r['budget']) for r in task_cases[0][0]['rows']})
        arrays = {}
        for key in keys:
            arrays[key] = np.asarray([[next(row['correct'] for row in case['rows'] if (row['mode'],row['condition'],row['budget'])==key) for case in subset] for subset in task_cases],dtype=float)
        rng = np.random.default_rng(910410)
        draws = rng.integers(0, manifest['count_per_task'], size=(4000,len(task_cases),manifest['count_per_task']))
        samples = {key: np.take_along_axis(value[None,:,:],draws,axis=2).mean(axis=(1,2)) for key,value in arrays.items()}
        rows = [row for subset in task_cases for case in subset for row in case['rows']]
        model_result = {'n': sum(map(len,task_cases)), 'conditions': [], 'paired_contrasts': [], 'task_accuracy': {}}
        for key in keys:
            subset = [row for row in rows if (row['mode'],row['condition'],row['budget'])==key]
            tokens, costs = [], []
            for row in subset:
                calls = row['preparation']+([row['evaluation']] if row['evaluation'] is not None else [])
                tokens.append(sum(call['prompt_tokens']+call['completion_tokens'] for call in calls))
                costs.append(sum(call['cost_usd'] for call in calls))
            model_result['conditions'].append({'mode': key[0], 'condition': key[1], 'budget': key[2], 'accuracy': float(arrays[key].mean()),
                                               'ci95': ci(samples[key]), 'pipeline_failures': sum(r['pipeline_failure'] for r in subset),
                                               'answer_parse_failures': sum(r['answer_parse_failure'] for r in subset),
                                               'mean_total_tokens': float(np.mean(tokens)), 'mean_logical_cost_usd': float(np.mean(costs))})
            for task, value in zip(manifest['tasks'],arrays[key],strict=True):
                model_result['task_accuracy'].setdefault(task,[]).append({'mode': key[0], 'condition': key[1], 'budget': key[2], 'accuracy':float(value.mean())})
            if key[1]!='self': continue
            baseline = (key[0],'zero',0)
            delta = arrays[key]-arrays[baseline]
            wins, losses = int((delta>0).sum()), int((delta<0).sum())
            discordant = wins+losses
            p = min(1.,2*sum(math.comb(discordant,j) for j in range(min(wins,losses)+1))/2**discordant)
            model_result['paired_contrasts'].append({'mode':key[0], 'budget':key[2], 'delta':float(delta.mean()), 'ci95':ci(samples[key]-samples[baseline]), 'wins':wins,'losses':losses,'mcnemar_two_sided_p':p})
        model_result['effective_slopes'] = []
        if len(manifest['budgets'])>2:
            for mode in manifest['modes']:
                shot_keys = [key for key in keys if key[0]==mode and key[1]=='self' and key[2]>0]
                ks = np.asarray([key[2] for key in shot_keys],dtype=float)
                errors = np.asarray([1-arrays[key].mean() for key in shot_keys])
                if np.all(errors>0):
                    slope = -float(np.polyfit(np.log(ks),np.log(errors),1)[0])
                    local = (-np.diff(np.log(errors))/np.diff(np.log(ks))).tolist()
                    model_result['effective_slopes'].append({'mode':mode, 'beta_zero_floor':slope, 'local_betas':local, 'interpretation':'Descriptive finite-window slope only; unknown floor, independent k-shot generations, and potentially nonmonotone curves.'})
        output['models'][model] = model_result
        bootstrap[model] = samples
    save_json(directory/'analysis.json',output)
    return output, bootstrap


def plot_replication(summary, destination, audit):
    fig, axes = plt.subplots(1,2,sharey=True,layout='constrained')
    for ax, mode in zip(axes,('direct','cot'),strict=True):
        for model, result in summary['models'].items():
            rows = [next(r for r in result['conditions'] if r['mode']==mode and r['condition']==condition) for condition in ('zero','self')]
            mean = np.asarray([r['accuracy'] for r in rows])
            low,high = np.asarray([r['ci95'] for r in rows]).T
            line, = ax.plot([0,3],mean,marker='o',label=NAMES[model])
            ax.fill_between([0,3],low,high,alpha=.15,color=line.get_color())
            if mode=='direct':
                reference=next(r for r in result['conditions'] if r['condition']=='official_real')
                ax.errorbar([3],[reference['accuracy']],yerr=[[reference['accuracy']-reference['ci95'][0]],[reference['ci95'][1]-reference['accuracy']]],fmt='s',label='Real examples')
            if mode=='cot' and audit is not None:
                checked=next(item for item in audit['option_text_cot_contrasts'] if item['model']==model and item['budget']==3)
                line,=ax.plot([0,3],[checked['zero_accuracy'],checked['self_accuracy']],linestyle='--',marker='s',label='Option-text audit')
                low,high=np.asarray([checked['zero_ci95'],checked['self_ci95']]).T
                ax.fill_between([0,3],low,high,alpha=.15,color=line.get_color())
        ax.set(title='Direct' if mode=='direct' else 'Chain of thought', xlabel='Demonstrations',xticks=[0,3])
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy across 23 BBH tasks')
    axes[0].legend()
    if audit is not None: axes[1].legend()
    fig.suptitle('Self-ICL: 368 held-out BBH questions')
    fig.savefig(destination)
    plt.close(fig)


def plot_scaling(summary, destination, mode_counts):
    modes=summary['manifest']['modes']
    fig, grid = plt.subplots(1,len(modes),sharey=True,squeeze=False,layout='constrained')
    axes=grid[0]
    for ax,mode in zip(axes,modes,strict=True):
        for model, result in summary['models'].items():
            rows = sorted([r for r in result['conditions'] if r['mode']==mode and r['condition'] in ('zero','self')],key=lambda r:r['budget'])
            x = [r['budget'] for r in rows]
            mean = np.asarray([r['accuracy'] for r in rows]); low, high = np.asarray([r['ci95'] for r in rows]).T
            line, = ax.plot(x,mean,marker='o',label='Self-ICL' if len(summary['models'])==1 else NAMES[model])
            ax.fill_between(x,low,high,alpha=.15,color=line.get_color())
            if mode=='direct':
                real = sorted([r for r in result['conditions'] if r['mode']=='direct' and r['condition']=='real'],key=lambda r:r['budget'])
                ax.plot([r['budget'] for r in real],[r['accuracy'] for r in real],linestyle='--',color=line.get_color(),label='Real examples' if len(summary['models'])==1 else NAMES[model]+' real')
        ax.set_xscale('symlog',base=2,linthresh=1)
        ax.set_xticks(summary['manifest']['budgets'],labels=summary['manifest']['budgets'])
        title='Direct' if mode=='direct' else 'Chain of thought'
        ax.set(title=f'{title}\n{mode_counts[mode]} questions',xlabel='Demonstrations')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Accuracy across compositional BBH tasks')
    axes[0].legend()
    fig.suptitle('Self-ICL demonstration-count curves')
    fig.savefig(destination)
    plt.close(fig)
    fig, grid = plt.subplots(1,len(modes),sharey=True,squeeze=False,layout='constrained')
    for ax,mode in zip(grid[0],modes,strict=True):
        for model,result in summary['models'].items():
            rows=sorted([r for r in result['conditions'] if r['mode']==mode and r['condition'] in ('zero','self')],key=lambda r:r['budget'])
            line,=ax.plot([r['mean_total_tokens'] for r in rows],[r['accuracy'] for r in rows],marker='o',label='Self-ICL')
            if mode=='direct':
                real=sorted([r for r in result['conditions'] if r['condition']=='real'],key=lambda r:r['budget'])
                ax.plot([r['mean_total_tokens'] for r in real],[r['accuracy'] for r in real],linestyle='--',marker='s',color=line.get_color(),label='Real examples')
        ax.set_xscale('log')
        ax.xaxis.set_major_locator(matplotlib.ticker.LogLocator(base=10,subs=(1,3)))
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set(title=('Direct' if mode=='direct' else 'Chain of thought'),xlabel='Mean total API tokens')
        ax.grid(alpha=.2)
    grid[0,0].set_ylabel('Accuracy across compositional BBH tasks')
    grid[0,0].legend()
    fig.suptitle('Self-ICL: generation, labelling and inference work')
    fig.savefig(destination.with_name('self-icl-work.png'))
    plt.close(fig)
    for model,result in summary['models'].items():
        fig,axes=plt.subplots(1,len(summary['manifest']['tasks']),sharex=True,sharey=True,squeeze=False,layout='constrained')
        for ax,task in zip(axes.flat,summary['manifest']['tasks'],strict=True):
            for mode in modes:
                rows=sorted([r for r in result['task_accuracy'][task] if r['mode']==mode and r['condition'] in ('zero','self')],key=lambda r:r['budget'])
                ax.plot([r['budget'] for r in rows],[r['accuracy'] for r in rows],marker='o',label='Self-ICL '+mode)
            real=sorted([r for r in result['task_accuracy'][task] if r['condition']=='real'],key=lambda r:r['budget'])
            ax.plot([r['budget'] for r in real],[r['accuracy'] for r in real],linestyle='--',marker='o',label='Real examples')
            ax.set_xscale('symlog',base=2,linthresh=1)
            ax.set_xticks([0,1,3,8,16],labels=[0,1,3,8,16])
            ax.set_title(SHORT[task]); ax.grid(alpha=.2)
        axes[0,0].legend()
        fig.supxlabel('Generated demonstrations'); fig.supylabel('Accuracy'); fig.suptitle(NAMES[model])
        fig.savefig(destination.with_name(f'self-icl-scaling-{model}.png'))
        plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    parser.add_argument('--plot',choices=['replication','scaling'])
    parser.add_argument('--cot-phase',type=Path)
    parser.add_argument('--audit',type=Path)
    args=parser.parse_args()
    summary,bootstrap=summarize(args.phase)
    if args.plot=='replication': plot_replication(summary,ROOT/'reports/self-icl-replication.png',json.loads(args.audit.read_text()) if args.audit else None)
    if args.plot=='scaling':
        combined=json.loads(json.dumps(summary))
        model=next(iter(combined['models']))
        counts={mode:combined['models'][model]['n'] for mode in combined['manifest']['modes']}
        if args.cot_phase:
            cot=json.loads((args.cot_phase/'analysis.json').read_text())
            combined['manifest']['modes'].append('cot')
            counts['cot']=cot['models'][model]['n']
            for model,result in cot['models'].items():
                combined['models'][model]['conditions'].extend(result['conditions'])
                for task,rows in result['task_accuracy'].items(): combined['models'][model]['task_accuracy'][task].extend(rows)
        plot_scaling(combined,ROOT/'reports/self-icl-scaling.png',counts)
    for model,result in summary['models'].items():
        print(model,'n=',result['n'])
        for contrast in result['paired_contrasts']: print(json.dumps(contrast))


if __name__=='__main__': main()
