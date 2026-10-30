"""Paired world-level analysis and curves for generated inverse-inference rehearsal."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from inverse_icl_task import decode_mask, mask_text
from self_icl_api import save_json

ROOT=Path(__file__).resolve().parents[1]
LABELS={'zero':'No examples','generated_parts':'Generated partial sums','teacher_whole':'Teacher whole examples',
        'teacher_parts':'Teacher same partial sums','teacher_random_parts':'Teacher random partial sums','long_direct':'Direct reasoning'}
NAMES={'qwen_flash':'Qwen3.5 Flash','qwen36':'Qwen3.6 35B A3B','gemini31':'Gemini 3.1 Flash Lite'}


def array_for(cases,condition,n):
    return np.asarray([np.mean([score['correct'] for score in next(row for row in case['rows'] if row['condition']==condition and row['budget']==n)['scores']]) for case in cases])


def coverage(case,n):
    world=case['world']; dimension=world['dimension']; half=dimension//2
    maps={0:{},1:{}}
    for generated,audit in zip(case['generated'],case['generation_audits'],strict=True):
        if generated['budget']>n: continue
        for row in audit['records']:
            mask=decode_mask(row['subset'],dimension)
            if mask is None: continue
            for side in (0,1):
                if not any(mask[(1-side)*half:(2-side)*half]): maps[side][row['subset']]=row['correct']
    covered=[]
    for test in world['tests']:
        left=mask_text(test['mask'][:half]+[0]*half)
        right=mask_text([0]*half+test['mask'][half:])
        covered.append(left in maps[0] and maps[0][left] and right in maps[1] and maps[1][right])
    return float(np.mean(covered))


def analyze(directory,allow_partial=False):
    manifest=json.loads((directory/'manifest.json').read_text())
    groups=defaultdict(list)
    cases=[json.loads(path.read_text()) for path in sorted((directory/'cases').glob('*.json'))]
    config=manifest['config']
    if not allow_partial: assert len(cases)==len(config['models'])*len(config['dimensions'])*config['world_count'],'Incomplete cohort.'
    for case in cases: groups[(case['model'],case['world']['dimension'])].append(case)
    results={}
    for (model,dimension),group in groups.items():
        group.sort(key=lambda case:case['world']['seed'])
        n_worlds=len(group)
        draws=np.random.default_rng(101910).integers(0,n_worlds,size=(6000,n_worlds))
        keys=sorted({(row['condition'],row['budget']) for row in group[0]['rows']},key=lambda key:(key[1],key[0]))
        arrays={key:array_for(group,*key) for key in keys}
        rows=[]
        for key in keys:
            records=[next(row for row in case['rows'] if (row['condition'],row['budget'])==key) for case in group]
            values=arrays[key]; boot=values[draws].mean(axis=1)
            tokens=[sum(call['prompt_tokens']+call['completion_tokens'] for call in row['preparation']+[row['evaluation']]) for row in records]
            completion=[sum(call['completion_tokens'] for call in row['preparation']+[row['evaluation']]) for row in records]
            delta=values-arrays[('zero',0)]
            rows.append({'condition':key[0],'budget':key[1],'accuracy':float(values.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
                         'delta_zero':float(delta.mean()),'delta_zero_ci95':np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist(),
                         'mean_total_tokens':float(np.mean(tokens)),'mean_completion_tokens':float(np.mean(completion)),
                         'parse_failures':sum(score['parse_failure'] for row in records for score in row['scores']),
                         'output_limit_hits':sum(row['evaluation']['finish_reason']=='length' for row in records)})
        comparisons=[]
        for condition,n in keys:
            if condition!='generated_parts': continue
            for control in ('teacher_whole','teacher_parts','teacher_random_parts','long_direct'):
                if (control,n) not in arrays: continue
                delta=arrays[(condition,n)]-arrays[(control,n)]
                comparisons.append({'budget':n,'control':control,'delta':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(axis=1),[.025,.975]).tolist()})
        diagnostics=[]
        for n in sorted({row['budget'] for row in rows if row['condition']=='generated_parts'}):
            audits=[audit for case in group for generation,audit in zip(case['generated'],case['generation_audits'],strict=True) if generation['budget']<=n]
            diagnostics.append({'budget':n,'expected_rows':sum(row['expected_count'] for row in audits),'parsed_rows':sum(row['parsed_count'] for row in audits),
                                'correct_rows':sum(row['correct_count'] for row in audits),'correct_partial_coverage':float(np.mean([coverage(case,n) for case in group]))})
        results[f'{model}/d{dimension}']={'model':model,'dimension':dimension,'worlds':n_worlds,'queries':len(group[0]['world']['tests']),
                                         'rows':rows,'paired_comparisons':comparisons,'generation_diagnostics':diagnostics}
    result={'manifest':manifest,'groups':results,'uncertainty':'6000 paired bootstrap resamples of complete random worlds; pointwise 95% intervals.'}
    save_json(directory/'analysis.json',result)
    return result


def plot(result,destination):
    for key,group in result['groups'].items():
        baseline=next(row for row in group['rows'] if row['condition']=='zero')
        fig,ax=plt.subplots(layout='constrained')
        for condition in ('generated_parts','teacher_whole','teacher_parts','teacher_random_parts','long_direct'):
            rows=sorted([row for row in group['rows'] if row['condition']==condition],key=lambda row:row['budget'])
            if not rows: continue
            rows=[baseline]+rows
            x=[row['budget'] for row in rows]; y=[row['accuracy'] for row in rows]
            line,=ax.plot(x,y,marker='o',linestyle='-' if condition=='generated_parts' else '--',label=LABELS[condition])
            if condition=='generated_parts': ax.fill_between(x,[r['ci95'][0] for r in rows],[r['ci95'][1] for r in rows],alpha=.15,color=line.get_color())
        budgets=sorted({row['budget'] for row in group['rows']})
        ax.set_xscale('symlog',base=2,linthresh=2); ax.set_xticks(budgets,labels=budgets)
        ax.set(xlabel='Rehearsal examples / matched allowance',ylabel='Exact inverse accuracy',ylim=(0,1.02),
               title=f"{NAMES[group['model']]}: {group['dimension']} symbols, {group['worlds']} worlds")
        ax.grid(alpha=.2); ax.legend()
        path=destination/f"inverse-icl-{group['model']}-d{group['dimension']}.png"
        fig.savefig(path); plt.close(fig)

        fig,axes=plt.subplots(1,2,layout='constrained')
        for condition in ('generated_parts','teacher_whole','teacher_parts','long_direct'):
            rows=sorted([row for row in group['rows'] if row['condition']==condition],key=lambda row:row['budget'])
            if not rows: continue
            rows=[baseline]+rows
            axes[0].plot([r['mean_total_tokens'] for r in rows],[r['accuracy'] for r in rows],marker='o',label=LABELS[condition])
        axes[0].set(xlabel='Mean total API tokens',ylabel='Exact inverse accuracy'); axes[0].set_xscale('log'); axes[0].grid(alpha=.2)
        diagnostics=group['generation_diagnostics']
        axes[1].plot([r['budget'] for r in diagnostics],[r['correct_rows']/r['expected_rows'] for r in diagnostics],marker='o',label='Forward arithmetic accuracy')
        axes[1].plot([r['budget'] for r in diagnostics],[r['correct_partial_coverage'] for r in diagnostics],marker='s',label='Both true partials present')
        axes[1].set(xlabel='Generated partial examples',ylabel='Fraction',ylim=(0,1.02)); axes[1].grid(alpha=.2)
        axes[1].set_xscale('symlog',base=2,linthresh=2)
        axes[1].set_xticks(budgets,labels=budgets)
        fig.suptitle('Computation and reusable partial examples')
        fig.savefig(destination/f"inverse-icl-{group['model']}-d{group['dimension']}-diagnostics.png"); plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    result=analyze(args.phase,args.allow_partial)
    if args.plot: plot(result,ROOT/'reports')
    for key,group in result['groups'].items():
        print(key,'worlds',group['worlds'])
        for row in group['rows']: print(row['condition'],row['budget'],round(row['accuracy'],4),'unparsed',row['parse_failures'],'limit hits',row['output_limit_hits'])
        for row in group['generation_diagnostics']: print('generation',json.dumps(row))


if __name__=='__main__': main()
