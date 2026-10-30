"""Compare the frozen cross-model tracking results and behavioral control gains."""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np

from analyze_training import estimate

ARMS=('expert','on_policy','random','frozen_policy')
LABELS={'expert':'Expert','on_policy':'Current model','random':'Random','frozen_policy':'Frozen model'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]/'results')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    load_dotenv(Path.home()/'.env')
    os.environ['MPLCONFIGDIR']=str(Path(os.environ['FAST_CACHE_DIR'])/'generative-learning/matplotlib')
    models={'qwen':('Qwen3-0.6B',args.root/'tracking_main_analysis/summary.json'),
            'smollm':('SmolLM2-135M',args.root/'tracking_smollm_analysis/summary.json')}
    summaries={key:json.loads(path.read_text()) for key,(label,path) in models.items()}
    probes={key:json.loads((args.root/f'feedback_experiment/{key}_controller_probe.json').read_text()) for key in models}
    exact={key:json.loads((args.root/f'feedback_experiment/{key}_exact_evaluation.json').read_text()) for key in models}
    assert all(summary['complete'] for summary in summaries.values())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args.output.mkdir(parents=True,exist_ok=True)
    paths=[]
    figure,axes=plt.subplots(1,2,sharey=True)
    for axis,(key,(label,path)) in zip(axes,models.items(),strict=True):
        summary=summaries[key]
        for arm in ARMS:
            rows=summary['curves'][arm]
            x=[row['budget'] for row in rows]
            y=[row['metrics']['recovery']['mean_absolute_error']['mean'] for row in rows]
            bounds=np.array([row['metrics']['recovery']['mean_absolute_error']['ci95'] for row in rows])
            axis.plot(x,y,marker='o',label=LABELS[arm])
            axis.fill_between(x,bounds[:,0],bounds[:,1],alpha=0.15)
        axis.axhline(summary['oracle_metrics']['recovery']['mean_absolute_error'],color='black',linestyle=':',label='Oracle')
        axis.set_xscale('symlog',linthresh=512)
        axis.set_xlim(left=0)
        axis.set_yscale('log')
        axis.set(xticks=[0,512,2048,8192],xticklabels=['0','512','2048','8192'],xlabel='Acquired state labels',title=label)
        axis.legend()
    axes[0].set_ylabel('Mean recovery error (log scale)')
    figure.tight_layout()
    path=args.output/'cross_model_recovery.png'
    figure.savefig(path);plt.close(figure);paths.append(str(path.resolve()))
    figure,axis=plt.subplots()
    order=('warm',*ARMS)
    for key,(label,path) in models.items():
        values=[[row['greedy_control_coefficients']['error'] for row in probes[key]['results'] if row['arm']==arm] for arm in order]
        line,=axis.plot(range(len(order)),[np.mean(value) for value in values],marker='o',label=label)
        for index,value in enumerate(values): axis.scatter([index]*len(value),value,color=line.get_color(),alpha=0.5)
    axis.axhline(-1,color='black',linestyle=':',label='Ideal correction')
    axis.set(xticks=range(len(order)),xticklabels=['Warm','Expert','Current','Random','Frozen'],ylabel='Effective coefficient on tracking error',title='Behavioral probe: points are training seeds')
    axis.legend();figure.tight_layout()
    path=args.output/'corrective_response.png'
    figure.savefig(path);plt.close(figure);paths.append(str(path.resolve()))
    report={'models':{},'plots':paths,'interpretation':'Original fresh-episode endpoints remain primary. Exact results refer to closed-domain lookup policies checked against every original final test action. Behavioral gains are not raw transformer weight coefficients. No power-law fit is asserted.'}
    for key,(label,path) in models.items():
        summary=summaries[key]
        results=exact[key]['results']
        report['models'][key]={'label':label,'parameter_count':probes[key]['results'][0]['parameter_count'],
            'original_endpoints':{arm:summary['curves'][arm][-1] for arm in ARMS},
            'exact_endpoints':{arm:{stratum:{field:estimate([row['metrics'][stratum][field] for row in results if row['arm']==arm]) for field in ('mean_absolute_error','stable_tracking')} for stratum in ('nominal','recovery','large_shift')} for arm in ARMS},
            'behavioral_error_gains':{arm:estimate([row['greedy_control_coefficients']['error'] for row in probes[key]['results'] if row['arm']==arm]) for arm in order},
            'original_actions_matched_by_exact_lookup':sum(row['recorded_actions_matched'] for row in results),
            'endpoint_contrasts':summary['endpoint_contrasts']}
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for path in paths:print(path)


if __name__=='__main__':main()
