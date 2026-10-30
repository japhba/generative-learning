"""Compare learned recognition with and without direct dream-to-cache insertion."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

import practice_drawing_library as m


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--both',type=Path,required=True)
    parser.add_argument('--proposal',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    roots=dict(both=args.both,proposal=args.proposal)
    conditions=['sleep','prior','frozen_proposal']
    runs={route:{c:[np.load(root/f'{c}_{i}.npz') for i in range(8)] for c in conditions} for route,root in roots.items()}
    boot=np.random.default_rng(20103134).integers(8,size=(20000,8))
    def estimate(values):
        values=np.array(values)
        return dict(mean=float(values.mean()),bootstrap95=np.quantile(values[boot].mean(axis=-1),[.025,.975]).tolist(),per_world=values.tolist())
    result=dict(retrospective=True,routes={})
    for route in roots:
        risk={c:np.array([r['records'][-1,2] for r in runs[route][c]]) for c in conditions}
        result['routes'][route]=dict(final_kl={c:estimate(v) for c,v in risk.items()},
            sleep_minus={c:estimate(risk['sleep']-risk[c]) for c in ['prior','frozen_proposal']},
            recognition_valid={c:estimate([r['records'][-1,6] for r in runs[route][c]]) for c in conditions},
            certificate={c:estimate([r['records'][-1,4] for r in runs[route][c]]) for c in conditions})
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(roots)
    print(json.dumps(result,indent=2))


def plot(roots):
    runs={route:{c:[np.load(root/f'{c}_{i}.npz') for i in range(8)] for c in ['sleep','frozen_proposal']} for route,root in roots.items()}
    grid=np.arange(0,1601,20)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for route,label,style in [('both','Both routes','-'),('proposal','Recognition route','--')]:
        for condition,role,color in [('sleep','learns R','C0'),('frozen_proposal','fixed R','C1')]:
            for ax,column in zip(axes,[2,6]):
                values=[np.interp(grid,r['records'][:,0],r['records'][:,column]) for r in runs[route][condition]]
                ax.plot(grid,np.mean(values,axis=0),linestyle=style,color=color,label=f'{label}: {role}')
    axes[0].set(ylabel='Population KL (nats)',yscale='log')
    axes[0].set_xscale('symlog',linthresh=20)
    axes[1].set(xlabel='Updates; stopped models held fixed',ylabel='Valid reconstruction rate')
    axes[1].set_xscale('symlog',linthresh=20)
    for ax in axes: ax.set_xlim(0,1600)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-sampled-routes.png'); plt.close(fig)


if __name__=='__main__': main()
