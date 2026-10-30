"""Primary contrasts for a fresh generation-recognition loop confirmation."""

import argparse
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
from practice_drawing_sampled_analysis import controls


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    rows=json.loads((args.source/'summary.json').read_text())
    n=len(rows); conditions=list(rows['0']['conditions'])
    runs={c:[np.load(args.source/f'run_{i}/{c}.npz') for i in range(n)] for c in conditions}
    cat=m.Catalogue(); fitted=[]
    for i in tqdm(range(n),desc='Fresh-loop analytic controls'):
        fitted.append(controls(cat,np.load(args.source/f'run_{i}/world.npz')))
        (args.output/'controls.json').write_text(json.dumps(fitted,indent=2)+'\n')
    boot=np.random.default_rng(20103135).integers(n,size=(20000,n))
    def estimate(values):
        values=np.array(values)
        means=values[boot].mean(axis=-1)
        return dict(mean=float(values.mean()),bootstrap95=np.quantile(means,[.025,.975]).tolist(),
                    bootstrap9875=np.quantile(means,[.00625,.99375]).tolist(),per_world=values.tolist())
    final={c:np.array([r['records'][-1,2] for r in runs[c]]) for c in conditions}
    cv=np.array([r['cv_kl'] for r in fitted])
    result=dict(worlds=n,initial=estimate([r['records'][0,2] for r in runs['sleep']]),
        final={c:estimate(v) for c,v in final.items()},direct=estimate(cv),controls=fitted,
        sleep_minus={c:estimate(final['sleep']-v) for c,v in final.items() if c!='sleep'},
        sleep_minus_direct=estimate(final['sleep']-cv),certificate={},recognition={},stopping={})
    for c in conditions:
        result['certificate'][c]=dict(final_gain=estimate([r['records'][-1,4] for r in runs[c]]),
            certified_worlds=int(sum(r['records'][-1,4]>0 for r in runs[c])),
            minimum_upper_slack=float(min(r['records'][0,1]+rows[str(i)]['upper_ll'] for i,r in enumerate(runs[c]))),
            maximum_elbo_decrease=float(max(np.diff(r['records'][:,3]).max() for r in runs[c])),
            minimum_likelihood_slack=float(min(np.min(-r['records'][:,1]+r['records'][:,3]) for r in runs[c])))
        assert result['certificate'][c]['maximum_elbo_decrease']<1e-8
        assert result['certificate'][c]['minimum_likelihood_slack']>-1e-8
        assert result['certificate'][c]['minimum_upper_slack']>=0
        result['recognition'][c]=dict(final_nll=estimate([r['records'][-1,5] for r in runs[c]]),
                                     final_valid=estimate([r['records'][-1,6] for r in runs[c]]))
        result['stopping'][c]=[dict(step=int(r['records'][-1,0]),status=str(r['status'])) for r in runs[c]]
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(args.source,result)
    print(json.dumps({key:result[key] for key in ['initial','final','direct','sleep_minus','sleep_minus_direct','certificate']},indent=2))


def plot(source,result):
    conditions=list(result['final']); n=result['worlds']
    runs={c:[np.load(source/f'run_{i}/{c}.npz') for i in range(n)] for c in conditions}
    labels=dict(sleep='Generation–recognition loop',prior='Initial-generator sleep',wake='Cached-cause wake',
                frozen_proposal='Frozen neural proposer',fixed_posterior='Entire posterior frozen',fixed_cache='No new causes')
    grid=np.arange(0,1601,20)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for c in conditions:
        for ax,column in zip(axes,[2,6]):
            values=[np.interp(grid,r['records'][:,0],r['records'][:,column]) for r in runs[c]]
            ax.plot(grid,np.mean(values,axis=0),label=labels[c])
    axes[0].axhline(result['direct']['mean'],color='black',linestyle=':',label='CV analytic fits')
    axes[0].set(ylabel='Population KL (nats)',yscale='log')
    axes[1].set(xlabel='Updates; stopped models held fixed',ylabel='Valid reconstruction rate')
    for ax in axes:
        ax.set_xscale('symlog',linthresh=20); ax.set_xlim(0,1600)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-inference-learning.png'); plt.close(fig)
    fig,ax=plt.subplots(layout='constrained')
    for i,r in enumerate(runs['sleep']):
        ax.plot(grid,np.interp(grid,r['records'][:,0],r['records'][:,4]),label=f'World {i}')
    ax.axhline(0,color='black',linestyle=':')
    ax.set(xlabel='Updates; stopped models held fixed',ylabel='Certified likelihood gain (nats)',xscale='symlog')
    ax.set_xscale('symlog',linthresh=20)
    ax.set_xlim(0,1600)
    ax.legend(ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-inference-certificate.png'); plt.close(fig)


if __name__=='__main__': main()
