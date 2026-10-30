"""Direct-fit controls and curves for practice with sampled initialization."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
import practice_drawing_library_analysis as direct
import practice_drawing_sampled as s
import practice_drawing_spectral as spectral


def spectral_fits(cat,observed,world):
    a,b=s.initial_parameters(cat.images[observed],world['noise'],world['initial_a'])
    code,image=cat.probabilities(a,b)
    frequency=np.bincount(observed,minlength=len(cat.images))/len(observed)
    posterior=code*frequency[cat.ids]/image[cat.ids]
    tensors=np.stack([np.bincount(cat.chunks[:,c],weights=posterior,minlength=64).reshape(4,4,4) for c in range(3)])
    candidates=[]
    for permutation in spectral.PERMUTATIONS:
        for projection in spectral.PROJECTIONS:
            b=spectral.decompose(tensors.mean(axis=0),permutation,projection)
            rank=len(b); indices=np.arange(3)%rank
            component=spectral.component_tensor(b).reshape(rank,64)
            mixing=np.maximum(tensors.reshape(3,64) @ np.linalg.pinv(component),0)+1e-6
            mixing/=mixing.sum(axis=-1,keepdims=True)
            a=m.row_normalize(mixing[:,indices]/np.bincount(indices,minlength=rank)[indices])
            b=m.row_normalize(b[indices])
            fitted=np.einsum('ck,kijl->cijl',a,spectral.component_tensor(b))
            _,p=cat.probabilities(a,b)
            candidates.append((float(np.sum((tensors-fitted)**2)),float(-np.log(p[observed]).mean()),p))
    return dict(spectral_moments=min(candidates,key=lambda c:c[0])[2],
                spectral_likelihood=min(candidates,key=lambda c:c[1])[2])


def controls(cat,world):
    observed=np.searchsorted(cat.images,world['observed'])
    mapping=cat.order[cat.offset[:-1]]
    def fit(data): return direct.direct(cat,data,mapping) | spectral_fits(cat,data,world)
    fitted=fit(observed)
    scores={key:0. for key in fitted}
    rng=np.random.default_rng(int(world['seed'])+342000)
    for test in np.array_split(rng.permutation(len(observed)),4):
        train=np.setdiff1d(np.arange(len(observed)),test)
        for key,p in fit(observed[train]).items(): scores[key]-=np.log(p[observed[test]]).sum()
    selected=min(scores,key=scores.__getitem__)
    risks={key:float(np.sum(world['teacher_p']*np.log(world['teacher_p']/p))) for key,p in fitted.items()}
    return dict(selected=selected,cv_nll=scores,kl=risks,cv_kl=risks[selected])


def plot(source,result,fixed_cache):
    n=result['worlds']; conditions=list(result['final'])
    runs={c:[np.load(source/f'run_{i}/{c}.npz') for i in range(n)] for c in conditions}
    conditions.append('fixed_cache')
    runs['fixed_cache']=[np.load(fixed_cache/f'fixed_cache_{i}.npz') for i in range(n)]
    labels=dict(sleep='Current-generator sleep',prior='Initial-generator sleep',wake='Cached-cause wake',
                frozen_proposal='Frozen proposal; posterior learns',fixed_posterior='Entire posterior frozen',
                fixed_cache='No new causes (retrospective)')
    grid=np.arange(0,1601,20)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for c in conditions:
        y=np.mean([np.interp(grid,r['records'][:,0],r['records'][:,2]) for r in runs[c]],axis=0)
        axes[0].plot(grid,y,label=labels[c])
    axes[0].axhline(result['direct']['mean'],color='black',linestyle=':',label='CV direct fits, including spectral')
    axes[0].set(ylabel='Population KL (nats)',yscale='log',xscale='symlog')
    axes[0].set_xscale('symlog',linthresh=20)
    for i,r in enumerate(runs['sleep']):
        axes[1].plot(grid,np.interp(grid,r['records'][:,0],r['records'][:,4]),label=f'World {i}')
    axes[1].axhline(0,color='black',linestyle=':')
    axes[1].set(xlabel='Updates; stopped models held fixed',ylabel='Certified gain (nats)')
    axes[1].set_xscale('symlog',linthresh=20)
    for ax in axes: ax.set_xlim(0,1600)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-sampled-learning.png'); plt.close(fig)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for c in conditions:
        for ax,column,label in zip(axes,[5,6],['True-program NLL (nats)','Valid reconstruction rate']):
            ax.plot(grid,np.mean([np.interp(grid,r['records'][:,0],r['records'][:,column]) for r in runs[c]],axis=0),label=labels[c])
            ax.set_ylabel(label)
    axes[-1].set_xlabel('Updates; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-sampled-recognition.png'); plt.close(fig)
    fig,ax=plt.subplots(layout='constrained')
    for i in range(n):
        history=np.load(source/f'run_{i}/initial.npz')['history']
        ax.plot(history[:,0],history[:,1],label=f'World {i}')
    ax.set(xlabel='Initial-generator draws',ylabel='Generated-program NLL (nats)',title='Learning initial recognition from generated examples')
    ax.legend(ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-sampled-initialization.png'); plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixed-cache',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        selection='Fourfold original-data CV. Canonical inverse is fixed lexicographic renderer map. Spectral initial parameters refitted from fold ink counts and independent noise.',
        spectral='Same fixed six permutations and eight projections, magnitude-rectified decomposition, two selection rules as the preceding spectral audit.'),indent=2)+'\n')
    rows=json.loads((args.source/'summary.json').read_text())
    n=len(rows); conditions=list(rows['0']['conditions'])
    runs={c:[np.load(args.source/f'run_{i}/{c}.npz') for i in range(n)] for c in conditions}
    cat=m.Catalogue(); fitted=[]
    for i in tqdm(range(n),desc='Sampled-study direct controls'):
        fitted.append(controls(cat,np.load(args.source/f'run_{i}/world.npz')))
        (args.output/'controls.json').write_text(json.dumps(fitted,indent=2)+'\n')
    boot=np.random.default_rng(20103132).integers(n,size=(20000,n))
    def estimate(values):
        values=np.array(values)
        return dict(mean=float(values.mean()),bootstrap95=np.quantile(values[boot].mean(axis=-1),[.025,.975]).tolist(),per_world=values.tolist())
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
        result['recognition'][c]=dict(initial_nll=estimate([r['records'][0,5] for r in runs[c]]),
            final_nll=estimate([r['records'][-1,5] for r in runs[c]]),final_valid=estimate([r['records'][-1,6] for r in runs[c]]))
        result['stopping'][c]=[dict(step=int(r['records'][-1,0]),status=str(r['status'])) for r in runs[c]]
    # U_0 - ell(theta_0) must be nonnegative on the certificate's event.
    result['initial_upper_slack']=[float(rows[str(i)]['upper_ll']+runs['sleep'][i]['records'][0,1]) for i in range(n)]
    assert min(result['initial_upper_slack'])>=0
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(args.source,result,args.fixed_cache)
    print(json.dumps({key:result[key] for key in ['initial','final','direct','sleep_minus','sleep_minus_direct','certificate']},indent=2))


if __name__=='__main__': main()
