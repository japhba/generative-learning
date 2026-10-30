"""Population controls and likelihood-certificate curves for balanced practice."""

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
import practice_drawing_spectral as spectral


def controls(cat,world):
    mapping=spectral.canonical_map(cat,world)
    def fit(data): return direct.direct(cat,data,mapping) | spectral.fits(cat,data,world,mapping)
    fitted=fit(world['observed'])
    scores={key:0. for key in fitted}
    rng=np.random.default_rng(int(world['seed'])+220000)
    for test in np.array_split(rng.permutation(len(world['observed'])),4):
        train=np.setdiff1d(np.arange(len(world['observed'])),test)
        for name,p in fit(world['observed'][train]).items(): scores[name]-=np.log(p[world['observed'][test]]).sum()
    selected=min(scores,key=scores.__getitem__)
    old=[key for key in fitted if not key.startswith('spectral')]
    selected_old=min(old,key=scores.__getitem__)
    risks={key:float(np.sum(world['teacher_p']*np.log(world['teacher_p']/p))) for key,p in fitted.items()}
    return dict(selected=selected,selected_without_spectral=selected_old,cv_nll=scores,kl=risks,
                cv_kl=risks[selected],cv_without_spectral_kl=risks[selected_old])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    shutil.copy2(spectral.__file__,args.output/'spectral_source.py')
    (args.output/'controls_protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        estimator='Initial-posterior three-command tensors; 6 view permutations times 8 fixed projections; simultaneous diagonalization with magnitude rectification; SVD relative rank threshold 1e-8, rank at most 3.',
        selection='Two fixed spectral variants choose by moment residual or training image likelihood. Fourfold CV selects among these and the nine prior direct fits.',
        leakage='Exposure frequencies and initial B are refitted within each fold, retaining only the independent initialization perturbations and canonical renderer-derived programs.',
        audit=spectral.audit()),indent=2)+'\n')
    rows=json.loads((args.source/'summary.json').read_text())
    n=len(rows); conditions=list(rows[0]['conditions'])
    runs={c:[np.load(args.source/f'{c}_{i}.npz') for i in range(n)] for c in conditions}
    cat=m.Catalogue()
    control=[]
    for i in tqdm(range(n),desc='Direct and spectral controls'):
        control.append(controls(cat,np.load(args.source/'worlds'/f'world_{i}.npz')))
        (args.output/'controls.json').write_text(json.dumps(control,indent=2)+'\n')
    boot=np.random.default_rng(20103107).integers(n,size=(20000,n))
    def estimate(values):
        values=np.array(values)
        return dict(mean=float(values.mean()),bootstrap95=np.quantile(values[boot].mean(axis=-1),[.025,.975]).tolist(),per_world=values.tolist())
    final={c:np.array([r['records'][-1,2] for r in runs[c]]) for c in conditions}
    cv=np.array([r['cv_kl'] for r in control])
    result=dict(source=str(args.source),worlds=n,initial=estimate([r['records'][0,2] for r in runs['practice']]),
        final={c:estimate(v) for c,v in final.items()},direct=estimate(cv),controls=control,
        practice_minus={c:estimate(final['practice']-v) for c,v in final.items() if c!='practice'},
        practice_minus_direct=estimate(final['practice']-cv),fixed_budgets={})
    for budget in [200,800,1600]:
        risk={c:np.array([np.interp(budget,r['records'][:,0],r['records'][:,2]) for r in runs[c]]) for c in conditions}
        risk['direct']=cv
        result['fixed_budgets'][budget]=dict(risk={c:estimate(v) for c,v in risk.items()},
            practice_minus={c:estimate(risk['practice']-v) for c,v in risk.items() if c!='practice'})
    result['certificate']={c:dict(min_final_certified_gain=float(min(r['records'][0,3]-r['records'][-1,3] for r in runs[c])),
        mean_final_certified_gain=float(np.mean([r['records'][0,3]-r['records'][-1,3] for r in runs[c]])),
        minimum_likelihood_slack=float(min(r['records'][:,4].min() for r in runs[c])),
        largest_bound_decrease=float(max(np.diff(r['records'][:,3]).max() for r in runs[c])),
        max_recomputed_bound_error=float(max(abs(r['final_bound_error']) for r in runs[c]))) for c in conditions}
    assert all(v['minimum_likelihood_slack']>=-1e-8 and v['largest_bound_decrease']<1e-8 and v['max_recomputed_bound_error']<1e-8 for v in result['certificate'].values())
    result['recognition_kl']={c:estimate([r['records'][-1,9] for r in runs[c]]) for c in conditions}
    result['stopping']={c:[dict(step=int(r['records'][-1,0]),status=str(r['status'])) for r in runs[c]] for c in conditions}
    (args.output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    plot(args.source,result)
    print(json.dumps({key:result[key] for key in ['initial','final','direct','practice_minus','practice_minus_direct','certificate']},indent=2))


def plot(source,result):
    conditions=list(result['final'])
    runs={c:[np.load(source/f'{c}_{i}.npz') for i in range(result['worlds'])] for c in conditions}
    cv=np.array(result['direct']['per_world'])
    labels=dict(practice='Generation–recognition loop',prior='Initial generator proposals',recognition_only='Recognition-only proposals',fixed='Frozen initial posterior')
    grid=np.arange(0,1601,20)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for c in conditions:
        axes[0].plot(grid,np.mean([np.interp(grid,r['records'][:,0],r['records'][:,2]) for r in runs[c]],axis=0),label=labels[c])
    axes[0].axhline(cv.mean(),color='black',linestyle=':',label='CV direct fits, including spectral')
    axes[0].set(ylabel='Population KL (nats)',yscale='log')
    for column,label in [(1,'Actual likelihood gain'),(3,'Certified lower bound')]:
        values=[np.interp(grid,r['records'][:,0],r['records'][0,column]-r['records'][:,column]) for r in runs['practice']]
        axes[1].plot(grid,np.mean(values,axis=0),label=label)
    axes[1].set(xlabel='Updates; stopped models held fixed',ylabel='Likelihood gain (nats)')
    axes[1].legend()
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-balanced-learning.png'); plt.close(fig)
    fig,axes=plt.subplots(2,1,layout='constrained')
    for c in conditions:
        for ax,col,label in zip(axes,[3,9],['Negative ELBO (nats)','Recognition KL (nats)']):
            ax.plot(grid,np.mean([np.interp(grid,r['records'][:,0],r['records'][:,col]) for r in runs[c]],axis=0),label=labels[c]); ax.set_ylabel(label)
    axes[-1].set_xlabel('Updates; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-balanced-recognition.png'); plt.close(fig)
    world=np.load(source/'worlds/world_0.npz')
    fig,axes=plt.subplots(3,1,layout='constrained')
    for ax,b,title in zip(axes,[world['initial_b'],runs['practice'][0]['b'],world['teacher_b']],['Initial motifs','After sampled balancing','Teacher motifs; evaluation only']):
        im=ax.imshow(b.reshape(3,12),vmin=0,vmax=1,aspect='auto')
        ax.set(title=title,xticks=range(12),xticklabels=['F','R','L','J']*3,yticks=range(3),ylabel='Motif')
    fig.colorbar(im,ax=axes,label='Command probability')
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-balanced-motifs.png'); plt.close(fig)


if __name__=='__main__': main()
