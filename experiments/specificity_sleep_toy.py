"""Equal-quality latent branches, restricted recognizers, identical sleep updates."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports'
DATA=ROOT/'results/task_search/specificity'


def main():
    rng=np.random.default_rng(120926);runs=4096;steps=256;eta=.1;initial=.25
    counts=[0,1,2,4,8,16,32,64,128,256]
    # phi_A/m=1, phi_B/m=(-1)^c. Each example has the same (m,c) in all arms.
    signs=rng.choice([-1.,1.],size=(runs,steps));magnitudes=rng.uniform(1,2,size=(runs,steps))
    labels=np.stack([magnitudes,signs*magnitudes],axis=1)
    assert np.all(np.abs(labels)==magnitudes[:,None,:])
    # Matrix dimensions are replicate, reader, source. No arm-specific step size.
    w=np.full((runs,2,2),initial);history=[]
    # Rich readers both have both features and a source-neutral initialization.
    rich=np.full((runs,2,2),initial);rich_history=[]
    for n in tqdm(range(steps+1),desc='Equal-quality sleep toy'):
        if n in counts:
            error=(np.abs(w)-1)**2
            history.append({'n':n,'mean_error':error.mean(axis=0).tolist(),'mean_weight':w.mean(axis=0).tolist(),'error_standard_error':(error.std(axis=0)/np.sqrt(runs)).tolist()})
            prediction_plus=rich[:,:,0]+rich[:,:,1];prediction_minus=rich[:,:,0]-rich[:,:,1]
            e=((np.abs(prediction_plus)-1)**2+(np.abs(prediction_minus)-1)**2)/2
            rich_history.append({'n':n,'mean_error':e.mean(axis=0).tolist()})
        if n==steps: break
        features=np.stack([np.ones(runs),signs[:,n]],axis=1)
        target=labels[:,:,n]/magnitudes[:,n,None]
        w+=eta*(features[:,:,None]*target[:,None,:]-w)
        predictions=np.einsum('rsf,rf->rs',rich,features)
        rich+=(eta/2)*(target-predictions)[:,:,None]*features[:,None,:]
    expected_own=np.array([(1-initial)**2*(1-eta)**(2*n) for n in counts])
    actual_own=np.array([r['mean_error'][0][0] for r in history]);assert np.allclose(expected_own,actual_own,atol=1e-13)
    assert np.allclose(w[:,0,0],w[:,1,1]) and np.allclose(w[:,0,1],w[:,1,0])
    assert max(rich_history[-1]['mean_error'])<1e-8
    result={'status':'Illustrative constructed mechanism, not a fit or an explanation established for the LLMs','runs':runs,'eta':eta,'initial_weight':initial,'counts':counts,'history':history,'rich_reader':rich_history,'source_validity':[1,1],'update':'w <- w + eta * ((phi_r / m)*(z_s / m) - w)','exact_own_error':'(1-w0)^2 * (1-eta)^(2*N)','seed':120926}
    (DATA/'sleep_toy.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(layout='constrained')
    ax.plot(counts,[r['mean_error'][0][0] for r in history],marker='o',label='Restricted reader: own source')
    ax.plot(counts,[r['mean_error'][0][1] for r in history],marker='o',label='Restricted reader: other source')
    ax.plot(counts,[np.mean(r['mean_error']) for r in rich_history],marker='o',label='Both features: either source')
    ax.set_xscale('symlog',linthresh=1);ax.set_xlabel('Replayed examples N');ax.set_ylabel('Normalized reconstruction error');ax.set_title('Both sources are 100% correct');ax.legend()
    fig.savefig(OUT/'task-search-specificity-sleep-toy.png');plt.close(fig)
    print(json.dumps({'exact_native_curve_verified':True,'source_quality':[1,1],'final_restricted_error':history[-1]['mean_error'],'final_rich_reader_error':rich_history[-1]['mean_error']}))


if __name__=='__main__': main()
