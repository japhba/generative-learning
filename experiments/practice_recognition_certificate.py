"""Audit a recognition-error certificate for actual likelihood improvement.

The XOR model here is analytically soluble and is only a theorem diagnostic,
not a candidate for the research specification.
"""

from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

import practice_drawing_library as m


def audit():
    rng = np.random.default_rng(20103051)
    n = 20000
    p = rng.uniform(.01, .99, n)
    recognition = rng.dirichlet(np.ones(4), size=n)
    backward = rng.dirichlet(np.ones(2), size=(n, 2)).reshape(n, 4)
    joint = np.stack([(1-p)**2, p**2], axis=-1)
    log_weight = np.log(np.repeat(joint, 2, axis=-1)*backward/recognition)
    span = np.ptp(log_weight, axis=-1)
    marginal = recognition.reshape(n, 2, 2).sum(axis=-1)
    next_p = .75*p+.25*marginal[:, 1]
    new_joint = np.stack([(1-next_p)**2, next_p**2], axis=-1)
    q_gain = np.sum(marginal*np.log(new_joint/joint), axis=-1)
    actual = np.log(new_joint.sum(axis=-1)/joint.sum(axis=-1))
    posterior = joint/joint.sum(axis=-1, keepdims=True)
    latent_kl = np.sum(marginal*np.log(marginal/posterior), axis=-1)
    path_kl = np.log(joint.sum(axis=-1))-np.sum(recognition*log_weight, axis=-1)
    certificate = q_gain-span**2/8
    assert np.min(path_kl-latent_kl) >= -1e-12
    assert np.min(span**2/8-path_kl) >= -1e-12
    assert np.min(actual-certificate) >= -1e-12
    assert np.all(actual[certificate>0] > 0)
    # A truncated posterior's missing mass cannot be inferred from its log-weight range.
    old, new = np.array([.01, .81]), np.array([.04, .64])
    mass = old[0]/old.sum()
    truncated_q_gain = np.log(new[0]/old[0])
    truncated_actual = np.log(new.sum()/old.sum())
    corrected = truncated_q_gain+np.log(mass)
    assert truncated_q_gain>0 and truncated_actual<0 and corrected<=truncated_actual
    return dict(random_cases=n, certified=int((certificate>0).sum()), actual_improvements=int((actual>0).sum()),
        minimum_certificate_slack=float((actual-certificate).min()),
        minimum_range_bound_slack=float((span**2/8-path_kl).min()),
        truncated_counterexample=dict(q_gain=float(truncated_q_gain), actual_likelihood_gain=float(truncated_actual),
                                     posterior_cache_mass=float(mass), corrected_certificate=float(corrected)))


def main():
    output=m.CACHE/'recognition_certificate'
    output.mkdir(parents=True,exist_ok=False)
    result=audit()
    result['uk_time']=datetime.now(ZoneInfo('Europe/London')).isoformat()
    result['scope']='Analytic theorem audit, not a successful practice model. A uniform residual bound and complete support are substantive assumptions.'
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    shutil.copy2(__file__,output/'source.py')
    p,r=np.meshgrid(np.linspace(.01,.99,301),np.linspace(.01,.99,301))
    old=np.stack([(1-p)**2,p**2],axis=-1)
    proposal=np.stack([1-r,r],axis=-1)
    next_p=.75*p+.25*r
    new=np.stack([(1-next_p)**2,next_p**2],axis=-1)
    gain=np.sum(proposal*np.log(new/old),axis=-1)
    actual=np.log(new.sum(axis=-1)/old.sum(axis=-1))
    bound=gain-np.ptp(np.log(old/proposal),axis=-1)**2/8
    classes=np.where(actual<=0,0,np.where(bound>0,2,1))
    np.savez_compressed(output/'grid.npz',generator_p=p,recognition_r=r,actual_gain=actual,certified_gain=bound)
    fig,ax=plt.subplots(layout='constrained')
    colors=['C3','C7','C2']
    ax.imshow(classes,origin='lower',extent=[.01,.99,.01,.99],aspect='auto',cmap=ListedColormap(colors),vmin=0,vmax=2)
    grid=np.linspace(.01,.99,301)
    ax.plot(grid,grid**2/(grid**2+(1-grid)**2),color='black',label='Exact posterior')
    ax.set(xlabel='Current generator probability',ylabel='Recognized probability of latent cause 11',title='When recognition certifies a likelihood improvement')
    handles=[Patch(color=c,label=l) for c,l in zip(colors,['Likelihood does not improve','Improvement not certified','Certified improvement'])]
    handles+=ax.get_legend_handles_labels()[0]
    fig.legend(handles=handles,loc='outside upper center',ncol=2)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-recognition-certificate.png')
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
