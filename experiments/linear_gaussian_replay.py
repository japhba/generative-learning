"""Exact population learning curves for teacher versus frozen self replay."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'task_search'/'linear_gaussian_replay'
PLOT=ROOT/'reports'/'task-search-linear-gaussian-replay.png'


def risk(weights, covariance):
    residual=np.eye(weights.shape[0])-weights@weights.T
    return float(.5*np.trace(residual@covariance@residual.T))


def gradient(weights, covariance):
    residual=weights@weights.T-np.eye(weights.shape[0])
    return (residual@covariance+covariance@residual)@weights


def check_gradient():
    rng=np.random.default_rng(120926)
    weights=rng.normal(size=(4,2))*.3
    generator=rng.normal(size=(4,3))
    covariance=generator@generator.T
    numeric=np.zeros_like(weights)
    epsilon=1e-6
    for index in np.ndindex(weights.shape):
        displacement=np.zeros_like(weights)
        displacement[index]=epsilon
        numeric[index]=(risk(weights+displacement,covariance)-risk(weights-displacement,covariance))/(2*epsilon)
    analytical=gradient(weights,covariance)
    assert np.allclose(numeric,analytical,rtol=1e-7,atol=1e-8)
    return float(np.max(np.abs(numeric-analytical)))


def main():
    gradient_error=check_gradient()
    eta=.02
    steps=400
    teacher=np.array([[1.],[0.]])
    teacher_covariance=teacher@teacher.T
    angle=np.deg2rad(30.)
    direction=np.array([[np.cos(angle)],[np.sin(angle)]])
    projector=direction@direction.T
    floor=.5*np.sin(angle)**2
    curves=[]
    for scale in tqdm([.5,1.],desc='Fixed illustrative cases',mininterval=5):
        initial=scale*direction
        own_covariance=initial@initial.T
        for source,covariance in [('own',own_covariance),('teacher',teacher_covariance)]:
            weights=initial.copy()
            records=[]
            for step in range(steps+1):
                records.append({'step':step,'teacher_risk':risk(weights,teacher_covariance),'source_loss':risk(weights,covariance),'scale':float(np.linalg.norm(weights)),'direction_alignment':float((weights.T@teacher).item()/np.linalg.norm(weights)),'weights':weights.tolist()})
                if step==steps: break
                weights-=eta*gradient(weights,covariance)
            losses=np.array([record['source_loss'] for record in records])
            assert np.isfinite(losses).all() and np.all(np.diff(losses)<1e-12)
            if source=='own':
                assert np.allclose(weights,projector@weights,atol=1e-12)
                assert np.allclose([record['direction_alignment'] for record in records],np.cos(angle),atol=1e-12)
                for record in records:
                    expected=.5*(np.sin(angle)**2+np.cos(angle)**2*(1-record['scale']**2)**2)
                    assert np.isclose(record['teacher_risk'],expected,atol=1e-12)
                assert abs(records[-1]['teacher_risk']-floor)<1e-6
            else:
                assert records[-1]['teacher_risk']<1e-7
            curves.append({'initial_scale':scale,'source':source,'records':records})
    summary=[{'initial_scale':curve['initial_scale'],'source':curve['source'],'initial_risk':curve['records'][0]['teacher_risk'],'final_risk':curve['records'][-1]['teacher_risk'],'final_scale':curve['records'][-1]['scale'],'final_direction_alignment':curve['records'][-1]['direction_alignment']} for curve in curves]
    result={'type':'Deterministic theoretical illustration; exact Gaussian expectations, no sampled data or fitted scaling laws','loss':'0.5 E ||x-W W^T x||^2','teacher_weights':teacher.tolist(),'initial_direction_degrees':30,'own_generator':'Frozen at the initial student weights; generated observations are constants during recognition learning','eta':eta,'steps':steps,'source_difference':'Covariance of the observations only; initialization, loss, optimizer and update count are identical','gradient_max_absolute_error':gradient_error,'analytic_own_risk_floor':float(floor),'summary':summary,'curves':curves}
    DATA.mkdir(parents=True,exist_ok=True)
    (DATA/'analysis.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,1,sharex=True,layout='constrained')
    for ax,scale in zip(axes,[.5,1.]):
        for curve in curves:
            if curve['initial_scale']!=scale: continue
            ax.plot([record['step'] for record in curve['records']],[record['teacher_risk'] for record in curve['records']],label='Own observations' if curve['source']=='own' else 'Teacher observations')
        ax.set_title(f'Initial scale {scale:g}; direction error 30 degrees')
        ax.set_ylabel('Held-out risk')
        ax.legend()
    axes[-1].set_xlabel('Population gradient updates (not sample count)')
    fig.savefig(PLOT)
    plt.close(fig)
    print(json.dumps({'checks':'Gradient, risk formula, own subspace invariance and teacher convergence passed','gradient_max_absolute_error':gradient_error,'summary':summary},indent=2))


if __name__=='__main__': main()
