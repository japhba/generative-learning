"""Online covariance learning with the correct singular values held fixed."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'/'task_search'/'linear_gaussian_online_rotation'
PLOT=ROOT/'reports'/'task-search-linear-gaussian-online-rotation.png'


def covariance(angle, eigenvalues):
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    return rotation@np.diag(eigenvalues)@rotation.T


def checks(eigenvalues):
    angle=np.pi/6
    observation=np.array([.7,-1.1])
    epsilon=1e-6
    plus=.5*np.square(covariance(angle+epsilon,eigenvalues)-np.outer(observation,observation)).sum()
    minus=.5*np.square(covariance(angle-epsilon,eigenvalues)-np.outer(observation,observation)).sum()
    coordinates=np.array([np.cos(angle)*observation[0]+np.sin(angle)*observation[1],-np.sin(angle)*observation[0]+np.cos(angle)*observation[1]])
    analytic=-2*(eigenvalues[0]-eigenvalues[1])*coordinates.prod()
    numeric=(plus-minus)/(2*epsilon)
    assert np.isclose(numeric,analytic,rtol=1e-8)
    teacher_covariance=np.diag(eigenvalues)
    risk=.5*np.square(covariance(angle,eigenvalues)-teacher_covariance).sum()
    assert np.isclose(risk,(eigenvalues[0]-eigenvalues[1])**2*np.sin(angle)**2)
    for angle in np.linspace(-np.pi,np.pi,13):
        assert np.allclose(np.linalg.eigvalsh(covariance(angle,eigenvalues)),np.sort(eigenvalues))
    return float(abs(numeric-analytic))


def main():
    eigenvalues=np.array([2.,1.])
    singular_values=np.sqrt(eigenvalues)
    gap=eigenvalues[0]-eigenvalues[1]
    initial=np.pi/6
    eta=.02
    steps=512
    runs=4096
    seed=20260912
    gradient_error=checks(eigenvalues)
    rng=np.random.default_rng(seed)
    angles=np.full((2,runs),initial)
    history=np.empty((steps+1,2,runs))
    history[0]=angles
    sources=['teacher','own']
    for step in tqdm(range(steps),desc='Online fixed-spectrum rotations',mininterval=5):
        # Paired independent Gaussian innovations; only the generating matrix differs.
        latent=rng.normal(size=(2,runs))
        generator_angle=np.stack([np.zeros(runs),angles[1]])
        generator_cos=np.cos(generator_angle)
        generator_sin=np.sin(generator_angle)
        x1=generator_cos*singular_values[0]*latent[0]-generator_sin*singular_values[1]*latent[1]
        x2=generator_sin*singular_values[0]*latent[0]+generator_cos*singular_values[1]*latent[1]
        # Recognition is W^T x. The angle gradient uses only these activations.
        reader_cos=np.cos(angles)
        reader_sin=np.sin(angles)
        q1=singular_values[0]*(reader_cos*x1+reader_sin*x2)
        q2=singular_values[1]*(-reader_sin*x1+reader_cos*x2)
        gradient=-2*gap*q1*q2/singular_values.prod()
        angles-=eta*gradient
        history[step+1]=angles
    assert np.isfinite(history).all()
    # Correctness is analyzed after completing the prospectively fixed cohort.
    risks=gap**2*np.sin(history)**2
    mean=risks.mean(axis=2)
    standard_error=risks.std(axis=2,ddof=1)/np.sqrt(runs)
    counts=np.arange(steps+1)
    increment_coefficient=2*gap*singular_values.prod()
    correlation=1/np.sqrt(1+4*eta**2*increment_coefficient**2)
    expected_own=.5*gap**2*(1-np.cos(2*initial)*correlation**counts)
    standardized=(mean[1:,1]-expected_own[1:])/standard_error[1:,1]
    assert np.max(np.abs(standardized))<6
    own_displacement=history[-1,1]-initial
    expected_variance=steps*(eta*increment_coefficient)**2
    assert abs(own_displacement.mean())<6*np.sqrt(expected_variance/runs)
    assert abs(own_displacement.var(ddof=1)/expected_variance-1)<.1
    curves=[{'source':source,'mean_risk':mean[:,index].tolist(),'standard_error':standard_error[:,index].tolist()} for index,source in enumerate(sources)]
    summary=[{'source':source,'initial_risk':float(mean[0,index]),'final_mean_risk':float(mean[-1,index]),'final_mean_risk_monte_carlo_standard_error':float(standard_error[-1,index])} for index,source in enumerate(sources)]
    result={'type':'Prospectively fixed numerical illustration of an analytic online-learning model; not an LLM result or a fitted scaling law','eigenvalues':eigenvalues.tolist(),'singular_values':singular_values.tolist(),'teacher_angle':0,'initial_angle_degrees':30,'eta':eta,'steps':steps,'runs':runs,'seed':seed,'sources':sources,'generation':'Each update uses a fresh Gaussian sample from the teacher or the current student; the generated observation is held fixed during differentiation','loss':'0.5 ||C(theta)-x x^T||_F^2','heldout_excess_risk':'0.5 ||C(theta)-C_T||_F^2 = (lambda1-lambda2)^2 sin(theta)^2','recognizer':'q=W(theta)^T x','angle_gradient':'-2 (lambda1-lambda2) q1 q2 / (sigma1 sigma2)','own_expected_angle_update':0,'teacher_expected_angle_update':'-eta (lambda1-lambda2)^2 sin(2 theta)','own_expected_risk':expected_own.tolist(),'own_random_walk_expected_final_variance':float(expected_variance),'own_random_walk_observed_final_variance':float(own_displacement.var(ddof=1)),'own_random_walk_observed_mean_displacement':float(own_displacement.mean()),'maximum_own_risk_error_in_monte_carlo_standard_errors':float(np.max(np.abs(standardized))),'gradient_absolute_error':gradient_error,'counts':counts.tolist(),'summary':summary,'curves':curves}
    DATA.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DATA/'trajectories.npz',angles=history)
    (DATA/'analysis.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(layout='constrained')
    ax.plot(counts,mean[:,0],label='Teacher samples: online SGD')
    ax.fill_between(counts,mean[:,0]-1.96*standard_error[:,0],mean[:,0]+1.96*standard_error[:,0],alpha=.2)
    ax.plot(counts,mean[:,1],label='Current-student samples: online SGD')
    ax.fill_between(counts,mean[:,1]-1.96*standard_error[:,1],mean[:,1]+1.96*standard_error[:,1],alpha=.2)
    ax.plot(counts,expected_own,linestyle='--',label='Current-student samples: exact mean')
    ax.axhline(mean[0,0],linestyle=':',label='No replay')
    ax.set_xlabel('Online rehearsal samples N')
    ax.set_ylabel('Held-out covariance excess risk')
    ax.set_title('Correct singular values; only orientation learns')
    ax.legend()
    fig.savefig(PLOT)
    plt.close(fig)
    print(json.dumps({'summary':summary,'analytic_self_final_risk':float(expected_own[-1]),'gradient_absolute_error':gradient_error,'maximum_own_risk_error_in_monte_carlo_standard_errors':float(np.max(np.abs(standardized)))},indent=2))


if __name__=='__main__': main()
