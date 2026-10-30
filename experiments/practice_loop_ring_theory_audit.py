"""Independent directional dynamic-programming Hessian at the cyclic alias."""

import json
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

from practice_loop_ring import exact, atomic, sigmoid, sampled


def directional(theta, direction, transitions, queries, horizon, beta):
    states=len(theta)
    delta=theta[None,:]-theta[:,None]
    perturbation=direction[None,:]-direction[:,None]
    p=sigmoid(beta*np.sin(delta))
    dp=beta*np.cos(delta)*p*(1-p)*perturbation
    ddp=(-beta*np.sin(delta)*p*(1-p)+beta**2*np.cos(delta)**2*p*(1-p)*(1-2*p))*perturbation**2
    value=np.eye(states)
    first=np.zeros_like(value)
    second=np.zeros_like(value)
    goals=np.arange(states)[None,:]
    for _ in range(horizon):
        right,left=[value[transitions[:,a,None],goals] for a in range(2)]
        dr,dl=[first[transitions[:,a,None],goals] for a in range(2)]
        ddr,ddl=[second[transitions[:,a,None],goals] for a in range(2)]
        second=ddp*(right-left)+2*dp*(dr-dl)+p*ddr+(1-p)*ddl
        first=dp*(right-left)+p*dr+(1-p)*dl
        value=p*right+(1-p)*left
        np.fill_diagonal(value,1)
        np.fill_diagonal(first,0)
        np.fill_diagonal(second,0)
    return np.array([x[queries[:,0],queries[:,1]].mean() for x in [value,first,second]])


def initial_score_covariance(theta,transitions,queries,horizon,beta,batch):
    states=len(theta)
    second=[]
    means=[]
    for start,goal in queries:
        stack=[(start,0,1.,np.zeros(states))]
        rows=[]
        while stack:
            state,step,probability,score=stack.pop()
            if state==goal or step==horizon:
                rows.append((state==goal,probability,score))
                continue
            delta=theta[goal]-theta[state]
            p=float(sigmoid(beta*np.sin(delta)))
            for action in (0,1):
                local=(action-p)*beta*np.cos(delta)
                value=score.copy()
                value[goal]+=local
                value[state]-=local
                stack.append((int(transitions[state,1-action]),step+1,probability*(p if action else 1-p),value))
        reward=np.array([row[0] for row in rows])
        probability=np.array([row[1] for row in rows])
        score=np.array([row[2] for row in rows])
        assert abs(probability.sum()-1)<1e-12
        rho=probability@reward
        m1=(probability*reward)@score
        m0=(probability*(1-reward))@score
        M1=score.T@((probability*reward)[:,None]*score)
        M0=score.T@((probability*(1-reward))[:,None]*score)
        second.append(.5*((1-rho)*M1+rho*M0-np.outer(m1,m0)-np.outer(m0,m1)))
        means.append(m1)
    mean=np.mean(means,axis=0)
    return mean,(np.mean(second,axis=0)-np.outer(mean,mean))/(batch/2)


def main():
    load_dotenv(Path.home()/".env")
    out=Path(os.environ["FAST_CACHE_DIR"])/"generative_practice_loop_ring"/"theory_audit"
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    shutil.copy2(__file__,out/"source.py")
    shutil.copy2(root/"experiments"/"practice_loop_ring.py",out/"learner_source.py")
    states,horizon,beta,rate,steps=12,6,6.,.1,3000
    theta=np.arange(states)*np.pi/2
    transitions=np.stack([(np.arange(states)+1)%states,(np.arange(states)-1)%states],axis=-1)
    queries=np.array([(s,g) for s in range(states) for g in range(states) if s!=g and g not in transitions[s]])
    basis=np.eye(states)
    diagonal=np.array([directional(theta,v,transitions,queries,horizon,beta)[2] for v in basis])
    hessian=np.diag(diagonal)
    for i in range(states):
        for j in range(i+1,states):
            joint=directional(theta,basis[i]+basis[j],transitions,queries,horizon,beta)[2]
            hessian[i,j]=hessian[j,i]=(joint-diagonal[i]-diagonal[j])/2
    finite_difference={}
    for epsilon in [1e-4,5e-5]:
        numerical=np.column_stack([(exact((theta+epsilon*v)[None],transitions[None],queries[None],horizon,beta,True)[1][0]-exact((theta-epsilon*v)[None],transitions[None],queries[None],horizon,beta,True)[1][0])/(2*epsilon) for v in basis])
        finite_difference[str(epsilon)]=float(abs(numerical-hessian).max())
    modes=[]
    for k in range(states):
        f=np.exp(2j*np.pi*k*np.arange(states)/states)/np.sqrt(states)
        eigenvalue=float(np.real(f.conj()@hessian@f))
        residual=float(np.linalg.norm(hessian@f-eigenvalue*f))
        modes.append({"angle_frequency":k,"goal_hessian_eigenvalue":eigenvalue,"eigenvector_residual":residual,"linearized_3000_step_amplification":float((1+rate*eigenvalue)**steps),"atomic_nll_hessian_eigenvalue":float(beta*sigmoid(-beta)/states*4*np.sin(np.pi*k/states)**2)})
    atomic_hessian=np.column_stack([-(atomic((theta+1e-4*v)[None],transitions[None],beta)[1][0]-atomic((theta-1e-4*v)[None],transitions[None],beta)[1][0])/(2e-4) for v in basis])
    laplacian=2*np.eye(states)-np.roll(np.eye(states),1,axis=0)-np.roll(np.eye(states),-1,axis=0)
    atomic_expected=beta*sigmoid(-beta)/states*laplacian
    result={"states":states,"horizon":horizon,"beta":beta,"rate":rate,"steps":steps,"initial_success":float(exact(theta[None],transitions[None],queries[None],horizon,beta)[0]),"initial_gradient_norm":float(np.linalg.norm(exact(theta[None],transitions[None],queries[None],horizon,beta,True)[1])),"independent_dp_success_error":float(abs(directional(theta,basis[0],transitions,queries,horizon,beta)[0]-exact(theta[None],transitions[None],queries[None],horizon,beta)[0])),"hessian_finite_difference_errors":finite_difference,"circulant_error":float(max(abs(hessian[i]-np.roll(hessian[0],i)).max() for i in range(states))),"atomic_nll_hessian_error":float(abs(atomic_hessian-atomic_expected).max()),"modes":modes,"interpretation":"The alias is a saddle of goal success but a global minimum of constrained atomic CE. Infinitesimal mean-flow instabilities are slow: largest3000step gain only1.77. Large escape requires noninfinitesimal stochastic diffusion/nonlinear drift. Fourier angle perturbation k creates first-order embedding sidebands3±k; unstablek2 includes desiredfrequency1 but fastestk6 does not. The learned graph already admits a direct winding1/traversal or spectral reconstruction bypass."}
    confirmation=np.load(out.parent/"confirmation"/"trajectories.npz")
    controls=np.load(out.parent/"controls"/"trajectories.npz")
    ordered=np.take_along_axis(controls["pretrain_angles"][-1],confirmation["ordering"],axis=1)
    differences=np.angle(np.exp(1j*(np.roll(ordered,-1,axis=1)-ordered)))
    winding=np.rint(differences.sum(axis=1)/(2*np.pi)).astype(int)
    values,counts=np.unique(winding,return_counts=True)
    result["natural_pretraining_winding_counts"]={str(k):int(n) for k,n in zip(values,counts)}
    rng=np.random.default_rng(731)
    nonlinear=[]
    tangent=[]
    rotation=np.array([[0.,1.],[-1.,0.]])
    for scale in [.01,.1,1.]:
        point=theta+rng.normal(0,scale,states)
        direction=rng.normal(size=states)
        direction/=np.linalg.norm(direction)
        learner=exact(point[None],transitions[None],queries[None],horizon,beta,True)[1][0]@direction
        independent=directional(point,direction,transitions,queries,horizon,beta)[1]
        nonlinear.append(float(abs(learner-independent)))
        us=np.array([np.cos(point[0]),np.sin(point[0])])
        ug=np.array([np.cos(point[1]),np.sin(point[1])])
        tangent_source=(np.eye(2)-np.outer(us,us))@rotation@ug
        angle_source=-np.cos(point[1]-point[0])*np.array([-np.sin(point[0]),np.cos(point[0])])
        tangent.append(float(abs(tangent_source-angle_source).max()))
    result["independent_nonlinear_directional_gradient_max_error"]=max(nonlinear)
    result["tangent_angle_update_max_error"]=max(tangent)
    mean,score_covariance=initial_score_covariance(theta,transitions,queries,horizon,beta,64)
    covariance_trace=float(np.trace(score_covariance))
    curvature_trace=float(np.trace(hessian@score_covariance))
    result["initial_batch64_noise"]={"mean_gradient_norm":float(np.linalg.norm(mean)),"covariance_trace":covariance_trace,"goal_hessian_covariance_trace":curvature_trace,"eta_point1_second_order_expected_goal_change":.5*.1**2*curvature_trace,"matched_isotropic_goal_hessian_covariance_trace":float(np.trace(hessian)*covariance_trace/(states-1)),"fourier_variances":np.real(np.diag(np.fft.fft(np.fft.ifft(score_covariance,axis=1),axis=0))).tolist()}
    repetitions=12000
    repeated_theta=np.repeat(theta[None],repetitions,axis=0)
    repeated_transitions=np.broadcast_to(transitions,(repetitions,states,2))
    repeated_queries=np.broadcast_to(queries,(repetitions,len(queries),2))
    draws=sampled(repeated_theta,repeated_transitions,repeated_queries,horizon,beta,np.random.default_rng(732),64,"practice")
    one_step=[]
    for eta in [.0001,.001,.003,.01,.03,.1]:
        change=exact(repeated_theta+eta*draws,repeated_transitions,repeated_queries,horizon,beta)-result["initial_success"]
        one_step.append({"rate":eta,"mean_goal_change":float(change.mean()),"sem":float(change.std(ddof=1)/np.sqrt(repetitions))})
    result["initial_one_step_monte_carlo"]={"independent_batches":repetitions,"seed":732,"updates":one_step}
    assert max(finite_difference.values())<1e-9
    assert result["circulant_error"]<1e-12
    assert result["atomic_nll_hessian_error"]<1e-9
    assert result["independent_nonlinear_directional_gradient_max_error"]<1e-14
    assert result["tangent_angle_update_max_error"]<1e-14
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    np.savez_compressed(out/"hessians.npz",goal_hessian=hessian,atomic_nll_hessian=atomic_expected,initial_batch64_score_covariance=score_covariance,monte_carlo_gradient_batches=draws,initial_angles=theta,transitions=transitions,queries=queries)
    fig,ax=plt.subplots()
    ax.stem(np.arange(states),[x["goal_hessian_eigenvalue"] for x in modes])
    ax.axhline(0,color="black")
    ax.set(xlabel="Angle Fourier mode",ylabel="Goal-success Hessian eigenvalue")
    fig.tight_layout()
    fig.savefig(out/"practice-loop-ring-hessian.png")
    plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
