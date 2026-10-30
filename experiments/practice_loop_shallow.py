"""Predeclared original-data shallow potential control for the line-world task.

No synthetic trajectories enter training.  A single scalar potential per state
defines p(action=1|state,goal)=sigmoid(2*(potential_goal-potential_state)).
The action permutation is learned by the orientation of that potential.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from practice_loop_candidate import make_world, sigmoid


CONFIG = {"seed_start":12500,"seeds":16,"states":12,"observations":5,"observation_noise":.05,"split":"all","horizon":12,"initial_standard_deviation":.5,"learning_rate":1.,"updates":10000,"evaluate_every":50,"specified_checkpoints":[500,10000]}


def policy(potential):
    p = sigmoid(2*(potential[None,:]-potential[:,None]))
    return np.stack([1-p,p],axis=-1)


def evaluate(potential,truth,learned,queries,actions,test):
    probabilities = policy(potential)
    target_p = probabilities[queries[:,0],queries[:,1],actions]
    success = []
    for transition, greedy in [(truth,False),(learned,False),(truth,True)]:
        selected = np.eye(2)[probabilities.argmax(axis=-1)] if greedy else probabilities
        values = np.eye(len(truth))
        goal = np.arange(len(truth))[None,:]
        for _ in range(CONFIG["horizon"]):
            values = sum(selected[:,:,a]*values[transition[:,a,None],goal] for a in range(2))
            np.fill_diagonal(values,1)
        success.append(float(values[test[:,0],test[:,1]].mean()))
    return np.array([*success,-np.log(target_p).mean(),target_p.mean()])


def gradient(potential,queries,actions):
    p = sigmoid(2*(potential[queries[:,1]]-potential[queries[:,0]]))
    residual = 2*(actions-p)/len(actions)
    change = np.zeros_like(potential)
    np.add.at(change,queries[:,1],residual)
    np.add.at(change,queries[:,0],-residual)
    return change


def audit(potential,queries,actions):
    analytic = gradient(potential,queries,actions)
    numerical = np.zeros_like(potential)
    def objective(x):
        logits=2*(x[queries[:,1]]-x[queries[:,0]])
        return np.mean(actions*logits-np.logaddexp(0,logits))
    for i in range(len(potential)):
        offset=np.eye(len(potential))[i]*1e-5
        numerical[i]=(objective(potential+offset)-objective(potential-offset))/2e-5
    error=float(abs(numerical-analytic).max())
    assert error<1e-9
    return error


def main():
    load_dotenv(Path.home()/".env")
    output=Path(os.environ["FAST_CACHE_DIR"])/"generative_practice_loop_shallow"
    output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    metadata={"uk_start":datetime.now(ZoneInfo("Europe/London")).isoformat(),"config":CONFIG,"protocol":"No tuning; one potential actor, original atomic inverse CE only, fixed lr1 and500/10000checkpoints matching the deep all-goal run. All exact goal rollouts are evaluation only.","metrics":["true_all_goal_success","learned_world_all_goal_success","true_greedy_all_goal_success","atomic_inverse_nll","atomic_action_probability"],"checkers":[]}
    (output/"protocol.json").write_text(json.dumps(metadata,indent=2)+"\n")
    shutil.copy2(__file__,output/"source.py")
    shutil.copy2(root/"experiments"/"practice_loop_candidate.py",output/"world_source.py")
    curves=[]
    weights=[]
    data={}
    for index in tqdm(range(CONFIG["seeds"]),desc="Shallow original-only potential"):
        seed=CONFIG["seed_start"]+index
        truth,learned,counts,ordering,queries,actions,train,test=make_world(np.random.default_rng(seed),CONFIG["states"],CONFIG["observations"],CONFIG["observation_noise"],CONFIG["split"])
        potential=np.random.default_rng(seed+10000).normal(0,CONFIG["initial_standard_deviation"],CONFIG["states"])
        if index==0: metadata["gradient_audit_error"]=audit(potential,queries,actions)
        curve=[]
        path=[]
        for step in range(CONFIG["updates"]+1):
            if step%CONFIG["evaluate_every"]==0:
                curve.append(evaluate(potential,truth,learned,queries,actions,test))
                path.append(potential.copy())
            if step<CONFIG["updates"]: potential+=CONFIG["learning_rate"]*gradient(potential,queries,actions)
        curves.append(curve)
        weights.append(path)
        metadata["checkers"].append(float(np.mean(truth==learned)))
        for name,value in [("truth",truth),("learned",learned),("counts",counts),("ordering",ordering),("queries",queries),("actions",actions),("test",test)]: data[f"{name}_{seed}"]=value
    curves=np.array(curves)
    weights=np.array(weights)
    steps=np.arange(0,CONFIG["updates"]+1,CONFIG["evaluate_every"])
    metadata["checkpoints"]={str(step):{"mean":curves[:,step//CONFIG["evaluate_every"]].mean(axis=0).tolist(),"sem":(curves[:,step//CONFIG["evaluate_every"]].std(axis=0,ddof=1)/np.sqrt(CONFIG["seeds"])).tolist(),"by_seed":curves[:,step//CONFIG["evaluate_every"]].tolist()} for step in [0,*CONFIG["specified_checkpoints"]]}
    metadata["uk_finish"]=datetime.now(ZoneInfo("Europe/London")).isoformat()
    (output/"summary.json").write_text(json.dumps(metadata,indent=2)+"\n")
    np.savez_compressed(output/"curves_and_worlds.npz",steps=steps,curves=curves,potentials=weights,**data)
    fig,axes=plt.subplots(1,2,layout="constrained")
    for ax,metric,ylabel in [(axes[0],0,"True all-goal success"),(axes[1],3,"Original atomic inverse NLL")]:
        values=curves[:,:,metric]
        ax.plot(steps,values.mean(axis=0))
        low,high=np.quantile(values,[.1,.9],axis=0)
        ax.fill_between(steps,low,high,alpha=.15)
        ax.set(xlabel="Original-data updates",ylabel=ylabel,xscale="symlog")
    fig.savefig(root/"tex"/"archive"/"figures"/"practice-loop-shallow.png")
    plt.close(fig)
    print(json.dumps({"gradient_error":metadata["gradient_audit_error"],"checker_mean":np.mean(metadata["checkers"]),"checkpoints":{key:value["mean"] for key,value in metadata["checkpoints"].items()}},indent=2))


if __name__=="__main__":
    main()
