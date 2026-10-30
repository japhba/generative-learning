"""Secondary held-out-goal transfer stress test for the fixed cyclic composer.

Three goal identities never receive practice feedback.  They remain available
as current states, and all atomic transitions were observed during acquisition.
The all-goal arm intentionally exposes the nominal held-out goals.
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
from tqdm.auto import trange

import practice_loop_ring as ring


CONFIG={"goal_seed":903715,"training_seed":904715,"held_out_goals":3,"beta":6.,"horizon":6,"practice_rate":.1,"atomic_rate":1.,"batch":64,"maximum_updates":6000,"evaluate_every":100,"minimum_stationarity_step":2000,"stationarity_evaluations":20,"stationarity_absolute_slope":.01,"required_stationary_evaluations":3}
ARMS=["partial_goals","all_goals","atomic_only"]
STARTS=["natural","alias"]


def values(theta,transition):
    seeds,states=theta.shape
    probability=ring.sigmoid(CONFIG["beta"]*np.sin(theta[:,None,:]-theta[:,:,None]))
    value=np.broadcast_to(np.eye(states),(seeds,states,states)).copy()
    sid=np.arange(seeds)[:,None,None]
    goal=np.arange(states)[None,None,:]
    for _ in range(CONFIG["horizon"]):
        value=probability*value[sid,transition[:,:,0,None],goal]+(1-probability)*value[sid,transition[:,:,1,None],goal]
        value[:,np.arange(states),np.arange(states)]=1
    return value


def metrics(theta,truth,learned,test,train,all_queries):
    sid=np.arange(len(theta))[:,None]
    output=[]
    for transition in [truth,learned]:
        table=values(theta,transition)
        output.extend([table[sid,q[:,:,0],q[:,:,1]].mean(axis=1) for q in [test,train,all_queries]])
    output.append(ring.atomic(theta,learned,CONFIG["beta"])[0])
    return np.stack(output,axis=1)


def main():
    load_dotenv(Path.home()/".env")
    fast=Path(os.environ["FAST_CACHE_DIR"])
    out=fast/"generative_practice_loop_ring"/"transfer"
    out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1]
    data=np.load(fast/"generative_practice_loop_ring"/"confirmation"/"trajectories.npz")
    previous=np.load(fast/"generative_practice_loop_ring"/"controls"/"trajectories.npz")
    truth,learned,all_queries=[data[key].copy() for key in ["truth","learned","queries"]]
    seeds,states=truth.shape[:2]
    goal_rng=np.random.default_rng(CONFIG["goal_seed"])
    held_goals=np.stack([goal_rng.choice(states,CONFIG["held_out_goals"],replace=False) for _ in range(seeds)])
    masks=np.stack([np.isin(q[:,1],goals) for q,goals in zip(all_queries,held_goals)])
    test=np.stack([q[mask] for q,mask in zip(all_queries,masks)])
    train=np.stack([q[~mask] for q,mask in zip(all_queries,masks)])
    assert test.shape[1]==27 and train.shape[1]==81
    assert all(not set(q[:,1])&set(goals) for q,goals in zip(train,held_goals))
    aliases=np.rint(data["initial"]/(np.pi/2)).astype(int)%4
    starts={"natural":previous["pretrain_angles"][-1].copy(),"alias":np.array([0.,np.pi/2,-np.pi,-np.pi/2])[aliases]}
    protocol={"uk_start":datetime.now(ZoneInfo("Europe/London")).isoformat(),"config":CONFIG,"seeds":seeds,"arms":ARMS,"starts":STARTS,"primary_start":"natural","metric_columns":["held_goal_true_success","train_goal_true_success","all_goal_true_success","held_goal_internal_success","train_goal_internal_success","all_goal_internal_success","atomic_inverse_nll"],"scope":"Secondary transfer stress test, no hyperparameter tuning. Held-out goal identities never enter partial-goal practice rewards but remain current states, and were present during original atomic acquisition. All-goal arm intentionally practices those goals. Natural initialization is the previously saved5000atomic-update endpoint. Alias secondary starts have onlyfourwrappedphasevalues. Cohort stopping uses internal objectives only; true outcomes are evaluation-only.","held_out_goals":held_goals.tolist(),"stopping":{}}
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    shutil.copy2(__file__,out/"source.py")
    shutil.copy2(Path(ring.__file__),out/"shared_source.py")
    steps=np.arange(0,CONFIG["maximum_updates"]+1,CONFIG["evaluate_every"])
    archive={"steps":steps,"held_out_goals":held_goals,"train_queries":train,"test_queries":test,"all_queries":all_queries,"truth":truth,"learned":learned,"counts":data["counts"],"ordering":data["ordering"]}
    summary={}
    for si,start in enumerate(STARTS):
        theta={arm:starts[start].copy() for arm in ARMS}
        rng={arm:np.random.default_rng(CONFIG["training_seed"]+100*si+ai) for ai,arm in enumerate(ARMS)}
        curves={arm:[] for arm in ARMS}
        paths={arm:[] for arm in ARMS}
        active={arm:True for arm in ARMS}
        stationary={arm:0 for arm in ARMS}
        stop={arm:{"step":CONFIG["maximum_updates"],"reason":"budget"} for arm in ARMS}
        for step in trange(CONFIG["maximum_updates"]+1,desc=f"Ring transfer: {start}"):
            if step%CONFIG["evaluate_every"]==0:
                for arm in ARMS:
                    score=metrics(theta[arm],truth,learned,test,train,all_queries)
                    curves[arm].append(score)
                    paths[arm].append(theta[arm].copy())
                    assert np.all(np.isfinite(score))
                    if active[arm] and step>=CONFIG["minimum_stationarity_step"]:
                        metric=6 if arm=="atomic_only" else 4 if arm=="partial_goals" else 5
                        recent=np.array(curves[arm])[-CONFIG["stationarity_evaluations"]:,:,metric].mean(axis=1)
                        loss=recent if arm=="atomic_only" else 1-recent
                        time=steps[step//CONFIG["evaluate_every"]-CONFIG["stationarity_evaluations"]+1:step//CONFIG["evaluate_every"]+1]
                        split=CONFIG["stationarity_evaluations"]//2
                        slopes=[np.polyfit(np.log(time[window]),np.log(loss[window]+1e-12),1)[0] for window in [slice(0,split),slice(split,None)]]
                        stationary[arm]=stationary[arm]+1 if max(abs(np.array(slopes)))<CONFIG["stationarity_absolute_slope"] else 0
                        if stationary[arm]>=CONFIG["required_stationary_evaluations"]:
                            active[arm]=False
                            stop[arm]={"step":step,"reason":"internal_objective_stationary","log_log_slopes":[float(x) for x in slopes]}
            if step==CONFIG["maximum_updates"]:
                break
            for arm in ARMS:
                if not active[arm]:
                    continue
                if arm=="atomic_only":
                    gradient=ring.atomic(theta[arm],learned,CONFIG["beta"])[1]
                    rate=CONFIG["atomic_rate"]
                else:
                    queries=train if arm=="partial_goals" else all_queries
                    gradient=ring.sampled(theta[arm],learned,queries,CONFIG["horizon"],CONFIG["beta"],rng[arm],CONFIG["batch"],"practice")
                    rate=CONFIG["practice_rate"]
                theta[arm]+=rate*gradient
                assert np.all(np.isfinite(theta[arm]))
        protocol["stopping"][start]=stop
        summary[start]={}
        for arm in ARMS:
            curve=np.array(curves[arm])
            archive[f"curves_{start}_{arm}"]=curve
            archive[f"angles_{start}_{arm}"]=np.array(paths[arm])
            summary[start][arm]={"initial_mean":curve[0].mean(axis=0).tolist(),"final_mean":curve[-1].mean(axis=0).tolist(),"final_sem":(curve[-1].std(axis=0,ddof=1)/np.sqrt(seeds)).tolist(),"final_by_seed":curve[-1].tolist()}
        np.savez_compressed(out/"trajectories.npz",**archive)
    audit=abs(values(starts["natural"],learned)[np.arange(seeds)[:,None],all_queries[:,:,0],all_queries[:,:,1]].mean(axis=1)-ring.exact(starts["natural"],learned,all_queries,CONFIG["horizon"],CONFIG["beta"]))
    assert audit.max()<1e-14
    boot_rng=np.random.default_rng(905715)
    for start in STARTS:
        partial=archive[f"curves_{start}_partial_goals"][-1,:,0]
        atomic=archive[f"curves_{start}_atomic_only"][-1,:,0]
        difference=partial-atomic
        indices=boot_rng.integers(seeds,size=(10000,seeds))
        summary[start]["paired_held_goal_practice_minus_atomic"]={"mean":float(difference.mean()),"bootstrap95":np.quantile(difference[indices].mean(axis=1),[.025,.975]).tolist()}
    result=protocol|{"uk_finish":datetime.now(ZoneInfo("Europe/London")).isoformat(),"evaluation_audit_max_error":float(audit.max()),"summary":summary}
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,axes=plt.subplots(3,2,layout="constrained")
    for col,start in enumerate(STARTS):
        for row,metric,label in [(0,0,"held-out goals"),(1,1,"training goals"),(2,6,"atomic reconstruction")]:
            for arm in ARMS:
                curve=archive[f"curves_{start}_{arm}"][:,:,metric]
                keep=steps<=protocol["stopping"][start][arm]["step"]
                line,=axes[row,col].plot(steps[keep],curve.mean(axis=1)[keep],label=arm.replace("_"," "))
                low,high=np.quantile(curve,[.1,.9],axis=1)
                axes[row,col].fill_between(steps[keep],low[keep],high[keep],color=line.get_color(),alpha=.15)
            axes[row,col].set(ylabel="NLL" if row==2 else "Success",xlabel="Update",title=f"{start}: {label}")
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc="outside lower center",ncols=3)
    fig.savefig(out/"practice-loop-ring-transfer.png")
    fig.savefig(root/"tex"/"figures"/"practice-loop-ring-transfer.png")
    plt.close(fig)
    print(json.dumps({"stopping":protocol["stopping"],"summary":{start:{arm:summary[start][arm]["final_mean"] for arm in ARMS} for start in STARTS},"held_goal_effect":{start:summary[start]["paired_held_goal_practice_minus_atomic"] for start in STARTS}},indent=2))


if __name__=="__main__":
    main()
