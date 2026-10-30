"""All-seed scaling, forecast failures, and measured weight geometry."""

import json

import matplotlib.pyplot as plt
import numpy as np

from audit_scaling_saved import family_grid
from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def read_history(directory):
    path = directory / "complete.json"
    complete = path.exists()
    if not complete: path = directory / "progress.json"
    return {"complete": complete, "history": json.loads(path.read_text())}


def slope(x, y):
    centered = np.log(x) - np.log(x).mean()
    return float(-centered @ np.log(y) / (centered @ centered))


def power_prediction(step, error, overlap, eta, gamma, targets):
    coefficient = overlap / error**gamma
    return (error**(1-gamma) + (gamma-1)*eta*coefficient*(targets-step))**(-1/(gamma-1))


def main():
    root = ROOT / "results/scaling_s4g/digit"
    data = {"updated_at_uk": stamp(), "pilot": {}, "confirmation": {}, "fits": {}, "bootstrap": {"windows":[]}, "matched_comparison":None,"late_forecasts":[],
            "interpretation": "Finite-window estimates. A stable asymptote additionally requires a known floor and stable gradient geometry. Retain failed forecasts and stopping outcomes."}
    for stage in ("pilot", "confirmation"):
        protocol = json.loads((root / stage / "protocol.json").read_text())
        for seed in protocol["seeds"]:
            for source in ("current", "frozen"):
                directory = root / stage / str(seed) / source
                if not (directory / "progress.json").exists(): continue
                result = read_history(directory); h = result["history"]
                key = f"{seed}_{source}"
                questions = json.loads((root / stage / str(seed) / "evaluation.json").read_text())
                labels = {r["id"]:r["label"] for r in questions}
                result["by_answer"] = {label:[{"step":r["step"],"error":float(np.mean([v["error"] for v in r["records"] if labels[v["id"]]==label]))} for r in h["evaluations"]] for label in ("0","1","2")}
                forecasts = json.loads((root / stage / str(seed) / "forecast.json").read_text())
                result["initial_forecast_rmse"] = float(np.sqrt(np.mean([(r["accuracy"] - forecasts["prediction"]["curves"][source][r["step"]]["accuracy"])**2 for r in h["evaluations"]])))
                first = json.loads((directory / "attempt0/train_0000.json").read_text())
                result["first_sampled_targets"] = first["records"]
                result["finite_window_slopes"] = []
                for start, end in ((32,128),(128,1024),(256,2048),(512,2048),(1024,2048)):
                    rows = [r for r in h["evaluations"] if start <= r["step"] <= end]
                    if h["step"] < end: continue
                    result["finite_window_slopes"].append({"start":start,"end":end,"beta":slope(np.array([r["step"] for r in rows]),np.array([r["error"] for r in rows]))})
                geometry = []
                for record in h["gradients"]:
                    error = 1-record["eval_success"]
                    overlap = record["current_success_gradient_overlap"]
                    geometry.append({**record,"error":error,"cubic_coefficient":overlap/error**3,"quadratic_coefficient":overlap/error**2,
                                     "train_eval_cosine":overlap/np.sqrt(record["train_gradient_norm_squared"]*record["eval_gradient_norm_squared"]),
                                     "eval_initial_direction_cosine":record["initial_direction_projection"]/np.sqrt(record["eval_gradient_norm_squared"])})
                result["geometry"] = geometry
                if source == "current" and 128 in [r["step"] for r in geometry]:
                    anchor = next(r for r in geometry if r["step"] == 128)
                    later = [r for r in h["evaluations"] if r["step"] > (384 if stage == "pilot" else 128)]
                    result["anchored_forecasts"] = {}
                    for gamma in (2,3,4):
                        targets = np.array([r["step"] for r in later])
                        predicted = power_prediction(128,anchor["error"],anchor["current_success_gradient_overlap"],h["eta"],gamma,targets)
                        result["anchored_forecasts"][str(gamma)] = [{"step":r["step"],"actual_error":r["error"],"predicted_error":float(p),"actual_over_prediction":r["error"]/p} for r,p in zip(later,predicted,strict=True)]
                data[stage][key] = result
        for seed in protocol["seeds"]:
            if all(f"{seed}_{s}" in data[stage] for s in ("current","frozen")):
                left,right=[data[stage][f"{seed}_{s}"] for s in ("current","frozen")]
                assert left["first_sampled_targets"] == right["first_sampled_targets"], "Matched-source first update differs."
                assert left["history"]["updates"][0]["loss"] == right["history"]["updates"][0]["loss"]
    current = [r for key,r in data["confirmation"].items() if key.endswith("_current")]
    if len(data["confirmation"])==6:
        common=sorted(set.intersection(*({r["step"] for r in h["history"]["evaluations"]} for h in data["confirmation"].values())))
        endpoint=common[-1];differences=[];means={"current":[],"frozen":[]}
        for seed in (99201,99202,99203):
            paired={source:next(r for r in data["confirmation"][f"{seed}_{source}"]["history"]["evaluations"] if r["step"]==endpoint) for source in ("current","frozen")}
            assert [r["id"] for r in paired["current"]["records"]]==[r["id"] for r in paired["frozen"]["records"]]
            differences.append(np.array([r["error"] for r in paired["frozen"]["records"]])-np.array([r["error"] for r in paired["current"]["records"]]))
            for source in means:means[source].append(paired[source]["accuracy"])
        rng=np.random.default_rng(99590);differences=np.array(differences)
        draws=[float(np.mean([differences[seed,rng.integers(differences.shape[1],size=differences.shape[1])].mean() for seed in rng.integers(3,size=3)])) for _ in range(2000)]
        data["matched_comparison"]={"updates":endpoint,"candidates":64*endpoint,"accuracy":{s:float(np.mean(v)) for s,v in means.items()},
                                    "paired_gain":float(differences.mean()),"gain_ci95":np.quantile(draws,[.025,.975]).tolist()}
    for path in sorted((ROOT/"results/scaling_s4g/late_forecast").glob("[0-9]*.json")):
        row=json.loads(path.read_text());key=f"{row['seed']}_current"
        if key not in data["confirmation"]:continue
        observed={r["step"]:r["error"] for r in data["confirmation"][key]["history"]["evaluations"]}
        for gamma,predictions in row["predictions"].items():
            for prediction in predictions:
                step=prediction["step"]
                if step not in observed:continue
                data["late_forecasts"].append({"seed":row["seed"],"gamma":int(gamma),"step":step,"predicted_error":prediction["error"],
                    "observed_error":observed[step],"observed_over_prediction":observed[step]/prediction["error"],"prospective":step>row["already_observed_through"],
                    "written_at_uk":row["written_at_uk"]})
    for source in ("current","frozen"):
        values = [r["history"] for key,r in data["confirmation"].items() if key.endswith("_"+source)]
        if len(values) != 3: continue
        common = sorted(set.intersection(*({r["step"] for r in h["evaluations"]} for h in values)))
        means = np.array([[next(r["error"] for r in h["evaluations"] if r["step"]==step) for step in common] for h in values]).mean(0)
        data["fits"][source] = {family:family_grid(64*np.array(common),means,family)[0] for family in ("power","exponential")}
    if len(current) == 3:
        common = sorted(set.intersection(*({r["step"] for r in x["history"]["evaluations"]} for x in current)))
        arrays = np.array([[[r["error"] for r in next(v for v in h["history"]["evaluations"] if v["step"]==step)["records"]] for step in common] for h in current])
        mean = arrays.mean(axis=(0,2)); rng = np.random.default_rng(99399)
        boot = []
        for _ in range(2000):
            chosen = rng.integers(3,size=3)
            boot.append(np.mean([arrays[seed][:,rng.integers(arrays.shape[2],size=arrays.shape[2])].mean(axis=1) for seed in chosen],axis=0))
        boot = np.array(boot)
        summaries=[]
        for start,end in ((128,1024),(256,2048),(512,2048),(1024,2048)):
            if end > common[-1]: continue
            select = (np.array(common)>=start)&(np.array(common)<=end)
            times = np.array(common)[select]
            beta = slope(times,mean[select]); uncertainty = [slope(times,row[select]) for row in boot]
            summaries.append({"start":start,"end":end,"beta":beta,"ci95":np.quantile(uncertainty,[.025,.975]).tolist()})
        data["bootstrap"] = {"method":"2000 hierarchical resamples of three training seeds and whole held-out questions, preserving time pairing.",
                             "common_steps":common,"mean_error":mean.tolist(),"error_ci95":np.quantile(boot,[.025,.975],axis=0).tolist(),"windows":summaries}
    fig,ax=plt.subplots()
    for source,color,style in (("current","C0","-"),("frozen","C1","--")):
        values=[r for key,r in data["confirmation"].items() if key.endswith("_"+source)]
        for index,result in enumerate(values):
            rows=[r for r in result["history"]["evaluations"] if r["step"]>0]
            ax.loglog([64*r["step"] for r in rows],[r["error"] for r in rows],style,color=color,alpha=.6,label=f"{source}: {len(values)} fresh seeds" if index==0 else None)
    ax.set(xlabel="Training candidate tokens",ylabel="Held-out error probability",title="Full-weight Qwen2.5-1.5B: counting")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-learning.png");plt.close(fig)
    fig,ax=plt.subplots()
    for index,result in enumerate(current):
        rows=[r for r in result["history"]["evaluations"] if r["step"]>0]
        ax.semilogx([64*r["step"] for r in rows],[r["accuracy"] for r in rows],color="C0",alpha=.6,label="Sampled accuracy" if index==0 else None)
        ax.semilogx([64*r["step"] for r in rows],[r["greedy_accuracy"] for r in rows],"--",color="C1",alpha=.6,label="Greedy accuracy" if index==0 else None)
    ax.set(xlabel="Training candidate tokens",ylabel="Held-out accuracy",title="Decision improvement versus confidence sharpening")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-greedy.png");plt.close(fig)
    fig,ax=plt.subplots()
    for label,color in (("0","C0"),("1","C1"),("2","C2")):
        for index,result in enumerate(current):
            rows=[r for r in result["by_answer"][label] if r["step"]>0]
            ax.loglog([64*r["step"] for r in rows],[r["error"] for r in rows],color=color,alpha=.6,label=f"Correct count {label}" if index==0 else None)
    ax.set(xlabel="Training candidate tokens",ylabel="Held-out error within each answer group",title="Different counting cases improve at different times")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-subtasks.png");plt.close(fig)
    fig,ax=plt.subplots()
    for result in current:
        h=result["history"];lookup={r["step"]:r["error"] for r in h["evaluations"]}
        ends=[n for n in (64,128,256,512,1024,2048) if n in lookup and n//2 in lookup]
        ax.semilogx([64*n for n in ends],[-np.log(lookup[n]/lookup[n//2])/np.log(2) for n in ends],"o-",label=f"Seed {h['seed']}")
    ax.axhline(.5,color="gray",linestyle="--",label="RBM population exponent 1/2")
    ax.axhline(1.,color="gray",linestyle=":",label="One-gate exponent 1")
    ax.set(xlabel="Candidate budget at end of each doubling",ylabel="Local error exponent",title="A scaling law needs a stable slope")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-exponents.png");plt.close(fig)
    fig,ax=plt.subplots()
    for result in current:
        rows=[r for r in result["geometry"] if r["step"]>0]
        if not rows: continue
        line,=ax.loglog([r["step"] for r in rows],[r["cubic_coefficient"] for r in rows],"o-",label=f"Seed {result['history']['seed']}")
        anchors=[r for r in rows if r["step"]==128]
        if anchors: ax.axhline(anchors[0]["cubic_coefficient"],color=line.get_color(),linestyle="--",alpha=.5)
    ax.set(xlabel="SGD updates",ylabel="Training/test gradient overlap ÷ error³",title="The RBM reduction requires this ratio to stabilize")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-geometry.png");plt.close(fig)
    fig,ax=plt.subplots()
    for result in current:
        rows=[r for r in result["geometry"] if r["step"]>0]
        if not rows: continue
        line,=ax.semilogx([r["step"] for r in rows],[r["weight_overlap"]["cosine"] for r in rows],"o-",label=f"Seed {result['history']['seed']}: displacement")
        ax.semilogx([r["step"] for r in rows],[r["eval_initial_direction_cosine"] for r in rows],"--",color=line.get_color(),label=f"Seed {result['history']['seed']}: gradient")
    ax.set(xlabel="SGD updates",ylabel="Cosine with initial held-out success gradient",title="The learning direction changes")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-weights.png");plt.close(fig)
    fig,ax=plt.subplots()
    for result in current:
        if "anchored_forecasts" not in result: continue
        rows=result["anchored_forecasts"]["3"]
        if not rows: continue
        line,=ax.loglog([64*r["step"] for r in rows],[r["actual_error"] for r in rows],"o-",label=f"Seed {result['history']['seed']}")
        ax.loglog([64*r["step"] for r in rows],[r["predicted_error"] for r in rows],"--",color=line.get_color())
    ax.set(xlabel="Training candidate tokens",ylabel="Held-out error probability",title="Solid: LLM; dashed: RBM forecast from update 128")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-forecasts.png");plt.close(fig)
    fig,ax=plt.subplots()
    for seed in (99201,99202,99203):
        rows=[r for r in data["late_forecasts"] if r["seed"]==seed and r["gamma"]==2 and r["prospective"]]
        if not rows:continue
        line,=ax.loglog([r["step"] for r in rows],[r["observed_error"] for r in rows],"o-",label=f"Seed {seed}")
        ax.loglog([r["step"] for r in rows],[r["predicted_error"] for r in rows],"--",color=line.get_color())
    ax.set(xlabel="SGD updates",ylabel="Held-out error probability",title="Solid: observations; dashed: late quadratic forecast")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-llm-late-forecasts.png");plt.close(fig)
    save(root/"analysis.json",data)
    print(json.dumps({"updated_at_uk":data["updated_at_uk"],"confirmation":{k:{"step":r["history"]["step"],"complete":r["complete"],"accuracy":r["history"]["evaluations"][-1]["accuracy"]} for k,r in data["confirmation"].items()},"windows":data["bootstrap"]},indent=2))


if __name__ == "__main__": main()
