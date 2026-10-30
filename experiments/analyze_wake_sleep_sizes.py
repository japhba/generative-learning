"""Check frozen size forecasts and independent-initialization replication."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_deep_wake_sleep import read_stage, RESULTS, ROOT, COLORS, LABELS


def target(run, threshold=.1):
    hit = [row for row in run["rows"] if row["posterior_kl"] <= threshold]
    return float(hit[0]["compute"]) if hit else None


def summary(runs, samples):
    sizes = sorted({2**sum(r["config"]["layers"]) for r in runs})
    matrix = []
    for size in sizes:
        group = sorted([r for r in runs if 2**sum(r["config"]["layers"]) == size], key=lambda r:r["config"]["seed"])
        matrix.append([target(r) for r in group])
    if any(value is None for row in matrix for value in row): return {"sizes": sizes, "times": matrix, "status": "censored; no unconditional mean exponent fitted"}
    a = np.array(matrix)
    x = np.log(sizes)
    slope, intercept = np.polyfit(x,np.log(a.mean(1)),1)
    slopes = [np.polyfit(x,np.log(a[:,s].mean(1)),1)[0] for s in samples]
    return {"sizes": sizes, "times": matrix, "mean_cost": a.mean(1).tolist(), "median_cost": np.median(a,axis=1).tolist(),
            "exponent": float(slope), "intercept": float(intercept), "exponent_ci": np.quantile(slopes,[.025,.975]).tolist(),
            "adjacent_slopes": (np.diff(np.log(a.mean(1)))/np.diff(x)).tolist(), "bootstrap_slopes": np.array(slopes).tolist(), "status": "all reached target"}


def main():
    controls = read_stage("state_scaling_controls")
    extrap = read_stage("state_extrapolation")
    replication = read_stage("size_replication")
    forecast = json.loads((RESULTS / "size_forecast.json").read_text())
    rng = np.random.default_rng(9598)
    b3, b16 = rng.integers(3,size=(4000,3)), rng.integers(16,size=(4000,16))
    result = {"forecast": forecast, "original": {}, "prospective_check": {}, "replication": {}, "zero_neglect": {}}
    for arm in ("elbo","sleep"):
        result["original"][arm] = summary([r for r in controls if r["config"]["neglect"]==3 and len(r["config"]["layers"])==2 and r["config"]["arm"]==arm], b3)
        future_runs = [r for r in extrap if r["config"]["arm"]==arm]
        future = [target(r) for r in future_runs]
        bounds = [r["config"]["budget"]*r["costs"]["wake"] if t is None else t for r,t in zip(future_runs,future,strict=True)]
        predicted = forecast["forecasts"][arm]["predicted_flops_at_4096"]
        measured = None if any(t is None for t in future) else float(np.mean(future))
        result["prospective_check"][arm] = {"per_seed_cost": future, "mean_cost": measured, "mean_lower_bound": float(np.mean(bounds)), "predicted_cost": predicted,
                                          "observed_over_predicted": None if measured is None else measured/predicted, "lower_bound_over_predicted": float(np.mean(bounds))/predicted}
        result["replication"][arm] = summary([r for r in replication if r["config"]["arm"]==arm], b16)
        result["zero_neglect"][arm] = summary([r for r in controls if r["config"]["neglect"]==0 and len(r["config"]["layers"])==2 and r["config"]["arm"]==arm], b3)
    if all(result["replication"][a]["status"]=="all reached target" for a in ("elbo","sleep")):
        delta = np.array(result["replication"]["elbo"]["bootstrap_slopes"]) - result["replication"]["sleep"]["bootstrap_slopes"]
        result["replicated_exponent_difference_ci"] = np.quantile(delta,[.025,.975]).tolist()
    fig, axs = plt.subplots(1,2,layout="constrained")
    for ax, name, title in zip(axs,("replication","zero_neglect"),("Moderate initial neglect","Diffuse initial recognition"),strict=True):
        for arm in ("elbo","sleep"):
            r=result[name][arm]
            if r["status"]!="all reached target": continue
            a=np.array(r["times"]); mean=a.mean(1); se=a.std(1,ddof=1)/np.sqrt(a.shape[1])
            ax.errorbar(r["sizes"],mean,yerr=se,marker="o",color=COLORS[arm],label=f'{LABELS[arm]} ({r["exponent"]:.2f})')
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Latent configurations"); ax.set_ylabel("FLOPs to posterior KL ≤ 0.1"); ax.set_title(title); ax.grid(True,alpha=.2)
        ax.legend()
    fig.suptitle("Problem-size scaling; fixed two-layer generator")
    fig.savefig(ROOT/"reports/deep-wake-sleep-size-scaling.png"); plt.close(fig)
    fig,ax=plt.subplots(layout="constrained")
    for arm in ("elbo","sleep"):
        r=result["original"][arm]; pred=result["prospective_check"][arm]
        x=np.geomspace(16,4096,100)
        ax.plot(x,np.exp(r["intercept"])*x**r["exponent"],linestyle="--",color=COLORS[arm],label=f'{LABELS[arm]} forecast')
        ax.plot(r["sizes"],r["mean_cost"],linestyle="none",marker="o",color=COLORS[arm])
        ax.scatter([4096],[pred["mean_lower_bound"]],marker="^" if pred["mean_cost"] is None else "D",color=COLORS[arm])
    ax.axvline(1024,color="gray",linestyle=":"); ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Latent configurations");ax.set_ylabel("FLOPs to posterior KL ≤ 0.1")
    ax.set_title("4,096-state check; triangle is a lower bound");ax.legend();ax.grid(True,alpha=.2)
    fig.savefig(ROOT/"reports/deep-wake-sleep-forecast.png");plt.close(fig)
    for name in ("original","replication","zero_neglect"):
        for arm in ("elbo","sleep"):
            if "bootstrap_slopes" in result[name][arm]: del result[name][arm]["bootstrap_slopes"]
    (RESULTS/"size_analysis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
