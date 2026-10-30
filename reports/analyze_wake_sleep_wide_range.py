"""Audit the wider budgets, separating transient slopes from stopped models."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("elbo", "sleep", "forward")
LABELS = {"elbo": "ELBO", "sleep": "ELBO + sleep", "forward": "Posterior teaching"}
COLORS = dict(zip(ARMS, ("C0", "C1", "C2"), strict=True))


def read(path): return json.loads(path.read_text())


def interpolate(rows, grid, cost, metric):
    positive = [r for r in rows if r[cost] > 0]
    return np.interp(np.log(grid), np.log([r[cost] for r in positive]), [r[metric] for r in positive])


def draw(ax, runs, grid, cost, metric, arm, complement=False):
    ys = np.stack([interpolate(r["rows"], grid, cost, metric) for r in runs])
    if complement: ys = 1 - ys
    mean, se = ys.mean(0), ys.std(0, ddof=1) / np.sqrt(len(ys))
    observed = grid <= min(r["rows"][-1][cost] for r in runs)
    ax.plot(grid[observed], mean[observed], color=COLORS[arm], label=LABELS[arm])
    tail = np.flatnonzero(~observed)
    if len(tail):
        start = max(0, tail[0] - 1)
        ax.plot(grid[start:], mean[start:], color=COLORS[arm], linestyle="--")
    ax.fill_between(grid, np.maximum(mean-se, np.finfo(float).tiny), mean+se, color=COLORS[arm], alpha=.15)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=.2)
    return ys


def load_llm(arm, seed):
    directory = ROOT / "results/llm_wake_sleep"
    old = directory / f"confirmation_{arm}_s{seed}"
    extended = directory / f"extended_{arm}_s{seed}"
    result = read(extended / "run.json")
    assert hashlib.sha256((extended / "runtime.py").read_bytes()).hexdigest() == result["protocol"]["source_sha256"]
    assert result["rows"][-1]["step"] == 4096 or result["stop_reason"] != "budget", f"Still running: {extended}"
    resume = read(extended / "resume.json")
    assert resume["step"] == result["rows"][-1]["step"] and Path(resume["persistent"]).exists()
    by_step = {r["step"]: r for r in result["rows"]}
    evaluations = {step:read((old if step < 512 else extended) / f'eval_{step}.json') for step in by_step}
    matched = read(directory / f"wide_matched_{arm}_s{seed}.json")
    step = matched["step"]
    if step not in by_step: by_step[step] = {"step":step, "flops":matched["actual_flops"]}
    by_step[step].update({k:v for k,v in matched["two_examples"].items() if k!="records"})
    evaluations[step] = matched["two_examples"]
    rows = [by_step[step] for step in sorted(by_step)]
    for row in rows:
        evaluation = evaluations[row["step"]]
        q = np.array([v["q"] for v in evaluation["records"]])
        p = np.array([v["model_posterior"] for v in evaluation["records"]])
        assert np.all(q > 0) and np.all(p > 0)
        assert np.allclose(q.sum(1), 1, atol=1e-6) and np.allclose(p.sum(1), 1, atol=1e-6)
        row["inference_forward_kl"] = float(np.mean(np.sum(p * np.log(p/q), axis=1)))
        row["neglected_model_mass"] = float(np.mean(np.sum(p * (q < .1*p), axis=1)))
        row["rule_error"] = [float(np.mean([1-v["compatible_mass"] for v in evaluation["records"] if v["latent"]==z])) for z in range(8)]
    result["rows"] = rows
    return result


def llm_analysis():
    seeds = (9610, 9611, 9612)
    with ThreadPoolExecutor(max_workers=9) as pool:
        values = list(tqdm(pool.map(lambda pair: load_llm(*pair), [(a,s) for a in ARMS for s in seeds]), total=9, desc="Extended LLM audit"))
    runs = {a: [r for r in values if r["protocol"]["arm"] == a] for a in ARMS}
    unit = runs["elbo"][0]["rows"][1]["flops"]
    grid = np.geomspace(2*unit, 4096*unit, 200)
    result = {"seeds": list(seeds), "baseline_flops_per_update": unit, "old_budget": 512*unit, "new_allocated_budget": 4096*unit,
              "policy": "Identical continuation from update 512; early-stopped weights held fixed only for allocated-budget comparisons. Dashed curves include such retention; no slopes fitted beyond actual observations.",
              "runs": {}, "budgets": {}, "effective_slopes": {}, "curves": {}}
    fig, axes = plt.subplots(1, 2, layout="constrained")
    for arm in ARMS:
        errors = draw(axes[0], runs[arm], grid, "flops", "compatible_mass", arm, True)
        gaps = draw(axes[1], runs[arm], grid, "flops", "inference_forward_kl", arm)
        result["curves"][arm] = {"flops":grid.tolist(), "probability_error":errors.tolist(), "model_inference_kl":gaps.tolist(),
                                  "neglected_model_mass":[interpolate(r["rows"],grid,"flops","neglected_model_mass").tolist() for r in runs[arm]]}
        result["runs"][arm] = [{"seed": r["protocol"]["seed"], "stop_reason": r["stop_reason"], "final": r["rows"][-1]} for r in runs[arm]]
        result["effective_slopes"][arm] = {}
        for low, high in ((32,128), (64,256), (128,512), (256,1024), (512,2048), (1024,4096)):
            if high*unit > min(r["rows"][-1]["flops"] for r in runs[arm]):
                result["effective_slopes"][arm][f"{low}-{high}"] = {"status": "Incomplete observed range; no exponent fitted"}
                continue
            x = np.geomspace(low*unit, high*unit, 32)
            exponents = {}
            for metric in ("compatible_mass", "inference_forward_kl", "posterior_kl"):
                y = np.stack([interpolate(r["rows"], x, "flops", metric) for r in runs[arm]])
                if metric == "compatible_mass": y = 1-y
                exponents[metric] = {"mean_curve": float(-np.polyfit(np.log(x), np.log(y.mean(0)), 1)[0]),
                                     "per_seed": [-float(np.polyfit(np.log(x), np.log(v), 1)[0]) for v in y]}
            result["effective_slopes"][arm][f"{low}-{high}"] = {"status": "Fully observed", "exponents": exponents}
    for ax in axes:
        ax.axvline(512*unit, color="gray", linestyle=":"); ax.set_xlabel("Allocated training FLOPs (approx.)")
    axes[0].set_ylabel("Probability of incompatible rule")
    axes[1].set_ylabel("KL(model posterior || recognition)")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Qwen2.5-1.5B: extending the old budget (dotted)")
    fig.savefig(ROOT / "reports/llm-wake-sleep-wide-range.png"); plt.close(fig)
    for updates in (128,256,512,1024,2048,4096):
        c = updates*unit
        result["budgets"][str(updates)] = {}
        for arm in ARMS:
            result["budgets"][str(updates)][arm] = {metric: [float(interpolate(r["rows"], [c], "flops", metric)[0]) for r in runs[arm]]
                                                  for metric in ("compatible_mass", "accuracy", "posterior_kl", "inference_forward_kl")}
    fig, ax = plt.subplots(layout="constrained")
    for arm in ARMS:
        draw(ax, runs[arm], grid, "flops", "posterior_kl", arm)
    ax.axvline(512*unit, color="gray", linestyle=":"); ax.set_xlabel("Allocated training FLOPs (approx.)"); ax.set_ylabel("True-posterior KL")
    ax.set_title("Same external task; three optimizer seeds"); ax.legend()
    fig.savefig(ROOT / "reports/llm-wake-sleep-wide-task-kl.png"); plt.close(fig)
    fig, axes = plt.subplots(1,2,layout="constrained",sharey=True)
    for ax, arm in zip(axes,("elbo","sleep"),strict=True):
        errors = []
        for run in runs[arm]:
            observed = [r for r in run["rows"] if r["flops"]>0]
            x = np.log([r["flops"] for r in observed])
            errors.append([np.interp(np.log(grid),x,[r["rule_error"][z] for r in observed]) for z in range(8)])
        mean=np.mean(errors,axis=0)
        mesh=ax.pcolormesh(grid/unit,np.arange(8),mean,shading="nearest",vmin=0,vmax=1,cmap="viridis")
        ax.set_xscale("log");ax.axvline(512,color="white",linestyle=":");ax.set_title(LABELS[arm])
    axes[0].set_yticks(np.arange(8),["add 1","add 2","add 3","subtract 1","subtract 2","subtract 3","multiply by 2","multiply by 3"])
    fig.colorbar(mesh,ax=axes,label="Probability of incompatible rule")
    fig.supxlabel("Allocated wake-equivalent updates")
    fig.suptitle("Do different causes recover at different times?")
    fig.savefig(ROOT / "reports/llm-wake-sleep-rule-recovery.png");plt.close(fig)
    fig,axes=plt.subplots(1,2,layout="constrained",sharey=True)
    for arm in ARMS:
        draw(axes[0],runs[arm],grid,"flops","generator_correct_probability",arm,True)
        draw(axes[1],runs[arm],np.geomspace(2,4096,200),"step","generator_correct_probability",arm,True)
    axes[0].set_xlabel("Training FLOPs (approx.)");axes[1].set_xlabel("Generator updates")
    axes[0].set_ylabel("Generator probability of wrong digit")
    axes[0].set_title("Equal compute");axes[1].set_title("Equal update count")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=3)
    fig.suptitle("Later training also improves the generator")
    fig.savefig(ROOT / "reports/llm-wake-sleep-generator-progress.png");plt.close(fig)
    (ROOT / "results/llm_wake_sleep/wide_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"llm": {"budgets":result["budgets"], "effective_slopes": result["effective_slopes"]}}, indent=2))


def load_toy(stage):
    path = ROOT / "results/deep_wake_sleep" / stage
    protocol = read(path / "protocol.json")
    assert hashlib.sha256((path / "runtime.py").read_bytes()).hexdigest() == (path / "runtime.sha256").read_text()
    results = []
    for case in protocol["cases"]:
        run = read(path / case["name"] / "run.json")
        assert (path / case["name"] / "weights.npz").exists(), f"Still running: {case['name']}"
        results.append(run)
    return results


def toy_analysis():
    groups = {stage: load_toy(stage) for stage in ("wide_error_range", "wide_rate_control")}
    result = {"policy": "Six preselected seeds. High rates from the earlier KL .1 tuning; second protocol reduces all rates tenfold at K=256. Stops and target censoring are retained. No extrapolated target time is fitted.", "groups": {}}
    fig, axes = plt.subplots(1, 2, layout="constrained", sharey=True)
    for ax, (stage, values), title in zip(axes, groups.items(), ("Original learning rates", "Tenfold lower rates"), strict=True):
        selected = [r for r in values if r["config"]["layers"] == [4,4]]
        unit = selected[0]["costs"]["wake"]
        grid = np.geomspace(2*unit, 262144*unit, 180)
        for arm in ARMS:
            runs = [r for r in selected if r["config"]["arm"] == arm]
            draw(ax, runs, grid, "compute", "posterior_kl", arm)
        ax.axhline(.03, color="gray", linestyle=":"); ax.set_xlabel("Allocated training FLOPs (approx.)"); ax.set_title(title)
        result["groups"][stage] = {}
        for bits in sorted({sum(r["config"]["layers"]) for r in values}):
            result["groups"][stage][str(2**bits)] = {}
            for arm in ARMS:
                runs = sorted([r for r in values if sum(r["config"]["layers"])==bits and r["config"]["arm"]==arm], key=lambda r:r["config"]["seed"])
                info = {"final_errors": [r["rows"][-1]["posterior_kl"] for r in runs],
                        "lowest_observed_errors": [min(v["posterior_kl"] for v in r["rows"]) for r in runs],
                        "stop_reasons": [r["stop_reason"] for r in runs], "actual_final_compute": [r["rows"][-1]["compute"] for r in runs],
                        "lr": runs[0]["config"]["lr_r"], "targets": {}}
                for target in (.1,.03,.01,.001,.0001,1e-5,1e-6,1e-7,1e-8,1e-9):
                    hits = [[v for v in r["rows"] if v["posterior_kl"] <= target] for r in runs]
                    times = [h[0]["compute"] if h else None for h in hits]
                    info["targets"][str(target)] = {"costs": times, "reached": sum(t is not None for t in times),
                                                  "mean_if_all_reached": float(np.mean(times)) if all(t is not None for t in times) else None}
                result["groups"][stage][str(2**bits)][arm] = info
    axes[0].set_ylabel("Posterior KL; 256 latent states")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("Toy: extending past the old error target (dotted)")
    fig.savefig(ROOT / "reports/deep-wake-sleep-wide-range.png"); plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for stage, style in (("wide_error_range", "-"), ("wide_rate_control", "--")):
        for arm in ARMS:
            info = result["groups"][stage]["256"][arm]
            reached = [(float(t), v["mean_if_all_reached"]) for t,v in info["targets"].items() if v["reached"] == 6]
            ax.plot([a for a,b in reached], [b for a,b in reached], color=COLORS[arm], linestyle=style, marker="o", label=f'{LABELS[arm]}, η={info["lr"]:g}')
    ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis(); ax.set_xlabel("Target posterior KL"); ax.set_ylabel("Mean actual FLOPs to first reach target")
    ax.set_title("Only targets reached by all six seeds are shown"); ax.legend(); ax.grid(True, alpha=.2)
    fig.savefig(ROOT / "reports/deep-wake-sleep-wide-targets.png"); plt.close(fig)
    (ROOT / "results/deep_wake_sleep/wide_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"toy": result}, indent=2))


def epsilon_analysis():
    values = load_toy("wide_epsilon_control")
    older = ROOT / "results/deep_wake_sleep/size_replication"
    values += [read(older / f"b{bits}_g2_{arm}_s{seed}/run.json") for bits in (4,6,8,10) for arm in ("elbo","sleep") for seed in range(9540,9546)]
    result = {"seeds": list(range(9540,9546)), "sizes": [16,64,256,1024,4096], "groups": {},
              "scope": "Numerical-parameter robustness and a larger-size check, not evidence for a universal law. Earlier-range and descriptive full-range fits are both reported; full-range fits require every target attained. Old default-epsilon comparisons use a slightly coarser evaluation grid."}
    samples=np.random.default_rng(9599).integers(6,size=(4000,6))
    fig, axes = plt.subplots(1,2,layout="constrained",sharey=True)
    for ax, optimizer, title in zip(axes, ("adam","adam_eps12"), ("Adam ε = 10⁻⁸", "Adam ε = 10⁻¹²"), strict=True):
        result["groups"][optimizer] = {}
        for arm in ("elbo","sleep"):
            records = []
            for size in result["sizes"]:
                runs = sorted([r for r in values if 2**sum(r["config"]["layers"])==size and r["config"]["arm"]==arm and r["config"]["optimizer"]==optimizer], key=lambda r:r["config"]["seed"])
                assert len(runs)==6
                hits = [[v for v in r["rows"] if v["posterior_kl"]<=.1] for r in runs]
                costs = [h[0]["compute"] if h else None for h in hits]
                bounds = [h[0]["compute"] if h else r["rows"][-1]["compute"] for h,r in zip(hits,runs,strict=True)]
                records.append({"size":size,"costs":costs,"reached":sum(c is not None for c in costs),"mean_lower_bound":float(np.mean(bounds)),
                                "mean_cost":float(np.mean(costs)) if all(c is not None for c in costs) else None})
            matrix = np.array([r["costs"] for r in records[:4]])
            slope = float(np.polyfit(np.log(result["sizes"][:4]), np.log(matrix.mean(1)), 1)[0])
            info = {"records":records,"exponent_16_to_1024":slope}
            result["groups"][optimizer][arm] = info
            if all(r["reached"]==6 for r in records):
                costs=np.array([r["costs"] for r in records]);x=np.log(result["sizes"])
                info["exponent_full_range"]=float(np.polyfit(x,np.log(costs.mean(1)),1)[0])
                bootstrap=[np.polyfit(x,np.log(costs[:,s].mean(1)),1)[0] for s in samples]
                info["full_range_seed_ci"]=np.quantile(bootstrap,[.025,.975]).tolist()
                info["adjacent_slopes"]=(np.diff(np.log(costs.mean(1)))/np.diff(x)).tolist()
            ax.errorbar([r["size"] for r in records[:4]], matrix.mean(1), yerr=matrix.std(1,ddof=1)/np.sqrt(6), color=COLORS[arm], marker="o",label=LABELS[arm])
            final=records[-1]
            if final["mean_cost"] is None:
                ax.scatter([4096],[final["mean_lower_bound"]],color=COLORS[arm],marker="^")
            else:
                ax.plot([1024,4096],[records[-2]["mean_cost"],final["mean_cost"]],color=COLORS[arm],marker="o")
                ax.errorbar([4096],[final["mean_cost"]],yerr=[np.std(final["costs"],ddof=1)/np.sqrt(6)],color=COLORS[arm])
        ax.set_title(title);ax.set_xlabel("Latent configurations");ax.set_xscale("log");ax.set_yscale("log");ax.grid(True,alpha=.2);ax.legend()
    axes[0].set_ylabel("FLOPs to posterior KL ≤ 0.1")
    fig.suptitle("Six-seed size extension and Adam stabilizer")
    fig.savefig(ROOT / "reports/deep-wake-sleep-wide-sizes.png");plt.close(fig)
    (ROOT / "results/deep_wake_sleep/wide_epsilon_analysis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


def expanded_replication_analysis():
    original = load_toy("size_replication")
    extra = load_toy("wide_size_replication")
    audited = [r for r in load_toy("wide_epsilon_control") if r["config"]["optimizer"]=="adam" and r["config"]["layers"]==[6,6]]
    runs = original+audited+extra
    assert len(runs)==160
    samples = np.random.default_rng(9597).integers(16,size=(4000,16))
    sizes = [16,64,256,1024,4096]
    result = {"seeds":list(range(9540,9556)),"sizes":sizes,"groups":{},
              "scope":"All original sixteen seeds, with the two-layer fixed generator, affine recognition and moderate neglect unchanged. New largest-size cases; descriptive cost-to-fixed-KL scaling, not error-versus-compute scaling. Preserve unattained targets as censored."}
    matrices={}
    fig,ax=plt.subplots(layout="constrained")
    for arm in ("elbo","sleep"):
        points=[]
        for size in sizes:
            group=sorted([r for r in runs if r["config"]["arm"]==arm and 2**sum(r["config"]["layers"])==size],key=lambda r:r["config"]["seed"])
            assert len(group)==16
            hits=[[v for v in r["rows"] if v["posterior_kl"]<=.1] for r in group]
            costs=[h[0]["compute"] if h else None for h in hits]
            bounds=[h[0]["compute"] if h else r["rows"][-1]["compute"] for h,r in zip(hits,group,strict=True)]
            points.append({"size":size,"costs":costs,"mean_lower_bound":float(np.mean(bounds)),"reached":sum(c is not None for c in costs),
                           "mean_cost":float(np.mean(costs)) if all(c is not None for c in costs) else None,
                           "median_cost":float(np.median(costs)) if all(c is not None for c in costs) else None})
        info={"points":points};result["groups"][arm]=info
        if all(p["reached"]==16 for p in points):
            a=np.array([p["costs"] for p in points]);matrices[arm]=a
            x=np.log(sizes);slope=float(np.polyfit(x,np.log(a.mean(1)),1)[0])
            slopes=np.array([np.polyfit(x,np.log(a[:,s].mean(1)),1)[0] for s in samples])
            info.update(exponent=slope,exponent_seed_ci=np.quantile(slopes,[.025,.975]).tolist(),adjacent_slopes=(np.diff(np.log(a.mean(1)))/np.diff(x)).tolist(),
                        status="All sixteen seeds reached every target",bootstrap_slopes=slopes.tolist())
            ax.errorbar(sizes,a.mean(1),yerr=a.std(1,ddof=1)/4,color=COLORS[arm],marker="o",label=f'{LABELS[arm]} (fit {slope:.2f})')
        else:
            info["status"]="Censored; no full-range unconditional mean exponent"
            complete=[p for p in points if p["reached"]==16]
            ax.errorbar([p["size"] for p in complete],[p["mean_cost"] for p in complete],yerr=[np.std(p["costs"],ddof=1)/4 for p in complete],color=COLORS[arm],marker="o",label=LABELS[arm])
            for point in points:
                if point["reached"]<16: ax.scatter([point["size"]],[point["mean_lower_bound"]],color=COLORS[arm],marker="^")
    e,s=result["groups"]["elbo"],result["groups"]["sleep"]
    if all(g["status"]=="All sixteen seeds reached every target" for g in (e,s)):
        result["exponent_gap_seed_ci"]=np.quantile(np.array(e["bootstrap_slopes"])-np.array(s["bootstrap_slopes"]),[.025,.975]).tolist()
        ratio=matrices["elbo"][-1,samples].mean(1)/matrices["sleep"][-1,samples].mean(1)
        result["largest_mean_cost_ratio"]=e["points"][-1]["mean_cost"]/s["points"][-1]["mean_cost"]
        result["largest_median_cost_ratio"]=e["points"][-1]["median_cost"]/s["points"][-1]["median_cost"]
        result["largest_ratio_seed_ci"]=np.quantile(ratio,[.025,.975]).tolist()
        for g in (e,s): del g["bootstrap_slopes"]
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Latent configurations");ax.set_ylabel("FLOPs to posterior KL ≤ 0.1")
    ax.set_title("Larger latent spaces: all sixteen original seeds");ax.legend();ax.grid(True,alpha=.2)
    fig.savefig(ROOT/"reports/deep-wake-sleep-expanded-replication.png");plt.close(fig)
    (ROOT/"results/deep_wake_sleep/expanded_size_analysis.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--only", choices=("toy", "llm", "epsilon", "sizes")); args = parser.parse_args()
    if args.only in (None,"toy"): toy_analysis()
    if args.only in (None,"llm"): llm_analysis()
    if args.only in (None,"epsilon"): epsilon_analysis()
    if args.only in (None,"sizes"): expanded_replication_analysis()
