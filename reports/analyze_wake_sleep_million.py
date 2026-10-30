"""Nine-size GPU study with paired seeds, censored targets and window checks."""

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
SEEDS = list(range(9540, 9546))
SIZES = [2**b for b in range(4, 21, 2)]
ARMS = ("elbo", "sleep")
OPTIMIZERS = ("adam", "adam_eps12")
LABELS = {"elbo": "ELBO", "sleep": "ELBO + sleep"}
COLORS = {"elbo": "C0", "sleep": "C1"}


def read(path): return json.loads(path.read_text())


def load_stage(stage, in_progress):
    directory = ROOT / "results/deep_wake_sleep" / stage
    protocol = read(directory / "protocol.json")
    assert hashlib.sha256((directory / "runtime.py").read_bytes()).hexdigest() == protocol["runtime_sha256"]
    assert hashlib.sha256((directory / "reference.py").read_bytes()).hexdigest() == protocol["reference_sha256"]
    def load(case):
        path = directory / case["name"]
        result = read(path / "run.json")
        assert result["config"] == {**protocol["base"], **case}
        assert in_progress or result["stop_reason"] != "running", path
        resume = read(path / "resume.json")
        assert Path(resume["persistent"]).exists()
        if result["stop_reason"] == "running": assert resume["step"] <= result["rows"][-1]["step"]
        else: assert resume["step"] == result["rows"][-1]["step"]
        result["path"] = str(path.relative_to(ROOT))
        return result
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(tqdm(pool.map(load, protocol["cases"]), total=len(protocol["cases"]), desc=stage))


def regression(x, curves):
    centered = x - x.mean()
    return centered @ np.log(curves) / (centered @ centered)


def summarize(runs, metric, samples, target, criterion="posterior_kl"):
    points = []
    for size in SIZES:
        group = sorted([r for r in runs if 2**sum(r["config"]["layers"]) == size], key=lambda r: r["config"]["seed"])
        assert [r["config"]["seed"] for r in group] == SEEDS
        direction = 1 if criterion == "recognition_weight_overlap" else -1
        hits = [[row for row in r["rows"] if direction * row[criterion] >= direction * target] for r in group]
        costs = [rows[0][metric] if rows else None for rows in hits]
        points.append({"size": size, "values": costs, "reached": sum(v is not None for v in costs),
                       "lower_bounds": [h[0][metric] if h else r["rows"][-1][metric] for h, r in zip(hits, group, strict=True)],
                       "stop_reasons": [r["stop_reason"] for r in group], "final_kl": [r["rows"][-1]["posterior_kl"] for r in group],
                       "initial_kl": [r["rows"][0]["posterior_kl"] for r in group], "paths": [r["path"] for r in group]})
    result = {"criterion": criterion, "target": target, "points": points, "fits": {}}
    for low, high in ((16, 4096), (16, 1048576), (4096, 1048576), (65536, 1048576)):
        selected = [p for p in points if low <= p["size"] <= high]
        key = f"{low}-{high}"
        if any(p["reached"] != 6 for p in selected):
            result["fits"][key] = {"status": "Censored: no unconditional mean exponent fitted"}
            continue
        matrix = np.array([p["values"] for p in selected])
        x = np.log([p["size"] for p in selected])
        bootstrap = regression(x, matrix[:, samples].mean(2))
        result["fits"][key] = {"status": "All targets attained", "mean_exponent": float(regression(x, matrix.mean(1))),
                              "mean_exponent_seed_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
                              "median_exponent": float(regression(x, np.median(matrix, axis=1)))}
    result["rolling_exponents"] = []
    for i in range(len(points) - 2):
        selected = points[i:i+3]
        if any(p["reached"] != 6 for p in selected): continue
        matrix = np.array([p["values"] for p in selected]); x = np.log([p["size"] for p in selected])
        result["rolling_exponents"].append({"low": selected[0]["size"], "high": selected[-1]["size"], "center": selected[1]["size"],
                                           "exponent": float(regression(x, matrix.mean(1))),
                                           "seed_ci": np.quantile(regression(x, matrix[:, samples].mean(2)), [.025, .975]).tolist()})
    return result


def points_plot(ax, info, arm, label, style="-"):
    complete = [p for p in info["points"] if p["reached"] == 6]
    ax.errorbar([p["size"] for p in complete], [np.mean(p["values"]) for p in complete],
                yerr=[np.std(p["values"], ddof=1) / np.sqrt(6) for p in complete], color=COLORS[arm], marker="o", linestyle=style, label=label)
    for p in info["points"]:
        if p["reached"] < 6: ax.scatter([p["size"]], [np.mean(p["lower_bounds"])], color=COLORS[arm], marker="^")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(True, alpha=.2)
    ax.axvline(4096, color="gray", linestyle=":")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--in-progress", action="store_true"); args = parser.parse_args()
    large = load_stage("million_size", args.in_progress)
    anchors = load_stage("million_anchors", args.in_progress)
    bridge = [r for r in large if r["config"]["name"].startswith("bridge")]
    runs = anchors + [r for r in large if not r["config"]["name"].startswith("bridge")]
    assert len(runs) == 216
    samples = np.random.default_rng(9873).integers(6, size=(4000, 6))
    result = {"seeds": SEEDS, "sizes": SIZES, "runs": len(runs), "bridge_runs": len(bridge), "running": sum(r["stop_reason"] == "running" for r in runs), "groups": {}, "largest_ratios": {}, "exponent_gaps": {},
              "scope": "Fixed useful two-layer sigmoid belief generator, affine full-state recognizer, exact population gradients, no sample selection. Cost to posterior KL <= .1 versus latent configuration count; not error versus compute or an LLM scaling experiment.",
              "accounting": "All nine sizes rerun using one float64 GPU implementation and a consistent dense evaluation schedule. Reference FLOPs retain previous uncached accounting; cached_dense_flops count recognition matrix products executed after caching the fixed generator. Setup, elementwise kernels and evaluations are excluded. Six paired seed bootstrap describes seed variation, not asymptotic-model uncertainty.",
              "old_numpy_results": "Preserved separately; not stitched to the GPU curve. First-hit costs can change with evaluation spacing, including occasional error dips.",
              "gpu_audit": read(ROOT / "results/deep_wake_sleep/gpu_audit.json")}
    for optimizer in OPTIMIZERS:
        result["groups"][optimizer] = {}
        for metric in ("compute", "cached_dense_flops", "step"):
            result["groups"][optimizer][metric] = {arm: summarize([r for r in runs if r["config"]["optimizer"] == optimizer and r["config"]["arm"] == arm], metric, samples, .1) for arm in ARMS}
        result["groups"][optimizer]["target_sensitivity"] = {str(target): {arm: summarize([r for r in runs if r["config"]["optimizer"] == optimizer and r["config"]["arm"] == arm], "cached_dense_flops", samples, target) for arm in ARMS} for target in (.2, .4)}
        result["groups"][optimizer]["weight_overlap"] = {str(target): {arm: summarize([r for r in runs if r["config"]["optimizer"] == optimizer and r["config"]["arm"] == arm], "cached_dense_flops", samples, target, "recognition_weight_overlap") for arm in ARMS} for target in (.5, .8)}
        result["largest_ratios"][optimizer] = {}
        for metric in ("compute", "cached_dense_flops", "step"):
            e, s = [result["groups"][optimizer][metric][arm]["points"][-1] for arm in ARMS]
            if min(e["reached"], s["reached"]) == 6:
                ev, sv = np.array(e["values"]), np.array(s["values"])
                result["largest_ratios"][optimizer][metric] = {"mean_ratio": float(ev.mean() / sv.mean()), "median_ratio": float(np.median(ev) / np.median(sv)),
                                                             "seed_ci": np.quantile(ev[samples].mean(1) / sv[samples].mean(1), [.025, .975]).tolist()}
        result["exponent_gaps"][optimizer] = {}
        for low, high in ((16, 4096), (16, 1048576), (4096, 1048576), (65536, 1048576)):
            selected = {arm: [p for p in result["groups"][optimizer]["cached_dense_flops"][arm]["points"] if low <= p["size"] <= high] for arm in ARMS}
            if any(p["reached"] != 6 for arm in ARMS for p in selected[arm]):
                result["exponent_gaps"][optimizer][f"{low}-{high}"] = {"status": "Censored: no exponent gap fitted"}
                continue
            x = np.log([p["size"] for p in selected["elbo"]])
            matrices = {arm: np.array([p["values"] for p in selected[arm]]) for arm in ARMS}
            gap = regression(x, matrices["elbo"].mean(1)) - regression(x, matrices["sleep"].mean(1))
            bootstrap = regression(x, matrices["elbo"][:, samples].mean(2)) - regression(x, matrices["sleep"][:, samples].mean(2))
            result["exponent_gaps"][optimizer][f"{low}-{high}"] = {"status": "All targets attained", "gap": float(gap), "paired_seed_ci": np.quantile(bootstrap, [.025, .975]).tolist()}
    for metric, filename, ylabel in (("compute", "deep-wake-sleep-million-scaling.png", "Reference FLOPs to posterior KL ≤ 0.1"),
                                    ("cached_dense_flops", "deep-wake-sleep-million-cached.png", "Recognition dense FLOPs to KL ≤ 0.1")):
        fig, ax = plt.subplots(layout="constrained")
        for optimizer, epsilon, style in (("adam", "10⁻⁸", "--"), ("adam_eps12", "10⁻¹²", "-")):
            for arm in ARMS: points_plot(ax, result["groups"][optimizer][metric][arm], arm, f'{LABELS[arm]}, ε={epsilon}', style)
        ax.set_xlabel("Latent configurations K"); ax.set_ylabel(ylabel)
        ax.set_title("16 to 1,048,576 configurations; six paired seeds")
        ax.legend()
        fig.savefig(ROOT / "reports" / filename); plt.close(fig)
    fig, axes = plt.subplots(1, 2, layout="constrained", sharey=True)
    for ax, optimizer, epsilon in zip(axes, OPTIMIZERS, ("10⁻⁸", "10⁻¹²"), strict=True):
        for arm in ARMS: points_plot(ax, result["groups"][optimizer]["step"][arm], arm, LABELS[arm])
        ax.set_title(f"Adam ε = {epsilon}")
    axes[0].set_ylabel("Updates to posterior KL ≤ 0.1")
    for ax in axes: ax.set_xlabel("Latent configurations K")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("Updates needed to reach the same posterior error")
    fig.savefig(ROOT / "reports/deep-wake-sleep-million-updates.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, layout="constrained", sharey=True)
    for ax, optimizer, epsilon in zip(axes, OPTIMIZERS, ("10⁻⁸", "10⁻¹²"), strict=True):
        for arm in ARMS:
            values = result["groups"][optimizer]["cached_dense_flops"][arm]["rolling_exponents"]
            x, y = [v["center"] for v in values], [v["exponent"] for v in values]
            ax.plot(x, y, color=COLORS[arm], marker="o", label=LABELS[arm])
            ax.fill_between(x, [v["seed_ci"][0] for v in values], [v["seed_ci"][1] for v in values], color=COLORS[arm], alpha=.15)
        ax.set_xscale("log"); ax.axhline(1, color="gray", linestyle=":"); ax.grid(True, alpha=.2); ax.set_title(f"Adam ε = {epsilon}")
    axes[0].set_ylabel("Cost exponent over three adjacent sizes")
    for ax in axes: ax.set_xlabel("Size-window midpoint K")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("Cost exponents across fitting windows")
    fig.savefig(ROOT / "reports/deep-wake-sleep-million-slopes.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, layout="constrained", sharey=True)
    for ax, target in zip(axes, (.5, .8), strict=True):
        for arm in ARMS: points_plot(ax, result["groups"]["adam_eps12"]["weight_overlap"][str(target)][arm], arm, LABELS[arm])
        ax.set_title(f"Weight overlap ≥ {target}")
    axes[0].set_ylabel("Recognition dense FLOPs to target")
    for ax in axes: ax.set_xlabel("Latent configurations K")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("Alignment with exact posterior weights; ε = 10⁻¹²")
    fig.savefig(ROOT / "reports/deep-wake-sleep-million-weight-overlap.png"); plt.close(fig)
    result["bridge"] = []
    for r in bridge:
        c = r["config"]
        old = read(ROOT / "results/deep_wake_sleep/wide_epsilon_control" / f'b12_g2_{c["arm"]}_adam_s{c["seed"]}' / "run.json")
        lookup = {v["step"]: v for v in r["rows"]}
        pairs = [(v, lookup[v["step"]]) for v in old["rows"] if v["step"] in lookup]
        result["bridge"].append({"seed": c["seed"], "arm": c["arm"], "old_first_hit_step": old["rows"][-1]["step"], "new_first_hit_step": r["rows"][-1]["step"],
                                "max_shared_checkpoint_kl_difference": max(abs(a["posterior_kl"] - b["posterior_kl"]) for a, b in pairs)})
    result["run_summaries"] = [{"path": r["path"], "stop_reason": r["stop_reason"], "final": r["rows"][-1]} for r in runs]
    result["total_training_gpu_seconds"] = sum(r["rows"][-1]["training_seconds"] for r in runs + bridge)
    (ROOT / "results/deep_wake_sleep/million_analysis.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({"largest_ratios": result["largest_ratios"], "fits": {o: {a: result["groups"][o]["cached_dense_flops"][a]["fits"] for a in ARMS} for o in OPTIMIZERS}}, indent=2))


if __name__ == "__main__": main()
