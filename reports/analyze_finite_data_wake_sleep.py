"""Real-data and compute curves, with paired seeds and explicit stopped runs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARMS = ("wake", "sleep", "forward", "predictive")
LABELS = {"wake": "Wake / real-data replay", "sleep": "Wake + sampled sleep", "forward": "Posterior teaching", "predictive": "Direct prediction"}
COLORS = dict(zip(ARMS, ("C0", "C1", "C2", "C3"), strict=True))


def read(path): return json.loads(path.read_text())


def point(run, budget):
    target = int(budget * run["costs"]["wake"] // run["costs"]["per_step"])
    if run["rows"][-1]["step"] < target:
        assert run["stop_reason"] == "stationary_train_and_validation"
        return run["rows"][-1], True
    return next(row for row in run["rows"] if row["step"] == target), False


def band(ax, x, values, label, **style):
    mean, se = values.mean(axis=0), values.std(axis=0, ddof=1) / np.sqrt(len(values))
    line, = ax.plot(x, mean, label=label, **style)
    ax.fill_between(x, np.maximum(mean - se, np.finfo(float).tiny), mean + se, color=line.get_color(), alpha=.15)


def configure(ax, xlabel, ylabel):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(True, alpha=.2)


def fit(values, sizes, mask, bootstrap):
    x = np.log(sizes[mask])
    beta = float(-np.polyfit(x, np.log(values[:, mask].mean(axis=0)), 1)[0])
    means = values[bootstrap][:, :, mask].mean(axis=1)
    powers = -(np.log(means) @ (x - x.mean())) / np.sum((x - x.mean()) ** 2)
    return {"beta": beta, "seed_ci": np.quantile(powers, [.025, .975]).tolist(), "bootstrap_beta": powers.tolist()}


def select():
    folders = [ROOT / "results/finite_data_wake_sleep" / name for name in ("pilot", "pilot_low_rates")]
    runs = [read(p) for folder in folders for p in sorted(folder.glob("*/*/run.json"))]
    assert len(runs) == 144 and all(r["stop_reason"] != "running" for r in runs)
    scores, selected = {}, {}
    for arm in ARMS:
        rates = sorted({r["lr"] for r in runs if r["arm"] == arm})
        scores[arm] = {str(lr): float(np.mean([np.log(point(r, b)[0]["test_prediction_kl"]) for r in runs if r["arm"] == arm and r["lr"] == lr for b in (256, 1024, 4096)])) for lr in rates}
        selected[arm] = float(min(scores[arm], key=scores[arm].__getitem__))
    result = {"selection": "Mean log observable test KL on separate tuning world, all three N and all three budgets; equally extended rate grid for every arm.", "scores": scores, "selected": selected, "tuning_world_seed": 9820, "tuning_dataset_seeds": [9821, 9822]}
    (ROOT / "results/finite_data_wake_sleep/selected.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


def factorized_followup():
    root = ROOT / "results/finite_data_wake_sleep"
    folder = root / "factorized_confirmation"
    p = read(folder / "protocol.json")
    runs = [read(path) for path in sorted(folder.glob("*/*/run.json"))]
    assert len(runs) == 480 and all(r["stop_reason"] != "running" for r in runs)
    sizes, seeds = np.array(p["data_sizes"]), p["seeds"]
    indexed = {(r["seed"], r["n"], r["arm"]): r for r in runs}
    values = {a: np.array([[point(indexed[s, n, a], 4096)[0]["test_prediction_kl"] for n in sizes] for s in seeds]) for a in ARMS}
    compiled = np.array([[read(folder / f"s{s}_n{n}/generator.json")["compiled_prediction_kl"] for n in sizes] for s in seeds])
    bootstrap = np.random.default_rng(985999).integers(len(seeds), size=(5000, len(seeds)))
    ratios = {}
    for i, n in enumerate(sizes):
        per_arm = {}
        for a in ("sleep", "forward", "predictive"):
            sampling_ratios = values[a][bootstrap, i].mean(axis=1) / values["wake"][bootstrap, i].mean(axis=1)
            per_arm[a] = {"over_wake": float(values[a][:, i].mean() / values["wake"][:, i].mean()), "seed_ci": np.quantile(sampling_ratios, [.025, .975]).tolist(), "wins": int(np.sum(values[a][:, i] < values["wake"][:, i]))}
        ratios[str(n)] = per_arm
    matched = [read(path) for path in sorted((root / "factorized_matched").glob("*/*/run.json"))]
    assert len(matched) == 108
    mi = {(r["seed"], r["n"], r["arm"]): r for r in matched}
    actual = {}
    for n in (512, 2048, 8192):
        vals = {a: np.array([mi[s, n, a]["rows"][-1]["test_prediction_kl"] for s in seeds]) for a in ("wake", "sleep", "forward")}
        actual[str(n)] = {"losses": {a: float(v.mean()) for a, v in vals.items()}, "sleep_wins": int(np.sum(vals["sleep"] < vals["wake"])), "wake_equivalent": {a: [mi[s, n, a]["rows"][-1]["wake_equivalent"] for s in seeds] for a in vals}}
        ratio = vals["sleep"][bootstrap].mean(axis=1) / vals["wake"][bootstrap].mean(axis=1)
        actual[str(n)]["sleep_over_wake"] = float(vals["sleep"].mean() / vals["wake"].mean())
        actual[str(n)]["ratio_seed_ci"] = np.quantile(ratio, [.025, .975]).tolist()
        assert all(min(actual[str(n)]["wake_equivalent"][a]) > 509 for a in vals)
        assert all(mi[s, n, a]["generator_data_sha256"] == indexed[s, n, a]["generator_data_sha256"] for s in seeds for a in vals)
    metrics = {}
    for a in ARMS:
        rows = [indexed[s, 8192, a]["rows"][-1] for s in seeds]
        metrics[a] = {key: float(np.mean([r[key] for r in rows])) for key in ("train_reverse_kl", "own_posterior_kl", "weight_overlap", "wake_equivalent")}
    capacity_runs = [read(path) for path in sorted((root / "paired_categorical").glob("*/*/run.json"))]
    assert len(capacity_runs) == 72 and all(r["stop_reason"] != "running" for r in capacity_runs)
    ci = {(r["seed"], r["n"], r["arm"]): r for r in capacity_runs}
    capacity = {}
    for n in (512, 2048, 8192):
        vals = {a: np.array([ci[s, n, a]["rows"][-1]["test_prediction_kl"] for s in seeds]) for a in ("wake", "sleep")}
        ratios_boot = vals["sleep"][bootstrap].mean(axis=1) / vals["wake"][bootstrap].mean(axis=1)
        capacity[str(n)] = {"losses": {a: float(v.mean()) for a, v in vals.items()}, "sleep_over_wake": float(vals["sleep"].mean() / vals["wake"].mean()), "ratio_seed_ci": np.quantile(ratios_boot, [.025, .975]).tolist()}
        for s in seeds:
            records = [ci[s, n, a] for a in ("wake", "sleep")] + [mi[s, n, a] for a in ("wake", "sleep")]
            costs = [r["rows"][-1]["recognition_flops"] for r in records]
            assert max(costs) / min(costs) < 1.004
            assert len({r["generator_data_sha256"] for r in records}) == 1
            first = np.load(root / f"paired_categorical/s{s}_n{n}/generator.npz")
            second = np.load(root / f"factorized_confirmation/s{s}_n{n}/generator.npz")
            assert np.array_equal(first["w"], second["w"]) and np.array_equal(first["b"], second["b"])
    result = {"protocol": p, "runs": len(runs), "matched_runs": len(matched), "capacity_runs": len(capacity_runs), "curves": {a: v.tolist() for a, v in values.items()}, "compiled": compiled.tolist(), "ratios": ratios, "actual_compute_check": actual, "paired_capacity_control": capacity, "large_n_metrics": metrics, "training_cpu_seconds": sum(r["rows"][-1]["training_cpu_seconds"] for r in runs), "interpretation": "Approximate-recognition follow-up on a fresh fixed world, with a paired categorical capacity control on the same world, data, generators and approximate compute. Forward posterior teaching is stronger than sampled sleep. Persistent representation/projection errors can change the loss floor; different asymptotic data exponents are not established."}
    (root / "factorized_analysis.json").write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(layout="constrained")
    for a in ("wake", "sleep", "forward"):
        band(ax, sizes, values[a], LABELS[a], color=COLORS[a], marker="o")
    ax.plot(sizes, compiled.mean(axis=0), color="black", linestyle=":", label="Learned generator, exact inference")
    configure(ax, "Number of real observations N", "Held-out excess prediction loss (nats)")
    ax.set_title("Finite data with a factorized recognizer")
    ax.legend()
    fig.savefig(ROOT / "reports/wake-sleep-finite-data-factorized.png"); plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    ns = np.array([512, 2048, 8192])
    for a in ("wake", "sleep", "forward"):
        v = np.array([[mi[s, n, a]["rows"][-1]["test_prediction_kl"] for n in ns] for s in seeds])
        band(ax, ns, v, LABELS[a], color=COLORS[a], marker="o")
    configure(ax, "Number of real observations N", "Held-out excess prediction loss (nats)")
    ax.set_title("Matched dense compute: 512 wake-update units")
    ax.set_xticks(ns, [f"{n:,}" for n in ns])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend()
    fig.savefig(ROOT / "reports/wake-sleep-finite-data-matched.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for ax, family, family_index in zip(axes, ("Full posterior", "Factorized posterior"), (ci, mi), strict=True):
        for a in ("wake", "sleep"):
            ys = np.array([[family_index[s, n, a]["rows"][-1]["test_prediction_kl"] for n in ns] for s in seeds])
            band(ax, ns, ys, LABELS[a], color=COLORS[a], marker="o")
        configure(ax, "Real observations N", "Excess prediction loss")
        ax.set_title(family)
        ax.set_xticks(ns, [f"{n:,}" for n in ns])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[1].set_ylabel("")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("Same task, data, generator and matched dense compute")
    fig.savefig(ROOT / "reports/wake-sleep-finite-data-capacity.png"); plt.close(fig)
    print(json.dumps({"largest_data_ratios": ratios["8192"], "actual_compute_check": actual, "capacity": capacity, "metrics": metrics}, indent=2))


def followup_section():
    r = read(ROOT / "results/finite_data_wake_sleep/factorized_analysis.json")
    sizes = r["protocol"]["data_sizes"]
    rows = []
    for i, n in enumerate(sizes):
        means = {a: np.mean(np.array(r["curves"][a])[:, i]) for a in ARMS}
        ratio = r["ratios"][str(n)]["sleep"]
        rows.append(f'<tr><td>{n:,}</td>{"".join(f"<td>{means[a]:.5f}</td>" for a in ARMS)}<td>{ratio["over_wake"]:.3f} [{ratio["seed_ci"][0]:.3f}, {ratio["seed_ci"][1]:.3f}]</td></tr>')
    actual_rows = []
    for n, v in r["actual_compute_check"].items():
        actual_rows.append(f'<tr><td>{int(n):,}</td>{"".join(f"<td>{v["losses"][a]:.5f}</td>" for a in ("wake", "sleep", "forward"))}<td>{v["sleep_over_wake"]:.3f} [{v["ratio_seed_ci"][0]:.3f}, {v["ratio_seed_ci"][1]:.3f}]</td></tr>')
    ratio = r["ratios"]["8192"]["sleep"]
    return f'''<h2>A targeted follow-up: an approximate recognizer</h2>
<p>The fully expressive recognizer below showed no sleep advantage. We then tested a standard mean-field recognition family: four independent binary probabilities conditioned on the six input bits, with 28 affine parameters. It cannot represent arbitrary posterior dependencies between causes. Both methods use this same restriction, start from small independent random weights with zero biases, and receive the same finite information. Rates were selected in the separate tuning world and frozen before testing a <strong>new fixed world with 12 fresh dataset seeds</strong>. We subsequently repeated fully expressive recognition on these exact same datasets and generators to check the effect of capacity directly.</p>
<p><strong>Here sampled sleep improves the real-data learning curve.</strong> At N=8,192 its excess loss is {100*(1-ratio["over_wake"]):.1f}% lower than wake, with a sleep/wake ratio {ratio["over_wake"]:.3f} [95% seed interval {ratio["seed_ci"][0]:.3f}, {ratio["seed_ci"][1]:.3f}]. The comparison gives each method the same maximum recognition-compute allocation. Wake often stops early after its prescribed training and validation stationarity check, so this headline is not a claim of identical executed compute. The actual-compute check below addresses that separately.</p>
<figure><img src="wake-sleep-finite-data-factorized.png" alt="Prediction loss against real sample count with approximate recognition"><figcaption>Same fixed architecture and data within each comparison; actual ancestral dreams, no checker, filtering, or temperature change. Means ± one seed standard error. Dotted black is the learned generator with exact inference. The emerging gap above it shows that fitting the generator is no longer the only limitation.</figcaption></figure>
<table><thead><tr><th>N</th><th>Wake</th><th>Sleep</th><th>Posterior teaching</th><th>Direct prediction</th><th>Sleep / wake [95% CI]</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p><strong>The decisive control is stronger than sleep:</strong> exact posterior teaching on the existing real examples achieves still lower task loss. This uses no new real information and no imagined contexts. It shows that the useful signal can be extracted from the learned model in more than one way; generation is one implementation, not established as necessary.</p>
<h3>Compute check without early stopping</h3><p>We replayed all 12 seeds at N=512, 2,048 and 8,192 to a budget of 512 wake-update equivalents, before early stopping can save compute. Wake and posterior teaching execute 512 updates; sleep executes 239 combined updates, including 15,296 newly generated contexts, costing 510.59 wake equivalents. Sleep therefore uses 0.28% less dense recognition compute. The shared generator and its fitting cost are identical. These counts cover executed leading matrix products and ancestral generation; they exclude elementwise work and evaluation, so they are an approximate compute proxy. This is a post-confirmation diagnostic using the same frozen rates and random streams, not another hyperparameter search.</p>
<figure><img src="wake-sleep-finite-data-matched.png" alt="Actual compute matched finite-data comparisons"></figure><table><thead><tr><th>N</th><th>Wake</th><th>Sleep</th><th>Posterior teaching</th><th>Sleep / wake [95% CI]</th></tr></thead><tbody>{"".join(actual_rows)}</tbody></table>
<h3>Paired recognition-capacity check</h3><figure><img src="wake-sleep-finite-data-capacity.png" alt="Paired full and factorized recognition on the same world and data"><figcaption>The true task, real observations, fitted generator, and compute are paired across recognition families. All executed dense budgets are within 0.4%. The fully expressive family uses its previously selected rates, with no tuning on this world. At N=8,192 its wake/sleep losses are {r["paired_capacity_control"]["8192"]["losses"]["wake"]:.5f}/{r["paired_capacity_control"]["8192"]["losses"]["sleep"]:.5f}. Each panel compares wake and sleep within the same architecture. Panels share the vertical scale.</figcaption></figure>
<p>At N=8,192, wake has lower real-data reverse KL ({r["large_n_metrics"]["wake"]["train_reverse_kl"]:.3f}) than sleep ({r["large_n_metrics"]["sleep"]["train_reverse_kl"]:.3f}), despite worse observable prediction. This supports a tradeoff between inference objectives under restricted recognition, rather than simply asserting wake failed to optimize its own criterion. It does not certify a global optimum.</p>
<p><strong>This is a learning-curve improvement, not an established change in the asymptotic data exponent.</strong> The restricted recognizer leaves an approximation error, and the methods approach different apparent floors. A power fit that ignored those floors would mix statistical learning with approximation bias. The next LLM match should retain an imperfect learned forward model and measure observable prediction through an approximate inverse, with posterior teaching as a control.</p>
<p><a href="finite-data-wake-sleep-data/factorized_analysis.json">Follow-up analysis</a> · <a href="finite-data-wake-sleep-data/factorized_confirmation/protocol.json">Frozen follow-up protocol</a> · <a href="finite-data-wake-sleep-data/factorized_pilot/selected.json">Separate rate selection</a> · <a href="finite-data-wake-sleep-data/factorized_matched/protocol.json">Actual-compute protocol</a> · <a href="finite-data-wake-sleep-data/factorized_confirmation/finite_data_wake_sleep.py">Follow-up training code</a></p>
<h2>Original comparison: a fully expressive affine recognizer</h2>'''


def analyze(folder):
    protocol = read(folder / "protocol.json")
    seeds, sizes = protocol["seeds"], np.array(protocol["data_sizes"])
    runs = [read(p) for p in sorted(folder.glob("*/*/run.json"))]
    assert len(runs) == len(seeds) * len(sizes) * len(ARMS)
    assert all(r["stop_reason"] != "running" for r in runs)
    by_key = {(r["seed"], r["n"], r["arm"]): r for r in runs}
    generator = {(s, n): read(folder / f"s{s}_n{n}" / "generator.json") for s in seeds for n in sizes}
    for s in seeds:
        for n in sizes:
            assert len({by_key[s, n, a]["generator_data_sha256"] for a in ARMS}) == 1
            assert all(by_key[s, n, a]["costs"]["shared_generator"] == generator[s, n]["leading_dense_flops"] for a in ARMS)
    compiled = np.array([[generator[s, n]["compiled_prediction_kl"] for n in sizes] for s in seeds])
    rng = np.random.default_rng(982999)
    bootstrap = rng.integers(len(seeds), size=(5000, len(seeds)))
    result = {"protocol": protocol, "runs": len(runs), "fits": {}, "curves": {}, "ratios": {}, "stopped": {}, "compiled_prediction_kl": compiled.tolist(), "total_training_cpu_seconds": sum(r["rows"][-1]["training_cpu_seconds"] for r in runs) + sum(g["training_cpu_seconds"] for g in generator.values()), "generator_costs": {str(n): [generator[s, n]["leading_dense_flops"] for s in seeds] for n in sizes}, "fit_caveat": "Descriptive mean excess-loss slopes at fixed subsequent recognition allocation, not asymptotic exponents; shared generator fitting varies with N. Seed intervals omit task and optimizer uncertainty."}
    plots = ROOT / "reports"
    for budget in protocol["config"]["report_budgets"]:
        values = {a: np.array([[point(by_key[s, n, a], budget)[0]["test_prediction_kl"] for n in sizes] for s in seeds]) for a in ARMS}
        result["curves"][str(budget)] = {a: v.tolist() for a, v in values.items()}
        result["stopped"][str(budget)] = {a: [sum(point(by_key[s, n, a], budget)[1] for s in seeds) for n in sizes] for a in ARMS}
        result["ratios"][str(budget)] = {}
        for n_idx, n in enumerate(sizes):
            numerator, denominator = values["sleep"][:, n_idx], values["wake"][:, n_idx]
            ratios = numerator[bootstrap].mean(axis=1) / denominator[bootstrap].mean(axis=1)
            differences = numerator[bootstrap].mean(axis=1) - denominator[bootstrap].mean(axis=1)
            result["ratios"][str(budget)][str(n)] = {"sleep_over_wake": float(numerator.mean() / denominator.mean()), "seed_ci": np.quantile(ratios, [.025, .975]).tolist(), "paired_difference": float((numerator - denominator).mean()), "difference_ci": np.quantile(differences, [.025, .975]).tolist(), "sleep_wins": int(np.sum(numerator < denominator))}
        result["fits"][str(budget)] = {}
        for low, high in ((32, 8192), (128, 2048), (512, 8192)):
            mask = (sizes >= low) & (sizes <= high)
            fits = {a: fit(v, sizes, mask, bootstrap) for a, v in {**values, "compiled": compiled}.items()}
            gap = np.array(fits["sleep"]["bootstrap_beta"]) - np.array(fits["wake"]["bootstrap_beta"])
            result["fits"][str(budget)][f"{low}-{high}"] = {"methods": {a: {k: v for k, v in f.items() if k != "bootstrap_beta"} for a, f in fits.items()}, "sleep_minus_wake_beta": {"value": fits["sleep"]["beta"] - fits["wake"]["beta"], "seed_ci": np.quantile(gap, [.025, .975]).tolist()}}
    budget = protocol["config"]["report_budgets"][-1]
    values = {a: np.array(result["curves"][str(budget)][a]) for a in ARMS}
    fig, ax = plt.subplots(layout="constrained")
    for a in ("wake", "sleep"): band(ax, sizes, values[a], LABELS[a], color=COLORS[a], marker="o")
    ax.plot(sizes, compiled.mean(axis=0), linestyle=":", color="black", label="Learned generator, exact inference")
    configure(ax, "Number of real observations N", "Held-out excess prediction loss (nats)")
    ax.set_title("Finite data; fixed task and model size")
    ax.legend()
    fig.savefig(plots / "wake-sleep-finite-data.png"); plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for a in ARMS: band(ax, sizes, values[a], LABELS[a], color=COLORS[a])
    configure(ax, "Number of real observations N", "Held-out excess prediction loss (nats)")
    ax.set_title("Controls at the same allocated compute")
    ax.legend()
    fig.savefig(plots / "wake-sleep-finite-data-controls.png"); plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for budget, linestyle in zip(protocol["config"]["report_budgets"], (":", "--", "-"), strict=True):
        for a in ("wake", "sleep"):
            v = np.array(result["curves"][str(budget)][a])
            ax.plot(sizes, v.mean(axis=0), color=COLORS[a], linestyle=linestyle, label=f'{LABELS[a]}, {budget} units')
    configure(ax, "Number of real observations N", "Held-out excess prediction loss (nats)")
    ax.set_title("Real-data curves at three compute allocations")
    ax.legend()
    fig.savefig(plots / "wake-sleep-finite-data-budgets.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, layout="constrained")
    for ax, n in zip(axes, (128, 2048), strict=True):
        for a in ("wake", "sleep", "forward"):
            cohort = [by_key[s, n, a] for s in seeds]
            grid = np.geomspace(2.2, protocol["config"]["budget"] - 2.2, 150)
            ys = np.array([np.interp(np.log(grid), np.log([r["wake_equivalent"] for r in run["rows"] if r["step"] > 0]), [r["test_prediction_kl"] for r in run["rows"] if r["step"] > 0]) for run in cohort])
            observed = grid <= min(run["rows"][-1]["wake_equivalent"] for run in cohort)
            ax.plot(grid[observed], ys.mean(axis=0)[observed], color=COLORS[a], label=LABELS[a])
            if not observed.all():
                tail = np.r_[np.flatnonzero(observed)[-1], np.flatnonzero(~observed)]
                ax.plot(grid[tail], ys.mean(axis=0)[tail], color=COLORS[a], linestyle="--")
            ax.fill_between(grid, ys.mean(axis=0) - ys.std(axis=0, ddof=1) / np.sqrt(len(seeds)), ys.mean(axis=0) + ys.std(axis=0, ddof=1) / np.sqrt(len(seeds)), color=COLORS[a], alpha=.15)
        ax.axhline(compiled[:, list(sizes).index(n)].mean(), color="black", linestyle=":")
        configure(ax, "Allocated recognition compute\n(wake-update equivalents)", "Excess prediction loss")
        ax.set_title(f"N = {n:,}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.savefig(plots / "wake-sleep-finite-data-compute.png"); plt.close(fig)
    fig, ax = plt.subplots(layout="constrained")
    for a in ("wake", "sleep", "forward"):
        overlaps = np.array([[point(by_key[s, n, a], protocol["config"]["budget"])[0]["weight_overlap"] for n in sizes] for s in seeds])
        band(ax, sizes, overlaps, LABELS[a], color=COLORS[a])
    ax.set_xscale("log"); ax.set_xlabel("Number of real observations N"); ax.set_ylabel("Weight overlap with learned posterior")
    ax.set_title("Did recognition recover the generator's knowledge?"); ax.grid(True, alpha=.2); ax.legend()
    fig.savefig(plots / "wake-sleep-finite-data-overlap.png"); plt.close(fig)
    (folder / "analysis.json").write_text(json.dumps(result, indent=2))
    build_report(result, folder)
    print(json.dumps({"runs": result["runs"], "training_cpu_seconds": result["total_training_cpu_seconds"], "fits": result["fits"][str(protocol["config"]["budget"])], "ratios": result["ratios"][str(protocol["config"]["budget"])]}, indent=2))


def build_report(result, folder):
    for filename in ("FINITE_DATA_WAKE_SLEEP_PROTOCOL.md", "FINITE_DATA_WAKE_SLEEP_THEORY.md", "FINITE_DATA_WAKE_SLEEP_RESULTS.md", "analyze_finite_data_wake_sleep.py", "confirm_finite_data_compute.py"):
        shutil.copyfile(ROOT / "experiments" / filename, ROOT / "reports" / filename)
    p = result["protocol"]
    budget = str(p["config"]["budget"])
    rows = []
    for i, n in enumerate(p["data_sizes"]):
        means = {a: np.array(result["curves"][budget][a])[:, i].mean() for a in ARMS}
        ratio = result["ratios"][budget][str(n)]
        cost = np.mean(result["generator_costs"][str(n)])
        rows.append(f'<tr><td>{n:,}</td>{"".join(f"<td>{means[a]:.5f}</td>" for a in ARMS)}<td>{ratio["sleep_over_wake"]:.3f} [{ratio["seed_ci"][0]:.3f}, {ratio["seed_ci"][1]:.3f}]</td><td>{cost:.3g}</td></tr>')
    fits = []
    for window, item in result["fits"][budget].items():
        cells = [f'{item["methods"][a]["beta"]:.2f} [{item["methods"][a]["seed_ci"][0]:.2f}, {item["methods"][a]["seed_ci"][1]:.2f}]' for a in ("wake", "sleep", "compiled")]
        fits.append(f'<tr><td>{window}</td>{"".join(f"<td>{cell}</td>" for cell in cells)}</tr>')
    timestamp = datetime.now(ZoneInfo("Europe/London")).strftime("%d %B %Y, %H:%M %Z")
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><title>Wake–sleep with finite real data</title><style>body{{font-family:system-ui,sans-serif;max-width:70rem;margin:auto;padding:2rem;line-height:1.6;color:#18212b}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:.5rem;text-align:left;border-bottom:1px solid #ccd3da}}figure{{margin:2rem 0}}figcaption{{color:#485868}}code{{background:#eef2f5}}a{{color:#145e99}}</style></head><body>
<h1>Does practice on its own generations improve learning from finite data?</h1><p>Updated {timestamp}. {result["runs"]} original confirmation runs, 480 approximation follow-up runs, 108 compute checks and 72 paired capacity checks; 12 paired dataset seeds per main cohort. All computation used the local CPU. Paid API/rental spend: $0.</p>
<p><strong>This is now held-out prediction loss versus the number of independent real observations, at fixed task distribution and model size.</strong> The generator is learned from those observations, rather than supplied the true distribution. Sleep uses actual unfiltered ancestral draws. The data curves fix subsequent recognition-compute allocation; generator fitting is common to all methods within an N/seed but its stopping time varies across N.</p>
{followup_section()}
<figure><img src="wake-sleep-finite-data.png" alt="Held-out excess prediction loss versus number of real observations"><figcaption>Means ± one standard error across paired dataset seeds. The dotted line predicts with the learned generator using exact inference. Recognition has an allocation of {p["config"]["budget"]:,} ordinary wake-update equivalents; generation and extra training are charged to sleep. N includes seven eighths training and one eighth validation. Duplicate binary values are legitimate independent observations. Models stopped by the prespecified validation stationarity rule retain their last value.</figcaption></figure>
<h2>What each method sees</h2><p>Four hidden binary causes generate eight observed bits through a single sigmoid layer. A complete real example contains eight bits; the task is to predict the last two from the first six. No real latent labels are provided. The generator starts from random, data-independent parameters and is fitted on the training portion by exact marginal MAP optimization. Its selected checkpoint is copied identically to every method, then frozen. Recognition uses a single affine 16-way softmax with small random weights and zero biases. Neither truth weights nor population probabilities enter training or stopping.</p>
<p>Wake replays the finite real contexts. Sleep adds generated cause–context pairs, with known cause labels from the generator's own choices. Posterior teaching uses exact learned-model posteriors on the existing real contexts. Direct prediction trains the observable conditional likelihood using real targets and the same frozen generator. It is a stronger task-directed control, not an additional source of data.</p>
<h2>Observable results and controls</h2><table><thead><tr><th>N</th><th>Wake</th><th>Sleep</th><th>Posterior teaching</th><th>Direct prediction</th><th>Sleep / wake [95% seed CI]</th><th>Shared fitting FLOPs</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>Loss is conditional KL to the true held-out observation distribution, in nats. Lower is better; a sleep/wake ratio above one favors wake. The primary criterion is observable prediction, not agreement with a self-generated posterior. The compiled predictor is a reference rather than an absolute lower bound: a recognizer can sometimes compensate for the generator's mistakes.</p>
<figure><img src="wake-sleep-finite-data-controls.png" alt="All four methods on the same finite-data task"></figure>
<figure><img src="wake-sleep-finite-data-budgets.png" alt="Data curves at three recognition budgets"><figcaption>Each line fixes recognition-compute allocation. Shared generator fitting is identical across methods within each N and seed; the table reports its additional cost. The data-axis experiment does not hold total compute constant across N.</figcaption></figure>
<h2>How training progresses at a fixed data size</h2><figure><img src="wake-sleep-finite-data-compute.png" alt="Held-out prediction loss versus compute at fixed real dataset sizes"><figcaption>Training and dreams use separate random streams. Dashed tails include stopped models held fixed at larger allocated budgets. Dotted horizontal lines show the imperfect learned generator with exact inference. No compute exponent is fitted through stopped tails.</figcaption></figure>
<h2>Descriptive data exponents</h2><p>Fits use L(N) = A N<sup>−β</sup>, with zero Bayes-excess loss as reference, at the largest recognition allocation. Windows are specified before confirmation. These are finite-range descriptions. Their confidence intervals resample dataset seeds jointly across N; they do not include uncertainty from changing task distributions or optimizers.</p><table><thead><tr><th>N window</th><th>Wake β</th><th>Sleep β</th><th>Compiled generator β</th></tr></thead><tbody>{"".join(fits)}</tbody></table>
<h2>Connection to the weight mechanism</h2><figure><img src="wake-sleep-finite-data-overlap.png" alt="Recognition coefficient alignment versus real data size"><figcaption>Centered affine coefficients are compared with the posterior implied by the learned generator, not with teacher parameters. This distinguishes incomplete transfer from finite-data errors in the generator itself.</figcaption></figure>
<p>The exact posterior of this one-layer model is affine in the context bits. When the real context design has full column rank, both wake and forward fitting can recover those coefficients from finite contexts. Once they do, observable error comes from the learned generator. Sleep cannot automatically remove that statistical error. In a regular local estimation regime, parameter fluctuations of order N<sup>−1/2</sup> yield quadratic KL error of order N<sup>−1</sup>; this conditional argument predicts a shared leading data exponent after inference catches up. It is not a guarantee of global optimization or a claim about every latent-variable model.</p>
<h2>Reproducibility and limits</h2><p>Hyperparameters were frozen using a separate tuning world and independent seeds, with the same learning-rate grid for every arm. Recognition learning rates decay with cumulative charged compute. Evaluation uses exact held-out population expectations only to avoid measurement noise; all training data and dreams are finite samples. Training stops using finite validation examples already counted in N. Checkpoints include optimizer moments, RNG states and data-stream position. This is a modular model with a frozen learned generator, not yet a joint-training LLM test.</p>
<p><a href="FINITE_DATA_WAKE_SLEEP_RESULTS.md">Results notes</a> · <a href="FINITE_DATA_WAKE_SLEEP_PROTOCOL.md">Protocol</a> · <a href="FINITE_DATA_WAKE_SLEEP_THEORY.md">Weight mechanism and statistical interpretation</a> · <a href="analyze_finite_data_wake_sleep.py">Analysis and plot code</a> · <a href="confirm_finite_data_compute.py">Compute and capacity controls</a></p>
<p><a href="finite-data-wake-sleep-data/{folder.name}/analysis.json">Analysis and paired observations</a> · <a href="finite-data-wake-sleep-data/{folder.name}/protocol.json">Frozen protocol</a> · <a href="finite-data-wake-sleep-data/selected.json">Separate hyperparameter selection</a> · <a href="finite-data-wake-sleep-data/{folder.name}/finite_data_wake_sleep.py">Training code</a> · <a href="finite-data-wake-sleep-data/factorized_audit.json">Gradient and resume checks</a> · <a href="finite-data-wake-sleep-data/source_manifest.json">Archived source hashes</a> · <a href="wake-sleep-scaling.html">Earlier toy and LLM experiments</a></p></body></html>'''
    (ROOT / "reports/wake-sleep-finite-data.html").write_text(html)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", action="store_true")
    parser.add_argument("--followup", action="store_true")
    parser.add_argument("--folder", type=Path, default=ROOT / "results/finite_data_wake_sleep/confirmation")
    args = parser.parse_args()
    if args.select: select()
    elif args.followup: factorized_followup()
    else: analyze(args.folder)


if __name__ == "__main__": main()
