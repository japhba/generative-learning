"""Independently regrade matched Zipf fine-tuning and compare with ICL and the toy."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from analyze_zipf import effective_fit, estimate, plt
from zipf_task import BUDGETS, LABELS, P, ideal_curves, label, make_world

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"qwen": "Qwen3-0.6B", "smollm": "SmolLM2-135M"}
ARMS = {"iid": "IID study", "ranked": "Ranked study", "historical_qwen": "Earlier Qwen queries", "on_policy": "Generated queries (EOS)"}
FIELDS = ("error", "marginalized_error", "coverage_error", "seen_error_mass", "seen_accuracy", "known_mass", "probability_mse", "functional_overlap")


def read(path):
    return json.loads(path.read_text())


def audit_case(folder, seed, repeats, lr, live):
    complete = read(folder / "complete.json")
    assert complete["complete"] and complete["seed"] == seed
    assert complete["repeats"] == repeats and complete["learning_rate"] == lr
    data = read(folder / "observations.json")
    assert len(data["queries"]) == len(data["targets"]) == 128
    assert data["targets"] == [LABELS.index(label({"seed": seed}, key)) for key in data["queries"]]
    replay = np.load(folder / "replay_indices.npy")
    assert replay.shape == (128, repeats, 32)
    for n, selected in enumerate(replay):
        assert np.all(selected[:, 0] == n) and selected.min() >= 0 and selected.max() <= n
    if live:
        assert [row["budget"] for row in data["generated"]] == BUDGETS
        for row in data["generated"]:
            assert row["previous"] + row["queries"] == data["queries"][:row["budget"]]
            assert row["guesses"] == np.argmax(row["guess_probabilities"], axis=1).tolist()
            assert row["feedback"] == [LABELS.index(label({"seed": seed}, key)) for key in row["queries"]]
    world = make_world(seed)
    records = []
    for budget in [0, *BUDGETS]:
        record = read(folder / f"n{budget}.json")
        assert record["budget"] == budget and record["queried"] == data["queries"][:budget]
        keys = record["keys"]
        seen = sorted(set(record["queried"]))
        assert keys == sorted(set(world["test_keys"]) | set(seen))
        assert record["counters"]["optimizer_updates"] == budget * repeats
        assert record["counters"]["training_presentations"] == 32 * budget * repeats
        probs = np.array(record["probabilities"])
        np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
        assert np.all(probs >= 0)
        lookup = {key: index for index, key in enumerate(keys)}
        targets = {key: LABELS.index(label({"seed": seed}, key)) for key in keys}
        predictions = dict(zip(keys, probs.argmax(axis=1), strict=True))
        raw = sum(weight * (predictions[key] != targets[key]) for key, weight in zip(world["test_keys"], world["test_weights"], strict=True))
        known_mass = sum(float(P[key - 1]) for key in seen)
        seen_mass = sum(float(P[key - 1]) * (predictions[key] != targets[key]) for key in seen)
        probability_mse = sum(weight * np.sum((probs[lookup[key]] - np.eye(4)[targets[key]]) ** 2) for key, weight in zip(world["test_keys"], world["test_weights"], strict=True))
        overlap = sum(weight * (probs[lookup[key], targets[key]] - .25) / .75 for key, weight in zip(world["test_keys"], world["test_weights"], strict=True))
        expected = {"error": raw, "known_mass": known_mass, "coverage_error": .75 * (1 - known_mass),
                    "seen_error_mass": seen_mass, "marginalized_error": .75 * (1 - known_mass) + seen_mass,
                    "probability_mse": probability_mse, "functional_overlap": overlap}
        if budget: expected["seen_accuracy"] = np.mean([predictions[key] == targets[key] for key in seen])
        for field, value in expected.items(): np.testing.assert_allclose(record["metrics"][field], value, rtol=1e-11, atol=1e-12)
        records.append(record)
    return records


def summarize(values, indices):
    point = estimate(values, indices)
    return {"mean": point["mean"].tolist(), "ci95": np.stack([point["lower"], point["upper"]], axis=1).tolist(), "by_world": values.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/zipf_finetuning")
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    parser.add_argument("--family", choices=FAMILIES, help="Audit and plot one completed model family while other runs continue.")
    args = parser.parse_args()
    protocol = read(args.root / "protocol.json")
    seeds = protocol["main_seeds"]
    budgets = np.array([0, *BUDGETS])
    indices = np.random.default_rng(20260906).integers(0, len(seeds), size=(1500, len(seeds)))
    result = {"complete": True, "world_seeds": seeds, "budgets": budgets.tolist(), "bootstrap_samples": len(indices), "families": {},
              "interpretation": "32 paired original ICL dictionaries; individual-key fine-tuning evaluation differs from the old batched ICL answers. Primary weighted error is regraded from predictions. Marginalized error adds exact unseen Bayes error and exhaustive observed-key mistakes. Fits over8–128 are descriptive finite-range exponents; all windows and irregularities retained. Model selection used disjoint pilot training accuracy only. Tokens and update counts are distinct from new-label budgets."}
    for family in tqdm((args.family,) if args.family else FAMILIES, desc="Auditing model families"):
        selection = read(args.root / family / "selection.json")
        assert selection["status"] == "passed"
        records = {}
        for arm in ARMS:
            cases = []
            for seed in tqdm(seeds, desc=f"{family}/{arm}", leave=False):
                folder = args.root / family / (f"live/{seed}" if arm == "on_policy" else f"main/{seed}_{arm}")
                case = audit_case(folder, seed, selection["repeats"], selection["learning_rate"], arm == "on_policy")
                assert read(folder / "complete.json")["source_sha256"] == protocol["source_sha256"]
                if arm == "iid": assert case[-1]["queried"] == make_world(seed)["passive_keys"]
                if arm == "ranked": assert case[-1]["queried"] == list(range(1, 129))
                if arm == "historical_qwen":
                    rows = read(ROOT / f"results/zipf_replication/world_results/qwen__qwen3-30b-a3b-instruct-2507__{seed}.json")
                    original = next(row["query_keys"] for row in rows if row["budget"] == 128 and row["condition"] == "active_generate")
                    assert case[-1]["queried"] == original
                cases.append(case)
            records[arm] = cases
        for i in range(len(seeds)):
            reference = records["iid"][i][0]["probabilities"]
            for arm in ARMS: np.testing.assert_array_equal(records[arm][i][0]["probabilities"], reference)
            replay = np.load(args.root / family / f"main/{seeds[i]}_iid/replay_indices.npy")
            for arm in ARMS:
                folder = args.root / family / (f"live/{seeds[i]}" if arm == "on_policy" else f"main/{seeds[i]}_{arm}")
                np.testing.assert_array_equal(np.load(folder / "replay_indices.npy"), replay)
        curves = {}
        for arm, cases in records.items():
            metrics = {}
            for field in FIELDS:
                selected = [case[1:] for case in cases] if field == "seen_accuracy" else cases
                values = np.array([[record["metrics"][field] for record in case] for case in selected])
                metrics[field] = summarize(values, indices)
                metrics[field]["budgets"] = BUDGETS if field == "seen_accuracy" else budgets.tolist()
            clocks = {field: summarize(np.array([[record["counters"][field] for record in case] for case in cases]), indices) for field in ("optimizer_updates", "training_presentations", "training_prompt_tokens", "training_padded_tokens", "train_seconds", "generation_seconds", "generation_prompt_tokens", "generation_output_tokens", "guess_prompt_tokens")}
            curves[arm] = {"metrics": metrics, "clocks": clocks,
                           "parameter_displacement_l2": summarize(np.array([[record["probes"]["parameter_displacement_l2"] for record in case] for case in cases]), indices),
                           "unique_queries_at128": [case[-1]["metrics"]["observed_unique"] for case in cases]}
        windows = []
        for minimum, maximum in ((8, 128), (4, 64), (16, 128)):
            fitted, draws = {}, {}
            for arm in ARMS:
                fitted[arm], draws[arm] = effective_fit(budgets, np.array(curves[arm]["metrics"]["error"]["by_world"]), indices, minimum, maximum)
            differences = {}
            for arm in ("ranked", "historical_qwen", "on_policy"):
                delta = draws[arm] - draws["iid"]
                differences[arm + "_minus_iid"] = {"mean": fitted[arm]["alpha"] - fitted["iid"]["alpha"], "ci95": np.quantile(delta, [.025, .975]).tolist()}
            diagnostics = {metric: {arm: effective_fit(budgets, np.array(curves[arm]["metrics"][metric]["by_world"]), indices, minimum, maximum)[0] for arm in ARMS}
                           for metric in ("coverage_error", "marginalized_error")}
            windows.append({"budget_range": [minimum, maximum], "fits": fitted, "paired_exponent_differences": differences, "diagnostic_fits": diagnostics})
        endpoint_differences = {}
        for arm in ("ranked", "historical_qwen", "on_policy"):
            difference = np.array(curves[arm]["metrics"]["error"]["by_world"]) - np.array(curves["iid"]["metrics"]["error"]["by_world"])
            endpoint_differences[arm + "_minus_iid"] = summarize(difference, indices)
        pilots = selection["pilot_results"]
        calibration = {"attempted_cases": len(pilots), "optimizer_updates": sum(row["counters"]["optimizer_updates"] for row in pilots),
                       "training_presentations": sum(row["counters"]["training_presentations"] for row in pilots),
                       "training_seconds": sum(row["counters"]["train_seconds"] for row in pilots), "pilot_results": pilots}
        result["families"][family] = {"selection": {key: value for key, value in selection.items() if key != "pilot_results"}, "curves": curves, "exponent_windows": windows,
                                       "paired_error_differences": endpoint_differences, "calibration": calibration,
                                       "audited_cases": len(ARMS) * len(seeds), "audited_checkpoints": len(ARMS) * len(seeds) * len(budgets)}
    args.output.mkdir(parents=True, exist_ok=True)
    theory = ideal_curves([0, *BUDGETS])
    result["toy"] = theory
    if args.family:
        family = result["families"][args.family]
        figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
        for axis, arms, title in zip(axes, (("iid", "ranked"), tuple(ARMS)), ("Fixed observations", "All acquisition conditions"), strict=True):
            for arm in arms:
                i = list(ARMS).index(arm)
                values = family["curves"][arm]["metrics"]["error"]
                limits = np.array(values["ci95"])[1:]
                axis.plot(BUDGETS, values["mean"][1:], marker="o", color=f"C{i}", label=ARMS[arm])
                axis.fill_between(BUDGETS, limits[:, 0], limits[:, 1], color=f"C{i}", alpha=.15)
            axis.set(title=title, xscale="log", yscale="log", xlabel="Oracle labels acquired")
        for i, key in enumerate(("iid_errors", "ranked_errors")):
            axes[0].plot(BUDGETS, theory[key][1:], linestyle="--", color=f"C{i}")
        axes[0].set_ylabel("Weighted test error; dashed: exact toy")
        figure.suptitle(FAMILIES[args.family] + ": 32 paired dictionaries")
        figure.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        figure.savefig(args.output / f"zipf-finetuning-{args.family}.png"); plt.close(figure)
        result["scope"] = "One completed model family; other families are reported separately."
        result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        (args.output / f"zipf-finetuning-{args.family}-data.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print((args.output / f"zipf-finetuning-{args.family}.png").resolve())
        return
    for metric, filename, ylabel in (("error", "zipf-finetuning-learning.png", "Weighted test error"),
                                     ("marginalized_error", "zipf-finetuning-coverage-retention.png", "Coverage error + seen error mass")):
        figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
        for axis, (family, label_) in zip(axes, FAMILIES.items(), strict=True):
            for i, arm in enumerate(("iid", "ranked")):
                values = result["families"][family]["curves"][arm]["metrics"][metric]
                limits = np.array(values["ci95"])[1:]
                axis.plot(BUDGETS, values["mean"][1:], marker="o", color=f"C{i}", label=f"Fine-tune: {'IID' if arm=='iid' else 'ranked'}")
                axis.fill_between(BUDGETS, limits[:, 0], limits[:, 1], color=f"C{i}", alpha=.15)
                axis.plot(BUDGETS, theory["iid_errors" if arm == "iid" else "ranked_errors"][1:], linestyle="--", color=f"C{i}", label=f"Toy: {'IID' if arm=='iid' else 'ranked'}")
            axis.set(title=label_, xscale="log", yscale="log", xlabel="Oracle labels acquired")
        axes[0].set_ylabel(ylabel)
        handles, labels_ = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels_, loc="outside lower center", ncols=2)
        figure.savefig(args.output / filename); plt.close(figure)
    figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for axis, (family, label_) in zip(axes, FAMILIES.items(), strict=True):
        for i, arm in enumerate(ARMS):
            values = result["families"][family]["curves"][arm]["metrics"]["error"]
            coverage = result["families"][family]["curves"][arm]["metrics"]["coverage_error"]
            axis.plot(BUDGETS, values["mean"][1:], marker="o", color=f"C{i}", label=ARMS[arm])
            axis.plot(BUDGETS, coverage["mean"][1:], linestyle="--", color=f"C{i}")
        axis.set(title=label_, xscale="log", yscale="log", xlabel="Oracle labels acquired")
    axes[0].set_ylabel("Test error; dashed: coverage prediction")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    figure.savefig(args.output / "zipf-finetuning-acquisition.png"); plt.close(figure)
    figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for axis, (family, label_) in zip(axes, FAMILIES.items(), strict=True):
        for i, arm in enumerate(ARMS):
            values = result["families"][family]["curves"][arm]["metrics"]["seen_accuracy"]
            limits = np.array(values["ci95"])
            axis.plot(BUDGETS, values["mean"], marker="o", color=f"C{i}", label=ARMS[arm])
            axis.fill_between(BUDGETS, limits[:, 0], limits[:, 1], color=f"C{i}", alpha=.15)
        axis.set(title=label_, xscale="log", xlabel="Oracle labels acquired", ylim=(0, 1.02))
    axes[0].set_ylabel("Accuracy on all observed keys")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    figure.savefig(args.output / "zipf-finetuning-retention.png"); plt.close(figure)
    for clock, filename, xlabel in (("training_presentations", "zipf-finetuning-presentations.png", "Training presentations, including replay"),
                                    ("total_training_and_generation_seconds", "zipf-finetuning-runtime.png", "Training + generation seconds")):
        figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
        for axis, (family, label_) in zip(axes, FAMILIES.items(), strict=True):
            for i, arm in enumerate(ARMS):
                curve = result["families"][family]["curves"][arm]
                if clock == "training_presentations": x = np.array(curve["clocks"][clock]["mean"])
                else: x = np.array(curve["clocks"]["train_seconds"]["mean"]) + np.array(curve["clocks"]["generation_seconds"]["mean"])
                axis.plot(x[1:], curve["metrics"]["error"]["mean"][1:], marker="o", color=f"C{i}", label=ARMS[arm])
            axis.set(title=label_, xscale="log", yscale="log", xlabel=xlabel)
        axes[0].set_ylabel("Weighted test error")
        figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
        figure.savefig(args.output / filename); plt.close(figure)
    figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for axis, (family, label_) in zip(axes, FAMILIES.items(), strict=True):
        for i, arm in enumerate(ARMS):
            values = result["families"][family]["curves"][arm]["metrics"]["functional_overlap"]
            axis.plot(BUDGETS, values["mean"][1:], marker="o", color=f"C{i}", label=ARMS[arm])
        axis.set(title=label_, xscale="log", xlabel="Oracle labels acquired")
    axes[0].set_ylabel("Centered output-probability overlap")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    figure.savefig(args.output / "zipf-finetuning-functional-overlap.png"); plt.close(figure)
    comparison = read(ROOT / "reports/icl-vs-toy-data.json")
    for relative, digest in comparison["source_sha256"].items(): assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
    assert comparison["budgets"] == BUDGETS
    result["icl_comparison_source_sha256"] = hashlib.sha256((ROOT / "reports/icl-vs-toy-data.json").read_bytes()).hexdigest()
    figure, axes = plt.subplots(1, 2, sharey=True, layout="constrained")
    for axis, arm, condition in zip(axes, ("iid", "ranked"), ("iid_study", "active_yoked"), strict=True):
        axis.plot(BUDGETS, theory["iid_errors" if arm == "iid" else "ranked_errors"][1:], linestyle="--", color="0.3", label="Exact linear toy")
        for i, (model, name) in enumerate((("google/gemini-2.5-flash-lite", "Gemini ICL"), ("mistralai/mistral-small-3.2-24b-instruct", "Mistral ICL"))):
            rows = comparison["icl"][model]["curves"]
            axis.plot(BUDGETS, [row[condition]["mean"] for row in rows], marker="o", color=f"C{i}", label=name)
        for i, (family, name) in enumerate(FAMILIES.items(), start=2):
            curve = result["families"][family]["curves"][arm]["metrics"]["error"]
            limits = np.array(curve["ci95"])[1:]
            axis.plot(BUDGETS, curve["mean"][1:], marker="o", color=f"C{i}", label=f"{name} FT")
            axis.fill_between(BUDGETS, limits[:, 0], limits[:, 1], color=f"C{i}", alpha=.15)
        axis.set(title="IID observations" if arm == "iid" else "Ranked observations", xscale="log", yscale="log", xlabel="Oracle labels acquired")
    axes[0].set_ylabel("Weighted test error")
    figure.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=2)
    figure.savefig(args.output / "zipf-finetuning-icl-toy.png"); plt.close(figure)
    icl = {**read(ROOT / "results/zipf_main/summary.json")["models"], **read(ROOT / "results/zipf_replication/summary.json")["models"]}
    figure, axis = plt.subplots(layout="constrained")
    labels_ = ["Toy", "Gemini ICL", "Mistral ICL", "Qwen FT", "SmolLM FT"]
    toy_ = read(ROOT / "results/weight_coverage_toy/simulation.json")["exact_eta1_exponents_n8_to128"]
    for i, (arm, condition, offset) in enumerate((("iid", "iid_study", -.15), ("ranked", "active_yoked", .15))):
        estimates, limits = [toy_[arm]], [[toy_[arm], toy_[arm]]]
        for model in ("google/gemini-2.5-flash-lite", "mistralai/mistral-small-3.2-24b-instruct"):
            fit = icl[model]["effective_exponents"]["windows"][0]["conditions"][condition]
            estimates.append(fit["alpha"]); limits.append(fit["ci95"])
        for family in FAMILIES:
            fit = result["families"][family]["exponent_windows"][0]["fits"][arm]
            estimates.append(fit["alpha"]); limits.append(fit["ci95"])
        values, bounds = np.array(estimates), np.array(limits)
        axis.errorbar(np.arange(5) + offset, values, yerr=np.array([values - bounds[:, 0], bounds[:, 1] - values]), fmt="o", color=f"C{i}", label="IID observations" if i == 0 else "Ranked / same-example observations")
    axis.set(xticks=np.arange(5), xticklabels=labels_, ylabel="Effective exponent over 8–128 labels", title="Matched observations: toy, ICL, weight training")
    figure.legend(*axis.get_legend_handles_labels(), loc="outside lower center", ncols=2)
    figure.savefig(args.output / "zipf-finetuning-exponents.png"); plt.close(figure)
    result["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output / "zipf-finetuning-data.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    save_path = args.root / "analysis.json"
    save_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    for path in args.output.glob("zipf-finetuning-*.png"): print(path.resolve())


if __name__ == "__main__": main()
