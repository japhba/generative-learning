"""Compare saved Zipf ICL curves with the exact one-layer Bayesian weight toy."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

from zipf_task import ALPHA, BUDGETS, LABELS, VOCABULARY, ideal_curves

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {"iid_study": "IID study", "active_generate": "Active generation", "active_yoked": "Same-example study"}
MODELS = {
    "google/gemini-2.5-flash-lite": "Gemini 2.5\nFlash Lite",
    "mistralai/mistral-small-3.2-24b-instruct": "Mistral\nSmall 3.2",
    "qwen/qwen3-30b-a3b-instruct-2507": "Qwen3 30B\n(irregular)",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    load_dotenv(Path.home() / ".env")
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["FAST_CACHE_DIR"]) / "generative-learning/matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, PercentFormatter, ScalarFormatter

    sources = [ROOT / "results" / run / "summary.json" for run in ("zipf_main", "zipf_replication")]
    summaries = [json.loads(path.read_text()) for path in sources]
    models = {key: value for summary in summaries for key, value in summary["models"].items()}
    raw_sources = [path.with_name("results.jsonl") for path in sources]
    raw = [json.loads(line) for path in raw_sources for line in path.read_text().splitlines()]
    assert set(models) == set(MODELS)
    assert len(raw) == 3 * 32 * 3 * len(BUDGETS)
    for key in tqdm(MODELS, desc="Checking plotted data"):
        model = models[key]
        assert model["world_seeds"] == list(range(4000, 4032))
        assert [point["budget"] for point in model["budgets"]] == BUDGETS
        for point in model["budgets"]:
            for condition in CONDITIONS:
                rows = [row for row in raw if row["model"] == key and row["condition"] == condition and row["budget"] == point["budget"]]
                assert len(rows) == 32
                np.testing.assert_allclose(point[condition]["error"]["mean"], np.mean([row["error"] for row in rows]), rtol=1e-13)

    theory = ideal_curves(BUDGETS)
    for summary in summaries:
        for field in ("iid_errors", "ranked_errors"):
            np.testing.assert_allclose(theory[field], summary["theory"][field], rtol=1e-13)
    toy_path = ROOT / "results/weight_coverage_toy/simulation.json"
    toy = json.loads(toy_path.read_text())
    mask = np.array(BUDGETS) >= 8
    toy_exponents = {policy: float(-np.polyfit(np.log(np.array(BUDGETS)[mask]), np.log(np.array(theory[field])[mask]), 1)[0]) for policy, field in (("iid", "iid_errors"), ("ranked", "ranked_errors"))}
    for policy, exponent in toy_exponents.items():
        np.testing.assert_allclose(exponent, toy["exact_eta1_exponents_n8_to128"][policy], rtol=1e-13)
    fits = {key: next(window["conditions"] for window in models[key]["effective_exponents"]["windows"] if window["budget_range"] == [8, 128]) for key in MODELS}
    smooth_theory = ideal_curves(list(range(1, max(BUDGETS) + 1)))

    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, sharex=True, sharey=True, layout="constrained")
    for axis, (key, label) in zip(axes, MODELS.items(), strict=True):
        for index, (condition, name) in enumerate(CONDITIONS.items()):
            points = [point[condition]["error"] for point in models[key]["budgets"]]
            bounds = np.array([point["ci95"] for point in points])
            axis.plot(BUDGETS, [point["mean"] for point in points], marker="o", color=f"C{index}", label=f"ICL: {name}")
            axis.fill_between(BUDGETS, bounds[:, 0], bounds[:, 1], color=f"C{index}", alpha=0.15)
        for index, (field, name) in enumerate((("iid_errors", "Toy: IID"), ("ranked_errors", "Toy: ranked / same examples"))):
            axis.plot(smooth_theory["budgets"], smooth_theory[field], linestyle="--", color=f"C{index}", label=name)
        axis.set(title=label, xlabel="Observed labels", xscale="log", yscale="log", xticks=[1, 8, 128], yticks=[0.05, 0.1, 0.2, 0.4, 0.8])
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.xaxis.set_minor_locator(NullLocator())
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        axis.yaxis.set_minor_locator(NullLocator())
    figure.suptitle("ICL vs. the linear weight-coverage toy\n32 worlds/model; bands: 95% intervals")
    figure.supylabel("Weighted test error (log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncols=2)
    curve_path = args.output / "icl-vs-toy.png"
    figure.savefig(curve_path)
    plt.close(figure)

    figure, axis = plt.subplots(layout="constrained")
    positions = np.arange(1, len(MODELS) + 1)
    for index, ((condition, label), offset) in enumerate(zip(CONDITIONS.items(), (-0.2, 0., 0.2), strict=True)):
        values = np.array([fits[key][condition]["alpha"] for key in MODELS])
        intervals = np.array([fits[key][condition]["ci95"] for key in MODELS])
        axis.errorbar(positions + offset, values, yerr=np.array([values - intervals[:, 0], intervals[:, 1] - values]), fmt="o", color=f"C{index}", label=label)
        axis.plot(offset, toy_exponents["iid" if condition == "iid_study" else "ranked"], marker="o", linestyle="none", color=f"C{index}")
    axis.set(xticks=range(len(MODELS) + 1), xticklabels=["Linear toy\n(exact)", *MODELS.values()], ylabel=r"Effective exponent $\beta$: error $\propto N^{-\beta}$", title="Same fit range: 8–128 observed labels\nICL: 95% world-bootstrap intervals")
    axis.legend(loc="upper left")
    exponent_path = args.output / "icl-vs-toy-exponents.png"
    figure.savefig(exponent_path)
    plt.close(figure)

    record = {
        "task": {"vocabulary": VOCABULARY, "classes": len(LABELS), "zipf_exponent": ALPHA},
        "budgets": BUDGETS, "icl": {key: {"curves": [{"budget": point["budget"], **{condition: point[condition]["error"] for condition in CONDITIONS}} for point in models[key]["budgets"]], "exponents_n8_to128": fits[key]} for key in MODELS},
        "toy": {"exact_curves": theory, "exponents_n8_to128": toy_exponents, "learning_rate": 1.0},
        "interpretation": "ICL means and pointwise 95% paired-world bootstrap intervals from 32 dictionaries per model. Toy lines are exact finite-vocabulary Bayesian expected classification errors, equal to weighted parameter MSE at unit-step SGD; they are not fits to ICL. Ranked and same-example toy curves coincide. Qwen does not reliably acquire the ranked curriculum. Exponents fit the same five budgets 8–128; irregular curves and window sensitivity prevent a universal scaling-law claim. Counts refer to labels, not tokens or compute.",
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in [*sources, *raw_sources, toy_path, Path(__file__)]},
    }
    (args.output / "icl-vs-toy-data.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    for path in (curve_path, exponent_path): print(path.resolve())


if __name__ == "__main__": main()
