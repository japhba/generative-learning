"""Analyze the prospectively specified finite-sample sweep and write reports."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from simulate_finite_sample import ARMS, initial_fisher, initial_weight, metric_hessians, metrics


LABELS = {"self_sampled": "Self-generated labels", "matched_gaussian": "Covariance-matched Gaussian", "isotropic_gaussian": "Isotropic Gaussian"}
COLORS = dict(zip(ARMS, plt.rcParams["axes.prop_cycle"].by_key()["color"][:3]))


def summary(values, critical=1.96):
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return {"mean": mean, "se": se, "ci95": [mean - critical*se, mean + critical*se], "median": float(np.median(values)),
            "q05": float(np.quantile(values, .05)), "q95": float(np.quantile(values, .95))}


def plot_curves(primary, key, ylabel, title, reports, name, multiplier=1):
    fig, ax = plt.subplots()
    for arm in ARMS:
        rows = primary["curves"][arm][key][1:]
        mean = multiplier * np.array([row["mean"] for row in rows])
        lo, hi = multiplier * np.array([row["ci95"] for row in rows]).T
        ax.plot(primary["samples"][1:], mean, label=LABELS[arm], color=COLORS[arm])
        ax.fill_between(primary["samples"][1:], lo, hi, alpha=.2, color=COLORS[arm])
    ax.axhline(multiplier * primary["curves"][ARMS[0]][key][0]["mean"], color="black", linestyle="--", label="Unchanged initial model")
    ax.set(xscale="log", xlabel="Presented inputs", ylabel=ylabel, title=title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / name)
    plt.close(fig)


def make_plots(cases, primary, scalar, reports):
    plot_curves(primary, "error", "Population classification error (%)", "Finite sampling worsens average error in this toy", reports, "finite-sample-error.png", 100)
    plot_curves(primary, "overlap", "Normalized weight overlap", "Weight alignment falls under self-generated noise", reports, "finite-sample-overlap.png")
    plot_curves(primary, "raw_overlap", "Raw overlap with the true signal", "Raw overlap: compare drift with sampling uncertainty", reports, "finite-sample-raw-overlap.png")
    plot_curves(primary, "norm_squared", "Squared weight norm", "Sampling increases the weight norm", reports, "finite-sample-norm.png")

    original = np.load(primary["raw_path"])
    values = metrics(original["weights"])
    delta = 100 * (values["error"][-1] - values["error"][0])
    edges = np.histogram_bin_edges(delta.ravel(), bins="auto")
    fig, ax = plt.subplots()
    for index, arm in enumerate(ARMS): ax.hist(delta[index], bins=edges, histtype="step", density=True, color=COLORS[arm], label=LABELS[arm])
    ax.axvline(0, color="black", linestyle="--")
    ax.set(xlabel="Change in error (percentage points; improvement < 0)", ylabel="Density", title="Some runs improve while the average worsens")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "finite-sample-distribution.png")
    plt.close(fig)

    rows = [(d, rho, batch) for d in (2, 8) for rho in (.3, .8) for batch in (1, 8, 64)]
    etas = [.025, .1, .4]
    grid = np.empty((len(rows), len(etas)))
    stars = np.zeros(grid.shape, dtype=bool)
    for case in cases:
        config = case["case"]
        i = rows.index((config["dimension"], config["initial_overlap"], config["batch"]))
        j = etas.index(config["eta"])
        endpoint = case["endpoint"][ARMS[0]]
        grid[i, j] = 100 * endpoint["error_change"]["mean"]
        lo, hi = endpoint["error_change_simultaneous_ci95"]
        stars[i, j] = lo > 0 or hi < 0
    fig, ax = plt.subplots()
    bound = np.max(np.abs(grid))
    raster = ax.imshow(grid, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    for i, j in np.ndindex(grid.shape): ax.text(j, i, f"{grid[i,j]:+.1f}" + ("*" if stars[i,j] else ""), ha="center", va="center", color="white" if abs(grid[i,j]) > .6*bound else "black")
    ax.set(xticks=range(len(etas)), xticklabels=etas, yticks=range(len(rows)), yticklabels=[f"d={d}, overlap={rho}, B={b}" for d, rho, b in rows],
           xlabel="Learning rate", title="Self-sampling: endpoint change in error (pp)")
    fig.colorbar(raster, ax=ax, label="Positive = worse; * simultaneous 95% CI excludes zero")
    fig.tight_layout()
    fig.savefig(reports / "finite-sample-sweep.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    for j, (batch, eta) in enumerate(((1,.025), (8,.1), (64,.4))):
        case = next(row for row in cases if row["case"]["name"] == f"d8_rho0.3_b{batch}_eta{eta:g}")
        color = plt.rcParams["axes.prop_cycle"].by_key()["color"][j]
        for arm, style in (("self_sampled", "-"), ("matched_gaussian", "--")):
            curve = case["curves"][arm]["error"]
            means = np.array([row["mean"] for row in curve])
            clock = np.array(case["steps"]) * eta**2 / batch
            ax.plot(clock[1:], 100 * (means[1:] - means[0]), linestyle=style, color=color, label=f"{'Self' if arm == 'self_sampled' else 'Gaussian'}: B={batch}, eta={eta:g}")
    ax.axhline(0, linestyle=":", color="black")
    ax.set(xscale="log", xlabel="Diffusion clock: updates × learning rate² / batch size", ylabel="Change in population error (pp)", title="Compare the same noise exposure across settings")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "finite-sample-noise-clock.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    selected = [row for row in scalar["rows"] if row["initial_mean"] == .1]
    steps = [row["step"] for row in selected]
    ax.plot(steps, [row["exact_mean_error"] for row in selected], label="Exact expected error")
    ax.plot(steps, [row["median_error"] for row in selected], label="Median run error")
    ax.axhline(.81, linestyle="--", color="black", label="Initial error")
    ax.set(xlabel="Self-sampled SGD updates", ylabel="Squared error of predicted mean", title="Separate nonlinear control: average and typical differ")
    ax.legend()
    fig.tight_layout()
    fig.savefig(reports / "finite-sample-scalar-control.png")
    plt.close(fig)


def write_reports(result, output, reports):
    primary = result["primary"]
    start = primary["curves"][ARMS[0]]["error"][0]["mean"]
    lines = ["# Equal-temperature finite-sample self-training", "", "The specified RBM classifier did not show a reproducible average benefit from finite self-sampling. Its normalized weight alignment deteriorated. This is a negative result for this model and grid, not an impossibility theorem for nonlinear networks or other optimizers.", "",
             f"The frozen grid contained 36 settings × 3 arms × 512 replicas = 55,296 trajectories. Sampling and training temperatures were both 1, with one plain SGD update per independent batch, no selection, and no external labels. Primary setting: d=8, initial overlap=0.3, norm=1, batch=8, learning rate=0.1, 512 updates (4,096 presented inputs). Initial error was {100*start:.3f}%.", "",
             "| Arm | Final error | Error change, 95% CI (pp) | Final overlap | Fraction improving |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        endpoint = primary["endpoint"][arm]
        delta = endpoint["error_change"]
        lines.append(f"| {LABELS[arm]} | {100*endpoint['error']['mean']:.3f}% | {100*delta['mean']:+.3f} [{100*delta['ci95'][0]:+.3f}, {100*delta['ci95'][1]:+.3f}] | {endpoint['overlap']['mean']:.4f} | {100*endpoint['fraction_error_improved']:.2f}% |")
    counts = result["sweep_counts"]
    lines += ["", f"Across the 36 self-sampling endpoints, {counts['mean_worse']} had a worse sample mean and {counts['mean_better']} a better sample mean. Using simultaneous 95% Bonferroni-adjusted normal intervals across these 36 endpoints, {counts['simultaneous_worse']} showed worsening and {counts['simultaneous_better']} showed improvement. These are Monte Carlo uncertainty intervals over stochastic training, with analytic population evaluation.", "",
              "## Interpretation", "", "Conditional score covariance predicts the initial drift. For the primary setting, the coefficient of eta²/B is " + f"{primary['local_drift']['self_sampled']['error']:.6f} for error and {primary['local_drift']['self_sampled']['overlap']:.6f} for normalized overlap. Thus the second-order effect initially points toward worse error and weaker alignment. Raw overlap is a martingale; norm growth is predicted by accumulated gradient variance.", "",
              "Matched Gaussian noise reproducing similar trajectories supports a gradient-noise explanation. The primary paired self-minus-Gaussian difference is " + f"{100*primary['paired_error_differences']['self_minus_matched']['mean']:+.3f} pp, with 95% CI [{100*primary['paired_error_differences']['self_minus_matched']['ci95'][0]:+.3f}, {100*primary['paired_error_differences']['self_minus_matched']['ci95'][1]:+.3f}]. An interval containing zero is not an equivalence proof. Higher moments can matter at larger steps; all grid comparisons remain in analysis.json and endpoints.csv.", "",
              "The isotropic control also worsens average error. This experiment supplies no evidence that the model's own anisotropic noise straightens its useful latent direction. The fraction of improved individual runs must not be substituted for an average benefit.", "",
              "The same curves are saved against update count, input count, and updates*eta²/B. No power-law exponent is fitted. Token counts, FLOPs, or LLM learning rates are not inferred from these CPU toy operations.", "",
              "## Separate nonlinear sanity control", "", "For p_theta(y)=Normal(theta²,1), the score also has zero mean, but the nonlinear prediction can change in expectation. The exact finite-step curve initially improves when the starting predicted mean is 0.1 and the evaluation truth is 1. Median performance worsens, and the expected error later increases. Starting at the correct mean 1 also worsens. This demonstrates the possibility of a finite-sample effect and the importance of distinguishing average and typical behavior; it does not rescue the RBM hypothesis or model latent straightening.", "",
              "## Validation and provenance", "", "Five numerical tests checked the score, Bernoulli covariance, actual noise controls, Hessians, quadrature convergence, and the scalar control's exact moment formula. All trajectory values were finite. The analysis verifies every raw file hash and the frozen simulation-source hash. The audit in analysis.json compares raw-overlap change and squared-norm growth with their martingale predictions.", "",
              f"Simulation wall time: {result['resources']['wall_seconds']:.2f} seconds using {result['resources']['workers']} local CPU workers. No paid API, rented compute, GPU, or new LLM run. Raw paths and persistent archive hashes are recorded separately. Plot sizes and fonts use matplotlib defaults.", ""]
    markdown = "\n".join(lines)
    (output / "results.md").write_text(markdown)
    Path("experiments/FINITE_SAMPLE_RESULTS.md").write_text(markdown)
    table_rows = []
    for case in result["cases"]:
        config = case["case"]
        cells = [str(config[key]) for key in ("dimension", "initial_overlap", "batch", "eta")]
        for arm in ARMS:
            delta = case["endpoint"][arm]["error_change"]
            cells.append(f"{100*delta['mean']:+.2f} [{100*delta['ci95'][0]:+.2f}, {100*delta['ci95'][1]:+.2f}]")
        table_rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    figures = [("finite-sample-error.png", "Primary setting; shaded bands are pointwise 95% Monte Carlo intervals."),
               ("finite-sample-overlap.png", "Normalized overlap controls the exact population classification error."),
               ("finite-sample-sweep.png", "All predeclared endpoints; stars use simultaneous intervals across 36 self-sampling settings."),
               ("finite-sample-distribution.png", "Individual improvements coexist with a worse average."),
               ("finite-sample-noise-clock.png", "A comparison of diffusion times, not a fit of learning exponents."),
               ("finite-sample-raw-overlap.png", "Mean raw overlap has sampling uncertainty and zero theoretical drift."),
               ("finite-sample-norm.png", "The expected squared norm increases with accumulated score variance."),
               ("finite-sample-scalar-control.png", "Separate nonlinear sanity control, not the RBM experiment.")]
    main_endpoint = primary["endpoint"][ARMS[0]]
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Finite-sample self-training</title><style>body{{font-family:system-ui,sans-serif;max-width:70rem;margin:auto;padding:2rem;line-height:1.5}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:.5em;text-align:right}}pre{{white-space:pre-wrap}}figure{{margin:2em 0}}.scroll{{overflow:auto}}</style>
<h1>Finite-sample self-training at equal temperatures</h1>
<p><strong>The tested RBM classifier loses alignment on average.</strong> Finite sampling moves the weights, but it does not yield a reproducible mean benefit in this specified grid.</p>
<p>Primary setting: error {100*start:.2f}% → {100*main_endpoint['error']['mean']:.2f}%; overlap 0.30 → {main_endpoint['overlap']['mean']:.3f}. {100*main_endpoint['fraction_error_improved']:.1f}% of individual runs improve. All temperatures are 1; no filtering or ground-truth labels enter training.</p>
<p>36 settings, three noise arms, 512 replicas each; analytic population evaluation. {counts['simultaneous_better']} settings show improved mean error and {counts['simultaneous_worse']} show worse mean error under simultaneous 95% intervals. This is a toy result, not a test of all nonlinear networks or LLM training.</p>
''' + "".join(f'<figure><img src="{filename}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>' for filename, caption in figures)
    report += '<h2>All endpoints</h2><p>Entries are change in error in percentage points, with pointwise 95% intervals. Negative means improvement.</p><div class="scroll"><table><thead><tr><th>Dimension</th><th>Initial overlap</th><th>Batch</th><th>Learning rate</th><th>Self-sampled</th><th>Matched Gaussian</th><th>Isotropic Gaussian</th></tr></thead><tbody>' + "".join(table_rows) + '</tbody></table></div>'
    report += '<h2>Detailed interpretation and provenance</h2><pre>' + html.escape(markdown) + '</pre></html>'
    (reports / "finite-sample-self-training.html").write_text(report)


def analyze(output, reports):
    protocol = json.loads((output / "protocol.json").read_text())
    run = json.loads((output / "run.json").read_text())
    scalar = json.loads((output / "scalar_control.json").read_text())
    source = Path(__file__).with_name("simulate_finite_sample.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == protocol["source_sha256"]
    simultaneous_z = NormalDist().inv_cdf(1 - .05 / (2 * len(run["cases"])))
    cases, csv_rows = [], []
    for record in tqdm(run["cases"], desc="Analyze finite samples"):
        raw = Path(record["raw_path"])
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == record["sha256"]
        arrays = np.load(raw)
        w = arrays["weights"]
        assert np.isfinite(w).all()
        values = metrics(w)
        config = record["case"]
        fisher = initial_fisher(w[0,0,0])
        h_rho, h_error = metric_hessians(w[0,0,0])
        local_drift = {}
        for arm in ARMS:
            covariance = fisher if arm != "isotropic_gaussian" else np.eye(config["dimension"]) * np.trace(fisher) / config["dimension"]
            local_drift[arm] = {"error": float(np.trace(covariance @ h_error)/2), "overlap": float(np.trace(covariance @ h_rho)/2), "norm_squared": float(np.trace(covariance))}
        case = {"case": config, "raw_path": str(raw), "steps": arrays["steps"].tolist(), "samples": (arrays["steps"] * config["batch"]).tolist(),
                "diffusion_clock": (arrays["steps"] * config["eta"]**2 / config["batch"]).tolist(), "curves": {}, "endpoint": {}, "audit": {},
                "initial_fisher": fisher.tolist(), "local_drift": local_drift}
        for arm_index, arm in enumerate(ARMS):
            case["curves"][arm] = {key: [summary(checkpoint[arm_index]) for checkpoint in matrix] for key, matrix in values.items()}
            endpoint = {key: summary(matrix[-1,arm_index]) for key, matrix in values.items()}
            delta = values["error"][-1,arm_index] - values["error"][0,arm_index]
            endpoint["error_change"] = summary(delta)
            endpoint["error_change_simultaneous_ci95"] = summary(delta, simultaneous_z)["ci95"]
            endpoint["fraction_error_improved"] = float(np.mean(delta < 0))
            case["endpoint"][arm] = endpoint
            residual = values["norm_squared"][-1,arm_index] - values["norm_squared"][0,arm_index] - arrays["quadratic_variation"][-1,arm_index]
            raw_change = values["raw_overlap"][-1,arm_index] - values["raw_overlap"][0,arm_index]
            case["audit"][arm] = {"norm_martingale_residual": summary(residual), "raw_overlap_change": summary(raw_change),
                                  "expected_norm_growth": float(arrays["quadratic_variation"][-1,arm_index].mean())}
            csv_rows.append({**config, "arm": arm, "initial_error": float(values["error"][0,arm_index,0]),
                             "final_mean_error": endpoint["error"]["mean"], "error_change": endpoint["error_change"]["mean"],
                             "error_change_ci_low": endpoint["error_change"]["ci95"][0], "error_change_ci_high": endpoint["error_change"]["ci95"][1],
                             "final_overlap": endpoint["overlap"]["mean"], "fraction_improved": endpoint["fraction_error_improved"]})
        case["paired_error_differences"] = {"self_minus_matched": summary(values["error"][-1,0] - values["error"][-1,1]),
                                            "self_minus_isotropic": summary(values["error"][-1,0] - values["error"][-1,2])}
        cases.append(case)
    primary = next(case for case in cases if case["case"]["name"] == protocol["primary_case"])
    ends = [case["endpoint"][ARMS[0]] for case in cases]
    counts = {"mean_worse": sum(end["error_change"]["mean"] > 0 for end in ends), "mean_better": sum(end["error_change"]["mean"] < 0 for end in ends),
              "simultaneous_worse": sum(end["error_change_simultaneous_ci95"][0] > 0 for end in ends),
              "simultaneous_better": sum(end["error_change_simultaneous_ci95"][1] < 0 for end in ends)}
    max_audit_z = {key: max(abs(case["audit"][arm][key]["mean"] / case["audit"][arm][key]["se"]) for case in cases for arm in ARMS)
                   for key in ("norm_martingale_residual", "raw_overlap_change")}
    result = {"analyzed_at_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "intervals": "Normal approximation over 512 independent replicas; pointwise 95%, with separate Bonferroni simultaneous 95% intervals over 36 self-sampled endpoints.",
              "simultaneous_critical_value": simultaneous_z, "primary": primary, "cases": cases, "sweep_counts": counts,
              "max_abs_martingale_z_over_108_arm_settings": max_audit_z,
              "resources": {key: run[key] for key in ("wall_seconds", "workers", "case_cpu_wall_seconds_sum")}, "scalar_control": scalar}
    (output / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "endpoints.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    make_plots(cases, primary, scalar, reports)
    write_reports(result, output, reports)
    print(json.dumps({"counts": counts, "primary": primary["endpoint"], "paired": primary["paired_error_differences"], "max_abs_audit_z": max_audit_z}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/finite_sample_20260907"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    analyze(args.output, args.reports)
