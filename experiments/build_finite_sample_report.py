"""Combine the frozen grid, precision check, and fresh axis confirmation."""

import hashlib
import html
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss

from analyze_finite_sample import ARMS, COLORS, LABELS, plot_curves, summary


ROOT = Path("results/finite_sample_20260907")
CONFIRMATION = Path("results/finite_sample_confirmation_20260907")
AXIS = Path("results/finite_sample_axis_20260907")
REPORTS = Path("reports")


def marginal_loss(w, nodes):
    """Marginal cross-entropy without the constant independent of w."""
    z, mass = hermgauss(nodes)
    q = np.sum(w**2, axis=-1)
    s = 1.5*w[...,0,None] + np.sqrt(2*q[...,None])*z
    return q/2 - np.sum((np.logaddexp(s, -s) - np.log(2))*mass/np.sqrt(np.pi), axis=-1)


def main():
    grid = json.loads((ROOT / "analysis.json").read_text())
    confirmation = json.loads((CONFIRMATION / "analysis.json").read_text())
    axis = json.loads((AXIS / "analysis.json").read_text())
    protocol = json.loads((AXIS / "protocol.json").read_text())
    for group in confirmation["groups"].values(): assert hashlib.sha256(Path(group["raw_path"]).read_bytes()).hexdigest() == group["sha256"]
    assert hashlib.sha256(Path(axis["raw_path"]).read_bytes()).hexdigest() == axis["sha256"]
    weights = np.load(axis["raw_path"])["weights"]
    loss100 = marginal_loss(weights[[0,-1]], 100)
    loss160 = marginal_loss(weights[[0,-1]], 160)
    quadrature_difference = float(np.max(np.abs(loss100 - loss160)))
    assert quadrature_difference < 1e-4
    marginal = {arm: summary(loss160[-1,index] - loss160[0,index]) for index, arm in enumerate(ARMS)}
    (AXIS / "marginal_likelihood_diagnostic.json").write_text(json.dumps({"status": "Post-hoc evaluation of the already completed axis confirmation; never used in training or selection.",
            "metric": "Change in population marginal cross-entropy, in nats per input; positive is worse. Integrates latent label and Gaussian input analytically up to 160-node quadrature.",
            "max_difference_100_vs_160_nodes": quadrature_difference, "arms": marginal}, indent=2) + "\n")

    plot_curves(axis, "axis_overlap", "Squared cosine to the signal axis", "Finite sampling improves recovery of the signal axis", REPORTS, "finite-sample-axis.png")
    plot_curves(axis, "overlap", "Signed normalized overlap", "The same update loses the signal's orientation", REPORTS, "finite-sample-axis-signed.png")
    plot_curves(axis, "error", "Population classification error (%)", "Axis recovery does not improve this classifier", REPORTS, "finite-sample-confirmed-error.png", 100)
    plot_curves(axis, "sign_flip", "Runs with reversed signal orientation (%)", "A growing fraction of runs reverse the signal", REPORTS, "finite-sample-sign-flips.png", 100)
    fig, ax = plt.subplots()
    rho = weights[-1,...,0] / np.linalg.norm(weights[-1], axis=-1)
    edges = np.histogram_bin_edges(rho.ravel(), bins="auto")
    for index, arm in enumerate(ARMS): ax.hist(rho[index], bins=edges, histtype="step", density=True, color=COLORS[arm], label=LABELS[arm])
    ax.axvline(.3, linestyle="--", color="black", label="Initial signed overlap")
    ax.axvline(0, linestyle=":", color="black")
    ax.set(xlabel="Signed normalized overlap", ylabel="Density", title="Noise changes the distribution of weight directions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "finite-sample-axis-distribution.png")
    plt.close(fig)

    endpoint = axis["endpoint"]
    contrast = axis["paired_axis_self_minus_isotropic"]
    matched = axis["paired_axis_self_minus_matched"]
    pre = confirmation["groups"]["primary"]
    audit = confirmation["groups"]["variance_audit"]
    prior_pair = pre["paired_self_minus_matched"]
    count = grid["sweep_counts"]
    lines = ["# Finite-sample self-training: axis recovery and prediction error", "",
             "Finite sampling produces a reproducible change in representation geometry in this toy. Sign-invariant alignment with the true latent axis improves and is reproduced by covariance-matched Gaussian noise. Signed overlap, mean classification accuracy, and marginal generative likelihood worsen. These outcomes must be kept separate.", "",
             "## What was run", "", "The learner is the one-vector Gaussian-visible/binary-hidden RBM conditional p(y|x)=exp(y w.x)/(2 cosh(w.x)). Inputs follow x=y_true*(1.5,0,...,0)+Normal(0,I), with balanced y_true. The true labels are discarded before training. Each batch supplies fresh, unfiltered self-sampled labels at temperature 1, followed by one plain SGD likelihood update at temperature 1. No clipping, normalization, reward, teacher validation, or selection is used.", "",
             "A frozen sweep varied dimension (2,8), initial signed overlap (0.3,0.8), batch (1,8,64), and learning rate (0.025,0.1,0.4), with 512 updates and 512 replicas per setting. All 36 settings and all three arms were retained. The unchanged initial model is the zero-expected-gradient baseline.", "",
             "The Gaussian control replaces each sampled label residual by a centered Gaussian of the same conditional variance while retaining the input vector. Its batch-average gradient covariance matches exactly at any given weights and input batch. The isotropic control preserves only the covariance trace. Both controls compute these quantities from their own current weights.", "",
             "## Original endpoints and independent precision check", "",
             f"The original primary endpoints were signed overlap and exact population classification error. Of 36 self-sampling settings, {count['mean_worse']} had worse mean error, {count['mean_better']} had better mean error, {count['simultaneous_worse']} showed worsening under simultaneous 95% intervals, and {count['simultaneous_better']} showed improvement. These intervals use Bonferroni correction across the 36 endpoints.", "",
             f"An independent 16,384-replica confirmation of d=8, initial overlap 0.3, norm 1, batch 8, eta 0.1, 512 updates found self-sampled error {100*pre['endpoint']['self_sampled']['error']['mean']:.3f}% from a 32.636% baseline. The self-minus-matched-Gaussian difference was {100*prior_pair['mean']:+.3f} pp [{100*prior_pair['ci95'][0]:+.3f}, {100*prior_pair['ci95'][1]:+.3f}]. Its entire 95% interval was inside the predeclared +/-1 pp practical-equivalence margin.", "",
             "A second 16,384-replica check targeted the largest squared-norm martingale diagnostic in the original grid. The original isotropic residual z=-3.866 became " + f"z={audit['audit']['isotropic_gaussian']['norm_martingale_residual']['mean']/audit['audit']['isotropic_gaussian']['norm_martingale_residual']['se']:.3f}. It remains a modest pointwise fluctuation; it is inside an interval adjusted for the six arm/group norm checks. The conditional covariance identity is exact and was independently checked numerically. The original outlier is retained.", "",
             f"In that d=2, initial-overlap=0.8 check, {100*audit['endpoint']['self_sampled']['fraction_error_improved']:.2f}% of runs improved and median error fell from 11.507% to {100*audit['endpoint']['self_sampled']['error']['median']:.3f}%, while mean error rose to {100*audit['endpoint']['self_sampled']['error']['mean']:.3f}%. A typical-run improvement can coexist with a worse expectation because damaging trajectories have larger effects.", "",
             "## Axis diagnostic: discovery followed by fresh confirmation", "",
             "Inspecting those completed runs suggested a distinction between the signal axis and its sign. The diagnostic rho² was introduced post hoc. Its confirmation protocol was then fixed before generating another 16,384 independent replicas of the original primary setting, using disjoint random seeds. It is not being relabeled as an original endpoint.", "",
             "The metric rho²=(w.mu)²/(||w||²||mu||²) measures alignment with the one-dimensional signal subspace and ignores its sign. Both mu and -mu define the same unlabeled Gaussian mixture. Random isotropic directions have expected rho²=1/d=0.125 here, so an increase from the initial 0.09 alone would not establish useful structure recovery. The isotropic comparison is essential.", "",
             "| Fresh axis-confirmation arm | Mean rho² | Signed overlap | Error | Reversed sign |", "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = endpoint[arm]
        lines.append(f"| {LABELS[arm]} | {row['axis_overlap']['mean']:.5f} | {row['overlap']['mean']:.5f} | {100*row['error']['mean']:.3f}% | {100*row['sign_flip']['mean']:.2f}% |")
    lines += ["", f"The self-minus-isotropic difference in mean rho² was {contrast['mean']:+.5f}, 95% paired interval [{contrast['ci95'][0]:+.5f}, {contrast['ci95'][1]:+.5f}]. The self-minus-matched-Gaussian difference was {matched['mean']:+.5f} [{matched['ci95'][0]:+.5f}, {matched['ci95'][1]:+.5f}], entirely within the predeclared +/-0.01 axis-overlap margin.", "",
              "This supports a covariance-driven effect on the latent signal axis. It does not establish that stochastic target tokens are uniquely necessary: the matched noise control reproduces the endpoint. Inputs still contain the useful mixture structure; the experiment does not generate both inputs and targets from an isolated incorrect model.", "",
              "## Local mechanism", "", "For a smooth metric f and a fresh self-sampled batch, the leading conditional drift is eta²/(2B) * trace(F(w) Hessian(f)(w)), despite E[delta w|w]=0. For rho², its Hessian is 2 grad(rho) grad(rho)^T + 2 rho Hessian(rho). The initial coefficients of eta²/B in the primary setting are:", "",
              f"- Signed overlap: {grid['primary']['local_drift']['self_sampled']['overlap']:.6f}.",
              f"- Classification error: {grid['primary']['local_drift']['self_sampled']['error']:.6f}.",
              f"- Squared axis overlap, self-sampled/matched: {protocol['expected_initial_axis_drift_coefficients']['self_sampled']:.6f}.",
              f"- Squared axis overlap, isotropic: {protocol['expected_initial_axis_drift_coefficients']['isotropic_gaussian']:.6f}.", "",
              "The opposite signs for signed and squared overlap explain why a zero-score argument alone is insufficient to characterize the representation. These are local small-step predictions, not a closed-form global learning curve. Norm also grows; neither true parameter recovery nor calibrated density learning follows from axis alignment.", "",
              "## Generative likelihood check", "", "As an additional post-hoc diagnostic, evaluate the marginal p_w(x), integrating out y. Up to a w-independent constant, its population cross-entropy is ||w||²/2 - E[log cosh(w.x)]. The following endpoint changes are in nats per input; positive is worse:", ""]
    for arm in ARMS:
        row = marginal[arm]
        lines.append(f"- {LABELS[arm]}: {row['mean']:+.5f} [{row['ci95'][0]:+.5f}, {row['ci95'][1]:+.5f}].")
    lines += ["", f"Thus the axis effect also does not imply improved marginal likelihood. Gaussian quadrature used 160 nodes; maximum discrepancy from 100 nodes was {quadrature_difference:.3g} nats over all evaluated replicas.", "",
              "## Scalar sanity control and scope", "", "The separately predeclared model p_theta(y)=Normal(theta²,1) supplies an exact nonlinear example where expected error can initially improve under zero-mean self-sampled scores. Its median error worsens and its expected error later grows. It is an existence check, not evidence for an RBM or LLM benefit.", "",
              "The central empirical finding is a finite-sample improvement in sign-invariant axis recovery, accompanied by worse predictions and density fit. It is a candidate component of a self-learning mechanism, not a successful end-to-end learner. No power-law exponent was fitted, and no conclusion about human or LLM learning rates is claimed.", "",
              "## Reproducibility", "", "The original simulation and five numerical tests were frozen before the sweep; the score, covariance controls, metric Hessians, quadrature, and scalar moment identities passed. The squared-axis Hessian was also checked against independent finite differences. Confirmations use disjoint case seed ranges (1000–1063 and 2000–2031), while the original sweep uses 0–35. Raw weights, accumulated gradient variance, code snapshots, protocols, and hashes are retained; archive.json records persistent copies under CACHE_DIR.", "",
              f"The three simulation phases took {grid['resources']['wall_seconds'] + confirmation['wall_seconds'] + axis['wall_seconds']:.2f} seconds of measured wall time using 8 local CPU workers, excluding analysis and reporting. They produced 202,752 classifier trajectories in total. No paid API, rented compute, GPU, or new LLM training was used. Plot element sizes and fonts use matplotlib defaults.", ""]
    markdown = "\n".join(lines)
    Path("experiments/FINITE_SAMPLE_RESULTS.md").write_text(markdown)
    (ROOT / "combined_results.md").write_text(markdown)
    data = {"grid": grid, "confirmation": confirmation, "axis": axis, "marginal_likelihood": marginal, "quadrature_difference": quadrature_difference,
            "builder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (REPORTS / "finite-sample-data.json").write_text(json.dumps(data) + "\n")
    (REPORTS / "finite-sample-endpoints.csv").write_bytes((ROOT / "endpoints.csv").read_bytes())
    figures = [("finite-sample-axis.png", "Fresh confirmation of the diagnostic discovered after the original sweep. Squared overlap is invariant to reversing the signal."),
               ("finite-sample-confirmed-error.png", "The same fresh runs have worse classification error."),
               ("finite-sample-axis-signed.png", "Signed overlap falls while squared axis overlap improves."),
               ("finite-sample-sign-flips.png", "The label orientation reverses in a growing fraction of runs."),
               ("finite-sample-sweep.png", "Original frozen grid; no simultaneous evidence of improved mean classification error."),
               ("finite-sample-axis-distribution.png", "The complete distribution, including trajectories that reverse the signal."),
               ("finite-sample-noise-clock.png", "A diffusion-clock comparison; no exponent fitting."),
               ("finite-sample-scalar-control.png", "Separate scalar control demonstrates why average and typical outcomes differ.")]
    report = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Finite-sample self-training: axis recovery and error</title>
<style>body{font-family:system-ui,sans-serif;max-width:70rem;margin:auto;padding:2rem;line-height:1.5}img{max-width:100%;height:auto}pre{white-space:pre-wrap}figure{margin:2em 0}table{border-collapse:collapse}td,th{padding:.6em;border:1px solid #ccc;text-align:right}</style>
<h1>Finite sampling improves axis recovery and worsens predictions</h1>
<p>Equal sampling and training temperatures; unfiltered self-generated labels; plain SGD; no external correctness feedback.</p>
<p><strong>The original classification endpoint is negative.</strong> A sign-invariant axis diagnostic, discovered later and confirmed on fresh seeds, shows a reproducible improvement over isotropic noise. Covariance-matched Gaussian noise reproduces this effect. It is a representation effect, with worse classification and marginal likelihood.</p>
<p>The table uses 16,384 fresh replicas per arm in the axis confirmation, each with 512 updates, batch size 8, and learning rate 0.1. Initial squared overlap was 0.09 and initial classification error was 32.636%.</p>
<table><tr><th>Arm</th><th>Squared overlap</th><th>Error</th><th>Reversed sign</th></tr>'''
    for arm in ARMS:
        row = endpoint[arm]
        report += f"<tr><td>{LABELS[arm]}</td><td>{row['axis_overlap']['mean']:.3f}</td><td>{100*row['error']['mean']:.2f}%</td><td>{100*row['sign_flip']['mean']:.1f}%</td></tr>"
    report += '</table><p><a href="finite-sample-data.json">Download data and protocols</a> · <a href="finite-sample-endpoints.csv">Original sweep CSV</a></p>'
    report += "".join(f'<figure><img src="{filename}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>' for filename, caption in figures)
    report += '<h2>Full methods, interpretation, and provenance</h2><pre>' + html.escape(markdown) + '</pre></html>'
    (REPORTS / "finite-sample-self-training.html").write_text(report)
    print(json.dumps({"axis_contrast": contrast, "marginal_likelihood": marginal, "quadrature_difference": quadrature_difference}, indent=2))


if __name__ == "__main__": main()
