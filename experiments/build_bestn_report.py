"""Canonical local report, including failures and prospective-test boundaries."""

from datetime import datetime
import html
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results/bestn_b2k"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(h))}</th>" for h in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</tbody></table>"


def percent(value): return f"{100 * value:.2f}%"
def pp(value): return f"{100 * value:+.2f} pp"
def ci(values): return f"[{100 * values[0]:+.2f}, {100 * values[1]:+.2f}] pp"
def figure(name, caption): return f'<figure><img src="{name}.png" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'


def main():
    result = json.loads((RESULTS / "analysis.json").read_text())
    gate = json.loads((RESULTS / "gate_prediction/analysis.json").read_text())
    training, final = result["training_analysis"], result["final_evaluation"]
    stage = "Completed experiment" if training["all_six_complete"] and final["complete"] else "Confirmation in progress"
    if final["complete"]:
        s = final["summary"]
        headline = f"At 512 training rollouts, current-source accuracy is {percent(s['current'])}, versus {percent(s['frozen'])} for the frozen source, on 128 untouched questions. The advantage is {pp(s['delta'])}, with hierarchical 95% interval {ci(s['ci95_hierarchical'])}."
        final_section = figure("bestn-final-test", "One fixed checkpoint; 128 untouched questions × eight responses. Gray lines join the three paired training seeds. Error bars resample questions, conditional on these trained models.")
        final_section += table(("Training seed", "Frozen", "Current", "Difference", "Question-bootstrap 95% interval"),
                              [(r["seed"], percent(r["frozen"]), percent(r["current"]), pp(r["delta"]), ci(r["ci95"])) for r in s["seeds"]])
        final_section += f"<p>Untrained accuracy: {percent(s['baseline'])}. Mean current-minus-frozen difference: {pp(s['delta'])}; fixed-run interval {ci(s['ci95_fixed_runs'])}; hierarchical seed/question interval {ci(s['ci95_hierarchical'])}. There are only three training seeds. Questions are shared across conditions, and resampling preserves that pairing.</p>"
    else:
        headline = "Current-source training improves the recurring evaluation curves. The fixed-budget check on untouched questions is still running; its result is not yet claimed."
        final_section = f"<p>{final['completed_conditions']} of seven conditions are complete. Every model uses checkpoint 16: 512 training candidates and 256 presentations. We evaluate the same 128 new questions, with eight responses per question. No result from this set changes the checkpoint, task or training protocol.</p>"
    final_section += "<p>This check was added after observing the earlier benefit, but before generating responses on the new cohort. Its checkpoint and analysis were fixed in advance. It is independent confirmation of a discovered effect, not an original preregistered endpoint.</p>"
    fresh = table(("Task", "N", "Untrained", "After update", "Change", "95% interval"),
                  [("Hard" if r["regime"] == "low" else "Easy", r["n"], percent(r["baseline"]), percent(r["success"]), pp(r["gain"]), ci(r["gain_ci95"])) for r in result["fresh"]])
    local = table(("Task", "N", "Forecast", "Importance gain", "Selection diagnostic"),
                  [(r["regime"], r["n"], pp(r["forecast"]), pp(r["actual"]), pp(r["corrected"])) for r in result["local"]])
    curves = table(("Seed", "Source", "Last update", "Last accuracy"),
                   [(h["seed"], h["source"], h["step"], percent(h["evaluations"][-1]["accuracy"])) for h in result["training"]])
    forecast_table = table(("Source", "Measurement", "Confirmation blocks", "Forecast RMSE", "Zero-gain RMSE"),
                           [(r["source"], r["measurement"], r["blocks"], pp(r["rmse"]), pp(r["zero_forecast_rmse"]))
                            for r in training["forecast_summary"] if r["scope"] == "confirmation"])
    decomposition = table(("Seed", "Source", "End update", "Total gain", "Mode-frequency part", "Conditional-accuracy part"),
                          [(r["seed"], r["source"], r["end_step"], pp(r["total_gain"]), pp(r["gate_frequency_component"]), pp(r["conditional_accuracy_component"]))
                           for r in gate["descriptive_decomposition"]])
    gate_assessment = table(("Source", "Prospective metric", "Checkpoints", "Mean absolute error"),
                            [(r["source"], r["metric"], ", ".join(map(str, r["steps"])), f"{r['mae']:.5f}") for r in gate["prospective_assessment"]])
    costs = table(("Seed", "Source", "Update", "Rollout tokens", "Training input tokens", "Token compute proxy"),
                  [(r["seed"], r["source"], r["step"], f"{r['rollout_tokens']:,}", f"{r['training_input_tokens']:,.0f}", f"{r['compute_proxy']:,.0f}")
                   for r in training["costs"] if r["step"] == 16])
    diagnostic_tokens = sum(h["diagnostic_tokens"] for h in result["training"])
    toy = next(r for r in json.loads((RESULTS / "toy/analysis.json").read_text())["endpoints"] if r["case"]["name"] == "rho0.3_n2")
    norm = json.loads((RESULTS / "norm_confirmation/complete.json").read_text())
    norm_table = table(("Initial overlap", "Arm", "Replicas", "Density change (nats)", "Final classification error"),
                       [(r["rho0"], r["source"], f"{r['replicas']:,}", f"{r['density_change']:+.4f}", percent(r["final_classification_error"])) for r in norm["summaries"]])
    updated = datetime.now(ZoneInfo("Europe/London")).strftime("%d %B %Y, %H:%M %Z")
    document = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Generative learning: selection, source refresh and weight dynamics</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.55;max-width:1000px;margin:auto;padding:2rem;color:#20252b}}img{{max-width:100%;height:auto;display:block}}figure{{margin:1.5rem 0}}figcaption,small{{color:#555}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{text-align:left;padding:.6rem;border-bottom:1px solid #ddd}}pre{{background:#f4f6f8;padding:1rem;overflow:auto;white-space:pre-wrap}}.status{{background:#eef4fa;padding:1rem}}a{{color:#075c9b}}</style></head><body>
<h1>A model can help itself when its samples receive feedback</h1>
<p><small>Research batch b2k · {updated} · {stage}</small></p>
<p class="status">{headline} The exact selection identity and initial weight-gradient overlaps provide a predictive connection to a conditional Boltzmann toy. A one-scalar model explains part of the behavior, but does not accurately predict the entire LLM curve.</p>
<p>The specific LLM claim tested is that <strong>refreshing the model that generates selected training traces improves learning over continuing to sample from its initial weights.</strong> Both conditions generate data and receive the same correctness feedback. This LLM contrast does not establish a benefit from unfiltered self-generation, or an advantage over every recognition or supervised-learning procedure. A separate no-feedback toy extension is reported below.</p>
<h2>Untouched questions at a fixed budget</h2>{final_section}
<h2>What was trained</h2>
<p>The model is <strong>Qwen/Qwen2.5-1.5B-Instruct</strong>, revision <code>989aa7980e4cf806f80c7fef2b1adb7bc71aa306</code>. All approximately 1.54 billion weights are trained in FP32, with TF32 disabled. Plain SGD uses rate 0.00012132033309492577, without adapters, clipping, decay or an adaptive optimizer. Summed completion-token negative log likelihood is divided by a fixed 128 and averaged over 16 selected responses.</p>
<p>Each seed supplies 64 training arithmetic programs and a separate 96-question recurring evaluation set. A program starts with a number and performs 12 additions, subtractions or multiplications with larger integers. The source generates two complete responses per training prompt at temperature 1. A deterministic simulator checks the final answer. A uniformly random maximum-correctness response supplies the training tokens. If both responses fail, a failed response is still selected. Equation formatting never selects examples.</p>
<p>The current source uses the learner’s latest weights. The frozen source generates fresh responses from the original weights throughout training. Prompts, candidate counts, tie schedules, update counts and feedback rules are matched. First candidate/target banks and losses are checked for exact equality within each pair. Test-time answers are sampled without best-of-N selection.</p>
{figure('bestn-training', 'All observed learning curves; individual seeds are shown, with no extension after stopping.')}
{curves}
<p>Runs stop when a recent log-error or log-training-loss slope is stationary, or after 32 updates. We corrected an erroneous requirement that both curves flatten to the intended rule that either can stop training. The first pair was stopped at saved updates 24 and 28; initial and amended protocols are retained. Confirmation training used the corrected rule. Comparisons use common observed budgets; the untouched check always uses update 16.</p>
<h2>The exact probabilistic connection</h2>
<p>For one prompt, let p be success probability and g(Y)=∇log π(Y) the full-trajectory score. With N independent draws and uniform maximum-reward ties, selected success is q<sub>N</sub>(p)=1−(1−p)<sup>N</sup>. Selection preserves the conditional distribution inside each reward class, giving:</p>
<pre>E[g(selected)] = c_N(p) ∇p
c_N(p) = 1 + (1−p) + ... + (1−p)^(N−2)
N = 1: expected update is zero
N = 2: expected update is exactly ∇p</pre>
<p>Across prompts, initial held-out gains depend on K<sub>ij</sub>=⟨∇p<sub>i</sub>,∇p<sub>j</sub>⟩. Positive weight-gradient overlaps transfer gains between questions. At equal candidate count the local per-prompt factor is c<sub>N</sub>(p)/N. N=4 versus N=2 reverses ordering at p≈0.382 in a homogeneous reduction; the actual LLM forecasts use measured overlaps for heterogeneous prompts.</p>
<h2>An RBM with eight interpretable weights</h2>
<pre>E_w(x,y) = ||x||²/2 − y w·x,    y ∈ {{−1,+1}}
Z(w) = 2 (2π)^(d/2) exp(||w||²/2)
p_w(y|x) = exp(y w·x) / [2 cosh(w·x)]
x = y_true μ + Gaussian noise
ρ = (w·μ) / (||w|| ||μ||)</pre>
<p>This Gaussian-visible, binary-hidden Boltzmann model has one weight vector. The same selected-output rule trains its conditional policy. Population drift dw/dt=Aμ+Bw gives dρ/dt=||μ||(1−ρ²)A/||w|| with A&gt;0 for N&gt;1. Selection rotates weights toward the useful direction. A frozen proposal has a fixed selected-target sufficient statistic.</p>
{figure('bestn-toy-error', 'Eight-weight RBM: population predictions were saved before simulation, without fitting a curve time constant.')}
{figure('bestn-toy-overlap', 'Normalized overlap shows the mechanism behind the classification gain.')}
<p>For initial overlap 0.3 and N=2, current-source error reaches 6.71%, versus 22.66% frozen. The difference is −15.95 percentage points, 95% interval [−16.07, −15.84]. This uses 256 replicas, 1,024 updates, batch 32 and rate 0.05. Six cases include N=1 and negative initial overlap. N=1 does not show comparable learning.</p>
{figure('bestn-toy-sampled', 'The same temperature-1 sampled-response metric used for the LLM. These population success forecasts were also saved before the simulation; this additional plot was produced later.')}
<p>Sampled toy accuracy is {percent(toy['current_sampled_success'])} current versus {percent(toy['frozen_sampled_success'])} frozen, a difference of {pp(toy['sampled_success_difference'])}. This metric depends on both weight norm and overlap, while the preceding greedy classification error depends only on overlap.</p>
<p>Binary feedback reveals the label in this toy, so its information complexity differs from checking long reasoning. Conditional reward learning also need not improve marginal density. This is not contrastive-divergence training: CD contrasts data-anchored and model statistics, whereas these updates fit reward-selected conditional samples. See <a href="https://www.cs.toronto.edu/~hinton/absps/tr00-004.pdf">Hinton’s contrastive-divergence paper</a>.</p>
<h2>Predictions made before LLM updates</h2>
<p>Before each four-update block, an independent rollout bank estimates the selected-training gradient. Its overlap with a held-out success gradient predicts the block’s gain. The forecast is saved before drawing training samples. Diagnostic tokens and gradients never enter the optimizer or choose the learning rate during these runs.</p>
{figure('bestn-block-forecasts', 'Prospective four-update forecasts versus paired trajectory-importance measurements. Deviations include sampling noise and finite-step curvature.')}
{forecast_table}
<p>The table uses the two confirmation seeds, reporting both importance estimates and changes in fresh sampled accuracy. It compares forecast error with predicting zero gain. Blocks are correlated, so these aggregate errors are descriptive. Current-source forecasts are more reliable; frozen-source forecasts contain a noisy off-policy ordinary-score component.</p>
<h2>Independent one-update sampling check</h2>
<p>The local probe used 32 training questions, independent forecast/training banks and 64 held-out questions. Hard and easy tasks had 12 and four operations. At equal candidate count, a single mean-loss step was scaled by 16/N; this is a local update-mass comparison, not 16/N sequential SGD updates.</p>
{figure('bestn-fresh', 'Each reconstructed one-step model generated 1,024 new responses. Intervals resample whole questions, conditional on the trained update.')}
{fresh}
<p>This confirms new response draws on the same probe questions, not additional training seeds. The original smaller fresh check was less favorable. Its outputs and the added paired unchanged-model baseline are retained.</p>
{local}
<details><summary>Finite-sample selection diagnostic</summary><p>An N=1 bank has a nonzero realized score despite zero expectation. Its ordinary-score component contributes 1/N of the N=1 update to the N contrast at matched candidates. After inspecting the importance results, we declared this first-order diagnostic before the larger fresh check:</p>
<pre>selection diagnostic = (p_N − p_0) − (p_1 − p_0)/N</pre>
{figure('bestn-local-selection', 'The subtraction is algebraically specified, but the diagnostic was chosen after observing the original data.')}
<p>This local decomposition is not a different trained model. Total gains remain the performance endpoint.</p></details>
<h2>Can one Boltzmann gate explain the curve?</h2>
<p>In the discovery seed, responses containing an equals sign rise from 58.6% to 91.7%, while their conditional correctness stays near 33%. Much of the gain may therefore come from using an existing strategy more often. An equals sign identifies an output mode, not valid reasoning.</p>
<pre>H ~ Bernoulli(λ),   λ = sigmoid(z)
Y | H ~ Q_H          (two fixed normalized distributions)
success = b + (a−b) λ
z_next = z + k [mean selected H − λ_train(z)]</pre>
<p>This normalized Boltzmann factor has one learned scalar. Current-source best-of-2 keeps increasing the gate when a&gt;b; a frozen source approaches its initial selected-mode frequency. Training and evaluation have distinct initial offsets and a shared logit displacement. k=1.3892 is calibrated from the discovery initialization’s pre-update gradient forecast, without fitting its later curve.</p>
<p>The numeric predictions were frozen at 12:48:21 BST on 7 September. Seed 86003 already had current-source checkpoints through update 8. Only later current checkpoints and frozen checkpoints after zero are prospective. The earlier region is shaded. Bands contain only finite-gate SGD variation; they exclude initial-parameter and fresh-evaluation uncertainty.</p>
{figure('bestn-gate-86003-accuracy', 'Frozen scalar predictions and the confirmation LLM. The scalar model underpredicts current-source learning; this is a quantitative limitation.')}
{figure('bestn-gate-86003-mode', 'Equation frequency tests the proposed intermediate mechanism.')}
{figure('bestn-gate-86003-weights', 'Weight displacement along the initial success-gradient direction tests the stronger fixed-direction approximation.')}
{gate_assessment}
{figure('bestn-gate-conditional', 'Conditional correctness motivates the reduction but does not prove that the experts are fixed.')}
{decomposition}
<p>The decomposition splits the product interaction equally between mode-frequency and conditional-accuracy changes. It is descriptive, not causal; conditional accuracy can also change through question composition. No scalar parameter was retuned after confirmation. Failure of this rank-one approximation does not invalidate the exact selection identity.</p>
<h2>Samples and compute</h2>
{figure('bestn-training-compute', 'Actual response lengths enter this token proxy. It is not a FLOP or wall-clock measurement.')}
{costs}
<p>The proxy counts generation tokens plus three times training input tokens. Current-source traces get longer, so equal candidates do not exactly match compute. Padding, attention length and hardware utilization are not modeled.</p>
<p>Recurring evaluation and forecasts generated {diagnostic_tokens:,} additional tokens across training so far. Their substantial diagnostic forward/backward work is excluded from the learner-only axis. Fresh-confirmation and untouched-test costs are also separate. The research procedure costs more than training alone.</p>
<h2>A separate positive no-feedback toy result</h2>
<p>While the last LLM confirmation ran, we tested one explicit inductive bias in the earlier no-feedback toy: project the weights back to their initial unit norm after every self-sampled score update. Sampling and fitting temperatures remain 1, no outputs are filtered, and no correctness feedback enters training. The input examples still come from the real Gaussian mixture.</p>
{figure('bestn-norm-density', 'Fresh confirmation with 16,384 replicas: fixed-norm self-sampling improves unlabeled density fit. Matched Gaussian noise reproduces the effect, implicating its data-dependent covariance.')}
<p>From initial overlap 0.3, marginal cross-entropy improves by 0.1141 nats per input, 95% interval [−0.1164, −0.1117]. The improvement exceeds isotropic noise by 0.0830 nats. Classification still worsens, from 32.64% error to 46.45%; sign reversals preserve mixture density but break label orientation. This is a probabilistic representation benefit, not a better classifier.</p>
{norm_table}
{figure('bestn-norm-limitation', 'The separately frozen limitation case starts from a good representation and deteriorates. This is not a uniformly improving learner.')}
<pre>P = I − w wᵀ,    F = E_x[x xᵀ sech²(w·x)]
b = −P F w − 0.5 tr(P F) w
E[ΔL] ≈ (eta²/B) [∇L·b + 0.5 tr(P F P ∇²L)]</pre>
<p>The initial analytic density-change coefficient is −0.6475 for self-sampled noise, versus −0.1100 for isotropic noise. An independent finite-difference audit agrees within Monte Carlo uncertainty. Normalization prevents the norm growth that spoiled density fit in the unconstrained study, and introduces a second-order weight drift. Thus zero expected score before projection does not imply zero change afterward.</p>
<p>This extension was discovered after the unconstrained failure and then confirmed with fresh seeds. It changes the algorithm through a norm constraint. It is not evidence that an isolated model creates information: real inputs provide the structure, and covariance-matched Gaussian noise reproduces the gain. No corresponding no-feedback LLM benefit has been demonstrated.</p>
<h2>Scope and failed checks</h2>
<p>The supported setting combines a conditional probabilistic learner, full-trace likelihood training, explicit success selection and a current-versus-frozen proposal contrast. The toy has exact overlap dynamics; local LLM overlaps predict part of learning. A single scalar does not yet predict the whole LLM trajectory.</p>
<p>No scaling exponent is measured. The gate’s continuous-time model can have 1/t decay of excess error near its attainable ceiling, while a frozen target approaches a different limit. This is a toy asymptotic. One architecture and arithmetic family cannot establish generality across ICL, humans or other models.</p>
<p>The first BF16 differential probe failed. FP32 restored small-step agreement; a separate generation-versus-scoring audit agreed within 0.001 nats on each tested trajectory. Accuracy may improve while likelihood of a canonical wording worsens. The earlier no-feedback finite-sample study recovered an unsigned axis but worsened classification and density fit; it did not establish useful learning without feedback.</p>
<p>Related algorithms include <a href="https://arxiv.org/abs/2304.06767">RAFT</a>, which fits reward-ranked generations, and <a href="https://arxiv.org/abs/2203.14465">STaR</a>, which bootstraps reasoning with correctness and answer-conditioned regeneration. No novelty claim is made for the basic algorithm or score identity.</p>
<p>Failures, sources, protocols, responses and checkpoint provenance are retained. Checkpoints include optimizer, scheduler, RNG and data position; resume reproduced the next update exactly. This batch used institutional <code>gpu_lowp</code> compute. <strong>Paid API and rental cost: $0.</strong> No repository or public site was pushed.</p>
<p><a href="bestn-data.json">Analysis data</a> · <a href="bestn-code.zip">Reproduction code</a> · <a href="finite-sample-self-training.html">Earlier no-feedback report</a></p>
</body></html>'''
    (ROOT / "reports/bestn-learning.html").write_text(document)
    (ROOT / "reports/bestn-data.json").write_text(json.dumps({"results": result, "gate": gate, "norm_confirmation": norm}, indent=2) + "\n")
    print("http://127.0.0.1:8095/bestn-learning.html")


if __name__ == "__main__": main()
