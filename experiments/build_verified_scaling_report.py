"""Latest canonical report: verified-only scaling, with the s4g record retained."""

from datetime import datetime
import html
import json
import os
from pathlib import Path

from run_elicitation_pilot import ROOT, save, setup
from train_bestn import stamp


def read(path): return json.loads(path.read_text())


def budget(value):
    if value["status"] == "interpolated": return f"{value['budget']:,.0f}"
    if value["status"] == "not_reached": return f"> {value['lower_bound']:,.0f}"
    return f"≤ {value['upper_bound']:,.0f}"


def main():
    setup(); root = ROOT / "results/scaling_r5s"; confirmation = root / "confirmation"
    protocol = read(confirmation / "protocol.json"); selection = read(confirmation / "selection.json")
    strict = read(confirmation / "analysis_strict.json"); full = read(confirmation / "analysis_test.json"); monitor = read(confirmation / "analysis_monitor.json")
    audit = read(confirmation / "audit.json"); budgets = read(confirmation / "budget_analysis.json")
    clean = read(confirmation / "strict_test_protocol.json"); gain = read(confirmation / "gain_prediction.json")
    toy = read(root / "toy/prospective.json"); toy_check = read(root / "toy/analysis.json"); geometry = read(root / "geometry/results.json")
    bias = read(root / "bias_probe/results.json"); bias_bound = read(root / "bias_probe/bound.json")
    pilot = read(root / "pilot/analysis.json")
    rate_collapse = read(root / "pilot/rate_collapse.json"); weight_laws = read(root / "toy/weight_laws.json")
    addendum = read(confirmation / "analysis_addendum.json")
    slope_diagnostic = read(confirmation / "slope_diagnostic.json")
    shared_skill = read(root / "shared_skill_theory/results.json")
    skill_compatibility = read(root / "shared_skill_theory/llm_compatibility.json")
    histories = []
    for cell in protocol["cells"]:
        directory = confirmation / str(cell["seed"]) / cell["name"]
        history = read(directory / "complete.json"); test = read(directory / "test.json")
        histories.append({"cell": cell, "history": history, "test": test})
    last_finish = max(datetime.fromisoformat(r["history"]["finished_at_uk"]) for r in histories)
    assert len(audit["runs"]) == 6 and all(r["all_selected_tokens_verified"] for r in audit["runs"])
    assert all(r["all_training_complete"] for r in (strict, full, monitor))
    assert all(datetime.fromisoformat(r["test"]["opened_final_test_at_uk"]) > last_finish for r in histories)
    main_point = next(r for r in strict["pointwise"] if r["step"] == 1024)
    last = strict["pointwise"][-1]
    direct = next(r for r in strict["cross_budget"] if r["current_updates"] == 1024 and r["frozen_updates"] == 4096)
    monitor_window = next(r for r in monitor["windows"] if r["start"] == 512 and r["end"] == 4096)
    strict_window = next(r for r in strict["windows"] if r["start"] == 512 and r["end"] == 4096)
    accepted_gain = next(r["speedup"] for r in budgets["axes"]["accepted"]["gains"] if r["error"] == .01)
    optimized_gain = next(r["speedup"] for r in budgets["axes"]["skip_rejected_work"]["gains"] if r["error"] == .01)
    point_rows = "".join(f"<tr><td>{r['step']:,}</td><td>{r['candidates']:,}</td><td>{100*r['current_error']:.3f}%</td><td>{100*r['frozen_error']:.3f}%</td><td>{r['frozen_over_current']:.2f} [{r['ratio_ci95'][0]:.2f}, {r['ratio_ci95'][1]:.2f}]</td></tr>" for r in strict["pointwise"] if r["step"] >= 128)
    crossing_rows = []
    for row in strict["crossings"]:
        ratio = "Censored"
        if "candidate_speedup" in row:
            ratio = f"{row['candidate_speedup']:.2f}"
            if "speedup_ci95" in row: ratio += f" [{row['speedup_ci95'][0]:.2f}, {row['speedup_ci95'][1]:.2f}]"
        crossing_rows.append(f"<tr><td>{100*row['error']:g}%</td><td>{budget(row['current'])}</td><td>{budget(row['frozen'])}</td><td>{ratio}</td></tr>")
    direct_rows = "".join(f"<tr><td>{r['current_updates']}</td><td>{r['frozen_updates']}</td><td>{r['budget_ratio']:g}</td><td>{100*r['current_error']:.3f}%</td><td>{100*r['frozen_error']:.3f}%</td><td>{100*r['frozen_minus_current_error']:.3f} [{100*r['difference_ci95'][0]:.3f}, {100*r['difference_ci95'][1]:.3f}]</td></tr>" for r in strict["cross_budget"])
    pilot_rows = "".join(f"<tr><td>{r['source']}</td><td>{r['eta']:.1e}</td><td>{r['steps']}</td><td>{100*r['final_error']:.3f}%</td><td>{r['log_error_auc']:.3f}</td></tr>" for r in pilot["runs"])
    windows = "".join(f"<tr><td>{r['start']}–{r['end']}</td><td>{r['current_beta']:.3f} [{r['current_ci95'][0]:.3f}, {r['current_ci95'][1]:.3f}]</td><td>{r['frozen_beta']:.3f} [{r['frozen_ci95'][0]:.3f}, {r['frozen_ci95'][1]:.3f}]</td></tr>" for r in strict["windows"])
    endpoints = "".join(f"<tr><td>{r['cell']['seed']}</td><td>{r['cell']['source']}</td><td>{r['history']['step']:,}</td><td>{100*r['history']['evaluations'][-1]['error']:.3f}%</td><td>{r['history']['accepted']:,}</td></tr>" for r in histories)
    geometry_rows = []
    for record in geometry["checkpoints"]:
        current, frozen = [record["sources"][s] for s in ("current", "frozen")]
        geometry_rows.append(f"<tr><td>{record['step']}</td><td>{current['gradient_cosine']:.3f}</td><td>{frozen['gradient_cosine']:.3f}</td><td>{current['predicted_population_gain']/frozen['predicted_population_gain']:.3f}</td><td>{current['observed_population_gain']/frozen['observed_population_gain']:.3f}</td></tr>")
    updated = stamp()
    data = {"updated_at_uk": updated, "protocol": protocol, "selection": selection, "strict_test": strict, "original_test": full, "monitor": monitor,
            "strict_test_definition": clean, "gain_prediction": gain, "toy_prediction": toy, "toy_check": toy_check, "audit": audit, "budgets": budgets,
            "geometry": geometry, "bias_probe": bias, "bias_bound": bias_bound, "pilot": pilot, "rate_collapse": rate_collapse,
            "toy_weight_laws": weight_laws, "analysis_addendum": addendum, "slope_diagnostic": slope_diagnostic, "shared_skill_theory": shared_skill, "shared_skill_compatibility": skill_compatibility,
            "endpoints": [{"cell": r["cell"], "steps": r["history"]["step"], "accepted": r["history"]["accepted"]} for r in histories]}
    save(ROOT / "reports/self-generation-scaling-followup.json", data)
    (ROOT / "reports/self-generation-scaling-followup-methods.md").write_bytes((ROOT / "experiments/VERIFIED_SCALING_THEORY.md").read_bytes())
    (ROOT / "reports/self-generation-scaling-shared-skill-methods.md").write_bytes((ROOT / "experiments/SHARED_SKILL_SCALING_THEORY.md").read_bytes())
    old_path = Path(os.environ["CACHE_DIR"]) / "generative-learning/scaling_s4g/observations/reports/self-generation-scaling.html"
    previous = old_path.read_text().split("<body>", 1)[1].rsplit("</body>", 1)[0]
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Self-generation: sample efficiency and scaling</title>
<style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#17212b}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #dce1e6}}code{{background:#f2f4f6;padding:.1rem .25rem}}.note{{background:#eef4f8;padding:1rem}}a{{color:#165b91}}summary{{cursor:pointer;font-weight:bold}}pre{{white-space:pre-wrap}}</style></head><body>
<h1>Self-generation improves sample efficiency</h1><p>Batch r5s · Updated {html.escape(updated)} · Three fresh training seeds, both methods, separate final evaluation</p>
<p class="note">Refreshing the generator achieves lower held-out error with <strong>four times fewer candidate answers</strong>: {100*direct['current_error']:.3f}% after 65,536 candidates, versus {100*direct['frozen_error']:.3f}% after 262,144 for the frozen-source learner. This comparison uses directly evaluated checkpoints. Both students train only on verified correct examples, and both continue improving.</p>
<p>At the matched budget of 65,536 candidates, strict-test error is {100*main_point['current_error']:.3f}% for current-source training and {100*main_point['frozen_error']:.3f}% for frozen-source training: an error ratio of {main_point['frozen_over_current']:.2f}, with a 95% paired hierarchical interval of {main_point['ratio_ci95'][0]:.2f}–{main_point['ratio_ci95'][1]:.2f}. Error ratios and sample-budget savings are different quantities.</p>
<p>This is feedback-guided self-generation. The model supplies its own candidate answers, and an external correctness check selects usable training examples. Question prompts are IID. The accepted input distribution changes as the generator improves. The result applies to this one-token counting task and model; it does not establish an advantage without validation or over direct uniform gold-label supervision.</p>
<img src="scaling-verified-strict.png" alt="Strict held-out learning curves for all three fresh seeds and both methods">
<h2>A control that can continue learning</h2>
<p>The previous s4g frozen control fitted some incorrect selected answers. Here both methods reject an entire two-candidate group if neither answer is correct. Otherwise they fit a uniformly selected correct candidate. Both students update all model weights. The generator uses either the student's current weights or an unchanged initial copy.</p>
<p>Qwen/Qwen2.5-1.5B-Instruct, revision <code>989aa7980e4cf806f80c7fef2b1adb7bc71aa306</code>; approximately 1.54 billion parameters; full FP32; TF32 off; no dropout, clipping, decay, adapter, additional head or vocabulary mask. Each update uses 32 IID question draws and 64 full-vocabulary temperature-1 candidate tokens after a fixed <code>Answer: </code> prefix. The task counts a target digit in a four-digit list; answers are 0, 1 or 2.</p>
<p>Loss is averaged over accepted examples, so increasing acceptance does not directly multiply the update scale. Rejected rows have zero loss weight, but their forward/backward computation is retained to match work. A symmetric pilot rate search selected the same constant plain-SGD rate, <code>5.6e-6</code>, for both methods. This is the best of three tested rates, not a claim of global optimizer optimality.</p>
<h2>Matched-budget error and savings in samples</h2>
<p>The primary metric is the expected probability of an incorrect sampled answer at temperature 1: the mean of <code>1 − p(correct token)</code>, evaluated exactly from the full vocabulary. The three answer classes receive equal weight. This differs from greedy decision error, which is plotted separately.</p>
<table><thead><tr><th>Updates</th><th>Candidate tokens</th><th>Current error</th><th>Frozen error</th><th>Frozen/current error, 95% interval</th></tr></thead><tbody>{point_rows}</tbody></table>
<p>The intervals resample training seeds and common whole held-out questions within answer strata, preserving method and checkpoint pairing. There are only three training seeds. These are pointwise intervals, without a multiple-comparison correction. Exact token probabilities remove action-sampling noise from evaluation, while uncertainty over inputs and training randomness remains.</p>
<table><thead><tr><th>Target error</th><th>Current candidates</th><th>Frozen candidates</th><th>Frozen/current candidate budget, 95% interval</th></tr></thead><tbody>{''.join(crossing_rows)}</tbody></table>
<p>Crossing budgets interpolate between observed adjacent checkpoints in log budget and log error. They never extrapolate. A bound or censored result means the crossing lies outside the measured positive-budget interval. No interval is reported when any bootstrap sample has a censored crossing.</p>
<p>Those intervals quantify sampling uncertainty, not uncertainty in interpolation between sparse saved checkpoints. The following secondary comparisons use only directly evaluated models and need no interpolation:</p>
<table><thead><tr><th>Current updates</th><th>Frozen updates</th><th>Budget ratio</th><th>Current error</th><th>Frozen error</th><th>Frozen minus current error, percentage points, 95% interval</th></tr></thead><tbody>{direct_rows}</tbody></table>
<p>All five discrete comparisons were specified after viewing monitoring curves and before reading final-test predictions; they are secondary analyses. The fourfold comparison at 1024 versus 4096 updates has an error difference of {100*direct['frozen_minus_current_error']:.3f} percentage points, with interval {100*direct['difference_ci95'][0]:.3f}–{100*direct['difference_ci95'][1]:.3f}. The eightfold comparison's interval crosses zero, so it does not establish an eightfold saving.</p>
<img src="scaling-verified-strict-greedy.png" alt="Strict held-out greedy decision error">
<p>Greedy decision error is shown separately. The later sampled-error tail can include confidence sharpening after greedy answers are already mostly correct.</p>
<h2>Evidence for different finite-range exponents</h2>
<table><thead><tr><th>Update window</th><th>Current exponent, 95% interval</th><th>Frozen exponent, 95% interval</th></tr></thead><tbody>{windows}</tbody></table>
<p>The strict final test supports a steeper current-source learning curve over the measured range. Over updates 512–4096, the exponent difference is {strict_window['beta_difference']:.2f}, with a paired bootstrap interval of {strict_window['difference_ci95'][0]:.2f}–{strict_window['difference_ci95'][1]:.2f}. These secondary estimates provide evidence for a finite-range exponent advantage, in addition to the sample-efficiency gain.</p>
<p>These are finite-window fits of error proportional to budget to a negative power, with zero floor and no fitted time shift. They are descriptive, with window sensitivity retained. A sample-efficiency gain does not require different asymptotic exponents or equality with a toy exponent.</p>
<img src="scaling-verified-error-ratio.png" alt="The frozen-to-current error ratio grows on the strict final test, while its later behavior depends on the evaluation cohort">
<p>This post-hoc diagnostic uses existing evaluations at the same five checkpoints for both cohorts. If both errors were exact power laws with the same exponent, their ratio would be constant. On the strict test it grows from 2.31 to 13.16. The difference between adjacent-interval slopes is smaller late in training: 0.16 over the last doubling, compared with 0.84 over updates 128–512. Pointwise uncertainty bands do not by themselves constitute a test of a constant ratio.</p>
<p>The ordering of fitted exponents also changes across evaluation cohorts. Over updates 512–4096, the strict-test slopes are {strict_window['current_beta']:.2f} for current and {strict_window['frozen_beta']:.2f} for frozen; the monitoring slopes are {monitor_window['current_beta']:.2f} and {monitor_window['frozen_beta']:.2f}, respectively. Both cohorts show a sample-efficiency advantage. A universal exponent improvement is not established.</p>
<h2>Accepted examples and compute accounting</h2>
<img src="scaling-verified-accepted-mix.png" alt="How generator refresh changes the accepted count-class distribution">
<p>The frozen initial generator rarely supplies correct examples from the initially difficult count-0 group. Refreshing brings that group's share of accepted examples toward its one-third share of the IID prompt distribution. Every accepted token was independently audited against its sampled candidate and correct answer; no correct target is supplied when both candidates fail.</p>
<img src="scaling-verified-budget-accepted.png" alt="Strict test curves against accepted examples">
<img src="scaling-verified-budget-skip_rejected_work.png" alt="Strict test curves using a proxy that skips rejected training rows">
<p>Actual forward/backward work is matched by the harness. The second plot uses a counterfactual optimized-filter proxy: proposal tokens plus three times accepted training tokens. It allows a frozen learner to save the work of rejected rows. This is a token-work estimate, not a measured FLOP count. Matching accepted examples also changes optimizer update count, so that axis does not alone isolate intrinsic per-example efficiency.</p>
<p>At 1% error, the interpolated saving is {accepted_gain:.2f} times on accepted examples and {optimized_gain:.2f} times on the optimized-filter work proxy. These secondary estimates use mean curves without new confidence intervals. The candidate advantage therefore persists under both alternative accounting choices; a wall-clock speedup was not measured.</p>
<h2>A normalized probabilistic toy predicts an improvement with the same exponent</h2>
<p>A single layer of independent Bernoulli/Boltzmann modes has linear energies and logistic normalization. Let each mode's initial chance of supplying a correct two-candidate example be <code>q0 = p0(2-p0)</code>. With equally frequent modes and accepted-mean updates, the derived asymptotic laws are:</p>
<pre>current mean error ~ m / t
frozen mean error ~ mean(q0) × sum(1/q0) / t
candidate-budget gain = mean(q0) × mean(1/q0) ≥ 1</pre>
<p>Both exponents are 1. Heterogeneous initial difficulty produces a better coefficient when the generator refreshes. With a single equally difficult mode, accepted-mean population updates coincide. The inequality therefore attributes the gain to heterogeneous usable practice, rather than merely multiplying the loss by a higher acceptance rate.</p>
<p>These gates are exactly conditional probabilities of an RBM with binary visible vector x, one binary hidden answer h, energy <code>E(x,h) = −h w·x</code>, and partition function <code>Z = 2^m + product(1+exp(w_j))</code>, evaluated on one-hot practice inputs. This supplies a normalized probabilistic model with explicit weights. The training objective is the verified conditional-answer objective; joint-density learning and contrastive divergence are separate questions.</p>
<img src="scaling-verified-toy-amplitudes.png" alt="Finite simulations and predicted inverse-budget amplitudes">
<p>For the fixed toy probabilities (0.2, 0.75, 0.95), the predicted gain is {toy['predicted_candidate_speedup']:.3f}. Independent finite-batch simulations retain the predicted amplitudes within about 0.31% at their endpoints. Their finite-window slopes are close to 1, with finite-time deviations fully recorded. The toy is a mechanism demonstration and its modes are independent.</p>
<img src="scaling-verified-toy-overlap.png" alt="Population and finite-batch overlap of mode weights with the equally positive direction">
<p>The toy also has an interpretable weight prediction. In population dynamics, refreshing removes the spread between mode logits as inverse time squared, whereas frozen sampling leaves a limiting spread set by the initial acceptance probabilities. Both weight directions eventually approach the equally positive direction, with different rates. These are overlaps with a balanced solution direction, not a finite true parameter. The unfitted formulas were derived and checked retrospectively on stored trajectories; finite-batch noise changes the faster population alignment tail.</p>
<p>A secondary hypothesis plugs the LLM's initial monitoring probabilities into this coefficient formula, giving {gain['predicted_asymptotic_candidate_speedup']:.2f}, with an initial-question bootstrap interval {gain['initial_question_bootstrap_ci95'][0]:.2f}–{gain['initial_question_bootstrap_ci95'][1]:.2f}. It was frozen at {html.escape(gain['frozen_at_uk'])}, after early confirmation checkpoints through 768 existed and before the 2048/4096 targets. It assumes independent equally weighted latent modes with a common gradient metric. Those assumptions are not established for the LLM; the interval excludes model misspecification, and this eventual prefactor is not an exact finite-budget forecast.</p>
<p>This coefficient formula is not quantitatively validated for the LLM. At the last strict-test checkpoint, the matched-budget error ratio is {last['frozen_over_current']:.2f}, whereas the toy-inspired eventual coefficient hypothesis is {gain['predicted_asymptotic_candidate_speedup']:.2f}. The horizontal saving to a chosen error threshold is a different quantity outside a common inverse-budget asymptote. The established connection is the usable-practice mechanism and a scaling improvement, rather than agreement of all coefficients or exponents.</p>
<h2>A shared skill can change the exponent</h2>
<p>A second construction, developed after the LLM results, uses the same verified-only update but replaces independent mode weights with one shared trainable scalar. Problems have difficulty <code>d ≥ 0</code> drawn IID with density <code>a exp(−a d)</code>. The model's correct-answer probability is <code>sigmoid(theta − d)</code>. Learning theta on easier questions also improves the chance of solving harder ones.</p>
<p>For <code>0 &lt; a &lt; 1</code>, the population dynamics have analytically different exponents:</p>
<pre>frozen generator: mean error ~ constant × t^(−a)
current generator: mean error ~ 2 / [a²(3−a) t]
shared weight: theta_frozen ~ log(t), theta_current ~ log(t)/a</pre>
<p>Time is learning rate times optimizer updates, proportional to candidate budget at fixed batch size and learning rate. Frozen proposals keep supplying relatively easy examples whose remaining gradients shrink. Current proposals supply correct answers near the model's increasing difficulty threshold, sustaining useful practice. Total-acceptance normalization remains present in both arms.</p>
<img src="scaling-shared-skill.png" alt="Population integration agrees with predicted current exponent one and frozen exponent one half in a shared probabilistic skill model">
<p>For the illustrative choice <code>a=1/2</code>, the predicted exponents are 1 for current and 1/2 for frozen. Numerical integration gives 1.000000 and 0.500006 over population times 1e8–1e12, with derived amplitudes also recovered. This is integration of population equations, not a simulation of that many updates or a new LLM run. Greedy error has the same exponent separation in this regime, so the construction is not solely about sharpening confidence.</p>
<p>The model is a normalized conditional Boltzmann model with one linear logit. A binary/continuous joint embedding has energy <code>a d + h(d−theta)</code> and partition function <code>1/a + exp(theta)/(a+1)</code>. Practice inputs still follow the prescribed IID distribution. This does not require joint-density training or contrastive divergence.</p>
<img src="scaling-shared-skill-cutoff.png" alt="The exponent separation persists with unbounded difficulty but disappears at sufficiently large budgets when difficulty is capped">
<p>The assumptions determine the outcome. If difficulty has a fixed upper bound, both sampled-error exponents eventually return to 1. They also agree at 1 when <code>a &gt; 1</code>; the boundary <code>a=1</code> has logarithmic corrections. All four checked settings, numerical-resolution checks, and an exact closed-form check at <code>a=1/2</code> are retained. A broad difficulty range can therefore give a long interval of different slopes even when the final slopes agree.</p>
<p>This is a concrete theoretical mechanism for an exponent improvement under the same checker-guided sampling rule. It is not yet a quantitative theory of the LLM: a shared correctness direction and the relevant difficulty tail have not been established there. The earlier independent-mode toy remains valid under its own assumptions. See the <a href="self-generation-scaling-shared-skill-methods.md">complete derivation, assumptions and numerical checks</a>.</p>
<p>The toy is a binary success model, not an implementation of counting. A direct compatibility check also rejects its literal one-skill description of the LLM: such a model would preserve every question's probability rank. At update 512, the initial-to-current rank correlation on 1023 already exposed monitoring questions is 0.466–0.506 across current-source seeds, and 0.848–0.862 across frozen-source students, instead of 1. The logit gains vary across questions, and rankings change within answer classes as well. All probability pairs have finite logits at this checkpoint. This is compatible with richer shared learning, but the scalar toy alone does not quantitatively describe the LLM.</p>
<h2>Weight-space diagnostic: useful, but not a complete explanation</h2>
<p>A read-only pilot probe compares current and frozen accepted-data gradients at the same model weights. It measures their overlap with the held-out success gradient, then applies one population step diagnostically and restores the weights.</p>
<table><thead><tr><th>Pilot step</th><th>Current cosine</th><th>Frozen cosine</th><th>Predicted gain ratio</th><th>Observed population-step gain ratio</th></tr></thead><tbody>{''.join(geometry_rows)}</tbody></table>
<p>Current-source gradients have a higher cosine at the later probed states, but their magnitudes differ. At step 512 the frozen gradient actually predicts a larger immediate gain. Gradient cosine alone therefore does not explain the accumulated learning-curve advantage. The changing accepted input mix is directly measured; a quantitative account of the full LLM weight dynamics remains incomplete.</p>
<p>A further read-only pilot control fits global answer-logit biases while keeping the pretrained model fixed, granting a restricted answer set of 0/1/2. Its greedy monitoring accuracy is {100*bias['after']['monitor']['greedy_accuracy']:.2f}%. Pairwise logit-margin constraints certify that no global answer-bias correction can exceed {100*bias_bound['global_bias_greedy_accuracy_upper_bound']:.2f}% on this cohort: at least {bias_bound['unavoidable_multiclass_errors']} questions must remain wrong. Full weight training goes beyond that correction. This does not distinguish changes to hidden features from changes to an input-dependent readout.</p>
<h2>Selection, held-out data and all runs</h2>
<img src="scaling-verified-pilot.png" alt="All six exploratory learning-rate pilot curves">
<img src="scaling-verified-rate-collapse.png" alt="Pilot learning curves measured against learning rate times updates">
<p>Rescaling the pilot's horizontal axis by learning rate checks whether its within-method trajectories behave like a common small-step learning process. All three rates remain visible. This is a descriptive pilot check, separate from the fresh-seed effect estimate.</p>
<table><thead><tr><th>Source</th><th>Rate</th><th>Pilot updates</th><th>Monitoring error</th><th>Log-error area</th></tr></thead><tbody>{pilot_rows}</tbody></table>
<p>The rate criterion was the mean area under log monitoring-error over log budget from updates 128 to 1024, independently for each method. All six pilot trajectories are retained. Confirmation uses seeds 99601, 99602 and 99603 with fixed selected rates and the predeclared horizon of 4096 updates.</p>
<table><thead><tr><th>Seed</th><th>Source</th><th>Completed updates</th><th>Final monitoring error</th><th>Accepted examples</th></tr></thead><tbody>{endpoints}</tbody></table>
<p>Each stage reserves 1023 monitoring questions and 4095 final-test questions. Training keeps the three answer counts equally likely, samples within each class conditional on avoiding held-out literal IDs, and retains incidental repeats. The final pool is excluded from all confirmation training and monitoring.</p>
<p>An identity audit found 138 final-pool questions overlapping the pilot's rate-selection monitor. Before any final-test model outputs existed, an additional strict subset retained all 2444 questions absent from both pilot training and pilot monitoring: 1130 count-0, 898 count-1 and 416 count-2 questions. It uses an equal-weight mean of class-conditional errors and a stratified paired bootstrap. The subset was defined after early confirmation monitoring but solely by IDs; all original-pool results remain available. Final-test evaluation began only after every confirmation trajectory was complete.</p>
<p>This is an exact-input holdout within r5s, not a claim of semantic out-of-distribution generalization or absence of related inputs from pretraining and earlier project explorations. The original pool and monitoring curves provide sensitivity checks:</p>
<img src="scaling-verified-test.png" alt="Original final-pool sensitivity analysis"><img src="scaling-verified-monitor.png" alt="Monitoring-cohort sensitivity analysis">
<h2>Execution and reproducibility</h2>
<p>Training ran on institutional gpu_lowp GPUs. No paid API call or rented instance was used. Full weights, optimizer, scheduler, RNG states, counters, data position and history were saved periodically to node-local storage, then asynchronously moved to SHA-verified durable checkpoints. Raw candidates, rejected groups, all rates, source snapshots, protocols, test records and diagnostics are retained.</p>
<p>Evaluation was regular throughout training. From update 64 onward, the latest four log-error and log-accepted-NLL checkpoints triggered a stop if either slope's magnitude fell below 0.02. A degraded or nonfinite learner fails loudly. All stopping outcomes are included; no stationary run was resumed to extend a curve.</p>
<p>Related filtered self-training methods include <a href="https://arxiv.org/abs/2304.06767">RAFT</a> and <a href="https://arxiv.org/abs/2203.14465">STaR</a>. These sources motivate the procedure; they do not establish the new numerical result or its novelty.</p>
<p><a href="self-generation-scaling-followup.json">All numerical results</a> · <a href="self-generation-scaling-followup-methods.md">Toy derivation and assumptions</a></p>
<details><summary>Earlier s4g study: RBM laws, initial LLM scaling and failed fixed-geometry forecasts</summary>{previous}</details>
</body></html>'''
    (ROOT / "reports/self-generation-scaling.html").write_text(page)
    print(json.dumps({"updated_at_uk": updated, "canonical_url": "http://127.0.0.1:8095/self-generation-scaling.html", "last_shared_updates": last["step"], "strict_questions": 2444}))


if __name__ == "__main__": main()
