"""Canonical, locally served s4g progress and results report."""

import html
import json

import matplotlib.pyplot as plt
import numpy as np

from run_elicitation_pilot import ROOT, save
from train_bestn import stamp


def main():
    root=ROOT/"results/scaling_s4g"
    theory=json.loads((root/"theory/results.json").read_text())
    saved=json.loads((root/"saved_audit/analysis.json").read_text())
    gate=json.loads((root/"gate_forecast/prospective.json").read_text())
    finite=json.loads((root/"finite_sgd/analysis.json").read_text())
    histories={}
    fig,ax=plt.subplots()
    for source in ("current","frozen"):
        directory=root/"arithmetic/pilot/95001"/source
        path=directory/"complete.json"
        complete=path.exists()
        if not complete: path=directory/"progress.json"
        if not path.exists(): continue
        history=json.loads(path.read_text())
        histories[source]={"complete":complete,"history":history}
        records=[r for r in history["evaluations"] if r["step"]>0]
        line,=ax.loglog([64*r["step"] for r in records],[1-r["accuracy"] for r in records],"o-",label=f"{source}: pilot observations")
        prediction=gate["population"][source][1:]
        ax.loglog([r["candidates"] for r in prediction],[1-r["accuracy"] for r in prediction],"--",color=line.get_color(),label=f"{source}: frozen Boltzmann forecast")
    ax.axvspan(64,1024,color="gray",alpha=.15)
    ax.set(xlabel="Training candidate rollouts",ylabel="Held-out sampled error",title="New arithmetic pilot and predeclared forecasts")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ROOT/"reports/scaling-arithmetic-learning.png")
    plt.close(fig)
    fig,ax=plt.subplots()
    rows=theory["rbm"][1:]
    ax.loglog([r["time"] for r in rows],[r["excess_error"] for r in rows],label="Gaussian RBM: population")
    gate_t=np.array(theory["gate"]["time"]); gate_e=np.array(theory["gate"]["error"]); keep=gate_t>0
    ax.loglog(gate_t[keep],gate_e[keep],label="One-logit Boltzmann gate")
    ax.set(xlabel="Population gradient-flow time",ylabel="Error above each known minimum",xlim=(.1,1e8))
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-theory.png");plt.close(fig)
    status_rows=[]
    for source,value in histories.items():
        history=value["history"];last=history["evaluations"][-1]
        status_rows.append(f"<tr><td>{source}</td><td>{history['step']}</td><td>{last['accuracy']:.1%}</td><td>{'Complete' if value['complete'] else 'Running'}</td></tr>")
    finite_rows="".join(f"<tr><td>{r['dimension']}</td><td>{r['batch']}</td><td>{r['eta']}</td><td>{r['beta']:.3f} [{r['beta_ci95'][0]:.3f}, {r['beta_ci95'][1]:.3f}]</td><td>{r['asymptotic_beta']:.3f}</td></tr>" for r in finite["cases"])
    confirmation_path=root/"finite_confirmation/analysis.json"
    confirmation=json.loads(confirmation_path.read_text()) if confirmation_path.exists() else None
    confirmation_html="<p>An independent, longer finite-SGD confirmation is in progress.</p>"
    if confirmation is not None:
        confirmation_html='<img src="scaling-finite-confirmation.png" alt="Independent finite SGD confirmation"><p>Fresh seeds and fixed asymptotic predictions; dashed lines have no fitted amplitude.</p>'
    digit=json.loads((root/"digit/analysis.json").read_text())
    gate_finite=json.loads((root/"gate_finite/analysis.json").read_text())
    class_geometry=json.loads((root/"class_geometry/results.json").read_text())
    dynamic_classes=json.loads((root/"class_geometry/dynamic/results.json").read_text())
    pilot=digit["pilot"]["99101_current"]["history"]
    fig,ax=plt.subplots()
    rows=[r for r in pilot["evaluations"] if r["step"]>0]
    ax.loglog([64*r["step"] for r in rows],[r["error"] for r in rows],"o-",label="Full-weight LLM pilot")
    rows=class_geometry["curve"][1:]
    ax.loglog([64*r["step"] for r in rows],[1-r["accuracy"] for r in rows],"--",label="Three fixed initial gradient modes")
    ax.set(xlabel="Training candidate tokens",ylabel="Held-out error probability",title="A retrospective three-mode reduction also misses the curve")
    ax.legend();fig.tight_layout();fig.savefig(ROOT/"reports/scaling-three-gate.png");plt.close(fig)
    digit_rows=[]
    for key,value in digit["confirmation"].items():
        history=value["history"];last=history["evaluations"][-1]
        digit_rows.append(f"<tr><td>{history['seed']}</td><td>{history['source']}</td><td>{history['eta']:.3g}</td><td>{history['step']}</td><td>{last['accuracy']:.2%}</td><td>{last['greedy_accuracy']:.2%}</td><td>{'Complete' if value['complete'] else 'Awaiting completion'}</td></tr>")
    window_rows="".join(f"<tr><td>{r['start']}–{r['end']}</td><td>{r['beta']:.3f}</td><td>{r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}</td></tr>" for r in digit["bootstrap"]["windows"])
    screens=[]
    for component in ("choice/calibration","digit/calibration","digit/shorter_count_calibration"):
        record=json.loads((root/component/"complete.json").read_text())
        screens.extend(record["summary"])
    screen_rows="".join(f"<tr><td>{r['cell']['name']}</td><td>{r['accuracy']:.1%}</td><td>{r['greedy_accuracy']:.1%}</td><td>{r['valid_mass']:.2%}</td></tr>" for r in screens)
    done=len(digit["confirmation"])==6 and all(r["complete"] for r in digit["confirmation"].values())
    status="Completed bounded study" if done else "Confirmation runs in progress"
    matched=digit["matched_comparison"]
    matched_html=f"<p>At a shared budget of {matched['candidates']:,} generated candidates ({matched['updates']} updates), current-source accuracy is {matched['accuracy']['current']:.2%}, compared with {matched['accuracy']['frozen']:.2%} for frozen-source training. The paired gain is {100*matched['paired_gain']:.2f} percentage points, with a 95% hierarchical interval of {100*matched['gain_ci95'][0]:.2f}–{100*matched['gain_ci95'][1]:.2f} points. This comparison uses identical budgets.</p>"
    current_endpoints=[r["history"]["evaluations"][-1]["accuracy"] for k,r in digit["confirmation"].items() if k.endswith("_current")]
    late_rows="".join(f"<tr><td>{r['seed']}</td><td>{100*r['predicted_error']:.4f}%</td><td>{100*r['observed_error']:.4f}%</td><td>{r['observed_over_prediction']:.3f}</td></tr>" for r in digit["late_forecasts"] if r["gamma"]==2 and r["step"]==2048)
    data={"updated_at_uk":stamp(),"theory":theory["summary"],"saved_audit":saved,"gate_forecast":gate,"finite_sgd":finite,
          "finite_confirmation":confirmation,"arithmetic":histories,"digit":digit,"gate_finite":gate_finite,"class_geometry":class_geometry,
          "dynamic_classes":dynamic_classes,"task_screens":screens,"confirmation_complete":done}
    save(ROOT/"reports/self-generation-scaling-data.json",data)
    (ROOT/"reports/self-generation-scaling-methods.md").write_bytes((ROOT/"experiments/SCALING_THEORY.md").read_bytes())
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Self-generation: scaling and probabilistic mechanisms</title><style>body{{font-family:system-ui,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;line-height:1.55;color:#17212b}}img{{max-width:100%;height:auto}}table{{border-collapse:collapse;width:100%;margin:1rem 0}}th,td{{padding:.5rem;text-align:left;border-bottom:1px solid #dce1e6}}code{{background:#f2f4f6;padding:.1rem .25rem}}.note{{background:#eef4f8;padding:1rem}}a{{color:#165b91}}</style></head><body>
<h1>Self-generation: scaling and probabilistic mechanisms</h1><p>Batch s4g · Updated {html.escape(data['updated_at_uk'])} · {status}</p>
<p class="note">We derived and independently checked scaling laws in normalized probabilistic toys. Full-weight LLM self-training also improves substantially, but its learning curve changes regime and rejects the simplest RBM extrapolation. The shared update identity is exact; a shared constant exponent does not follow from it.</p>
<p>This study uses an external correctness checker. Both LLM conditions generate candidates and update their students. The comparison tests refreshing the generator against keeping its initial weights frozen; it does not isolate generation versus recognition, or establish improvement without feedback. The earlier ICL experiments are a separate setting and were not rerun here.</p>
{matched_html}
<p>The three current-source runs reached a mean sampled accuracy of {np.mean(current_endpoints):.3%} at 2048 updates. Their finite-window exponents are about 1.46 over 128–1024 updates and1.71 over 256–2048 updates; local slopes change substantially, so these fits are not a stable shared toy/LLM law.</p>
<h2>The shared mechanism</h2><p>With current-policy sampling and a binary correctness checker, the expected score of the selected one of two trajectories equals the success gradient. Population gradient flow therefore satisfies <code>dp/dt = ||grad p||²</code>. If squared gradient norm scales as excess error to power gamma, the corresponding error exponent is <code>1/(gamma−1)</code>, under the stated population assumptions.</p>
<p>The one-logit Boltzmann gate gives exponent 1. The existing Gaussian RBM gives exponent 1/2 because its continuous distribution of near-boundary margins changes the gradient geometry. Numerical population exponents are {theory['summary']['gate_beta']:.6f} and {theory['summary']['rbm_beta']:.6f}, respectively, over the predeclared late-time window. Their amplitudes were derived before integration.</p>
<img src="scaling-theory.png" alt="Population scaling in two normalized probabilistic models">
<p>The RBM's normalized weight overlap improves while its norm grows. These are sampled-correctness dynamics, not convergence of joint-density parameters to the true data-generating weights.</p>
<h2>Actual finite-sample toy updates</h2><p>A finite-batch calculation predicts an additional noisy, multi-weight regime with exponent 1/3. This prediction includes tangential sampling noise and its radial drift. The pilot uses actual two-sample conditional draws and full weight updates.</p>
<table><thead><tr><th>Weights</th><th>Batch</th><th>Rate</th><th>Finite-window exponent, 95% interval</th><th>Predicted eventual exponent</th></tr></thead><tbody>{finite_rows}</tbody></table>
<img src="scaling-finite-sgd.png" alt="Finite sample toy scaling"><p>Fits use the final factor8 in steps; intervals bootstrap whole replicas. An intermediate-noise regime may not yet reach its eventual asymptote. Frozen-source controls stopped on stationarity and are retained in the raw data.</p>{confirmation_html}
<p>The longer confirmation uses 512 fresh replicas and262144 updates. Its finite-window exponents are 0.50192 (one weight),0.35314 (eight weights, batch 1), and0.34016 (eight weights, batch 2). Errors at the endpoint divided by the analytically predicted errors are1.00113,0.99710, and1.03500. The confidence intervals describe finite-window slopes; some narrowly exclude the eventual asymptotic exponent. This supports the predicted regimes and amplitudes without claiming exact finite-time powers.</p>
<p>The mechanism predicts weights as well as errors: norm grows as time to power 1/4 in the population regime and1/3 in the noisy regime. Angular error follows the independently derived finite-noise expression. Dashed lines below have no fitted coefficient.</p>
<img src="scaling-rbm-weights.png" alt="Predicted and observed RBM weight-norm growth"><img src="scaling-rbm-alignment.png" alt="Predicted and observed alignment with the true generative weights">
<p>The error criterion matters. Once population weights are aligned, the remaining1/√N sampled-error tail comes from confidence sharpening through norm growth; greedy error is already at its Bayes floor. In the noisy multi-weight regime, angular error dominates, so the 1/N^(1/3) tail also improves greedy decisions. This distinguishes a change in decision quality from a sharper sampling distribution.</p>
<img src="scaling-rbm-greedy.png" alt="Greedy and sampled errors in the noisy RBM">
<p>A separate512-replica finite-sample check of the one-logit gate gives exponents1.00073 and1.00084 for batches 1 and8. Endpoint errors agree with the unfitted inverse-time amplitude within 0.04%. The gate's selected-score variance vanishes as error squared, while the Gaussian RBM retains noise from examples near its boundary. Finite sampling alone therefore does not imply a universal1/3 exponent.</p>
<img src="scaling-gate-finite.png" alt="Actual finite-sample Boltzmann gate updates and inverse-time prediction">
<h2>Why the old arithmetic curves were insufficient</h2><p>All six old trajectories were retained. Shifted powers and exponentials with adjustable floors both describe their short plateaus. Power exponents throughout the tested 0.05–8 range come within 0.5 percentage-point RMSE of the best fit. This is an ambiguity diagnostic, not a confidence interval.</p>
<img src="scaling-saved-audit.png" alt="Power and exponential ambiguity in the old arithmetic experiment">
<h2>New full-weight LLM pilot</h2><p>Qwen2.5-1.5B-Instruct; four-operation small-integer arithmetic; fresh IID problem batches; full FP32 plain SGD; batch 32; rate 1.8291510011e−5; two unrestricted temperature 1 trajectories per question. Both students train, with either their own current weights or a frozen initial copy generating practice. Checkpoints, raw outputs, stopping diagnostics, and complete resume state are retained.</p>
<table><thead><tr><th>Source</th><th>Updates</th><th>Held-out accuracy</th><th>Status</th></tr></thead><tbody>{''.join(status_rows)}</tbody></table>
<img src="scaling-arithmetic-learning.png" alt="Arithmetic learning curves against probabilistic predictions">
<p>The one-weight success-gate forecast uses only initial success and the pre-update training/test gradient overlap. Its numeric forecast was frozen at {html.escape(gate['created_at_uk'])}. Current checkpoints through 16 and frozen checkpoints through 8 already existed. The shaded area conservatively marks budgets through 16 updates; only later observations are prospective for both conditions. The model assumes a stable success-logit metric and an attainable success ceiling of 1. Those assumptions are being tested, not asserted.</p>
<p>Both full-trace runs stopped at 64 updates because selected-trace likelihood went stationary, although correctness was still improving. Current-source accuracy was92.285%, versus its92.272% forecast; frozen-source accuracy was88.379%, versus90.866% forecast. Agreement at the current endpoint is encouraging but one pilot seed and a short curve do not identify an exponent.</p>
<h2>Full-weight counting: exact output probabilities and fresh seeds</h2>
<p>The redesigned task asks how many copies of a digit occur in a four-digit list, with answers0,1,2. Qwen2.5-1.5B-Instruct generates one token from its full vocabulary after the fixed prefix <code>Answer: </code>. There is no vocabulary mask, extra head, temperature change, or adapter: all 1.54 billion weights train in FP32 using plain SGD. Each batch contains32 questions and64 sampled candidates. Uniform correctness ties retain all-failure groups; only the selected sampled tokens enter cross-entropy.</p>
<p>Each seed has1024 distinct held-out questions, excluded from a fresh training stream that retains incidental repeats. Evaluations compute exact correct-token probabilities, eliminating rollout Monte Carlo error but not finite-question uncertainty. A separate initial gradient calibration sets a fixed learning rate for each run; these diagnostic gold gradients never enter learner updates. Full initial conditional-gate forecasts are frozen before training.</p>
<p>The first current/frozen update has been checked for identical candidates, selections, and loss. Training stops when either recent log-error or log-selected-loss slope has magnitude below 0.02. A scratch-disk interruption is recorded separately: prior full checkpoints were retained, scratch copies were relocated to their own durable copies with their old paths preserved as symlinks, and runs resumed with unchanged learning settings. Incomplete checkpoint files and failed attempts are retained.</p>
<table><thead><tr><th>Fresh seed</th><th>Proposal source</th><th>SGD rate</th><th>Updates</th><th>Sampled accuracy</th><th>Greedy accuracy</th><th>Status</th></tr></thead><tbody>{''.join(digit_rows)}</tbody></table>
<img src="scaling-llm-learning.png" alt="All fresh-seed LLM learning curves">
<p>All scaling fits use sampled correctness at unchanged temperature 1. Greedy accuracy is shown separately: increasing confidence can reduce sampled error even after most greedy decisions are correct.</p>
<img src="scaling-llm-greedy.png" alt="Greedy versus sampled LLM accuracy">
<p>The exploratory pilot improved from 64.200% to 99.327% by 1024 updates. Its frozen-source student stopped at 128 updates at 69.074%; these endpoints use different budgets and are not a matched-budget effect estimate. The raw data retain every shared checkpoint for matched comparisons. The fresh-seed table likewise shows actual stopping budgets.</p>
<h2>Why a short straight segment is insufficient</h2>
<p>The counting pilot initially suggested the RBM's cubic gradient/error relation and a1/√N curve. A forecast anchored to its update128 error and gradient overlap was frozen after results through 384 existed. It predicted6.469% error at 512 updates; the subsequently observed error was2.844%. The fresh-seed protocol therefore tests this specific gamma 3 reduction prospectively, alongside gamma 2 and gamma 4 sensitivity alternatives. These alternatives are not fitted exponents.</p>
<img src="scaling-llm-forecasts.png" alt="Predeclared cubic-geometry predictions against fresh observations">
<img src="scaling-llm-exponents.png" alt="Local slopes across candidate budget doublings">
<table><thead><tr><th>Update window</th><th>Mean-curve error exponent</th><th>95% hierarchical interval</th></tr></thead><tbody>{window_rows}</tbody></table>
<p>Intervals resample training seeds and whole held-out questions while preserving pairing across checkpoints. They quantify finite-window slopes, not an identified asymptotic power law. Flexible power-plus-floor and exponential-plus-floor comparisons and all per-seed values are included in the downloadable numerical record.</p>
<h2>A narrower prospective test of the late learning rate</h2>
<p>After the early cubic forecast failed, a secondary protocol used the already recorded update1024 gradient overlap to forecast1536 and2048. All numerical forecasts were saved before those target checkpoints. The primary late hypothesis used quadratic geometry, the one-gate limit; cubic and quartic alternatives are retained in the numerical record.</p>
<table><thead><tr><th>Seed</th><th>Predicted error at 2048</th><th>Observed error</th><th>Observed / predicted</th></tr></thead><tbody>{late_rows}</tbody></table>
<img src="scaling-llm-late-forecasts.png" alt="Secondary prospective forecasts from late gradient measurements">
<p>These endpoint error forecasts are within about 23% in relative terms. They show useful local predictive information, but a twofold window and a refreshed coefficient do not establish a global exponent. Gradient measurements also use a finite512-question observer cohort; exact token probabilities remove action-sampling noise from the diagnostic, not uncertainty over inputs.</p>
<h2>The weight-space connection, and where the simple reduction fails</h2>
<p>For a common cubic relation, the training/test success-gradient overlap divided by error cubed should stabilize. Measuring this quantity directly tests the proposed RBM-to-LLM bridge more strongly than matching slopes by eye.</p>
<img src="scaling-llm-geometry.png" alt="Measured gradient-error coefficient across training">
<p>The LLM's displacement initially aligns with its initial held-out success gradient, but later learning changes direction. This is a measurable weight-space diagnostic. It is not an overlap with known true LLM weights, which we do not have; the RBM's overlap with its known generative parameter is a different observable. Gradient rotation can also occur in a fixed-feature classifier, so this observation alone does not prove that the model learned new features.</p>
<img src="scaling-llm-weights.png" alt="Changes in weight displacement and gradient direction">
<p>Answer groups expose a further failure of a homogeneous gate. In the pilot, correct-count 0 performance rose from 20.7% to 87.7% by 128 updates. Count1 performance was then about 79.0%, and improved sharply later. A single aggregate slope can conceal these successive changes.</p>
<img src="scaling-llm-subtasks.png" alt="Learning curves separated by the correct count">
<h2>Direct evidence from class-gradient overlaps</h2>
<p>A final retrospective probe tests the suggested low-dimensional explanation. The gradients for count 0 and count 1 initially have cosine +0.295, but this changes to −0.195 by 32 updates and −0.588 by 128 updates. An update helping one class can therefore locally hurt the other, while the total update can still improve both through their own gradient terms. This is a concrete change in the measured coupling between learning modes.</p>
<img src="scaling-class-interactions.png" alt="Measured class-gradient interactions change sign">
<p>Three fixed initial class-gradient directions capture only {100*class_geometry['projection_squared']/class_geometry['final_displacement_squared']:.1f}% of the pilot's final squared parameter displacement. A coupled three-gate model using those initial overlaps, initial probabilities and the actual learning rate predicts about 94.0% accuracy at 64 updates, versus 83.4% observed. Its late endpoint is closer, but it misses the intervening behavior. This reduction was designed after seeing the class curves and is explicitly retrospective; it is not presented as an independently validated forecast.</p>
<img src="scaling-three-gate.png" alt="A constant three-mode model fails to reproduce the full LLM curve">
<p>The exact class-level relation is <code>d p_a / d t = sum_b P(b) &lt;grad p_a, grad p_b&gt;</code> within the corresponding population or empirical observer distribution. This gives a principled route from overlap measurements to learning rates. The missing ingredient in the fixed toy reductions is the evolution of these overlaps, not evidence that the selected-score identity fails.</p>
<h2>Task selection and scope</h2>
<p>Before observing short-output learning curves, the screen required correct-token probability between 0.3 and0.9, valid-answer mass at least 0.95, and greedy accuracy at least 0.6. The four-entry count with maximum count 2 was the first declared cell to qualify. Every rejected screen is retained below. A generation audit confirmed that direct logits agree exactly with the standard generation path when repetition penalties and truncation processors are explicitly disabled.</p>
<table><thead><tr><th>Screened task</th><th>Sampled correctness</th><th>Greedy correctness</th><th>Valid-answer mass</th></tr></thead><tbody>{screen_rows}</tbody></table>
<p>The Gaussian RBM has a known irreducible Bayes floor and a continuous margin distribution. The counting task has deterministic answers and a large model whose training direction changes. Those statistical and geometric differences matter: the selected-score identity alone cannot make their exponents equal. The held-out monitoring questions are used for rate calibration, diagnostics and stopping, so they are not an untouched final test set. The one-token experiment also does not establish scaling for multistep reasoning or transfer to another model.</p>
<p>Full-trace candidate counts match while token counts can differ as response lengths change. In the one-token task, learner token budgets track candidate counts; prompt computation is shared for the two independent draws. Evaluations, gradient calibration, checkpoint I/O and recovery overhead are additional experimental costs and are kept separate. No rented instance has been created and this batch has used no paid API calls.</p>
<h2>Related primary work and provenance</h2><p>The tabular softmax1/t connection is consistent with <a href="https://proceedings.mlr.press/v119/mei20b.html">Mei et al. (2020)</a>. Related iterative selection and resource-allocation studies include <a href="https://arxiv.org/abs/2410.20727">Faster WIND</a> and <a href="https://arxiv.org/abs/2603.12151">IsoCompute Playbook</a>. These sources do not validate our new LLM forecasts or establish their novelty.</p>
<p><a href="self-generation-scaling-data.json">Download numerical summary</a> · <a href="self-generation-scaling-methods.md">Derivations and assumptions</a> · <a href="bestn-learning.html">Completed b2k experiment</a></p>
</body></html>'''
    (ROOT/"reports/self-generation-scaling.html").write_text(page)
    print(json.dumps({"updated_at_uk":data["updated_at_uk"],"arithmetic":{s:{"step":v["history"]["step"],"complete":v["complete"]} for s,v in histories.items()}},indent=2))


if __name__=="__main__":main()
