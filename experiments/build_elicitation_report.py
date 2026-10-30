"""Standalone report for the elicitation, linear-toy and consolidation pilots."""

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
LABELS = {"frozen_reasoning": "Frozen reasoning source", "on_policy_reasoning": "Current reasoning source", "direct_self_imitation": "Direct self-imitation"}


def read(path):
    return json.loads((ROOT / path).read_text())


def figure(name, caption):
    encoded = base64.b64encode((REPORTS / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def percent(value):
    return f"{100 * value:.1f}%"


def build():
    elicitation = read("results/elicitation_pilot/general_confirmation/confirmation/general/explicit_answer_analysis.json")
    e = elicitation["summary"]["arithmetic"]
    toy = read("results/elicitation_toy/analysis.json")
    refresh = read("results/posterior_refresh/analysis.json")
    learned = read("results/elicitation_consolidation/analysis.json")
    execution = read("results/elicitation_consolidation/execution.json")
    traces = read("results/elicitation_traces/analysis.json")
    trace_execution = read("results/elicitation_traces/execution.json")
    confirmation = read("results/elicitation_trace_confirmation/analysis.json")
    confirmation_execution = read("results/elicitation_trace_confirmation/execution.json")
    resources = read("results/elicitation_trace_confirmation/resources.json")
    assert learned["audit"]["cases"] == 9 and execution["workers_completed"]
    assert traces["audit"]["cases"] == 3 and trace_execution["workers_completed"]
    assert confirmation["audit"]["cases"] == 6 and confirmation_execution["workers_completed"]
    fig, ax = plt.subplots()
    labels = ["Short answer", "One reasoning\ntrace", "Four-trace\nconsensus"]
    values = [e[key] for key in ("direct_greedy", "reason_greedy", "consensus")]
    ax.bar(labels, values)
    ax.set(ylabel="Accuracy on 144 fresh questions", ylim=(0,1), title="The same weights contain more usable competence")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.tight_layout()
    fig.savefig(REPORTS / "elicitation-gap.png")
    plt.close(fig)
    for name in ("linear-elicitation.png", "linear-elicitation-gap.png"):
        shutil.copyfile(ROOT / "results/elicitation_toy" / name, REPORTS / name)
    for name in ("posterior-refresh-mse.png", "posterior-refresh-normalized_overlap.png"):
        shutil.copyfile(ROOT / "results/posterior_refresh" / name, REPORTS / name)
    fig, ax = plt.subplots()
    for arm, label in (("sleep", "Own generator → recognition"), ("self_imitation", "Imitate own recognition"), ("wrong_generator", "Incorrect internal generator")):
        rows = toy["records"][1:]
        values = np.array([r["arms"][arm]["normalized_overlap"] for r in rows])
        x = [r["generated_pairs"] for r in rows]
        line, = ax.plot(x, values.mean(axis=1), label=label)
        se = values.std(axis=1, ddof=1) / np.sqrt(len(values[0]))
        ax.fill_between(x, values.mean(axis=1)-1.96*se, values.mean(axis=1)+1.96*se, color=line.get_color(), alpha=.2)
    ax.set(xscale="log", xlabel="Internally generated training pairs", ylabel="Normalized overlap with optimal inverse",
           title="How recognition weights approach the internal target")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "linear-elicitation-overlap.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    baseline = learned["cases"][0]["checkpoints"][0]
    for representation in ("answers", "traces"):
        rows = [row for row in learned["icl"] if row["representation"] == representation]
        for mode, style in (("direct", "-"), ("reason", "--")):
            x = [0, *[row["budget"] for row in rows]]
            y = [baseline["metrics"][mode]["accuracy"], *[row["metrics"][mode]["accuracy"] for row in rows]]
            ax.plot(x,y,marker="o",linestyle=style,label=f"{representation.capitalize()} in context · {mode}")
    ax.set(xlabel="Self-generated practice examples in context", ylabel="Held-out accuracy", ylim=(0,1), title="Exploratory ICL transfer of self-generated practice")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.2))
    fig.tight_layout()
    fig.savefig(REPORTS / "elicitation-icl.png")
    plt.close(fig)
    final_table = []
    for arm, label in LABELS.items():
        rows = learned["curves"][arm]
        selected = [c for c in learned["cases"] if c["arm"] == arm]
        practice_accuracy = np.mean([c["checkpoints"][-1]["practice"]["accuracy"] for c in selected])
        self_agreement = np.mean([c["checkpoints"][-1]["practice"]["latest_self_target_agreement"] for c in selected])
        final_table.append(f"<tr><td>{label}</td><td>{percent(rows['direct'][-1]['accuracy'])}</td><td>{percent(rows['reason'][-1]['accuracy'])}</td><td>{percent(practice_accuracy)}</td><td>{percent(self_agreement)}</td></tr>")
    contrast = learned["contrasts"]["direct"]
    trace_labels = {"frozen_reasoning": "Frozen reasoning source", "on_policy_reasoning": "Current reasoning source", "raw_reasoning_self": "Unfiltered reasoning self-imitation"}
    trace_baseline = traces["curves"]["frozen_reasoning"]["reason"][0]
    trace_contrast = traces["contrasts"]["reason"]
    confirmation_contrast = confirmation["contrasts"]["reason"]
    confirmation_baseline = confirmation["curves"]["frozen_reasoning"]["reason"][0]
    confirmation_labels = {arm:LABELS[arm] for arm in ("frozen_reasoning", "on_policy_reasoning")}
    confirmation_table = "".join(
        f"<tr><td>{label}</td><td>{percent(confirmation['curves'][arm]['reason'][-1]['accuracy'])}</td>"
        f"<td>{', '.join(percent(v) for v in confirmation['curves'][arm]['reason'][-1]['seed_accuracies'])}</td>"
        f"<td>{percent(confirmation['curves'][arm]['direct'][-1]['accuracy'])}</td>"
        f"<td>{confirmation['curves'][arm]['reason'][-1]['mean_output_tokens']:.1f}</td></tr>"
        for arm,label in confirmation_labels.items())
    trace_table = "".join(
        f"<tr><td>{label}</td><td>{percent(traces['curves'][arm]['reason'][-1]['accuracy'])}</td>"
        f"<td>{100*traces['baseline_contrasts'][arm]['reason']['final_minus_baseline']:+.1f} pp</td>"
        f"<td>{percent(traces['curves'][arm]['direct'][-1]['accuracy'])}</td>"
        f"<td>{traces['curves'][arm]['reason'][-1]['mean_output_tokens']:.1f}</td>"
        f"<td>{percent(traces['curves'][arm]['reason'][-1]['completion'])}</td></tr>"
        for arm,label in trace_labels.items())
    trace_figures = [figure(name, caption) for name,caption in (
        ("elicitation-traces-reason.png", "One exploratory optimization seed and 144 new test questions. Bands resample paired questions within program length; they exclude training-seed uncertainty."),
        ("elicitation-traces-tokens.png", "Actual practice generation tokens plus training input tokens. These operations have different costs, so this sum is a work counter rather than FLOPs. Evaluation work is recorded separately."),
        ("elicitation-traces-teacher.png", "Correctness is measured offline after training. No gold answers or verifier rewards select these targets."))]
    toy_end = toy["records"][-1]["arms"]
    def toy_metric(arm, key): return float(np.mean(toy_end[arm][key]))
    table_icl = "".join(f"<tr><td>{row['budget']}</td><td>{row['representation']}</td><td>{percent(row['metrics']['direct']['accuracy'])}</td><td>{percent(row['metrics']['reason']['accuracy'])}</td><td>{percent(row['metrics']['direct']['completion'])}</td></tr>" for row in learned["icl"])
    figures = [figure("elicitation-gap.png", "No weight updates or new labels. The high-compute procedures use only the model's own reasoning. Consensus is not selected using the correct answer."),
               figure("elicitation-learning-direct.png", "Three optimization seeds and a separate 144-question test set. Bands resample seeds and questions, stratified by program length. Only 96 unique practice questions are revisited."),
               figure("elicitation-learning-reason.png", "Longer reasoning is evaluated separately after training the short-answer policy, so any loss of the model's original reasoning capability is visible."),
               figure("elicitation-target-quality.png", "Targets are graded only for this diagnostic after all training. Their correctness never determines selection or updates."),
               figure("elicitation-icl.png", "One exploratory demonstration sequence. Answers and traces come from the same selected model-generated candidates; neither is checked against a gold answer."),
               figure("linear-elicitation.png", "A fixed linear Gaussian generator can train a more accurate inverse. An incorrect generator can consolidate incorrect latent semantics."),
               figure("linear-elicitation-gap.png", "The population recurrence is exact; finite-batch SGD approaches a noise floor. No power-law exponent is fitted.")]
    provenance = {"elicitation": elicitation, "consolidation": learned, "traces": traces, "confirmation": confirmation, "toy": toy, "posterior_refresh": refresh,
                  "execution": execution, "trace_execution": trace_execution,
                  "confirmation_execution": confirmation_execution,
                  "resources": resources,
                  "amendments": [read("results/elicitation_pilot/amendment_1.json"), read("results/elicitation_pilot/amendment_2.json")]}
    data = json.dumps(provenance).replace("</", "<\\/")
    page = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Self-generated practice and the elicitation gap</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.55;max-width:80ch;margin:3em auto;padding:0 1em;color:#20242a}}h1,h2{{line-height:1.2}}figure{{margin:2em 0}}img{{max-width:100%;height:auto}}figcaption{{color:#555}}table{{border-collapse:collapse;width:100%;margin:1em 0}}th,td{{text-align:left;padding:.5em;border-bottom:1px solid #ddd}}code{{overflow-wrap:anywhere}}.note{{border-left:solid #697b8c;padding-left:1em}}a{{color:#145d93}}button,select{{font:inherit}}details{{margin:1em 0}}</style>
<body><nav><a href="zipf-finetuning.html">Earlier coverage experiment</a> · <a href="generation-and-learning.html">Earlier learning report</a></nav>
<h1>Self-generated practice and the elicitation gap</h1>
<p>Qwen2.5-1.5B-Instruct · fixed arithmetic questions · linear Gaussian mechanism · September 6, 2026</p>
<p>The same initial model answered <strong>{percent(e['direct_greedy'])}</strong> of fresh questions correctly with a short answer, <strong>{percent(e['reason_greedy'])}</strong> with one reasoning trace, and <strong>{percent(e['consensus'])}</strong> with four-trace consensus. This confirms a gap in access to existing competence under the specified response procedures.</p>
<p><strong>Answer-only consolidation failed.</strong> Across three optimization seeds, final-answer training produced almost no accuracy on new questions and sharply reduced reasoning under the reasoning prompt. Fresh self-produced targets then became mostly wrong. Retaining full self-generated traces in context did help: reasoning accuracy rose from 84.0% to 94.4% in an exploratory check.</p>
<p>A new full-trace training recipe tests whether reasoning survives when its actual computations are retained. From a common {percent(trace_baseline['accuracy'])} reasoning baseline, final accuracy is <strong>{percent(traces['curves']['frozen_reasoning']['reason'][-1]['accuracy'])}</strong> with a frozen source and <strong>{percent(traces['curves']['on_policy_reasoning']['reason'][-1]['accuracy'])}</strong> with a current source. This follow-up has one optimization seed; its full results and limits appear below.</p>
<p>The unchanged full-trace recipe was then repeated on new practice and evaluation questions with three optimization seeds. From a {percent(confirmation_baseline['accuracy'])} baseline, final reasoning accuracy averaged <strong>{percent(confirmation['curves']['on_policy_reasoning']['reason'][-1]['accuracy'])}</strong> with the current source and <strong>{percent(confirmation['curves']['frozen_reasoning']['reason'][-1]['accuracy'])}</strong> with the frozen source. The paired difference is <strong>{100*confirmation_contrast['on_policy_minus_frozen']:+.1f} percentage points</strong>, 95% bootstrap interval <strong>[{100*confirmation_contrast['ci95'][0]:+.1f}, {100*confirmation_contrast['ci95'][1]:+.1f}]</strong>.</p>
<p><strong>The current-source advantage repeated in all three seeds, with a smaller effect than in the exploratory pilot.</strong> This supports repeated elicitation through self-generated reasoning in this setting. It does not show that unfiltered self-imitation helps, that short-answer reasoning has been compressed, or that the learning curves have different scaling exponents.</p>
<p class="note">These studies separate elicitation, consolidation, and policy refresh. A successful high-compute answer is not itself a learning result. A better learned answer is not automatically an on-policy advantage. No scaling exponents are fitted here.</p>
<h2>Existing information, more useful computation</h2>
<p>Each arithmetic program supplies a starting integer and 4, 6, or 8 addition, subtraction, or multiplication operations. All rules are already in the question. The model receives no oracle corrections. Exact answers are used only by an offline evaluator. Plurality selection uses completed answers, includes mistakes, and breaks ties by first occurrence.</p>
{figures[0]}
<p>The confirmation uses 144 fresh questions, 48 at each depth. The consensus gain over the short response is {100*e['gain']:.1f} points, 95% paired interval [{100*e['gain_ci95'][0]:.1f}, {100*e['gain_ci95'][1]:.1f}]. All short and sampled reasoning responses completed. Short answers permit 96 new tokens; reasoning permits 512 per attempt. This is an intentionally different inference-compute budget, not a comparison at equal inference cost.</p>
<details><summary>Calibration failures and answer-extraction amendments</summary><p>The initially screened math-specialized model ignored the short-answer instruction and exhausted the 96-token budget on every calibration response. Its apparent gap was an interface failure, and those runs are retained separately. The general model obeyed short mode. Inspection then found valid explicit final integers without LaTeX boxes; a parser extension was frozen before this fresh confirmation. Original raw responses and strict-box scores remain intact. The alternate permutation-machine task did not qualify. Neither a useful-looking curve nor an exponent was a selection criterion.</p></details>
<h2>Answer-only training suppressed reasoning</h2>
<p>Each arm starts from the same pinned Qwen2.5-1.5B-Instruct weights and sees the same 96 new practice questions, in four shuffled passes. Three seeds vary the ordering and sampling. A separate 144-question test set was frozen before training.</p>
<ul><li><strong>Frozen reasoning source:</strong> four fresh reasoning samples from the initial model for each scheduled presentation; train the short-answer policy on their plurality answer.</li><li><strong>Current reasoning source:</strong> the same procedure, with the source weights refreshed after each 16-question batch.</li><li><strong>Direct self-imitation:</strong> sample a short response and train on its actual tokens, without correctness selection.</li></ul>
<p>All arms update full model weights with AdamW, learning rate 1e-5, four updates per 16-question batch, FP32 weights/state and BF16 autocast. Completion-token NLL is divided by a fixed normalization; the ordinary LM output head is unchanged. The main pair has matched question, rollout-count, and optimizer schedules. Response lengths and actual compute still differ and are retained in the data.</p>
<table><thead><tr><th>Final condition</th><th>Test: short</th><th>Test: reasoning</th><th>Practice: true answer</th><th>Practice: latest self-target</th></tr></thead><tbody>{''.join(final_table)}</tbody></table>
<p>Training accuracy and agreement with self-targets distinguish fitting the internal targets from learning an accurate procedure for new questions. Ground-truth diagnostics were computed after the frozen runs and did not influence their targets or hyperparameters.</p>
<p>The common reasoning baseline was 84.0%. After just the first four updates, reasoning-prompt responses shrank from roughly 144 tokens to about 7–9 tokens and accuracy collapsed. The frozen-source learner partly memorized practice answers, but this did not transfer to new programs. The current-source learner also lost the ability to produce useful practice targets. The final short-answer current-minus-frozen difference was {100*contrast['on_policy_minus_frozen']:+.1f} percentage points, paired 95% interval [{100*contrast['ci95'][0]:+.1f}, {100*contrast['ci95'][1]:+.1f}]; the main finding is the failure of all three recipes.</p>
{figures[1]}{figures[2]}{figures[3]}
<h2>Practice retained in context</h2>
<p>The initial model also receives 4 or 16 of its own practice examples in context, as either selected final answers or their actual sampled reasoning traces. This is one demonstration sequence and an exploratory transfer check. It is not an independent three-seed replication of the weight-learning result.</p>
<table><thead><tr><th>Examples</th><th>Context</th><th>Short accuracy</th><th>Reasoning accuracy</th><th>Short completion</th></tr></thead><tbody>{table_icl}</tbody></table>
<p>With full traces in context, reasoning responses average about 179 tokens with four examples and 152 with sixteen, compared with 144 without examples. Input contexts are also longer. Under the 96-token short-answer limit, completion falls to 0% and 4.2%, respectively. Those short-mode failures are truncations, so they do not show that the model forgot arithmetic. The reasoning improvement is not established at equal total inference compute.</p>
{figures[4]}
<h2>Follow-up: teach the actual reasoning trace</h2>
<p>This recipe uses 96 new practice questions, four passes, and a separate 144-question evaluation cohort. The model now trains under its original reasoning prompt on the actual sampled reasoning tokens. The two main arms still select from four samples by internal plurality. The control imitates one unfiltered current-model reasoning trace, including incomplete attempts.</p>
<p>All model weights are updated with AdamW at learning rate 1e-6, one update per batch of 16, and completion NLL divided by the fixed value 128 × batch size. FP32 weights/state, BF16 autocast, and clipping at 1 are retained. The prompt, target representation, update count, learning rate, and normalization all change from the failed recipe. This is a new recipe, not a causal isolation of trace preservation alone.</p>
<table><thead><tr><th>Final condition</th><th>Reasoning accuracy</th><th>Change from baseline</th><th>Short accuracy</th><th>Reasoning tokens</th><th>Completion</th></tr></thead><tbody>{trace_table}</tbody></table>
<p>The current-minus-frozen reasoning difference is {100*trace_contrast['on_policy_minus_frozen']:+.1f} percentage points, with a paired question-bootstrap 95% interval [{100*trace_contrast['ci95'][0]:+.1f}, {100*trace_contrast['ci95'][1]:+.1f}]. These intervals describe this test cohort conditional on one realized training run. They do not include training-seed variability.</p>
{''.join(trace_figures)}
<p class="note">The raw self-imitation control uses one rollout rather than four. The expected raw score gradient is zero at the sampling parameters under exact matched sampling and likelihood, but finite sampling, numerical precision, clipping, and AdamW can still move the weights. A realized gain or loss in this control is not evidence against that identity.</p>
<h2>Confirmation with new questions and three training seeds</h2>
<p>The learner, prompts, target rule, 96-question practice budget, four passes, optimizer, response limits, and checkpoints remain unchanged. New cohorts use practice seed 71100 and evaluation seed 71200; optimization seeds 71000–71002 vary ordering and sampling. The one-rollout control is not repeated. All six runs finished before correctness analysis.</p>
<table><thead><tr><th>Final condition</th><th>Mean reasoning accuracy</th><th>Three seed accuracies</th><th>Mean short accuracy</th><th>Reasoning tokens</th></tr></thead><tbody>{confirmation_table}</tbody></table>
<p>This is a fresh-data confirmation for the same architecture. The three seeds share 144 evaluation questions; they are not 432 independent questions. The paired bootstrap resamples seeds and questions, with questions stratified by program length. It is limited by three seeds and one new cohort. Perfect observed scores can produce degenerate bootstrap bands, which do not establish perfect population accuracy.</p>
{figure('elicitation-confirmation-reason.png', 'Unchanged full-trace recipe, fresh practice/evaluation cohorts, and three new optimization seeds. Bands are paired seed-and-question bootstrap intervals. All checkpoints were specified in advance.')}
{figure('elicitation-confirmation-teacher.png', 'Mean correctness of the internally produced targets across the three confirmation seeds. This is an offline diagnostic, never training feedback.')}
{figure('elicitation-confirmation-tokens.png', 'The main pair has matched question, rollout-count, and update budgets. Actual response lengths differ; generation plus training input tokens are measured work counts, not FLOPs.')}
<p>At the same final presentation budget, the current-source arm used about 9.2% more generation-plus-training input tokens. At the prespecified 192-presentation checkpoint it already achieved 96.5% reasoning accuracy using about 246,000 such tokens; the frozen-source arm reached 92.6% after 384 presentations and about 452,000 tokens. This is favorable under the recorded token-work proxy, but it does not establish a hardware-FLOP scaling law.</p>
<label for="budget">Inspect a confirmation checkpoint: </label><select id="budget"><option>0</option><option>16</option><option>64</option><option>192</option><option selected>384</option></select><div id="checkpoint"></div>
<h2>An explicit linear mechanism and its limits</h2>
<p>The toy assumes a useful internal generator already exists: z is standard Gaussian and x=Az+noise. A separate linear recognition map Bx estimates z. Internally generated (x,z) pairs can train B toward the posterior-mean inverse B*=Aᵀ(AAᵀ+σ²I)⁻¹. The generator has supplied no new observations of the world; it has made existing knowledge available as a supervised target.</p>
<pre>E||Bx-z||² = irreducible error + tr((B-B*) C (B-B*)ᵀ)
grad_B(half MSE) = B C - Aᵀ,    C = A Aᵀ + σ²I</pre>
<p>Across 32 SGD runs, prediction error falls from 8 to {toy_metric('sleep','mse'):.4f}, near the optimal {toy['protocol']['bayes_mse']:.4f}. Normalized weight overlap reaches {toy_metric('sleep','normalized_overlap'):.4f}. Self-imitation supplies no expected corrective gradient; an incorrect generator can actively worsen the inverse.</p>
{figures[5]}{figures[6]}
{figure('linear-elicitation-overlap.png', 'Overlap is tr(B C B*ᵀ) / tr(B* C B*ᵀ): zero for the initial zero weights and one for the optimal inverse. This is a normalized projection, not a cosine similarity. Prediction error also checks the remaining weight distance.')}
<p class="note">The toy keeps A fixed. It therefore demonstrates closure of an inference gap but cannot, by itself, explain an advantage from refreshing an autoregressive policy. It also uses a different task from the LLM arithmetic pilot. A complete three-setting equivalence remains a research objective, not an established conclusion.</p>
<h2>A linear extension that does test sampler refresh</h2>
<p>Keep the same generative model, but initialize a latent particle at the current recognition mean Bx plus Gaussian noise. Take one posterior-refinement step using the model's reconstruction residual and prior, then train B on that sampled particle. The frozen control always initializes at the original B₀x. Both take one posterior step per internal training pair and receive the same questions and coupled randomness; the latent used to generate each input is discarded before target creation.</p>
<pre>H = I + AᵀA / σ²,     R = I - δH
z₀ = Bx + ε₀
z₁ = Rz₀ + δAᵀx/σ² + ε₁
Cov(ε₁) = 2δI - δ²H</pre>
<p>This transition preserves the exact Gaussian posterior, but one step from an imperfect initialization remains biased. With current initialization, the expected weight error evolves as Eₜ₊₁ = Eₜ − ηδH Eₜ C and approaches zero at stable step sizes. Frozen initialization approaches B = δAᵀ/σ², the mean of its permanently shallow teacher. The prediction is analytic and was recorded before simulation.</p>
{figure('posterior-refresh-mse.png', 'Same 8×8 generator, 32 SGD repetitions, one posterior transition per internal pair. Solid lines are mean stochastic outcomes; dashed lines are exact population recurrences. Bands are 1.96 standard errors across repetitions.')}
<p>At 262,144 internal pairs, current initialization gives mean error 2.1024 versus 5.3060 for frozen initialization, from an initial error of 8. The optimal error is 1.8422. This demonstrates a refresh mechanism without adding facts: better recognition initializes better posterior samples, which further improve recognition.</p>
<p class="note">The posterior-refinement operation is supplied by the assumed useful generative model. It is the extra computation that makes these targets differ from unfiltered recognition samples. This toy models refreshing posterior initialization; it is not a matched autoregressive LLM experiment. Its geometric dynamics and biased frozen limit do not establish different scaling exponents.</p>
<h2>What the three settings currently establish</h2>
<table><thead><tr><th>Setting</th><th>What self-generation contributes</th><th>Remaining limitation</th></tr></thead><tbody>
<tr><td>Linear Gaussian learner</td><td>Internal latent targets align recognition weights with the implied posterior. A separate posterior-rollout extension benefits from refreshing its initialization.</td><td>The useful generative model and posterior-refinement operation are supplied; the arithmetic LLM has not been reduced to this mechanism.</td></tr>
<tr><td>LLM in-context learning</td><td>Keeping full self-generated worked examples improves subsequent reasoning on new arithmetic programs.</td><td>One exploratory sequence, longer context, no demonstrated short-answer compression.</td></tr>
<tr><td>LLM weight learning</td><td>Fresh versus frozen-source practice is directly compared; the full-trace recipe receives a fresh-data, three-seed confirmation.</td><td>One architecture, one confirmation cohort, and a short saturating budget range. Short-answer compression still fails.</td></tr>
</tbody></table>
<p>The shared hypothesis is that computation can expose competence already implicit in a model, then practice can make it easier to use. Shannon information need not increase. A distinct advantage from refreshing the sampler requires additional evidence: useful changes in the model's own practice distribution that improve subsequent learning beyond the frozen-source control.</p>
<h2>Reproducibility and resources</h2>
<p>Audited {learned['audit']['cases']+traces['audit']['cases']+confirmation['audit']['cases']} learning cases and {learned['audit']['checkpoints']+traces['audit']['checkpoints']+confirmation['audit']['checkpoints']} checkpoints, including every target's source, causal presentation order, counters, matching initial teacher draws, and paired baseline predictions. Final answer-only seed 68000 models, all full-trace pilot models, and confirmation seed 71000 models were copied to shared storage and SHA-256 verified before their jobs exited.</p>
<p>Across screening, confirmation, evaluation, training, and archival, completed allocations totaled {resources['allocated_gpu_minutes']:.1f} H100 GPU-minutes. This includes reserved time for evaluation and model copies, rather than only optimizer work. New paid API/Vast spending was $0.</p>
<details><summary>Versions, artifacts, and compute</summary><p>Model revision: <code>989aa7980e4cf806f80c7fef2b1adb7bc71aa306</code>. Full-weight training uses torch 2.10.0 and transformers 4.57.6 on H100 GPUs. Run outputs and immutable source snapshots are under <code>results/elicitation_pilot</code>, <code>results/elicitation_consolidation</code>, <code>results/elicitation_traces</code>, and <code>results/elicitation_toy</code>, <code>results/posterior_refresh</code>, and <code>results/elicitation_trace_confirmation</code>. Raw generated texts and token IDs remain in those local outputs; the download below contains analyzed predictions, curves, methods, and provenance. Archives: <code>{html.escape(execution['archive'])}</code> and <code>{html.escape(trace_execution['archive'])}</code>, and <code>{html.escape(confirmation_execution['archive'])}</code>.</p><p>New paid API/Vast spending: $0. Institutional computation used only gpu_lowp. All three learning allocations completed and released their GPUs. Actual token counts distinguish generated completions, prompts, supervised tokens, training input tokens, and padding. Timing records include inference and updates; evaluation work is kept separate. Token totals are not FLOPs. No learning-curve exponents are fitted to these short, saturating or deteriorating trajectories.</p></details>
<p><button id="download">Download analyzed predictions, methods, and provenance JSON</button></p>
<p>Background: <a href="https://arxiv.org/abs/2002.10689">usable information under computational constraints</a>, <a href="https://www.cs.toronto.edu/~hinton/absps/ws.pdf">wake–sleep</a>, and <a href="https://arxiv.org/abs/2510.18814">reward-free self-training</a>. Model: <a href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct">Qwen2.5-1.5B-Instruct</a>.</p>
<script type="application/json" id="data">{data}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);
const names={json.dumps(confirmation_labels)};
function render(){{const n=Number(document.getElementById('budget').value);let s='<table><tr><th>Condition</th><th>Short accuracy</th><th>Reasoning accuracy</th><th>Generation tokens</th><th>Training tokens</th></tr>';for(const [arm,name] of Object.entries(names)){{const d=data.confirmation.curves[arm].direct.find(r=>r.budget===n),r=data.confirmation.curves[arm].reason.find(r=>r.budget===n);s+=`<tr><td>${{name}}</td><td>${{(100*d.accuracy).toFixed(1)}}%</td><td>${{(100*r.accuracy).toFixed(1)}}%</td><td>${{Math.round(d.generation_tokens).toLocaleString()}}</td><td>${{Math.round(d.training_tokens).toLocaleString()}}</td></tr>`;}}document.getElementById('checkpoint').innerHTML=s+'</table><p>Tokens are measured work counts, not FLOPs; evaluation costs are separate.</p>';}}
document.getElementById('budget').addEventListener('change',render);render();
document.getElementById('download').addEventListener('click',()=>{{const u=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{{type:'application/json'}}));const a=document.createElement('a');a.href=u;a.download='elicitation-pilot-data.json';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);}});
</script></body></html>'''
    (REPORTS / "elicitation-pilot.html").write_text(page)
    summary = {"built_at_utc": datetime.now(timezone.utc).isoformat(), "elicitation": e,
               "learning_contrasts": learned["contrasts"], "learning_endpoints": {arm:{mode:rows[-1] for mode,rows in value.items()} for arm,value in learned["curves"].items()},
               "trace_contrasts": traces["contrasts"],
               "confirmation_contrasts": confirmation["contrasts"],
               "report": str(REPORTS / "elicitation-pilot.html"), "new_paid_spend": 0}
    (ROOT / "results/elicitation_consolidation/report_summary.json").write_text(json.dumps(summary,indent=2)+'\n')
    results_markdown = f'''# Self-generated practice and elicitation: results

The full-trace recipe received a fresh-data confirmation with three optimization
seeds. Final reasoning accuracy averaged **{percent(confirmation['curves']['on_policy_reasoning']['reason'][-1]['accuracy'])} with a current source**
versus **{percent(confirmation['curves']['frozen_reasoning']['reason'][-1]['accuracy'])} with a frozen source**, from a common
{percent(confirmation_baseline['accuracy'])} baseline. The paired difference is {100*confirmation_contrast['on_policy_minus_frozen']:+.1f} percentage
points, with a 95% paired seed-and-question bootstrap interval of
[{100*confirmation_contrast['ci95'][0]:+.1f}, {100*confirmation_contrast['ci95'][1]:+.1f}] points. This is one model and a small arithmetic task;
it does not establish distinct scaling exponents or successful short-answer compression.

The [standalone report](../reports/elicitation-pilot.html) includes the full curves,
failed recipes, an interactive confirmation checkpoint table, and downloadable
analyzed predictions and provenance.

## Model and training

The learner is **Qwen/Qwen2.5-1.5B-Instruct**, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. All weights are fine-tuned with AdamW;
this is not adapter training or an added classifier. The successful recipe uses
FP32 weights/state, BF16 autocast, learning rate 1e-6, no decay, clipping at 1, one
update per batch of 16, and completion-token NLL summed then divided by the fixed
value `128 * batch_size`.

Every arm receives the same 96 fixed arithmetic programs in four shuffled passes.
Each program specifies a starting integer and 4, 6, or 8 arithmetic operations.
Both main arms obtain four fresh reasoning samples per presentation. Internal
plurality over completed parsed final answers selects an actual full reasoning
trace. Incorrect targets remain; no exact answer or reward selects them. The only
main-pair intervention is whether the source retains its initial weights or uses
the current learner after each batch. This is on-policy self-training with a
likelihood loss, not RLVR. Exact answers are used only for offline evaluation.

## Three distinct observations

1. **Elicitation without weight updates:** on 144 fresh confirmation questions,
   the original weights achieved {percent(e['direct_greedy'])} with short answers,
   {percent(e['reason_greedy'])} with greedy reasoning, and {percent(e['consensus'])}
   with four-sample consensus. The response procedures use different compute.
2. **ICL with self-produced examples:** retaining 4 or 16 full traces raised
   reasoning accuracy from 84.0% to 94.4% on a separate cohort. This is one
   exploratory demonstration sequence. Answer-only context did not help.
   Full-trace context caused almost all short-mode responses to truncate.
3. **Weight learning:** the initial one-seed full-trace pilot reached 100.0%
   current-source versus 87.5% frozen-source reasoning accuracy, from 84.0%.
   It motivated the unchanged fresh-data, three-seed confirmation above.

Confirmation seed accuracies, in order 71000, 71001, 71002:

| Source | Reasoning accuracy by seed | Mean short-answer accuracy |
| --- | --- | --- |
| Frozen | {', '.join(percent(v) for v in confirmation['curves']['frozen_reasoning']['reason'][-1]['seed_accuracies'])} | {percent(confirmation['curves']['frozen_reasoning']['direct'][-1]['accuracy'])} |
| Current | {', '.join(percent(v) for v in confirmation['curves']['on_policy_reasoning']['reason'][-1]['seed_accuracies'])} | {percent(confirmation['curves']['on_policy_reasoning']['direct'][-1]['accuracy'])} |

The confirmation uses new practice seed 71100 and evaluation seed 71200. Its three
training seeds share 144 test questions; these are not 432 independent questions.
All six runs finished before correctness scoring. Intervals are limited by three
seeds and one fresh cohort. Perfect observed scores can yield degenerate bootstrap
bands without proving perfect population accuracy. The confirmation repeats the
same architecture, not another model family.

All three optimization seeds favor the current source, but the mean gap is smaller
than the exploratory pilot's 12.5 points. At 192 presentations the current arm
already achieved 96.5% using about 246,000 generation-plus-training input tokens;
the frozen arm reached 92.6% at 384 presentations and about 452,000 tokens. At equal
final presentations the current arm used about 9.2% more such tokens. These are
measured work proxies, not FLOPs or a fitted scaling law.

## Retained negative results

Answer-only consolidation used three seeds, full-weight AdamW at 1e-5, and four
updates per batch. It did not improve answers on new programs. Reasoning-prompt
outputs shrank from about 144 tokens to 7–9 tokens after the first batch. The
frozen-source learner partially memorized practice answers; the current-source
learner also lost the ability to produce useful practice targets.

In the full-trace pilot, unfiltered one-rollout self-imitation fell to 16.7%
reasoning accuracy. It is a mechanism control with fewer rollouts, not an equal-cost
competitor. A zero expected raw score gradient does not imply zero finite-sample
parameter motion, especially with clipping, AdamW, and finite precision.

The full-trace follow-up changed the prompt, target representation, learning rate,
normalization, and update count together. Its difference from the failed recipe
cannot be attributed solely to retaining reasoning traces. The fresh-versus-frozen
comparison within each recipe does hold these choices fixed.

The screened math-specialized model ignored the short-answer instruction; its
96-token responses truncated. A parser amendment later admitted unambiguous final
integers without LaTeX boxes, before the general model's fresh confirmation. Raw
responses, original strict scores, and both dated amendments remain available.

## Linear probabilistic mechanisms

The [original toy](LINEAR_ELICITATION_TOY.md) supplies an already useful generator
`x = Az + noise` and an imperfect recognition map `Bx`. Internally generated pairs
align B with `B* = Aᵀ(AAᵀ + sigma²I)⁻¹`. Mean error falls from 8 to 1.8566, near
the optimal 1.8422. The weighted error is exactly the squared distance from B to
B*. This toy alone does not compare refreshing a sampler.

The [posterior-refresh extension](POSTERIOR_REFRESH_TOY.md) initializes one latent
posterior step with current versus frozen recognition weights. It uses the same
8×8 generator and an exact-invariant linear Gaussian posterior transition. At the
same internal sampling budget, current initialization gives error 2.1024 versus
5.3060 for frozen initialization. Normalized weight overlap is 0.9300 versus
0.2969. Stable population dynamics approach the optimal inverse in the current
arm and a biased one-step teacher in the frozen arm.

The extra posterior computation is an explicit asymmetry. No new facts or oracle
latents enter these updates. This supplies a possible refresh mechanism within a
linear model; it does not reduce LLM arithmetic to those same weight dynamics.
The toy's geometric modes and biased frozen limit do not establish different
power-law exponents. The LLM curves are too short and saturate too quickly for a
credible exponent comparison.

## Reproduction and resources

Methods: [elicitation](ELICITATION_PROTOCOL.md),
[answer-only consolidation](ELICITATION_CONSOLIDATION_PROTOCOL.md),
[full-trace pilot](ELICITATION_TRACES_PROTOCOL.md), and
[fresh-data confirmation](ELICITATION_TRACE_CONFIRMATION_PROTOCOL.md).

Audits cover 18 learning cases and 90 checkpoints: source and data hashes, exact
sampled target tokens, causal schedules, counters, paired initial teacher samples,
and baseline predictions. Eight task/parser checks pass. The toy additionally
verifies posterior normal equations, risk decomposition, transition invariance,
and population-update stability. No paid APIs or Vast rentals were used; new paid
spend is $0. Training uses institutional H100s through `gpu_lowp` only.
All completed allocations together used {resources['allocated_gpu_minutes']:.1f} H100 GPU-minutes, including
screening, evaluation, and archival; optimizer work alone is recorded separately.

Final models retained for answer-only seed 68000, full-trace pilot seed 69000, and
confirmation seed 71000 were archived to the configured shared cache with SHA-256
verification. All training allocations completed and released their GPUs. Full
token and timing records remain in the run outputs. Counts of generation tokens
plus training input tokens are work counters, not FLOPs; the main pair matches
question, rollout, and optimizer counts but allows actual trace lengths to differ.

Rebuild the report after the independent analyzers have completed:

```bash
"$VENV_LOCAL/generative-learning/bin/python" experiments/build_elicitation_report.py
```
'''
    (ROOT / "experiments/ELICITATION_RESULTS.md").write_text(results_markdown)
    print(summary['report'])


if __name__ == "__main__": build()
