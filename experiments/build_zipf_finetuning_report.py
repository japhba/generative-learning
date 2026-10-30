"""Build a self-contained report from the completed and audited matched experiment."""

import base64
from datetime import datetime, timezone
import html
import json
from pathlib import Path

from analyze_zipf_finetuning import ARMS, FAMILIES

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
RESULTS = ROOT / "results/zipf_finetuning"


def read(path): return json.loads(path.read_text())


def interval(value, bounds, digits=3):
    return f"{value:.{digits}f} [{bounds[0]:.{digits}f}, {bounds[1]:.{digits}f}]"


def table(headers, rows):
    return "<table><thead><tr>" + "".join(f"<th>{html.escape(str(value))}</th>" for value in headers) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in row) + "</tr>" for row in rows) + "</tbody></table>"


def figure(name, caption):
    encoded = base64.b64encode((REPORTS / name).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>'


def main():
    data = read(REPORTS / "zipf-finetuning-data.json")
    protocol = read(RESULTS / "protocol.json")
    archive = read(RESULTS / "archive.json")
    execution = read(RESULTS / "execution.json")
    newline_pilot = read(RESULTS / "newline_pilot_summary.json")
    query_diagnostics = {family: read(RESULTS / family / "query_termination_diagnostic.json") for family in FAMILIES}
    assert data["complete"] and archive["complete"] and execution["allocations_released"]
    audits = {family: read(RESULTS / family / "same_data_audit/4000/audit.json") for family in FAMILIES}
    assert all(row["passed"] for row in audits.values())
    exponents, endpoints, costs, windows, pilots, corrected_queries = [], [], [], [], [], []
    supported = []
    for family, name in FAMILIES.items():
        result = data["families"][family]
        primary = result["exponent_windows"][0]
        difference = primary["paired_exponent_differences"]["ranked_minus_iid"]
        supported.append(difference["ci95"][0] > 0)
        exponents.append([name, *[interval(primary["fits"][arm]["alpha"], primary["fits"][arm]["ci95"]) for arm in ("iid", "ranked")], interval(difference["mean"], difference["ci95"])])
        selected = result["selection"]
        costs.append([name, selected["repeats"], f'{selected["learning_rate"]:g}', 128 * selected["repeats"], 4096 * selected["repeats"], result["calibration"]["attempted_cases"]])
        for arm, arm_name in ARMS.items():
            curve = result["curves"][arm]
            metrics = curve["metrics"]
            unique = curve["unique_queries_at128"]
            endpoints.append([name, arm_name, interval(metrics["error"]["mean"][-1], metrics["error"]["ci95"][-1]),
                              f'{metrics["seen_accuracy"]["mean"][-1]:.3f}', f'{sum(unique)/len(unique):.1f}',
                              f'{metrics["coverage_error"]["mean"][-1]:.4f}', f'{metrics["seen_error_mass"]["mean"][-1]:.4f}',
                              f'{curve["clocks"]["train_seconds"]["mean"][-1]:.1f}', f'{curve["clocks"]["generation_seconds"]["mean"][-1]:.1f}'])
        for window in result["exponent_windows"]:
            for arm, fit in window["fits"].items():
                windows.append([name, ARMS[arm], "–".join(map(str, window["budget_range"])), interval(fit["alpha"], fit["ci95"]), f'{fit["log_error_rmse"]:.3f}'])
        for pilot in result["calibration"]["pilot_results"]:
            pilots.append([name, pilot["seed"], pilot["curriculum"], pilot["repeats"], f'{pilot["learning_rate"]:g}', f'{pilot["metrics"]["seen_accuracy"]:.4f}'])
        for pilot in newline_pilot["families"][family]["pilots"]:
            corrected_queries.append([name, pilot["seed"], pilot["unique_keys"], f'{pilot["known_mass"]:.4f}', f'{pilot["seen_accuracy"]:.3f}', ", ".join(map(str,pilot["queries"][:8]))])
    toy = read(ROOT / "results/weight_coverage_toy/simulation.json")["exact_eta1_exponents_n8_to128"]
    exponents.insert(0, ["Exact finite toy", f'{toy["iid"]:.3f}', f'{toy["ranked"]:.3f}', f'{toy["ranked"]-toy["iid"]:.3f}'])
    headline = "Ranked observations produce faster finite learning curves in both fine-tuned models." if all(supported) else "The matched fine-tuning experiment gives a mixed replication of the exponent difference."
    live_sentences, retention_sentences = [], []
    for family, name in FAMILIES.items():
        result = data["families"][family]
        delta = result["paired_error_differences"]["on_policy_minus_iid"]
        bounds = delta["ci95"][-1]
        verdict = "lower" if bounds[1] < 0 else "higher" if bounds[0] > 0 else "not clearly different"
        live_sentences.append(f'{name}: final error with its own generated queries is {verdict} relative to IID study; paired error difference {interval(delta["mean"][-1], bounds)}.')
        seen = result["curves"]["ranked"]["metrics"]["seen_error_mass"]["mean"]
        late = result["exponent_windows"][2]["fits"]["ranked"]
        retention_sentences.append(f'{name}: ranked observed-key error mass is {seen[data["budgets"].index(8)]:.4f} at 8 labels and {seen[data["budgets"].index(16)]:.4f} at 16; its ranked exponent over 16–128 is {interval(late["alpha"], late["ci95"])}.')
    generated = datetime.now(timezone.utc).isoformat()
    body = f"""
<p class="eyebrow">Matched task · 32 paired dictionaries · Two full-weight learners · {generated[:10]}</p>
<h1>Does the Zipf learning curve survive training the weights?</h1>
<p class="lead">{headline}</p>
<p>We replaced storage in the prompt with full-weight fine-tuning while retaining the original dictionaries, example streams, oracle labels and weighted test items. The experiment distinguishes how observations are acquired from how the learner stores them.</p>
<h2>Primary result</h2>
<p>Effective exponents fit <code>error ∝ N⁻ᵝ</code> over the five prespecified budgets 8, 16, 32, 64 and 128. Brackets are 95% intervals from 1,500 paired whole-dictionary bootstrap samples. These finite-range slopes do not establish asymptotic scaling laws or exponent equivalence.</p>
{table(['Learner','IID β [95% CI]','Ranked β [95% CI]','Ranked − IID'], exponents)}
{figure('zipf-finetuning-icl-toy.png', 'Identical observation streams compared across the exact linear toy, earlier ICL and new weight learning. Shading shows fine-tuning uncertainty; the exponent comparison below includes ICL intervals.')}
{figure('zipf-finetuning-learning.png', 'Full-weight fine-tuning versus exact population toy curves. Shading is pointwise uncertainty across dictionaries. Both arms receive correct labels and use the same update rule.')}
{figure('zipf-finetuning-exponents.png', 'The same coverage comparison across the linear toy, earlier ICL, and fine-tuning. ICL ranked points use the same-example observation control. Different model sizes and answer formats prevent attributing all differences to where memory is stored.')}
<h2>Query generation: a decoder bias and a corrected pilot</h2>
<p>{' '.join(live_sentences)}</p>
<p>The current fine-tuned model greedily emits a legal key and guesses its label before feedback. Its prompt explains the rank frequencies and lists previously requested IDs, with no correct-label history. Each request costs one label, including repetitions. Decoding constrains valid IDs but does not enforce useful choices, novelty or sorting. A block is acquired before any of that block's labels update the model.</p>
<p><strong>The original decoder has a material limitation.</strong> Requiring EOS after the integer biases the models toward continuing the number. With unchanged weights and no feedback, Qwen repeatedly chose 10000 and SmolLM repeatedly chose 12345. Allowing a newline at a legal integer boundary changes both models' choices to repeated key 1. Thus the original generated-query curves combine termination bias with the policy's behavior.</p>
{figure('zipf-finetuning-acquisition.png', 'The original four acquisition arms. The generated-query arm uses the biased EOS boundary. Dashed lines are exact coverage predictions, and the historical Qwen arm retains the earlier larger model’s actual choices.')}
{table(['Learner','Acquisition','Final error [95% CI]','Seen accuracy','Unique / 128','Unseen error','Seen error mass','Train s','Generate s'], endpoints)}
<p>We then allowed newline or EOS and ran actual feedback and weight updates in four fresh dictionaries per model, seeds 52000–52003. This follow-up retained the prompt, legal key prefixes, chosen learning settings and repeated-query costs. It used no held-out test predictions or exponent fitting. The acquisition-calibration requirement was to exceed the exact expected IID coverage at 128 labels in every pilot: {newline_pilot['protocol']['expected_iid_known_mass_at128']:.4f} of test probability mass.</p>
{table(['Learner','Fresh pilot','Unique / 128','Known mass','Seen accuracy','First 8 queries'], corrected_queries)}
<p>Corrected acquisition calibration passed: {', '.join(name + ' — ' + ('yes' if newline_pilot['families'][family]['passes_acquisition_calibration'] else 'no') for family,name in FAMILIES.items())}. These fresh pilot outcomes characterize the corrected selector; the original EOS curves remain retained separately. The fixed IID/ranked learning-curve comparison is unaffected by this decoder issue.</p>
<h2>Mechanism: coverage and retention</h2>
<p>For independent four-way labels, an unobserved key contributes expected error 3/4. Thus coverage error is exactly <code>(3/4) × unobserved probability mass</code>. We also enumerate every acquired key to measure its actual test-weighted mistakes. Their sum marginalizes unknown labels and explains the raw test curve without replacing the primary measurement.</p>
<p>{' '.join(retention_sentences)} Optimization transients and finite test sampling can shift fitted slopes away from the exact toy; the primary 8–128 fits remain unchanged.</p>
{figure('zipf-finetuning-coverage-retention.png', 'Exact unobserved-label error plus every observed-key mistake, compared with the toy. This is a diagnostic conditional expectation, distinct from the measured 40-item test error.')}
{figure('zipf-finetuning-retention.png', 'Accuracy on every unique acquired key. Training does not automatically imply perfect storage; calibration and main-world retention are shown separately.')}
<p><strong>Actual weight control:</strong> in the prospectively selected dictionary 4000, replaying each original live model's identical correct observations and minibatches while omitting generation reproduced every checkpoint probability and every saved parameter tensor exactly, for both model families. This checks the implementation on one paired world per family; it is not another independent replication.</p>
{figure('zipf-finetuning-functional-overlap.png', 'Centered overlap of the output probabilities with the true label vectors. Chance probabilities give zero and perfect predictions give one. This is a functional measure, not overlap with a known true transformer weight vector.')}
<p>The explicit true weights belong to the linear toy. Transformer parameter-displacement norms are retained in the downloadable data; their magnitudes do not identify alignment with a unique teacher parameter vector.</p>
<h2>Exactly what was trained</h2>
<p>Qwen/Qwen3-0.6B-Base and HuggingFaceTB/SmolLM2-135M are reset to the same pretrained revision for every dictionary and arm. We update all model parameters using AdamW, FP32 parameters and state, BF16 autocast, zero weight decay and norm-1 gradient clipping. The loss is cross-entropy over the four ordinary LM output tokens for A/B/C/D. There is no added classifier head. Evaluation supplies only <code>Dictionary key: k\\nLabel:</code>, with no association examples.</p>
<p>Each new oracle observation enters a multiset buffer. Each of its R updates uses 32 presentations: the newest observation plus 31 uniform replay draws, including duplicates. Replay index sequences and resets are paired across arms. At N labels this costs RN updates and 32RN training presentations.</p>
<p>Within a model, these training budgets are exactly matched across acquisition arms. A presentation has different computational cost in the two architectures. Because R is fixed within each model, a power-law fit against updates or presentations has the same exponent as its label-budget fit; this change of units is not an independent scaling result.</p>
{table(['Learner','R updates / label','Learning rate','Updates at 128','Presentations at 128','Calibration cases'], costs)}
{figure('zipf-finetuning-presentations.png', 'The same errors versus actual training presentations, including replay. R is fixed within each model, so changing this horizontal unit leaves its label-budget exponent unchanged.')}
{figure('zipf-finetuning-runtime.png', 'Measured training plus generation time on shared institutional H100 GPUs; evaluation, calibration and model loading are excluded. Worker concurrency varied during execution, so these times describe this run and do not supply a controlled compute scaling law.')}
<p>The calibration criterion was at least 99% accuracy on all observed keys at 128 labels in each of four disjoint pilot dictionaries under both IID and ranked acquisition. Configurations are eliminated at the first failure. Selection uses no test predictions or exponent fits; every attempted pilot is retained below. Any execution amendment is preserved with the protocol artifacts.</p>
<details><summary>All attempted calibration cases</summary>{table(['Learner','Pilot world','Curriculum','R','Learning rate','Seen accuracy'], pilots)}</details>
<h2>Checkpoints</h2>
<p>Select a model and budget to inspect means across the same 32 worlds.</p>
<label>Model <select id="family">{''.join(f'<option value="{key}">{name}</option>' for key,name in FAMILIES.items())}</select></label>
<label>Labels acquired <select id="budget">{''.join(f'<option value="{i}" {"selected" if n==128 else ""}>{n}</option>' for i,n in enumerate(data['budgets']))}</select></label>
<div id="checkpoint"></div>
<h2>Fit sensitivity and limits</h2>
<p>The 8–128 window was fixed before confirmation. The two additional windows are sensitivity checks on overlapping data. Log-error residuals below expose curvature that a single exponent can hide. Intervals are pointwise and unadjusted for multiple contrasts. The 32 dictionaries are reused from ICL, enabling paired comparisons but not an independent draw of task worlds.</p>
<details><summary>All fit windows and residuals</summary>{table(['Learner','Acquisition','Fit window','β [95% CI]','Log-error RMSE'], windows)}</details>
<p>The earlier ICL models emitted 40 answers together in a context containing examples. The fine-tuned models answer each of those same weighted test items independently. They are smaller base models, not the original hosted instruction-tuned models. These differences limit a causal comparison of ICL versus weight storage.</p>
<p><strong>Scope of the conclusion:</strong> this tests acquisition and coverage with verified feedback. Ranked study and model-generated practice can supply the same information. It does not isolate a benefit of producing an answer, distinguish generative and discriminative architectures, or demonstrate learning from unverified self-generated labels.</p>
<h2>Reproducibility and resources</h2>
<p>All 256 model/dictionary/arm runs and 2,304 checkpoints were independently regraded from saved probabilities. Curricula, target labels, replay causality, paired resets and source hashes were checked. Eight preselected final models were archived to the configured shared cache, with SHA-256 verification of every copied file. All dictionaries retain their predictions, acquisition records and replay indices.</p>
<p>New paid API spending: $0. New Vast rental spending: $0. Institutional allocations {', '.join(map(str,execution['slurm_jobs']))} used the permitted gpu_lowp partition and have been released. Earlier OpenRouter spending remains $0.37499892. The existing report servers remain available.</p>
<p><button id="download">Download complete analysis JSON</button> <button id="methods">Download protocol and execution records</button></p>
<p><a href="generation-and-learning.html">Earlier feedback and tracking investigation</a></p>
<p class="muted">Generated {generated}. Source revisions and hashes are included in the downloadable records.</p>
"""
    payload = json.dumps(data, allow_nan=False).replace("</", "<\\/")
    metadata = json.dumps({"protocol": protocol, "execution": execution, "archive": archive, "runtime_environment": read(RESULTS / "runtime_environment.json"),
                           "query_termination_diagnostics": query_diagnostics, "corrected_boundary_pilot": newline_pilot,
                           "same_data_audits": {family: {key:value for key,value in audit.items() if key != "weight_tensors"} for family,audit in audits.items()}}, allow_nan=False).replace("</", "<\\/")
    script = """
const data=JSON.parse(document.getElementById('data').textContent), metadata=JSON.parse(document.getElementById('metadata').textContent);
const arms={iid:'IID study',ranked:'Ranked study',historical_qwen:'Earlier Qwen queries',on_policy:'Generated queries (EOS)'};
function render(){const family=document.getElementById('family').value, i=Number(document.getElementById('budget').value); let rows='';
for(const [arm,label] of Object.entries(arms)){const c=data.families[family].curves[arm], m=c.metrics; const seen=i===0?'—':m.seen_accuracy.mean[i-1].toFixed(3);
rows+=`<tr><td>${label}</td><td>${m.error.mean[i].toFixed(4)}</td><td>${m.coverage_error.mean[i].toFixed(4)}</td><td>${m.seen_error_mass.mean[i].toFixed(4)}</td><td>${seen}</td><td>${c.clocks.training_presentations.mean[i].toFixed(0)}</td></tr>`;}
document.getElementById('checkpoint').innerHTML='<table><thead><tr><th>Acquisition</th><th>Test error</th><th>Unseen error</th><th>Seen error mass</th><th>Seen accuracy</th><th>Presentations</th></tr></thead><tbody>'+rows+'</tbody></table>';}
function download(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);}
document.getElementById('family').addEventListener('change',render);document.getElementById('budget').addEventListener('change',render);
document.getElementById('download').addEventListener('click',()=>download(data,'zipf-finetuning-data.json'));document.getElementById('methods').addEventListener('click',()=>download(metadata,'zipf-finetuning-methods.json'));render();
"""
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Zipf learning in context and in weights</title><style>body{font-family:system-ui,sans-serif;color:#1c2534;background:#fafafa;margin:auto;padding:2rem;max-width:1050px;line-height:1.6}h1,h2{line-height:1.2}h2{margin-top:2.5rem}.eyebrow,.muted,figcaption{color:#526078}.lead{font-size:1.25em}figure{margin:1.8rem 0}img{max-width:100%;height:auto;display:block;margin:auto}figcaption{margin-top:.7rem}table{border-collapse:collapse;display:block;overflow:auto;width:100%;margin:1rem 0;font-variant-numeric:tabular-nums}td,th{padding:.55rem .75rem;border-bottom:1px solid #ccd4df;text-align:left;white-space:nowrap}th{background:#edf1f6}label{display:inline-block;margin:.5rem 1rem .5rem 0}select,button{font:inherit;padding:.35rem .6rem}code{background:#edf1f6;padding:.1rem .3rem}summary,button{cursor:pointer}details{margin:1rem 0}</style><main>' + body + '</main><script id="data" type="application/json">' + payload + '</script><script id="metadata" type="application/json">' + metadata + '</script><script>' + script + '</script></html>'
    (REPORTS / "zipf-finetuning.html").write_text(page)
    markdown = ["# Matched Zipf fine-tuning results", "", headline, "", "Effective exponents over the prespecified 8–128-label window; 95% paired world-bootstrap intervals:", "", "| Learner | IID | Ranked | Difference |", "|---|---|---|---|"]
    markdown.extend("| " + " | ".join(row) + " |" for row in exponents)
    markdown.extend(["", *live_sentences, "", "The original generated-query decoder required EOS and biased both models toward long, rare keys. A newline boundary removes that bias. Its separate fresh-world acquisition calibration is reported in the HTML; the fixed-curriculum exponent comparison is unaffected.", "", "Both world-4000 passive replays reproduced the original live condition's saved weights and checkpoint probabilities exactly.", "", "Full methods, all arms, calibration failures, sensitivity windows, resource accounting and downloadable provenance are in [the standalone report](../reports/zipf-finetuning.html).", "", "These are finite-range coverage effects under verified feedback, not an isolated generation-versus-recognition effect or a demonstrated asymptotic law.", "", "New API and Vast spending: $0. Institutional allocations released.", ""])
    (ROOT / "experiments/ZIPF_FINETUNING_RESULTS.md").write_text("\n".join(markdown))
    print(REPORTS / "zipf-finetuning.html")


if __name__ == "__main__": main()
