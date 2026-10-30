"""One canonical local report for the deep toy and matched LLM experiment."""

from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def million_size_section():
    study = json.loads((ROOT / "results/deep_wake_sleep/million_analysis.json").read_text())
    completion = f'{study["running"]} runs are still training; their target costs are lower bounds in this live update.' if study["running"] else 'Every run has finished; no live trajectories are included.'
    control = study["groups"]["adam_eps12"]["cached_dense_flops"]
    e, s = [control[arm]["fits"]["16-1048576"] for arm in ("elbo", "sleep")]
    assert all(f["status"] == "All targets attained" for f in (e, s))
    overlap = study["groups"]["adam_eps12"]["weight_overlap"]["0.8"]
    overlap_ratio = np.mean(overlap["elbo"]["points"][-1]["values"]) / np.mean(overlap["sleep"]["points"][-1]["values"])
    fits, counts, ratios = [], [], []
    for optimizer, epsilon in (("adam", "10⁻⁸"), ("adam_eps12", "10⁻¹²")):
        for arm, label in (("elbo", "ELBO"), ("sleep", "ELBO + sleep")):
            group = study["groups"][optimizer]["cached_dense_flops"][arm]
            cells = []
            for window in ("16-4096", "16-1048576", "4096-1048576", "65536-1048576"):
                f = group["fits"][window]
                cells.append(f'{f["mean_exponent"]:.2f} [{f["mean_exponent_seed_ci"][0]:.2f}, {f["mean_exponent_seed_ci"][1]:.2f}]' if f["status"] == "All targets attained" else "censored; not fitted")
            fits.append(f'<tr><td>{epsilon}</td><td>{label}</td>{"".join(f"<td>{c}</td>" for c in cells)}</tr>')
            point = study["groups"][optimizer]["step"][arm]["points"][-1]
            steps = f'{np.mean(point["values"]):,.0f}' if point["reached"] == 6 else f'≥ {np.mean(point["lower_bounds"]):,.0f}'
            counts.append(f'<tr><td>{epsilon}</td><td>{label}</td><td>{point["reached"]}/6</td><td>{steps}</td></tr>')
        r = study["largest_ratios"][optimizer]
        if "cached_dense_flops" in r:
            v = r["cached_dense_flops"]
            ratios.append(f'With ε={epsilon}, the million-state mean dense-compute ratio is {v["mean_ratio"]:,.1f}× (median ratio {v["median_ratio"]:,.1f}×; paired-seed interval [{v["seed_ci"][0]:,.1f}, {v["seed_ci"][1]:,.1f}]).')
        else: ratios.append(f'With ε={epsilon}, some million-state targets remain censored; no unconditional mean ratio is reported.')
    return f'''<h2>Scaling to a million latent configurations</h2>
<p><strong>The new range is 16–1,048,576 configurations: 65,536-fold in problem size and 256 times the previous maximum.</strong> The same six prespecified seeds are tested at all nine sizes, with both learning rules and both Adam stabilizers: {study["runs"]} training runs, plus {study["bridge_runs"]} CPU-to-GPU bridge runs. {completion} Every point on the new curve uses the same float64 GPU implementation and evaluation schedule. Earlier CPU results remain below as historical comparisons.</p>
<p><strong>The size-dependent cost advantage survives the numerical control.</strong> With ε=10⁻¹², full-range fitted powers are {e["mean_exponent"]:.2f} for ELBO and {s["mean_exponent"]:.2f} for ELBO + sleep. The seed-bootstrap intervals are [{e["mean_exponent_seed_ci"][0]:.2f}, {e["mean_exponent_seed_ci"][1]:.2f}] and [{s["mean_exponent_seed_ci"][0]:.2f}, {s["mean_exponent_seed_ci"][1]:.2f}]. The baseline’s exponent changes as the fitting window moves, so this supports a clear finite-range scaling advantage without establishing one fixed exponent for all sizes.</p>
<figure><img src="deep-wake-sleep-million-cached.png" alt="Cost to fixed posterior error over nine sizes through one million latent configurations"><figcaption>Mean recognition matrix-product FLOPs to first recorded posterior KL ≤ 0.1, with one seed standard error. Dashed curves use Adam ε=10⁻⁸; solid curves use ε=10⁻¹². The vertical dotted line is the earlier 4,096-state limit. The fixed generator is cached; elementwise kernels, one-time setup and evaluation are excluded from this leading-dense cost axis. The same accounting is used at every size. Triangles, if present, are lower bounds for unattained targets.</figcaption></figure>
<p>{" ".join(ratios)}</p>
<figure><img src="deep-wake-sleep-million-updates.png" alt="Number of optimizer updates needed as latent configuration count grows"><figcaption>Update count separates growing output-space cost from extra optimization work. ELBO executes two recognition matrix products per update; the combined method executes three. Sleep uses the exact expected generative gradient here, rather than a finite Monte Carlo batch.</figcaption></figure>
<table><thead><tr><th>Adam ε</th><th>Method</th><th>Million-state targets</th><th>Mean updates to target</th></tr></thead><tbody>{"".join(counts)}</tbody></table>
<p><strong>The numerical control matters:</strong> as probability is divided across more coordinates, per-coordinate gradients can become small relative to Adam’s ε. Lowering ε tests whether a growing recovery cost is partly caused by that absolute gradient scale. It is not a temperature change or a change to generated-data selection. Learning rates remain those previously selected on smaller sizes: 0.1 for ELBO and 0.3 for ELBO + sleep; the large sizes have not been separately retuned.</p>
<table><thead><tr><th>Adam ε</th><th>Method</th><th>Exponent, 16–4,096</th><th>16–1,048,576</th><th>4,096–1,048,576</th><th>65,536–1,048,576</th></tr></thead><tbody>{"".join(fits)}</tbody></table>
<p>These fits use mean dense recognition cost. Brackets give 95% paired-seed bootstrap intervals; they measure variation across six seeds and do not establish an asymptotic law. Fits are omitted whenever any target in the window is unattained. Median fits, reference-cost fits and all individual observations are retained in the <a href="deep-wake-sleep-data/million_analysis.json">analysis data</a>.</p>
<figure><img src="deep-wake-sleep-million-slopes.png" alt="Rolling cost exponents and bootstrap uncertainty over the larger size range"><figcaption>Each slope uses three adjacent sizes, spanning a factor of 16 in K. Shading is the paired-seed bootstrap interval. A straight-looking full-range plot can coexist with changing local slopes; the numerical control helps distinguish those effects.</figcaption></figure>
<figure><img src="deep-wake-sleep-million-weight-overlap.png" alt="Compute to reach a specified overlap with the exact posterior coefficients"><figcaption>Direct weight-space measurement using the smaller Adam stabilizer. The exact posterior’s affine coefficients are compiled from the generative model and centered across output classes to remove the common-logit gauge. Cosine overlap includes weights and intercepts. These two descriptive targets assess overall coefficient alignment; posterior KL also strongly penalizes explanations assigned extremely small recognition probability. The corresponding target costs need not have the same ratios or slopes. Alternative posterior-KL targets 0.2 and 0.4 are also recorded in the analysis data.</figcaption></figure>
<p>At a million configurations, sleep uses {overlap_ratio:.1f}× less dense compute to reach weight overlap 0.8. Over the new 4,096–1,048,576 range, both overlap-target cost curves are approximately linear: their fitted powers are {overlap["elbo"]["fits"]["4096-1048576"]["mean_exponent"]:.2f} and {overlap["sleep"]["fits"]["4096-1048576"]["mean_exponent"]:.2f}. This makes the interpretation more specific: the larger posterior-KL advantage concerns recovery of severely underestimated explanations, rather than equally faster scaling of every measure of weight learning.</p>
<details><summary>Reference FLOPs, validation and interpretation</summary>
<figure><img src="deep-wake-sleep-million-scaling.png" alt="Million-state comparison using the earlier reference FLOP convention"><figcaption>This axis retains the earlier uncached algorithm’s approximate FLOP convention, including generator work. The current GPU execution caches that fixed distribution, so these are explicitly reference FLOPs, not actual measured GPU arithmetic or runtime.</figcaption></figure>
<p>The GPU audit covers three generative depths and three objectives. Gradients and 100-step weight trajectories agree with the NumPy implementation to numerical precision; restoring the complete optimizer tensors reproduces continuation exactly. Full checkpoints include recognition weights, Adam moments/counter, the constant learning-rate specification and RNG states. They are written locally and copied asynchronously to persistent storage. Evaluation and stationarity checks continue throughout long runs.</p>
<p>The bridge check exposed sensitivity of first-hit target cost to evaluation spacing: a brief error dip may be missed by a coarse schedule even when shared-checkpoint errors agree closely. For that reason all smaller sizes were rerun with the new schedule; the new large points were not simply appended to older coarse measurements. The raw trajectories and bridge discrepancies remain available.</p>
<p>This remains a fixed-generator population inference experiment. The two-layer sigmoid belief network is normalized, and its exact posterior is realizable by the affine recognizer. No verifier, rejection, reward or temperature adjustment enters either objective. The recognizer explicitly allocates one output to every latent configuration, and the posterior coefficients could also be compiled directly from the generator. The comparison therefore concerns specified learning rules and their recovery behavior; it is not an algorithm-independent computational lower bound, a finite-sample scaling law or a new LLM result. The earlier LLM experiment below remains unchanged.</p>
<p><a href="deep-wake-sleep-data/million_size/protocol.json">Large-size protocol</a> · <a href="deep-wake-sleep-data/million_anchors/protocol.json">Consistent smaller-size reruns</a> · <a href="deep-wake-sleep-data/gpu_audit.json">GPU validation</a> · <a href="deep-wake-sleep-data/million_execution.json">Execution and compute record</a> · <a href="deep_wake_sleep_gpu.py">Training implementation</a> · <a href="analyze_wake_sleep_million.py">Analysis</a> · <a href="DEEP_WAKE_SLEEP_THEORY.md">Weight mechanism and Adam scaling calculation</a> · <a href="WAKE_SLEEP_SCALING_RESULTS.md">Results notes</a></p>
</details>
<h2>Earlier 4,096-state and longer-budget LLM comparison</h2>'''


def wider_range_section():
    wide = json.loads((ROOT / "results/llm_wake_sleep/wide_analysis.json").read_text())
    actual = json.loads((ROOT / "results/llm_wake_sleep/wide_matched_evaluation.json").read_text())
    epsilon = json.loads((ROOT / "results/deep_wake_sleep/wide_epsilon_analysis.json").read_text())
    expanded = json.loads((ROOT / "results/deep_wake_sleep/expanded_size_analysis.json").read_text())
    e,s=expanded["groups"]["elbo"],expanded["groups"]["sleep"]
    assert all(g["status"]=="All sixteen seeds reached every target" for g in (e,s))
    def mean(arm, cohort, key): return float(np.mean([r[cohort][key] for r in actual if r["arm"]==arm]))
    table = []
    for arm, label in (("elbo","ELBO"),("sleep","ELBO + sleep"),("forward","Exact posterior teaching")):
        runs = [r for r in actual if r["arm"]==arm]
        table.append(f'<tr><td>{label}</td><td>{100*mean(arm,"two_examples","accuracy"):.2f}%</td><td>{1-mean(arm,"two_examples","compatible_mass"):.4f}</td><td>{mean(arm,"two_examples","inference_forward_kl"):.4f}</td><td>{100*mean(arm,"three_examples","accuracy"):.2f}%</td><td>{", ".join(str(r["step"]) for r in runs)}</td></tr>')
    slope_rows = []
    for window in ("64-256","128-512","256-1024","512-2048","1024-4096"):
        cells=[]
        for arm in ("elbo","sleep"):
            item=wide["effective_slopes"][arm][window]
            cells.append(f'{item["exponents"]["compatible_mass"]["mean_curve"]:.2f}' if item["status"]=="Fully observed" else "stopped; not fitted")
        slope_rows.append(f'<tr><td>{window}</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>')
    eps_rows=[]
    for optimizer, label in (("adam","10⁻⁸"),("adam_eps12","10⁻¹²")):
        for arm in ("elbo","sleep"):
            item=epsilon["groups"][optimizer][arm]; last=item["records"][-1]
            cost=f'{last["mean_cost"]:.3g}' if last["mean_cost"] is not None else f'≥ {last["mean_lower_bound"]:.3g} (censored)'
            eps_rows.append(f'<tr><td>{label}</td><td>{arm}</td><td>{item["exponent_16_to_1024"]:.2f}</td><td>{last["reached"]}/6</td><td>{cost}</td></tr>')
    return f'''<h3>The 4,096-state extension and eightfold LLM budget</h3>
<p><strong>The full original 16-seed toy cohort now extends from 16 to 4,096 latent states.</strong> Every run reaches posterior KL ≤ 0.1. Across that 256-fold range in state count, descriptive fitted cost exponents are {e["exponent"]:.2f} for ELBO and {s["exponent"]:.2f} for ELBO + sleep. Paired-seed bootstrap intervals are [{e["exponent_seed_ci"][0]:.2f}, {e["exponent_seed_ci"][1]:.2f}] and [{s["exponent_seed_ci"][0]:.2f}, {s["exponent_seed_ci"][1]:.2f}]. At 4,096 states the cost ratio is {expanded["largest_mean_cost_ratio"]:.1f}× using means and {expanded["largest_median_cost_ratio"]:.1f}× using medians; the mean ratio’s seed-bootstrap interval is [{expanded["largest_ratio_seed_ci"][0]:.1f}, {expanded["largest_ratio_seed_ci"][1]:.1f}].</p>
<figure><img src="deep-wake-sleep-expanded-replication.png" alt="Full sixteen-seed toy size comparison extended through 4096 latent states"><figcaption>Actual compute to a fixed error target, with means ± one seed standard error. This measures cost versus problem size, not error versus training compute. The fixed exact generator, population objectives, affine recognizer and earlier selected rates are unchanged; the largest size was not separately retuned. The baseline’s adjacent-size slopes still vary, so these are finite-range descriptions. Raw data: <a href="deep-wake-sleep-data/expanded_size_analysis.json">full expanded cohort</a>.</figcaption></figure>
<p>We continued all nine LLM runs from their saved optimizer and RNG states, increasing the cap from 512 to 4,096 updates. The task, data, model, learning rates, temperature and objectives stayed fixed. Initial checkpoint probabilities were reproduced on resume. Evaluation was made more frequent, with the existing stationarity rule retained.</p>
<figure><img src="llm-wake-sleep-wide-range.png" alt="Wider LLM learning curves, with the earlier budget marked"><figcaption>The dotted vertical line is the old 512-update baseline compute allowance. Left: probability assigned to an incompatible rule on the external task. Right: recognition’s forward KL from the current generator’s exact posterior, which measures inference error within the learned probabilistic model. Means ± one seed standard error; interpolation is between evaluated checkpoints. Dashed tails, if present, retain early-stopped weights rather than imply further training.</figcaption></figure>
<p><strong>The early task-accuracy gap disappears with more training.</strong> The table uses actual saved weights under the same larger compute allowance, including the three-example cohort that never enters training. The larger-budget accuracies are {100*mean("elbo","two_examples","accuracy"):.2f}% for ELBO and {100*mean("sleep","two_examples","accuracy"):.2f}% with sleep. On three examples, they are {100*mean("elbo","three_examples","accuracy"):.2f}% and {100*mean("sleep","three_examples","accuracy"):.2f}%. Sleep still has lower within-model inference KL on the two-example cohort, but this does not translate into a lasting external-task advantage.</p>
<table><thead><tr><th>Method</th><th>Two-example accuracy</th><th>Probability error</th><th>Model-posterior inference KL</th><th>Three-example accuracy</th><th>Actual updates, each seed</th></tr></thead><tbody>{"".join(table)}</tbody></table>
<p>The allowance is {wide["new_allocated_budget"]:.4g} estimated FLOPs per run, eight times the earlier baseline allowance. We use the latest saved checkpoint within it; sleep uses 0.78% less compute because checkpoints are spaced 32 updates apart. All nine continuations reached 4,096 updates without triggering stationarity stopping. The model-posterior KL uses different moving targets across arms because their generators also learn. It is not an arithmetic correctness metric, and the model posterior is not a lower bound on external task error when the generator is misspecified.</p>
<details><summary>Why additional generator learning matters</summary><figure><img src="llm-wake-sleep-generator-progress.png" alt="Generator digit error by training compute and by generator updates"><figcaption>The generators improve substantially after the old cutoff. At the same 4,096 update count, generator correctness is similar across methods; at the matched compute allowance, sleep has made fewer generator updates. This is consistent with a transition from recovering the inverse mapping to improving the generator, although it is not a separate causal intervention isolating that transition.</figcaption></figure></details>
<details><summary>Effective exponents and recovery of individual rules</summary><table><thead><tr><th>Baseline-update-equivalent compute window</th><th>ELBO probability-error exponent</th><th>Sleep probability-error exponent</th></tr></thead><tbody>{"".join(slope_rows)}</tbody></table><p>These are descriptive log–log slopes of mean error. Only fully observed windows are fitted. A persistent power law would require stability when the window moves; early onset, delayed recovery and late floors can all change these slopes. Three optimization seeds are insufficient for a precise universal exponent claim.</p><figure><img src="llm-wake-sleep-wide-task-kl.png" alt="Wider external true-posterior KL curves"></figure><figure><img src="llm-wake-sleep-rule-recovery.png" alt="Probability error for each of the eight rules during training"><figcaption>Same mean error, separated by the true generating rule for evaluation only. No rule labels from these evaluation observations enter training.</figcaption></figure></details>

<p>For the toy, we increased the budget to 262,144 wake-equivalent updates and lowered the accuracy stopping target from KL 0.03 to 10⁻⁹. The 54 cases use six fixed replication seeds, 64/256/1,024 latent states, the same two-layer fixed generator and affine recognizer, and all three objectives. A further 18 runs at 256 states reduce every method’s learning rate tenfold to diagnose the late optimization floor.</p>
<figure><img src="deep-wake-sleep-wide-range.png" alt="Toy posterior error over a larger compute and accuracy range"><figcaption>The middle state count, chosen before these outcomes, with six seeds. The dotted horizontal line is the earlier error stopping target. Smaller learning rates lower the eventual plateau. Solid means include actual training for every seed; dashed portions include models retained at their stopping weights. These curves show recovery transitions and optimization floors, rather than one power law spanning the full range.</figcaption></figure>
<figure><img src="deep-wake-sleep-wide-targets.png" alt="Compute to progressively tighter posterior error targets"><figcaption>Actual first-hit compute, averaged only at targets reached by all six seeds. Missing tighter targets remain failures or right-censored observations in the raw tables; they are not extrapolated. Dashed lines here denote the explicitly tested lower learning rates. Hyperparameter search compute is not included in the per-run cost axis.</figcaption></figure>

<details><summary>Numerical stabilizer and weight-space mechanism checks</summary>
<p>A 72-run six-seed audit checks Adam’s numerical stabilizer as per-state gradients shrink: ε = 10⁻⁸ versus 10⁻¹², with all other hyperparameters held fixed. A further 20 runs complete the larger-size point for the full 16-seed cohort shown above. The fitted exponents in this audit table use only 16–1,024; every six-seed 4,096-state target was also attained. The large gap survives the stabilizer change.</p>
<figure><img src="deep-wake-sleep-wide-sizes.png" alt="Larger latent-state count and Adam numerical-stabilizer audit"><figcaption>Same six seeds in each panel; means ± one seed standard error. Every target in this audit was reached. This robustness check does not turn a finite-size descriptive fit into an asymptotic theorem.</figcaption></figure>
<table><thead><tr><th>Adam ε</th><th>Method</th><th>Exponent, 16–1,024</th><th>4,096 targets reached</th><th>4,096 mean FLOPs to KL 0.1</th></tr></thead><tbody>{"".join(eps_rows)}</tbody></table>
<p>There is also an exact weight-space identity for the affine toy. If e is the difference from the compiled posterior weights, wake’s recognition gradient is <code>M(B)e</code>, where M is the current recognition Fisher matrix. The expected forward/sleep gradient is <code>[∫ M(B* + s(B−B*)) ds]e</code>: an average of Fisher matrices along the path to the posterior. This explains why neglected directions can receive weak wake updates while forward fitting still moves them. Both operators agree locally near the correct posterior. We checked posterior compilation, both gradient identities and the weight-distance contraction across 36 cases, with maximum error below 1.5×10⁻¹⁵. This is an affine-toy identity, not a global identity or exponent theorem for the nonlinear LLM. <a href="DEEP_WAKE_SLEEP_THEORY.md">Derivation</a> · <a href="deep-wake-sleep-data/weight_mechanism_audit.json">Audit results</a>.</p>
</details>
<p><a href="llm-wake-sleep-data/wide_analysis.json">All wider LLM curves and window slopes</a> · <a href="llm-wake-sleep-data/wide_matched_evaluation.json">Actual larger-budget evaluations</a> · <a href="deep-wake-sleep-data/wide_analysis.json">Toy accuracy targets and censoring</a> · <a href="deep-wake-sleep-data/wide_epsilon_analysis.json">Larger-size and numerical-parameter audit</a></p>
<h2>Earlier shorter-budget results, for comparison</h2>'''


def main():
    sizes = json.loads((ROOT / "results/deep_wake_sleep/size_analysis.json").read_text())
    llm = json.loads((ROOT / "results/llm_wake_sleep/analysis.json").read_text())
    matched = json.loads((ROOT / "results/llm_wake_sleep/matched_evaluation.json").read_text())
    def mean(arm,cohort,key): return float(np.mean([r[cohort][key] for r in matched if r["arm"]==arm]))
    rows = []
    for arm, label in (("elbo","ELBO"),("sleep","ELBO + sleep"),("forward","Exact posterior teaching")):
        rows.append(f'<tr><td>{label}</td><td>{100*mean(arm,"two_examples","accuracy"):.1f}%</td><td>{mean(arm,"two_examples","posterior_kl"):.3f}</td><td>{100*mean(arm,"three_examples","accuracy"):.1f}%</td></tr>')
    e,s = sizes["replication"]["elbo"],sizes["replication"]["sleep"]
    ci_e,ci_s=e["exponent_ci"],s["exponent_ci"]
    small_ratio=e["mean_cost"][0]/s["mean_cost"][0];large_ratio=e["mean_cost"][-1]/s["mean_cost"][-1]
    source_names = ["deep_wake_sleep.py","analyze_deep_wake_sleep.py","analyze_wake_sleep_sizes.py","llm_wake_sleep.py","analyze_llm_wake_sleep.py",
                    "evaluate_llm_wake_sleep_matched.py","run_llm_wake_sleep.py","tune_llm_wake_sleep.py","confirm_llm_wake_sleep.py",
                    "DEEP_WAKE_SLEEP_THEORY.md","build_wake_sleep_scaling_report.py","extend_llm_wake_sleep.py",
                    "analyze_wake_sleep_wide_range.py","evaluate_wide_llm_wake_sleep.py","audit_wake_sleep_weights.py",
                    "deep_wake_sleep_gpu.py","analyze_wake_sleep_million.py","WAKE_SLEEP_SCALING_RESULTS.md"]
    for name in source_names: shutil.copy2(ROOT/"experiments"/name,REPORTS/name)
    for label,destination in (("deep-wake-sleep-data","deep_wake_sleep"),("llm-wake-sleep-data","llm_wake_sleep")):
        path=REPORTS/label
        if not path.is_symlink(): path.symlink_to(ROOT/"results"/destination,target_is_directory=True)
    code_links=" · ".join(f'<a href="{name}">{html.escape(name)}</a>' for name in source_names)
    report=f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="cache-control" content="no-cache, no-store, must-revalidate"><meta http-equiv="pragma" content="no-cache"><meta http-equiv="expires" content="0">
<title>Wake–sleep scaling and the LLM match</title>
<style>body{{font-family:system-ui,sans-serif;line-height:1.55;max-width:1080px;margin:auto;padding:2rem;color:#20252b;background:#fafbfc}}h1,h2{{line-height:1.2}}figure{{margin:2rem 0}}img{{max-width:100%;height:auto;background:white}}figcaption,.muted{{color:#52606d}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:.7rem;text-align:left;border-bottom:1px solid #dce2e7}}.finding{{padding:1rem;border-left:4px solid #287a68;background:#edf7f3}}a{{color:#165f9d}}code{{background:#eef1f4;padding:.1em .3em}}details{{margin:1rem 0}}.links{{overflow-wrap:anywhere}}</style></head><body>
<h1>Wake–sleep scaling and a matched LLM experiment</h1>
<p class="muted">Updated {datetime.now(ZoneInfo("Europe/London")).strftime("%d %B %Y, %H:%M %Z")}. Local CPU and institutional gpu_lowp H100s. Paid APIs/rental spend: $0.</p>
<div class="finding"><strong>New: finite-data learning curves with an imperfect learned generator.</strong>
The new test fixes task and architecture and varies real sample count. Fully expressive recognition shows no sleep advantage; a standard factorized recognizer benefits from unfiltered dreams, with a matched-compute check. Posterior teaching on existing real examples performs similarly or better, so generated inputs are not established as essential. <a href="wake-sleep-finite-data.html">Read the current finite-data report and controls.</a></div>
<p>The older million-state experiment below measures cost versus problem size, not loss versus real data. Its exponents should not be interpreted as data-scaling exponents. No new LLM experiment is included in the finite-data update.</p>
{million_size_section()}
{wider_range_section()}

<h2>The more reproducible toy result</h2>
<p>A two-layer directed sigmoid belief network generates six binary observations from hidden binary causes. <strong>For this size experiment, the useful generator is fixed and exactly matches the data distribution; only recognition weights learn.</strong> The recognizer is an affine softmax over complete latent configurations. At moderate initial recognition bias, 16 new initialization seeds give finite-range fits <code>C(K) ∝ K^{e["exponent"]:.2f}</code> for ELBO and <code>C(K) ∝ K^{s["exponent"]:.2f}</code> for ELBO plus sleep, where K ranges from 16 to 1,024 latent configurations and the target is posterior KL ≤ 0.1. Seed-bootstrap intervals are [{ci_e[0]:.2f}, {ci_e[1]:.2f}] and [{ci_s[0]:.2f}, {ci_s[1]:.2f}]; they do not cover model or extrapolation uncertainty.</p>
<p>The measured mean compute ratio grows from {small_ratio:.1f}× at 16 configurations to {large_ratio:.1f}× at 1,024. These are actual compute-to-target ratios, unlike error ratios in the previous Gaussian experiment. Every replication run reached the primary target. A further 64-run audit retuned each method separately at every size, using an equal rate grid and independent tuning seeds; it selected the same rates at all four sizes, so no confirmation cells needed rerunning. The baseline’s adjacent-size slopes vary substantially, so the fitted exponent is a finite-range description.</p>
<figure><img src="deep-wake-sleep-size-scaling.png" alt="Compute needed to reach fixed posterior KL, with moderate and diffuse initial recognition"><figcaption>Left: 16 fresh initialization seeds. Right: the three-seed diffuse-initialization control. Means and one seed standard error. Both arms use the same model and data distribution within each case; sleep costs are included. The benefit depends on an inference deficit.</figcaption></figure>

<p>The initial three-seed fit suggested exponents 2.02 and 1.15. Before running 4,096 configurations, we saved its extrapolation. Sleep required 2.37 billion estimated FLOPs versus a forecast of 2.36 billion, agreeing within checkpoint resolution. ELBO exceeded its point forecast: two runs reached the target and one remained above it after its budget, giving a mean-cost lower bound of 744 billion FLOPs versus a forecast of 472 billion. We therefore have no observed finite mean target cost for that ELBO check, and do not fit the censored limit as if it were a success.</p>
<figure><img src="deep-wake-sleep-forecast.png" alt="Frozen forecasts and prospective 4096-state measurements"><figcaption>Dashed curves were fixed using the original smaller sizes. The triangle is a lower bound; it is not a measured target attainment. The prediction is not refit to this point.</figcaption></figure>

<h2>What the deeper experiments establish</h2>
<p>The broader sweep includes one- and three-layer generative models, one- and three-layer recognizers, SGD and Adam, population and sampled gradients, and fixed and coupled generators. There are 1,296 main runs, 324 rate follow-ups, 48 initial size runs, 96 size controls, 128 independent size replications and six extrapolation runs, plus 376 pilot/tuning runs. Rates were searched on separate seeds, including checks beyond the original grid boundaries.</p>
<p>Random recognition logit biases deliberately create unequal confidence: standard deviations 0, 3 and 6. This is an explicit controlled initialization, not a claim that standard pretraining naturally produces these exact biases. Strong neglect gives large recovery improvements, but sampled sleep can leave a noise floor and the fitted error-versus-compute slopes vary or reverse later. Adding depth alone did not establish a new asymptotic exponent.</p>
<figure><img src="deep-wake-sleep-sampled.png" alt="Sampled shallow and deep learning curves"><figcaption>Fixed useful generators, sampled real observations and dreams, strong initial neglect. Means ± one seed standard error. Stopped models retain their last error for larger allocated budgets; no extra updates are implied. Exact posterior teaching often improves on sleep.</figcaption></figure>
<details><summary>Population and coupled-generator curves</summary><figure><img src="deep-wake-sleep-population.png" alt="Population learning curves"></figure><figure><img src="deep-wake-sleep-coupled.png" alt="Generator marginal error during coupled learning"></figure></details>

<h2>An actual LLM match</h2>
<p>We trained <strong>Qwen2.5-1.5B-Instruct</strong> using two separate rank-four LoRA adapters on its query/value projections. The pretrained backbone is frozen. One adapter predicts an output digit given an input and a named arithmetic rule; the other infers a rule from two input–output examples. The latent prior is uniform over eight rules. There are 480 training observations and 240 held-out observations, with disjoint ordered input pairs.</p>
<p>Wake updates use unlabeled real observations: the hidden rule identity is removed before training. ELBO expectations enumerate all eight rules, so the baseline does not rely on a poor discrete-gradient estimator. In sleep, the model samples a rule and its own output digits, then learns to recover that rule from the imagined examples. <strong>All generated examples are retained, including arithmetically incorrect ones. Temperature is one; no correctness checker, reward, or true rule label enters the updates.</strong> Both directions train in the final comparison.</p>
<p>The conditional output supports are explicitly restricted to ten digit tokens and eight letter tokens, identically across arms. This makes a tractable probabilistic model inside an LLM; it is not unrestricted chain-of-thought self-training. The generator’s initial exact posterior identifies a compatible rule on about 82% of the held-out examples, while direct recognition is initially at 12.5%.</p>
<figure><img src="llm-wake-sleep-learning.png" alt="LLM inference error and posterior KL against full training compute"><figcaption>Three fresh optimization seeds. Compute includes generation/scoring and activation backpropagation. Lines interpolate recorded checkpoints; shading is one seed standard error. The table below uses actual checkpoints rather than interpolated values.</figcaption></figure>
<table><thead><tr><th>Method</th><th>Two-example accuracy</th><th>Two-example posterior KL</th><th>New three-example accuracy</th></tr></thead><tbody>{"".join(rows)}</tbody></table>
<p>We replayed sleep from saved state to update 459 and evaluated ELBO/posterior teaching at update 512. Estimated compute differs by only 0.13%, with sleep using slightly less. Sleep improves greedy rule accuracy in all three seeds on both cohorts. Initializations are paired and both methods draw from the same IID training pool, but real-batch orders differ because sleep consumes extra random draws. The new cohort contains 640 observations with three examples each, generated after choosing the training setup; it was never used in training or rate selection. Exact posterior teaching remains a strong competitor and has lower posterior KL.</p>
<p>The LLM’s finite-window error exponents are approximately 0.61 versus 0.88 over 64–256 baseline-update-equivalent compute, but 0.41 versus 0.46 over 128–512. This is evidence of a learning-curve advantage, not a stable exponent separation. Generator marginal likelihood does not show a comparable sleep advantage; the demonstrated improvement is inference.</p>

<h2>Mechanism and the next discriminating test</h2>
<p>A generated cause–observation pair provides the inverse model with a cause label it may rarely propose itself. Forward posterior fitting penalizes missing such explanations. Near a correctly learned posterior, however, both KL directions have the same Fisher curvature, even through a deep recognition network. Any stronger asymptotic scaling claim must therefore involve persistent recovery of increasingly difficult directions, changing representations, or a nonlocal regime—not depth by itself.</p>
<p>The main remaining limitation is structural: the toy explicitly enumerates latent configurations and gives the recognizer one output per configuration. Its cost is exponential in the number of latent bits. Its affine posterior can also be compiled directly from the generative weights, so this is a comparison between specified learning rules, not an algorithm-independent advantage. A stronger follow-up should use an autoregressive recognizer over compositional causes, grow the same hypothesis family in the LLM, and test whether the advantage survives equal-compute posterior approximations. The present eight-rule LLM experiment establishes the mechanism’s practical match, not that larger hypothesis spaces reproduce the toy’s numerical exponents.</p>
<p>Related foundations: <a href="https://proceedings.mlr.press/v115/le20a.html">Le et al., reweighted wake–sleep</a> distinguish inference objectives and model-versus-data mismatch. <a href="https://arxiv.org/html/2409.17858v2">Bordelon et al., feature learning and scaling</a> motivate examining evolving difficult directions. <a href="https://arxiv.org/abs/2006.08381">DreamCoder</a> connects wake–sleep to learning program representations. These papers do not establish the scaling claim tested here.</p>

<h2>Evidence and reproduction</h2>
<p>Gradient audits cover 144 directional derivatives, maximum discrepancy below 10⁻¹⁰, and local forward/reverse curvature convergence. Runtime copies, source hashes, protocols, trajectories and final weights are retained for every toy stage. LLM runs save adapters, optimizer state, RNG state and data-stream position every 32 updates; the actual matched-budget check exercised resume. Checkpoints and model downloads use configured cache storage. Nothing was pushed or published externally.</p>
<p><a href="deep-wake-sleep-data/size_analysis.json">Size analysis and raw target costs</a> · <a href="deep-wake-sleep-data/size_forecast.json">Forecast saved before extrapolation</a> · <a href="deep-wake-sleep-data/per_size_rate_tuning/selected.json">Separate rate searches at each size</a> · <a href="deep-wake-sleep-data/analysis.json">Full toy comparisons</a> · <a href="deep-wake-sleep-data/gradient_audit.json">Gradient audit</a> · <a href="llm-wake-sleep-data/analysis.json">LLM curves</a> · <a href="llm-wake-sleep-data/matched_evaluation.json">Actual matched-budget evaluations</a> · <a href="deep-wake-sleep-data/">All toy run artifacts</a> · <a href="llm-wake-sleep-data/">All LLM run artifacts</a></p>
<details><summary>Code and derivations</summary><p class="links">{code_links}</p></details>
</body></html>'''
    (REPORTS/"wake-sleep-scaling.html").write_text(report)
    print(REPORTS/"wake-sleep-scaling.html")


if __name__=="__main__": main()
