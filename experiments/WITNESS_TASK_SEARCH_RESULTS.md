# Additional constructive tasks and a strategy-state toy

Completed 11 September 2026, 20:18 BST.

Twelve new families were tested at K=4 and K=0 with fixed questions in the teacher/own comparison. No additional task confirmed a source advantage over both controls. All model inference and audits are complete, the local GPU server is stopped, and the original data archive is preserved.

| Fresh confirmation (256 questions each, N=32) | Initial | Teacher | Own | Own − teacher, percentage points [95% paired-episode CI] |
|---|---:|---:|---:|---|
| witness_hamiltonian_path_l1, K=4 | 12.50% | 12.89% | 12.89% | +0.00 [-4.69, +4.30] |
| witness_bezout_l0, K=0 | 23.05% | 28.12% | 30.08% | +1.95 [-3.91, +7.81] |

The Bézout own-minus-instruction difference is +7.03 points [1.17, 12.11] pointwise, but the source difference is inconclusive. Both fresh joint tests fail even without multiplicity correction; joint Holm p=1 when earlier local primary attempts are included. This does not establish a power-law scaling advantage. The unselected topological-order K=0 screen was also positive in 32 queries, but the prospectively fixed one-candidate cap selected Bézout by the recorded tie-break; it remains exploratory.

All 16,192 distinct referenced raw requests passed prompt reconstruction. Every procedural world and saved score was independently checked. There was no context censoring; 43 generated and 175 evaluation call uses reached the output cap. Truncated generated slots retained empty answers and truncated evaluations scored incorrect under the unchanged protocol. All raw text remains available. The post-hoc last-JSON-array sensitivity gives Hamiltonian teacher=own=15.625%; Bézout is unchanged.

## Mechanistic hypothesis

A minimal fixed learner keeps strategy logits h. Each own rehearsal style is sampled independently from softmax(h0). Reading either source applies h <- h + diag(w) one_hot(style), without correctness feedback; final strategy probabilities are softmax(h). Correct teacher solutions can elicit a strategy that the limited reader executes poorly. A rare useful own cue can dominate when its mean logit increment w_j*p_j is larger. A salient invalid strategy can likewise cause eventual deterioration.

The two- and three-strategy curves were summed exactly, using illustrative unfitted parameters. Equal-cue and matched-strategy-teacher controls remove the source advantage. Identical replayed content always produces identical predictions. The model supplies a possible mechanism and falsifiable controls, not evidence that Qwen implements it.

A normalized cue likelihood reproduces the update. Generation and recognition assume different cue distributions. The coherent posterior-predictive null leaves the toy’s mean linear readout unchanged; posterior averaging by itself is not a no-learning theorem for externally observed data. Internally generated rehearsal adds no task information conditional on the model and initial data, but a limited decoder can benefit from its computation.

For a weight-space analogue, the existing linear Gaussian model generates z and x=Az+noise and trains a recognizer B by B <- B-eta*(Bx-z)*x^T. This closes a generative/recognition gap in the sense of [wake–sleep](https://www.cs.utoronto.ca/~hinton/absps/ws.htm). Ordinary exactly on-policy SFT instead has zero mean score gradient at its sampling parameters; finite-sample updates, temperature differences, and other asymmetries need separate analysis.

Finite mixtures do not imply power laws. A weighted density of slow modes proportional to r^(alpha-1) near zero can yield N^(-alpha); a steeper exponent requires changing that small-rate structure. No such spectrum has been established here.

## Literature and artifacts

[Self-ICL](https://aclanthology.org/2023.emnlp-main.968/) provides evidence for unjudged self-generated demonstrations improving zero-shot prompting. It does not establish superiority to exact teacher answers on identical rehearsal questions or a steeper asymptotic power law. The canonical HTML report gives the matched protocol, full curves, and ten actual rehearsals plus ten paired evaluations for each fresh candidate.

Report: http://127.0.0.1:8095/task-search-report.html

Full experiment archive and hash: results/task_search/witness_archive.json. Post-review presentation files are packaged separately without overwriting that archive. Toy assumptions: experiments/STRATEGY_CONTEXT_TOY.md. Full numerical results: results/task_search/witness_analysis/analysis.json.
