# PROC confirmatory results (7 October 2026)

Protocol: `PROC_PROTOCOL.md`. Data: `results/task_search/proc_main` (34 of 40 episodes usable). Figure:
`reports/proc-confirmatory.png`. Six ~8B models, two per language (English, Japanese, Chinese), write worked
executions of a hidden 3–4-step number procedure they are told. Readers see only 1, 2 or 4 worked examples and then
either recognise the steps (4-way questions) or deploy the procedure on new numbers.

Contrasts are on the double-centred reader × source matrix (reader skill and general source quality removed); 95%
episode bootstrap CIs and two-sided sign-flip p over 34 episodes.

| | N = 1 | N = 2 | N = 4 |
|---|---|---|---|
| **Own − same-language peer, deployment** | **+6.5 [+3.4, +9.6]** | +4.8 [+1.1, +8.8] | +4.7 [+1.7, +7.9] |
| Own − same-language peer, recognition | −2.7 [−4.7, −0.6] | +2.3 [−0.8, +5.4] | +5.5 [+3.3, +7.6] |
| Own − peer, deployment, cells where every step was recognised | +10.0 [+6.1, +13.9] | +9.3 [+4.8, +14.3] | +10.5 [+6.6, +14.7] |
| Same-language peer − other language, deployment | +3.0 [−0.1, +6.4] | +2.6 | +0.6 |
| Same-language peer − other language, recognition | +4.9 [+2.3, +7.3] | +2.8 [+0.5, +5.1] | +0.1 |

- **Primary 1 (self in deployment, mean over N): +5.3 pp [+2.6, +8.3], p = 0.001, Holm p = 0.002.**
- **Primary 2 (dissociation, self in deployment − self in recognition): +3.6 pp [+0.5, +6.8], p = 0.034, Holm p = 0.035.**
- The effect grows with procedure length. Self in deployment is +8.3, +7.6 and +7.4 pp for 4-step procedures, against
  +5.0, +2.6 and +2.5 pp for 3-step ones.
- **By reader:** positive for 5 of 6 at N = 1. The English and Chinese pairs are positive at every N. The Japanese pair
  is mixed: Gemma2-Swallow is negative at N = 2 and 4.
- **Not verbatim copying:** the readers' test solutions overlap their own demonstrations no more than a peer's (4-gram
  overlap 0.65 vs 0.64).
- **References:** input/output pairs alone give 0%. The procedure stated in the reader's language gives 38–85%.
  Worked examples give 62–74%.

**Reading.** This is the first pre-registered own advantage in the project. A model deploys a procedure better from its
own worked executions than from a same-language peer's, with correctness, inputs and source quality controlled. The
advantage is not mirrored in recognition: after one example a reader recognises its own demonstration's steps slightly
*worse*, yet deploys them 6.5 pp better. Among fully recognised cases the gap is about 10 pp. Language (other vs same)
mainly affects recognition after a single example.

**Caveats:**

- One task family; ~8B models.
- The panel differs from LANG (Llama-3.1-8B and Swallow-8B were too weak as writers).
- The dissociation is modest (p = 0.034) and carried by N = 1. At N = 4 recognition shows a similar own advantage.
- "Every step recognised" is a selected comparison on raw accuracy.

## More recognition data (PROC-REC, `PROCREC_PROTOCOL.md`, 7 October 2026)

73 episodes: the 34 above plus 39 fresh ones run through the full pipeline. Each cell got 3 × L step-identification
questions and 4 verification items (number chains, half with one wrong step). Figure: `reports/proc-recognition.png`.

- **Deployment self effect replicates** on the fresh episodes: +5.5 pp [+3.1, +7.8], p < 0.001. Pooled: +5.4 pp
  [+3.6, +7.2].
- **Verification is at chance for every reader** (50–55%, 48% "yes"), so it is uninformative. The pre-registered
  primary, self on the recognition composite, is therefore null: −0.3 pp [−1.0, +0.3].
- **Step identification is at ceiling for four readers** (OLMo-3-7B 99%, Phi-4-mini 97%, GLM-4-9B 100%,
  InternLM3-8B 100%). For them own − peer recognition is 0.0 pp, yet own − peer deployment is **+7.6 pp
  [+5.2, +10.0]**: +7.5, +7.1 and +8.0 at N = 1, 2 and 4, all p < 0.001. On the fresh episodes alone (pre-registered
  by amendment 1 before they were analysed) it is +7.3 pp [+4.4, +10.3].

  Per reader the deployment gain is OLMo +14.8, GLM +12.6, Phi-4-mini +5.7 and InternLM −2.9 pp.
- **The sign-changing recognition self effect** (−3.0, +1.2, +4.3 pp at N = 1, 2, 4) comes entirely from the two
  Japanese readers, who recognise poorly and in opposite directions: Gemma2-Swallow (29% of steps) recognises its own
  examples 29 pp *worse*, ABEJA (59%) 34 pp better. Neither shows a deployment own effect.
- Dissociation (self in deployment − self on the composite): +5.7 pp [+3.8, +7.6], p < 0.001.

**Reading.** For readers who understand both their own and a peer's demonstration completely, the own demonstration
still leads to better deployment, by about 7–8 pp, in three of the four. The ambiguity in the earlier recognition
numbers was idiosyncratic behaviour of the two weak recognisers, not a general pattern.
