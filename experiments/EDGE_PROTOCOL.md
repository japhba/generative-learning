# Edge of capability: own versus peers' unchecked attempts, matched in number correct

Frozen 6 October 2026 (UK) before any inference for this study.

**Hypothesis (from the user).** On problems at the edge of a model's capability, a model learns more easily from its
own attempts — right or wrong — because it comprehends its own reasoning. Earlier studies (MLAYOUT, MTHINK, MSFT,
RENDER) showed only all-correct practice, mostly with capable 9–30B models for which the operator task was not at the
edge. Strong models' foreign examples help everyone (a source main effect), which must not be mistaken for an own
effect.

**Design choices that follow from this.**

- **Panel of peers.** Four small instruction models of similar strength from four families: Qwen3-1.7B (thinking off),
  Llama-3.2-3B-Instruct, SmolLM3-3B (thinking off), Phi-4-mini-instruct. No strong teacher is in the panel, so no
  source can dominate by quality. Same pinned revisions as SMALL; vLLM 0.28 on gpu_lowp, greedy at test.
- **Edge by construction.** Episodes are hidden-rule problems (the rule is shown by K = 4 solved examples). Each model
  makes 6 attempts (step by step, T = 0.8, top-p 0.9, 1,024 tokens) at each of 12 pool problems. A pool problem is
  usable as a *correct* slot if every model has a finished, correct attempt on it, and as an *incorrect* slot if every
  model has a finished attempt with a wrong final answer. An episode enters the study only if every model both
  succeeds and fails on its problems, which places it at the edge for the whole panel.
- **Matched number correct.** The *mixed* practice set has 2 correct and 2 incorrect slots: the same 4 problems, the
  same correct/incorrect pattern and the same (seeded) order for every source. Per slot, each source contributes the
  qualifying attempt closest to the slot's median length across sources (ties by sample index). Slot choice: the first
  two incorrect-eligible pool problems, then the first two correct-eligible problems not already used; if that fails,
  correct first, then incorrect. The *all-correct* set (secondary) uses the first four correct-eligible problems.
- **Test.** The reader sees the K examples, one source's practice set labelled "Earlier attempts at practice problems
  (unchecked; some may be wrong)", then a new problem of the same episode, and reasons step by step to a final
  "Answer:" line (greedy, 1,024 tokens; 12 test problems per episode). The no-practice condition is exactly the attempt
  prompt. Attempts are never labelled by author. Truncated or unparseable answers count as wrong.

## Phase 1: calibration (12 episodes per candidate family; fresh seeds)

Candidate families: `operator` (a◇b = p·a + q·b, operands 11–49), `offset` (p·a + q·b + r with q possibly negative,
operands 10–59), `units` (two hidden conversion ratios 4–12), `cipher` (Caesar shift). Calibration runs attempts,
matching, the no-practice test and the practice conditions. **Selection rule (uses no own-vs-foreign quantity):** a
family is eligible if its mixed-set yield (fraction of episodes with a feasible mixed set) is at least 25% and every
model's mean no-practice test accuracy lies in [5%, 90%]. The (up to) two eligible families with the highest mixed
yield go to the main phase, each with E = min(150, ceil(48 / yield)) fresh episodes (about 48 included episodes per
family expected). Calibration own-vs-foreign numbers are reported as exploratory only.

## Phase 2: main (confirmatory)

Conditions per reader and included episode: none; mixed practice from each of the 4 sources; all-correct practice from
each source (episodes with a feasible all-correct set).

**Primary.** Per included episode, the 4 × 4 reader × source accuracy matrix of the mixed arm, and
S = mean(diagonal) − mean(off-diagonal), which cancels additive reader and source effects (in particular "this source's
attempts help everyone"). Primary = mean S over included episodes, pooled over the selected families; one-sided
sign-flip test over episodes (200,000 seeded random sign vectors), α = 0.05; 95% episode bootstrap CI (20,000).

**Secondary (95% CIs; per-reader tests Holm-corrected).**

1. S per family.
2. S in the all-correct arm, and S(mixed) − S(all-correct) on episodes with both sets.
3. Quality-adjusted own effect: for reader r, Q = [A(r,r) − mean_{s≠r} A(r,s)] − [h(r) − mean_{s≠r} h(s)], where h(x)
   is source x's mean accuracy for third-party readers (readers other than r and x). This removes how much more the
   own source helps *other* models than the foreign sources do.
4. Edge dose-response: own effect d = A(r,r) − mean_{s≠r} A(r,s) by tertile of the reader's no-practice accuracy on
   the episode, and the linear + quadratic regression of d on that accuracy.
5. Logit-scale S: cell accuracies (12·A + 0.5)/13 on the logit scale, double-centred per episode, mean diagonal.
6. Practice effect: mean over readers and sources of A(mixed) − A(none), and of A(all-correct) − A(none).
7. Own effect per reader (one-sided sign-flip, Holm over 4).
8. Exploratory: familiarity (reader's mean token log-likelihood of the practice attempts) as a mediator of the own
   effect.

## Amendment 1 (6 October 2026, 15:21 UK; after the calibration attempts, before any calibration test result was scored)

Calibration attempts show that `operator`, `offset` and `cipher` are beyond this panel: attempt accuracy 0–6%, mostly
truncated rambling at 1,024 tokens, and no episode has a feasible mixed set. `units` is at the edge for every model
(attempt accuracy Llama-3.2-3B 16%, Phi-4-mini 54%, Qwen3-1.7B 64%, SmolLM3-3B 80%), with mixed sets in 12/12 episodes.
At most one family can therefore be selected. To keep the confirmatory test adequately powered, a single selected family
gets E = min(150, ceil(120 / yield)) main episodes (about 120 included) instead of about 48. Nothing else changes.

## Amendment 2 (6 October 2026, 15:22 UK; after the exploratory calibration analysis, before any main-phase test was run)

Calibration (12 units episodes, exploratory) gave S(mixed) = −14.0 pp [−21.4, −6.5], S(all-correct) = −4.2 pp and
S(mixed) − S(all-correct) = −12.6 pp, the opposite of the hypothesis. The one-sided primary test stays as frozen; in
addition the main phase reports:

- the **two-sided** sign-flip p for S(mixed), S(all-correct) and S(mixed) − S(all-correct);
- **error adoption** (mechanism, exploratory): each incorrect practice attempt on "How many X are in k Y?" implies a
  wrong factor f = answer / k for its conversion type; a wrong test answer to a question of the same conversion type
  with value k′·f counts as adopting a practice error. Adoption rate by own vs foreign source (mixed arm), with an
  episode bootstrap; and the same predicted values in the no-practice answers, to check whether own errors are simply the
  errors the reader makes anyway.

## Amendment 3 (6 October 2026, 15:24 UK; before any main-phase test was run)

The calibration examples suggest that a reader is most persuaded by reasoning that matches its own tendencies,
including its errors, so own *unchecked* attempts may be self-reinforcing. If comprehending one's own reasoning helps
learning, it should show once attempts are marked: a model should learn most from its own mistakes when told they are
mistakes. **Labelled arm** (`edge_label.py`, run after the main phase on the same main episodes, mixed sets and readers):
the practice block is headed "Earlier attempts at practice problems, each checked against the true answer:" and every
attempt is followed by "Check: correct." or "Check: wrong." (the true answer is not shown). Analysis: S(labelled) with
two-sided sign-flip p and bootstrap CI; S(labelled) − S(unlabelled mixed) on the same episodes; own effect per reader;
practice effect of labelled vs unlabelled.

## Amendment 4 (6 October 2026, 15:44 UK; after the main and labelled results, which turned out to be invalid)

The frozen scorer (normalised exact match of the final "Answer:" line) mis-scored answer formats: Qwen3-1.7B often
writes `Answer: **Answer: 30**`, SmolLM3-3B and Llama-3.2-3B often write `4 * 72 = 288` or `126 tazzs` on the answer
line. As a result 108/202 of Qwen3-1.7B's and 46/202 of SmolLM3-3B's "incorrect" practice slots had the right number,
so the correct/incorrect pattern was **not** matched across sources, and readers that copied a format from the practice
were scored wrong (Qwen3-1.7B reading its own mixed set: 66% strict, 96% numeric). The main and labelled results are
therefore void as tests of the hypothesis.

Fix, applied to both slot selection and test scoring: an answer is correct if the **last number on the final
"Answer:" line** equals the true answer (an answer line without a number is wrong). Phase `main2` re-matches the same
120 main episodes from the same attempts with this scorer (all selection rules otherwise unchanged) and re-runs every
test condition; the labelled arm is re-run on the new sets (`label2`). All primary and secondary analyses (including
amendments 2–3) are repeated unchanged on `main2`/`label2`. The capability-matched checks (pairwise H among
Qwen3-1.7B, SmolLM3-3B and Phi-4-mini, and S without Llama-3.2-3B), first run post hoc on the void data, are now
declared secondary.
