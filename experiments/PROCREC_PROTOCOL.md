# More recognition data for PROC (PROC-REC)

Frozen 7 October 2026, 21:30 UK, before any inference for this extension.

**Why.** In PROC (PROC_RESULTS.md) the own − same-language-peer effect on recognition changed sign with the number of
worked examples: −2.7 pp at N = 1, +2.3 at N = 2, +5.5 at N = 4. Recognition per reader × source cell rested on only
L = 3–4 four-way questions, and some readers were at 100%.

**More items per cell.** All items are seeded independently of the frozen episodes, from `rec|<episode seed>`.

- *Step identification*: two more option sets per step (new distractors for arithmetic steps, new option order for
  all), each asked as in PROC. Together with the original set that makes 3 × L questions per cell.
- *Verification* (new, discriminative): 4 executions of a procedure on new numbers, shown as neutral number chains
  ("4726 → 6274 → 5711 → 1157"). Two follow the hidden procedure; two have exactly one step replaced by a
  same-family alternative, and all values stay positive. The reader answers "Answer: yes/no" in its language:
  "Is every step consistent with the procedure used in the worked examples?"

**More episodes.**

- Set A: the 34 usable PROC confirmatory episodes.
- Set B: 40 fresh episodes (seed prefix `proc_main2_20261007`) through the full PROC pipeline: writers, matching,
  deployment, original probes. This also replicates the deployment effect.

Conditions per reader: input/output pairs only, and N ∈ {1, 2, 4} worked examples from each of the 6 sources (as in
PROC).

**Measures per cell:**

- step identification = accuracy over the 3 × L questions;
- verification = accuracy over the 4 chains;
- **recognition composite** = mean of the two.

**Statistic.** As in PROC: on the double-centred 6 × 6 reader × source matrix, **self** = own − same-language peer and
**language** = same-language peer − other language. Two-sided sign-flip over episodes (100,000 draws), 95% episode
bootstrap CIs, sets A and B pooled unless stated.

- **Primary:** self on the recognition composite, averaged within episode over N ∈ {1, 2, 4}.
- **Secondary:**
  1. Self on the composite at each N, and the trend N = 4 − N = 1.
  2. Self on step identification (all 3 sets) and on verification separately.
  3. Dissociation = self in deployment − self on the composite (mean over N).
  4. Replication of the PROC deployment self effect on set B alone.
  5. Language contrast on each recognition measure.
  6. Ceiling and bias checks: per-reader accuracy on each measure, and the verification "yes" rate.

## Amendment 1 (7 October 2026, 22:10 UK; after set A, before any set B analysis)

Set A showed two things.

- **Verification is at chance for every reader** (50–54% correct, 48% "yes"). These models cannot check a number chain
  against a procedure without working, so the composite is half noise. The primary stays as frozen and is reported, but
  step identification is the informative recognition measure.
- **Step identification is at ceiling for four readers** (OLMo-3-7B, Phi-4-mini, GLM-4-9B, InternLM3-8B: 96–100%) and
  low for the two Japanese readers (Gemma2-Swallow 29%, ABEJA 57%). For the ceiling readers, own and peer examples are
  recognised equally by construction.

Added secondaries (pooled sets A + B):

- (a) Self (own − same-language peer) per reader, on step identification and on deployment.
- (b) The **ceiling-reader analysis.** The four readers whose mean step-identification accuracy on set A is ≥ 95%:
  their deployment own − same-language peer (raw accuracy, row-wise), and the same on step identification.
  Two-sided sign-flip over episodes.
