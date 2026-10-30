# Recognised but applied worse? Holding recognition fixed (LANG-APPLY)

Frozen 7 October 2026, 08:40 UK, before any inference for this study.

**Background.** In LANG (LANG_RESULTS.md), other-language worked solutions were recognised less often (the hidden-rule
probe was correct in 46% vs 57% of cases) and were applied worse. Among recognised cases, the deficit was +4–6 pp at
N ≥ 2 on the interaction scale and not significant at N = 1. That evidence is weak in three ways:

1. Recognition was probed in a separate call, so the reader need not "know" the rule while solving.
2. Conditioning on recognition compares different subsets of cells.
3. There was one binary probe per cell.

**Design.** Same six models, task, prompts and practice selection as LANG; 2 H100s on gpu_saxe. For every reader,
source and N ∈ {1, 2, 4}, two new conversation conditions:

- `inj:s:N`, **rule injected.**
  1. User turn: the practice from source s and the rule probe (as in LANG).
  2. Assistant turn: the *correct* rule, written as if the reader had said it (`Ratios: x, y, z`, cumulative factors).
  3. User turn: the test instruction and problem.

  Recognition is thereby 100% and identical for every source. Any remaining source-language effect on test accuracy is
  a difference in *applying* the demonstrations.
- `self:s:N`, **self-stated rule.** The same conversation, but the assistant turn is the reader's own greedy answer to
  the probe (identical to its LANG probe call). The reader has just stated a rule itself, right or wrong.
- `inj:none`: the correct rule is injected with no practice (application with the rule known and no demonstration).

**Episodes.**

- Set A: the 29 usable LANG episodes (practice and probes reused).
- Set B: 40 fresh episodes run through the full LANG pipeline (seed prefix `lang_main2_20261007`), including its
  standard conditions. This also gives an independent replication of the LANG primary.

**Statistic.** D is the double-centred 6 × 6 reader × source accuracy matrix per episode and N (reader skill and
general source quality removed). The contrast C is mean D(same-language peer cells) − mean D(other-language cells).

- **Primary:** C under rule injection, averaged within each episode over N ∈ {1, 2, 4}, then over all usable episodes
  of sets A and B. Two-sided sign-flip test over episodes (100,000 draws), 95% episode bootstrap CI.
- **Secondary:**
  1. C under injection for each N and each set.
  2. Replication of the LANG primary (standard condition, C at N = 1) on set B.
  3. Self-stated: C restricted to cells whose inserted self-stated rule was correct (interaction-scale residuals), and
     C over all self-stated cells.
  4. C under injection by test-item type: forward one-step, forward multi-step, reverse.
  5. Demonstration value beyond the rule: accuracy under `inj:s:N` minus `inj:none`, by relation (own, same-language
     peer, other language).
  6. Own − same-language peer under injection.
  7. Ceiling check: mean accuracy under injection per reader. If most readers exceed 95%, the injected test has little
     room, and this is reported.
