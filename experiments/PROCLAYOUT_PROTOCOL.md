# PROC-LAYOUT: does the own advantage follow the layout or the authorship of the worked example?

Drafted 5 November 2026 before any rewrite was produced. **Frozen 05 November 2026, 15:30 UK**, after the rewrite pilot and before any
reader saw a rewritten example.

**Question (user, 5 Nov).** In PROC a reader applies a demonstrated procedure better from its own worked example than
from a same-language peer's (+6.5 pp at N = 1), although it recognises the steps from both. In the clearest episode
(L3__11, OLMo-3-7B: 8/8 after its own example, 0/8 after Phi-4-mini's) the reader copied the layout of whichever example
it saw, and only its own layout wrote out the digits of the swap step it otherwise gets wrong. Two accounts:

- **H1, layout.** The reader imitates the example's layout, and its own layout spells out what it needs.
- **H2, authorship.** Beyond layout, the reader reconstructs its own latent computation from text it wrote itself (the
  user's hypothesis), so its own wording helps even in someone else's layout.

**Design.** The 73 usable PROC episodes (34 confirmatory in `proc_main`, 39 fresh in `proc_main2`), N = 1, the first
matched worked example of each source. For each language pair (OLMo-3-7B / Phi-4-mini, Gemma2-Swallow-9B / ABEJA-Qwen2.5-7B,
GLM-4-9B / InternLM3-8B) and episode, the two examples A and B are rewritten by one fixed model into four texts:
A's wording in A's layout, A's wording in B's layout, B's wording in B's layout, B's wording in A's layout.

- **Rewriter:** `anthropic/claude-sonnet-5.5`, provider pinned to Anthropic, temperature 0, cached; one instruction for
  every transplant (`proc_layout.py`, `PROMPT`), which first maps each line of the layout source to the phrase of the
  wording source it will use (or NEW) and then writes the text. Layout = structure and markup and which elements each step contains (e.g.
  digits listed one by one, a spaced digit string, before/after lines, code blocks, introduction, closing line). Wording =
  the sentences and phrases, kept verbatim from the source text where the target layout has a place for them.
- **Own wording in own layout** is the original example. In the pilot every identity rewrite came back verbatim (24/24),
  so OO and PP use the originals; passing through the rewriter then cancels in L and W and enters only the interaction.
  All four cells are run afresh in the same server session; the stored PROC outputs for OO and PP serve as a
  reproducibility check.
- **Validity:** a rewrite must keep the right final answer and every intermediate value (the PROC criteria). An invalid
  rewrite is retried up to three times at temperature 0.3; if any of a pair's four rewrites stays invalid, that pair is
  dropped from that episode (both readers).
- **Readers:** the same six models, prompts, test numbers, seeds and settings as PROC (greedy, 1,024 tokens, numeric
  scoring of the final answer line). Each reader sees one example in one of four cells, relative to the reader:
  OO (own wording, own layout), OP (own wording, peer layout), PO (peer wording, own layout), PP (peer wording, peer
  layout). Accuracy per reader × episode × cell is the mean over the 8 test numbers.

**Primary (two-sided, Holm over the two):**

1. **Own layout** L = mean[(OO + PO) − (OP + PP)] / 2.
2. **Own wording** W = mean[(OO + OP) − (PO + PP)] / 2.

Averaged within each episode over the readers (both members of a pair, so a layout's or wording's general quality
cancels), then over episodes: sign-flip test over episodes (100,000 draws), 95% episode bootstrap CI. H1 predicts L > 0
with W near 0; H2 predicts W > 0.

**Secondary:** the interaction (OO − OP − PO + PP); OO − PP (the own effect after rewriting) against the original
own − peer on the same episodes; each contrast per reader, per language pair, per procedure length, and for the four
readers that recognise every step (OLMo, Phi-4-mini, GLM, InternLM); the confirmatory and fresh episode sets separately.

**Manipulation checks** (reported, not used for exclusion): a layout signature per text (the sequence of line types:
heading, bullet, numbered, code, spaced digit string, plain text, blank) and its similarity to each source; the share
of character 4-grams of the words (digits and markup removed) shared with each source. A transplant should be closer
to its layout source in layout and to its wording source in wording.

**Exploratory:** layout similarity between a reader's test solutions and the example it saw, by cell.

**Budget:** rewrites ≤ USD 5 on OpenRouter (expected under USD 1). Readers on gpu_saxe, at most 2 GPUs.

## Rewrite pilot (before freezing; 2 episodes per set, 24 transplants)

| Rewriter, instruction | layout similarity to layout source / wording source | digit listings follow layout source | wording from wording source* |
|---|---|---|---|
| Haiku 5.5, first instruction | 0.85 / 0.51 | n/a | n/a |
| Haiku 5.5, "exactly the elements of B" | 0.95 / 0.48 | 15 / 18 | 0.67 |
| Sonnet 5.5, same | 0.97 / 0.47 | 7 / 18 | 0.82 |
| **Sonnet 5.5, with a line-by-line plan (used)** | **1.00 / 0.47** | **15 / 18** | **0.77** |

\* Of the transplant's character 4-grams found in exactly one of the two sources, the share from the wording source
(identity: 1.00). Transplants of a terse wording source into a verbose layout borrow most (English pair, minimum 0.25),
so W is somewhat attenuated there; the per-pair wording shares are reported with the results. All 24 pilot transplants
kept the right answer and every step result. Pilot spend USD 1.10; full run expected about USD 5.
