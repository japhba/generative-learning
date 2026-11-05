# PROC-LAYOUT results (5 November 2026)

Protocol: `PROCLAYOUT_PROTOCOL.md` (frozen 15:30 UK). Code: `proc_layout.py`. Data: `results/task_search/proc_layout`.
Figure: `reports/proc-layout.png`. 73 PROC episodes (34 confirmatory + 39 fresh), N = 1, six readers, 8 test numbers each;
no pair was dropped (all 438 transplants kept the right answer and every step result).

**The own advantage follows the layout of the worked example, not its wording.**

| Cell (relative to the reader) | own layout | peer layout |
|---|---|---|
| own wording | 73.9% | 67.3% |
| peer wording | 73.2% | 67.9% |

- **Primary 1, own layout L: +6.0 pp [+4.1, +8.0], p < 10⁻⁵ (Holm < 10⁻⁵).**
- **Primary 2, own wording W: +0.1 pp [−1.2, +1.3], p = 0.95.**
- Interaction +1.3 [−0.9, +3.4]. OO − PP reproduces the original own effect exactly (+6.1 pp; the fresh greedy runs of
  the original examples matched the stored PROC outputs in every case).
- Both episode sets: L +6.1 (confirmatory), +5.9 (fresh); W +0.4, −0.2. Larger for 4-step procedures (L +7.7) than
  3-step ones (+4.5). The four readers that recognise every step: L +7.5 [+5.0, +10.1], W +0.0.
- Per reader, L: OLMo +13.5, GLM +15.6, Phi-4-mini +3.3, ABEJA +3.6, Swallow +2.3, InternLM −2.4; W between −1.9 and
  +2.7 for all. In two of the three language pairs both members prefer their own layout, so this is a match between
  reader and layout, not one layout being better for everyone (that part cancels in L by construction).
- **Manipulation checks.** Transplants match the layout source (line-structure similarity 1.00 vs 0.40 to the wording
  source); where the two sources differ in writing digits out one by one, the transplant follows the layout source in
  76% and the wording source in 7%. Wording: 73% of the source-specific phrases come from the wording source (70–80% by
  pair), so W is attenuated by a factor of about 0.46; its upper CI bound still corresponds to at most about +3 pp for a
  complete wording swap.
- **Exploratory (mechanism).** Readers copy the shown layout almost perfectly: their test solutions are closer in layout
  to the shown example than to the same wording in the other layout in 99.3% of cases (similarity 0.96 vs 0.40). They
  write digits out one by one whenever the example does (100%) and almost never otherwise (6%). Readers are more
  accurate when they write digits out (73% vs 65%; GLM 70% vs 37%, OLMo 78% vs 64%), a correlational pattern consistent
  with the layout carrying the scaffolding a reader needs.

**Reading.** In PROC a model does better from its own worked example because the example is laid out the way it
works: the reader imitates the layout, and its own layout spells out the sub-steps it needs. A peer's example re-laid
out in the reader's layout is as good as the reader's own (73.2% vs 73.9%); the reader's own example in the peer's
layout is as poor as the peer's (67.3% vs 67.9%). Authorship of the sentences adds nothing measurable. If "own latents"
are reconstructed from an own example, they live in its step structure, not in its wording.

**Caveats.** One task family; ~8B models; one rewriter (Claude Sonnet 5.5) and an imperfect wording separation; "layout"
includes which sub-steps are written out, so it is not purely typographic. Rewrites cost USD 7.4 (pilots included).
