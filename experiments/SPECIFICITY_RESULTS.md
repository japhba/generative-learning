# Reader–source specificity: completed 12 September 2026

The fixed fresh cohort does **not** show both readers preferring their own
samples in any of the three settings at the prespecified N=32 endpoint. The
largest benefit transfers from Qwen-generated examples to Gemma.

Qwen3.5-9B (pinned local BF16) and Gemma 4 26B A4B (OpenRouter, DekaLLM BF16)
each read the same questions with Qwen, Gemma, or exact teacher rehearsal answers.
All prompts forbid two or more 1s in factorization. Generation is offline and
unjudged; all saved content, including truncated prefixes, is replayed. There
are 16 fresh episodes × 12 test queries per setting. K counts initial teacher
pairs. All curves N=0,2,8,32 are retained.

| Setting / reader | Initial only | Qwen source | Gemma source | Teacher source |
|---|---:|---:|---:|---:|
| Factor K=0 / Qwen | 9/192 | 36/192 | 7/192 | 5/192 |
| Factor K=0 / Gemma | 1/192 | 115/192 | 1/192 | 1/192 |
| Factor K=4 / Qwen | 8/192 | 16/192 | 1/192 | 7/192 |
| Factor K=4 / Gemma | 3/192 | 2/192 | 2/192 | 2/192 |
| Bézout K=0 / Qwen | 53/192 | 48/192 | 66/192 | 55/192 |
| Bézout K=0 / Gemma | 143/192 | 137/192 | 131/192 | 125/192 |

Only the initial column has N=0; other columns have N=32. Own-source preference
means own minus the other model's source. Pointwise 95% paired episode-bootstrap
intervals, in percentage points:

| Setting | Qwen preference | Gemma preference | Mean preference H | Primary Holm p |
|---|---:|---:|---:|---:|
| Factor K=0 | +15.10 [3.12,29.17] | −59.38 [−69.79,−50.00] | −22.14 [−29.17,−15.36] | 1.000 |
| Factor K=4 | +7.81 [3.65,12.50] | 0.00 [0.00,0.00] | +3.91 [1.82,6.25] | 0.00586 |
| Bézout K=0 | −9.38 [−12.50,−6.25] | −3.12 [−9.38,3.12] | −6.25 [−10.42,−2.60] | 1.000 |

H is half the additive reader × source interaction. The positive K=4 interaction
is entirely due to Qwen's preference, while Gemma is at a floor and has identical
episode-level accuracy under both generated sources. It is not evidence of a
mutual own-source advantage. The zero-width empirical interval for Gemma at K=4
does not prove that the population difference is zero. The one-sided sign-flip
test assumes symmetric/exchangeable episode contrasts; it is not a guaranteed
randomized-treatment test.

## What seems to transfer

At K=0, all 36 correct Qwen answers and 108/115 correct Gemma answers with Qwen
rehearsal use exactly one 1. This is allowed by the explicit rule. The result is
consistent with portable elicitation of an easy valid decomposition strategy.
It does not demonstrate improved general factorization or knowledge of authorship.

For example, the first Qwen rehearsal in episode 0 answers target 19,371,704
with [1,2,9685852], while its teacher answer is [748,563,46]. On the held-out
target 8,625,606, both readers with Qwen rehearsal answer [1,2,4312803].
Gemma with its own rehearsal answers [186,211,214], and with teacher rehearsal
[154,211,266]; both products are wrong. This is the first successful Gemma
transfer case in file order, selected for illustration after outcomes were known.
The report also shows the unselected first 10 rehearsals and first 10 tests
in episode 0 of every setting, with complete text and every reader/source condition.

## Important limits

- Source correctness and length differ substantially. Factor K=0 rehearsal
  validity is Qwen 86/512 (16.8%), Gemma 4/512 (0.8%), teacher 512/512.
  Mean answer characters are respectively 16.1, 355.5, and 14.7.
- The reasoning channel is disabled, but Gemma still sometimes writes visible
  working. Its factor K=0 initial-only responses are truncated in 150/192 cases.
  At N=32, truncations are 33/192 with Qwen, 16/192 with Gemma, and 10/192 with
  teacher rehearsal. Thus the large Qwen-source advantage over the other two
  rehearsal sources is not explained by *fewer* truncated final responses.
  The initial-only comparison is strongly affected by the output cap.
- Across all settings and N, generation truncations are Qwen 3/1536 and Gemma
  415/1536; final truncations are Qwen 41/5760 and Gemma 541/5760. These are
  modeling/format limitations, not dropped requests. The new 256-token cap and
  verbatim truncated-prefix policy differ from the earlier task-search protocol.
- Last-complete-JSON-array rescoring leaves all factorization endpoint accuracies
  unchanged. In Bézout, Gemma's own and other-source accuracies both become
  71.875%; Qwen's preference remains negative. Primary strict scores are unchanged.
- These are two previously selected task families with 16 fresh episodes each.
  Floor effects and wide intervals limit conclusions about other tasks/models.
  There is no power-law fit or evidence for a changed asymptotic exponent.
- The remote serving revision and kernels are not pinned; complete API metadata
  and the fixed provider are recorded.

## Mechanistic toy

The new specificity_sleep_toy.py is a separate constructed example. Inputs are
(m,c), m>0 and c binary, and any z with |z|=m is valid. Generators choose m and
(−1)^c m; their sources are both 100% correct. The two restricted readers use
features m and (−1)^c m, respectively, and learn one scalar with the identical
squared-error recognition update for either source.

Own feature–target correlations equal 1; cross-source correlations are equally
likely ±1 and cancel in expectation. Both readers therefore prefer their own
source, without a source-dependent reward or correctness filter. Giving both
readers both features lets them learn either source. This is a fixed-generator
sleep phase; native representation alignment is assumed, not learned. It is a
possible mechanism, not a fit or a mechanism established by the LLM results.
See SPECIFICITY_SLEEP_TOY.md for the derivation and limitations.

## Evidence and checks

All 14,592 request uses reconstructed exactly from the frozen worlds and saved
answers; 14,400 unique requests because identical full requests are cached.
Every remote experimental response was from DekaLLM, with zero reported
reasoning-channel tokens. Context overflow: zero. Gemma experiment cost:
USD 0.58486518, plus USD 0.00000336 for the unrelated infrastructure probe.
Three transient remote failure records were retained and resolved by retries.

The full inference sources and protocol were frozen before generation. The
analyzer was frozen before final inference and before any correctness inspection.
All data are under results/task_search/specificity. The canonical one-page
report has expandable full curves, source diagnostics, exact prompts, and
30 rehearsal examples plus 60 complete paired evaluation rows.
