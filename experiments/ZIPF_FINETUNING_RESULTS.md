# Matched Zipf fine-tuning results

Ranked observations produce faster finite learning curves in both fine-tuned models.

Effective exponents over the prespecified 8–128-label window; 95% paired world-bootstrap intervals:

| Learner | IID | Ranked | Difference |
|---|---|---|---|
| Exact finite toy | 0.327 | 0.502 | 0.175 |
| Qwen3-0.6B | 0.354 [0.312, 0.390] | 0.574 [0.521, 0.625] | 0.220 [0.157, 0.284] |
| SmolLM2-135M | 0.338 [0.298, 0.375] | 0.492 [0.472, 0.511] | 0.154 [0.114, 0.202] |

Qwen3-0.6B: final error with its own generated queries is higher relative to IID study; paired error difference 0.559 [0.493, 0.626].
SmolLM2-135M: final error with its own generated queries is higher relative to IID study; paired error difference 0.680 [0.619, 0.732].

The original generated-query decoder required EOS and biased both models toward long, rare keys. A newline boundary removes that bias. Its separate fresh-world acquisition calibration is reported in the HTML; the fixed-curriculum exponent comparison is unaffected.

Both world-4000 passive replays reproduced the original live condition's saved weights and checkpoint probabilities exactly.

Full methods, all arms, calibration failures, sensitivity windows, resource accounting and downloadable provenance are in [the standalone report](../reports/zipf-finetuning.html).

These are finite-range coverage effects under verified feedback, not an isolated generation-versus-recognition effect or a demonstrated asymptotic law.

New API and Vast spending: $0. Institutional allocations released.
