# Cross-task reader–source specificity search: completed 5 October 2026

Protocol: XSPEC_PROTOCOL.md. Report: reports/task-specificity-search.html.
Data: results/task_search/xspec. OpenRouter spend: USD 3.91.

**No setting shows a mutual own-source advantage** (both Qwen3.5-9B and Gemma 4
26B A4B learning better from their own unjudged rehearsal answers than from the
other model's, same questions, no feedback).

- Calibration: 408 length-feasible reasoning_gym settings (spec × K∈{0,4}) at
  N=0 → 217 eligible → 80 chosen by baseline spread → 76 screened (4 exclusions).
- Screen (4 episodes × 8 tests, N=16): 3 settings had both preferences ≥ 6.25 pp;
  a within-pair label-swap null predicts 2.6 (P = 0.48). Both > 0: 10 vs 9.5.
- Confirmation (16 fresh episodes × 12 tests, N = 0, 2, 8, 16, 32; H at N=16):

| Setting | Qwen own − Gemma src | Gemma own − Qwen src | H | Holm p |
|---|---:|---:|---:|---:|
| knights_knaves K=4 | −2.1 [−6.8, 3.1] | +1.0 [−3.1, 5.2] | −0.5 [−3.1, 2.1] | 0.72 |
| color_cube_rotation L1 K=4 | +2.1 [−3.1, 6.2] | +6.8 [1.6, 12.5] | +4.4 [0.8, 7.8] | 0.071 |
| largest_island K=4 | +1.0 [−1.6, 4.2] | +0.5 [−3.1, 4.2] | +0.8 [−1.8, 3.6] | 0.72 |

- Colour cube: one-sided (Gemma only), concentrated in the Novita provider block,
  and tied to Gemma writing working into its answer under the 256-token cap.
- Large screen effects are one-way source effects (they cancel out of H), e.g. simple
  integration K=0: Gemma 91% with Qwen's examples vs 66% with its own.
  The extreme graph-colour K=4 point (Gemma 3% own vs 69%) is a ```json-fence
  artefact under the strict validity check; fence-stripped rescoring leaves the
  screen selection and every confirmation score unchanged (xspec/fence_sensitivity.json).
- Changes made during the run (scheduling/transport only, plus provider blocking in the
  confirmation) are listed in XSPEC_PROTOCOL.md and in each phase's design.json.
