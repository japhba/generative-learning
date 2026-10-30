# Local Qwen task search

Unjudged own answers improve three-factor product decomposition on fresh data. With 32 additional examples, Qwen3.5-9B scores 27.7%, versus 3.1% with teacher answers and 8.4% with the initial examples alone (512 held-out queries in 32 paired episodes). The gain over teacher is 24.6 percentage points, 95% episode-bootstrap interval [17.2, 32.8]; the gain over initial-only is 19.3 points [13.1, 26.0]. The joint test survives Holm correction across all 15 attempted confirmations (p=0.00015).

The task asks for exactly three positive integers whose product equals the given target T. The teacher draws three factors independently from 1 through 999 and supplies those valid factors. The learner may return any valid decomposition, including [1, 1, T]. In the own condition, only 6.3% of rehearsal answers are correct. Valid identity answers account for 80.3% of correct final predictions and 90.9% of the net gain over initial-only. This descriptive breakdown suggests a shift in solution strategy; it does not establish causal mediation. The questions and their order are identical between source conditions, so greater question coverage cannot explain the difference.

![Product decomposition learning curve](../reports/task-search-product-decomposition.png)

Across the completed fresh confirmations, 1 fixed-question and 0 complete-example comparisons pass the prespecified joint tests and correction within each comparison family. A positive effect must exceed both the teacher and initial-only conditions. The finite-range curves do not establish an asymptotic power law.

Qwen3.5-9B ran locally in BF16 through vLLM on the RTX 4500 Ada 24 GB. No paid API calls, weight updates, generated-sample filtering, correction, or correctness feedback were used. The earlier matched-source study remains separate.

The final learner always receives four initial correct examples, the same final questions and native answer instructions, temperature zero, thinking disabled, and a 1,024-token output allowance. Rehearsal generations use temperature 0.7. Independent generation sees the initial examples only; the explicitly identified online experiment additionally sees previous unjudged own answers. Only example content changes between source conditions at final evaluation; source names do not appear in prompts. Equal sample counts do not equal token or compute budgets.

## Fresh fixed-question confirmations

| Task | N | Queries | Initial only | Teacher | Own | Own − teacher, points [95% CI] | Own − initial, points | Joint Holm p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Prime count | 8 | 512 | 6.4% | 6.8% | 5.5% | -1.37 [-2.54, -0.20] | -0.98 | 1.0000 |
| Graph coloring | 8 | 512 | 58.0% | 59.2% | 60.9% | +1.76 [-1.56, +5.27] | +2.93 | 1.0000 |
| Maze distance | 8 | 512 | 20.5% | 21.9% | 22.9% | +0.98 [-1.76, +3.71] | +2.34 | 1.0000 |
| Number format | 8 | 512 | 85.2% | 88.5% | 84.6% | -3.91 [-7.81, -0.39] | -0.59 | 1.0000 |
| Palindrome partitions | 8 | 512 | 31.2% | 34.2% | 28.3% | -5.86 [-8.20, -3.71] | -2.93 | 1.0000 |
| Graph paths | 8 | 512 | 37.5% | 35.5% | 36.1% | +0.59 [-1.95, +3.32] | -1.37 | 1.0000 |
| Product of 2 integers | 32 | 512 | 29.7% | 14.8% | 36.1% | +21.29 [+13.48, +29.10] | +6.45 | 0.4785 |
| Product of 3 integers | 32 | 512 | 8.4% | 3.1% | 27.7% | +24.61 [+17.19, +32.81] | +19.34 | 0.0001 |
| Color allocation | 8 | 512 | 64.5% | 63.5% | 63.3% | -0.20 [-5.66, +5.47] | -1.17 | 1.0000 |

![Primary confirmation](../reports/task-search-primary-confirmation.png)

The additional rehearsal questions and their order are literally identical in this comparison. Number format used 32 new episodes of 16 tests and N=0,1,2,4,8,16,32, with primary endpoint N=8. The other five original tasks used 32 new episodes of 16 tests and N=0,2,8. The wording confirmation uses the seven-point curve with endpoint N=8; any online or arithmetic decomposition confirmations use the same seven-point curve with endpoint N=32. All 9 selected primary hypotheses remain in one multiplicity correction, including failed attempts.

## Fresh complete-example confirmations

| Task | N | Queries | Initial only | Teacher | Own | Own − teacher, points [95% CI] | Own − initial, points | Joint Holm p |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| Coin probability | 8 | 256 | 41.0% | 43.8% | 40.6% | -3.12 [-7.42, +1.17] | -0.39 | 1.0000 |
| Cube rotation | 8 | 256 | 30.5% | 32.4% | 32.0% | -0.39 [-4.30, +3.52] | +1.56 | 1.0000 |
| Bit count | 8 | 256 | 25.8% | 27.3% | 25.4% | -1.95 [-5.86, +1.56] | -0.39 | 1.0000 |
| Graph coloring | 8 | 256 | 55.9% | 57.0% | 53.9% | -3.12 [-8.20, +2.34] | -1.95 | 1.0000 |
| Graph paths | 8 | 256 | 31.6% | 36.7% | 32.4% | -4.30 [-8.98, +0.00] | +0.78 | 1.0000 |
| Self reference | 8 | 256 | 34.8% | 35.2% | 30.9% | -4.30 [-11.72, +2.73] | -3.91 | 1.0000 |

![Complete-example confirmation](../reports/task-search-secondary-confirmation.png)

These six tasks each used 32 fresh episodes of eight tests and N=0,2,8. The model generated both questions and answers with a generic same-family request; there were no targeted curriculum hints or test access. This comparison allows a change in question distribution. It cannot isolate a fixed-question advantage or an intrinsic effect of authorship.

## Number-format scaling and notation control

The selecting screen showed own 15/16 versus teacher 13/16; the 512-query confirmation reversed that difference. At N=8, own accuracy was 84.6%, canonical teacher 88.5%, initial-only 85.2%, and a valid teacher copying the winning candidate in input notation 85.5%. At N=32, these three rehearsal sources scored 86.1%, 91.0%, and 81.4%, respectively. Changing the valid teacher representation can change the curve substantially.

The prespecified finite-window fit log(error)=a−beta log(N), N=2,4,8,16,32, gives teacher beta 0.166, own beta 0.034, and own-minus-teacher beta -0.131 with 95% episode-bootstrap interval [-0.22945618526016276, -0.038992947241149475]. The own curve does not show reliably decreasing error over this window. The stricter nearest-displayed-candidate scoring diagnostic changed zero predictions.

![Number-format curve](../results/task_search/number_format_analysis/number_format_learning_curve.png)

## Broad search and focused strategy test

Baseline calibration examined 311 context-eligible settings across 99 families from 385 attempted procedural configurations. One setting per family was selected from initial-only accuracy, retaining floor and ceiling tasks. The source screen completed 194 episodes across 98 families: two episodes per setting where preflight succeeded, eight tests, N=0,2,8. Six fixed-question and six complete-example tasks exceeded both controls in this small exploratory screen, and every one was advanced to fresh confirmation. Polynomial multiplication lost both episodes to oracle/scorer disagreement; two other source-screen episodes were excluded before inference for context or oracle failure.

The focused graph-coloring extension varied the palette and edge density on six-vertex graphs, after a larger baseline-only grid. All nine compact settings received four episodes, eight tests, and N=0,2,8,32. None exceeded both controls at N=32. When six or twelve colors were available, the model spontaneously produced an injective, always-valid coloring in 23/768 rehearsal samples; only 424/768 of those rehearsals were valid overall. The extra strategy variety did not establish a generation benefit.

| Palette setting | Initial only | Teacher, N=32 | Own, N=32 | Own − teacher, points |
|---|---:|---:|---:|---:|
| graph_palette_v6_c12_p15__default | 87.5% | 84.4% | 78.1% | -6.25 |
| graph_palette_v6_c12_p40__default | 62.5% | 43.8% | 43.8% | +0.00 |
| graph_palette_v6_c12_p70__default | 40.6% | 53.1% | 37.5% | -15.62 |
| graph_palette_v6_c3_p15__default | 71.9% | 87.5% | 81.2% | -6.25 |
| graph_palette_v6_c3_p40__default | 53.1% | 62.5% | 50.0% | -12.50 |
| graph_palette_v6_c3_p70__default | 43.8% | 25.0% | 31.2% | +6.25 |
| graph_palette_v6_c6_p15__default | 81.2% | 87.5% | 81.2% | -6.25 |
| graph_palette_v6_c6_p40__default | 56.2% | 46.9% | 50.0% | +3.12 |
| graph_palette_v6_c6_p70__default | 28.1% | 46.9% | 46.9% | +0.00 |

## Shared task wording

The same six-entity allocation problem was expressed as meeting slots, storage bins, record codes, radio channels, or vertex colors. Wording was identical between source arms. Four baseline queries selected one geometry per wording without source comparisons; all five wordings then received four new episodes and eight tests, N=0,2,8. These are five wordings of one mathematical task, not five new task families. Underlying graphs are paired across wordings at a fixed geometry.

| Shared wording | Initial only | Teacher, N=8 | Own, N=8 |
|---|---:|---:|---:|
| meeting_slots_v6_c12_p70 | 34.4% | 25.0% | 21.9% |
| radio_channels_v6_c6_p70 | 37.5% | 71.9% | 40.6% |
| record_codes_v6_c6_p70 | 34.4% | 56.2% | 40.6% |
| storage_bins_v6_c6_p70 | 40.6% | 62.5% | 34.4% |
| vertex_colors_v6_c6_p40 | 59.4% | 62.5% | 68.8% |

Only the vertex-color wording exceeded both controls in the selecting screen. Its fresh 512-query confirmation appears in the primary table regardless of outcome, with all seven sample counts retained.

## Sequential unjudged rehearsal

The online screen repeats all 36 original palette episodes and the same assigned questions, but each own rehearsal answer can condition on previous unjudged own answers. No future question, final test, teacher rehearsal answer, correction, or strategy hint enters generation. Initial-only and teacher condition records match exactly between the independent and sequential screens. Every first generation request is identical. This changes generation input-token cost; the final source comparison still uses the same questions and decoding.

| Palette setting | Initial only | Teacher, N=32 | Independent own, N=32 | Sequential own, N=32 |
|---|---:|---:|---:|---:|
| graph_palette_v6_c12_p15__default | 87.5% | 84.4% | 78.1% | 84.4% |
| graph_palette_v6_c12_p40__default | 62.5% | 43.8% | 43.8% | 43.8% |
| graph_palette_v6_c12_p70__default | 40.6% | 53.1% | 37.5% | 34.4% |
| graph_palette_v6_c3_p15__default | 71.9% | 87.5% | 81.2% | 78.1% |
| graph_palette_v6_c3_p40__default | 53.1% | 62.5% | 50.0% | 56.2% |
| graph_palette_v6_c3_p70__default | 43.8% | 25.0% | 31.2% | 28.1% |
| graph_palette_v6_c6_p15__default | 81.2% | 87.5% | 81.2% | 78.1% |
| graph_palette_v6_c6_p40__default | 56.2% | 46.9% | 50.0% | 50.0% |
| graph_palette_v6_c6_p70__default | 28.1% | 46.9% | 46.9% | 46.9% |

The frozen rule selected 0 uncensored candidates exceeding both controls at N=32, up to a cap of two. Every selected candidate received 32 new episodes of 16 tests and the seven-point curve; every attempted confirmation remains in the primary table and correction.

![Online screen learning curve](../results/task_search/palette_online_screen/online_learning_curve.png)

## Valid teacher strategy control

On the same 192 graphs from the six settings with at least as many colors as vertices, replacing the standard valid teacher answers with a canonical distinct-color assignment produced 76.0%, 88.5%, and 92.2% accuracy at N=2,8,32. At N=32 the standard teacher scored 60.4%, independent own answers 56.3%, and initial-only 59.4%. All 576 new evaluation prompts have exactly the same token counts and byte counts as their standard-teacher counterparts; all 768 constructed teacher examples are valid.

This diagnostic demonstrates sensitivity to the teacher solution representation. It does not demonstrate an advantage of unjudged generation. The exact distinct-color assignment appears in 71.9% of final control answers at N=32, versus 4.2% with own examples and 1.0% with the standard teacher. All graphs use IDs 0 through 5, so the assignment is a universally valid constant answer; transfer to new vertex labels or sizes has not been tested.

![Valid teacher strategy learning curve](../results/task_search/palette_injective_control/strategy_teacher_learning_curve.png)

## Arithmetic decomposition strategy test

Two additional task operations ask for a JSON array of two or three integers with a specified sum or product. Sum parts may be zero; product factors must be positive and may be one. The task statement gives no solution strategy. A sum teacher samples uniformly from all ordered weak compositions of a random target. A product teacher samples independent uniform positive factors and presents their product as the target. The sampler does not exclude zero parts or factors equal to one. All mathematically valid decompositions receive full credit under exact arithmetic.

The twelve baseline configurations vary operation, arity, and digit scale. The scale describes target digits for sums and teacher-factor digits for products. Four baseline queries select one magnitude per operation/arity, retaining all four settings even at floor or ceiling. Sums scored 100% in this small baseline calibration; two-factor products had useful headroom, while three-factor products were at floor. All source conditions still receive identical initial examples, assigned questions, final prompts and decoder settings. Own answers are independent, unjudged generations with no strategy hint or test access.

| Arithmetic setting | Initial only | Teacher, N=32 | Own, N=32 |
|---|---:|---:|---:|
| product_decomposition_k2_d12__default | 21.9% | 9.4% | 31.2% |
| product_decomposition_k3_d3__default | 3.1% | 0.0% | 12.5% |
| sum_decomposition_k2_d3__default | 100.0% | 100.0% | 100.0% |
| sum_decomposition_k3_d3__default | 100.0% | 100.0% | 100.0% |

The screen uses four fresh episodes of eight tests per setting and N=0,2,8,32. Its frozen rule selects 2 uncensored settings exceeding both controls, capped at two, for fresh 32-episode confirmations of 16 tests each. Every attempted confirmation is included in the primary table and correction, regardless of result. Identity-answer frequencies are measured only after full inference. These are two operations with several settings, not twelve independent families.

In the selecting screen, only 10/128 own two-factor answers and 3/128 own three-factor answers were correct. The positive point estimates therefore do not rely on highly accurate rehearsal labels. Most output strings are well-formed integer arrays. The post-hoc factor-size audit measures use of a factor at most 10, including incorrect answers: it is a description of solution representation, not a correctness filter or a causal intervention. In the two-factor screen, identity-answer frequency itself did not increase relative to initial-only; the broader shift toward small factors is a more plausible hypothesis to examine.

The teacher here supplies random valid decompositions, not a pedagogically optimized policy. The task permits identity solutions, so this experiment concerns selecting an easy valid solution rather than acquiring a general-purpose factoring algorithm. The problem family identifier includes arity and magnitude, and the initial examples demonstrate the teacher distribution. These limits matter when interpreting a generation advantage.

The fresh three-factor confirmation gives the following final-answer breakdown. All identity answers counted here are arithmetically valid; they are a subset of total accuracy. The audit occurs after inference and supplies no feedback.

| Context | Correct answers | Valid identity answers |
|---|---:|---:|
| Initial only | 43/512 (8.4%) | 24/512 (4.7%) |
| 32 teacher examples | 16/512 (3.1%) | 10/512 (2.0%) |
| 32 own examples | 142/512 (27.7%) | 114/512 (22.3%) |

Of 1024 own rehearsal answers, 65 are correct and 35 are valid identity decompositions. All teacher rehearsal answers are correct; none is an identity decomposition. No three-factor rehearsal answer was truncated.

The two-factor confirmation exceeds teacher at N=32, but its advantage over initial-only has a confidence interval crossing zero. Its own curve peaks at N=8 and then declines. It fails the prespecified joint confirmation criterion; selecting the peak afterwards would change the test.

![Decomposition learning curve](../results/task_search/final/plots/product_decomposition_k2_d12_self_fixed.png)

## Finite-window slopes for all longer confirmations

The common descriptive window is N=2,4,8,16,32, fitting log(error)=intercept−beta log(N) without a pseudocount or error floor. This window was fixed before completion of the new longer confirmations. Intervals resample whole episodes and preserve both sources. These descriptive fits are not asymptotic laws or additional multiplicity-corrected discovery tests. A positive slope difference alone does not establish an absolute accuracy benefit.

| Task | Teacher beta | Own beta | Own − teacher beta [95% CI] |
|---|---:|---:|---|
| Number format | 0.166 | 0.034 | -0.131 [-0.229, -0.039] |
| Color allocation | 0.079 | -0.010 | -0.089 [-0.159, -0.018] |
| Product of 2 integers | -0.009 | 0.020 | +0.028 [-0.026, +0.080] |
| Product of 3 integers | -0.009 | 0.070 | +0.079 [+0.043, +0.125] |

## Rehearsal answer quality measured afterwards

These diagnostics are computed after full inference and never determine inclusion, repair, feedback, or reweighting. Every assigned rehearsal slot remains present. Differences in answer quality are descriptive and do not identify the mechanism of a source effect.

| Fresh confirmation | Generated answers | Correct fraction | Truncated answers |
|---|---:|---:|---:|
| number_format__default | 1024 | 84.4% | 0 |
| count_primes__0 | 256 | 7.0% | 0 |
| graph_color__1 | 256 | 56.6% | 0 |
| maze__default | 256 | 14.8% | 0 |
| palindrome_partitioning__1 | 256 | 34.8% | 39 |
| path_star__default | 256 | 32.8% | 3 |
| vertex_colors_v6_c6_p40__default | 1024 | 60.7% | 0 |
| product_decomposition_k2_d12__default | 1024 | 12.3% | 1 |
| product_decomposition_k3_d3__default | 1024 | 6.3% | 0 |

## Scoring, controls, and limits

The independent audit recomputed selected-task teacher solutions using prime enumeration, bit counts, BFS, palindrome enumeration, truth-count enumeration, combinatorial probability, graph constraints, and the cube benchmark convention. Coin-flip native prefix credit can incorrectly equal one; confirmation scoring requires absolute probability error at most 1e-4 against the exact fraction, as specified before fresh inference. Other primary metrics retain the native full-credit criterion. Graph colorings and palindrome partitions also undergo stricter output-constraint sensitivity checks, kept separately in the final JSON. All native outputs and scores are retained.

Constraint-audit score changes across screen and confirmations: {}. Generated sample statuses across the reported source phases: {'ok': 12828, 'invalid_output': 75, 'truncated': 89}. Final response statuses: {'ok': 62702, 'truncated': 994}. Generated questions exactly matching a held-out question: 23; within-pool duplicate generated questions: 481. None was removed or repaired. Full raw outputs, including truncations and invalid JSON, are retained; incomplete samples occupy explicit empty-answer slots.

The cube task does not explicitly specify the rotation axis when moving bottom to top; its grader assumes front/back swap with left/right retained. Its result therefore concerns that benchmark convention. These are answer-only, non-thinking, frozen-weight experiments with one checkpoint. Floor settings and the small exploratory screen limit sensitivity to small effects. New episode seeds do not imply globally unique question strings in finite task spaces. All 512 palindrome-confirmation answers have a valid compact serialization of at most 430 tokens, below the shared 1,024-token allowance.

These procedural episodes generally sample a known task family rather than introduce a new episode-specific hidden rule. That limits what they test about latent-model learning. The model retains no state beyond the supplied context: replaying the same demonstration text produces the same experimental condition. Any source effect here concerns the generated content and its conditioning, not an independent memory of authorship.

Uncertainty clusters by episode; intervals are pointwise, while joint p-values account for testing both required contrasts and all 9 primary confirmation attempts. The machine-readable table also includes separate per-contrast corrections and Holm correction across all 15 primary and secondary hypotheses. The sign-flip tests use within-episode source exchangeability under the null. Per-task learning curves remain in the final plots directory. Three sample counts in most original confirmations are insufficient to identify a power law.

Exact request reconstruction verifies identical final instructions and decoding, the intended independent or sequential generation contexts, fixed-question matching, unchanged generated prefixes, source hashes, cached payload hashes, and raw answer retention. Ten fresh prefix-cache-salt replays produced ten exact full-response matches. Seven offline control tests cover sample retention, held-out isolation, alternative valid solutions, malformed scoring, and the probability-prefix defect. A separate offline check verifies that sequential rehearsal sees only previous own answers.

## Complete exploratory screen

| Task | Initial only | Teacher, N=8 | Own answers, N=8 | Own complete examples, N=8 |
|---|---:|---:|---:|---:|
| ab__default | 0.0% | 6.2% | 0.0% | 6.2% |
| acre__default | 62.5% | 56.2% | 62.5% | 62.5% |
| advanced_geometry__default | 0.0% | 0.0% | 0.0% | 0.0% |
| aiw__1 | 81.2% | 87.5% | 87.5% | 75.0% |
| arc_1d__1 | 6.2% | 0.0% | 6.2% | 0.0% |
| base_conversion__default | 0.0% | 0.0% | 0.0% | 0.0% |
| basic_arithmetic__default | 56.2% | 56.2% | 56.2% | 56.2% |
| bf__3 | 0.0% | 0.0% | 0.0% | 0.0% |
| binary_alternation__default | 43.8% | 31.2% | 31.2% | 25.0% |
| binary_matrix__default | 0.0% | 0.0% | 0.0% | 0.0% |
| bitwise_arithmetic__0 | 37.5% | 31.2% | 31.2% | 31.2% |
| caesar_cipher__0 | 0.0% | 0.0% | 0.0% | 0.0% |
| calendar_arithmetic__1 | 50.0% | 56.2% | 43.8% | 43.8% |
| chain_sum__3 | 50.0% | 56.2% | 56.2% | 56.2% |
| circuit_logic__default | 68.8% | 68.8% | 56.2% | 56.2% |
| coin_flip__default | 31.2% | 37.5% | 31.2% | 43.8% |
| color_cube_rotation__0 | 37.5% | 37.5% | 31.2% | 43.8% |
| complex_arithmetic__default | 87.5% | 93.8% | 93.8% | 87.5% |
| count_bits__0 | 18.8% | 25.0% | 25.0% | 37.5% |
| count_primes__0 | 6.2% | 6.2% | 12.5% | 6.2% |
| countdown__0 | 25.0% | 31.2% | 18.8% | 25.0% |
| course_schedule__1 | 43.8% | 37.5% | 43.8% | 43.8% |
| cryptarithm__default | 0.0% | 0.0% | 0.0% | 0.0% |
| decimal_arithmetic__1 | 12.5% | 12.5% | 12.5% | 12.5% |
| decimal_chain_sum__default | 31.2% | 37.5% | 37.5% | 37.5% |
| dice__default | 0.0% | 0.0% | 0.0% | 0.0% |
| emoji_mystery__0 | 12.5% | 6.2% | 6.2% | 6.2% |
| family_relationships__3 | 87.5% | 87.5% | 81.2% | 87.5% |
| figlet_font__0 | 0.0% | 0.0% | 0.0% | 0.0% |
| fraction_simplification__0 | 93.8% | 87.5% | 87.5% | 87.5% |
| futoshiki__0 | 0.0% | 6.2% | 0.0% | 0.0% |
| game_of_life__default | 18.8% | 25.0% | 18.8% | 12.5% |
| game_of_life_halting__0 | 100.0% | 100.0% | 100.0% | 100.0% |
| gcd__1 | 87.5% | 93.8% | 93.8% | 93.8% |
| graph_color__1 | 62.5% | 75.0% | 87.5% | 81.2% |
| group_anagrams__0 | 93.8% | 68.8% | 81.2% | 93.8% |
| gsm_symbolic__default | 31.2% | 31.2% | 25.0% | 31.2% |
| intermediate_integration__3 | 37.5% | 50.0% | 25.0% | 31.2% |
| isomorphic_strings__default | 62.5% | 62.5% | 56.2% | 62.5% |
| jugs__0 | 0.0% | 6.2% | 0.0% | 0.0% |
| kakurasu__default | 0.0% | 0.0% | 0.0% | 0.0% |
| knight_swap__default | 0.0% | 0.0% | 0.0% | 0.0% |
| knights_knaves__default | 62.5% | 81.2% | 50.0% | 56.2% |
| largest_island__default | 50.0% | 43.8% | 43.8% | 37.5% |
| lcm__0 | 25.0% | 31.2% | 31.2% | 31.2% |
| leg_counting__0 | 37.5% | 56.2% | 56.2% | 37.5% |
| letter_counting__3 | 18.8% | 31.2% | 31.2% | 31.2% |
| letter_jumble__default | 12.5% | 12.5% | 12.5% | 6.2% |
| list_functions__default | 50.0% | 50.0% | 37.5% | 31.2% |
| mahjong_puzzle__default | 37.5% | 25.0% | 37.5% | 37.5% |
| manipulate_matrix__default | 12.5% | 12.5% | 12.5% | 12.5% |
| maze__default | 37.5% | 37.5% | 50.0% | 31.2% |
| mini_sudoku__default | 25.0% | 18.8% | 25.0% | 6.2% |
| n_queens__0 | 6.2% | 43.8% | 12.5% | 12.5% |
| needle_haystack__default | 100.0% | 100.0% | 100.0% | 100.0% |
| number_filtering__0 | 43.8% | 62.5% | 62.5% | 43.8% |
| number_format__default | 81.2% | 81.2% | 93.8% | 81.2% |
| number_sequence__0 | 43.8% | 56.2% | 56.2% | 43.8% |
| number_sorting__default | 100.0% | 100.0% | 93.8% | 100.0% |
| palindrome_generation__default | 18.8% | 12.5% | 12.5% | 12.5% |
| palindrome_partitioning__1 | 31.2% | 31.2% | 37.5% | 31.2% |
| path_star__default | 31.2% | 37.5% | 43.8% | 43.8% |
| polynomial_equations__0 | 93.8% | 93.8% | 93.8% | 87.5% |
| pool_matrix__default | 37.5% | 37.5% | 37.5% | 37.5% |
| power_function__0 | 37.5% | 56.2% | 43.8% | 31.2% |
| prime_factorization__3 | 68.8% | 68.8% | 68.8% | 62.5% |
| products__default | 87.5% | 81.2% | 81.2% | 81.2% |
| propositional_logic__1 | 100.0% | 100.0% | 100.0% | 87.5% |
| puzzle24__1 | 31.2% | 43.8% | 12.5% | 25.0% |
| quantum_lock__0 | 6.2% | 18.8% | 6.2% | 6.2% |
| ransom_note__0 | 62.5% | 62.5% | 62.5% | 56.2% |
| rectangle_count__default | 37.5% | 43.8% | 37.5% | 31.2% |
| rotate_matrix__default | 18.8% | 25.0% | 25.0% | 25.0% |
| rotten_oranges__0 | 25.0% | 25.0% | 18.8% | 0.0% |
| rubiks_cube__default | 0.0% | 0.0% | 0.0% | 0.0% |
| self_reference__0 | 18.8% | 12.5% | 18.8% | 37.5% |
| sentence_reordering__default | 18.8% | 18.8% | 12.5% | 12.5% |
| shortest_path__1 | 43.8% | 25.0% | 31.2% | 31.2% |
| simple_equations__0 | 43.8% | 50.0% | 50.0% | 50.0% |
| simple_geometry__default | 12.5% | 12.5% | 12.5% | 6.2% |
| simple_integration__1 | 37.5% | 37.5% | 31.2% | 37.5% |
| sokoban__1 | 6.2% | 6.2% | 0.0% | 6.2% |
| spell_backward__1 | 68.8% | 75.0% | 75.0% | 75.0% |
| spiral_matrix__default | 12.5% | 18.8% | 18.8% | 18.8% |
| string_insertion__0 | 81.2% | 81.2% | 81.2% | 81.2% |
| string_manipulation__default | 18.8% | 25.0% | 6.2% | 6.2% |
| string_splitting__default | 0.0% | 0.0% | 0.0% | 0.0% |
| string_synthesis__default | 6.2% | 12.5% | 0.0% | 6.2% |
| sudoku__default | 0.0% | 0.0% | 0.0% | 0.0% |
| survo__default | 6.2% | 12.5% | 12.5% | 12.5% |
| syllogism__default | 87.5% | 75.0% | 75.0% | 81.2% |
| time_intervals__3 | 43.8% | 50.0% | 50.0% | 43.8% |
| tower_of_hanoi__1 | 25.0% | 18.8% | 18.8% | 25.0% |
| tsumego__0 | 12.5% | 18.8% | 12.5% | 12.5% |
| word_ladder__0 | 6.2% | 6.2% | 6.2% | 6.2% |
| word_sequence_reversal__0 | 62.5% | 43.8% | 43.8% | 50.0% |
| word_sorting__default | 56.2% | 31.2% | 43.8% | 43.8% |
| zebra_puzzles__1 | 68.8% | 56.2% | 56.2% | 68.8% |

## Reproduction and evidence

The prospective design and amendments are in TASK_SEARCH_PROTOCOL.md. The pinned checkpoint revision is c202236235762e1c871ad0ccb60c8ee5ba337b9a; dependencies and the private runtime setup are in task_search_runtime/README.md and its uv.lock. Re-run finish_task_search.py, finish_frame_confirmation_task_search.py, finish_online_confirmation_task_search.py, and finish_decomposition_task_search.py to resume their cached cohorts. The finalizer audits completed phases and regenerates this report. Raw evidence lives in results/task_search on local NVMe, with a persistent CEPH archive recorded in archive.json. No model weights are included in the evidence archive.
