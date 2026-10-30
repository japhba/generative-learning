# Constructive task search after the two-1 factorization control

The objective is to test additional task families under the existing source-only contrast. The local Qwen3.5-9B checkpoint, vLLM configuration, four initial teacher pairs, generation temperature 0.7, evaluation temperature zero, disabled thinking, and 1,024-token output limit are unchanged. The paired final prompts differ only in the additional answer strings. Rehearsal questions and order are identical; independent own generation sees neither tests nor previous generated answers. Every own slot remains present. No correctness filter, repair, external judge, or source-specific strategy cue is used.

`witness_plan.json` was saved before the first new inference call. Two levels of twelve families receive eight initial-only calibration queries each. One level per family is chosen using accuracy closest to one half, then prompt length and spec ID as tie breaks. Floor and ceiling families remain in the search. Every selected level receives four fresh episodes, eight tests each, and N=0,2,8,32. The fixed selecting endpoint is N=32. At most two complete candidates strictly exceeding both initial-only and teacher at that endpoint advance to sixteen fresh episodes of sixteen tests and N=0,1,2,4,8,16,32. The endpoint remains N=32, rather than a selected curve peak.

The exact teacher policies are:

| Family | Teacher construction | Accepted alternatives |
|---|---|---|
| Bézout coefficients | Extended Euclid for two coprime positive coefficients | Any integer pair satisfying ax+by=1 |
| Simultaneous congruences | Sample an integer in the permitted interval and present its two residues | Any bounded integer satisfying both congruences |
| Bounded linear allocation | Sample a bounded positive pair and form its weighted total | Any positive bounded pair meeting that total |
| Three squares | Sample three bounded positive integers and form their squared sum | Any bounded positive triple with that squared sum |
| Rational interval | Sample a positive fraction and construct strict rational bounds around it | Any bounded fraction strictly within the interval |
| Balanced satisfiability | Plant a half-true assignment and generate clauses satisfied by it | Any half-true assignment satisfying every clause |
| Perfect matching | Plant a random matching and add allowed edges | Any bijective assignment using allowed edges |
| Hamiltonian path | Plant a random directed path and add random edges | Any path visiting every vertex exactly once |
| Topological order | Plant a random order and sample forward edges | Any ordering respecting all directed edges |
| Equal-sum groups | Sample three positive compositions of the same capacity, then shuffle all items | Any assignment of every item to three equal-sum groups |
| Common supersequence | Dynamic programming gives one minimum-length supersequence | Any supersequence within that minimum length |
| Register program | Execute a sampled bounded-length program, rejecting identity start/target instances before inference | Any bounded-length program reaching the same target |

No family admits an instance-independent constant answer that satisfies all possible questions. This does not prohibit some individual questions from having particularly easy solutions. Initial, rehearsal, and test questions are disjoint within every episode. Different seeded episodes may revisit questions from a finite procedural distribution; “fresh” means newly sampled episodes, not guaranteed global novelty.

Before inference, 4,800 teacher certificates passed their exact scorers after JSON serialization, alongside 33,600 malformed-output checks. A further 25 independent hand-specified semantic checks cover valid alternative certificates and violations of direction, bounds, number usage, balancing, ordering, and program execution. These checks contain no model outputs. The checks and source hashes are saved under `results/task_search`.

Screen results are exploratory. Fresh confirmation uses a joint one-sided paired-episode sign-flip test: take the larger p-value for own-versus-teacher and own-versus-initial. Report Holm within this round and with the nine prior local primary attempts, plus a conservative 21-comparison Bonferroni sensitivity (twelve new screened families plus nine prior attempts). Bootstrap intervals are pointwise, with whole episodes as units. Finite-window error slopes on N=2,4,8,16,32 are descriptive and make no asymptotic power-law claim. No data-dependent change to cohort size or stopping is made.

The accompanying strategy-state toy is separate from the empirical test. Its parameters are illustrative and unfitted; none enter task construction, inference, filtering, selection, or scoring.

## Separately frozen zero-initial-example follow-up

Before inspecting any K=4 confirmation outcomes, a follow-up motivated by Self-ICL was recorded in witness_zero_plan.json. It retains all twelve calibrated task settings and uses K=0 in both conditions. Newly seeded episodes receive the same four-episode, eight-test, N=0,2,8,32 screen. At most one candidate positive against both controls at N=32 advances to sixteen fresh episodes, sixteen tests, and all seven N values. This tests a different initial-context setting while preserving the exact source-only contrast. All current confirmation attempts join the nine prior local primary attempts for Holm; a conservative Bonferroni sensitivity now counts 33 possible settings (24 K/task screens plus nine prior attempts). The original K=4 plan and data remain unchanged.
