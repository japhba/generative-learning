# Can an observed branching rule direct offline learning?

26 November 2026, UK time. Written before the pilot. The experimental unit is a teacher world and learner initialization. Five attempts use the same two-level sign tree. The teacher uses the same unknown signed branching motif at the two levels. A polarity reference in each bottom group makes the motif identifiable from observations. The learner receives only the visible string. The generators' root traces are reserved for classic sleep and evaluation.

1. Local wake after a deep threshold, with no reuse of the motif.
2. A deep reader that can directly use the learned shallow motif, with ordinary local wake.
3. Local coincidences verified separately at each synapse by the shallow motif.
4. One signed scalar score supplied by the learned shallow motif.
5. One root-label-invariant pair score, with smooth deep activation. Test whether a shallow estimate gives the true direction positive growth while orthogonal directions decay.

Start with finite IID Watching. Branch the identical learner into all-own Practice, fresh teacher, independent twin, a symmetry-transformed source with exactly the recipient's population KL, classic alternate wake/sleep, and sleep alone. All arms count one visible string per step including discarded/silent trials. Measure at initialization, at the fork, and throughout continuation. The main measures are exact visible population KL and inferred root accuracy on fresh teacher strings (a silent unit guesses). Also record shallow inference, deep alignment, confidence/eligibility and mode coverage. Compare the fixed horizons of 128 and 512 continuation strings, with 32 watched strings, and show complete trajectories. The primary unit for differences is paired worlds, with uncertainty across worlds. A construction counts as a joint success only if own improves both endpoints from the fork and beats fresh teacher on both at a common horizon. Audit own versus twin separately.

Use independent pilot and evaluation seed ranges. Keep every evaluation world, including stalls and negative results. Probe teacher sharing, Watching, deep initialization, and the eligibility gate. Examine source mixtures at the same total budget; report whether chosen criteria are transient. Keep per-world results on gatsbystor. The final note should distinguish algebraic properties of the teacher, idealized drift calculations, and measured dynamics of the actual learner.

00:08 GMT supplement, before independent evaluation. Development ensemble seed 11 (24 worlds). The pilot motivated a stronger foreign control: an independent copy of the Own arm supplies every sample to the matched arm through a single fixed signed permutation preserving the teacher law. Its source KL then matches the Own arm at every step. For the final run use seed 261126, 192 worlds, branching factor 10, one deep threshold of 0.55, 0.18 deep step size, sign fidelity 0.94 and independent, weak random shallow readout before Watching. Add ordinary teacher wake, ordinary alternating wake/sleep, independent random deep exploration, removal of the noiseless reference, a one-generative-layer lateral reader, destruction of the learned verdict, and gate and initialization controls. Without the reference use rigorous lower/upper bounds for visible KL. At each horizon report mean paired differences and world-level outcomes; the two horizons and all failures stay in the report. The heatmaps vary the length of the initial all-teacher block, followed by pure Practice. All learning rates and horizons are fixed across seeds and sources.

00:10 GMT follow-up after the main results. Recheck the gate-by-pair-score interaction on 192 new worlds (seed 261127) with the full set of source controls. Extend the local and scalar learners to three levels, with the same motif at every level, separately initialized middle and top synapses, and silence propagated upward. Test 96 further worlds (seed 261128) with fresh teacher, Own, and Own with the middle reader and writer frozen at the fork. Measure shallow, middle, and top teacher-cause accuracy throughout, and estimate visible KL with a fixed independent teacher panel and a marginalized likelihood. Count the entire generated tree as one string in every arm and average the update across repeated sites. This extension and the gate sensitivity are exploratory.

00:12 GMT diagnostic. On the original evaluation ensemble delay the shallow verdict by one string during Practice. This keeps the sample budget and tests whether the verdict must refer to the current trial. Report this as an exploratory ablation.

00:15 GMT diagnostic. The exact match for the local filter uses a synaptic bound equal to the teacher's sign fidelity. Repeat with a starting guess of 0.8, estimate the bound online from the within-row correlations of the watched strings, and keep that estimate at the fork. Use the same estimate in the shallow writer. This is an exploratory check on the noise prior.

00:19 GMT diagnostic. Open the gate only after the common watched checkpoint. In the flat control freeze only its lateral motif estimate at the fork, allowing the deep reader and writer to keep learning. Also evaluate the specific 50 teacher plus 50 own versus 100 teacher budget in the spec. These comparisons distinguish an architectural requirement for depth from the ability to preserve a learned statistic.

## Reproduction

Run from the repository root with its existing project environment. The main, probes, budgets, watching, lag, calibration, and challenges suites used seed 261126 and 192 vectorized worlds. The independent gate comparison used seed 261127 and 192 worlds. The stack script fixes seed 261128 and 96 worlds. Each array carries one record per world and checkpoint. The suite, method, continuation, variant, watch count, and times appear in its archive key. Metrics, seed, and world count are separate entries for each suite. The archive stores exact visible KL under `kl` for the reference model. For the noisy-reference ablation use `visible_lower` and `visible_upper`.

```bash
stage=/var/tmp/jbauer/repeated_branch
interpreter=/var/tmp/jbauer/venvs/generative-learning/bin/python
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$stage"
for suite in main probes budgets watching lag calibration challenges; do
    "$interpreter" experiments/repeated_branch.py --suite "$suite" --seed 261126 --worlds 192 --out "$stage/$suite.npz"
done
"$interpreter" experiments/repeated_branch.py --suite pair_gate --seed 261127 --worlds 192 --out "$stage/pair_gate.npz"
"$interpreter" experiments/repeated_branch_stack.py --out "$stage/stack.npz"
"$interpreter" experiments/repeated_branch_review.py --pack "$stage" --data "$stage/study.npz"
```

The maintained study archive is `/nfs/gatsbystor/jbauer/generative-learning/results/repeated_branch/study.npz`. Running the review script without arguments validates this archive and regenerates the figures.
