# Strict Hebbian learning in a feedforward sign tree

26 November 2026, before pilots. Keep the repeated-branch teacher and the exact visible KL and teacher-root accuracy measures. Test ordinary deep coactivity against a structured feedforward reader. A private unit proposes a root from a visible string. A shallow motif learned within the string can be reused to bias branch units. A second root unit reads these branch activities. All increments to a plastic weight must be a constant times its actual two endpoint activities, summed over uses of a shared weight. No weight clipping, decay, score multiplier, backpropagation, or root trace in Own. The decoder can be clamped to the independently read states during wake, and this convention must be disclosed.

Compare fresh teacher, Own, a twin, a foreign source with exactly the Own stream's KL, and ordinary alternating wake/sleep. Use 32 watched strings and examine +128 and +512 trials. Use independent pilot and evaluation ensembles. Require improvements from the fork in both generation and held-out cause recognition. Test whether the effect depends on verifying the current sample, reusing the correct motif, and private reader/writer alignment. Test an ordinary dense hidden layer as a distinct comparator. Report failed designs and which wiring had to be specified. All worlds, curves, and uncertainty use the same sample accounting as the preceding study.

Before confirmation, pilot seed 73, 32 worlds. The simplest positive candidate used an equal current from the sensed sign and a top-down prediction in a branch cell. With a saturated prediction the currents cancel on disagreement. Adding an extra sensory bypass weakened generation. Confirm on seed 261129, 192 paired worlds, with both original horizons. Include ordinary dense, a learned mixing strength, a dense hidden layer with known branch correspondence and matched private starting pattern, and an exact transpose-tied version that learns only during forward recognition. Compare removing deep sampling noise and letting the decoder learn on its own free forward pass. Vary sensory gain, reader/writer coupling, availability of the learned motif, and intrinsic channel noise. The depth extension will use 96 independent worlds, seed 261130, with a frozen middle-level control. These architectures and controls are fixed before inspecting the confirmation ensemble.

Separate replication after viewing the first ensemble. Own beat classic wake/sleep early for the additive model and the dense model with matched starting code, while classic narrowed or reversed the difference later. Test +32 as a fixed follow-up horizon on a new seed (261131, 256 worlds), keep +128 and +512, and report all three models (additive, vanilla dense, structured dense with matched start). Compare separate and transpose-tied weights. Tied classic wake applies the clamped decoder increment to the transposed reader parameters, sleep trains on writing states. The main tied learner continues to learn from forward recognition alone.

Depth diagnostic on seed 261129. Collapse the tree to directly observed branch signs. A single lateral unit learns the motif from the same string using a local Hebbian pair. Compare retaining plasticity and holding the lateral weights at the end of Watching. In the separate-weight tree also test initial read, write, and proposal vectors exactly orthogonal to the teacher. This is an exploratory counterexample search for depth necessity.

## Reproduction

The main, tied, probes, budgets, and depth suites use 192 worlds and seed 261129. The independent confirmation uses 256 worlds and seed 261131. The three-level study fixes seed 261130, 96 worlds, and 512 independent evaluation strings per world. Budget keys contain the watch count followed by the practice count. Other two-level suite keys contain the model, variant, source and initial watch count, with continuation times under the suite's marks entry. Root accuracy also appears before latent-polarity relabeling. Visible KL is exact for two levels, Monte Carlo for three levels. No example is dropped. Development files are excluded from the confirmation archive.

```bash
stage=/var/tmp/jbauer/strict_hebb_branch
interpreter=/var/tmp/jbauer/venvs/generative-learning/bin/python
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
mkdir -p "$stage"
for suite in main tied probes budgets depth; do
    "$interpreter" experiments/strict_hebb_branch.py --suite "$suite" --seed 261129 --worlds 192 --out "$stage/$suite.npz"
done
"$interpreter" experiments/strict_hebb_branch.py --suite confirm --seed 261131 --worlds 256 --out "$stage/confirm.npz"
"$interpreter" experiments/strict_hebb_stack.py --out "$stage/stack.npz"
"$interpreter" experiments/strict_hebb_review.py --pack "$stage" --data "$stage/study.npz"
```

The durable archive is `/nfs/gatsbystor/jbauer/generative-learning/results/strict_hebb_branch/study.npz`. Running the review script without arguments reads that archive, checks forward endpoint increments, mask pooling, fork equality and the ideal zero-drive teacher state, repeats two trajectories, and regenerates the figure.
