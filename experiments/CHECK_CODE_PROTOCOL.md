# Learning to sample from local parity checks

28 November 2026, 12:15 GMT. Design recorded before simulation. The question is whether overlapping leaf constraints make full-coverage generation deeper than local recognition, and whether the resulting recognition lead gives a bounded learner an informative update on an own string.

Use the tensor product of two independently relabeled binary [7,4,3] Hamming codes. Four designated information coordinates remain an information set on each axis. Watching sees IID words, with the 33 redundant coordinates erased together with probability 1/4. The root is the bottom-right clean bit. It can equivalently be drawn first and the fair message drawn conditional on its root parity. All continuations have exactly the same mask law. A shallow reader learns the four-site parity dictionary on each axis from fully visible positive strings. Activate the dictionary after eight complete words. Before activation collect only the shallow evidence. Thereafter apply the same innovation update to every complete observation, including during Watching. Record false checks and all failures.

The deep writer initially generates independent redundancy bits conditional on the cue. A separately initialized linear reader proposes a completion of the cue. Contrast one observed word with that proposal. If they obey all previously installed checks and disagree on a learned check, append one correction along their visible difference to both circuits. Choice among tied checks is randomized by a fixed private ordering. Count every presented string. No written noise bits or teacher latents enter the update. Score exact visible forward KL, full-word validity, teacher-root accuracy on erased trials, and the fraction of all erased bits read correctly. Log old-check acceptance, useful directions, and full validity of source words.

Use 16 common Watching strings and 0 through 100 continuation strings. The 64-string continuation is the primary horizon. Fork Own, fresh teacher, independent twin, and a foreign source drawn uniformly from a random supercode with exactly the reference Own source's rank at each step. Compare a deterministic compiler of the watched checks, a passive solver of the observed input–output equations, and a writer that receives only whole-word acceptance. Also check 50 Watching plus 50 follow-up and 100 teacher. The exceptional case of fewer than eight complete words at a fork continues Watching until calibration using only externally observed teacher data during Watching or in fresh teacher continuations. Offline arms with insufficient evidence retain the last uncalibrated state. No primary worlds are discarded.

Run a small implementation check, then 256 paired worlds starting at seed 281126. Retain per-world curves and uncertainty over worlds. Check rank, conservation of previously learned constraints, the nullspace dimension and the analytic probabilities independently. Distinguish inference on a masked teacher observation from reading an already visible root. A positive comparison to the same bounded update and a loss to a passive algebraic learner are compatible outcomes.

12:20 GMT, before the finite-network experiment. Independently test neural learnability of opposite directions on precisely the same unlabeled pairs of views. In a 16-bit chain the innovation is the parity of two neighboring states and the last state is the parity of all innovations. Fit the last coordinate in each direction with the same one-hidden-layer ReLU MLP, width 64, mean squared error, Adam at .003, batches of 64 fresh pairs for 512 updates. Tie initialization across directions and use independent paired worlds (32 seeds starting at 281127). Log mean squared error and sign accuracy on a held-out panel throughout. Compare the finite networks with the exact population squared-loss rates of the isotropic two-layer ReLU tangent kernel with a constant bias coordinate. The chain is a conditional-direction test, since its full joint has an easy sampler.

12:23 GMT, exploratory topology comparison after the primary results. Keep the same visible grid, mask frequency, base code, and update rules, but remove the column checks and draw the seven encoded rows independently. The information set expands to the first four entries of every row. Run 256 new worlds starting at 281383, both with and without column checks. This manipulates code rate as well as constraint overlap. Evaluate the rank prediction and include an analytic counterexample with independent copies of a fixed block to distinguish rate effects from a need for growing depth.

12:26 GMT, after seeing the foreign-source deficit. Reuse precisely the equal-KL foreign words in a second recipient which first projects each word through its current correction layers. This spends an additional pass through the recipient and changes the distribution seen by its update. The same operation is the identity on Own and teacher words. Use the original paired worlds to measure how much of the deficit comes from throwing away an incompatible foreign word. Also derive the distribution obtained by adding projected private noise to a fresh teacher completion.

## Reproduction

Use the existing project interpreter with single-threaded BLAS. Each run uses only CPU. From the repository root, with `py=/var/tmp/jbauer/venvs/generative-learning/bin/python` and `stage=/var/tmp/jbauer/check_code_practice`, run

```bash
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
"$py" experiments/check_code_practice.py audit
"$py" experiments/check_code_practice.py run --out "$stage/main.npz"
"$py" experiments/check_code_practice.py run --watch 50 --steps 50 --out "$stage/budget.npz"
"$py" experiments/check_code_relu.py --out "$stage/relu.npz"
"$py" experiments/check_code_practice.py run --seed 281383 --out "$stage/overlap.npz"
"$py" experiments/check_code_practice.py run --seed 281383 --independent-lines --out "$stage/independent.npz"
"$py" experiments/check_code_practice.py run --reproject --out "$stage/foreign.npz"
"$py" experiments/check_code_practice.py plot --main "$stage/main.npz" --relu "$stage/relu.npz" --out tex/figures/check_code_practice.png
```

The six named arrays are bundled in `/nfs/gatsbystor/jbauer/generative-learning/results/check_code_practice/study.zip`. Practice curves have axes world, continuation count (zero is the fork), arm, metric. Event indicators have axes world, attempted continuation, the first five source arms, event. The independent-network archive has axes seed, checkpoint, direction (read, write), score (MSE, accuracy). The supplement reproduces the original seven arm trajectories bit for bit and appends the reprojected-foreign recipient.
