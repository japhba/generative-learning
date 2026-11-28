# Practice identity and the tied two-level network

28 November 2026. Not pre-registered: written after the runs, in the order the decisions were taken (UK times).

## Identity (14:30-15:00)
Derived first, then checked. For a teacher uniform on S, a verdict 1_S and own samples credited with their writing state, the expected
practice update is sum_{x in S} p(x) phibar(x); watching with the exact posterior gives |S|^-1 sum_S phibar(x). Equal up to p(S) when p is flat
on S. Checks: `practice_identity.py` (exact enumeration, writers flat on S by construction, and random writers), `practice_identity_sampled.py`
(order-m world, one group of q^(m-1) states, q/m in (2,3) (2,4) (2,5) (3,3), shared random start 0.01 / 0.3 / 1 / 4, half own samples,
8 seeds, time to KL < 0.1). Prediction written before the sampled run: no practice variant beats teacher strings alone per sample.

## Tied network (15:10-16:30)
`tied_binding.py`. Two checks of three Z_3 blocks, tied (classes equal, 27 strings) or free (81). Two level-1 groups of 9 states wired to all
blocks, one level-2 group of 3 states; separate reader and writer; rule P <- P + eps (e - P), eps 0.003, start tilt 0.1, batches of 64.
Sequence of runs and what each changed:
1. pilot, 8 seeds, 6e5 samples, no schedule: free world practice 5/8, tied world nothing works.
2. accepted own samples no longer move the level-2 writer (the verdict checks level 1 only), 2e6 samples: tied world still nothing.
3. level 2 frozen until 1e6 samples (`--stage 1e6`), 16 seeds: tied practice 12/16, all teacher-driven arms <= 2/16.
4. robustness of the schedule, 16 seeds: `--slow2 0.1` (practice 6/16), `--gate2 1` and `4` (0-3/16 everywhere), `--pass_on 0.9`
   (practice 6, wake-sleep 7). Inhibition rates c 3 / 30 / 300 tried in the free world; c 3 works there.
5. main run: `--stage 1e6 --seeds 32 --samples 2e6`, all arms incl. equal-validity foreign and offline practice; shares of own samples
   (`'arm @ share'`) 16 seeds.
6. trap test `tied_binding_trap.py`: 1e6 samples of exact-posterior watching (level 2 frozen), then 1e6 more of the same, of practice, of sleep
   only, of practice with the learner's reader, 16 seeds. The drive norms it prints use level-2 state 0 instead of the mixture over level-2
   states and are not used in the note.

Solved = KL(teacher || writer) < 0.1 at 2e6 samples. KL exact by enumeration over the 729 strings.

## Reproduction
```bash
export OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg
python3 practice_identity.py --out /var/tmp/jbauer/practice_identity/exact.json
python3 practice_identity_sampled.py --out /var/tmp/jbauer/practice_identity/sampled.json
python3 practice_identity_plot.py --exact /var/tmp/jbauer/practice_identity/exact.json --sampled /var/tmp/jbauer/practice_identity/sampled.json --out ../tex/figures/practice_identity
python3 tied_binding.py --out /var/tmp/jbauer/tied_binding/main.json --samples 2e6 --stage 1e6 --seeds 32 --arms '<arms of the note, | separated>'
python3 tied_binding_trap.py --out /var/tmp/jbauer/tied_binding/trap.json
python3 tied_binding_plot.py --dir /var/tmp/jbauer/tied_binding --out ../tex/figures/tied_binding
```
All outputs: `/nfs/gatsbystor/jbauer/generative-learning/results/practice_identity/`.

## Later runs (16:40-17:30)
7. novelty-weighted verdict (`'offline practice, novelty-weighted verdict'`, `'practice, novelty-weighted verdict'`): each check unit keeps a
   running share of own triples (rate 0.02 per batch); an accepted own sample is kept with probability min(1, w / 9),
   w = prod_c teacher share(t_c) / own share(t_c). 16 seeds, stage 1e6. Motivated by the identity's on-policy term after plain offline practice
   narrowed in the main run.
8. `--explain` (level 2 learns only from strings the level-1 code rewrites exactly), no stage, 16 seeds: fails.
9. `--part G2` / `--part R2` with `--stage 1e6`: the stage freezes only the writer's or only the reader's level 2, 16 seeds, tied world.
10. `--nosleep2` (own samples do not move the reader's level 2), no stage, 16 seeds.
11. `--learned_wiring`: the verifier counts all 20 triples of blocks and treats a triple as a check once it has shown <= 9 patterns in >= 100
    teacher strings. 32 seeds, stage 1e6, verdict arms.
12. Twin fixed to practise on its own samples in parallel (before, it only watched); rerun 32 seeds (`twin_practising.json`); the figure uses
    `main_v2.json` = main.json with the twin arm replaced by this rerun.
13. Rate separation instead of a schedule (19:00): `--slow2 0.03` and `0.01`, no stage, 16 seeds, tied world; then the main configuration
    becomes `--slow2 0.01` (32 seeds, all arms, both worlds; shares 16 seeds): `slow_main.json`, `slow_shares.json`. The figure is drawn from
    these with `--main slow_main.json --shares slow_shares.json --stage 0 --probe 3.4e5`.
