"""Preregister the partial-memory amendment and prepare every unchanged fresh world."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
import practice_drawing_sampled as s
import practice_drawing_unmasked as u
import practice_local_cause_partial as partial
from practice_drawing_sampled_analysis import controls
from practice_drawing_fixed_posterior import counts, mstep, complete_q


def prepare(job):
    output, index, seed = job
    target = output / f"run_{index}"
    target.mkdir()
    m.recognize = u.recognize
    cat = m.Catalogue()
    world = s.world(cat, seed)
    np.savez_compressed(target / "world.npz", **world)
    initial = partial.initialize(world["observed"], world["initial_a"], world["initial_b"], seed)
    np.savez_compressed(target / "initial.npz", **{key: value for key, value in initial.items() if key != "caches"},
                        caches=np.concatenate(initial["caches"]), cache_sizes=np.array([len(c) for c in initial["caches"]]))
    assert np.max(abs(initial["recognition"])) == 0
    observed = np.searchsorted(cat.images, world["observed"])
    acount, bcount, p0 = counts(cat, world["initial_a"], world["initial_b"], observed)
    a, b = mstep(acount, bcount)
    _, p = cat.probabilities(a, b)
    q_gain = complete_q(a, b, acount, bcount) - complete_q(world["initial_a"], world["initial_b"], acount, bcount)
    likelihood_gain = np.log(p[observed] / p0[observed]).mean()
    assert likelihood_gain >= q_gain - 1e-10 and q_gain >= -1e-10
    np.savez_compressed(target / "full_initial_posterior_one_step.npz", a=a, b=b, acount=acount, bcount=bcount)
    one_step = dict(population_kl=float(np.sum(world["teacher_p"] * np.log(world["teacher_p"] / p))),
                    observed_nll=float(-np.log(p[observed]).mean()), conditional_gain=float(q_gain),
                    likelihood_gain=float(likelihood_gain))
    analytic = controls(cat, world)
    assert len(analytic["kl"]) == 11
    result = dict(seed=seed, distinct_drawings=len(initial["targets"]), cached_programs=sum(map(len, initial["caches"])),
                  minimum_initial_matches=int(initial["hits"].min()), bootstrap_rounds=initial["bootstrap_rounds"],
                  empty_initial_caches=len(initial["empty_targets"]),
                  initial_inverse_maximum=float(np.max(abs(initial["recognition"]))), initial_upper_ll=initial["upper_ll"],
                  initial_population_kl=float(np.sum(world["teacher_p"] * np.log(world["teacher_p"] / p0))),
                  one_step=one_step, analytic=analytic)
    (target / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20104200)
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.workers <= 4
    args.output.mkdir(parents=True, exist_ok=False)
    names = ["practice_local_cause_fresh.py", "practice_local_cause_partial.py", "practice_local_cause.py", "practice_drawing_unmasked.py",
             "practice_drawing_sampled.py", "practice_drawing_library.py", "practice_drawing_fixed_posterior.py",
             "practice_drawing_sampled_analysis.py", "practice_drawing_library_analysis.py",
             "practice_drawing_spectral.py", "practice_recursive_grammar_screen.py"]
    hashes = {}
    for name in names:
        source = Path(__file__).parent / name
        shutil.copy2(source, args.output / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = dict(
        uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(),
        status="Declared amendment before rerunning all twelve unchanged worlds; no hyperparameter tuning.",
        amendment_of="/var/tmp/jbauer/generative_practice_drawing_library/gradual_fresh_inputs/protocol.json",
        original_failure="/var/tmp/jbauer/generative_practice_drawing_library/gradual_fresh_inputs/failure.json",
        amendment_reason="The original initializer required a valid remembered program for every observation and missed one in seed 20104211. Retain empty memories with a rigorous constant lower bound, rather than dropping observations or supplying causes. Retain the failed attempt and rerun every unchanged seed.",
        seed=args.seed, seeds=list(range(args.seed, args.seed + args.worlds)), worlds=args.worlds,
        finite_data="128 original drawings per world; no teacher program or motif label enters training.",
        generator="Three motifs, three chunks of three commands; unchanged 0.02 categorical probability floor.",
        initialization="Existing ink-count generator initialization, 1,048,576 generated draws and zero neural inverse. No conditional bootstrap or coverage requirement. Preserve each empty memory and its empirical observation mass.",
        partial_bound="For an empty memory use kappa=(0.02/4)^9, since every observed in-class drawing has at least one program and every program has at least that probability. Nonempty memories use summed program probability. Empty observations supply zero parameter counts until a cause is discovered.",
        comparison_initialization="Main and fixed-cache share the same zero inverse, initial generator, and sampled memories, including empty ones.",
        training=dict(rate=.15, steps=6000, workers=args.workers, conditions=["sleep", "fixed_cache"],
                      config=s.CONFIG),
        comparisons=[
            "Full gradual loop versus fixed initial program cache with posterior and generator learning.",
            "Full gradual loop versus one exact E-step over every program at initial parameters followed by exact constrained categorical M-step.",
            "Full gradual loop versus unchanged eleven-fit menu selected by four-fold original-data cross-validation."
        ],
        primary="Paired final population KL differences for the three comparisons; negative favors practice.",
        uncertainty="20,000 paired world bootstrap replicates with seed 20104299; 95% and Bonferroni 98.3333333333333% intervals for three comparisons. No world selection.",
        secondary="Observed-data likelihood, all-time sampled likelihood certificate, inferred-program validity, motif entropy and permutation-aligned motif MSE.",
        evaluation="Teacher distributions and exhaustive catalogue are evaluation-only for the practice learner. Direct controls are deliberately allowed exhaustive inference.",
        stopping="Unchanged log-log stationarity on negative memory bound and generated-program inverse cross entropy; cap 6000. No teacher-risk stopping.",
        failure_policy="Any actual invariant or implementation failure stops loudly. Empty memories are an explicitly modeled state. Do not discard, replace, reseed, or silently omit a world.",
        source_sha256=hashes)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(prepare, (args.output, index, args.seed + index)): index for index in range(args.worlds)}
        for future in tqdm(as_completed(jobs), total=args.worlds, desc="Fresh local-practice worlds"):
            index = jobs[future]
            summaries[str(index)] = future.result()
            (args.output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(dict(world=index, result=summaries[str(index)])), flush=True)


if __name__ == "__main__": main()
