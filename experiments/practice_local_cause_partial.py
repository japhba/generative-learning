"""Gradual local practice with a rigorous likelihood bound for empty cause memories."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
import practice_drawing_sampled as s
import practice_drawing_unmasked as u
import practice_local_cause as local
from practice_drawing_fixed_posterior import complete_q

KAPPA = (.02 / 4) ** 9


def initialize(observed, a, b, seed):
    """Unconditional generated experience; uncovered observations remain present."""
    geometry = s.Geometry()
    targets, frequency = np.unique(observed, return_counts=True)
    mass = frequency / len(observed)
    rng = np.random.default_rng(seed + 320000)
    caches = [np.array([], dtype=np.int64) for _ in targets]
    recognition = np.zeros((9, 4, 129))
    hits = np.zeros(len(targets), dtype=int)
    probe = m.draw(a, b, np.random.default_rng(seed + 320001), 512)
    probe_actions = s.decode(probe)
    probe_images = m.render(probe_actions)
    history = []
    for offset in tqdm(range(0, s.CONFIG["initial_dreams"], s.CONFIG["initial_batch"]),
                       desc="Initial generated experience", leave=False):
        codes = m.draw(a, b, rng, s.CONFIG["initial_batch"])
        bitmaps = m.render(s.decode(codes))
        hits += np.array([(bitmaps == target).sum() for target in targets])
        s.merge(caches, targets, codes, bitmaps, a, b)
        if offset % (s.CONFIG["initial_batch"] * 16) == 0:
            lp = u.recognize(geometry, recognition, probe_images, rng, probe_actions)[2]
            history.append([offset + s.CONFIG["initial_batch"], -lp.mean(), sum(len(c) > 0 for c in caches)])
    upper = s.upper_binomial(hits / s.CONFIG["initial_dreams"], s.CONFIG["initial_dreams"],
                             s.CONFIG["delta"] / len(targets))
    return dict(targets=targets, mass=mass, caches=caches, recognition=recognition, hits=hits,
                upper_ll=float(mass @ np.log(upper)), upper_probabilities=upper, history=np.array(history),
                bootstrap_rounds=0, empty_targets=targets[np.array([len(c) == 0 for c in caches])], kappa=KAPPA)


def posterior(caches, mass, a, b):
    sizes = np.array([len(c) for c in caches])
    codes = np.concatenate(caches)
    probabilities = [s.program_probability(c, a, b) for c in caches]
    sums = np.array([p.sum() for p in probabilities])
    lower = np.maximum(sums, KAPPA)
    weights = np.concatenate(probabilities) * np.repeat(mass / lower, sizes)
    return codes, weights, float(mass @ np.log(lower))


def expected_counts(codes, weights, a, b):
    actions = s.decode(codes).reshape(-1, 3, 3)
    acount, bcount = np.zeros((3, 3)), np.zeros((3, 3, 4))
    for chunk in range(3):
        component = np.prod(b[np.arange(3)[None, :, None], np.arange(3)[None, None, :],
                              actions[:, None, chunk, :]], axis=-1)
        responsibility = component * a[chunk]
        responsibility /= responsibility.sum(axis=-1, keepdims=True)
        weighted = responsibility * weights[:, None]
        acount[chunk] = weighted.sum(axis=0)
        for position in range(3):
            for action in range(4):
                bcount[:, position, action] += weighted[actions[:, chunk, position] == action].sum(axis=0)
    active_mass = weights.sum()
    assert np.max(abs(acount.sum(axis=-1) - active_mass)) < 1e-10
    assert abs(bcount.sum() - 9 * active_mass) < 1e-10
    assert active_mass <= 1 + 1e-10
    return acount, bcount


def sleep_update(geometry, recognition, codes, rng):
    actions = s.decode(codes)
    gradient = u.recognize(geometry, recognition, m.render(actions), rng, actions, update=True)[3]
    recognition += s.CONFIG["rate"] * gradient


def train(world, initial, condition, evaluate, steps, rate):
    seed = int(world["seed"])
    geometry = s.Geometry()
    rng, eval_rng = np.random.default_rng(seed + 330000), np.random.default_rng(seed + 330001)
    a, b = world["initial_a"].copy(), world["initial_b"].copy()
    a_logits, b_logits = local.logits_from_probability(a), local.logits_from_probability(b)
    caches = [c.copy() for c in initial["caches"]]
    recognition = initial["recognition"].copy()
    targets, mass = initial["targets"], initial["mass"]
    _, _, bound = posterior(caches, mass, a, b)
    records, parameters, empty_records, gain_records, norm_records = [], [], [], [], []
    start = time.perf_counter()
    status = "maximum updates"
    for step in tqdm(range(steps + 1), desc=condition, leave=False):
        if step % s.CONFIG["eval_every"] == 0 or step == steps:
            nll, risk, recognition_ce, valid = evaluate(a, b, geometry, recognition, eval_rng)
            generated = m.draw(a, b, eval_rng, 512)
            actions = s.decode(generated)
            dream_ce = -u.recognize(geometry, recognition, m.render(actions), eval_rng, actions)[2].mean()
            assert -nll >= bound - 1e-9
            records.append([step, nll, risk, -bound, bound - initial["upper_ll"], recognition_ce, valid,
                            dream_ce, time.perf_counter() - start])
            parameters.append(np.r_[a.ravel(), b.ravel()])
            empty = np.array([len(c) == 0 for c in caches])
            empty_records.append([step, int(empty.sum()), float(mass[empty].sum())])
            assert np.isfinite(records[-1]).all()
            if step >= s.CONFIG["min_steps"]:
                recent = np.array(records[-s.CONFIG["window"]:])
                slopes = [np.polyfit(np.log(recent[:, 0]), np.log(recent[:, j]), 1)[0] for j in [3, 7]]
                if max(abs(value) for value in slopes) < s.CONFIG["slope"]:
                    status = "stationary variational bound and dream recognition"
                    break
        if step == steps: break
        previous_bound = bound
        if condition == "sleep":
            sleep_update(geometry, recognition, m.draw(a, b, rng, s.CONFIG["batch"]), rng)
            questions = np.repeat(targets, s.CONFIG["trials"])
            proposals, valid, _, _ = u.recognize(geometry, recognition, questions, rng)
            s.merge(caches, targets, proposals[valid], questions[valid], a, b)
        codes, weights, bound = posterior(caches, mass, a, b)
        assert bound >= previous_bound - 1e-10
        acount, bcount = expected_counts(codes, weights, a, b)
        ag, bg = local.local_gradient(a_logits, acount), local.local_gradient(b_logits, bcount)
        gradient_square = np.sum(ag ** 2) + np.sum(bg ** 2)
        a_logits += rate * ag
        b_logits += rate * bg
        next_a, next_b = local.probabilities(a_logits), local.probabilities(b_logits)
        gain = complete_q(next_a, next_b, acount, bcount) - complete_q(a, b, acount, bcount)
        assert gain >= rate * (1 - .75 * rate) * gradient_square - 1e-10
        gain_records.append(gain)
        norm_records.append(gradient_square)
        a, b = next_a, next_b
        _, _, bound = posterior(caches, mass, a, b)
        assert bound >= previous_bound + gain - 1e-9
    return dict(records=np.array(records), weights=np.array(parameters), a=a, b=b, recognition=recognition,
                status=status, caches=np.concatenate(caches), cache_sizes=np.array([len(c) for c in caches]),
                conditional_gains=np.array(gain_records), gradient_square=np.array(norm_records),
                empty_records=np.array(empty_records), kappa=KAPPA)


def run(job):
    source, output, index, steps, rate, conditions = job
    target = output / f"run_{index}"
    target.mkdir()
    world = np.load(source / f"run_{index}/world.npz")
    saved = np.load(source / f"run_{index}/initial.npz")
    initial = dict(saved)
    initial["caches"] = list(np.split(saved["caches"], np.cumsum(saved["cache_sizes"])[:-1]))
    cat = m.Catalogue()
    observed = np.searchsorted(cat.images, world["observed"])
    seed = int(world["seed"])
    probe = m.draw(world["teacher_a"], world["teacher_b"], np.random.default_rng(seed + 340000), 512)

    def evaluate(a, b, geometry, recognition, rng):
        _, p = cat.probabilities(a, b)
        lp = u.recognize(geometry, recognition, cat.canvas[probe], rng, cat.actions[probe])[2]
        valid = u.recognize(geometry, recognition, cat.canvas[probe], rng)[1].mean()
        return -np.log(p[observed]).mean(), np.sum(world["teacher_p"] * np.log(world["teacher_p"] / p)), -lp.mean(), valid

    summary = {}
    for condition in conditions:
        result = train(world, initial, condition, evaluate, steps, rate)
        assert np.all(m.render(s.decode(result["caches"])) == np.repeat(initial["targets"], result["cache_sizes"]))
        np.savez_compressed(target / f"{condition}.npz", **result)
        summary[condition] = dict(final=result["records"][-1].tolist(), status=result["status"],
                                  initial_empty=int(result["empty_records"][0, 1]),
                                  final_empty=int(result["empty_records"][-1, 1]))
        (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=12)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--rate", type=float, default=.15)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    assert args.workers <= 4 and 0 < args.rate < 4 / 3
    args.output.mkdir(parents=True, exist_ok=False)
    names = ["practice_local_cause_partial.py", "practice_local_cause.py", "practice_drawing_unmasked.py",
             "practice_drawing_sampled.py", "practice_drawing_library.py", "practice_drawing_fixed_posterior.py",
             "practice_recursive_grammar_screen.py"]
    hashes = {}
    for name in names:
        source = Path(__file__).parent / name
        shutil.copy2(source, args.output / name)
        hashes[name] = hashlib.sha256(source.read_bytes()).hexdigest()
    conditions = ["sleep", "fixed_cache"]
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    arguments["conditions"] = conditions
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(), arguments=arguments,
                    preregistration=str(args.source / "protocol.json"), source_sha256=hashes,
                    bound="Each empty cause memory contributes empirical mass times log kappa; kappa=(0.02/4)^9. Nonempty memories use their exact summed program probability.",
                    active_counts="Retain original empirical mass. Do not renormalize counts over covered images.",
                    gradient_gain="Every update asserts Q gain >= eta*(1-3*eta/4)*squared_gradient_norm. No update is accepted or rejected by this diagnostic.",
                    stopping="Unchanged observed bound and generated-program recognition stationarity; no population criterion.")
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(run, (args.source, args.output, i, args.steps, args.rate, conditions)): i for i in range(args.worlds)}
        for future in tqdm(as_completed(jobs), total=args.worlds, desc="Fresh partial-memory practice"):
            index = jobs[future]
            results[str(index)] = future.result()
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(dict(world=index, results=results[str(index)])), flush=True)


if __name__ == "__main__": main()
