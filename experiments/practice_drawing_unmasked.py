"""Test learned drawing inference with no hand-coded action-validity mask.

Only completed programs are checked against remembered drawings. The forward
interpreter, feature coordinates, initialization and learning rules otherwise
match practice_drawing_sampled.py. No GPU or external model calls are used.
"""

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
from practice_drawing_sampled_analysis import controls


def recognize(cat, weights, target, rng, actions=None, update=False):
    n = len(target)
    x, y, direction = np.full(n, 3), np.full(n, 3), np.zeros(n, dtype=int)
    canvas = np.full(n, np.uint64(1) << np.uint64(27), dtype=np.uint64)
    output, log_probability = np.zeros((n, m.LENGTH), dtype=np.uint8), np.zeros(n)
    gradients = np.zeros_like(weights)
    for step in range(m.LENGTH):
        phi = m.features(cat, target, canvas, x, y, direction)
        logits = phi @ weights[step].T
        p = np.exp(logits-logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        if actions is None:
            proposal = .9*p+.1/4
            action = (rng.random((n, 1)) > proposal.cumsum(axis=-1)).sum(axis=-1)
        else: action = actions[:, step]
        log_probability += np.log(p[np.arange(n), action])
        if update:
            error = -p
            error[np.arange(n), action] += 1
            gradients[step] = error.T @ phi/n
        output[:, step] = action
        move = (action == 0) | (action == 3)
        x = (x+move*m.DX[direction]) % m.SIDE
        y = (y+move*m.DY[direction]) % m.SIDE
        canvas |= (action == 0).astype(np.uint64) << (y*m.SIDE+x).astype(np.uint64)
        direction = (direction+(action == 1)-(action == 2)) % 4
    return output @ m.POWERS, canvas == target, log_probability, gradients


def audit():
    rng = np.random.default_rng(20103290)
    geometry = s.Geometry()
    weights = rng.normal(0, .1, (9, 4, 129))
    actions = rng.integers(4, size=(37, 9), dtype=np.uint8)
    target = m.render(actions)
    codes, valid, lp, gradient = recognize(geometry, weights, target, rng, actions, True)
    assert valid.all() and np.array_equal(codes, actions @ m.POWERS)
    direction = rng.normal(size=weights.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-5
    upper = recognize(geometry, weights+eps*direction, target, rng, actions)[2].mean()
    lower = recognize(geometry, weights-eps*direction, target, rng, actions)[2].mean()
    analytic = np.sum(direction*gradient)
    numerical = (upper-lower)/(2*eps)
    assert abs(analytic-numerical) < 1e-8
    repeated = np.repeat(target, 100)
    sampled, accepted, _, _ = recognize(geometry, np.zeros_like(weights), repeated, rng)
    assert np.array_equal(m.render(s.decode(sampled)) == repeated, accepted)
    uniform_lp = recognize(geometry, np.zeros_like(weights), target, rng, actions)[2]
    assert np.max(abs(uniform_lp+9*np.log(4))) < 1e-12
    assert accepted.mean() < .1
    return dict(directional_gradient_error=float(abs(analytic-numerical)),
                zero_weight_valid_rate=float(accepted.mean()),
                zero_weight_program_nll=float(-uniform_lp.mean()))


def pretrain(a, b, seed, steps):
    geometry = s.Geometry()
    weights = np.zeros((9, 4, 129))
    rng = np.random.default_rng(seed+350000)
    probe = m.draw(a, b, np.random.default_rng(seed+350001), 512)
    actions = s.decode(probe)
    images = m.render(actions)
    history = []
    for step in tqdm(range(steps+1), desc="Train once on initial generator", leave=False):
        if step % s.CONFIG["eval_every"] == 0 or step == steps:
            loss = -recognize(geometry, weights, images, rng, actions)[2].mean()
            fresh = s.decode(m.draw(a, b, rng, 512))
            fresh_loss = -recognize(geometry, weights, m.render(fresh), rng, fresh)[2].mean()
            history.append([step, loss, fresh_loss])
            if step >= s.CONFIG["min_steps"]:
                recent = np.array(history[-s.CONFIG["window"]:])
                slopes = [np.polyfit(np.log(recent[:, 0]), np.log(recent[:, j]), 1)[0] for j in [1, 2]]
                if max(abs(v) for v in slopes) < s.CONFIG["slope"]: break
        if step == steps: break
        s.sleep_update(geometry, weights, m.draw(a, b, rng, s.CONFIG["batch"]), rng)
    return weights, np.array(history)


def run(job):
    root, seed, index, steps, do_controls = job
    output = root/f"run_{index}"
    output.mkdir()
    # Deliberate experimental substitution, isolated inside this worker process.
    m.recognize = recognize
    cat = m.Catalogue()
    data = s.world(cat, seed)
    np.savez_compressed(output/"world.npz", **data)
    initial = s.initialize(data["observed"], data["initial_a"], data["initial_b"], seed, False)
    np.savez_compressed(output/"initial.npz",
                        **{key: value for key, value in initial.items() if key != "caches"},
                        caches=np.concatenate(initial["caches"]),
                        cache_sizes=np.array([len(c) for c in initial["caches"]]))
    observed = np.searchsorted(cat.images, data["observed"])
    probe = m.draw(data["teacher_a"], data["teacher_b"], np.random.default_rng(seed+340000), 512)

    def evaluate(a, b, geometry, recognition, rng):
        _, p = cat.probabilities(a, b)
        lp = recognize(geometry, recognition, cat.canvas[probe], rng, cat.actions[probe])[2]
        valid = recognize(geometry, recognition, cat.canvas[probe], rng)[1].mean()
        return (-np.log(p[observed]).mean(),
                np.sum(data["teacher_p"]*np.log(data["teacher_p"]/p)), -lp.mean(), valid)

    summary = dict(seed=seed, upper_ll=initial["upper_ll"], conditions={})
    for condition in ["sleep", "wake", "prior", "frozen_proposal", "fixed_cache", "pretrained_frozen"]:
        starting = initial
        training_condition = condition
        if condition == "pretrained_frozen":
            learned, calibration = pretrain(data["initial_a"], data["initial_b"], seed, steps)
            np.savez_compressed(output/"pretraining.npz", recognition=learned, records=calibration)
            starting = initial | {"recognition": learned}
            training_condition = "frozen_proposal"
        result = s.train(data["observed"], data["initial_a"], data["initial_b"], starting,
                         seed, training_condition, evaluate, steps, cache_dreams=False)
        assert np.all(m.render(s.decode(result["caches"])) == np.repeat(initial["targets"], result["cache_sizes"]))
        np.savez_compressed(output/f"{condition}.npz", **result)
        summary["conditions"][condition] = dict(final=result["records"][-1].tolist(), status=result["status"])
        (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    if do_controls: summary["direct"] = controls(cat, data)
    (output/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--controls", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for filename in ["practice_drawing_unmasked.py", "practice_drawing_sampled.py",
                     "practice_drawing_library.py", "practice_drawing_fixed_posterior.py",
                     "practice_recursive_grammar_screen.py", "practice_drawing_sampled_analysis.py",
                     "practice_drawing_library_analysis.py", "practice_drawing_spectral.py"]:
        source = Path(__file__).parent/filename
        shutil.copy2(source, args.output/filename)
        sources[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(),
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    config=s.CONFIG, source_sha256=sources,
                    architecture="Unmasked neural inverse; only completed program equality filters proposals. No direct dream-to-cache route.",
                    primary="Population KL of sleep minus frozen proposal, fixed cache, and original-data CV analytic menu.",
                    secondary="Sleep minus cached-cause wake, initial-generator sleep, and initial-generator pretraining followed by frozen neural weights; recognition accuracy and finite-data certificate.",
                    stopping="Unchanged online stationarity rule; stopped models held fixed. Not equal computational work.",
                    guarantee="Inherited all-time certificate for likelihood versus initial model, not monotonic likelihood or population guarantee.",
                    limits="Known renderer and geometric features remain supplied; no emergent verification-generation gap is assumed.",
                    audits=audit())
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    results = {}
    with ProcessPoolExecutor(args.workers) as executor:
        jobs = {executor.submit(run, (args.output, args.seed+i, i, args.steps, args.controls)): i for i in range(args.worlds)}
        for future in tqdm(as_completed(jobs), total=args.worlds, desc="Unmasked drawing worlds"):
            i = jobs[future]
            results[str(i)] = future.result()
            (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
            print(json.dumps(dict(world=i, result=results[str(i)])), flush=True)


if __name__ == "__main__": main()
