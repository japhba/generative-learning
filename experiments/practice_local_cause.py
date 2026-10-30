"""Gradual local coactivity learning in the sampled drawing-cause model.

The existing probability-count maximization is replaced by one small logit
update. Every other part of the unmasked generation/recognition loop is shared.
Existing worlds are a retrospective mechanism test, not fresh confirmation.
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
import practice_drawing_unmasked as u
from practice_drawing_fixed_posterior import complete_q


def probabilities(logits):
    raw = np.exp(logits - logits.max(axis=-1, keepdims=True))
    raw /= raw.sum(axis=-1, keepdims=True)
    return .98 * raw + .02 / logits.shape[-1]


def logits_from_probability(p): return np.log((p - .02 / p.shape[-1]) / .98)


def local_gradient(logits, counts):
    p = probabilities(logits)
    raw = (p - .02 / p.shape[-1]) / .98
    learned_counts = counts * (1 - .02 / (p.shape[-1] * p))
    return learned_counts - raw * learned_counts.sum(axis=-1, keepdims=True)


def audit():
    rng = np.random.default_rng(20104001)
    errors = []
    for shape in [(3, 3), (3, 3, 4)]:
        logits, counts, direction = [rng.normal(size=shape), rng.random(shape), rng.normal(size=shape)]
        direction /= np.linalg.norm(direction)
        eps = 1e-5
        numerical = np.sum(counts * (np.log(probabilities(logits + eps * direction)) -
                                     np.log(probabilities(logits - eps * direction)))) / (2 * eps)
        analytic = np.sum(local_gradient(logits, counts) * direction)
        errors.append(abs(numerical - analytic))
    assert max(errors) < 1e-8
    return dict(local_gradient_errors=errors)


def run(job):
    source, output, index, steps, rate, conditions = job
    target = output / f"run_{index}"
    target.mkdir()
    data = np.load(source / f"run_{index}/world.npz")
    saved = np.load(source / f"run_{index}/initial.npz")
    initial = dict(saved)
    initial["caches"] = list(np.split(saved["caches"], np.cumsum(saved["cache_sizes"])[:-1]))
    cat = m.Catalogue()
    observed = np.searchsorted(cat.images, data["observed"])
    seed = int(data["seed"])
    probe = m.draw(data["teacher_a"], data["teacher_b"], np.random.default_rng(seed + 340000), 512)
    m.recognize = u.recognize

    def evaluate(a, b, geometry, recognition, rng):
        _, p = cat.probabilities(a, b)
        lp = u.recognize(geometry, recognition, cat.canvas[probe], rng, cat.actions[probe])[2]
        valid = u.recognize(geometry, recognition, cat.canvas[probe], rng)[1].mean()
        return (-np.log(p[observed]).mean(), np.sum(data["teacher_p"] * np.log(data["teacher_p"] / p)),
                -lp.mean(), valid)

    summaries = {}
    for condition in conditions:
        a_logits, b_logits = logits_from_probability(data["initial_a"]), logits_from_probability(data["initial_b"])
        update_checks = []

        def gradual_step(acount, bcount):
            nonlocal a_logits, b_logits
            old_a, old_b = probabilities(a_logits), probabilities(b_logits)
            a_logits += rate * local_gradient(a_logits, acount)
            b_logits += rate * local_gradient(b_logits, bcount)
            a, b = probabilities(a_logits), probabilities(b_logits)
            gain = complete_q(a, b, acount, bcount) - complete_q(old_a, old_b, acount, bcount)
            assert gain >= -1e-10, "Local generator step decreased the conditional objective"
            update_checks.append(gain)
            return a, b

        s.mstep = gradual_step
        result = s.train(data["observed"], data["initial_a"], data["initial_b"], initial, seed,
                         condition, evaluate, steps, cache_dreams=False)
        assert np.all(m.render(s.decode(result["caches"])) ==
                      np.repeat(initial["targets"], result["cache_sizes"]))
        np.savez_compressed(target / f"{condition}.npz", **result, conditional_gains=update_checks)
        summaries[condition] = dict(final=result["records"][-1].tolist(), status=result["status"],
                                    minimum_conditional_gain=float(min(update_checks)))
        (target / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--steps", type=int, default=1600)
    parser.add_argument("--rate", type=float, default=.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conditions", nargs="+", default=["sleep", "fixed_cache", "fixed_posterior"],
                        choices=["sleep", "fixed_cache", "fixed_posterior", "frozen_proposal"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for filename in ["practice_local_cause.py", "practice_drawing_unmasked.py", "practice_drawing_sampled.py",
                     "practice_drawing_library.py", "practice_drawing_fixed_posterior.py",
                     "practice_recursive_grammar_screen.py"]:
        path = Path(__file__).parent / filename
        shutil.copy2(path, args.output / filename)
        sources[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(),
                    arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    purpose="Retrospective test of gradual local generator updates on the existing 16 worlds.",
                    changes="One softmax residual/coactivity update per cycle replaces exact conditional count maximization.",
                    stopping="Inherited observed bound and generated-recognition log-log stationarity; no population metric.",
                    audits=audit(), source_sha256=sources)
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    results = {}
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(run, (args.source, args.output, i, args.steps, args.rate, args.conditions)): i
                for i in range(args.worlds)}
        for future in tqdm(as_completed(jobs), total=args.worlds, desc="Gradual local learning"):
            index = jobs[future]
            results[str(index)] = future.result()
            (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
            print(json.dumps(dict(world=index, results=results[str(index)])), flush=True)


if __name__ == "__main__": main()
