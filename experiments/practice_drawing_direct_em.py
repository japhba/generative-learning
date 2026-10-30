"""Original-data likelihood fitting without dream generation or neural inference."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
from practice_drawing_fixed_posterior import counts, mstep


def run(job):
    source, output, index = job
    world = np.load(source/f"run_{index}/world.npz")
    cat = m.Catalogue()
    observed = np.searchsorted(cat.images, world["observed"])
    a, b = world["initial_a"].copy(), world["initial_b"].copy()
    records, parameters = [], []
    start = time.perf_counter()
    status = "update limit"
    for step in range(401):
        acount, bcount, p = counts(cat, a, b, observed)
        nll = -np.log(p[observed]).mean()
        risk = np.sum(world["teacher_p"]*np.log(world["teacher_p"]/p))
        records.append([step, nll, risk, time.perf_counter()-start])
        parameters.append(np.r_[a.ravel(), b.ravel()])
        assert np.isfinite(records[-1]).all()
        if step > 0: assert records[-1][1] <= records[-2][1]+1e-10
        if step >= 40:
            recent = np.array(records[-20:])
            slope = np.polyfit(np.log(recent[:, 0]), np.log(recent[:, 1]), 1)[0]
            if abs(slope) < .003:
                status = "stationary original-data likelihood"
                break
        if step == 400: break
        a, b = mstep(acount, bcount)
    np.savez_compressed(output/f"world_{index}.npz", records=np.array(records), weights=np.array(parameters),
                        a=a, b=b, status=status)
    return dict(seed=int(world["seed"]), final=records[-1], status=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/Path(__file__).name)
    n = json.loads((args.source/"protocol.json").read_text())["arguments"]["worlds"]
    protocol = dict(uk_time=datetime.now(ZoneInfo("Europe/London")).isoformat(), source=str(args.source),
                    purpose="Retrospective stronger shortcut: enumerate programs and alternate exact posterior counts with categorical M steps.",
                    assumptions="Same model, data, initial generator and probability floor. No neural recognizer, dreams, cache, or likelihood line search.",
                    stopping="Observed-data log-NLL/log-step slope below .003 over 20 consecutive updates after 40; maximum 400.",
                    interpretation="A deterministic original-data likelihood fit, not an unconditional impossibility theorem for sampling.")
    (args.output/"protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    results = {}
    with ProcessPoolExecutor(args.workers) as executor:
        jobs = {executor.submit(run, (args.source, args.output, i)): i for i in range(n)}
        for future in tqdm(as_completed(jobs), total=n, desc="Original-data exact EM"):
            i = jobs[future]
            results[str(i)] = future.result()
            (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
