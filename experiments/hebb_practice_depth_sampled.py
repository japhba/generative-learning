"""Single-sample version of hebb_practice_depth.py: one teacher sample (and, with practice, one own sample) per update, every unit sampled.

Checks that what the exact expected updates show survives sampling noise. Stages and arms as in hebb_practice_depth.py
(watch, reread, reread_shuffle, kept, kept_only, lateral); the measurements are exact (enumeration) at the logged steps.
"""
import argparse
import json
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

import hebb_practice_depth as model


def draw(rng, field):
    return np.where(rng.random(field.shape) < 1 / (1 + np.exp(-2 * field)), 1., -1.)


def read_deep(rng, r2, lat, h1):
    h2 = np.zeros(len(r2))
    for j in range(len(r2)):
        h2[j] = draw(rng, r2[j] @ h1 - lat[j, :j] @ h2[:j])
    return h2


def run(job):
    seed, stages, args = job
    t = s = model.Shape(args['n'], args['m'], args['k'], args['block'])
    rng = np.random.default_rng(seed)
    a1, a2 = t.mask * rng.normal(0, args['g1'], (t.n, t.m)), rng.normal(0, args['g2'], (t.m, t.k))
    teacher = model.teacher_tables(t, a1, a2)
    cumulative = np.cumsum(teacher['px'])
    r1, r2, g1, g2, lat = model.start(s, rng, 'blank', 0., args['init'])
    rate, history, step = args['rate'], [], 0
    for number, (name, steps) in enumerate(stages):
        arm = dict(own=None, source='own', watch=1., lateral=False) | model.ARMS[name]
        for local in range(steps + 1):
            if local % args['every'] == 0:
                history.append(dict(step=step + local, stage=number) | model.evaluate(s, arm, [r1, r2, g1, g2, lat], teacher))
            if local == steps:
                break
            x = s.X[min(np.searchsorted(cumulative, rng.random()), len(s.X) - 1)]
            h1 = draw(rng, r1 @ x)
            h2 = read_deep(rng, r2, lat, h1)
            dr1, dr2 = arm['watch'] * (np.outer(h1, x) - np.tanh(r1)), arm['watch'] * (np.outer(h2, h1) - np.tanh(r2))
            dg1, dg2 = np.outer(x - np.tanh(g1 @ h1), h1), np.outer(h1 - np.tanh(g2 @ h2), h2)
            if arm['lateral']:
                lat = lat + rate * np.tril(np.outer(h2, h2), -1)
            if arm['own']:
                top = draw(rng, np.zeros(s.k))
                mid = draw(rng, np.zeros(s.m) if arm['source'] == 'shuffle' else g2 @ top)
                own = draw(rng, g1 @ mid)
                if arm['own'] == 'reread':
                    mid = draw(rng, r1 @ own)
                    top = read_deep(rng, r2, lat, mid)
                dr1, dr2 = dr1 + np.outer(mid, own) - np.tanh(r1), dr2 + np.outer(top, mid) - np.tanh(r2)
            r1, r2, g1, g2 = r1 + rate * s.mask.T * dr1, r2 + rate * dr2, g1 + rate * s.mask * dg1, g2 + rate * dg2
        step += steps
    return dict(seed=seed, stages=stages, curves={key: [row[key] for row in history] for key in history[0]})


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--block', type=int, default=3)
    parser.add_argument('--g1', type=float, default=1.8)
    parser.add_argument('--g2', type=float, default=1.5)
    parser.add_argument('--init', type=float, default=.01)
    parser.add_argument('--rate', type=float, default=.01)
    parser.add_argument('--watch_steps', type=int, default=40000)
    parser.add_argument('--steps', type=int, default=120000)
    parser.add_argument('--every', type=int, default=2000)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--arms', nargs='+', default=['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral'])
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--tag', default='sampled')
    args = vars(parser.parse_args())
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_hebb_practice' / f"{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(args, indent=2))
    jobs = [(20261119 + world, [('watch', args['watch_steps']), (arm, args['steps'])], args) for world in range(args['worlds']) for arm in args['arms']]
    with Pool(args['workers']) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=args, runs=results)))
    print(out)


if __name__ == '__main__':
    main()
