"""hebb_practice_depth.py with couplings across layers: the deep units also touch the visible units directly.

Teacher and learner alike: x is written from h1 and from h2 (generation weights G1 and T), h2 is read from h1 and from x
(recognition weights R2 and S). Rules as in hebb_practice_depth.py, with S learnt by the same Hebb rule as R2 and T by the
same wake step as G1. Arms: watch, reread, reread_shuffle, kept, lateral; protocol: watch, then the arm.
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

from hebb_practice_depth import ARMS, DEFAULT, Shape, kl, mutual_information, softmax


def generator(s, g1, g2, skip):
    """p(h1 | h2), the mean of x given (h1, h2), p(x | h1, h2) and the law of x."""
    p21 = softmax(s.H2 @ g2.T @ s.H1.T, 1)
    field = (s.H1 @ g1.T)[:, None, :] + (s.H2 @ skip.T)[None]
    px3 = np.exp(field @ s.X.T - np.logaddexp(field, -field).sum(2, keepdims=True))
    return p21, np.tanh(field), px3, np.einsum('km,mkx->x', p21, px3) / len(s.H2)


def reader(s, r1, r2, skip, lat):
    """q(h1 | x) and q(h2 | x, h1); deep unit j is read after units i < j and sees -lat[j, i] h2_i."""
    field = (s.H1 @ r2.T)[None, :, None, :] + (s.X @ skip.T)[:, None, None, :] - (s.H2 @ np.tril(lat, -1).T)[None, None]
    return softmax(s.X @ r1.T @ s.H1.T, 1), np.exp((s.H2[None, None] * field - np.logaddexp(field, -field)).sum(3))


def read(s, law, q1, q2):
    joint = (law[:, None] * q1)[:, :, None] * q2
    return s.H1.T @ joint.sum(2).T @ s.X, s.H2.T @ joint.sum(0).T @ s.H1, s.H2.T @ joint.sum(1).T @ s.X, joint.sum(0)


def update(s, arm, w, pstar):
    r1, r2, rs, g1, g2, gs, lat = w
    cfg = DEFAULT | arm
    p21, mean, px3, px = generator(s, g1, g2, gs)
    q1, q2 = reader(s, r1, r2, rs, lat)
    hebb1, hebb2, hebbs, codes = read(s, pstar, q1, q2)
    written = np.einsum('mk,mkn->mkn', codes, mean)
    dg1, dgs = hebb1.T - np.einsum('mkn,ma->na', written, s.H1), hebbs.T - np.einsum('mkn,kb->nb', written, s.H2)
    dg2 = hebb2.T - np.tanh(s.H2 @ g2.T).T @ (codes.sum(0)[:, None] * s.H2)
    dr = [cfg['watch'] * (h - np.tanh(r)) for h, r in ((hebb1, r1), (hebb2, r2), (hebbs, rs))]
    dlat = np.tril((s.H2 * codes.sum(0)[:, None]).T @ s.H2, -1) * cfg['lateral']
    if cfg['own'] == 'reread':
        law = px if cfg['source'] == 'own' else generator(s, g1, 0 * g2, 0 * gs)[3]
        dr = [d + cfg['share'] * (h - np.tanh(r)) for d, h, r in zip(dr, read(s, law, q1, q2)[:3], (r1, r2, rs))]
    if cfg['own'] == 'kept':
        pair = p21.T / len(s.H2)
        kept = np.einsum('mk,ma,mkn->an', pair, s.H1, mean), s.H2.T @ pair.T @ s.H1, np.einsum('mk,kb,mkn->bn', pair, s.H2, mean)
        dr = [d + cfg['share'] * (h - np.tanh(r)) for d, h, r in zip(dr, kept, (r1, r2, rs))]
    return [s.mask.T * dr[0], dr[1], dr[2], s.mask * dg1, dg2, dgs, dlat]


def evaluate(s, w, teacher):
    r1, r2, rs, g1, g2, gs, lat = w
    q1, q2 = reader(s, r1, r2, rs, lat)
    top = teacher['top_x'] @ (q1[:, :, None] * q2).reshape(len(s.X), -1)
    top = top.reshape(len(s.H2), len(s.H1), len(s.H2)).sum(1)
    code = top.sum(0)
    rows = np.concatenate([r2, rs], 1)
    sign = 1. if rows[0] @ rows[1] >= 0 else -1.
    return dict(kl=kl(teacher['px'], generator(s, g1, g2, gs)[3]), read=mutual_information(top), read_pairs=[[mutual_information(s.bit[i].T @ top @ s.bit[j]) for j in range(s.k)] for i in range(s.k)],
                unit_corr=float(code @ (s.H2[:, 0] * s.H2[:, 1])), prior_gap=float(np.log(len(s.H2)) + code @ np.log(np.maximum(code, 1e-300))), split=float(np.linalg.norm(rows[0] - sign * rows[1])),
                norm_r2=float(np.linalg.norm(r2)), norm_rs=float(np.linalg.norm(rs)), norm_gs=float(np.linalg.norm(gs)))


def run(job):
    args = job['args']
    s = Shape(args['n'], args['m'], args['k'], args['block'])
    rng = np.random.default_rng(job['seed'])
    a1, a2, askip = s.mask * rng.normal(0, args['g1'], (s.n, s.m)), rng.normal(0, args['g2'], (s.m, s.k)), rng.normal(0, args['gskip'], (s.n, s.k))
    p21, _, px3, px = generator(s, a1, a2, askip)
    teacher = dict(px=px, top_x=np.einsum('km,mkx->kx', p21, px3) / len(s.H2))
    shapes = [(s.m, s.n), (s.k, s.m), (s.k, s.n), (s.n, s.m), (s.m, s.k), (s.n, s.k)]
    masks = [s.mask.T, 1., 1., s.mask, 1., 1.]
    w = [mask * rng.normal(0, args['init'], shape) for mask, shape in zip(masks, shapes)] + [np.zeros((s.k, s.k))]
    jitter = [mask * rng.normal(0, args['jitter'], shape) for mask, shape in zip(masks, shapes)] + [0.]
    history, step = [], 0
    for number, (name, steps) in enumerate(job['stages']):
        arm = ARMS[name]
        if number:
            w = [v + e for v, e in zip(w, jitter)]
        for local in range(steps + 1):
            if local % args['every'] == 0:
                history.append(dict(step=step + local, stage=number) | evaluate(s, w, teacher))
            if local == steps:
                break
            w = [v + args['rate'] * d for v, d in zip(w, update(s, arm, w, teacher['px']))]
        step += steps
    return dict(seed=job['seed'], stages=job['stages'], curves={key: [row[key] for row in history] for key in history[0]})


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=12)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--block', type=int, default=3)
    parser.add_argument('--g1', type=float, default=1.8)
    parser.add_argument('--g2', type=float, default=1.5)
    parser.add_argument('--gskip', type=float, default=.5)
    parser.add_argument('--init', type=float, default=.01)
    parser.add_argument('--jitter', type=float, default=1e-3)
    parser.add_argument('--rate', type=float, default=.1)
    parser.add_argument('--watch_steps', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--every', type=int, default=250)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--arms', nargs='+', default=['watch', 'reread', 'reread_shuffle', 'kept', 'lateral'])
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--tag', default='skip')
    args = vars(parser.parse_args())
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_hebb_practice' / f"{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(args, indent=2))
    jobs = [dict(seed=20261119 + world, stages=[('watch', args['watch_steps']), (arm, args['steps'])], args=args) for world in range(args['worlds']) for arm in args['arms']]
    with Pool(args['workers']) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=args, runs=results)))
    print(out)


if __name__ == '__main__':
    main()
