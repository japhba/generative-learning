"""Directed stack h2 -> h1 -> x of +-1 units, teacher and learner of the same shape: a Hebb rule on recognition, and what own samples add.

The learner has bottom-up recognition weights (R1, R2) and top-down generation weights (G1, G2); every h1 unit has its own
block of visible units. All updates are exact expected updates (enumeration over x, h1, h2), so the arms differ only in rule:
  watch          Hebb with decay on the bottom-up reading of teacher samples, dR = <post pre> - decay(R) at both layers
  reread         watch + the same rule on the bottom-up reading of own samples (the rule sees the sample only)
  reread_shuffle watch + the same rule on samples of the learner's first layer alone (h1 drawn uniformly: no deep structure)
  reread_twin    watch + the same rule on the samples of a twin trained alike from another start
  kept           watch + Hebb on the generating pass of own samples (post = the unit's top-down state, pre = the layer below)
  kept_only      kept without the Hebb rule on teacher samples
  lateral        watch + inhibition between deep units that grows while their readings of teacher samples are correlated
  ws / watch_ws  classic wake-sleep (recognition by the delta rule on own samples) / the same plus watch
  exact          reference: the generator follows the likelihood gradient with its own exact posterior as recognition
The generator always learns by the wake step (delta rule) on teacher samples as read by recognition.
decay(R) is tanh(R) (the weight then ends at the log-odds of the correlation) or R (--decay linear).
"""
import argparse
import itertools
import json
import os
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

ARMS = dict(watch=dict(), reread=dict(own='reread'), reread_shuffle=dict(own='reread', source='shuffle'), reread_twin=dict(own='reread', source='twin'), kept=dict(own='kept'),
            kept_only=dict(own='kept', watch=0.), lateral=dict(lateral=True), ws=dict(own='delta', watch=0.), watch_ws=dict(own='delta'), exact=dict(exact=True), exact_shallow=dict(exact=True, shallow=True))
DEFAULT = dict(own=None, source='own', watch=1., share=1., lateral=False, exact=False, shallow=False, decay='tanh')


def spins(n):
    return np.array(list(itertools.product((-1., 1.), repeat=n)))


def softmax(logits, axis):
    p = np.exp(logits - logits.max(axis=axis, keepdims=True))
    return p / p.sum(axis=axis, keepdims=True)


def kl(p, q):
    keep = p > 0
    return float(np.sum(p[keep] * (np.log(p[keep]) - np.log(q[keep]))))


def mutual_information(joint):
    outer = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    keep = joint > 0
    return float(np.sum(joint[keep] * (np.log(joint[keep]) - np.log(outer[keep]))))


class Shape:
    def __init__(self, n, m, k, block):
        self.n, self.m, self.k = n, m, k
        self.X, self.H1, self.H2 = spins(n), spins(m), spins(k)
        self.mask = (np.arange(n)[:, None] // block == np.arange(m)[None]) * 1.
        self.bit = [np.stack([self.H2[:, j] < 0, self.H2[:, j] > 0], 1) * 1. for j in range(k)]


def generator(s, g1, g2):
    """p(h1 | h2), p(x | h1), the law of h1 and the law of x; the top units are independent and unbiased."""
    p21, p10 = softmax(s.H2 @ g2.T @ s.H1.T, 1), softmax(s.H1 @ g1.T @ s.X.T, 1)
    return p21, p10, p21.mean(0), p21.mean(0) @ p10


def reader(s, r1, r2, lat):
    """q(h1 | x) and q(h2 | h1); with lateral inhibition deep unit j is read after units i < j and sees -lat[j, i] h2_i."""
    field = (s.H1 @ r2.T)[:, None, :] - (s.H2 @ np.tril(lat, -1).T)[None]
    return softmax(s.X @ r1.T @ s.H1.T, 1), np.exp((s.H2[None] * field - np.logaddexp(field, -field)).sum(2))


def posterior(p21, p10, ph1, px):
    return (p10 * ph1[:, None]).T / px[:, None], p21.T / p21.sum(0)[:, None]


def read(s, law, q1, q2):
    """<h1 x^T> and <h2 h1^T> when x ~ law is read bottom-up, with the joint laws of (x, h1) and (h1, h2)."""
    joint1 = law[:, None] * q1
    joint2 = joint1.sum(0)[:, None] * q2
    return s.H1.T @ joint1.T @ s.X, s.H2.T @ joint2.T @ s.H1, joint1, joint2


def decayed(cfg, r):
    return np.tanh(r) if cfg['decay'] == 'tanh' else r


def update(s, arm, w, pstar, foreign=None):
    """Expected change of (R1, R2, G1, G2, lateral) per unit learning rate."""
    r1, r2, g1, g2, lat = w
    cfg = DEFAULT | arm
    p21, p10, ph1, px = generator(s, g1, g2)
    q1, q2 = posterior(p21, p10, ph1, px) if cfg['exact'] else reader(s, r1, r2, lat)
    hebb1, hebb2, joint1, joint2 = read(s, pstar, q1, q2)
    dg1 = hebb1.T - (joint1.sum(0)[:, None] * np.tanh(s.H1 @ g1.T)).T @ s.H1
    dg2 = (hebb2.T - np.tanh(s.H2 @ g2.T).T @ (joint2.sum(0)[:, None] * s.H2)) * (not cfg['shallow'])
    dr1, dr2 = cfg['watch'] * (hebb1 - decayed(cfg, r1)), cfg['watch'] * (hebb2 - decayed(cfg, r2))
    dlat = np.tril((s.H2 * joint2.sum(0)[:, None]).T @ s.H2, -1) * cfg['lateral']
    if cfg['own'] == 'reread':
        law = dict(own=px, twin=foreign, shuffle=p10.mean(0))[cfg['source']]
        own1, own2, _, _ = read(s, law, q1, q2)
        dr1, dr2 = dr1 + cfg['share'] * (own1 - decayed(cfg, r1)), dr2 + cfg['share'] * (own2 - decayed(cfg, r2))
    if cfg['own'] in ('kept', 'delta'):
        kept1, kept2 = (ph1[:, None] * s.H1).T @ np.tanh(s.H1 @ g1.T), s.H2.T @ p21 @ s.H1 / len(s.H2)
        if cfg['own'] == 'kept':
            back1, back2 = decayed(cfg, r1), decayed(cfg, r2)
        else:
            back1, back2 = (px[:, None] * np.tanh(s.X @ r1.T)).T @ s.X, (ph1[:, None] * np.tanh(s.H1 @ r2.T)).T @ s.H1
        dr1, dr2 = dr1 + cfg['share'] * (kept1 - back1), dr2 + cfg['share'] * (kept2 - back2)
    return [s.mask.T * dr1, dr2, s.mask * dg1, dg2, dlat]


def split_rates(s, cfg, w, pstar, px, q1):
    """Two deep units that read alike: size of their difference and its predicted growth rate under each rule.

    With delta = r_1 -+ r_2 (recognition rows) and gamma = g_1 -+ g_2 (generation columns), A(law) = <h1 h1^T sech^2(r h1)> over the
    codes of that law and D the slope of the decay: watch gives d delta/dt = (A_T - D) delta, re-read own samples add
    share (A_own - D) delta, the generating pass adds share (gamma - D delta), and the wake step gives d gamma/dt = A_T delta - u gamma
    with u = <sech^2(r h1)> over teacher codes, the rate at which the two units disagree about a teacher sample.
    """
    r1, r2, g1, g2, lat = w
    sign = 1. if r2[0] @ r2[1] >= 0 else -1.
    mean = (r2[0] + sign * r2[1]) / 2
    slack = 1 - np.tanh(s.H1 @ mean) ** 2
    spread = [(s.H1 * (law @ q1 * slack)[:, None]).T @ s.H1 for law in (pstar, px)]
    undecided = float(pstar @ q1 @ slack)
    slope = np.diag(1 - np.tanh(mean) ** 2) if cfg['decay'] == 'tanh' else np.eye(s.m)
    share, eye = cfg['share'], np.eye(s.m)
    loop = lambda watch: float(np.linalg.eigvals(np.block([[watch * spread[0] - (watch + share) * slope, share * eye], [spread[0], -undecided * eye]])).real.max())
    delta = r2[0] - sign * r2[1]
    return dict(split=float(np.linalg.norm(delta)), split_g=float(np.linalg.norm(g2[:, 0] - sign * g2[:, 1])), split_align=float(abs(np.linalg.eigh(spread[0])[1][:, -1] @ delta) / max(np.linalg.norm(delta), 1e-300)), undecided=undecided, spread=float(np.linalg.eigvalsh(spread[0])[-1] / undecided),
                rate_watch=float(np.linalg.eigvalsh(spread[0] - slope)[-1]), rate_reread=float(np.linalg.eigvalsh(spread[0] + share * spread[1] - (1 + share) * slope)[-1]), rate_kept=loop(1.), rate_kept_only=loop(0.))


def teacher_tables(t, a1, a2):
    p21, p10, ph1, px = generator(t, a1, a2)
    return dict(px=px, top_x=p21 @ p10 / len(t.H2), shape=t)


def evaluate(s, arm, w, teacher):
    r1, r2, g1, g2, lat = w
    cfg = DEFAULT | arm
    t, pstar = teacher['shape'], teacher['px']
    p21, p10, ph1, px = generator(s, g1, g2)
    q1, q2 = posterior(p21, p10, ph1, px) if cfg['exact'] else reader(s, r1, r2, lat)
    top = teacher['top_x'] @ q1 @ q2
    code = top.sum(0)
    pairs = [[mutual_information(t.bit[i].T @ top @ s.bit[j]) for j in range(s.k)] for i in range(t.k)]
    corr = [float(code @ (s.H2[:, i] * s.H2[:, j])) for i in range(s.k) for j in range(i)]
    drive = lambda law: float(np.linalg.eigvalsh((s.H1 * (law @ q1)[:, None]).T @ s.H1 - np.eye(s.m))[-1])
    out = dict(kl=kl(pstar, px), kl_no_deep=kl(pstar, p10.mean(0)), read=mutual_information(top), read_pairs=pairs, unit_corr=corr, prior_gap=float(np.log(len(s.H2)) + code @ np.log(np.maximum(code, 1e-300))),
               reach_own=mutual_information(p21 @ p10 @ q1 @ q2 / len(s.H2)), norm_r1=float(np.linalg.norm(r1)), norm_r2=float(np.linalg.norm(r2)), norm_g2=float(np.linalg.norm(g2)),
               align=float((r2 * g2.T).sum() / max(np.linalg.norm(r2) * np.linalg.norm(g2), 1e-12)), drive_teacher=drive(pstar), drive_own=drive(px))
    return out | (split_rates(s, cfg, w, pstar, px, q1) if s.k == 2 and not cfg['exact'] else {})


def start(s, rng, kind, amplitude, init):
    w = [s.mask.T * rng.normal(0, init, (s.m, s.n)), rng.normal(0, init, (s.k, s.m)), s.mask * rng.normal(0, init, (s.n, s.m)), rng.normal(0, init, (s.m, s.k)), np.zeros((s.k, s.k))]
    if kind in ('coupled', 'decoupled'):
        deep = rng.normal(0, amplitude, (s.k, s.m))
        w[1], w[3] = deep, (deep if kind == 'coupled' else rng.normal(0, amplitude, (s.k, s.m))).T.copy()
    return w


def run(job):
    args = job['args']
    t, s = Shape(args['n'], args['m'], args['k'], args['block']), Shape(args['n'], args['m'], job['k'], args['block'])
    rng = np.random.default_rng(job['seed'])
    a1, a2 = t.mask * rng.normal(0, args['g1'], (t.n, t.m)), rng.normal(0, args['g2'], (t.m, t.k))
    teacher = teacher_tables(t, a1, a2)
    learners = [start(s, rng, job['kind'], job['amplitude'], args['init']) for _ in range(2)][:1 + any(ARMS[name].get('source') == 'twin' for name, _ in job['stages'])]
    jitter = [[rng.normal(0, args['jitter'], v.shape) for v in w] for w in learners]
    history, step = [[] for _ in learners], 0
    for number, (name, steps) in enumerate(job['stages']):
        arm = ARMS[name] | dict(share=job['share'], decay=args['decay'])
        if number:
            learners = [[w[0] + s.mask.T * e[0], w[1] + e[1], w[2] + s.mask * e[2], w[3] + e[3], w[4]] for w, e in zip(learners, jitter)]
        for local in range(steps + 1):
            if local % args['every'] == 0 or local in args['early']:
                for j, w in enumerate(learners):
                    history[j].append(dict(step=step + local, stage=number) | evaluate(s, arm, w, teacher))
            if local == steps:
                break
            laws = [generator(s, w[2], w[3])[3] for w in learners][::-1] if arm.get('source') == 'twin' else [None]
            learners = [[v + args['rate'] * d for v, d in zip(w, update(s, arm, w, teacher['px'], laws[j]))] for j, w in enumerate(learners)]
        step += steps
    keys = history[0][0].keys()
    return {key: job[key] for key in ('seed', 'stages', 'kind', 'amplitude', 'k', 'share')} | dict(teacher=[a1.tolist(), a2.tolist()], weights=[[v.tolist() for v in w] for w in learners],
                                                                                                    curves=[{key: [row[key] for row in h] for key in keys} for h in history])


def audit(args):
    """The linear laws of the write-up on a random state: watch grows small deep weights by (C1 - 1) R2; re-read own samples act at second order in the generator's deep weights, the generating pass at first order."""
    s = Shape(args['n'], args['m'], args['k'], args['block'])
    rng = np.random.default_rng(0)
    w = start(s, rng, 'blank', 0, .5)
    pstar = rng.dirichlet(np.ones(len(s.X)))
    small = 1e-5 * rng.normal(size=(s.k, s.m))
    q1, _ = reader(s, w[0], small, w[4])
    code = (s.H1 * (pstar @ q1)[:, None]).T @ s.H1
    watch = update(s, ARMS['watch'], [w[0], small, w[2], w[3], w[4]], pstar)[1]
    linear_error = float(np.abs(watch - small @ (code - np.eye(s.m))).max() / np.abs(watch).max())
    assert linear_error < 1e-6, linear_error
    orders, direction = {}, rng.normal(size=(s.m, s.k))
    for name in ('reread', 'kept'):
        sizes = [np.linalg.norm(update(s, ARMS[name] | dict(watch=0.), [w[0], small, w[2], scale * direction, w[4]], pstar)[1] + small * (name == 'kept')) for scale in (1e-3, 2e-3)]
        orders[name] = float(np.log2(sizes[1] / sizes[0]))
    assert abs(orders['reread'] - 2) < .01 and abs(orders['kept'] - 1) < .01, orders
    return dict(linear_error=linear_error, order_reread=orders['reread'], order_kept=orders['kept'])


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
    parser.add_argument('--jitter', type=float, default=1e-3)
    parser.add_argument('--rate', type=float, default=.1)
    parser.add_argument('--decay', default='tanh', choices=['linear', 'tanh'])
    parser.add_argument('--watch_steps', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=20000)
    parser.add_argument('--scratch_steps', type=int, default=12000)
    parser.add_argument('--started_worlds', type=int, default=8)
    parser.add_argument('--every', type=int, default=100)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--fork', nargs='*', default=['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral', 'ws', 'watch_ws'])
    parser.add_argument('--shares', type=float, nargs='*', default=[.25, 4.])
    parser.add_argument('--shared', nargs='*', default=['reread', 'kept'])
    parser.add_argument('--scratch', nargs='*', default=['watch', 'reread', 'kept', 'kept_only', 'lateral', 'ws', 'watch_ws', 'exact', 'exact_shallow'])
    parser.add_argument('--amplitudes', type=float, nargs='*', default=[.3, 1., 3.])
    parser.add_argument('--started', nargs='*', default=['watch', 'reread', 'reread_twin', 'kept'])
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--tag', default='main')
    args = vars(parser.parse_args())
    args['early'] = [int(v) for v in np.unique(np.round(np.geomspace(1, args['every'], 16)))]
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_hebb_practice' / f"{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    checks = audit(args)
    (out / 'protocol.json').write_text(json.dumps(args | checks, indent=2))
    base = dict(kind='blank', amplitude=0., k=args['k'], share=1., args=args)
    total, jobs = args['scratch_steps'], []
    for world, seed in enumerate(20261119 + world for world in range(args['worlds'])):
        jobs += [base | dict(seed=seed, stages=[('watch', args['watch_steps']), (arm, args['steps'])]) for arm in args['fork']]
        jobs += [base | dict(seed=seed, stages=[('watch', args['watch_steps']), (arm, args['steps'])], share=share) for arm in args['shared'] for share in args['shares']]
        jobs += [base | dict(seed=seed, stages=[(arm, total)]) for arm in args['scratch']] + [base | dict(seed=seed, stages=[('exact', total)], k=1)] * bool(args['scratch'])
        jobs += [base | dict(seed=seed, stages=[(arm, total)], kind=kind, amplitude=amplitude) for arm in args['started'] for kind in ('coupled', 'decoupled') for amplitude in args['amplitudes']] * (world < args['started_worlds'])
    with Pool(args['workers']) as pool:
        results = list(tqdm(pool.imap_unordered(run, sorted(jobs, key=lambda job: -sum(steps for _, steps in job['stages']))), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=args, checks=checks, runs=results)))
    print(out)


if __name__ == '__main__':
    main()
