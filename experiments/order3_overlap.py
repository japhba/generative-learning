"""Overlapping checks of order three, deep units that write through the check units, plain Hebb rule.

World. Components of six tokens (positions scattered at random) with three checks that overlap in one token each:
(t0, t1, t2), (t2, t3, t4), (t4, t5, t0), each summing to zero mod 3. 27 of the 729 configurations of a component are valid; single tokens
and token pairs are exactly uniform (order3_paper_checks.py). Control world 'apart': two checks per component that share no token.

First layer (learned by watching). Check units, found by coincidence from teacher strings as in order3_practice.py: one unit per allowed
triple of a check site (9 per site). A check unit fires when its three tokens show its triple, and writes its triple onto its tokens.

Deep layer. K units with weights W[k, site, check unit] >= 0, used for writing and reading (tied), all-to-all.
Writing: at each site the active deep unit draws one check unit from the softmax of its weights there; the check unit drives its three
tokens with the weight it was drawn with; a token takes a value from the softmax of the summed drive (a token shared by two checks
gets two drives). Reading: the check units that fire on the string; a deep unit's drive is the summed weight to those (above the smallest weight of each site), as
a share of the mean drive its own pattern would give if every site came back; the unit with the largest drive fires if the share reaches theta.
Start: one random check unit per site at weight w0.

Rule (two factors): when a deep unit fires, every one of its weights moves toward wmax if that check unit fired on the string and
toward zero otherwise, W += eps * (wmax * fired - W). Practice applies it to own samples (one unit at a time writes), watching to
teacher strings. 'store': the unit with the largest drive takes the teacher string's code in one shot, no threshold.

Scores, exact per component: kl, when every component is written by one unit drawn among the units sharp on all its checks;
kl_one_unit, when one unit writes the whole string; per_check_valid, the share of valid components when every check is written by its
own randomly drawn unit (what the first layer can do without consistent deep units).

    python order3_overlap.py --out /var/tmp/jbauer/generative_order3/overlap_main
"""
import argparse
import itertools
import json
import os
import time
import zlib
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
from scipy.special import logsumexp
from tqdm.auto import tqdm

from order3_practice import F, Judge, Surface, onehot

Q, V = 3, 9
CHECKS = dict(overlap=((0, 1, 2), (2, 3, 4), (4, 5, 0)), apart=((0, 1, 2), (3, 4, 5)))


class World:
    def __init__(self, seed, comps, kind):
        rng = np.random.default_rng([seed, 11])
        self.C, self.kind, self.L = comps, kind, 6 * comps
        self.pos = rng.permutation(self.L).reshape(comps, 6)
        self.local = np.array(CHECKS[kind])
        self.sites = self.pos[:, self.local].reshape(-1, 3)                                     # [C * checks, 3], component by component
        self.entropy = np.log(Q) * comps * (3 if kind == 'overlap' else 4)

    def teacher(self, rng, n):
        t = rng.integers(Q, size=(n, self.C, 6))
        if self.kind == 'overlap':
            t[..., 1], t[..., 3], t[..., 5] = (-t[..., 0] - t[..., 2]) % Q, (-t[..., 2] - t[..., 4]) % Q, (-t[..., 4] - t[..., 0]) % Q
        else:
            t[..., 2], t[..., 5] = (-t[..., 0] - t[..., 1]) % Q, (-t[..., 3] - t[..., 4]) % Q
        x = np.zeros((n, self.L), int)
        x[:, self.pos] = t
        return x

    def valid(self, x):
        """[N, C] components in which every check holds."""
        return (x[:, self.sites].sum(2) % Q == 0).reshape(len(x), self.C, -1).all(2)


class Learner:
    def __init__(self, rng, world, judge, units, cfg):
        self.world, self.cfg, self.K, self.S = world, cfg, units, len(world.sites)
        found = {tuple(sorted(s)): i for i, s in enumerate(judge.sites)}
        assert set(found) == {tuple(sorted(s)) for s in world.sites}, 'the judge did not find exactly the check sites'
        self.vals = np.zeros((self.S, V, 3), int)                                               # values of each check unit on its site's tokens
        self.index = -np.ones((self.S, Q ** 3), int)                                            # check unit that fires on a value triple, -1: none
        for s, site in enumerate(world.sites):
            order = np.argsort(site)                                                            # the judge keeps the tokens of a site in ascending position
            combos = np.nonzero(judge.allowed[found[tuple(sorted(site))]])[0]
            assert len(combos) == V
            v = np.zeros((V, 3), int)
            v[:, order] = np.stack([combos // 9, combos // 3 % 3, combos % 3], 1)
            self.vals[s] = v
            self.index[s, v[:, 0] * 9 + v[:, 1] * 3 + v[:, 2]] = np.arange(V)
        self.W = np.zeros((units, self.S, V), F)
        self.W[np.arange(units)[:, None], np.arange(self.S), rng.integers(V, size=(units, self.S))] = cfg['w0']

    def tokens(self, rng, tau, g):
        """strings from one check unit per site (tau [N, S]) driving its tokens with strength g [N, S]."""
        n = len(tau)
        logits = np.zeros((n, self.world.L, Q))
        for j in range(3):
            np.add.at(logits, (np.arange(n)[:, None], self.world.sites[:, j][None], self.vals[np.arange(self.S), tau, j]), g)
        return (logits + rng.gumbel(size=logits.shape)).argmax(2)

    def write(self, rng, k):
        w = self.W[k]
        tau = (w + rng.gumbel(size=w.shape)).argmax(2)
        return self.tokens(rng, tau, np.take_along_axis(w, tau[..., None], 2)[..., 0]), tau

    def code(self, x):
        """[N, S] the check unit that fires at each site, -1 where none does."""
        s = self.world.sites
        return self.index[np.arange(self.S), x[:, s[:, 0]] * 9 + x[:, s[:, 1]] * 3 + x[:, s[:, 2]]]

    def code1(self, code):
        out = np.zeros((len(code), self.S * V), F)
        n, s = np.nonzero(code >= 0)
        out[n, s * V + code[n, s]] = 1
        return out

    def drive(self, h1):
        wt = self.W - self.W.min(2, keepdims=True)                                              # weight above the smallest weight of the site: a flat site drives nothing
        p = np.exp(self.W - self.W.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
        norm = (p * wt).sum((1, 2))
        return (h1 @ wt.reshape(self.K, -1).T) / np.maximum(norm, 1e-9), norm >= self.cfg['norm_min'] * self.cfg['wmax']

    def read(self, h1):
        rel, able = self.drive(h1)
        rel = np.where(able, rel, -1.)
        best = rel.argmax(1)
        fire = np.zeros_like(rel, dtype=F)
        n = np.nonzero(rel[np.arange(len(rel)), best] >= self.cfg['theta'])[0]
        fire[n, best[n]] = 1
        return fire

    def hebb(self, fire, h1, eps):
        keep = (1 - eps) ** fire.sum(0)
        self.W = np.clip(self.W * keep[:, None, None].astype(F) + (eps * self.cfg['wmax'] * (fire.T @ h1)).reshape(self.W.shape).astype(F), 0, self.cfg['wmax'])

    def store(self, code):
        h1 = self.code1(code)
        for i in range(len(code)):
            rel, _ = self.drive(h1[i:i + 1])
            self.W[int(rel[0].argmax())] = self.cfg['wmax'] * h1[i].reshape(self.S, V)

    def sharp(self):
        return self.W.max(2) >= .5 * self.cfg['wmax']                                           # [K, S]

    def table(self):
        """log P[k, cluster, configuration]: exact probability that unit k alone writes each valid configuration of a cluster of checks
        (a component of three overlapping checks, or a single check in the control world)."""
        p = np.exp(self.W - self.W.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
        tok = lambda g, hit: np.exp(g * hit) / (np.exp(g) + Q - 1)                             # a token driven by one check unit shows the value x
        if self.world.kind == 'apart':
            out = np.zeros((self.K, self.S, V))
            for s in range(self.S):
                for t in range(V):
                    hit = self.vals[s] == self.vals[s, t]                                       # [V, 3]
                    out[:, s, t] = (p[:, s] * np.prod([tok(self.W[:, s], hit[:, j]) for j in range(3)], 0)).sum(1)
            return np.log(out)
        out = np.zeros((self.K, self.world.C, 27))

        def shared(ga, va, gb, vb, x):                                                          # a token driven by two check units: [K, V, V]
            num = np.exp(ga[:, :, None] * (va == x)[None, :, None] + gb[:, None, :] * (vb == x)[None, None, :])
            same = (va[:, None] == vb[None, :])[None]
            den = np.where(same, np.exp(ga[:, :, None] + gb[:, None, :]) + Q - 1, np.exp(ga[:, :, None]) + np.exp(gb[:, None, :]) + Q - 2)
            return num / den
        for c in range(self.world.C):
            a, b, d = 3 * c, 3 * c + 1, 3 * c + 2                                               # checks (t0 t1 t2), (t2 t3 t4), (t4 t5 t0)
            for n, (x0, x2, x4) in enumerate(itertools.product(range(Q), repeat=3)):
                x1, x3, x5 = (-x0 - x2) % Q, (-x2 - x4) % Q, (-x4 - x0) % Q
                mab = (p[:, a] * tok(self.W[:, a], self.vals[a, :, 1] == x1))[:, :, None] * shared(self.W[:, a], self.vals[a, :, 2], self.W[:, b], self.vals[b, :, 0], x2)
                mbd = (p[:, b] * tok(self.W[:, b], self.vals[b, :, 1] == x3))[:, :, None] * shared(self.W[:, b], self.vals[b, :, 2], self.W[:, d], self.vals[d, :, 0], x4)
                mda = (p[:, d] * tok(self.W[:, d], self.vals[d, :, 1] == x5))[:, :, None] * shared(self.W[:, d], self.vals[d, :, 2], self.W[:, a], self.vals[a, :, 0], x0)
                out[:, c, n] = np.einsum('kab,kbd,kda->k', mab, mbd, mda)
        return np.log(np.maximum(out, 1e-300))


def evaluate(rng, world, learner, cfg, surface):
    n, per = cfg['n_eval'], len(world.local)
    k = rng.integers(learner.K, size=n)
    x, tau = learner.write(rng, k)
    code = learner.code(x)
    fire = learner.read(learner.code1(code))
    xt = world.teacher(rng, n)
    fire_t = learner.read(learner.code1(learner.code(xt)))
    sharp = learner.sharp()
    logp = learner.table()                                                                      # [K, clusters, configurations]
    hold = sharp.reshape(learner.K, -1, per).all(2) if world.kind == 'overlap' else sharp
    kl = 0.
    for c in range(logp.shape[1]):
        ks = np.nonzero(hold[:, c])[0]
        kl += np.mean(-np.log(logp.shape[2]) - (logsumexp(logp[ks, c], axis=0) - np.log(len(ks)))) if len(ks) else np.log(Q ** (3 * per if world.kind == 'overlap' else 3) / logp.shape[2])
    t = xt[:cfg['n_kl']][:, world.pos]
    cfg_t = t[..., 0] * 9 + t[..., 2] * 3 + t[..., 4] if world.kind == 'overlap' else learner.code(xt[:cfg['n_kl']])
    kl_one = -np.mean(logsumexp(sum(logp[:, c, cfg_t[:, c]] for c in range(logp.shape[1])), axis=0) - np.log(learner.K)) - world.entropy
    sk = np.stack([np.nonzero(sharp[:, s])[0][rng.integers(max(sharp[:, s].sum(), 1), size=n)] if sharp[:, s].any() else rng.integers(learner.K, size=n) for s in range(learner.S)], 1)   # one unit per site
    w = learner.W[sk, np.arange(learner.S)]                                                     # [n, S, V]
    tau_s = (w + rng.gumbel(size=w.shape)).argmax(2)
    xs = learner.tokens(rng, tau_s, np.take_along_axis(w, tau_s[..., None], 2)[..., 0])
    return dict(valid=float(world.valid(x).mean()), kept=float((code == tau).mean()), silent=float((code < 0).mean()), writer_fires=float(fire[np.arange(n), k].mean()), other_fires=float(fire.sum(1).mean() - fire[np.arange(n), k].mean()),
                teacher_fires=float(fire_t.any(1).mean()), kl=float(kl), kl_one_unit=float(kl_one), per_check_valid=float(world.valid(xs).mean()), holders=float(hold.sum(0).mean()), sites_sharp=float(sharp.mean()),
                held_valid=float(np.mean([np.exp(logp[np.nonzero(hold[:, c])[0], c]).sum(1).mean() if hold[:, c].any() else 0. for c in range(logp.shape[1])])), mmd=surface.z(x))


def run(job):
    name, seed, cfg = job
    t0 = time.time()
    world = World(seed, cfg['comps'], cfg['kind'])
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    judge = Judge(world.teacher(np.random.default_rng([seed, 13]), cfg['n_judge']), Q)
    learner = Learner(rng, world, judge, cfg['units'], cfg)
    twin = Learner(np.random.default_rng([seed, 99]), world, judge, cfg['units'], cfg) if cfg['source'] == 'twin' else None
    surface = Surface(np.random.default_rng([seed, 15]), world.teacher, {'single tokens': (lambda x: x, Q, 1), 'token pairs': (lambda x: x, Q, 2), 'token checks': (lambda x: (x[:, world.sites] * Q ** np.arange(3)).sum(2), Q ** 3, 1)}, cfg['n_mmd'])
    bt, curve = cfg['batch'], []
    n_steps = cfg['samples'] // bt
    evals = set(np.unique(np.round(np.geomspace(1, n_steps, cfg['n_evals'])).astype(int)).tolist()) | {0, n_steps}
    for step in range(n_steps + 1):
        if step in evals:
            curve.append(dict(samples=step * bt, **evaluate(rng, world, learner, cfg, surface)))
        if step == n_steps:
            break
        if cfg['source'] == 'own':
            h1 = learner.code1(learner.code(learner.write(rng, rng.integers(learner.K, size=bt))[0]))
            learner.hebb(learner.read(h1), h1, cfg['eps'])
        elif cfg['source'] == 'teacher':
            h1 = learner.code1(learner.code(world.teacher(rng, bt)))
            learner.hebb(learner.read(h1), h1, cfg['eps'])
        elif cfg['source'] == 'twin':
            h1 = twin.code1(twin.code(twin.write(rng, rng.integers(twin.K, size=bt))[0]))
            twin.hebb(twin.read(h1), h1, cfg['eps'])
            learner.hebb(learner.read(h1), h1, cfg['eps'])
        elif cfg['source'] == 'store':
            learner.store(learner.code(world.teacher(rng, bt)))
    return dict(name=name, seed=seed, cfg=cfg, curve=curve, seconds=time.time() - t0)


BASE = dict(comps=54, kind='overlap', units=1024, w0=8., wmax=8., theta=.3, norm_min=1.5, eps=.3, source='own', samples=262144, batch=64, n_evals=14, n_eval=2048, n_mmd=1024, n_kl=512, n_judge=256)


def arms(only):
    a = dict(practice={}, watching=dict(source='teacher'), twin=dict(source='twin'), store_teacher=dict(source='store', samples=16384), apart_practice=dict(kind='apart'), apart_watching=dict(kind='apart', source='teacher'))
    small = dict(comps=27, samples=131072, worlds=4)
    for c in (3, 9, 27):
        a[f'width{c}_practice'] = dict(comps=c, samples=131072)
        a[f'width{c}_watching'] = dict(comps=c, source='teacher', samples=131072)
    for eps in (.03, .1, 1.):
        a[f'eps{eps}_practice'] = dict(small, eps=eps, samples=262144 if eps < .1 else 131072)
    for k in (64, 256, 4096):
        a[f'units{k}_practice'] = dict(small, units=k, samples=max(32768, 128 * k), batch=min(64, max(8, k // 16)))
    return {k: dict(BASE, **v) for k, v in a.items() if any(k.startswith(o) for o in only)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--only', nargs='*', default=[''])
    ap.add_argument('--samples', type=int)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed, {k: v for k, v in dict(cfg, **({'samples': args.samples} if args.samples else {})).items() if k != 'worlds'}) for name, cfg in arms(args.only).items() for seed in range(min(args.worlds, cfg.get('worlds', args.worlds)))]
    jobs.sort(key=lambda j: -j[2]['samples'] * j[2]['comps'] * j[2]['units'])
    with Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs)))
    json.dump(dict(time=datetime.now(ZoneInfo('Europe/London')).isoformat(), results=results), open(out / 'results.json', 'w'))
    for name in arms(args.only):
        rs = [r for r in results if r['name'] == name]
        med = lambda key, at: float(np.median([r['curve'][at][key] for r in rs]))
        print(f"{name:20s} valid {med('valid', 0):.3f} -> {med('valid', -1):.3f}  kl {med('kl', 0):6.1f} -> {med('kl', -1):6.1f}  one unit {med('kl_one_unit', 0):6.1f} -> {med('kl_one_unit', -1):6.1f}  per-check valid {med('per_check_valid', 0):.3f} -> {med('per_check_valid', -1):.3f}  "
              f"writer fires {med('writer_fires', 0):.2f} -> {med('writer_fires', -1):.2f}  teacher fires {med('teacher_fires', 0):.4f} -> {med('teacher_fires', -1):.4f}  holders {med('holders', -1):.0f}  {np.mean([r['seconds'] for r in rs]):.0f}s")
        print('    mmd', {k: round(v, 1) for k, v in rs[0]['curve'][0]['mmd'].items()}, '->', {k: round(v, 1) for k, v in rs[0]['curve'][-1]['mmd'].items()})


if __name__ == '__main__':
    main()
