"""Order-three world with overlapping checks, learner = a network of winner-take-all groups with a composed deep code.

World: components of six tokens with q values each and three checks that share one token each, (t0, t1, t2), (t2, t3, t4), (t4, t5, t0),
each summing to zero mod q. q^3 of the q^6 components are valid; single tokens and token pairs are uniform. Positions scattered at random.

Network, three layers of units, each layer made of winner-take-all groups (one unit of a group is active):
  tokens (one group of q per position) - check units (one group of q^2 per check site, wired to the three tokens of the site, one unit per
  triple the site showed in the first teacher strings) - deep units (K groups of n units, all-to-all with the check units).
Weights between a deep unit and a check unit are log conditional probabilities, w = log P(check unit | deep unit). Separate arrays for
reading (R) and writing (G), same start.
  writing: every deep group draws a unit from a uniform prior; a check group draws its unit from softmax(sum over deep groups of
           G - log of the group's average probability of that check unit); a check unit writes its triple; a token that two check
           units write takes one of the two values at random.
  reading: a check unit fires if its triple is on the string (a site whose triple is not allowed stays silent); a deep group draws its
           unit from softmax(sum over firing check units of R).
Rule (two factors, the same for R and G): when a deep unit is active, its weights move by eps * (pre * exp(-w) - 1), i.e. the stored
probabilities move toward what the check groups showed, P <- P + eps * (pre - P), at the sites whose check group fired (a silent site
changes nothing).

Sources of the (deep, check) pairs the rule is applied to:
  teacher string: deep = reading of the string.
  own sample:     check = what the first layer reads back from the written string (readback) or the check units as drawn;
                  deep = the units that wrote (credit 'writer') or the reading of the own string (credit 'read').
  twin:           a second network runs the same schedule from another start; its samples replace the own samples (deep = this network's reading).
teacher_trains / own_trains say which of R and G a source moves. Watching: teacher strings move both. Classic wake-sleep: teacher
strings move G only, own samples R only. The first `watch_first` samples are always watching; a stored set of `set` teacher strings
replaces fresh ones if set > 0.

    python order3_net.py --out /var/tmp/jbauer/generative_order3/net_main
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
from tqdm.auto import tqdm

from order3_practice import F, Judge, Surface

LOCAL = np.array(((0, 1, 2), (2, 3, 4), (4, 5, 0)))
SHARED = ((0, 0, 0, 2, 2), (2, 0, 2, 1, 0), (4, 1, 2, 2, 0))                                     # (local token, site a, slot in a, site b, slot in b)


class World:
    def __init__(self, seed, comps, q):
        rng = np.random.default_rng([seed, 11])
        self.C, self.q, self.L = comps, q, 6 * comps
        self.pos = rng.permutation(self.L).reshape(comps, 6)
        self.sites = self.pos[:, LOCAL].reshape(-1, 3)                                          # [3 C, 3], component by component
        self.valid_idx = np.array([sum(v * q ** j for j, v in enumerate((a, (-a - b) % q, b, (-b - c) % q, c, (-c - a) % q))) for a, b, c in itertools.product(range(q), repeat=3)])

    def teacher(self, rng, n):
        t = rng.integers(self.q, size=(n, self.C, 6))
        t[..., 1], t[..., 3], t[..., 5] = (-t[..., 0] - t[..., 2]) % self.q, (-t[..., 2] - t[..., 4]) % self.q, (-t[..., 4] - t[..., 0]) % self.q
        x = np.zeros((n, self.L), int)
        x[:, self.pos] = t
        return x

    def valid(self, x):
        return (x[:, self.sites].sum(2) % self.q == 0).reshape(len(x), self.C, 3).all(2)

    def stats(self, x):
        """sum over components of KL(teacher || learner) from the sample, share of valid components the sample shows, share of valid components."""
        n, v = len(x), len(self.valid_idx)
        cfg = (x[:, self.pos] * self.q ** np.arange(6)).sum(2)
        counts = np.stack([np.bincount(cfg[:, c], minlength=self.q ** 6)[self.valid_idx] for c in range(self.C)])
        return float(np.sum(np.mean(-np.log(v) - np.log(np.maximum(counts / n, 1 / (2 * n))), 1))), float((counts >= n / (4 * v)).mean()), float(counts.sum() / (n * self.C))


class FirstLayer:
    """check units of every site, stored from the triples the site showed; rendering of tokens and reading back."""

    def __init__(self, world, strings):
        q = world.q
        self.world, self.S, self.L, self.V = world, len(world.sites), world.L, q * q
        judge = Judge(strings, q, sites=world.sites)
        self.vals = np.zeros((self.S, self.V, 3), int)
        self.index = -np.ones((self.S, q ** 3), int)
        for s in range(self.S):
            combos = np.nonzero(judge.allowed[s])[0]
            assert len(combos) == self.V, 'a site has not shown all its triples yet'
            self.vals[s] = np.stack([combos // q ** 2, combos // q % q, combos % q], 1)
            self.index[s, combos] = np.arange(self.V)
        cover = [[] for _ in range(self.L)]
        for s, site in enumerate(world.sites):
            for j, p in enumerate(site):
                cover[p].append((s, j))
        self.c1 = np.array([c[0] for c in cover]); self.c2 = np.array([c[1] if len(c) > 1 else (-1, 0) for c in cover])
        self.has2 = self.c2[:, 0] >= 0

    def render(self, rng, tau):
        """check units tau [N, S] -> tokens [N, L]."""
        x = self.vals[self.c1[:, 0], tau[:, self.c1[:, 0]], self.c1[:, 1]]
        x2 = self.vals[self.c2[:, 0], tau[:, self.c2[:, 0]], self.c2[:, 1]]
        return np.where(self.has2 & (rng.random(x.shape) < .5), x2, x)

    def code(self, x):
        s, q = self.world.sites, self.world.q
        return self.index[np.arange(self.S), x[:, s[:, 0]] * q * q + x[:, s[:, 1]] * q + x[:, s[:, 2]]]

    def onehot(self, code):
        out = np.zeros((len(code), self.S * self.V), F)
        n, s = np.nonzero(code >= 0)
        out[n, s * self.V + code[n, s]] = 1
        return out


class Net:
    def __init__(self, rng, first, cfg):
        self.f, self.cfg, self.K, self.n, self.S, self.V = first, cfg, cfg['groups'], cfg['states'], first.S, first.V
        p = np.exp(cfg['init'] * rng.normal(size=(self.K, self.n, self.S, self.V)))
        self.R = p / p.sum(3, keepdims=True)
        self.G = self.R.copy()

    def logw(self, P, ratio=False):
        """log P(check unit | deep unit); for writing, relative to the group's average over its units, so that what all units of a group
        agree on is not counted once per group."""
        w = np.log(P + self.cfg['floor'])
        return (w - (np.log(P.mean(1, keepdims=True) + self.cfg['floor']) if ratio else 0)).reshape(self.K * self.n, self.S * self.V).astype(F)

    def z1(self, z):
        out = np.zeros((len(z), self.K * self.n), F)
        out[np.arange(len(z))[:, None], np.arange(self.K) * self.n + z] = 1
        return out

    def write(self, rng, z):
        """deep units z [N, K] -> tokens, read-back code [N, S] (-1 silent), check units as drawn."""
        logits = (self.z1(z) @ self.logw(self.G, ratio=True)).reshape(len(z), self.S, self.V)
        tau = (logits + rng.gumbel(size=logits.shape)).argmax(2)
        x = self.f.render(rng, tau)
        return x, self.f.code(x), tau

    def read(self, rng, code):
        logits = (self.f.onehot(code) @ self.logw(self.R).T).reshape(len(code), self.K, self.n)
        return (logits + rng.gumbel(size=logits.shape)).argmax(2), logits

    def hebb(self, which, z, code, eps):
        """for every active deep unit and every site whose check group fired: P <- P + eps * (pre - P), sample after sample."""
        z1, h1 = self.z1(z), self.f.onehot(code)
        count = (z1.T @ (code >= 0).astype(F)).astype(np.float64)                               # [K n, S] samples in which the unit was active and the site fired
        keep = np.repeat((1 - eps) ** count, self.V, 1)
        mean = (z1.T @ h1).astype(np.float64) / np.repeat(np.maximum(count, 1), self.V, 1)
        for name in which:
            P = getattr(self, name).reshape(self.K * self.n, self.S * self.V)
            setattr(self, name, (keep * P + (1 - keep) * mean).reshape(self.K, self.n, self.S, self.V))

    def loadings(self, P):
        """how much the units of a deep group tell the value of each token: [K, C, 6] (1 = one unit per value), and for the shared tokens
        how much the two sites of the token disagree."""
        C, q = self.f.world.C, self.f.world.q
        p = P / P.sum(3, keepdims=True)
        m = np.zeros((self.S, self.V, 3, q)); m[np.arange(self.S)[:, None, None], np.arange(self.V)[None, :, None], np.arange(3)[None, None], self.f.vals] = 1
        tok = np.einsum('gksu,suji->gksji', p, m).reshape(self.K, self.n, C, 3, 3, q)            # [K, n, component, site, slot, value]
        dev = lambda a: np.sqrt(((a - a.mean(1, keepdims=True)) ** 2).mean((1, -1)) / ((q - 1) / q ** 2))
        load, clash = np.zeros((self.K, C, 6)), np.zeros((self.K, C, 3))
        for j, (t, sa, ja, sb, jb) in enumerate(SHARED):
            a, b = tok[:, :, :, sa, ja], tok[:, :, :, sb, jb]
            load[:, :, t], clash[:, :, j] = dev((a + b) / 2), dev((a - b) / 2)
        for s in range(3):
            load[:, :, 2 * s + 1] = dev(tok[:, :, :, s, 1])
        return load, clash


def evaluate(rng, world, first, net, cfg, surface, final=False):
    n = cfg['n_eval']
    z = rng.integers(net.n, size=(n, net.K))
    x, code, tau = net.write(rng, z)
    kl, cover, valid = world.stats(x)
    zr, _ = net.read(rng, code)
    x2, code2, _ = net.write(rng, zr)                                                           # own sample read and written again
    xt = world.teacher(rng, n)
    ct = first.code(xt)
    zt, logit_t = net.read(rng, ct)
    xt2, ct2, _ = net.write(rng, zt)                                                            # teacher string read and written again
    conf = lambda lg: float(np.mean(np.exp(lg - lg.max(2, keepdims=True)).max(2) / np.exp(lg - lg.max(2, keepdims=True)).sum(2)))
    out = dict(valid=valid, kl=kl, coverage=cover, silent=float((code < 0).mean()), kept=float((code == tau).mean()),
               own_read=float((zr == z).mean()), own_trip=float((code2 == code)[code >= 0].mean()), teacher_trip=float((ct2 == ct).mean()),
               teacher_trip_tokens=float((xt2 == xt).mean()), teacher_trip_valid=float(world.valid(xt2).mean()), teacher_conf=conf(logit_t),
               usage=float(np.mean(np.abs(np.stack([(zt == k).mean(0) for k in range(net.n)]) - 1 / net.n).max(0))), mmd=surface.z(x))
    for side, P in (('', net.G), ('r_', net.R)):
        load, clash = net.loadings(P)
        sh = load[:, :, ::2].reshape(net.K, -1)                                                 # [K, shared tokens]
        top = np.sort(sh, 0)[::-1]
        out.update({side + 'token_top': float(top[0].mean()), side + 'tokens_owned': float((top[0] > .7).mean()), side + 'tokens_shared': float((top[1] > .7).mean()),
                    side + 'groups_free': float((sh.max(1) < .3).mean()), side + 'group_pr': float(np.median((sh ** 2).sum(1) ** 2 / np.maximum((sh ** 4).sum(1), 1e-12))),
                    side + 'group_top': float(np.median(sh.max(1))), side + 'private': float(load[:, :, 1::2].max(0).mean()), side + 'clash': float(clash.max(0).mean())})
        if final or cfg['trace']:
            out[side + 'loadings'] = np.round(sh, 3).tolist()
        if not side:                                                                            # two groups on one token: how often they read a teacher string the same way (up to naming)
            two = np.nonzero(top[1] > .7)[0]
            g = np.argsort(sh[:, two], 0)[-2:]
            out['co_owner_agree'] = float(np.mean(np.abs(2 * (zt[:, g[0]] == zt[:, g[1]]).mean(0) - 1))) if len(two) else float('nan')
    return out


def run(job):
    name, seed, cfg = job
    t0 = time.time()
    world = World(seed, cfg['comps'], cfg['q'])
    q = world.q
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    first = FirstLayer(world, world.teacher(np.random.default_rng([seed, 13]), cfg['n_first']))
    net = Net(rng, first, cfg)
    twin = Net(np.random.default_rng([seed, 99]), first, cfg) if cfg['twin'] else None
    surface = Surface(np.random.default_rng([seed, 15]), world.teacher, {'single tokens': (lambda x: x, q, 1), 'token pairs': (lambda x: x, q, 2), 'token checks': (lambda x: (x[:, world.sites] * q ** np.arange(3)).sum(2), q ** 3, 1)}, cfg['n_mmd'])
    stored = world.teacher(np.random.default_rng([seed, 17]), cfg['set']) if cfg['set'] else None
    draw = lambda n: stored[rng.integers(len(stored), size=n)] if cfg['set'] else world.teacher(rng, n)
    bt, eps = cfg['batch'], cfg['eps']
    n_steps = cfg['samples'] // bt
    evals = set(np.unique(np.round(np.geomspace(1, n_steps, cfg['n_evals'])).astype(int)).tolist()) | {0, n_steps, cfg['watch_first'] // bt}
    curve, own_share = [], 0.

    def watch(m, which):
        code = first.code(draw(bt))
        m.hebb(tuple(which), m.read(rng, code)[0], code, eps)

    def practice(m, source):
        """own samples of `source` teach m: the deep side of the pair is the writing units (credit 'writer', only for m's own samples) or m's reading."""
        z = rng.integers(source.n, size=(bt, source.K))
        x, code, tau = source.write(rng, z)
        pre = code if cfg['readback'] else tau
        m.hebb(tuple(cfg['own_trains']), z if cfg['credit'] == 'writer' and source is m else m.read(rng, pre)[0], pre, eps * cfg['eps_own'])

    for step in range(n_steps + 1):
        if step in evals:
            curve.append(dict(samples=step * bt, **evaluate(rng, world, first, net, cfg, surface, final=step == n_steps)))
        if step == n_steps:
            break
        first_phase = step * bt < cfg['watch_first']
        own = not first_phase and own_share < cfg['own'] * (step + 1 - cfg['watch_first'] // bt)        # own samples make up the share cfg['own'] after the first teacher strings
        own_share += own
        for m in (net, twin) if twin else (net,):
            if not own:
                watch(m, 'RG' if first_phase else cfg['teacher_trains'])
            elif m is twin or not twin:
                practice(m, m)
            else:
                practice(net, twin)
    return dict(name=name, seed=seed, cfg=cfg, curve=curve, seconds=time.time() - t0)


BASE = dict(q=2, comps=18, groups=54, states=2, init=.05, floor=1e-3, eps=.003, eps_own=1., twin=False, credit='writer', teacher_trains='RG', own_trains='RG', readback=True, own=0., watch_first=0, set=0, trace=False,
            samples=262144, batch=64, n_evals=18, n_eval=4096, n_mmd=1024, n_first=256)


def arms(only):
    ws = dict(own=.5, teacher_trains='G', own_trains='R')                                       # classic wake-sleep
    a = dict(watching={}, ws=ws, ws_intended=dict(ws, readback=False), ws_twin=dict(ws, twin=True), watch_sleep=dict(own=.5, own_trains='R'),
             practice=dict(own=1.), practice_read=dict(own=1., credit='read'), mix=dict(own=.5), mix_read=dict(own=.5, credit='read'), mix_g=dict(own=.5, own_trains='G'))
    a['mix_intended'] = dict(own=.5, readback=False)
    for share in (.03125, .0625, .125, .25, .75, .875, .9375):                                  # own samples teach the reader only, teacher strings both
        a[f'sleep_own{share}'] = dict(own=share, own_trains='R')
    for share in (.125, .25, .75, .875):                                                        # budget: share of own samples at a fixed total
        a[f'mix_own{share}'] = dict(own=share)
        a[f'ws_own{share}'] = dict(ws, own=share)
    for first in (16384, 65536):
        a[f'first{first}_mix'] = dict(watch_first=first, own=.5, trace=True)                                # watching, then own samples in turn with teacher strings
        a[f'first{first}_mix_twin'] = dict(watch_first=first, own=.5, twin=True)
        a[f'first{first}_offline'] = dict(watch_first=first, own=1., trace=True)                            # watching, then own samples only
        a[f'first{first}_offline_intended'] = dict(watch_first=first, own=1., readback=False)   # ... without the first layer reading the own string back
        a[f'first{first}_offline_read'] = dict(watch_first=first, own=1., credit='read')        # ... credited by reading instead of to the writing units
        a[f'first{first}_offline_twin'] = dict(watch_first=first, own=1., twin=True)
    for n in (256, 1024, 4096):                                                                 # a stored teacher set, shown again and again
        a[f'set{n}_watching'] = dict(set=n)
        a[f'set{n}_mix'] = dict(set=n, own=.5)
        a[f'set{n}_ws'] = dict(ws, set=n)
    return {k: dict(BASE, **v) for k, v in a.items() if any(k == o or (o.endswith('*') and k.startswith(o[:-1])) for o in only)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--only', nargs='*', default=['*'])
    ap.add_argument('--set', nargs='*', default=[], help='key=value overrides for every arm')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    over = {k: json.loads(v) for k, v in (s.split('=') for s in args.set)}
    jobs = [(name, seed, dict(cfg, **over)) for name, cfg in arms(args.only).items() for seed in range(args.worlds)]
    with Pool(min(args.workers, len(jobs))) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs)))
    json.dump(dict(time=datetime.now(ZoneInfo('Europe/London')).isoformat(), results=results), open(out / 'results.json', 'w'))
    keys = ('valid', 'kl', 'coverage', 'own_read', 'teacher_trip', 'tokens_owned', 'tokens_shared', 'groups_free', 'group_pr', 'usage')
    for name in arms(args.only):
        rs = [r for r in results if r['name'] == name]
        med = lambda key, at: float(np.median([r['curve'][at][key] for r in rs]))
        print(f"{name:22s} " + '  '.join(f"{k} {med(k, 0):.2f}->{med(k, -1):.2f}" for k in keys) + f"  mmd {[round(float(np.median([r['curve'][-1]['mmd'][k] for r in rs])), 1) for k in rs[0]['curve'][0]['mmd']]}  {np.mean([r['seconds'] for r in rs]):.0f}s")


if __name__ == '__main__':
    main()
