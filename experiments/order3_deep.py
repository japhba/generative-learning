"""Order-three world with a second level of checks, learner = a stack of winner-take-all groups (the network of order3_net.py, deeper).

World (binary tokens). Three top bits (a, b, c) per upper ring; the ring turns them into six mid bits (a, a+b, b, b+c, c, c+a). Every lower
ring takes three mid bits from three different upper rings as its tokens t0, t2, t4 and adds t1 = t0+t2, t3 = t2+t4, t5 = t4+t0. Mid bits and
top bits are therefore tokens themselves; single tokens and token pairs are uniform. Two kinds of checks: the token checks of a lower ring
(level 1, the sites of the first layer) and the checks between the mid bits of an upper ring (level 2: three tokens from three lower rings).
`upper=False` leaves the mid bits free, which is the two-level world of order3_net.py.

Network: tokens - check units (wiring given, triples stored from the first teacher strings) - layers of winner-take-all groups, all-to-all
between neighbouring layers; by default 54 groups of 2 - 27 groups of 4 - 27 groups of 2, enough for mid bits, level-2 checks and top bits,
but nothing says which group is for which. Between an upper unit and a lower unit the weight is log P(lower unit | upper unit), separate
arrays for reading (R) and writing (G), same weak start.
  writing: the deepest groups draw from a uniform prior; going down, a group draws its unit from softmax(sum over the groups above of
           G - log of the group's average); check units write their triples (a token two check units write takes one of the two values).
  reading: going up, a group draws its unit from softmax(sum over the active units below of R).
Rule (two factors, the same at every layer and for R and G): for an active upper unit, P <- P + eps * (active below - P).

Pairs the rule is applied to:
  teacher string: every layer as read.
  own sample:     upper = the units that wrote (credit 'writer') or the reading of the own string (credit 'read'); lower = the units below
                  as drawn, at the first layer what the check units read back from the string (readback; 'all': the reading of the own
                  string at every layer).
  twin:           a second network on the same schedule from another start supplies the samples (upper and lower = this network's reading).
teacher_trains / own_trains: which of R and G a source moves. Ours: teacher strings move both, own samples the reader. Classic wake-sleep:
teacher strings move G only, own samples R only. own_levels restricts the own-sample rule to some layers; own samples are used from
`watch_first` to `own_stop` samples. Further switches: layer_from (samples at which a layer starts to learn), eps_layers (rate of a layer
relative to eps), theta (a group passes its unit upward only if its probability is above theta), draws ('read': the deepest units of an
own sample are the reading of a fresh teacher string instead of independent draws; used with one learned layer, arms shallow_*).
With the default three layers the stack does not hold: see order3_ladder.py for two learned levels with given check units.

    python order3_deep.py --out /var/tmp/jbauer/generative_order3/deep_main
"""
import argparse
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

from order3_net import LOCAL, FirstLayer
from order3_practice import F, Surface

ring = lambda b: np.stack([b[..., 0], b[..., 0] ^ b[..., 1], b[..., 1], b[..., 1] ^ b[..., 2], b[..., 2], b[..., 2] ^ b[..., 0]], -1)
VALID = (ring(np.array(np.unravel_index(np.arange(8), (2, 2, 2))).T) * 2 ** np.arange(6)).sum(1)


class World:
    def __init__(self, seed, supers, upper=True):
        rng = np.random.default_rng([seed, 11])
        self.U, self.C, self.q, self.upper = supers, 2 * supers, 2, upper
        self.L = 6 * self.C
        self.pos = rng.permutation(self.L).reshape(self.C, 6)
        self.sites = self.pos[:, LOCAL].reshape(-1, 3)                                          # level-1 checks, lower ring by lower ring
        while True:                                                                             # the three free tokens of a lower ring come from three different upper rings
            self.assign = rng.permutation(6 * supers).reshape(self.C, 3)
            if all(len(set(a)) == 3 for a in (self.assign // 6).tolist()):
                break
        where = np.zeros(6 * supers, int)
        where[self.assign.ravel()] = self.pos[:, ::2].ravel()
        self.midpos = where.reshape(supers, 6)                                                  # position of the token that carries each mid bit
        self.sites2 = self.midpos[:, LOCAL].reshape(-1, 3)                                      # level-2 checks as token triples

    def teacher(self, rng, n):
        mid = ring(rng.integers(2, size=(n, self.U, 3))).reshape(n, -1) if self.upper else rng.integers(2, size=(n, 6 * self.U))
        x = np.zeros((n, self.L), int)
        x[:, self.pos] = ring(mid[:, self.assign])
        return x

    @staticmethod
    def stats(x, pos):
        """for rings at positions pos [R, 6]: sum over rings of KL(teacher || learner) from the sample, share of valid rings the sample shows, share of valid rings."""
        n = len(x)
        cfg = (x[:, pos] * 2 ** np.arange(6)).sum(2)
        counts = np.stack([np.bincount(cfg[:, c], minlength=64)[VALID] for c in range(len(pos))])
        return float(np.sum(np.mean(-np.log(8) - np.log(np.maximum(counts / n, 1 / (2 * n))), 1))), float((counts >= n / 32).mean()), float(counts.sum() / (n * len(pos)))

    def targets(self, x):
        """what the layers should carry, read off the tokens: mid bits [N, 6 U], level-2 checks as pairs of mid bits [N, 3 U], top bits [N, 3 U]."""
        m = x[:, self.midpos]                                                                   # [N, U, 6]
        return dict(mid=(m.reshape(len(x), -1), 2), check=((2 * m[:, :, ::2] + m[:, :, (2, 4, 0)]).reshape(len(x), -1), 4), top=(m[:, :, ::2].reshape(len(x), -1), 2))


class Net:
    def __init__(self, rng, first, cfg):
        self.f, self.cfg = first, cfg
        self.shape = [(first.S, first.V)] + [tuple(l) for l in cfg['layers']]                   # (groups, units per group); layer 0 = check units
        self.depth = len(self.shape) - 1
        self.R = [None] + [self.start(rng, up, lo) for lo, up in zip(self.shape[:-1], self.shape[1:])]
        self.G = [None] + [r.copy() for r in self.R[1:]]

    def start(self, rng, up, lo):
        p = np.exp(self.cfg['init'] * rng.normal(size=(up[0] * up[1], lo[0], lo[1])))
        return (p / p.sum(2, keepdims=True)).reshape(up[0] * up[1], lo[0] * lo[1])

    def logw(self, P, l, ratio=False):
        """log P(lower unit | upper unit); for writing, relative to the upper group's average over its units."""
        K, n = self.shape[l]
        w = np.log(P + self.cfg['floor'])
        return (w - (np.repeat(np.log(P.reshape(K, n, -1).mean(1) + self.cfg['floor']), n, 0) if ratio else 0)).astype(F)

    def onehot(self, h, l):
        K, n = self.shape[l]
        out = np.zeros((len(h), K * n), F)
        i, g = np.nonzero(h >= 0)
        out[i, g * n + h[i, g]] = 1
        return out

    def draw(self, rng, logits, l):
        lg = logits.reshape(len(logits), *self.shape[l])
        return (lg + rng.gumbel(size=lg.shape)).argmax(2)

    def read(self, rng, code, upto=None):
        """check units as read from a string (-1 silent) -> for every layer up to `upto` the units drawn, and what the layer passes on:
        the unit drawn if its probability is above theta, nothing (-1) otherwise."""
        h, out = [code], [code]
        for l in range(1, (upto or self.depth) + 1):
            lg = (self.onehot(out[-1], l - 1) @ self.logw(self.R[l], l).T).reshape(len(code), *self.shape[l])
            s = (lg + rng.gumbel(size=lg.shape)).argmax(2)
            p = np.exp(lg - lg.max(2, keepdims=True))
            h.append(s); out.append(np.where(np.take_along_axis(p, s[..., None], 2)[..., 0] / p.sum(2) > self.cfg['theta'], s, -1))
        return h, out

    def write(self, rng, z, top=None):
        """units z of layer `top` (default: the deepest) -> units of every layer below as drawn (dict), tokens, check units read back."""
        top = top or self.depth
        h = {top: z}
        for l in range(top, 0, -1):
            h[l - 1] = self.draw(rng, self.onehot(h[l], l) @ self.logw(self.G[l], l, ratio=True), l - 1)
        x = self.f.render(rng, h[0])
        return h, x, self.f.code(x)

    def prior(self, rng, n):
        return rng.integers(self.shape[-1][1], size=(n, self.shape[-1][0]))

    def hebb(self, which, l, up, lo, eps):
        """for every active unit of layer l and every group below that fired: P <- P + eps * (active below - P), sample after sample."""
        nl = self.shape[l - 1][1]
        u1, l1 = self.onehot(up, l), self.onehot(lo, l - 1)
        count = (u1.T @ (lo >= 0).astype(F)).astype(np.float64)
        keep = np.repeat((1 - eps) ** count, nl, 1)
        mean = (u1.T @ l1).astype(np.float64) / np.repeat(np.maximum(count, 1), nl, 1)
        for name in which:
            getattr(self, name)[l] = keep * getattr(self, name)[l] + (1 - keep) * mean


def info(a, na, b, nb):
    """mutual information between every group of a [N, A] (na units) and every variable of b [N, B] (nb values), in units of log nb: [A, B]."""
    one = lambda v, n: (v[:, :, None] == np.arange(n)).reshape(len(v), -1).astype(F)
    j = (one(a, na).T @ one(b, nb)).astype(np.float64).reshape(a.shape[1], na, b.shape[1], nb) / len(a)
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(j > 0, j * np.log(j / (j.sum(3, keepdims=True) * j.sum(1, keepdims=True))), 0).sum((1, 3)) / np.log(nb)


def evaluate(rng, world, first, net, cfg, surface, final=False):
    n = cfg['n_eval']
    h, x, code = net.write(rng, net.prior(rng, n))
    xt = world.teacher(rng, n)
    ct = first.code(xt)
    r, passed = net.read(rng, ct)
    out = dict(silent=float((code < 0).mean()), mmd=surface.z(x))
    out['kl1'], out['cover1'], out['valid1'] = world.stats(x, world.pos)
    out['kl2'], out['cover2'], out['valid2'] = world.stats(x, world.midpos)
    out['checks2'] = float((x[:, world.sites2].sum(2) % 2 == 0).mean())
    tw, tr = world.targets(x), world.targets(xt)
    for l, name in zip(range(1, net.depth + 1), cfg['targets']):
        units = net.shape[l][1]
        for side, states, (target, values) in (('write', h[l], tw[name]), ('read', r[l], tr[name])):
            m = info(states, units, target, values)                                             # [groups, targets]
            top = np.sort(m, 0)[::-1]
            out.update({f'{side}{l}_cov': float((top[0] > .8).mean()), f'{side}{l}_dup': float((top[1] > .8).mean()), f'{side}{l}_best': float(top[0].mean()),
                        f'{side}{l}_free': float((m.max(1) < .2).mean()), f'{side}{l}_part': float(((m.max(1) >= .2) & (m.max(1) <= .8)).mean())})
            if final:
                out[f'{side}{l}_info'] = np.round(m, 2).tolist()
        out[f'fire{l}'] = float((passed[l] >= 0).mean())
        out[f'usage{l}'] = float(np.mean(np.abs(np.stack([(r[l] == k).mean(0) for k in range(units)]) - 1 / units).max(0)))
        # a string read up to layer l and written again from there: share of tokens that come back
        out[f'trip_t{l}'] = float((net.write(rng, r[l], top=l)[1] == xt).mean())
        out[f'trip_o{l}'] = float((net.write(rng, net.read(rng, code, upto=l)[0][l], top=l)[1] == x).mean())
    return out


def run(job):
    name, seed, cfg = job
    t0 = time.time()
    world = World(seed, cfg['supers'], cfg['upper'])
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    first = FirstLayer(world, world.teacher(np.random.default_rng([seed, 13]), cfg['n_first']))
    net = Net(rng, first, cfg)
    twin = Net(np.random.default_rng([seed, 99]), first, cfg) if cfg['twin'] else None
    code3 = lambda sites: lambda x: (x[:, sites] * 2 ** np.arange(3)).sum(2)
    surface = Surface(np.random.default_rng([seed, 15]), world.teacher, {'single tokens': (lambda x: x, 2, 1), 'token pairs': (lambda x: x, 2, 2), 'level-1 checks': (code3(world.sites), 8, 1), 'level-2 checks': (code3(world.sites2), 8, 1)}, cfg['n_mmd'])
    bt, eps = cfg['batch'], cfg['eps']
    n_steps = cfg['samples'] // bt
    evals = set(np.unique(np.round(np.geomspace(1, n_steps, cfg['n_evals'])).astype(int)).tolist()) | {0, n_steps, cfg['watch_first'] // bt, cfg['own_stop'] // bt}
    levels = cfg['own_levels'] or range(1, net.depth + 1)
    curve, n_own = [], 0

    def watch(m, which, on):
        h, passed = m.read(rng, first.code(world.teacher(rng, bt)))
        for l in on:
            m.hebb(tuple(which), l, h[l], passed[l - 1], eps * cfg['eps_layers'][l - 1])

    def practice(m, source, on):
        # the deepest units of an own sample: independent draws, or (draws='read') the reading of a fresh teacher string, which is what a finished level above would hand down
        h, x, code = source.write(rng, source.prior(rng, bt) if cfg['draws'] == 'prior' else source.read(rng, first.code(world.teacher(rng, bt)))[0][-1])
        low = code if cfg['readback'] else h[0]
        if cfg['credit'] == 'writer' and source is m:
            up, lo = h, m.read(rng, code)[1] if cfg['readback'] == 'all' else {**h, 0: low}
        else:
            up, lo = m.read(rng, low)
        for l in on:
            if l in levels:
                m.hebb(tuple(cfg['own_trains']), l, up[l], lo[l - 1], eps * cfg['eps_own'] * cfg['eps_layers'][l - 1])

    for step in range(n_steps + 1):
        if step in evals:
            curve.append(dict(samples=step * bt, **evaluate(rng, world, first, net, cfg, surface, final=step == n_steps)))
        if step == n_steps:
            break
        phase = cfg['watch_first'] <= step * bt < (cfg['own_stop'] or cfg['samples'])           # own samples make up the share cfg['own'] inside this window
        own = phase and n_own < cfg['own'] * (step + 1 - cfg['watch_first'] // bt)
        n_own += own
        on = [l for l in range(1, net.depth + 1) if step * bt >= cfg['layer_from'][l - 1]]         # a layer learns from its starting time on
        for m in (net, twin) if twin else (net,):
            if not own:
                watch(m, cfg['teacher_trains'] if phase else 'RG', on)
            elif m is twin or not twin:
                practice(m, m, on)
            else:
                practice(net, twin, on)
    return dict(name=name, seed=seed, cfg=cfg, curve=curve, seconds=time.time() - t0)


BASE = dict(supers=9, upper=True, layers=((54, 2), (27, 4), (27, 2)), targets=('mid', 'check', 'top'), init=.05, floor=1e-3, eps=.003, eps_own=1., twin=False, credit='writer', teacher_trains='RG', own_trains='RG',
            readback=True, own=0., watch_first=0, own_stop=0, own_levels=None, theta=0., draws='prior', layer_from=(0, 0, 0), eps_layers=(1., 1., 1.), samples=1048576, batch=64, n_evals=22, n_eval=4096, n_mmd=1024, n_first=256)


def arms(only):
    ours, ws = dict(own=.5, own_trains='R'), dict(own=.5, teacher_trains='G', own_trains='R')
    a = dict(watching={}, ours=ours, ws=ws, mix=dict(own=.5), ours_twin=dict(ours, twin=True), ws_twin=dict(ws, twin=True), ours_all=dict(ours, readback='all'), ours_drawn=dict(ours, readback=False), ws_drawn=dict(ws, readback=False),
             ours_level1=dict(ours, own_levels=(1,)), ours_level23=dict(ours, own_levels=(2, 3)), ours_level12=dict(ours, own_levels=(1, 2)))
    for stop in (65536, 131072, 262144):                                                        # own samples only up to here, teacher strings alone afterwards
        a[f'ours_stop{stop}'] = dict(ours, own_stop=stop)
    for first in (65536, 262144):                                                               # teacher strings alone up to here
        a[f'first{first}_ours'] = dict(ours, watch_first=first)
        a[f'first{first}_ws'] = dict(ws, watch_first=first)
    for share in (.125, .25, .75, .875):
        a[f'ours_own{share}'] = dict(ours, own=share)
        a[f'ws_own{share}'] = dict(ws, own=share)
    a['shallow_watching'] = dict(layers=((54, 2),), targets=('mid',), layer_from=(0,), eps_layers=(1.,))                           # the two-layer network of order3_net.py on this world
    shallow = dict(layers=((54, 2),), targets=('mid',), layer_from=(0,), eps_layers=(1.,))
    a['shallow_ours'] = dict(ours, **shallow)
    a['shallow_ws'] = dict(ws, **shallow)
    a['shallow_ours_read'] = dict(ours, **shallow, draws='read')                                # own samples written from the reading of fresh teacher strings
    a['shallow_ws_read'] = dict(ws, **shallow, draws='read')
    a['shallow_first_ours'] = dict(ours, **shallow, watch_first=65536)
    a['shallow_first_ours_read'] = dict(ours, **shallow, watch_first=65536, draws='read')
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
    for name in arms(args.only):
        rs = [r for r in results if r['name'] == name]
        med = lambda key, at: float(np.median([r['curve'][at][key] for r in rs]))
        keys = ['kl1', 'kl2', 'valid1', 'checks2'] + [f'{s}{l}_{k}' for l in range(1, len(rs[0]['cfg']['layers']) + 1) for s, k in (('read', 'cov'), ('read', 'dup'))] + [f"trip_t{len(rs[0]['cfg']['layers'])}"]
        print(f"{name:18s} " + '  '.join(f"{k} {med(k, 0):.2f}->{med(k, -1):.2f}" for k in keys) + f"  {np.mean([r['seconds'] for r in rs]):.0f}s")


if __name__ == '__main__':
    main()
