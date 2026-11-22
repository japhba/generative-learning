"""Two learned levels on the order-three world with a second level of checks (World of order3_deep.py), check units of both levels given.

World: top bits -> mid bits (upper rings) -> tokens (lower rings); mid bits and top bits are tokens themselves, see order3_deep.py.
The first layer has check units for the token checks of both levels (wiring given, triples stored from the first teacher strings): 54
level-1 sites (the three tokens of a check in a lower ring) and 27 level-2 sites (three mid tokens of a check in an upper ring).

Learned, with winner-take-all groups of two units, weights = log conditional probabilities, separate reader and writer, same weak start:
  mid groups (54) - level-1 check units:   R1, G1 = P(level-1 check unit | mid unit)     (the network of order3_net.py)
  top groups (27) - level-2 check units:   R3, G3 = P(level-2 check unit | top unit)     (the same network one level up)
  level-2 check units -> mid groups:       G2 = P(mid unit | level-2 check unit)         (how the upper level writes into the lower one)
Writing runs through the whole stack: top groups draw from a uniform prior -> level-2 check units -> mid groups -> level-1 check units ->
tokens (a token two check units write takes one of the two values). Every draw is from softmax(sum over the groups above of log G - log of
the group's average). Reading: the check units of both levels read the string (a site whose triple is not allowed stays silent); mid groups
read the level-1 check units, top groups the level-2 check units.
Rule, the same everywhere: for an active upper unit, P <- P + eps * (active below - P) at the groups below that fired.

Pairs the rule is applied to:
  teacher string: (mid units as read, level-1 check units), (top units as read, level-2 check units), (level-2 check units, mid units as read).
  own sample:     (mid units as drawn, level-1 check units read back from the string), (top units as drawn, level-2 check units read back).
                  draws: 'top' = written from the top; 'mid' = mid units drawn independently, the upper level left out (teaches R1 only);
                  'both' = in turn; 'each' = in turn, and a sample teaches only the level at which its units were drawn independently
                  (from the top: R3, from the mid units: R1).
  twin:           a second network on the same schedule supplies the samples; both sides as this network reads them.
teacher_trains / own_trains: which of R and G a source moves (ours: teacher both, own samples the reader; classic wake-sleep: teacher G,
own R). The upper level learns with rate eps * eps_upper, R3 and G3 from `upper_from` samples on, G2 from `link_from` samples on.

    python order3_ladder.py --out /var/tmp/jbauer/generative_order3/ladder_main
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

from order3_deep import World, info
from order3_net import FirstLayer
from order3_practice import F, Surface


class Sites:
    """check units of the level-2 sites: one unit per triple a site showed in the first teacher strings."""

    def __init__(self, sites, strings):
        self.sites, self.S, self.V = sites, len(sites), 4
        combo = (strings[:, sites] * (4, 2, 1)).sum(2)                                          # [n, S]
        self.index = -np.ones((self.S, 8), int)
        self.vals = np.zeros((self.S, 4, 3), int)
        for s in range(self.S):
            seen = np.unique(combo[:, s])
            assert len(seen) == 4, 'a site has not shown all its triples yet'
            self.index[s, seen] = np.arange(4)
            self.vals[s] = np.stack([seen // 4, seen // 2 % 2, seen % 2], 1)

    def code(self, x):
        return self.index[np.arange(self.S), (x[:, self.sites] * (4, 2, 1)).sum(2)]


def onehot(h, n):
    out = np.zeros((len(h), h.shape[1] * n), F)
    i, g = np.nonzero(h >= 0)
    out[i, g * n + h[i, g]] = 1
    return out


def draw(rng, logits, n):
    lg = logits.reshape(len(logits), -1, n)
    return (lg + rng.gumbel(size=lg.shape)).argmax(2)


def hebb(P, up, n_up, lo, n_lo, eps):
    """P [upper units, lower units]: for every active upper unit and every lower group that fired, P <- P + eps * (active below - P)."""
    u1 = onehot(up, n_up)
    count = (u1.T @ (lo >= 0).astype(F)).astype(np.float64)
    keep = np.repeat((1 - eps) ** count, n_lo, 1)
    return keep * P + (1 - keep) * (u1.T @ onehot(lo, n_lo)).astype(np.float64) / np.repeat(np.maximum(count, 1), n_lo, 1)


class Net:
    def __init__(self, rng, first, second, cfg):
        self.f, self.s, self.cfg, (self.KM, self.KU) = first, second, cfg, cfg['groups']
        start = lambda up, groups, n: (lambda p: (p / p.sum(2, keepdims=True)).reshape(up, groups * n))(np.exp(cfg['init'] * rng.normal(size=(up, groups, n))))
        self.R1 = start(2 * self.KM, first.S, 4); self.G1 = self.R1.copy()
        self.R3 = start(2 * self.KU, second.S, 4); self.G3 = self.R3.copy()
        self.G2 = start(4 * second.S, self.KM, 2)

    def lift(self, P, n):
        """log P(lower unit | upper unit) relative to the upper group's average over its n units."""
        return (np.log(P + self.cfg['floor']) - np.repeat(np.log(P.reshape(-1, n, P.shape[1]).mean(1) + self.cfg['floor']), n, 0)).astype(F)

    def write(self, rng, top=None, mid=None):
        """from top units (or from mid units, leaving the upper level out) -> everything drawn on the way, tokens, check units of both levels read back."""
        h = dict(top=top, mid=mid)
        if mid is None:
            h['tau2'] = draw(rng, onehot(top, 2) @ self.lift(self.G3, 2), 4)
            h['mid'] = draw(rng, onehot(h['tau2'], 4) @ self.lift(self.G2, 4), 2)
        h['tau1'] = draw(rng, onehot(h['mid'], 2) @ self.lift(self.G1, 2), 4)
        x = self.f.render(rng, h['tau1'])
        return h, x, self.f.code(x), self.s.code(x)

    def read(self, rng, c1, c2):
        logw = lambda P: np.log(P + self.cfg['floor']).astype(F).T
        return draw(rng, onehot(c1, 4) @ logw(self.R1), 2), draw(rng, onehot(c2, 4) @ logw(self.R3), 2)


def evaluate(rng, world, net, cfg, surface):
    n, f, s = cfg['n_eval'], net.f, net.s
    h, x, c1, c2 = net.write(rng, top=rng.integers(2, size=(n, net.KU)))
    xt = world.teacher(rng, n)
    mid_t, top_t = net.read(rng, f.code(xt), s.code(xt))
    out = dict(mmd=surface.z(x), silent1=float((c1 < 0).mean()), silent2=float((c2 < 0).mean()))
    out['kl1'], out['cover1'], out['valid1'] = world.stats(x, world.pos)
    out['kl2'], out['cover2'], out['valid2'] = world.stats(x, world.midpos)
    out['checks2'] = float((x[:, world.sites2].sum(2) % 2 == 0).mean())
    out['kept2'] = float((x[:, s.sites] == s.vals[np.arange(s.S), h['tau2']]).mean())            # share of mid tokens that are what the level-2 check units drew
    tw, tr = world.targets(x), world.targets(xt)
    for name, target, wrote, readt in (('mid', 'mid', h['mid'], mid_t), ('top', 'top', h['top'], top_t)):
        for side, states, (tg, values) in (('write', wrote, tw[target]), ('read', readt, tr[target])):
            m = info(states, 2, tg, values)
            top2 = np.sort(m, 0)[::-1]
            out.update({f'{side}_{name}_cov': float((top2[0] > .8).mean()), f'{side}_{name}_dup': float((top2[1] > .8).mean()), f'{side}_{name}_free': float((m.max(1) < .2).mean())})
    # a teacher string read and written again from the top units, and from the mid units alone: share of tokens that come back
    out['trip_top'] = float((net.write(rng, top=top_t)[1] == xt).mean())
    out['trip_mid'] = float((net.write(rng, mid=mid_t)[1] == xt).mean())
    xm = net.write(rng, mid=rng.integers(2, size=(n, net.KM)))[1]                               # samples with independent mid units
    out['kl1_mid'] = world.stats(xm, world.pos)[0]
    return out


def run(job):
    name, seed, cfg = job
    t0 = time.time()
    world = World(seed, cfg['supers'], True)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    strings = world.teacher(np.random.default_rng([seed, 13]), cfg['n_first'])
    first, second = FirstLayer(world, strings), Sites(world.sites2, strings)
    net = Net(rng, first, second, cfg)
    twin = Net(np.random.default_rng([seed, 99]), first, second, cfg) if cfg['twin'] else None
    code3 = lambda sites: lambda x: (x[:, sites] * 2 ** np.arange(3)).sum(2)
    surface = Surface(np.random.default_rng([seed, 15]), world.teacher, {'single tokens': (lambda x: x, 2, 1), 'token pairs': (lambda x: x, 2, 2), 'level-1 checks': (code3(world.sites), 8, 1), 'level-2 checks': (code3(world.sites2), 8, 1)}, cfg['n_mmd'])
    bt, eps = cfg['batch'], cfg['eps']
    n_steps = cfg['samples'] // bt
    evals = set(np.unique(np.round(np.geomspace(1, n_steps, cfg['n_evals'])).astype(int)).tolist()) | {0, n_steps, cfg['watch_first'] // bt, cfg['upper_from'] // bt, cfg['link_from'] // bt}
    curve, n_own = [], 0

    def learn(m, which, mid, c1, top, c2, tau2, e, upper, link, lower=True):
        for name in which:
            if lower:
                setattr(m, name + '1', hebb(getattr(m, name + '1'), mid, 2, c1, 4, e))
            if upper and top is not None:
                setattr(m, name + '3', hebb(getattr(m, name + '3'), top, 2, c2, 4, e * cfg['eps_upper']))
        if link and 'G' in which and tau2 is not None:
            m.G2 = hebb(m.G2, tau2, 4, mid, 2, e * cfg['eps_upper'])

    def watch(m, which, upper, link):
        x = world.teacher(rng, bt)
        c1, c2 = first.code(x), second.code(x)
        mid, top = m.read(rng, c1, c2)
        learn(m, which, mid, c1, top, c2, c2, eps, upper, link)

    def practice(m, source, upper, link, k):
        from_mid = cfg['draws'] == 'mid' or cfg['draws'] in ('both', 'each') and k % 2
        lower = from_mid or cfg['draws'] != 'each'
        h, x, c1, c2 = source.write(rng, mid=rng.integers(2, size=(bt, source.KM))) if from_mid else source.write(rng, top=rng.integers(2, size=(bt, source.KU)))
        if source is m:
            learn(m, cfg['own_trains'], h['mid'], c1, h['top'], c2, h.get('tau2'), eps * cfg['eps_own'], upper, link, lower)
        else:
            mid, top = m.read(rng, c1, c2)
            learn(m, cfg['own_trains'], mid, c1, top if not from_mid else None, c2, c2, eps * cfg['eps_own'], upper, link, lower)

    for step in range(n_steps + 1):
        if step in evals:
            curve.append(dict(samples=step * bt, **evaluate(rng, world, net, cfg, surface)))
        if step == n_steps:
            break
        phase, upper, link = step * bt >= cfg['watch_first'], step * bt >= cfg['upper_from'], step * bt >= cfg['link_from']
        own = phase and n_own < cfg['own'] * (step + 1 - cfg['watch_first'] // bt)
        n_own += own
        for m in (net, twin) if twin else (net,):
            if not own:
                watch(m, cfg['teacher_trains'] if phase else 'RG', upper, link)
            elif m is twin or not twin:
                practice(m, m, upper, link, n_own)
            else:
                practice(net, twin, upper, link, n_own)
    return dict(name=name, seed=seed, cfg=cfg, curve=curve, seconds=time.time() - t0)


BASE = dict(supers=9, groups=(54, 27), init=.05, floor=1e-3, eps=.003, eps_own=1., eps_upper=1., upper_from=0, link_from=0, twin=False, teacher_trains='RG', own_trains='RG', draws='top', own=0., watch_first=0,
            samples=524288, batch=64, n_evals=20, n_eval=4096, n_mmd=1024, n_first=256)


def arms(only):
    ours, ws = dict(own=.5, own_trains='R'), dict(own=.5, teacher_trains='G', own_trains='R')
    a = dict(watching={}, ours=ours, ws=ws, ours_twin=dict(ours, twin=True), ws_twin=dict(ws, twin=True))
    for d in ('mid', 'both', 'each'):                                                                   # where the own samples start
        a[f'ours_{d}'] = dict(ours, draws=d)
        a[f'ws_{d}'] = dict(ws, draws=d)
    for late in (32768, 65536, 131072, 262144):                                                 # the upper level starts to learn late
        a[f'ours_late{late}'] = dict(ours, upper_from=late, link_from=late)
        a[f'ws_late{late}'] = dict(ws, upper_from=late, link_from=late)
        a[f'watching_late{late}'] = dict(upper_from=late, link_from=late)
    a['ours_link131072'] = dict(ours, link_from=131072)                                         # only the writing from the upper level into the lower one starts late
    a['ours_tops131072'] = dict(ours, upper_from=131072)                                        # only the top groups start late
    a['ours_late131072_twin'] = dict(ours, upper_from=131072, link_from=131072, twin=True)
    a['ours_each_twin'] = dict(ours, draws='each', twin=True)
    a['ours_each_rg'] = dict(own=.5, draws='each')                                              # own samples move the writer too
    for slow in (.3, .1, .03, .01):                                                             # ... or learns more slowly
        a[f'ours_slow{slow}'] = dict(ours, eps_upper=slow)
        a[f'ws_slow{slow}'] = dict(ws, eps_upper=slow)
    for first in (65536, 262144):                                                               # teacher strings alone first
        a[f'first{first}_ours'] = dict(ours, watch_first=first)
        a[f'first{first}_ours_both'] = dict(ours, watch_first=first, draws='both')
        a[f'first{first}_ours_mid'] = dict(ours, watch_first=first, draws='mid')
        a[f'first{first}_ours_each'] = dict(ours, watch_first=first, draws='each')
        a[f'first{first}_ws_each'] = dict(ws, watch_first=first, draws='each')
        a[f'first{first}_ws'] = dict(ws, watch_first=first)
        a[f'first{first}_ws_both'] = dict(ws, watch_first=first, draws='both')
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
    keys = ('kl1', 'kl2', 'valid1', 'checks2', 'read_mid_cov', 'read_mid_dup', 'read_top_cov', 'read_top_dup', 'trip_top')
    for name in arms(args.only):
        rs = [r for r in results if r['name'] == name]
        med = lambda key, at: float(np.median([r['curve'][at][key] for r in rs]))
        print(f"{name:20s} " + '  '.join(f"{k} {med(k, 0):.2f}->{med(k, -1):.2f}" for k in keys) + f"  {np.mean([r['seconds'] for r in rs]):.0f}s")


if __name__ == '__main__':
    main()
