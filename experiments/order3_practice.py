"""Practice judged by checks of order three: wide thresholded deep units, lesson per check, credit by reading alone.

World. S = 3 B slots (token positions) with q values (q prime). Block b is a triple of positions (scattered at random) with a + b + c = 0
(mod q): single tokens and token pairs are exactly uniform, the rule sits in the triples (order3_paper_checks.py). kind 'rule2' adds a
second level: blocks come in threes whose symbols s = a - b sum to zero, a rule of order six in tokens and of order three in the code of
the blocks. kind 'free' has no rule (the judge accepts every triple).

Judge. Check units are learned by coincidence from teacher strings: every triple of slots keeps which value combinations it has seen; a
triple with at least a third of its combinations unseen is a check site, and a sample passes the check if its combination has been seen.

Learner. K deep units; unit k has weights W[k, slot, value] >= floor on the slots it is wired to (all of them, or a random `fanin`), used
for writing and reading (tied). Writing: the logits of a slot are the summed weights of the active units, values are drawn from the
softmax. Reading: a unit's drive is sum_i w~[k, i, x_i] with w~ the weight above the smallest weight of that slot, as a share of the mean
drive of the unit's own samples; the unit with the largest drive (in its group) fires if the share is at least theta.
Start: every unit has one random preferred value per slot at weight w0 (sharp, random direction).

Practice (one unit at a time writes; lesson per check; credit by reading): for each unit that fires on the sample and each check, the
weights of the sampled values of the check's slots are multiplied by 1 + eta if the check passed and by 1 - lam if it failed (bounded
to [floor, wmax]). rule = 'hebb' has two factors only: the weights of the sampled values that a check unit confirmed are multiplied by
1 + eta, and every weight of a unit is multiplied by 1 - lam each time the unit fires. Watching uses the same reader and the same rule on teacher strings (all checks pass). 'store': the unit with the
largest drive takes a teacher string in one shot, no threshold. 'twin': the learner reads the samples of a second learner that practices.

Scores. kl: exact KL(teacher || learner) for the generator in which every check site is written by one unit drawn among the units that
are sharp there (units compete per check). kl_one_unit: the same if one unit, drawn uniformly, writes the whole string. mmd: unbiased
MMD^2 between samples and teacher strings with single-token, token-pair and check features, in standard deviations of the same
statistic between two teacher sets.

level = 2: the same learner on the code of the blocks (the finished level below, taken as given), judged by checks over three blocks.
groups > 1: groups of units, one unit per active group writes with summed logits, the number of active groups follows how well the
writers are read (an attempt at units that share the string; it does not work).

    python order3_practice.py --out /var/tmp/jbauer/generative_order3/run_main
    python order3_practice.py --out /var/tmp/jbauer/generative_order3/run_few --only few
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

F = np.float32


def onehot(x, v):
    """[N, S] values -> [N, S * v] float32."""
    out = np.zeros((x.shape[0], x.shape[1] * v), F)
    out[np.arange(len(x))[:, None], np.arange(x.shape[1]) * v + x] = 1
    return out


class World:
    def __init__(self, seed, blocks, q, kind):
        rng = np.random.default_rng([seed, 11])
        self.B, self.q, self.kind, self.L = blocks, q, kind, 3 * blocks
        self.pos = rng.permutation(self.L).reshape(blocks, 3)                 # token positions of each block
        self.sup = rng.permutation(blocks).reshape(blocks // 3, 3) if kind == 'rule2' else None
        self.entropy = np.log(q) * dict(free=3 * blocks, rule=2 * blocks, rule2=5 * blocks / 3)[kind]

    def teacher(self, rng, n):
        q = self.q
        a, b, c = (rng.integers(q, size=(n, self.B)) for _ in range(3))
        if self.kind == 'rule2':
            s = rng.integers(q, size=(n, self.B))
            s[:, self.sup[:, 2]] = (-s[:, self.sup[:, 0]] - s[:, self.sup[:, 1]]) % q
            b = (a - s) % q
        if self.kind != 'free':
            c = (-a - b) % q
        x = np.zeros((n, self.L), int)
        x[:, self.pos[:, 0]], x[:, self.pos[:, 1]], x[:, self.pos[:, 2]] = a, b, c
        return x

    def code(self, x):
        """tokens -> code of each block (a q + b), -1 where the block's triple breaks the rule."""
        a, b, c = (x[:, self.pos[:, k]] for k in range(3))
        return np.where((a + b + c) % self.q == 0, a * self.q + b, -1)

    def decode(self, u):
        a, b = u // self.q, u % self.q
        x = np.zeros((len(u), self.L), int)
        x[:, self.pos[:, 0]], x[:, self.pos[:, 1]], x[:, self.pos[:, 2]] = a, b, (-a - b) % self.q
        return x


class Judge:
    """check units over triples of slots, learned by coincidence from the strings shown."""

    def __init__(self, strings, v, accept_all=False, sites=None):
        n, s = strings.shape
        trip = np.array(list(itertools.combinations(range(s), 3))) if sites is None else np.asarray(sites)
        seen = np.zeros((len(trip), v ** 3), bool)
        step = max(1000, 4_000_000 // n)
        for lo in range(0, len(trip), step):
            t = trip[lo:lo + step]
            combo = strings[:, t[:, 0]] * v * v + strings[:, t[:, 1]] * v + strings[:, t[:, 2]]             # [n, T]
            seen[lo:lo + step] = np.bincount((np.arange(len(t)) * v ** 3 + combo).ravel(), minlength=len(t) * v ** 3).reshape(len(t), -1) > 0
        site = (~seen).mean(1) >= 1 / 3 if sites is None else np.ones(len(trip), bool)
        self.sites, self.allowed, self.v, self.s = trip[site], seen[site], v, s
        if accept_all:
            self.allowed[:] = True
        self.n_strings = n

    def passed(self, x):
        """[N, C] which checks fire on the strings."""
        t = self.sites
        return self.allowed[np.arange(len(t)), x[:, t[:, 0]] * self.v ** 2 + x[:, t[:, 1]] * self.v + x[:, t[:, 2]]]

    def sign(self, x):
        """[N, S] +1 where every check of the slot fired, -1 where one failed, 0 where the slot is in no check."""
        ok = self.passed(x)
        fail, has = np.zeros(x.shape, bool), np.zeros(x.shape[1], bool)
        for k in range(3):
            np.logical_or.at(fail, (slice(None), self.sites[:, k]), ~ok)
            has[self.sites[:, k]] = True
        return np.where(fail, -1, 1) * has


class Learner:
    def __init__(self, rng, s, v, groups, n, cfg):
        self.S, self.V, self.G, self.n, self.K, self.cfg = s, v, groups, n, groups * n, cfg
        fan = cfg['fanin'] or s                                                                 # each unit is wired to a random set of slots (all of them if fanin = 0)
        self.mask = np.zeros((self.K, s), bool)
        self.mask[np.arange(self.K)[:, None], np.argsort(rng.random((self.K, s)), 1)[:, :fan]] = True
        self.W = np.full((self.K, s, v), cfg['floor'], F)
        pref = rng.integers(v, size=(self.K, s))
        k, i = np.nonzero(self.mask)
        self.W[k, i, pref[k, i]] = cfg['w0']

    def write(self, rng, units):
        """units [N, G] index of the active unit within each group (-1: group silent) -> strings [N, S]."""
        act = np.zeros((len(units), self.K), F)
        r, g = np.nonzero(units >= 0)
        act[r, g * self.n + units[r, g]] = 1
        logits = (act @ self.W.reshape(self.K, -1)).reshape(len(units), self.S, self.V)
        return (logits + rng.gumbel(size=logits.shape)).argmax(2), act

    def drive(self, x1):
        """[N, K] drive of every unit, as a share of the mean drive of the unit's own samples (written alone); and which units have a pattern at all."""
        wt = self.W - self.W.min(2, keepdims=True)
        p = np.exp(self.W - self.W.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
        norm = (p * wt).sum((1, 2))
        return (x1 @ wt.reshape(self.K, -1).T) / np.maximum(norm, 1e-9), norm >= self.cfg['norm_min'] * self.cfg['wmax']

    def read(self, x1, theta=None):
        """[N, K] float 0/1: in each group the unit with the largest drive fires if the drive reaches the threshold."""
        rel, able = self.drive(x1)
        rel = np.where(able, rel, -1.).reshape(len(x1), self.G, self.n)
        best = rel.argmax(2)
        fire = np.zeros_like(rel, dtype=F)
        r, g = np.nonzero(np.take_along_axis(rel, best[..., None], 2)[..., 0] >= (self.cfg['theta'] if theta is None else theta))
        fire[r, g, best[r, g]] = 1
        return fire.reshape(len(x1), self.K)

    def lesson(self, x1, fire, sign, eta, lam):
        gain = (sign == 1).astype(F)
        if self.cfg['surprise']:                                                                # reinforce a passed value only as far as the units that fired did not predict it
            logits = (fire @ self.W.reshape(self.K, -1)).reshape(len(x1), self.S, self.V)
            p = np.exp(logits - logits.max(2, keepdims=True)); p /= p.sum(2, keepdims=True)
            gain = gain * (1 - (p.reshape(len(x1), -1) * x1).reshape(len(x1), self.S, self.V).sum(2))
        up = fire.T @ (x1 * np.repeat(gain, self.V, 1))
        down = fire.T @ (x1 * np.repeat(sign == -1, self.V, 1))
        if self.cfg['rule'] == 'hebb':                                                          # two factors: every synapse of a unit decays when the unit fires; `sign` only says which tokens a check unit confirmed
            down = np.repeat(fire.sum(0)[:, None], self.S * self.V, 1)
        self.W = np.where(self.mask[:, :, None], np.clip(self.W * np.exp(up * np.log1p(eta) + down * np.log1p(-lam)).reshape(self.W.shape).astype(F), self.cfg['floor'], self.cfg['wmax']), F(self.cfg['floor']))

    def store(self, x, theta):
        """one string at a time: the unit with the largest drive takes the string in one shot (if its drive reaches theta)."""
        x1 = onehot(x, self.V)
        for i in range(len(x)):
            rel, _ = self.drive(x1[i:i + 1])
            k = int(rel[0].argmax())
            if rel[0, k] >= theta:
                self.W[k] = self.cfg['floor']
                self.W[k, np.nonzero(self.mask[k])[0], x[i][self.mask[k]]] = self.cfg['wmax']

    def log_prob(self, rng, x):
        """log probability of strings under 'one unit per group, drawn uniformly, writes': exact for one group; for several a lower bound
        (factorized posterior over the groups' units, two coordinate sweeps, then four draws)."""
        x1 = onehot(x, self.V)
        if self.G == 1:
            logp = self.W - logsumexp(self.W, axis=2, keepdims=True)
            return logsumexp(x1 @ logp.reshape(self.K, -1).T.astype(F), axis=1) - np.log(self.K)
        n_, idx = len(x), np.arange(self.S)
        ll_of = lambda logits: np.take_along_axis(logits - logsumexp(logits, axis=-1, keepdims=True), x.reshape((n_,) + (1,) * (logits.ndim - 3) + (self.S, 1)), -1)[..., 0].sum(-1)
        rel, _ = self.drive(x1)
        z = rel.reshape(n_, self.G, self.n).argmax(2)
        wg = self.W.reshape(self.G, self.n, self.S, self.V)
        tot = wg[np.arange(self.G), z].sum(1)
        q = np.zeros((n_, self.G, self.n))
        for sweep in range(2):
            for g in range(self.G):
                base = tot - wg[g, z[:, g]]
                ll = ll_of(base[:, None] + wg[g][None])                                         # [N, n]
                q[:, g] = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
                z[:, g] = ll.argmax(1)
                tot = base + wg[g, z[:, g]]
        kl = (q * (np.log(np.maximum(q, 1e-300)) + np.log(self.n))).sum((1, 2))
        draws = []
        for _ in range(4):
            zd = (q.cumsum(2) < rng.random((n_, self.G, 1))).sum(2).clip(max=self.n - 1)
            draws.append(ll_of(wg[np.arange(self.G), zd].sum(1)))
        return np.mean(draws, 0) - kl

    def holders(self, sites):
        """[K, C] units with a sharp value on every slot of a check site."""
        wt = self.W - self.W.min(2, keepdims=True)
        return (wt.max(2) >= .5 * self.cfg['wmax'])[:, sites].all(2)

    def write_sites(self, rng, sites, n):
        """strings in which every check site is written by one unit, drawn among the units that hold the site (no holder: uniform values)."""
        hold, x = self.holders(sites), rng.integers(self.V, size=(n, self.S))
        for c, site in enumerate(sites):
            ks = np.nonzero(hold[:, c])[0]
            if len(ks):
                logits = self.W[ks[rng.integers(len(ks), size=n)]][:, site]
                x[:, site] = (logits + rng.gumbel(size=logits.shape)).argmax(2)
        return x

    def log_prob_sites(self, x, sites):
        """exact log probability of strings under write_sites (every slot is in exactly one site)."""
        logp, hold, out = self.W - logsumexp(self.W, axis=2, keepdims=True), self.holders(sites), np.zeros(len(x))
        for c, site in enumerate(sites):
            ks = np.nonzero(hold[:, c])[0]
            out += logsumexp(sum(logp[ks][:, i, x[:, i]] for i in site), axis=0) - np.log(len(ks)) if len(ks) else -3 * np.log(self.V)
        return out

    def ownership(self, blocks_of, valid):
        """blocks_of [B, 3] slots of each check; valid(values [.., 3]) -> bool. For each block the number of units per group that are sharp on a valid
        triple there; returns the share of the leading group among holders, its share of units that hold, and the blocks led per group."""
        wt = self.W - self.W.min(2, keepdims=True)
        sharp = wt.max(2) >= .5 * self.cfg['wmax']                                               # [K, S]
        top = self.W.argmax(2)
        hold = sharp[:, blocks_of].all(2) & valid(top[:, blocks_of])                             # [K, B]
        per = hold.reshape(self.G, self.n, -1).sum(1)                                            # [G, B]
        lead = per.argmax(0)
        return dict(holders=float(per.sum(0).mean()), purity=float((per.max(0) / np.maximum(per.sum(0), 1)).mean()), lead_cover=float(per.max(0).mean() / self.n),
                    blocks_per_group=np.bincount(np.bincount(lead[per.max(0) > 0], minlength=self.G), minlength=6)[:6].tolist(), fields=float(hold.sum(1).mean()))


class Surface:
    """unbiased MMD^2 between own and teacher samples for feature maps given as (categorical index arrays, order): the features are the indicators
    of all order-tuples of the indexed slots, so the kernel is binom(number of agreeing slots, order). Reported in units of the standard deviation
    of the same statistic between two teacher sets of the same size."""

    def __init__(self, rng, draw, maps, n, n_null=8):
        self.maps, self.n = maps, n
        ref = [draw(rng, n) for _ in range(n_null + 1)]
        self.ref = {name: f(ref[0]) for name, (f, v, order) in maps.items()}
        self.null = {}
        for name, (f, v, order) in maps.items():
            fr = [f(r) for r in ref]
            self.null[name] = float(np.std([self.stat(fr[i], fr[i + 1], v, order) for i in range(n_null)]))

    @staticmethod
    def stat(a, b, v, order):
        a1, b1 = onehot(a, v), onehot(b, v)
        k = lambda m: m if order == 1 else m * (m - 1) / 2
        kaa, kbb, kab = k(a1 @ a1.T), k(b1 @ b1.T), k(a1 @ b1.T)
        na, nb = len(a), len(b)
        return float((kaa.sum() - np.trace(kaa)) / (na * (na - 1)) + (kbb.sum() - np.trace(kbb)) / (nb * (nb - 1)) - 2 * kab.mean())

    def z(self, own):
        return {name: self.stat(f(own[:self.n]), self.ref[name], v, order) / self.null[name] for name, (f, v, order) in self.maps.items()}


def evaluate(rng, world, judge, learner, cfg, space):
    n, sites = cfg['n_eval'], judge.sites
    allowed = lambda t: judge.allowed[np.arange(len(sites)), t[..., 0] * learner.V ** 2 + t[..., 1] * learner.V + t[..., 2]]
    units = np.full((n, learner.G), -1)
    units[:, 0] = rng.integers(learner.n, size=n)
    if learner.G > 1:                                                                           # several groups: every group has one active unit
        units = rng.integers(learner.n, size=(n, learner.G))
    x, act = learner.write(rng, units)                                                          # practice samples
    x1 = onehot(x, learner.V)
    fire = learner.read(x1)
    xt = space['teacher'](rng, n)
    fire_t = learner.read(onehot(xt, learner.V))
    hold = learner.holders(sites)                                                               # [K, C]
    good = hold & allowed(learner.W.argmax(2)[:, sites])
    inside = learner.mask[:, sites].all(2)
    sharp = (learner.W - learner.W.min(2, keepdims=True)).max(2) >= .5 * cfg['wmax']
    in_held = np.zeros_like(sharp)
    for k in range(3):
        np.logical_or.at(in_held, (slice(None), sites[:, k]), hold)
    xg = learner.write_sites(rng, sites, n)                                                     # the generator: units compete per check site
    ok, okg = judge.passed(x), judge.passed(xg)
    out = dict(practice_pass=float(ok.mean()), writer_fires=float((fire * act).sum() / act.sum()), other_fires=float((fire * (1 - act)).sum(1).mean()),
               teacher_fires=float(fire_t.any(1).mean()), teacher_units=float(fire_t.sum(1).mean()),
               check_pass=float(okg.mean()), all_pass=float(okg.all(1).mean()), kl=float(-learner.log_prob_sites(xt[:cfg['n_kl']], sites).mean() - space['entropy']),
               kl_one_unit=float(-learner.log_prob(rng, xt[:cfg['n_kl']]).mean() - space['entropy']),
               sites_held=float(hold.sum(1).mean()), held_valid=float(good.sum() / max(hold.sum(), 1)), inside_valid=float((good & inside).sum() / max(inside.sum(), 1)),
               sharp_outside=float((sharp & ~in_held).sum(1).mean()), holders_per_site=float(hold.sum(0).mean()), sites_without=float((good.sum(0) == 0).mean()),
               mmd_practice=space['surface'].z(space['to_tokens'](x)), mmd=space['surface'].z(space['to_tokens'](xg)))
    if learner.G > 1:
        out.update(learner.ownership(sites, allowed))
    return out


def setup(seed, cfg):
    """world, judge and the space the deep layer lives in (tokens, or the code of the blocks for the level above)."""
    world = World(seed, cfg['blocks'], cfg['q'], cfg['kind'])
    rng = np.random.default_rng([seed, 13])
    q, v = cfg['q'], cfg['q'] if cfg['level'] == 1 else cfg['q'] ** 2
    blockcfg = lambda x: (x[:, world.pos] * q ** np.arange(3)).sum(2)
    maps = {'single tokens': (lambda x: x, q, 1), 'token pairs': (lambda x: x, q, 2), 'token checks': (blockcfg, q ** 3, 1)}
    if cfg['level'] == 1:
        judge = Judge(world.teacher(rng, cfg['n_judge']), q, accept_all=cfg['kind'] == 'free', sites=world.pos if cfg['kind'] == 'free' else None)
        space = dict(teacher=world.teacher, to_tokens=lambda x: x, entropy=world.entropy, slots=world.L, values=q)
        truth = set(map(tuple, np.sort(world.pos, 1)))
    else:                                                                                       # the deep layer sits on the code of the finished level below
        judge = Judge(world.code(world.teacher(rng, cfg['n_judge'])), v)
        maps = dict(maps, **{'block pairs': (world.code, v, 2), 'block checks': (lambda x: (world.code(x)[:, world.sup] * v ** np.arange(3)).sum(2), v ** 3, 1)})
        space = dict(teacher=lambda r, m: world.code(world.teacher(r, m)), to_tokens=world.decode, entropy=world.entropy, slots=world.B, values=v)
        truth = set(map(tuple, np.sort(world.sup, 1)))
    space['surface'] = Surface(rng, world.teacher, maps, cfg['n_mmd'])
    found = set(map(tuple, np.sort(judge.sites, 1)))
    judge_info = dict(n_strings=judge.n_strings, sites_found=len(found), sites_true=len(truth), sites_exact=found == truth, allowed_share=float(judge.allowed.mean()))
    return world, judge, space, judge_info


def run(job):
    name, seed, cfg = job
    t0 = time.time()
    world, judge, space, judge_info = setup(seed, cfg)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    learner = Learner(rng, space['slots'], space['values'], cfg['groups'], cfg['units'] // cfg['groups'], cfg)
    twin = Learner(np.random.default_rng([seed, 99]), space['slots'], space['values'], cfg['groups'], cfg['units'] // cfg['groups'], cfg) if cfg['source'] == 'twin' else None
    bt, curve, m, read_ok = cfg['batch'], [], cfg['m0'], 1.
    n_steps = cfg['samples'] // bt
    evals = set(np.unique(np.round(np.geomspace(1, n_steps, cfg['n_evals'])).astype(int)).tolist()) | {0, n_steps}
    for step in range(n_steps + 1):
        if step in evals:
            curve.append(dict(samples=step * bt, coactive=m, **evaluate(rng, world, judge, learner, cfg, space)))
        if step == n_steps:
            break
        if cfg['source'] == 'own':
            units = np.full((bt, learner.G), -1)
            units[:, 0] = rng.integers(learner.n, size=bt)
            if learner.G > 1:                                                                   # each group is active with probability m / G; m follows how well the writers are read
                on = rng.random((bt, learner.G)) < m / learner.G
                on[np.arange(bt), rng.integers(learner.G, size=bt)] = True
                units = np.where(on, rng.integers(learner.n, size=(bt, learner.G)), -1)
            x, act = learner.write(rng, units)
            x1 = onehot(x, learner.V)
            fire = act if cfg['credit'] == 'writer' else learner.read(x1)
            learner.lesson(x1, fire, judge.sign(x), cfg['eta'], cfg['lam'])
            read_ok = .9 * read_ok + .1 * float((fire * act).sum() / act.sum())
            m = float(np.clip(m * (1 + cfg['ramp'] * ((read_ok > cfg['read_hi']) - (read_ok < cfg['read_lo']))), 1, learner.G))
        elif cfg['source'] == 'teacher':
            x = space['teacher'](rng, bt)
            x1 = onehot(x, learner.V)
            learner.lesson(x1, learner.read(x1), judge.sign(x), cfg['eta'], cfg['lam'])
        elif cfg['source'] == 'twin':                                                           # the twin practices on its own samples, the learner only reads them
            units = np.full((bt, twin.G), -1)
            units[:, 0] = rng.integers(twin.n, size=bt)
            x, act = twin.write(rng, units)
            x1 = onehot(x, twin.V)
            sign = judge.sign(x)
            twin.lesson(x1, twin.read(x1), sign, cfg['eta'], cfg['lam'])
            learner.lesson(x1, learner.read(x1), sign, cfg['eta'], cfg['lam'])
        elif cfg['source'] == 'store':                                                          # teacher strings, best unit takes the string in one shot, no threshold
            learner.store(space['teacher'](rng, bt), -1.)
    return dict(name=name, seed=seed, cfg=cfg, judge=judge_info, curve=curve, seconds=time.time() - t0)


BASE = dict(q=3, blocks=81, kind='rule', level=1, groups=1, units=1024, w0=4., wmax=6., floor=.05, theta=.7, norm_min=1.5, eta=1., lam=.2, source='own', credit='read',
            samples=262144, batch=64, n_evals=14, n_eval=2048, n_mmd=1024, n_kl=512, n_judge=256, surprise=False, m0=1., ramp=.02, read_hi=.8, read_lo=.6, fanin=0, rule='three')


def judge_scan(job):
    """how many teacher strings the coincidence rule needs to find exactly the check sites and all their allowed combinations."""
    seed, blocks, q = job
    out = []
    for level, kind, ns in ((1, 'rule', (16, 32, 64, 128, 256)), (2, 'rule2', (256, 512, 1024, 2048, 4096))):
        world, rng = World(seed, blocks, q, kind), np.random.default_rng([seed, 17])
        truth = set(map(tuple, np.sort(world.pos if level == 1 else world.sup, 1)))
        for n in ns:
            x = world.teacher(rng, n)
            judge = Judge(x if level == 1 else world.code(x), q if level == 1 else q * q)
            out.append(dict(seed=seed, level=level, strings=n, sites_exact=set(map(tuple, np.sort(judge.sites, 1))) == truth, allowed_exact=bool(len(judge.sites) and np.allclose(judge.allowed.mean(), 1 / q))))
    return out


def arms(only):
    a = {}
    a['practice'] = {}
    a['practice_writer_credit'] = dict(credit='writer')
    a['watching'] = dict(source='teacher')
    a['twin'] = dict(source='twin')
    a['store_teacher'] = dict(source='store', samples=8192)
    a['practice_free_world'] = dict(kind='free')
    lvl2 = dict(kind='rule2', level=2, n_judge=4096)
    a['level2_practice'] = dict(lvl2, samples=524288)
    a['level2_watching'] = dict(lvl2, source='teacher', samples=524288)
    a['level2_twin'] = dict(lvl2, source='twin')
    a['level2_store_teacher'] = dict(lvl2, source='store', samples=8192)
    for b in (3, 9, 27):
        a[f'width{b}_practice'] = dict(blocks=b, samples=131072)
        a[f'width{b}_watching'] = dict(blocks=b, source='teacher', samples=131072)
    small = dict(blocks=27, samples=131072, worlds=4)
    for eta, lam in itertools.product((.1, .3, 1., 3.), (.05, .2, .4, .6)):
        a[f'step_eta{eta}_lam{lam}'] = dict(small, eta=eta, lam=lam)
    for k in (256, 4096):
        a[f'units{k}_practice'] = dict(small, units=k, samples=128 * k)
    a['softstart_practice'] = dict(small, w0=2.)
    for k in (32, 64, 256):                                                                     # few wide units at the full width
        a[f'few{k}_practice'] = dict(units=k, samples=1024 * k, batch=max(8, k // 2))
    a['few64_watching'] = dict(units=64, source='teacher', samples=65536, batch=32)
    a['sparse_lam0.2'] = dict(small, fanin=27, lam=.2, theta=.75, samples=262144)                # each unit wired to a third of the slots
    a['sparse_lam0.4'] = dict(small, fanin=27, lam=.4, theta=.75, samples=262144)
    a['groups_coactive'] = dict(small, groups=54, units=972, theta=.55, surprise=True, eta=3., n_kl=128)
    hebb = dict(rule='hebb', lam=.05)                                                           # Hebb on the tokens the check units confirm, decay with the unit's own firing
    a['hebb_practice'] = dict(hebb)
    a['hebb_watching'] = dict(hebb, source='teacher')
    a['hebb_twin'] = dict(hebb, source='twin')
    a['hebb_free_world'] = dict(hebb, kind='free')
    a['hebb_level2_practice'] = dict(lvl2, rule='hebb', eta=3., lam=.02, samples=524288)
    a['hebb_level2_watching'] = dict(lvl2, rule='hebb', eta=3., lam=.02, source='teacher', samples=524288)
    for eta, lam in itertools.product((.3, 1., 3.), (.01, .03, .1, .3)):
        a[f'hebbstep_eta{eta}_lam{lam}'] = dict(small, rule='hebb', eta=eta, lam=lam)
    return {k: dict(BASE, **v) for k, v in a.items() if any(k.startswith(o) for o in only)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--worlds', type=int, default=8)
    ap.add_argument('--workers', type=int, default=28)
    ap.add_argument('--only', nargs='*', default=[''])
    ap.add_argument('--samples', type=int)
    ap.add_argument('--blocks', type=int)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jobs = [(name, seed, {k: v for k, v in dict(cfg, **({'samples': args.samples} if args.samples else {}), **({'blocks': args.blocks} if args.blocks and not name.startswith('width') else {})).items() if k != 'worlds'})
            for name, cfg in arms(args.only).items() for seed in range(min(args.worlds, cfg.get('worlds', args.worlds)))]
    jobs.sort(key=lambda j: -j[2]['samples'] * j[2]['blocks'])
    with Pool(args.workers) as pool:
        scan = sum(pool.map(judge_scan, [(seed, 81, BASE['q']) for seed in range(args.worlds)]), []) if args.only == [''] else []
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs)))
    json.dump(dict(time=datetime.now(ZoneInfo('Europe/London')).isoformat(), results=results, judge_scan=scan), open(out / 'results.json', 'w'))
    for level in (1, 2):
        print('judge, level', level, {n: f"{np.mean([s['sites_exact'] and s['allowed_exact'] for s in scan if s['level'] == level and s['strings'] == n]):.2f}" for n in sorted({s['strings'] for s in scan if s['level'] == level})})
    for name in arms(args.only):
        rs = [r for r in results if r['name'] == name]
        last = lambda key: float(np.median([r['curve'][-1].get(key, np.nan) for r in rs]))
        first = lambda key: float(np.median([r['curve'][0].get(key, np.nan) for r in rs]))
        print(f"{name:28s} check pass {first('check_pass'):.3f} -> {last('check_pass'):.3f}  all pass {last('all_pass'):.3f}  kl {first('kl'):7.1f} -> {last('kl'):7.1f}  one unit {first('kl_one_unit'):7.1f} -> {last('kl_one_unit'):7.1f}  "
              f"writer fires {last('writer_fires'):.2f}  teacher fires {first('teacher_fires'):.4f} -> {last('teacher_fires'):.4f} ({last('teacher_units'):.1f} units)  sites held {first('sites_held'):.1f} -> {last('sites_held'):.1f}  "
              f"sharp outside {first('sharp_outside'):.1f} -> {last('sharp_outside'):.1f}  judge exact {np.mean([r['judge']['sites_exact'] for r in rs]):.2f}  {np.mean([r['seconds'] for r in rs]):.0f}s")
        print('    mmd practice', {k: round(v, 1) for k, v in rs[0]['curve'][0]['mmd_practice'].items()}, '->', {k: round(v, 1) for k, v in rs[0]['curve'][-1]['mmd_practice'].items()}, ' generator', {k: round(v, 1) for k, v in rs[0]['curve'][-1]['mmd'].items()})


if __name__ == '__main__':
    main()
