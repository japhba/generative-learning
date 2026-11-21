"""First-layer units that share visible tokens: what the first layer's write-read loop teaches the layer above, in a two-layer writer and reader.

World 'ring'. A string has L tokens (Q values) on a ring. First-layer slot i covers positions (i, i+1); its units are the token pairs the
teacher uses there (V = Q * DEG per slot, every token has DEG successors). Neighbouring slots share a token, so a code (one unit per slot)
is consistent only if neighbours agree on it. The teacher writes a fraction F of the consistent strings, equally often (F = 1: the deep rule
is nothing but agreement on shared tokens; F < 1: part of the deep rule cannot be seen in any token pair).
World 'tree' (control). L / 2 slots with two tokens of their own; no shared tokens, every code is consistent; the teacher writes N_TREE of
the V^(L/2) codes.

Learner. First layer (taken as learned in Watching): one unit per token pair seen in teacher strings, reading it iff both tokens are there.
Writer layer 0: K deep units, logits W0[unit, slot, first-layer unit]; a deep unit draws one first-layer unit per slot (softmax); each
first-layer unit writes its two tokens; where two units write different tokens to one position the string shows one of them (1/2 each).
Reader: the first layer reads the string back (slot silent if its pair is no unit); deep unit a is driven by R0[a] . code / slots and the
best-driven unit fires if its drive is at least THETA (THETA = 0: the best unit always fires).
Rules. Wake at layer 0 (delta rule): dW0[a, slot] = ETA * g * (code read in the slot - softmax(W0[a, slot])). Instar on the reader:
dR0[a] = ETA_S * g * (code read - R0[a]). g is the verdict of the first layer: 1 if every slot was read, else 0 (a threshold unit on the
number of active first-layer units). A unit's share in the writer's prior is 1 + the number of teacher strings it fired on.
Arms = which samples, and which deep unit they are credited to:
  own / writer     own samples, credited to the unit that wrote the sample (sleep as routing), with the verdict ('accept'), without it
                   ('plain': g = 1, a silent slot has target zero) or with g = the writer's own drive by the sample ('drive')
  own / read       own samples, instar on the writer (sleep), wake on the unit that fires when the sample is read
  teacher, stored  fresh teacher strings, or P stored ones re-read; wake and instar on the unit that fires. With 'recruit' a string that
                   fires no unit is written into the unit that has fired least often (one-shot storage, a reader that needs no own samples)
  twin             the samples of a second learner (other start, own / writer / accept), read like teacher strings
  then_teacher     own / writer / accept for SWITCH rounds, then teacher strings read like any string (teacher_then_own: the other order)
  probe            codes with one first-layer unit per slot drawn uniformly, written and read back like own samples; credited to a random
                   deep unit with the verdict, or read like teacher strings
Starts of layer 0: 'sharp' (one random first-layer unit per slot at logit SCALE, reader coupled), 'diffuse' (all zero), or ('damage', d):
a consistent string with d slots replaced.
Scores are exact (2 x 2 transfer matrices over which of its two writers a position shows).
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import itertools
import json
import zlib
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

L, Q, DEG = 8, 4, 2
V = Q * DEG
N_TREE = 256
CLIP = 6.


def regular01(rng):
    while True:
        m = np.zeros((Q, Q), int)
        for _ in range(DEG):
            m[np.arange(Q), rng.permutation(Q)] += 1
        if m.max() == 1:
            return m


class World:
    def __init__(self, seed, kind, f):
        rng = np.random.default_rng(seed)
        self.kind, self.f = kind, f
        self.S = S = L if kind == 'ring' else L // 2
        self.idx = np.arange(S)
        allowed = np.stack([regular01(rng) for _ in range(S)])
        self.left, self.right = (np.stack([np.nonzero(allowed[i])[k] for i in range(S)]) for k in (0, 1))  # [S, V] tokens of each unit
        self.unit_of = -np.ones((S, Q, Q), int)
        for i in range(S):
            self.unit_of[i, self.left[i], self.right[i]] = np.arange(V)
        if kind == 'ring':
            strings = np.array(list(itertools.product(range(Q), repeat=L)))
            codes = self.unit_of[self.idx, strings, np.roll(strings, -1, 1)]
            self.cons = codes[(codes >= 0).all(1)]                       # codes of all consistent strings
            n_valid = max(1, round(f * len(self.cons)))
        else:
            self.cons = np.array(list(itertools.product(range(V), repeat=S)))
            n_valid = N_TREE
        while True:                                                       # the teacher's strings use every first-layer unit
            self.valid = np.sort(rng.choice(len(self.cons), n_valid, replace=False))
            if all(len(np.unique(self.cons[self.valid][:, i])) == V for i in range(S)):
                break
        self.is_valid = np.zeros(len(self.cons), bool)
        self.is_valid[self.valid] = True
        self.cons_id = {c.tobytes(): n for n, c in enumerate(self.cons)}
        self.n_all = Q ** L if kind == 'ring' else V ** S
        self.lmat = np.eye(Q)[self.left]                                  # [S, V, Q]
        self.rmat = np.eye(Q)[self.right]
        self.cons_left = self.left[self.idx, self.cons]                   # [N, S] left token of each slot's unit
        self.cons_right = self.right[self.idx, self.cons]

    def teacher(self, rng, n):
        return self.cons[rng.choice(self.valid, n)]

    def readback(self, rng, h):
        """codes [B, S] -> codes the first layer reads from the strings they write (-1: slot silent)."""
        if self.kind == 'tree':
            return h
        lt, rt = self.left[self.idx, h], self.right[self.idx, h]
        x = np.where(rng.random(h.shape) < .5, np.roll(rt, 1, 1), lt)     # position j shows left(slot j) or right(slot j - 1)
        return self.unit_of[self.idx, x, np.roll(x, -1, 1)]

    def string_probs(self, p):
        """p [K, S, V] -> probability [K, N] with which each deep unit writes each consistent string."""
        out = np.empty((len(p), len(self.cons)))
        for s in range(0, len(p), 128):
            pk = p[s:s + 128]
            both = pk[:, self.idx, self.cons]                             # [k, N, S]
            if self.kind == 'tree':
                out[s:s + 128] = both.prod(-1)
                continue
            pl, pr = np.einsum('ksv,svq->ksq', pk, self.lmat), np.einsum('ksv,svq->ksq', pk, self.rmat)
            lo, ro = pl[:, self.idx, self.cons_left], pr[:, self.idx, self.cons_right]
            t = np.stack([np.stack([lo, both], -1), np.stack([np.ones_like(lo), ro], -1)], -2)   # [k, N, S, 2, 2]; row: is the slot's left token shown (0) or not, column: is its right token shown (1) or not
            m = t[:, :, 0]
            for j in range(1, self.S):
                m = m @ t[:, :, j]
            out[s:s + 128] = np.trace(m, axis1=-2, axis2=-1) / 2 ** self.S
        return out


def softmax(w):
    e = np.exp(w - w.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def onehot_codes(code):
    t = np.zeros(code.shape + (V,))
    b = np.nonzero(code >= 0)
    t[b + (code[b],)] = 1
    return t


class Learner:
    def __init__(self, world, rng, k, start, scale):
        self.world, self.S = world, world.S
        self.w0 = np.zeros((k, self.S, V))
        if start == 'sharp':
            self.w0[:] = scale * onehot_codes(rng.integers(V, size=(k, self.S)))
        elif start != 'diffuse':
            code = world.cons[rng.integers(len(world.cons), size=k)].copy()
            for a in range(k):
                slots = rng.choice(self.S, start[1], replace=False)
                code[a, slots] = (code[a, slots] + rng.integers(1, V, size=len(slots))) % V
            self.w0[:] = scale * onehot_codes(code)
        self.r0 = softmax(self.w0)
        self.count = np.zeros(k)

    def prior(self):
        return (1 + self.count) / (1 + self.count).sum()

    def practice_round(self, rng, eta, eta_s, elig='accept'):
        """every deep unit writes one sample and is credited for what the first layer reads back; returns the codes read and the eligibilities."""
        p = softmax(self.w0)
        h = (p.cumsum(-1) < rng.random(p.shape[:-1] + (1,))).sum(-1).clip(max=V - 1)
        code = self.world.readback(rng, h)
        target = onehot_codes(code)
        g = (code >= 0).all(1).astype(float) if elig == 'accept' else np.ones(len(p)) if elig == 'plain' else (self.r0 * target).sum((1, 2)) / self.S
        self.w0 = np.clip(self.w0 + eta * g[:, None, None] * (target - p), -CLIP, CLIP)
        self.r0 += eta_s * (1. if elig == 'drive' else g[:, None, None]) * (target - self.r0)
        return code, g

    def credit(self, code, eta, eta_s):
        """codes [K, S], one per deep unit: wake and instar step on unit a for code a if the first layer read every slot."""
        target, g = onehot_codes(code), (code >= 0).all(1).astype(float)[:, None, None]
        self.w0 = np.clip(self.w0 + eta * g * (target - softmax(self.w0)), -CLIP, CLIP)
        self.r0 += eta_s * g * (target - self.r0)

    def write_one(self, rng, a):
        p = softmax(self.w0[a])
        h = (p.cumsum(-1) < rng.random((self.S, 1))).sum(-1).clip(max=V - 1)
        return self.world.readback(rng, h[None])[0]

    def read(self, rng, code, theta, eta, eta_s, teacher, recruit=False):
        """a string the first layer reads completely drives the deep units; the best-driven one fires if its drive reaches theta and takes a
        wake and an instar step; returns it (or -1)."""
        if (code < 0).any():
            return -1
        drive = self.r0[:, self.world.idx, code].sum(1) / self.S
        a = int(np.argmax(drive + 1e-9 * rng.random(len(drive))))
        target = onehot_codes(code)
        if drive[a] < theta and recruit:
            a = int(np.argmin(self.count + 1e-9 * rng.random(len(drive))))
            self.w0[a] = CLIP * target
            self.r0[a] = softmax(self.w0[a])
            self.count[a] += teacher
            return a
        if drive[a] < theta:
            return -1
        self.w0[a] = np.clip(self.w0[a] + eta * (target - softmax(self.w0[a])), -CLIP, CLIP)
        self.r0[a] += eta_s * (target - self.r0[a])
        self.count[a] += teacher
        return a

    def score(self):
        w = self.world
        p = softmax(self.w0)
        law = self.prior() @ w.string_probs(p)
        tmpl = p.argmax(-1)
        consistent = np.array([c.tobytes() in w.cons_id for c in tmpl])
        on_valid = np.array([w.is_valid[w.cons_id[c.tobytes()]] if ok else False for c, ok in zip(tmpl, consistent)])
        conflicts = (w.right[w.idx, tmpl] != np.roll(w.left[w.idx, tmpl], -1, 1)).sum(1) if w.kind == 'ring' else np.zeros(len(tmpl))
        best = np.stack([self.r0[:, w.idx, c].sum(1).max() / self.S for c in w.cons[w.valid]])
        return dict(kl=float(np.mean(-np.log(law[w.valid] * len(w.valid)))), valid=float(law[w.valid].sum()), consistent=float(law.sum()),
                    covered=float(np.mean(law[w.valid] * len(w.valid) >= .1)), units_consistent=float(consistent.mean()), units_valid=float(on_valid.mean()),
                    conflicts=float(conflicts.mean()), sharp=float(p.max(-1).mean()), strings_held=int(len({c.tobytes() for c, ok in zip(tmpl, on_valid) if ok})),
                    teacher_drive=float(best.mean()), teacher_pass=[float(np.mean(best >= th)) for th in (.5, .75, 1. - 1e-9)])


def run(job):
    seed, kind, f, name, arm, cfg = job
    world = World(seed, kind, f)
    rng = np.random.default_rng([seed, zlib.crc32(name.encode())])
    k, eta, eta_s, theta = cfg['k'], arm.get('eta', cfg['eta']), cfg['eta_s'], arm.get('theta', cfg['theta'])
    eta_t = arm.get('eta_teacher', cfg['eta_teacher'])
    me = Learner(world, rng, k, arm.get('start', 'sharp'), arm.get('scale', cfg['scale']))
    twin = Learner(world, rng, k, 'sharp', cfg['scale']) if arm['source'] == 'twin' else None
    stored = world.teacher(rng, arm['stored']) if arm['source'] == 'stored' else None
    rounds = arm.get('rounds', cfg['rounds'])
    marks = {0, rounds} if arm.get('final_only') else {0, rounds} | {int(round(x)) for x in np.geomspace(1, rounds, 28)}
    curve, fired, own_drive, done = [], [], [], np.full(min(k, 256), -1)
    for r in range(rounds + 1):
        if r in marks:
            curve.append(dict(samples=r * k, fired=float(np.mean(fired)) if fired else None, own_drive=float(np.mean(own_drive)) if own_drive else None, **me.score()))
            fired, own_drive = [], []
        if r == rounds:
            break
        source = {'then_teacher': ('own', 'teacher'), 'teacher_then_own': ('teacher', 'own')}.get(arm['source'], (arm['source'],) * 2)[r >= arm.get('switch', 0)]
        if source == 'own' and arm.get('credit', 'writer') == 'writer':
            fired.append(me.practice_round(rng, eta, eta_s, arm.get('elig', 'accept'))[1].mean())
            if arm.get('trace'):                                          # first round after which a unit writes one consistent string with probability > 1/2
                tmpl = softmax(me.w0[:len(done)])
                ok = np.array([c.tobytes() in world.cons_id for c in tmpl.argmax(-1)]) & (tmpl.max(-1).prod(-1) > .5)
                done[(done < 0) & ok] = r + 1
            continue
        if source == 'own':
            for a in rng.permutation(k):
                code = me.write_one(rng, a)
                if (code >= 0).all():
                    own_drive.append(me.r0[a, world.idx, code].sum() / world.S)
                    me.r0[a] += eta_s * (onehot_codes(code) - me.r0[a])
                fired.append(me.read(rng, code, theta, eta, eta_s, 0) >= 0)
            continue
        if source == 'probe':
            codes = world.readback(rng, rng.integers(V, size=(k, world.S)))
            fired.append(float((codes >= 0).all(1).mean()))
            if arm.get('credit', 'writer') == 'writer':
                me.credit(codes, eta, eta_s)
            else:
                for code in codes:
                    me.read(rng, code, theta, eta_t, eta_s, 0)
            continue
        codes = twin.practice_round(rng, cfg['eta'], eta_s)[0][rng.permutation(k)] if source == 'twin' else world.teacher(rng, k) if source != 'stored' else stored[rng.integers(len(stored), size=k)]
        for code in codes:
            fired.append(me.read(rng, code, theta, eta_t, eta_s, source != 'twin', arm.get('recruit', False)) >= 0)
    out = dict(seed=seed, kind=kind, f=f, arm=name, n_cons=len(world.cons), n_valid=len(world.valid), n_all=world.n_all, curve=curve)
    if arm.get('trace'):
        out['done_round'] = done.tolist()
    if twin is not None:
        out['twin'] = twin.score()
    return out


def references(job):
    """exact scores of a blank deep layer, the number of teacher strings until every first-layer unit has been seen, and the drive of the
    deep units at a sharp random start by teacher strings and by their own samples."""
    seed, kind, f, cfg = job
    world = World(seed, kind, f)
    rng = np.random.default_rng([seed, 7])
    blank = Learner(world, rng, 4, 'diffuse', 0.).score()
    codes = world.teacher(rng, 4096)
    first = np.full((world.S, V), 4096)
    for n in range(4095, -1, -1):
        first[world.idx, codes[n]] = n + 1
    sharp = Learner(world, rng, cfg['k'], 'sharp', cfg['scale'])
    tmpl = softmax(sharp.w0).argmax(-1)
    own = np.stack([world.readback(rng, tmpl) for _ in range(8)])                                          # [8, K, S]
    own_match = (own == tmpl[None]).sum(-1)
    teach = world.cons[world.valid]
    teach_match = (teach[:, None] == tmpl[None]).sum(-1)                                                   # [N, K]
    return dict(seed=seed, kind=kind, f=f, n_cons=len(world.cons), n_valid=len(world.valid), n_all=world.n_all, blank=blank, strings_until_all_units=int(first.max()),
                own_match_hist=np.bincount(own_match.ravel(), minlength=world.S + 1).tolist(), teacher_match_hist=np.bincount(teach_match.ravel(), minlength=world.S + 1).tolist(),
                teacher_best_hist=np.bincount(teach_match.max(1), minlength=world.S + 1).tolist(), own_unread=float(np.mean(own < 0)), own_changed=float(np.mean(own != tmpl[None])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--k', type=int, default=1024)
    parser.add_argument('--rounds', type=int, default=200)
    parser.add_argument('--eta', type=float, default=3.)
    parser.add_argument('--eta-teacher', type=float, default=.3)
    parser.add_argument('--eta-s', type=float, default=.3)
    parser.add_argument('--theta', type=float, default=.75)
    parser.add_argument('--scale', type=float, default=4.)
    parser.add_argument('--stored', type=int, default=64)
    parser.add_argument('--workers', type=int, default=28)
    parser.add_argument('--only', default='')
    parser.add_argument('--out', default='')
    args = parser.parse_args()
    cfg = vars(args)
    main_arms = dict(own_accept=dict(source='own', trace=True), own_accept_diffuse=dict(source='own', start='diffuse'), own_plain=dict(source='own', elig='plain'), own_drive=dict(source='own', elig='drive'),
                     own_read=dict(source='own', credit='read'), own_read_wta=dict(source='own', credit='read', theta=0.), teacher_gate=dict(source='teacher'), teacher_wta=dict(source='teacher', theta=0.),
                     stored_wta=dict(source='stored', stored=args.stored, theta=0.), stored_gate=dict(source='stored', stored=args.stored), twin_gate=dict(source='twin'), twin_wta=dict(source='twin', theta=0.))
    main_arms['teacher_recruit'] = dict(source='teacher', recruit=True, rounds=args.rounds // 4)
    main_arms |= dict(probe_writer=dict(source='probe'), probe_wta=dict(source='probe', credit='read', theta=0.), wta_then_own=dict(source='teacher_then_own', theta=0., switch=args.rounds // 2))
    for eta in (.3, 1.):
        main_arms[f'own_accept_eta{eta}'] = dict(source='own', eta=eta)
        main_arms[f'own_plain_eta{eta}'] = dict(source='own', elig='plain', eta=eta)
    for eta in (.1, 1., 3.):
        main_arms[f'teacher_wta_eta{eta}'] = dict(source='teacher', theta=0., eta_teacher=eta)
    for scale in (2., 6.):
        main_arms[f'own_accept_scale{scale}'] = dict(source='own', scale=scale)
    for theta in (1., .875, .625, .5, .375):
        main_arms[f'teacher_theta{theta}'] = dict(source='teacher', theta=theta, rounds=args.rounds // 2)
    for d in (0, 1, 2, 3, 4, 6, 8):
        main_arms[f'damage{d}_read'] = dict(source='own', credit='read', start=('damage', d), rounds=args.rounds // 2)
        main_arms[f'damage{d}_writer'] = dict(source='own', start=('damage', d), rounds=args.rounds // 2)
        main_arms[f'damage{d}_teacher'] = dict(source='teacher', start=('damage', d), rounds=args.rounds // 2)
    totals = [args.rounds * 2 ** -n for n in range(4, -1, -1)]
    budget_arms = {f'budget_theta{theta}_total{int(total)}_own{share}': dict(source='then_teacher', theta=theta, rounds=int(total), switch=int(round(share * total)), final_only=True)
                   for theta in (args.theta, 0.) for total in totals for share in (0., .25, .5, .75, 1.)}
    part_arms = {name: main_arms[name] for name in ('own_accept', 'teacher_wta', 'teacher_gate')} | dict(then_teacher=dict(source='then_teacher', switch=args.rounds // 2))
    tree_arms = {name: main_arms[name] for name in ('own_accept', 'own_accept_diffuse', 'own_read', 'teacher_wta', 'teacher_gate')}
    jobs = [(seed, 'ring', 1., name, arm, cfg) for seed in range(args.worlds) for name, arm in (main_arms | budget_arms).items()]
    jobs += [(seed, 'ring', f, name, arm, cfg) for seed in range(args.worlds) for f in (1., .5, .25) for name, arm in part_arms.items() if (f, name) != (1., 'own_accept') and (f < 1 or name == 'then_teacher')]
    jobs += [(seed, 'tree', 0., name, arm, cfg) for seed in range(args.worlds) for name, arm in tree_arms.items()]
    if args.only:
        jobs = [job for job in jobs if any(job[3].startswith(prefix) for prefix in args.only.split(','))]
    ref_jobs = [(seed, kind, f, cfg) for seed in range(args.worlds) for kind, f in (('ring', 1.), ('ring', .5), ('ring', .25), ('tree', 0.))]
    stamp = datetime.now(ZoneInfo('Europe/London')).strftime('%Y%m%d_%H%M%S')
    out = Path(args.out or f'/var/tmp/jbauer/generative_shared_visibles/run_{stamp}')
    out.mkdir(parents=True, exist_ok=True)
    cost = lambda job: -job[4].get('rounds', args.rounds) * (1 + 9 * (job[4].get('credit') == 'read' or job[4]['source'] != 'own'))
    with Pool(args.workers) as pool:
        refs = list(tqdm(pool.imap_unordered(references, ref_jobs), total=len(ref_jobs), desc='references'))
        runs = list(tqdm(pool.imap_unordered(run, sorted(jobs, key=cost)), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=cfg, constants=dict(L=L, Q=Q, DEG=DEG, V=V, N_TREE=N_TREE, CLIP=CLIP), references=refs, runs=runs)))
    print(out / 'results.json')


if __name__ == '__main__':
    main()
