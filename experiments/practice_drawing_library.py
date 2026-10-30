"""Latent drawing motifs, learned program inference, and exact likelihood checks."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

load_dotenv(Path.home()/'.env')
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_drawing_library'
LENGTH, SIDE, MOTIFS = 9, 8, 3
POWERS = 4**np.arange(LENGTH-1, -1, -1)
MICRO = ((np.arange(64)[:, None]//(4**np.arange(2, -1, -1))) % 4).astype(np.uint8)
DX, DY = np.array([1, 0, -1, 0]), np.array([0, 1, 0, -1])
CONFIG = dict(examples=128, cache_size=16, trials=16, recognition_batch=128, recognition_rate=.15,
              generator_step=.25, floor=.02, eval_every=20, min_steps=200, max_steps=600,
              stationary_window=8, stationary_slope=.003, calibration_steps=600, seed=20103000)


def render(actions):
    n = len(actions)
    x, y, direction = np.full(n, 3), np.full(n, 3), np.zeros(n, dtype=int)
    canvas = np.full(n, np.uint64(1) << np.uint64(27), dtype=np.uint64)
    for action in actions.T:
        move = (action == 0) | (action == 3)
        x = (x+move*DX[direction]) % SIDE
        y = (y+move*DY[direction]) % SIDE
        canvas |= (action == 0).astype(np.uint64) << (y*SIDE+x).astype(np.uint64)
        direction = (direction+(action == 1)-(action == 2)) % 4
    return canvas


class Catalogue:
    def __init__(self):
        self.actions = ((np.arange(4**LENGTH)[:, None]//POWERS) % 4).astype(np.uint8)
        self.chunks = self.actions.reshape(-1, 3, 3) @ np.array([16, 4, 1])
        self.canvas = render(self.actions)
        self.images, self.ids, self.counts = np.unique(self.canvas, return_inverse=True, return_counts=True)
        self.order = np.argsort(self.ids, kind='stable')
        self.offset = np.r_[0, np.cumsum(self.counts)]
        self.uniform = self.counts/len(self.actions)
        # Pixel indices in turtle-relative coordinates; axes rotate with heading.
        dy, dx = np.divmod(np.arange(64), SIDE)
        dx, dy = dx-3, dy-3
        self.indices = np.empty((64, 4, 64), dtype=np.uint64)
        for position in range(64):
            y, x = divmod(position, SIDE)
            for direction in range(4):
                xx = (x+dx*DX[direction]-dy*DY[direction]) % SIDE
                yy = (y+dx*DY[direction]+dy*DX[direction]) % SIDE
                self.indices[position, direction] = yy*SIDE+xx

    def aliases(self, image): return self.order[self.offset[image]:self.offset[image+1]]

    def probabilities(self, a, b):
        component = np.prod(b[:, np.arange(3)[None, :], MICRO[:, :]], axis=-1)
        chunk = a @ component
        code = np.prod(chunk[np.arange(3), self.chunks], axis=-1)
        return code, np.bincount(self.ids, weights=code, minlength=len(self.images))


def row_normalize(counts):
    return (1-CONFIG['floor'])*counts/counts.sum(axis=-1, keepdims=True)+CONFIG['floor']/counts.shape[-1]


def draw(a, b, rng, n):
    motif = np.stack([rng.choice(MOTIFS, n, p=row) for row in a], axis=-1)
    p = b[motif]
    action = (rng.random((n, 3, 3, 1)) > p.cumsum(axis=-1)).sum(axis=-1)
    return action.reshape(n, LENGTH).astype(np.uint8) @ POWERS


def features(cat, target, canvas, x, y, direction):
    shifts = cat.indices[y*SIDE+x, direction]
    all_pixels = ((target[:, None] >> shifts) & np.uint64(1)).astype(float)
    remaining = (((target & ~canvas)[:, None] >> shifts) & np.uint64(1)).astype(float)
    return np.c_[np.ones(len(target)), all_pixels, remaining]


def recognize(cat, weights, target, rng, actions=None, update=False):
    n = len(target)
    x, y, direction = np.full(n, 3), np.full(n, 3), np.zeros(n, dtype=int)
    canvas = np.full(n, np.uint64(1) << np.uint64(27), dtype=np.uint64)
    output, log_probability = np.zeros((n, LENGTH), dtype=np.uint8), np.zeros(n)
    gradients = np.zeros_like(weights)
    for step in range(LENGTH):
        phi = features(cat, target, canvas, x, y, direction)
        logits = phi @ weights[step].T
        nx, ny = (x+DX[direction]) % SIDE, (y+DY[direction]) % SIDE
        allowed = ((target >> (ny*SIDE+nx).astype(np.uint64)) & np.uint64(1)).astype(bool)
        logits[~allowed, 0] = -np.inf
        p = np.exp(logits-logits.max(axis=-1, keepdims=True))
        p /= p.sum(axis=-1, keepdims=True)
        if actions is None:
            proposal = .9*p+.1*(p > 0)/(p > 0).sum(axis=-1, keepdims=True)
            action = (rng.random((n, 1)) > proposal.cumsum(axis=-1)).sum(axis=-1)
        else: action = actions[:, step]
        assert np.all(allowed | (action != 0))
        log_probability += np.log(p[np.arange(n), action])
        if update:
            error = -p
            error[np.arange(n), action] += 1
            gradients[step] = error.T @ phi/n
        output[:, step] = action
        move = (action == 0) | (action == 3)
        x, y = np.where(move, nx, x), np.where(move, ny, y)
        canvas |= (action == 0).astype(np.uint64) << (y*SIDE+x).astype(np.uint64)
        direction = (direction+(action == 1)-(action == 2)) % 4
    return output @ POWERS, canvas == target, log_probability, gradients


def fit_counts(cat, a, b, codes, weights):
    component = np.prod(b[:, np.arange(3)[None, :], MICRO[:, :]], axis=-1)
    acount, bcount = np.zeros_like(a), np.zeros_like(b)
    for chunk in range(3):
        observed = np.bincount(cat.chunks[codes, chunk], weights=weights, minlength=64)
        responsibilities = a[chunk, :, None]*component
        responsibilities /= responsibilities.sum(axis=0, keepdims=True)
        expected = responsibilities*observed
        acount[chunk] = expected.sum(axis=-1)
        for j in range(3):
            for symbol in range(4): bcount[:, j, symbol] += expected[:, MICRO[:, j] == symbol].sum(axis=-1)
    return row_normalize(acount), row_normalize(bcount)


def cache_posterior(caches, mass, code_probability):
    codes = np.concatenate(caches)
    weights = np.concatenate([count*code_probability[cache]/code_probability[cache].sum() for count, cache in zip(mass, caches)])
    return codes, weights/weights.sum()


def train(cat, world, condition, rng, steps):
    a, b, recognition = world['initial_a'].copy(), world['initial_b'].copy(), np.zeros((9, 4, 129))
    unique, mass = np.unique(world['observed'], return_counts=True)
    caches = [np.array([code]) for code in world['canonical']]
    target, history, weights_history = cat.images[unique], [], []
    code_p, image_p = cat.probabilities(a, b)
    old_nll = -np.average(np.log(image_p[unique]), weights=mass)
    accepted, valid_proposals, attempts = 0, 0, 0
    start = time.perf_counter()
    calibration = []
    if condition == 'calibrated':
        calibration_probe = draw(a, b, rng, 512)
        for calibration_step in range(CONFIG['calibration_steps']+1):
            if calibration_step % CONFIG['eval_every'] == 0:
                lp = recognize(cat, recognition, cat.canvas[calibration_probe], rng, cat.actions[calibration_probe])[2]
                calibration.append([calibration_step, -lp.mean()])
            if calibration_step == CONFIG['calibration_steps']: break
            examples = draw(a, b, rng, CONFIG['recognition_batch'])
            gradient = recognize(cat, recognition, cat.canvas[examples], rng, cat.actions[examples], update=True)[3]
            recognition += CONFIG['recognition_rate']*gradient
    # Identical fixed latent-program probe across conditions, held outside training.
    probe_rng = np.random.default_rng(int(world['seed'])+11000)
    probes = draw(world['teacher_a'], world['teacher_b'], probe_rng, 512)
    eval_rng = np.random.default_rng(int(world['seed'])+12000)
    status = 'maximum updates'
    for step in tqdm(range(steps+1), desc=condition, leave=False):
        if step % CONFIG['eval_every'] == 0 or step == steps:
            _, _, logp, _ = recognize(cat, recognition, cat.canvas[probes], eval_rng, cat.actions[probes])
            _, correct, _, _ = recognize(cat, recognition, cat.canvas[probes], eval_rng)
            truth = world['teacher_p']
            kl = np.sum(truth*np.log(truth/image_p))
            history.append([step, old_nll, kl, -logp.mean(), correct.mean(), np.mean([len(c) for c in caches]),
                            accepted/max(step, 1), valid_proposals/max(attempts, 1), time.perf_counter()-start])
            weights_history.append(np.r_[a.ravel(), b.ravel()])
            assert np.isfinite(history[-1]).all()
            if step >= CONFIG['min_steps']:
                recent = np.array(history[-CONFIG['stationary_window']:])
                slopes = [np.polyfit(np.log(recent[:, 0]), np.log(recent[:, col]), 1)[0] for col in [1, 5]]
                if max(abs(s) for s in slopes) < CONFIG['stationary_slope']:
                    status = 'stationary training likelihood and cache size'
                    break
        if step == steps: break
        if condition == 'sleep':
            examples = draw(a, b, rng, CONFIG['recognition_batch'])
            _, _, _, gradient = recognize(cat, recognition, cat.canvas[examples], rng, cat.actions[examples], update=True)
            recognition += CONFIG['recognition_rate']*gradient
        elif condition == 'wake':
            codes, posterior = cache_posterior(caches, mass, code_p)
            examples = rng.choice(codes, CONFIG['recognition_batch'], p=posterior)
            _, _, _, gradient = recognize(cat, recognition, cat.canvas[examples], rng, cat.actions[examples], update=True)
            recognition += CONFIG['recognition_rate']*gradient
        if condition in ['sleep', 'wake', 'frozen', 'calibrated']:
            candidates, valid, _, _ = recognize(cat, recognition, np.repeat(target, CONFIG['trials']), rng)
            candidates, valid = candidates.reshape(-1, CONFIG['trials']), valid.reshape(-1, CONFIG['trials'])
            valid_proposals += valid.sum(); attempts += valid.size
            for i in range(len(caches)):
                combined = np.union1d(caches[i], candidates[i, valid[i]])
                caches[i] = combined[np.argsort(code_p[combined], kind='stable')[-CONFIG['cache_size']:]]
        if condition == 'exact':
            codes = np.concatenate([cat.aliases(i) for i in unique])
            weights = np.concatenate([count*code_p[cat.aliases(i)]/image_p[i] for count, i in zip(mass, unique)])
            weights /= weights.sum()
        else: codes, weights = cache_posterior(caches, mass, code_p)
        proposed_a, proposed_b = fit_counts(cat, a, b, codes, weights)
        amount = CONFIG['generator_step']
        for _ in range(12):
            new_a, new_b = (1-amount)*a+amount*proposed_a, (1-amount)*b+amount*proposed_b
            new_code_p, new_image_p = cat.probabilities(new_a, new_b)
            new_nll = -np.average(np.log(new_image_p[unique]), weights=mass)
            if new_nll <= old_nll+1e-12:
                a, b, code_p, image_p, old_nll = new_a, new_b, new_code_p, new_image_p, new_nll
                accepted += 1
                break
            amount /= 2
    return dict(records=np.array(history), weights=np.array(weights_history), recognition=recognition, a=a, b=b,
                caches=np.concatenate(caches), cache_sizes=np.array([len(c) for c in caches]), status=status,
                calibration=np.array(calibration))


def world(cat, seed):
    rng = np.random.default_rng(seed)
    prototypes = rng.choice(np.flatnonzero((MICRO == 0).any(axis=-1)), 3, replace=False)
    teacher_b = .94*np.eye(4)[MICRO[prototypes]]+.06/4
    teacher_a = .9*rng.dirichlet(np.full(3, .6), size=3)+.1/3
    _, teacher_p = cat.probabilities(teacher_a, teacher_b)
    observed = rng.choice(len(cat.images), CONFIG['examples'], p=teacher_p)
    unique = np.unique(observed)
    canonical = np.array([rng.choice(cat.aliases(i)) for i in unique])
    # Exposure provides one valid explanation per distinct image, shared by all arms.
    mass = np.unique(observed, return_counts=True)[1]
    freq = np.stack([np.bincount(cat.actions[canonical].reshape(-1, 3, 3)[:, :, j].ravel(),
                                weights=np.repeat(mass, 3), minlength=4) for j in range(3)])
    freq = row_normalize(freq)
    initial_b = np.stack([row_normalize(freq*rng.lognormal(0, .35, (3, 4))) for _ in range(3)])
    initial_a = row_normalize(rng.dirichlet(np.full(3, 6.), size=3))
    return dict(seed=seed, teacher_a=teacher_a, teacher_b=teacher_b, teacher_p=teacher_p,
                observed=observed, canonical=canonical, initial_a=initial_a, initial_b=initial_b, prototypes=prototypes)


def direct_controls(cat, observed):
    initial = cat.uniform
    alphas = np.array([.1, 1., 10., 100., 1000.])
    scores = np.zeros(len(alphas))
    rng = np.random.default_rng(20103001)
    for test in np.array_split(rng.permutation(len(observed)), 4):
        train = np.setdiff1d(np.arange(len(observed)), test)
        counts = np.bincount(observed[train], minlength=len(cat.images))
        for i, alpha in enumerate(alphas): scores[i] -= np.log((counts[observed[test]]+alpha*initial[observed[test]])/(len(train)+alpha)).sum()
    counts = np.bincount(observed, minlength=len(cat.images))
    alpha = alphas[np.argmin(scores)]
    return (counts+alpha*initial)/(len(observed)+alpha), dict(alpha=float(alpha), cv_nll=scores.tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, default=4)
    parser.add_argument('--seed', type=int, default=CONFIG['seed'])
    parser.add_argument('--steps', type=int, default=CONFIG['max_steps'])
    parser.add_argument('--conditions', nargs='+', choices=['sleep', 'wake', 'frozen', 'calibrated', 'canonical', 'exact'],
                        default=['sleep', 'wake', 'frozen', 'canonical', 'exact'])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'source.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        config=CONFIG, worlds=args.worlds, seed=args.seed, steps=args.steps, conditions=args.conditions,
        primary='Exploratory screen: final true image KL and own-dream versus old-cache recognition; no outcome-based selection.',
        caveat='Exact enumeration supplies a likelihood acceptance check and initial valid causes; this is an additional global operation, not a local neural learning rule.'), indent=2)+'\n')
    cat, summary = Catalogue(), []
    print(json.dumps(dict(programs=len(cat.actions), distinct_images=len(cat.images), max_aliases=int(cat.counts.max()))), flush=True)
    for i in tqdm(range(args.worlds), desc='worlds'):
        data = world(cat, args.seed+i)
        np.savez_compressed(args.output/f'world_{i}.npz', **data)
        direct, details = direct_controls(cat, data['observed'])
        row = dict(seed=args.seed+i, conditions={}, histogram=dict(kl=float(np.sum(data['teacher_p']*np.log(data['teacher_p']/direct))), **details))
        for condition in args.conditions:
            result = train(cat, data, condition, np.random.default_rng(args.seed+i+100000), args.steps)
            np.savez_compressed(args.output/f'{condition}_{i}.npz', **result)
            row['conditions'][condition] = dict(final=result['records'][-1].tolist(), status=result['status'])
            print(json.dumps(dict(world=i, condition=condition, **row['conditions'][condition])), flush=True)
        summary.append(row)
        (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__': main()
