"""A visible unit written straight by the root (a skip connection in the data): what a first-layer verdict on own samples can teach the layer above.

World, two levels of one-hot groups. One of N0 root productions t (equally likely) names a symbol pair (a_t, b_t); each symbol is written
by one of M synonym units of the first layer (unit = symbol * M + synonym, V * M units per slot, each with a token patch of its own); the
root also writes a trace token tag(t) with RHO values straight to the surface (RHO = 1: no trace, RHO = N0: the root is visible).
A unit's token patch is unique, so a string is its two units plus the trace, and everything is scored on unit pairs.

Learner. First layer, from the stored teacher strings: the units seen, how often each is used per slot, and Hebbian counts unit x trace.
Its verdict on a string: the two units have been seen with a common trace value (with RHO = 1: each has been seen in its slot).
With --verdict pairs the first layer instead keeps which pairs of units it has seen together (a lateral memory, no trace needed).
Layer 0: K units with a fixed uniform prior, each writing the two slots independently, w[c, slot, unit]. It starts at the slot
frequencies (what a layer without deep units knows) and a credited sample moves the rows of unit c towards the sample's units,
dw = ETA * credit_c * (onehot - w). The arms differ in which samples are credited to which deep unit:
  gate_fresh             teacher strings, credited to the best deep unit only if its evidence exceeds the blank start by GATE nats
                         (a fresh teacher string with a unit the first layer has not seen adds that unit to the first layer)
  em / sem / wta (fresh or stored)  teacher strings without a gate, credited by the posterior over deep units (em), to one deep unit
                         drawn from that posterior (sem) or to its maximum (wta)
  practice               own samples that pass the verdict, credited to the deep unit that wrote them
  practice_w             the same, weighted by 1 / (the learner's probability of the sample), normalised by a running mean and capped
  practice_f             the same, weighted by (the first layer's probability of the pair, from its unit x trace frequencies) / (the
                         learner's probability of the sample), capped: a graded verdict instead of pass / fail
  practice_w_*_fresh / _stored  practice_w and teacher strings (gate, em or sem) in turn, half each
  probe, draw_probe      pairs of uniformly drawn units that pass the verdict, credited to a random deep unit (probe) or to one drawn
                         from the learner's posterior (draw_probe)
  reread_own, draw_own   own samples that pass the verdict, credited by the learner's posterior over its deep units (reread) or to one
                         deep unit drawn from it (draw), weighted as in practice_w
  draw_twin              two learners from different starts; each is credited, as in draw_own, for the samples of the other
References in closed form, from the unit x trace counts and with the learner's floor EPS: draw the trace, then each slot's unit
given the trace, by the counted frequencies (inversion) or uniformly over the units seen with that trace (inversion_set).
With SKEW > 0 the productions and the synonyms of a symbol are not equally likely (log-probabilities Gaussian with s.d. SKEW).
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

ARMS = dict(gate_fresh=dict(teacher='fresh', credit='gate'), em_fresh=dict(teacher='fresh'), em_stored=dict(teacher='stored'), sem_fresh=dict(teacher='fresh', credit='draw'), sem_stored=dict(teacher='stored', credit='draw'),
            wta_fresh=dict(teacher='fresh', credit='best'), wta_stored=dict(teacher='stored', credit='best'), practice=dict(own=1.), practice_w=dict(own=1., weight=True), practice_f=dict(own=1., weight='judge'),
            practice_w_gate_fresh=dict(own=.5, weight=True, teacher='fresh', credit='gate'), practice_w_em_fresh=dict(own=.5, weight=True, teacher='fresh'), practice_w_sem_fresh=dict(own=.5, weight=True, teacher='fresh', credit='draw'),
            practice_w_gate_stored=dict(own=.5, weight=True, teacher='stored', credit='gate'), practice_w_sem_stored=dict(own=.5, weight=True, teacher='stored', credit='draw'),
            probe=dict(own=1., source='uniform'), draw_probe=dict(own=1., source='uniform', own_credit='draw'), reread_own=dict(own=1., weight=True, own_credit='soft'), draw_own=dict(own=1., weight=True, own_credit='draw'), draw_twin=dict(own=1., weight=True, own_credit='draw', source='twin'))
DEFAULT = dict(own=0., teacher=None, credit='soft', weight=False, source='own', own_credit='writer')


def mutual_information(joint):
    outer = joint.sum(1, keepdims=True) * joint.sum(0, keepdims=True)
    keep = joint > 0
    return float(np.sum(joint[keep] * (np.log(joint[keep]) - np.log(outer[keep]))))


def make_world(rng, args, rho):
    v, m, n0 = args['v'], args['m'], args['n0']
    pairs = rng.choice(v * v, n0, replace=False)
    a, b = pairs // v, pairs % v
    owner = -np.ones((v * m, v * m), int)
    for t in range(n0):
        owner[a[t] * m:(a[t] + 1) * m, b[t] * m:(b[t] + 1) * m] = t
    tag = rng.permutation(n0) % rho
    root, synonym = np.exp(args['skew'] * rng.normal(size=n0)), np.exp(args['skew'] * rng.normal(size=(v, m)))
    root, synonym = root / root.sum(), synonym / synonym.sum(1, keepdims=True)
    truth = np.zeros(owner.shape)
    for t in range(n0):
        truth[a[t] * m:(a[t] + 1) * m, b[t] * m:(b[t] + 1) * m] = root[t] * np.outer(synonym[a[t]], synonym[b[t]])
    return dict(a=a, b=b, tag=tag, rho=rho, owner=owner, valid=owner >= 0, root=root, synonym=synonym, truth=truth)


def teacher(world, rng, args, size):
    t = rng.choice(args['n0'], size=size, p=world['root'])
    draw = lambda symbols: (world['synonym'][symbols].cumsum(1) > rng.random((size, 1))).argmax(1)
    return t, world['a'][t] * args['m'] + draw(world['a'][t]), world['b'][t] * args['m'] + draw(world['b'][t])


def first_layer(world, args, t, u1, u2):
    n1 = len(world['owner'])
    counts = np.zeros((2, n1, world['rho']))
    np.add.at(counts[0], (u1, world['tag'][t]), 1)
    np.add.at(counts[1], (u2, world['tag'][t]), 1)
    seen = counts.sum((0, 2)) > 0
    flat, trace = seen / seen.sum(), counts[0].sum(0) / len(t)
    floored = lambda rows: (1 - args['eps']) * rows / np.maximum(rows.sum(1, keepdims=True), 1e-300) + args['eps'] * flat[None, :, None]
    verdict = ((counts[0] > 0) * 1.) @ ((counts[1] > 0) * 1.).T > 0
    if args['verdict'] == 'pairs':
        verdict = np.zeros_like(verdict)
        verdict[u1, u2] = True
    slot = counts / np.maximum(counts.sum(1, keepdims=True), 1)
    return dict(seen=seen, flat=flat, usage=counts.sum(2) / len(t), verdict=verdict, judge=np.einsum('r,ar,br->ab', trace, slot[0], slot[1]),
                inversion=np.einsum('r,ar,br->ab', trace, *floored(counts)), inversion_set=np.einsum('r,ar,br->ab', trace, *floored((counts > 0) * 1.)))


def start(layer, args, rng):
    w = layer['usage'][None] * (1 + args['noise'] * rng.random((args['k'], *layer['usage'].shape)))
    return w / w.sum(2, keepdims=True)


def score(model, world, args):
    valid, truth = world['valid'], world['truth']
    scored = (1 - args['delta']) * model + args['delta'] / valid.size
    return dict(kl=float((truth[valid] * np.log(truth[valid] / scored[valid])).sum()), valid=float(model[valid].sum()), covered=float(truth[valid][model[valid] >= .5 * truth[valid]].sum()))


def evaluate(w, world, layer, args):
    mix = (1 - args['eps']) * w + args['eps'] * layer['flat']
    units = np.einsum('ca,cb->cab', mix[:, 0], mix[:, 1])
    model, valid = units.mean(0), world['valid']
    posterior = units / np.maximum(units.sum(0), 1e-300)
    joint = np.stack([(posterior * world['truth'])[:, world['owner'] == t].sum(1) for t in range(args['n0'])])
    accept = float((model * layer['verdict']).sum())
    return score(model, world, args) | dict(read=mutual_information(joint) / np.log(args['n0']), accept=accept, precision=float((model * layer['verdict'] * valid).sum() / accept),
                                            units_valid=int(((units * valid).sum((1, 2)) > .9).sum()), productions=int((np.array([model[world['owner'] == t].sum() for t in range(args['n0'])]) >= .5 * world['root']).sum()))


def credited(kind, likelihood, rng, threshold=None):
    """Which deep units a sample is credited to: all by the posterior, one drawn from it, its maximum, or its maximum if above a threshold."""
    if kind == 'soft':
        return likelihood / likelihood.sum()
    credit = np.zeros(len(likelihood))
    if kind == 'draw':
        credit[(likelihood.cumsum() > rng.random() * likelihood.sum()).argmax()] = 1.
    elif kind == 'best':
        credit[int((likelihood * (1 + 1e-9 * rng.random(len(likelihood)))).argmax())] = 1.
    else:
        best = int(likelihood.argmax())
        credit[best] = likelihood[best] > threshold
    return credit


def simulate(cfg, learners, world, layer, stored, args, rng):
    """One sample per learner and step; with source 'twin' the sample comes from the other learner. Curves are those of the first learner."""
    k, eps = args['k'], args['eps']
    layer = layer | dict(seen=layer['seen'].copy())
    flat, blank = layer['flat'], np.maximum(layer['usage'], 1e-3)
    marks = {0} | {int(s) for s in np.round(np.geomspace(1, args['steps'], args['points']))}
    history, scale = [], [None] * len(learners)
    for step in range(args['steps'] + 1):
        if step in marks:
            history.append(dict(step=step) | evaluate(learners[0], world, layer, args))
        if step == args['steps']:
            break
        for i, w in enumerate(learners):
            mix = (1 - eps) * w + eps * flat
            if rng.random() < cfg['own']:
                c = rng.integers(k)
                rows = dict(own=mix[c], uniform=np.stack([flat, flat]), twin=((1 - eps) * learners[i - 1][c] + eps * flat))[cfg['source']]
                a, b = (rows.cumsum(1) > rng.random((2, 1)) * rows.sum(1, keepdims=True)).argmax(1)
                if not layer['verdict'][a, b]:
                    continue
                likelihood, weight = mix[:, 0, a] * mix[:, 1, b], 1.
                if cfg['weight'] == 'judge':
                    weight = min(args['cap'], layer['judge'][a, b] / likelihood.mean())
                elif cfg['weight']:
                    inverse = 1 / likelihood.mean()
                    scale[i] = inverse if scale[i] is None else scale[i] + args['track'] * (inverse - scale[i])
                    weight = min(args['cap'], inverse / scale[i])
                credit = weight * (np.eye(k)[c] if cfg['own_credit'] == 'writer' else credited(cfg['own_credit'], likelihood, rng))
            else:
                if cfg['teacher'] == 'stored':
                    j = rng.integers(len(stored[0]))
                    a, b = stored[1][j], stored[2][j]
                else:
                    _, (a,), (b,) = teacher(world, rng, args, 1)
                    if not (layer['seen'][a] and layer['seen'][b]):
                        layer['seen'][[a, b]] = True
                        flat = layer['flat'] = layer['seen'] / layer['seen'].sum()
                        mix = (1 - eps) * w + eps * flat
                likelihood = mix[:, 0, a] * mix[:, 1, b]
                credit = credited(cfg['credit'], likelihood, rng, np.exp(args['gate']) * blank[0, a] * blank[1, b])
            rate = np.minimum(args['eta'] * credit, 1.)[:, None]
            w[:, 0] *= 1 - rate
            w[:, 1] *= 1 - rate
            w[:, 0, a] += rate[:, 0]
            w[:, 1, b] += rate[:, 0]
    return history


def run(job):
    args, cfg = job['args'], DEFAULT | ARMS[job['arm']]
    rng = np.random.default_rng(job['seed'])
    world = make_world(rng, args, job['rho'])
    stored = [values[:job['stored']] for values in teacher(world, rng, args, max(args['stored']))]
    layer = first_layer(world, args, *stored)
    learners = [start(layer, args, np.random.default_rng([job['seed'], 1 + i])) for i in range(1 + (cfg['source'] == 'twin'))]
    history = simulate(cfg, learners, world, layer, stored, args, np.random.default_rng([job['seed'], 4, job['stored']]))
    verdict, valid = layer['verdict'], world['valid']
    reference = dict(seen=int(layer['seen'].sum()), verdict_size=int(verdict.sum()), verdict_precision=float((verdict & valid).sum() / verdict.sum()), verdict_recall=float((verdict & valid).sum() / valid.sum()),
                     inversion=score(layer['inversion'], world, args), inversion_set=score(layer['inversion_set'], world, args), target=score(verdict / verdict.sum(), world, args))
    return {key: job[key] for key in ('seed', 'arm', 'rho', 'stored')} | dict(reference=reference, curves={key: [row[key] for row in history] for key in history[0]})


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', type=int, default=8)
    parser.add_argument('--m', type=int, default=4)
    parser.add_argument('--n0', type=int, default=16)
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--skew', type=float, default=0.)
    parser.add_argument('--verdict', default='trace', choices=['trace', 'pairs'])
    parser.add_argument('--eta', type=float, default=.01)
    parser.add_argument('--eps', type=float, default=.02)
    parser.add_argument('--delta', type=float, default=1e-3)
    parser.add_argument('--noise', type=float, default=.01)
    parser.add_argument('--gate', type=float, default=1.)
    parser.add_argument('--cap', type=float, default=5.)
    parser.add_argument('--track', type=float, default=.01)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--points', type=int, default=40)
    parser.add_argument('--stored', type=int, nargs='+', default=[128, 256, 512, 1024, 4096])
    parser.add_argument('--rhos', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--scan', nargs='*', default=['practice', 'practice_w'])
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--workers', type=int, default=30)
    parser.add_argument('--tag', default='main')
    args = vars(parser.parse_args())
    out = Path(os.environ['FAST_CACHE_DIR']) / 'generative_skip_verdict' / f"{args['tag']}_{datetime.now(ZoneInfo('Europe/London')):%Y%m%d_%H%M%S}"
    out.mkdir(parents=True)
    (out / 'protocol.json').write_text(json.dumps(args, indent=2))
    jobs = [dict(seed=20261121 + world, arm=arm, rho=rho, stored=stored, args=args) for world in range(args['worlds']) for stored in args['stored']
            for rho, arms in [(args['n0'], args['arms'])] + [(rho, args['scan']) for rho in args['rhos']] for arm in arms]
    with Pool(args['workers']) as pool:
        results = list(tqdm(pool.imap_unordered(run, jobs), total=len(jobs), desc='runs'))
    (out / 'results.json').write_text(json.dumps(dict(args=args, runs=results)))
    print(out)


if __name__ == '__main__':
    main()
