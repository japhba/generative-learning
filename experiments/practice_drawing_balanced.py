"""Sample-selected posterior balancing with an initially tight likelihood bound.

Every recognition block update and generator M step increases the same ELBO.
Full likelihoods are evaluated for audits, never to accept training updates.
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from zoneinfo import ZoneInfo

import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m
from practice_drawing_fixed_posterior import counts, mstep, complete_q

LABELS = ((np.arange(27)[:, None]//np.array([9, 3, 1])) % 3).astype(np.uint8)
CONFIG = dict(batch=4096, eval_every=20, min_steps=200, max_steps=1600, stationary_window=10, slope=.003)


def joint_probability(cat, a, b, codes, labels):
    actions = cat.actions[codes].reshape(-1, 3, 3)
    result = np.ones(len(codes))
    for c in range(3):
        result *= a[c, labels[:, c]]
        for t in range(3): result *= b[labels[:, c], t, actions[:, c, t]]
    return result


def sample_joint(a, b, rng, n):
    labels = np.stack([rng.choice(3, n, p=row) for row in a], axis=-1)
    p = b[labels]
    actions = (rng.random((n, 3, 3, 1)) > p.cumsum(axis=-1)).sum(axis=-1)
    codes = actions.reshape(n, 9) @ m.POWERS
    return codes, labels, codes*27+labels @ np.array([9, 3, 1])


def coactivity(cat, states, weights):
    codes, labels = states//27, LABELS[states % 27]
    actions = cat.actions[codes].reshape(-1, 3, 3)
    a, b = np.zeros((3, 3)), np.zeros((3, 3, 4))
    for c in range(3):
        a[c] = np.bincount(labels[:, c], weights=weights, minlength=3)
        for t in range(3): b[:, t] += np.bincount(labels[:, c]*4+actions[:, c, t], weights=weights, minlength=12).reshape(3, 4)
    return a, b


class Recognition:
    def __init__(self, cat, world):
        self.cat = cat
        self.initial_a, self.initial_b = world['initial_a'], world['initial_b']
        code, image = cat.probabilities(self.initial_a, self.initial_b)
        self.initial_image = image
        self.probability = np.empty(len(code)*27)
        # Exact initial posterior; enumeration is an explicit initialization cost.
        for start in range(0, len(code), 4096):
            codes = np.repeat(np.arange(start, min(start+4096, len(code))), 27)
            labels = np.tile(LABELS, (len(codes)//27, 1))
            self.probability[start*27:(start+len(codes)//27)*27] = joint_probability(cat, self.initial_a, self.initial_b, codes, labels)/image[cat.ids[codes]]
        self.rank = np.argsort(cat.order)
        self.base = 1 << (len(self.probability)-1).bit_length()
        self.tree = np.zeros(2*self.base)
        self.tree[self.base:self.base+len(self.probability)] = self.probability.reshape(-1,27)[cat.order].ravel()
        start = self.base//2
        while start:
            self.tree[start:2*start] = self.tree[2*start:4*start:2]+self.tree[2*start+1:4*start:2]
            start //= 2
        mass = np.bincount(cat.ids, weights=self.probability.reshape(-1, 27).sum(axis=-1), minlength=len(image))
        assert np.max(abs(mass-1)) < 1e-10

    def prefix_mass(self, positions):
        nodes, value = self.base+positions, np.zeros(len(positions))
        while np.any(nodes > 1):
            value += np.where(nodes % 2, self.tree[nodes-1], 0.)
            nodes //= 2
        return value

    def samples(self, images, rng):
        lo = self.prefix_mass(27*self.cat.offset[images])
        hi = self.prefix_mass(27*self.cat.offset[images+1])
        uniform = lo+rng.random(len(images))*(hi-lo)
        node = np.ones(len(images), dtype=int)
        for _ in range(self.base.bit_length()-1):
            left = 2*node
            right = uniform >= self.tree[left]
            uniform -= np.where(right, self.tree[left], 0.)
            node = left+right
        position = node-self.base
        states = self.cat.order[position//27]*27+position % 27
        assert np.array_equal(self.cat.ids[states//27], images)
        return states

    def balance(self, states, a, b, data_mass):
        states = np.unique(states)
        codes, labels = states//27, LABELS[states % 27]
        images = self.cat.ids[codes]
        old = self.probability[states].copy()
        joint = joint_probability(self.cat, a, b, codes, labels)
        block_mass = np.bincount(images, weights=old, minlength=len(data_mass))[images]
        joint_mass = np.bincount(images, weights=joint, minlength=len(data_mass))[images]
        new = block_mass*joint/joint_mass
        gain = np.sum(data_mass[images]*old*np.log(old/new))
        assert gain >= -1e-10 and np.all(new > 0)
        self.probability[states] = new
        nodes = self.base+self.rank[codes]*27+states % 27
        self.tree[nodes] = new
        while np.any(nodes > 1):
            nodes = np.unique(nodes//2)
            self.tree[nodes] = self.tree[2*nodes]+self.tree[2*nodes+1]
        change_a, change_b = coactivity(self.cat, states, data_mass[images]*(new-old))
        return change_a, change_b, float(gain), len(states)

    def exact_bound(self, a, b, data_mass):
        bound = 0.
        for start in range(0, len(self.probability), 131072):
            states = np.arange(start, min(start+131072, len(self.probability)))
            images = self.cat.ids[states//27]
            positive = data_mass[images] > 0
            selected, images = states[positive], images[positive]
            posterior = self.probability[selected]
            joint = joint_probability(self.cat, a, b, selected//27, LABELS[selected % 27])
            bound += np.sum(data_mass[images]*posterior*(np.log(joint)-np.log(posterior)))
        return bound


def train(cat, world, condition, steps, rng):
    recognition = Recognition(cat, world)
    data_mass = np.bincount(world['observed'], minlength=len(cat.images))/len(world['observed'])
    a, b = world['initial_a'].copy(), world['initial_b'].copy()
    acount, bcount, initial_p = counts(cat, a, b, world['observed'])
    initial_ll = np.log(initial_p[world['observed']]).mean()
    assert abs(recognition.exact_bound(a,b,data_mass)-initial_ll) < 1e-10
    bound_gain, recognition_gain, generated, data_hits = 0., 0., 0, 0
    history, params = [], []
    teacher_code, _ = cat.probabilities(world['teacher_a'], world['teacher_b'])
    eval_rng = np.random.default_rng(int(world['seed'])+210000)
    start = time.perf_counter()
    status = 'maximum steps'
    for step in tqdm(range(steps+1), desc=condition, leave=False):
        if step % CONFIG['eval_every'] == 0 or step == steps:
            _, p = cat.probabilities(a, b)
            ll = np.log(p[world['observed']]).mean()
            risk = np.sum(world['teacher_p']*np.log(world['teacher_p']/p))
            bound = initial_ll+bound_gain
            dream_codes, _, dream_states = sample_joint(a,b,eval_rng,4096)
            dream_ce = -np.log(recognition.probability[dream_states]).mean()
            program_recognition = recognition.probability.reshape(-1,27).sum(axis=-1)
            recognition_kl = np.sum(teacher_code*np.log(teacher_code/(world['teacher_p'][cat.ids]*program_recognition)))
            assert ll >= bound-1e-8
            history.append([step, -ll, risk, -bound, ll-bound, recognition_gain, data_hits/max(generated,1), time.perf_counter()-start, dream_ce, recognition_kl])
            params.append(np.r_[a.ravel(), b.ravel()])
            assert np.isfinite(history[-1]).all()
            if step >= CONFIG['min_steps']:
                recent = np.array(history[-CONFIG['stationary_window']:])
                slopes = [np.polyfit(np.log(recent[:,0]), np.log(recent[:,j]), 1)[0] for j in [3,8]]
                if max(abs(s) for s in slopes) < CONFIG['slope']:
                    status = 'stationary variational bound and dream recognition'; break
        if step == steps: break
        if condition != 'fixed':
            if condition == 'prior':
                codes, labels, proposals = sample_joint(world['initial_a'], world['initial_b'], rng, CONFIG['batch'])
            elif condition == 'recognition_only':
                images = rng.choice(len(cat.images), CONFIG['batch'], p=data_mass)
                proposals = recognition.samples(images,rng)
                codes = proposals//27
            else: codes, labels, proposals = sample_joint(a,b,rng,CONFIG['batch'])
            # Execute the generated program; the catalogue only identifies its bitmap.
            canvas = m.render(cat.actions[codes])
            images = np.searchsorted(cat.images,canvas)
            assert np.array_equal(images,cat.ids[codes])
            generated += len(images); data_hits += (data_mass[images] > 0).sum()
            partners = recognition.samples(images,rng)
            ca, cb, gain, _ = recognition.balance(np.r_[proposals,partners],a,b,data_mass)
            acount += ca; bcount += cb
            bound_gain += gain; recognition_gain += gain
        new_a,new_b = mstep(acount,bcount)
        gain = complete_q(new_a,new_b,acount,bcount)-complete_q(a,b,acount,bcount)
        assert gain >= -1e-10
        bound_gain += float(gain)
        a,b = new_a,new_b
    measured = recognition.exact_bound(a,b,data_mass)
    assert abs(measured-(initial_ll+bound_gain)) < 1e-8
    mass = np.bincount(cat.ids,weights=recognition.probability.reshape(-1,27).sum(axis=-1),minlength=len(cat.images))
    assert np.max(abs(mass-1)) < 1e-9
    return dict(records=np.array(history),weights=np.array(params),a=a,b=b,status=status,
                final_bound_error=measured-(initial_ll+bound_gain),posterior_row_error=np.max(abs(mass-1)),
                recognition=recognition.probability,program_recognition=program_recognition)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worlds',type=int,default=4)
    parser.add_argument('--first-world',type=int,default=0)
    parser.add_argument('--steps',type=int,default=CONFIG['max_steps'])
    parser.add_argument('--conditions',nargs='+',choices=['practice','prior','recognition_only','fixed'],default=['practice','prior','recognition_only','fixed'])
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'source.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),config=CONFIG,
        source=str(args.source),worlds=args.worlds,steps=args.steps,conditions=args.conditions,
        guarantee='Every sampled block projection and exact categorical M-step increases an initially tight ELBO; true data likelihood stays above the initial likelihood plus cumulative gains. No likelihood acceptance check.',
        stopping='Only the locally accumulated ELBO and dream recognition cross entropy control stopping. Exact data likelihood and teacher risk are audit diagnostics.',
        initialization='Exact initial posterior over all program and motif causes, computed by enumeration. This is a supplied computation, not a free learned recognition capability.',
        controls='Fixed initial posterior; generation proposals held at the initial prior; current-recognition-only proposals on original images. Every adaptive condition uses the current recognition distribution for comparison causes.',
        scope='Exploratory follow-up on existing worlds, not a fresh replication.'),indent=2)+'\n')
    cat,summary=m.Catalogue(),[]
    for i in tqdm(range(args.first_world,args.first_world+args.worlds),desc='Balanced worlds'):
        world=np.load(args.source/f'world_{i}.npz')
        row=dict(seed=int(world['seed']),conditions={})
        for condition in args.conditions:
            run=train(cat,world,condition,args.steps,np.random.default_rng(int(world['seed'])+200000))
            posterior = run.pop('recognition')
            if i == 0 and condition == 'practice': np.savez_compressed(args.output/'recognition_0.npz',posterior=posterior)
            np.savez_compressed(args.output/f'{condition}_{i}.npz',**run)
            row['conditions'][condition]=dict(final=run['records'][-1].tolist(),status=run['status'],bound_error=float(run['final_bound_error']),row_error=float(run['posterior_row_error']))
            print(json.dumps(dict(world=i,condition=condition,**row['conditions'][condition])),flush=True)
        summary.append(row)
        (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
