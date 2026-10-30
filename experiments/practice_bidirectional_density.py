"""Exact small-model audit for a normalized bidirectional latent generator.

This is a mathematical screen, not a sampled-practice experiment. It checks
whether correcting the global normalizer can improve likelihood after the
usual observed-example bound has stopped improving.
"""

import argparse
from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


D, H = 6, 3
X = np.array(list(product((0., 1.), repeat=D)))
Z = np.array(list(product((0., 1.), repeat=H)))
XF = np.column_stack([2*X-1, np.ones(len(X))])
ZF = np.column_stack([2*Z-1, np.ones(len(Z))])
PARAMETERS = H+D*(H+1)+H*(D+1)


def logsumexp(a, axis):
    peak = a.max(axis=axis, keepdims=True)
    return np.squeeze(peak+np.log(np.exp(a-peak).sum(axis=axis, keepdims=True)), axis=axis)


def sigmoid(a): return np.exp(-np.logaddexp(0., -a))


def unpack(theta): return theta[:H], theta[H:H+D*(H+1)].reshape(D, H+1), theta[H+D*(H+1):].reshape(H, D+1)


def forward(theta):
    prior, generator, recognition = unpack(theta)
    emission_logits, recognition_logits = ZF @ generator.T, XF @ recognition.T
    log_prior = Z @ prior-np.logaddexp(0., prior).sum()
    log_g = X @ emission_logits.T-np.logaddexp(0., emission_logits).sum(axis=-1)[None]+log_prior[None]
    log_r = recognition_logits @ Z.T-np.logaddexp(0., recognition_logits).sum(axis=-1)[:, None]
    half = .5*(log_g+log_r)
    log_s = logsumexp(half, axis=1)
    mu = np.exp(half-log_s[:, None])
    log_normalizer = logsumexp(2*log_s, axis=0)
    log_p = 2*log_s-log_normalizer
    assert abs(np.exp(log_g).sum()-1) < 1e-10
    assert np.max(np.abs(np.exp(log_r).sum(axis=-1)-1)) < 1e-10
    assert log_normalizer <= 1e-10
    return dict(log_g=log_g, log_r=log_r, log_s=log_s, mu=mu, log_p=log_p,
                log_normalizer=log_normalizer, emission=sigmoid(emission_logits), recognition=sigmoid(recognition_logits))


def score_average(theta, state, weights):
    prior, _, _ = unpack(theta)
    gp = np.einsum('xz,zh->h', weights, Z-sigmoid(prior))
    residual_g = X[:, None]-state['emission'][None]
    gg = np.einsum('xz,xzd,zk->dk', weights, residual_g, ZF)
    residual_r = Z[None]-state['recognition'][:, None]
    gr = np.einsum('xz,xzh,xk->hk', weights, residual_r, XF)
    return np.concatenate([gp, gg.reshape(-1), gr.reshape(-1)])


def objectives(theta, frequency, truth):
    state = forward(theta)
    observed = frequency[:, None]*state['mu']
    free = np.exp(state['log_p'])[:, None]*state['mu']
    positive = score_average(theta, state, observed)
    negative = score_average(theta, state, free)
    records = np.array([frequency @ state['log_p'], 2*frequency @ state['log_s'],
        truth @ (np.log(truth)-state['log_p']),
        truth @ (np.log(truth)-logsumexp(state['log_g'], axis=1)),
        np.sum(free*(np.log(state['mu'])-state['log_r'])), np.linalg.norm(positive), np.linalg.norm(positive-negative)])
    return records, positive, positive-negative, state


def fit(theta, frequency, truth, objective, iterations=2400):
    history, states, steps = [], [], []
    status = 'iteration cap'
    step_size = 1.
    column = 1 if objective == 'bound' else 0
    for step in range(iterations+1):
        record, bound_gradient, likelihood_gradient, _ = objectives(theta, frequency, truth)
        gradient = bound_gradient if objective == 'bound' else likelihood_gradient
        if step % 10 == 0:
            history.append(record); states.append(theta.copy()); steps.append(step)
            assert np.isfinite(record).all()
            if step >= 200:
                slopes = np.polyfit(np.log(np.array(steps[-20:])+1), np.log(-np.array(history[-20:])[:, [column]]), 1)[0]
                if np.max(np.abs(slopes)) < 1e-5:
                    status = 'stationary internally evaluated objective'; break
        if np.linalg.norm(gradient) < 1e-7: status = 'stationary gradient'; break
        if step == iterations: break
        for _ in range(40):
            proposed = theta+step_size*gradient
            state = forward(proposed)
            value = 2*frequency @ state['log_s'] if objective == 'bound' else frequency @ state['log_p']
            if value >= record[column]+1e-4*step_size*(gradient @ gradient): break
            step_size *= .5
        else: raise AssertionError('Line search failed')
        theta = proposed
        step_size = min(1.1*step_size, 10.)
    return dict(theta=theta, history=np.array(history), states=np.array(states), steps=np.array(steps), status=np.array(status))


def audit(output):
    rng = np.random.default_rng(20102871)
    theta = rng.normal(0, .5, PARAMETERS)
    frequency = rng.dirichlet(np.ones(len(X)))
    _, gb, gl, state = objectives(theta, frequency, frequency)
    numerical = np.zeros((2, PARAMETERS))
    for j in range(PARAMETERS):
        delta = np.zeros(PARAMETERS); delta[j] = 1e-5
        plus, minus = forward(theta+delta), forward(theta-delta)
        numerical[0, j] = 2*frequency @ (plus['log_s']-minus['log_s'])/2e-5
        numerical[1, j] = frequency @ (plus['log_p']-minus['log_p'])/2e-5
    error = np.max(np.abs(numerical-np.stack([gb, gl])))
    assert error < 1e-8
    # The two-latent lift eliminates the latent sum from the free joint score.
    log_pair = .5*(state['log_g'][:, :, None]+state['log_r'][:, :, None]+state['log_g'][:, None, :]+state['log_r'][:, None, :])
    log_pair -= state['log_normalizer']
    joint = np.exp(log_pair)
    marginal_error = np.max(np.abs(joint.sum(axis=(1, 2))-np.exp(state['log_p'])))
    assert marginal_error < 1e-12
    # An own-generation proposal is G(x,z1) R(z2|x); its MH ratio uses only
    # the opposite assignment of generation and recognition, not a teacher.
    log_proposal = state['log_g'][:, :, None]+state['log_r'][:, None, :]
    log_ratio = .5*(state['log_g'][:, None, :]+state['log_r'][:, :, None]-state['log_g'][:, :, None]-state['log_r'][:, None, :])
    ratio_error = np.max(np.abs(log_pair-log_proposal-log_ratio+state['log_normalizer']))
    assert ratio_error < 1e-12
    flux_error = 0.
    logp, logq, ratio = log_pair.reshape(-1), log_proposal.reshape(-1), log_ratio.reshape(-1)
    for _ in range(10000):
        i, j = rng.integers(len(logp), size=2)
        forward_flux = logp[i]+logq[j]+min(0., ratio[j]-ratio[i])
        reverse_flux = logp[j]+logq[i]+min(0., ratio[i]-ratio[j])
        flux_error = max(flux_error, abs(forward_flux-reverse_flux))
    assert flux_error < 1e-12
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), gradient_error=float(error),
        pair_marginal_error=float(marginal_error), own_generation_ratio_error=float(ratio_error), detailed_balance_log_flux_error=float(flux_error))
    (output/'audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_bidirectional_density/exact_screen')
    parser.add_argument('--worlds', type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'source.py')
    protocol = dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), worlds=args.worlds, seed=20102880, examples=512,
        purpose='Exact mathematical screen only: does normalized likelihood improve after the original-example BiHM bound is fitted?',
        assumptions='Binary sigmoid generator, three independent binary latent causes, six visible bits; factorized learned causal recognizer.',
        limitation='No sampled-practice or shortcut-separation claim. Teacher data and exact population risk are used only for diagnostics.')
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    audit(args.output)
    rows = []
    for world in tqdm(range(args.worlds), desc='Bidirectional density, exact screen'):
        rng = np.random.default_rng(protocol['seed']+world)
        teacher = rng.normal(0, 1.2, PARAMETERS)
        teacher[:H] *= .25
        prior, generator, _ = unpack(teacher)
        generator[:, -1] *= .2
        true_joint = np.exp(forward(teacher)['log_g'])
        truth = true_joint.sum(axis=-1)
        data = rng.choice(len(X), size=protocol['examples'], p=truth)
        frequency = np.bincount(data, minlength=len(X))/len(data)
        initial = rng.normal(0, .3, PARAMETERS)
        _, generator, recognition = unpack(initial)
        means = (X[data].sum(axis=0)+.5)/(len(data)+1)
        generator[:, -1] = np.log(means/(1-means))
        np.savez_compressed(args.output/f'world_{world}.npz', teacher=teacher, truth=truth, data=data, frequency=frequency, initial=initial)
        bound = fit(initial, frequency, truth, 'bound')
        np.savez_compressed(args.output/f'bound_{world}.npz', **bound)
        likelihood = fit(bound['theta'], frequency, truth, 'likelihood')
        np.savez_compressed(args.output/f'likelihood_{world}.npz', **likelihood)
        before = objectives(bound['theta'], frequency, truth)
        after = objectives(likelihood['theta'], frequency, truth)
        negative = before[1]-before[2]
        rows.append(dict(world=world, before=before[0].tolist(), after=after[0].tolist(),
            only_negative_likelihood_direction=float(before[2] @ -negative), only_negative_bound_direction=float(before[1] @ -negative),
            bound_status=str(bound['status']), likelihood_status=str(likelihood['status'])))
        (args.output/'summary.json').write_text(json.dumps(rows, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    runs = [np.load(args.output/f'likelihood_{w}.npz') for w in range(args.worlds)]
    grid = np.arange(0, max(r['steps'][-1] for r in runs)+1, 10)
    for column, ax in zip([0, 2], axes):
        ax.plot(grid, np.mean([np.interp(grid, r['steps'], r['history'][:, column]) for r in runs], axis=0), label='Normalized likelihood fitting')
        ax.axhline(np.mean([r['history'][0, column] for r in runs]), linestyle=':', label='Initial observed-example bound fit')
    axes[0].set(ylabel='Observed-data log likelihood')
    axes[1].set(ylabel='True predictive KL (nats)', xlabel='Exact gradient iterations after bound fitting')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center')
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-bidirectional-density-screen.png')
    plt.close(fig)
    print(json.dumps(rows, indent=2))


if __name__ == '__main__': main()
