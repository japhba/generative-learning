"""Joint generation and latent-cause recognition through a learned noisy channel.

Finite input/output exposure identifies a Bayesian channel memory. Offline
practice receives no new world observations. Every teaching target is a latent
cause sampled by the learner itself. The oracle decoder is evaluation-only.
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm


def project(w, gradient): return gradient - np.sum(w*gradient, axis=-1, keepdims=True)*w


def normalize(w): return w/np.linalg.norm(w, axis=-1, keepdims=True)


def risk(w, r, mean, noise):
    signal = np.sum(w*mean, axis=-1)
    return (1-r*signal)**2 + noise*r*r, noise/(signal*signal+noise)


def drift(w, r, mean, noise):
    signal = np.sum(w*mean, axis=-1)
    error_gain = 1-r*signal
    return project(w, (r*error_gain)[:, None]*mean), error_gain*signal-noise*r


def mean_step(w, r, mean, noise, dt, condition):
    def field(x, y):
        dw, dr = drift(x, y, mean, noise)
        if condition == 'recognition_only': dw = np.zeros_like(dw)
        if condition == 'frozen_recognizer': dr = np.zeros_like(dr)
        return dw, dr
    k1, l1 = field(w, r)
    k2, l2 = field(w+dt*k1/2, r+dt*l1/2)
    k3, l3 = field(w+dt*k2/2, r+dt*l2/2)
    k4, l4 = field(w+dt*k3, r+dt*l3)
    return normalize(w+dt*(k1+2*k2+2*k3+k4)/6), r+dt*(l1+2*l2+2*l3+l4)/6


def imagined_batch(w, r, mean, gamma, action_var, env_var, z, rng):
    innovation = np.sqrt(action_var)*rng.standard_normal((2, *z.shape, w.shape[-1]))
    command = w[None, :, None, :]*z[None, :, :, None]+innovation
    channel = mean[None, :, None, :]+np.sqrt(gamma)*rng.standard_normal(command.shape)
    observation = np.sum(channel*command, axis=-1)+np.sqrt(env_var)*rng.standard_normal(command.shape[:-1])
    error = z[None]-r[None, :, None]*observation
    return innovation, observation, error


def sampled_step(w, r, mean, gamma, action_var, env_var, pairs, dt, rng, scramble):
    z = rng.standard_normal((w.shape[0], pairs))
    innovation, observation, error = imagined_batch(w, r, mean, gamma, action_var, env_var, z, rng)
    reward = -.5*error**2
    if scramble:
        _, _, unrelated_error = imagined_batch(w, r, mean, gamma, action_var, env_var, z, rng)
        reward = -.5*unrelated_error**2
    dw = np.mean(.5*(reward[0]-reward[1])[:, :, None]*(innovation[0]-innovation[1])*z[:, :, None]/action_var, axis=1)
    dr = np.mean(error*observation, axis=(0, 2))
    return normalize(w+dt*project(w, dw)), r+dt*dr


def worlds(seeds, seed, d=4, repeats=4, env_var=.1, action_var=.03):
    rng = np.random.default_rng(seed)
    true_channel = rng.standard_normal((seeds, d))
    commands = np.repeat(np.eye(d), repeats, axis=0)
    commands[1::2] *= -1
    observations = true_channel@commands.T + np.sqrt(env_var)*rng.standard_normal((seeds, d*repeats))
    gamma = 1/(1+repeats/env_var)
    mean = gamma*(observations@commands)/env_var
    noise = env_var+gamma+action_var*(np.sum(mean**2, axis=-1)+d*gamma)
    w0 = normalize(rng.standard_normal((seeds, d)))
    return true_channel, commands, observations, gamma, mean, noise, w0


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=32)
    parser.add_argument('--seed-start', type=int, default=191127)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--dt', type=float, default=.01)
    parser.add_argument('--pairs', type=int, default=128)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    action_var = .03; env_var = .1
    true_channel, commands, observations, gamma, mean, noise, w0 = worlds(args.seeds, args.seed_start)
    source = Path(__file__).read_bytes()
    shutil.copy2(__file__, args.output/'source.py')
    protocol = vars(args) | {'output': str(args.output), 'uk_started': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'source_sha256': hashlib.sha256(source).hexdigest(), 'd': 4, 'n_observations': 16, 'prior_channel_covariance': 'identity', 'env_var': env_var, 'action_var': action_var, 'power': 1, 'posterior_variance': gamma, 'recognizer_initialization': 'zero except frozen_recognizer, which is initialized to the best linear recognizer for the starting generator', 'primary': 'Bayes optimal-linear-decoder MSE for the generated channel; this isolates generation from recognizer improvement.', 'secondary': ['actual current-recognizer Bayes MSE', 'actual-world optimal-linear-decoder MSE', 'actual-world current-recognizer MSE', 'weight alignment'], 'design': 'All configurations declared before these new worlds. Equal generator/recognizer step sizes; no truth in training; no replay of original observations after computing the channel posterior.', 'stopping': 'After time 20 stop a condition if the last 20 evaluation means have absolute double-log slopes below .01 for both joint Bayes MSE and excess generation MSE. Evaluation every .2 time units. Abort on nonfinite or mean MSE exceeding 5. Frozen/independent controls may plateau and stop earlier.', 'theorem_scope': 'Continuous exact expected projected gradient flow, not individual finite-batch steps.'}
    (args.output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    np.savez_compressed(args.output/'worlds.npz', true_channel=true_channel, commands=commands, observations=observations, mean=mean, noise=noise, w0=w0, gamma=gamma)
    optimum = noise/(noise+np.sum(mean**2, axis=-1))
    true_noise = env_var+action_var*np.sum(true_channel**2, axis=-1)
    initial_signal = np.sum(w0*mean, axis=-1)
    initial_gen = noise/(noise+initial_signal**2)
    names = ['joint_mean_field', 'joint_sampled', 'recognition_only', 'frozen_recognizer', 'independent_reward']
    histories = {}; summary = {'worlds': args.seeds, 'observations': 16, 'initial_generation_mse': float(initial_gen.mean()), 'optimal_generation_mse': float(optimum.mean()), 'gamma': gamma}
    for condition_index, name in enumerate(tqdm(names, desc='Cause-recognition conditions', mininterval=2)):
        rng = np.random.default_rng(args.seed_start+10000+condition_index)
        w = w0.copy(); r = np.zeros(args.seeds)
        if name == 'frozen_recognizer': r = initial_signal/(initial_signal**2+noise)
        records = []
        for step in tqdm(range(args.steps+1), desc=name, leave=False, mininterval=2):
            if step % 20 == 0:
                joint, generation = risk(w, r, mean, noise)
                true_joint, true_generation = risk(w, r, true_channel, true_noise)
                alignment = np.sum(w*mean, axis=-1)**2/np.sum(mean**2, axis=-1)
                assert np.isfinite(joint).all() and joint.mean() < 5, (name, step, joint)
                records.append((step, w.copy(), r.copy(), joint, generation, true_joint, true_generation, alignment))
                np.savez_compressed(args.output/f'{name}_checkpoint.npz', step=step, w=w, r=r, rng_state=json.dumps(rng.bit_generator.state))
                if step*args.dt >= 20 and len(records) >= 21:
                    t = np.array([v[0]*args.dt for v in records[-20:]])
                    js = np.array([v[3].mean() for v in records[-20:]])
                    gs = np.array([(v[4]-optimum).mean() for v in records[-20:]])
                    slopes = [np.polyfit(np.log(t), np.log(v+1e-14), 1)[0] for v in (js, gs)]
                    if max(abs(s) for s in slopes) < .01: break
            if step == args.steps: break
            if name in ['joint_sampled', 'independent_reward']:
                w, r = sampled_step(w, r, mean, gamma, action_var, env_var, args.pairs, args.dt, rng, name == 'independent_reward')
            else: w, r = mean_step(w, r, mean, noise, args.dt, name)
        arrays = {key: np.array([row[j] for row in records]) for j, key in enumerate(['step', 'w', 'r', 'joint_mse', 'generation_mse', 'true_joint_mse', 'true_generation_mse', 'alignment'])}
        np.savez_compressed(args.output/f'{name}.npz', **arrays)
        histories[name] = arrays
        gain = initial_gen-arrays['generation_mse'][-1]
        summary[name] = {'final_time': float(arrays['step'][-1]*args.dt), 'final_joint_mse': float(arrays['joint_mse'][-1].mean()), 'final_generation_mse': float(arrays['generation_mse'][-1].mean()), 'final_true_joint_mse': float(arrays['true_joint_mse'][-1].mean()), 'final_true_generation_mse': float(arrays['true_generation_mse'][-1].mean()), 'initial_true_generation_mse': float(arrays['true_generation_mse'][0].mean()), 'mean_generation_gain': float(gain.mean()), 'se_generation_gain': float(gain.std(ddof=1)/np.sqrt(args.seeds)), 'bayes_worse_worlds': int(np.sum(gain < -1e-10)), 'actual_generation_worse_worlds': int(np.sum(arrays['true_generation_mse'][-1] > arrays['true_generation_mse'][0]+1e-10)), 'maximum_generation_mse_increase': float(np.diff(arrays['generation_mse'], axis=0).max()), 'alignment': float(arrays['alignment'][-1].mean())}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    figures = root/'tex/archive/figures'
    for metric, filename, ylabel in [('generation_mse', 'practice-latent-channel-generation.png', 'Generation MSE with an optimal linear decoder'), ('joint_mse', 'practice-latent-channel-joint.png', 'MSE using the currently learned recognizer')]:
        fig, ax = plt.subplots(layout='constrained')
        for name, label in [('joint_mean_field', 'Joint mean field'), ('joint_sampled', 'Joint sampled'), ('recognition_only', 'Learn recognition only'), ('frozen_recognizer', 'Freeze initially optimal recognizer'), ('independent_reward', 'Independent reward')]:
            a = histories[name]; values = a[metric]; m = values.mean(axis=1); se = values.std(axis=1, ddof=1)/np.sqrt(args.seeds)
            line, = ax.plot(a['step']*args.dt, m, label=label)
            ax.fill_between(a['step']*args.dt, m-1.96*se, m+1.96*se, color=line.get_color(), alpha=.15)
        ax.axhline(optimum.mean(), linestyle=':', label='Analytic optimum')
        ax.set(xlabel='Offline learning time', ylabel=ylabel, title='Generation and latent-cause recognition learn together')
        ax.legend(); fig.savefig(figures/filename); plt.close(fig)
    fig, ax = plt.subplots(layout='constrained')
    a = histories['joint_mean_field']; signal = np.einsum('tsd,sd->ts', a['w'], mean)
    for label, values in [('Learned recognizer gain', np.abs(a['r'][:, 0])), ('Optimal recognizer gain', np.abs(signal[:, 0]/(signal[:, 0]**2+noise[0]))), ('Generator direction overlap', a['alignment'][:, 0])]: ax.plot(a['step']*args.dt, values, label=label)
    ax.set(xlabel='Offline learning time', ylabel='Diagnostic value', title='First predeclared world: coadaptation')
    ax.legend(); fig.savefig(figures/'practice-latent-channel-representation.png'); plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
