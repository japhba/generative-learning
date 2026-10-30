"""Full uncompressed linear-attention memory reproduces Gibbs learning.

The memory learner has scalar keys and vector values, never a stored weight
parameter. Every generation and recognition reads every occupied token.
The explicit learner is used only for a common-randomness comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange


def sigmoid(value): return np.exp(-np.logaddexp(0, -value))


def log_distribution(weight, states):
    logits = .5 * (weight @ states.T) ** 2
    return logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)


class ContextLearner:
    """Only memory and Gibbs visible states persist; no weight cache exists."""

    def __init__(self, initial, covariance, capacity):
        seeds, dimension = initial.shape
        self.keys = np.zeros((seeds, capacity))
        self.values = np.zeros((seeds, capacity, dimension))
        self.keys[:, 0] = 1.
        self.values[:, 0] = initial
        self.length = 1
        self.covariance = covariance.copy()

    def generate(self, latent):
        attention = latent[:, :, None] * self.keys[:, None, :self.length]
        return np.einsum('sbm,smd->sbd', attention, self.values[:, :self.length], optimize=False)

    def recognize(self, visible):
        # Swap the vector values into keys, and scalar keys into values.
        attention = np.einsum('sbd,smd->sbm', visible, self.values[:, :self.length], optimize=False)
        return np.einsum('sbm,sm->sb', attention, self.keys[:, :self.length], optimize=False)

    def prepare_update(self, visible, rate):
        # A temporary activation obtained by attention, never a parameter/cache.
        current = self.generate(np.ones((len(visible), 1)))[:, 0]
        positive = np.einsum('sij,sj->si', self.covariance, current)
        negative_keys = self.recognize(visible)
        keys = np.concatenate([np.ones((len(visible), 1)), negative_keys], axis=1)
        values = np.concatenate([rate * positive[:, None], -rate * visible / visible.shape[1]], axis=1)
        return keys, values

    def append(self, keys, values):
        following = self.length + keys.shape[1]
        self.keys[:, self.length:following] = keys
        self.values[:, self.length:following] = values
        self.length = following


def evaluation(weight, states, true_logp, frequency, covariance):
    logq = log_distribution(weight, states)
    kl = np.sum(np.exp(true_logp) * (true_logp - logq), axis=1)
    empirical_nll = -np.sum(frequency * logq, axis=1)
    norm_squared = np.sum(weight ** 2, axis=1)
    hch = np.einsum('si,sij,sj->s', weight, covariance, weight)
    reconstruction = states.shape[1] + (norm_squared - 2) * hch
    return np.stack([kl, empirical_nll, reconstruction], axis=1)


def main():
    load_dotenv(Path.home() / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_theory/curves_and_data.npz')
    parser.add_argument('--output', type=Path, default=Path(os.environ['FAST_CACHE_DIR']) / 'generative_practice_loop_context_covariance')
    parser.add_argument('--figure-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'tex/archive/figures')
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--sweeps', type=int, default=4)
    parser.add_argument('--burn-in', type=int, default=100)
    parser.add_argument('--rate', type=float, default=.1)
    parser.add_argument('--random-seed', type=int, default=810931)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True); args.figure_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(args.input)
    states = np.array(list(itertools.product([-1., 1.], repeat=3)))
    examples = states[archive['samples'][:args.seeds]]
    frequency = archive['empirical_frequencies'][:args.seeds].copy()
    covariance = np.einsum('sni,snj->sij', examples, examples) / examples.shape[1]
    assert np.max(np.abs(covariance - archive['empirical_covariances'][:args.seeds])) < 1e-14
    _, eigenvectors = np.linalg.eigh(covariance)
    initial = eigenvectors[:, :, -1].copy()
    # Deterministic gauge choice uses only the empirical eigenvector.
    initial *= np.sign(initial[:, 0])[:, None]
    explicit = initial.copy()
    context = ContextLearner(initial.copy(), covariance, 1 + args.steps * (args.batch + 1))
    true_logp = log_distribution(archive['true_h'][None], states)
    rng = np.random.default_rng(args.random_seed)
    visible_explicit = rng.choice([-1., 1.], size=(args.seeds, args.batch, 3))
    visible_context = visible_explicit.copy()
    errors = []
    weight_history = [explicit.copy()]
    context_history = [context.generate(np.ones((args.seeds, 1)))[:, 0]]
    curve = [evaluation(explicit, states, true_logp, frequency, covariance)]
    context_curve = [evaluation(context_history[-1], states, true_logp, frequency, covariance)]
    metadata = {'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, 'start_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(), 'environment': {'python': sys.version, 'python_executable': sys.executable, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__}, 'metric_columns': ['true_forward_kl', 'empirical_nll', 'tied_autoencoder_mse'], 'error_columns': ['recognition', 'generation', 'visible_mismatch', 'weight_read', 'update', 'density_metric'], 'protocol': 'All occupied tokens are read at every call. Fixed 300-step implementation audit, not an optimization-to-convergence run. True teacher used only by evaluation. Positive helper stores only empirical covariance; the context object never retains or reads old examples. Negative keys use old posterior means; values are signed. Original raw-example-helper audit remains preserved in generative_practice_loop_context with its original source snapshot.'}
    (args.output / 'protocol.json').write_text(json.dumps(metadata, indent=2) + '\n')
    started = perf_counter()
    for step in trange(-args.burn_in + 1, args.steps + 1, desc='Full growing context cycles'):
        sweeps = 1 if step <= 0 else args.sweeps
        recognition_error = generation_error = 0.
        mismatch = 0
        for _ in range(sweeps):
            noise = rng.normal(size=(args.seeds, args.batch))
            uniforms = rng.random((args.seeds, args.batch, 3))
            recognized_explicit = np.einsum('sbd,sd->sb', visible_explicit, explicit)
            recognized_context = context.recognize(visible_context)
            recognition_error = max(recognition_error, float(np.max(np.abs(recognized_context - recognized_explicit))))
            hidden_explicit = recognized_explicit + noise
            hidden_context = recognized_context + noise
            generated_explicit = hidden_explicit[:, :, None] * explicit[:, None]
            generated_context = context.generate(hidden_context)
            generation_error = max(generation_error, float(np.max(np.abs(generated_context - generated_explicit))))
            visible_explicit = np.where(uniforms < sigmoid(2 * generated_explicit), 1., -1.)
            visible_context = np.where(uniforms < sigmoid(2 * generated_context), 1., -1.)
            mismatch += int(np.count_nonzero(visible_explicit != visible_context))
        if step <= 0: continue
        explicit_negative = np.einsum('sbd,sb->sd', visible_explicit, np.einsum('sbd,sd->sb', visible_explicit, explicit)) / args.batch
        explicit_delta = args.rate * (np.einsum('sij,sj->si', covariance, explicit) - explicit_negative)
        keys, values = context.prepare_update(visible_context, args.rate)
        context_delta = np.einsum('sm,smd->sd', keys, values)
        update_error = float(np.max(np.abs(context_delta - explicit_delta)))
        explicit += explicit_delta
        context.append(keys, values)
        diagnostic_read = context.generate(np.ones((args.seeds, 1)))[:, 0]
        metric = evaluation(explicit, states, true_logp, frequency, covariance)
        memory_metric = evaluation(diagnostic_read, states, true_logp, frequency, covariance)
        error = [recognition_error, generation_error, mismatch, float(np.max(np.abs(diagnostic_read - explicit))), update_error, float(np.max(np.abs(memory_metric - metric)))]
        assert mismatch == 0, (step, error)
        assert max(error) < 1e-9, (step, error)
        assert np.isfinite(memory_metric).all()
        errors.append(error); curve.append(metric); context_curve.append(memory_metric)
        weight_history.append(explicit.copy()); context_history.append(diagnostic_read)
    curve, context_curve, errors = np.array(curve), np.array(context_curve), np.array(errors)
    metadata.update({'duration_seconds': perf_counter() - started, 'finish_uk': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'memory_tokens_per_seed': context.length, 'maximum_errors': dict(zip(metadata['error_columns'], errors.max(axis=0).tolist())), 'initial_mean_metrics': curve[0].mean(axis=0).tolist(), 'final_mean_metrics': curve[-1].mean(axis=0).tolist(), 'final_metrics_by_seed': curve[-1].tolist(), 'true_h': archive['true_h'].tolist()})
    np.savez_compressed(args.output / 'trajectories.npz', explicit_metrics=curve, context_metrics=context_curve, errors=errors, explicit_weights=weight_history, context_weight_diagnostics=context_history, memory_keys=context.keys[:, :context.length], memory_values=context.values[:, :context.length], examples=examples, empirical_frequencies=frequency, covariance=covariance, initial=initial, visible_final=visible_context)
    (args.output / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (args.output / 'run_source.py').write_bytes(Path(__file__).read_bytes())
    fig, axes = plt.subplots(2, 1, layout='constrained')
    times = np.arange(args.steps + 1)
    axes[0].plot(times, curve[:, :, 0].mean(axis=1), label='Explicit weights')
    axes[0].plot(times, context_curve[:, :, 0].mean(axis=1), linestyle='--', label='Growing context memory')
    axes[0].axhline(curve[0, :, 0].mean(), color='black', linestyle=':', label='Continue optimal reconstruction')
    axes[0].set(ylabel='True forward KL'); axes[0].legend()
    for index in [0, 1, 3]: axes[1].plot(times[1:], errors[:, index], label=metadata['error_columns'][index])
    axes[1].set(xlabel='Update', ylabel='Maximum absolute discrepancy')
    axes[1].legend()
    figure = args.output / 'practice-loop-context-covariance.png'
    fig.savefig(figure); plt.close(fig)
    (args.figure_dir / figure.name).write_bytes(figure.read_bytes())
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__': main()
