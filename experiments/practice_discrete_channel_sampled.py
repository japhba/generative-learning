"""Actual internal trials for the discrete cause-coding screen."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import trange

from practice_discrete_channel import evaluate, plot_runs, softmax


def main():
    load_dotenv(Path.home()/'.env')
    base = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_discrete_channel/screen'
    out = base/'sampled'; out.mkdir(exist_ok=False)
    worlds = np.load(base/'worlds.npz')
    g_logits, r_logits = worlds['initial_g'].copy(), worlds['initial_r'].copy()
    channel, truth = worlds['channel'], worlds['truth']
    world_count, messages, actions = g_logits.shape
    outputs = channel.shape[-1]
    batch, rate, maximum = 512, 10., 6000
    protocol = {'uk_start': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'seed': 602126, 'batch_per_world': batch, 'rate_both_networks': rate, 'maximum_steps': maximum, 'baseline': 'Previous-trial exponential mean reward, updated at rate .05 after the current batch.', 'scope': 'Exploratory follow-up on the same worlds, not an independent confirmation. Both updates use only chosen cause, generated action, imagined observation, recognized cause and their internal equality reward. Fixed maximum; stop on joint/gen error log-log stationarity after step 1000.'}
    (out/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    shutil.copy2(__file__, out/'source.py')
    rng = np.random.default_rng(protocol['seed'])
    baseline = np.full(world_count, 1/messages)
    w = np.arange(world_count)[:, None]
    records, gs, rs, steps = [], [], [], []
    for iteration in trange(maximum+1, desc='Sampled joint cause coding'):
        g, r = softmax(g_logits), softmax(r_logits)
        if iteration % 20 == 0:
            record = np.concatenate([evaluate(g, r, channel), evaluate(g, r, truth)], axis=-1)
            assert np.isfinite(record).all()
            records.append(record); gs.append(g_logits.copy()); rs.append(r_logits.copy()); steps.append(iteration)
            if iteration >= 1000:
                error = 1-np.array(records[-20:])[:, :, :2].mean(axis=1)
                slopes = np.polyfit(np.log(np.array(steps[-20:])+1), np.log(error), 1)[0]
                if np.max(np.abs(slopes)) < .01: break
        if iteration == maximum: break
        z = rng.integers(messages, size=(world_count, batch))
        a = (rng.random((world_count, batch, 1)) > g[w, z].cumsum(axis=-1)).sum(axis=-1)
        y = (rng.random((world_count, batch, 1)) > channel[w, a].cumsum(axis=-1)).sum(axis=-1)
        recognized = (rng.random((world_count, batch, 1)) > r[w, y].cumsum(axis=-1)).sum(axis=-1)
        reward = (z == recognized).astype(float)
        advantage = reward-baseline[:, None]
        generator_events = np.zeros((world_count, messages, actions))
        recognition_events = np.zeros((world_count, outputs, messages))
        np.add.at(generator_events, (w, z, a), advantage)
        np.add.at(recognition_events, (w, y, recognized), advantage)
        g_logits += rate*(generator_events-generator_events.sum(axis=-1, keepdims=True)*g)/batch
        r_logits += rate*(recognition_events-recognition_events.sum(axis=-1, keepdims=True)*r)/batch
        baseline = .95*baseline + .05*reward.mean(axis=-1)
    run = {'records': np.array(records), 'steps': np.array(steps), 'g': np.array(gs), 'r': np.array(rs)}
    np.savez_compressed(out/'trajectory.npz', **run)
    baselines = dict(np.load(base/'baselines.npz'))
    rng = np.random.default_rng(611129)
    comparisons = {}
    for name in ['spectral', 'greedy', 'greedy_swap']:
        difference = baselines[name][:, 0]-run['records'][-1, :, 1]
        bootstrap = difference[rng.integers(world_count, size=(20000,world_count))].mean(axis=1)
        comparisons[name] = {'mean_direct_minus_sampled': float(difference.mean()), 'paired_world_bootstrap_95ci': np.quantile(bootstrap, [.025,.975]).tolist(), 'direct_wins': int(np.sum(difference > 0))}
    summary = protocol | {'uk_finish': datetime.now(ZoneInfo('Europe/London')).isoformat(), 'last_step': int(steps[-1]), 'initial': run['records'][0].mean(axis=0).tolist(), 'final': run['records'][-1].mean(axis=0).tolist(), 'metric_order': ['posterior_joint_success','posterior_optimal_decoder_success','true_joint_success','true_optimal_decoder_success'], 'comparisons': comparisons, 'minimum_recorded_generator_accuracy_change': float(np.diff(run['records'][:, :, 1], axis=0).min())}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    runs = {name: dict(np.load(base/f'{name}.npz')) for name in ['joint', 'recognition_only', 'frozen_recognizer']}
    runs['joint_sampled'] = run
    plot_runs(runs, baselines)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
