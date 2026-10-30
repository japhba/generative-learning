"""Controls and interpretation for the hidden-sequence mechanism screen."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

import practice_hidden_sequences as model


def main():
    load_dotenv(Path.home()/'.env')
    base = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_hidden_sequences'
    source, output = base/'internal_stopping', base/'analysis_internal'
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((source/'summary.json').read_text())
    n = len(rows)
    bootstrap = np.random.default_rng(20102694).integers(n, size=(20000, n))
    def interval(value): return dict(mean=float(value.mean()), bootstrap95=np.quantile(value[bootstrap].mean(axis=-1), [.025, .975]).tolist())
    conditions = ['sleep', 'wake', 'frozen', 'exact']
    runs = {c: [np.load(source/f'{c}_{i}.npz') for i in range(n)] for c in conditions}
    final = {c: np.array([r['records'][-1, 1] for r in runs[c]]) for c in conditions}
    report = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), scope='Post-screen analysis, not independent confirmation.',
                  endpoint_means={c: np.mean([r['records'][-1] for r in runs[c]], axis=0).tolist() for c in conditions},
                  sleep_minus_control={c: interval(final['sleep']-final[c]) for c in conditions[1:]})
    for c in ['order_1', 'order_2', 'order_3', 'hard_em', 'em']:
        report['sleep_minus_control'][c] = interval(final['sleep']-np.array([r['controls'][c][1] for r in rows]))
    common_step = min(r['steps'][-1] for c in conditions for r in runs[c])
    matched = {c: np.array([np.interp(common_step, r['steps'], r['records'][:, 1]) for r in runs[c]]) for c in conditions}
    common_cost = min(r['costs'][-1] for c in ['sleep', 'wake', 'frozen'] for r in runs[c])
    budgeted = {c: np.array([np.interp(common_cost, r['costs'], r['records'][:, 1]) for r in runs[c]]) for c in ['sleep', 'wake', 'frozen']}
    report['common_step'] = dict(step=int(common_step), means={c: float(v.mean()) for c, v in matched.items()},
                               sleep_minus_wake=interval(matched['sleep']-matched['wake']))
    report['common_proposal_budget'] = dict(proposals=int(common_cost), means={c: float(v.mean()) for c, v in budgeted.items()},
                                          sleep_minus_wake=interval(budgeted['sleep']-budgeted['wake']),
                                          note='Counts include discarded dream draws in controls; not wall-clock cost or operation count.')
    em = [np.load(source/f'em_{i}.npz') for i in range(n)]
    report['maximum_em_nll_increase'] = float(max(np.diff(r['records'][:, 0]).max() for r in em))
    assert report['maximum_em_nll_increase'] < 1e-10
    report['stopping'] = {c: [dict(step=int(r['steps'][-1]), reason=str(r['status'])) for r in runs[c]] for c in conditions}
    report['reconstruction'] = 'Exactly zero for every compatible hidden path, by the emission mask. This is a structural property of the toy, not evidence of a learned compression representation.'
    report['audit_harness_correction'] = 'The first audit standardized floating-point roundoff in theoretically constant score coordinates. Those coordinates are now checked by absolute error; nonconstant coordinates use standard errors. No learning estimator changed. The failed source is preserved in audit/source_failed.py.'
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    table = [r'\begin{center}', r'\begin{tabular}{lrr}', r'\toprule',
             r'Condition & Data NLL & True KL\\', r'\midrule']
    initial = np.mean([r['initial'] for r in rows], axis=0)
    table.append(f'Initial single-path count fit & {initial[0]:.5f} & {initial[1]:.5f}'+r'\\')
    for name, label in [('sleep', 'Recognition from dreams'), ('wake', 'Recognition from old data'), ('frozen', 'Frozen recognition'), ('exact', 'Exact likelihood gradient')]:
        v = report['endpoint_means'][name]
        table.append(f'{label} & {v[0]:.5f} & {v[1]:.5f}'+r'\\')
    for name, label in [('order_1', 'Direct bigram counts'), ('order_2', 'Direct order-two counts'), ('order_3', 'Direct order-three counts'), ('hard_em', 'Single-path Viterbi fitting'), ('em', 'Direct posterior EM')]:
        v = np.mean([r['controls'][name] for r in rows], axis=0)
        table.append(f'{label} & {v[0]:.5f} & {v[1]:.5f}'+r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{center}', ''])
    table.extend(['Sleep recognition minus each control, in true predictive KL:', r'\begin{center}', r'\begin{tabular}{lrrr}', r'\toprule', r'Control & Mean difference & Lower 95\% & Upper 95\%\\', r'\midrule'])
    for name, label in [('wake', 'Wake recognition'), ('frozen', 'Frozen recognition'), ('order_3', 'Order-three counts'), ('em', 'Posterior EM')]:
        v = report['sleep_minus_control'][name]
        table.append(f"{label} & ${v['mean']:.5f}$ & ${v['bootstrap95'][0]:.5f}$ & ${v['bootstrap95'][1]:.5f}$"+r'\\')
    table.extend([r'\bottomrule', r'\end{tabular}', r'\end{center}'])
    table.append('These paired bootstrap intervals over eight worlds are descriptive and unadjusted for multiple comparisons. Endpoint comparisons include different stopping times; they are not estimates at a common training budget.\n')
    for key, label, budget in [('common_step', 'updates', 'step'), ('common_proposal_budget', 'generated proposals', 'proposals')]:
        v = report[key]
        table.append(f"At {v[budget]:,} common {label}, mean predictive KL is ${v['means']['sleep']:.5f}$ for sleep recognition and ${v['means']['wake']:.5f}$ for wake recognition.\n")
    table.append('Proposal counts are a sampling-cost proxy, not measured runtime. The exact-gradient baseline can stop on a shallow plateau before a stochastic trajectory leaves that region; its endpoint should not be interpreted as the maximum achievable likelihood.\n')
    (Path(__file__).resolve().parents[1]/'tex/archive/practice_hidden_sequences_results.tex').write_text('\n'.join(table)+'\n')
    model.plot(source, n)
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for method, label in [('em', 'Exact posterior EM'), ('hard_em', 'Single-path fit')]:
        values = [np.load(source/f'{method}_{i}.npz')['records'] for i in range(n)]
        steps = np.arange(max(map(len, values)))
        for m, ax in enumerate(axes):
            means = np.mean([np.interp(steps, np.arange(len(v)), v[:, m]) for v in values], axis=0)
            ax.plot(steps, means, label=label)
    axes[0].set(ylabel='Observed-data NLL (nats)')
    axes[1].set(ylabel='True predictive KL (nats)', xlabel='Direct fitting iterations; stopped models held fixed')
    axes[0].legend()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-hidden-sequences-direct-fit.png')
    plt.close(fig)
    fig, ax = plt.subplots(layout='constrained')
    for c, label in [('sleep', 'Recognition from dreams'), ('wake', 'Recognition from old-data posteriors'), ('frozen', 'Frozen recognition')]:
        steps = np.arange(0, max(r['steps'][-1] for r in runs[c])+1, model.CONFIG['eval_every'])
        ax.plot(steps, np.mean([np.interp(steps, r['steps'], r['records'][:, 2]) for r in runs[c]], axis=0), label=label)
    ax.set(ylabel='Cause-posterior KL estimate (nats)', xlabel='Offline update budget; stopped models held fixed')
    ax.legend()
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-hidden-sequences-recognition.png')
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, layout='constrained')
    world = np.load(source/'world_0.npz')
    for ax, w, title in zip(axes, [world['initial_w'], runs['sleep'][0]['w'][-1]], ['After fitting one valid path per observation', 'After learning with generated practice']):
        color = ax.imshow(model.softmax(w), vmin=0, vmax=1, aspect='auto')
        ax.set(title=title, xlabel='Next hidden state', ylabel='Current state', xticks=np.arange(4), yticks=np.arange(5), yticklabels=['0', '1', '2', '3', 'start'])
    fig.colorbar(color, ax=axes, label='Transition probability')
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-hidden-sequences-weights.png')
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ['endpoint_means', 'sleep_minus_control', 'common_step', 'common_proposal_budget', 'maximum_em_nll_increase']}, indent=2))


if __name__ == '__main__': main()
