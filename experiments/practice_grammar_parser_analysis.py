"""Paired outcomes, common-budget checks and interpretable grammar weights."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np

import practice_grammar_parser as model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--moments', type=Path)
    parser.add_argument('--write-tex', action='store_true')
    args = parser.parse_args()
    output = args.output if args.output is not None else args.source.parent/(args.source.name+'_analysis')
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.source/'summary.json').read_text())
    conditions = list(rows[0]['conditions'])
    n = len(rows)
    runs = {c: [np.load(args.source/f'{c}_{i}.npz') for i in range(n)] for c in conditions}
    bootstrap = np.random.default_rng(20102902).integers(n, size=(20000, n))
    def stats(v): return dict(mean=float(v.mean()), bootstrap95=np.quantile(v[bootstrap].mean(axis=-1), [.025, .975]).tolist(), per_world=v.tolist())
    final = {c: np.array([r['records'][-1] for r in runs[c]]) for c in conditions}
    initial = np.array([r['records'][0] for r in runs['sleep']])
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), source=str(args.source), worlds=n,
        metric_columns=['training NLL', 'true predictive KL', 'current-generator posterior recognition KL', 'dream recognition cross entropy', 'observed-data posterior recognition KL'],
        initial_mean=initial.mean(axis=0).tolist(), final_means={k: v.mean(axis=0).tolist() for k, v in final.items()},
        endpoint_sleep_minus={c: stats(final['sleep'][:, 1]-final[c][:, 1]) for c in conditions if c != 'sleep'},
        sleep_minus_direct={name: stats(final['sleep'][:, 1]-np.array([r['direct'][name][1] for r in rows])) for name in rows[0]['direct']})
    sampled = [c for c in conditions if c != 'exact']
    result['common_budget'] = {}
    for axis in ['steps', 'costs', 'wall_seconds']:
        budgets = np.array([min(runs[c][i][axis][-1] for c in sampled) for i in range(n)])
        values = {c: np.array([np.interp(budgets[i], runs[c][i][axis], runs[c][i]['records'][:, 1]) for i in range(n)]) for c in sampled}
        result['common_budget'][axis] = dict(budgets=budgets.tolist(), predictive_kl={c: stats(v) for c, v in values.items()},
                                            sleep_minus={c: stats(values['sleep']-values[c]) for c in sampled if c != 'sleep'})
    result['fixed_budget'] = {}
    for axis, budget in [('steps', 500), ('wall_seconds', 30), ('costs', 100000)]:
        values = {c: np.array([np.interp(budget, runs[c][i][axis], runs[c][i]['records'][:, 1]) for i in range(n)]) for c in sampled}
        result['fixed_budget'][axis] = dict(budget=budget, predictive_kl={c: stats(v) for c, v in values.items()},
                                           sleep_minus={c: stats(values['sleep']-values[c]) for c in sampled if c != 'sleep'})
    result['budget_caution'] = 'Proposal counts omit unequal table/score costs. Wall times include evaluation and are implementation-specific. The original pilot also computed unused dreams in wake/frozen arms, so its wall-time comparison is not an algorithm-efficiency claim.'
    result['diagnostics'] = {c: dict(mean_proposals_per_step=float(np.mean([r['diagnostics'][:, 0].mean() for r in runs[c]])),
        mean_meeting=float(np.mean([r['diagnostics'][:, 2].mean() for r in runs[c]])), maximum_meeting=int(max(r['diagnostics'][:, 3].max() for r in runs[c])),
        median_gradient_variance=float(np.median(np.concatenate([r['diagnostics'][:, 4] for r in runs[c]])))) for c in sampled}
    result['stopping'] = {c: [dict(step=int(r['steps'][-1]), status=str(r['status'])) for r in runs[c]] for c in conditions}
    if args.moments is not None:
        moments = json.loads((args.moments/'summary.json').read_text())
        assert len(moments['rows']) == n
        moment_kl = np.array([r['predictive_kl'] for r in moments['rows']])
        moment_cv_kl = np.array([r['cv_predictive_kl'] for r in moments['rows']])
        result['structural_controls'] = dict(moment=stats(moment_kl), augmented_cv=stats(moment_cv_kl),
            sleep_minus_moment=stats(final['sleep'][:, 1]-moment_kl), sleep_minus_augmented_cv=stats(final['sleep'][:, 1]-moment_cv_kl))
        fig, axes = plt.subplots(2, 1, layout='constrained')
        for c in conditions:
            grid = np.arange(0, max(r['steps'][-1] for r in runs[c])+1, model.CONFIG['eval_every'])
            for column, ax in zip([1, 2], axes):
                ax.plot(grid, np.mean([np.interp(grid, r['steps'], r['records'][:, column]) for r in runs[c]], axis=0), label=c)
        axes[0].axhline(moment_cv_kl.mean(), linestyle=':', color='black', label='CV direct fit including structural moments')
        axes[0].axhline(moment_kl.mean(), linestyle='--', color='C4', label='Analytic structural moments')
        axes[0].set(ylabel='True predictive KL (nats)')
        axes[1].set(ylabel='Causal posterior recognition KL', xlabel='Updates; stopped models held fixed')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=2)
        fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-grammar-parser-learning.png')
        plt.close(fig)
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for c in conditions:
        grid = np.arange(0, max(r['steps'][-1] for r in runs[c])+1, model.CONFIG['eval_every'])
        axes[0].plot(grid, np.mean([np.interp(grid, r['steps'], r['records'][:, 0]) for r in runs[c]], axis=0), label=c)
        contrast = []
        for r in runs[c]:
            values = np.array([np.linalg.norm(model.generator_probabilities(w)[1][0]-model.generator_probabilities(w)[1][1]) for w in r['w']])
            contrast.append(np.interp(grid, r['steps'], values))
        axes[1].plot(grid, np.mean(contrast, axis=0), label=c)
    axes[0].set(ylabel='Training NLL (nats)')
    axes[1].set(ylabel='Difference between parent rules', xlabel='Updates; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=4)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-grammar-parser-training.png')
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, layout='constrained')
    world = np.load(args.source/'world_0.npz')
    matrices = [model.generator_probabilities(world['initial'])[1], model.generator_probabilities(runs['sleep'][0]['w'][-1])[1], world['rules']]
    for ax, matrix, title in zip(axes, matrices, ['Initial generator', 'After own-generation practice', 'Teacher (evaluation only)']):
        picture = ax.imshow(matrix.reshape(2, 4), vmin=0, vmax=1, aspect='auto')
        ax.set(title=title, xticks=range(4), xticklabels=['00', '01', '10', '11'], yticks=[0, 1], ylabel='Parent category')
    axes[-1].set_xlabel('Child categories')
    fig.colorbar(picture, ax=axes, label='Production probability')
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-grammar-parser-weights.png')
    plt.close(fig)
    if args.write_tex:
        contrast = result['fixed_budget']['steps']['sleep_minus']
        def estimate(v): return f"{v['mean']:+.4f} [{v['bootstrap95'][0]:+.4f}, {v['bootstrap95'][1]:+.4f}]"
        table = [r'\begin{center}', r'\begin{tabular}{lr}', r'\toprule', r'Method & Mean predictive KL \\', r'\midrule',
                 f"Initial generator & {initial[:, 1].mean():.4f}" + r' \\']
        for c in conditions: table.append(f"{c.capitalize()} recognition & {final[c][:, 1].mean():.4f}" + r' \\')
        table += [f"Original CV direct fit & {np.mean([r['direct']['cv_selected'][1] for r in rows]):.4f}" + r' \\',
                  f"Analytic structural moments & {moment_kl.mean():.4f}" + r' \\',
                  f"CV including structural moments & {moment_cv_kl.mean():.4f}" + r' \\',
                  r'\bottomrule', r'\end{tabular}', r'\end{center}',
                  f"On {n} fresh worlds, the predeclared primary contrast at 500 updates, sleep minus wake, is {estimate(contrast['wake'])} nats.",
                  f"The corresponding sleep-minus-frozen contrast is {estimate(contrast['frozen'])} nats.",
                  f"At internally stopped endpoints, sleep minus the augmented direct control is {estimate(result['structural_controls']['sleep_minus_augmented_cv'])} nats.",
                  r'Intervals are paired 95\% bootstrap intervals across worlds.']
        (Path(__file__).resolve().parents[1]/'tex/archive/practice_grammar_parser_results.tex').write_text('\n'.join(table)+'\n')
    print(json.dumps({k: result[k] for k in ['source', 'initial_mean', 'final_means', 'endpoint_sleep_minus', 'common_budget']}, indent=2))


if __name__ == '__main__': main()
