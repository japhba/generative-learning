"""Figure and summary numbers for hebb_practice_depth.py and its single-sample and cross-layer versions."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

HUE = dict(watch='#2a78d6', reread='#eb6834', kept='#1baf7a', lateral='#6250d6', ws='#e87ba4', reference='#52514e')
ARMS = dict(watch=('watch', '-', 'teacher samples only (Hebb)'), reread=('reread', '-', '+ own samples, re-read'), reread_shuffle=('reread', '--', '+ first-layer samples, re-read'),
            reread_twin=('reread', ':', "+ a twin's samples, re-read"), kept=('kept', '-', '+ own samples, generating pass'), kept_only=('kept', '--', 'own generating pass, no Hebb on teacher samples'),
            lateral=('lateral', '-', '+ lateral inhibition, no own samples'), ws=('ws', '-', 'wake-sleep'), watch_ws=('ws', '--', 'wake-sleep + Hebb on teacher samples'),
            exact=('reference', '-', 'likelihood ascent, exact posterior'), exact_shallow=('reference', '--', 'the same without the deep layer'), exact_one=('reference', ':', 'the same with one deep unit'))


def load(directory):
    return json.loads((Path(directory) / 'results.json').read_text())


def curve(run):
    return run['curves'] if isinstance(run['curves'], dict) else run['curves'][0]


def pick(data, arm, stages, share=1., kind='blank', amplitude=0., k=None):
    k = data['args']['k'] if k is None else k
    keep = lambda run: len(run['stages']) == stages and run['stages'][-1][0] == arm and run.get('share', 1.) == share and run.get('kind', 'blank') == kind and run.get('amplitude', 0.) == amplitude and run.get('k', k) == k
    return sorted(filter(keep, data['runs']), key=lambda run: run['seed'])


def series(runs, key, last_stage=True):
    """Steps counted from the start of the last stage, and one row per world."""
    stage = np.array(curve(runs[0])['stage'])
    keep = stage == stage.max() if last_stage else np.ones(len(stage), bool)
    steps = np.array(curve(runs[0])['step'])[keep]
    return steps - steps[0], np.array([np.squeeze(np.array(curve(run)[key])[keep]) for run in runs])


def merged(run):
    """Number of pairs of deep units whose readings of a teacher sample agree (or disagree) in more than three of four cases at the end."""
    return int((np.abs(np.atleast_1d(curve(run)['unit_corr'][-1])) > .5).sum())


def table(data, arms, stages=2, **where):
    out = {}
    for arm in arms:
        runs = pick(data, arm, stages, **where)
        if not runs:
            continue
        _, loss = series(runs, 'kl')
        _, gap = series(runs, 'prior_gap')
        _, read = series(runs, 'read')
        out[arm] = dict(worlds=len(runs), kl_start=float(np.median(loss[:, 0])), kl_end=float(np.median(loss[:, -1])), kl_end_worlds=np.round(loss[:, -1], 4).tolist(), halved=int((loss[:, -1] < .5 * loss[:, 0]).sum()),
                        no_merged_units=int(sum(merged(run) == 0 for run in runs)), prior_matched=int((gap[:, -1] < .1).sum()), prior_gap_end=float(np.median(gap[:, -1])), read_end=float(np.median(read[:, -1])))
    return out


def growth(data, arm):
    """Per world: measured growth rate of the difference between the two deep units while it is small, and the closed-form rate over the same steps.

    Steps from 60 after the fork (the part of the state that both units share has settled by then) for as long as the difference is below 0.02; of these the later half, where the fastest mode leads.
    """
    rate, rows = data['args']['rate'], []
    for run in pick(data, arm, 2):
        steps, split = series([run], 'split')
        _, predicted = series([run], 'rate_' + arm)
        split, predicted = split[0], predicted[0]
        keep = np.flatnonzero((steps[:-1] >= 60) & (np.maximum.accumulate(split)[1:] < .02) & (split[1:] > 1e-11))
        slope = np.diff(np.log(np.maximum(split, 1e-300))) / (np.diff(steps) * rate)
        keep = keep[len(keep) // 2:]
        if len(keep) >= 4:
            rows.append((run['seed'], float(np.median(slope[keep])), float(np.median(((predicted[:-1] + predicted[1:]) / 2)[keep]))))
    return rows


def lines(ax, data, arms, key, stages):
    for arm in arms:
        runs = pick(data, 'exact' if arm == 'exact_one' else arm, stages, k=1 if arm == 'exact_one' else None)
        hue, style, label = ARMS[arm]
        steps, values = series(runs, key)
        ax.plot(np.maximum(steps, .5), np.median(values, 0), color=HUE[hue], linestyle=style, label=label)


def sina(ax, position, values, color, low, high, filled=True):
    """Dots spread by a kernel density of fixed width (0.05 of the axis range), over a box from the smallest to the largest value with the median and quartiles."""
    logs = np.log10(values)
    kernel = gaussian_kde(logs, bw_method=lambda kde: .05 * (high - low) / max(kde.dataset.std(), 1e-6))
    grid = np.linspace(low, high, 400)
    top = kernel(grid).max()
    ax.fill_betweenx(10 ** grid, position - .4 * kernel(grid) / top, position + .4 * kernel(grid) / top, color=color, alpha=.15, linewidth=0)
    jitter = np.random.default_rng(0).uniform(-1, 1, len(logs)) * .4 * kernel(logs) / top
    ax.scatter(position + jitter, values, facecolor=color if filled else 'none', edgecolor=color, zorder=3)
    ax.add_patch(plt.Rectangle((position - .42, values.min()), .84, values.max() - values.min(), fill=False, edgecolor=HUE['reference']))
    ax.plot([position - .42, position + .42], [np.median(values)] * 2, color=HUE['reference'])
    for quartile in np.quantile(values, [.25, .75]):
        ax.plot([position - .12, position + .12], [quartile] * 2, color=HUE['reference'])


def main():
    parser = argparse.ArgumentParser()
    for name in ('main', 'rates', 'linear', 'sampled', 'three', 'skip'):
        parser.add_argument('--' + name)
    parser.add_argument('--out', default=str(Path(__file__).resolve().parents[1] / 'reports' / 'hebb-practice-depth'))
    paths = parser.parse_args()
    data = load(paths.main)
    args = data['args']
    fork_arms = ['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral', 'ws', 'watch_ws']
    scratch_arms = ['watch', 'reread', 'kept', 'kept_only', 'lateral', 'ws', 'watch_ws', 'exact', 'exact_shallow']
    numbers = dict(checks=data['checks'], fork=table(data, fork_arms), scratch=table(data, scratch_arms, stages=1), one_deep_unit=table(data, ['exact'], stages=1, k=1),
                   shares={share: table(data, ['reread', 'kept'], share=share) for share in args['shares']},
                   started={f'{kind} {amplitude}': table(data, args['started'], stages=1, kind=kind, amplitude=amplitude) for kind in ('coupled', 'decoupled') for amplitude in args['amplitudes']})
    marks = [100, 200, 300, 500, 1000, 2000, 12000]
    numbers['scratch_kl_at'] = dict(steps=marks) | {arm: [float(np.median(series(pick(data, arm, 1), 'kl')[1][:, list(series(pick(data, arm, 1), 'kl')[0]).index(mark)])) for mark in marks] for arm in scratch_arms}
    numbers['reach_at_fork'] = {key: float(np.median(series(pick(data, 'watch', 2), key)[1][:, 0])) for key in ('reach_own', 'read', 'prior_gap', 'undecided', 'spread')}
    numbers['reach_after_kept_only'] = {key: float(np.median(series(pick(data, 'kept_only', 2), key)[1][:, -1])) for key in ('reach_own', 'read')}

    figure, axes = plt.subplots(2, 3, figsize=(20, 11), constrained_layout=True)
    ax = axes[0, 0]
    lines(ax, data, ['watch', 'lateral', 'kept_only', 'ws', 'watch_ws', 'exact', 'exact_shallow', 'exact_one'], 'kl', 1)
    ax.set(xscale='log', yscale='log', xlabel='teacher samples', ylabel=r'KL(teacher $\Vert$ learner), nats', title='From a blank start: Hebb on teacher samples is early, then stalls')
    ax.legend()

    ax = axes[0, 1]
    lines(ax, data, ['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral'], 'kl', 2)
    ax.set(xscale='log', yscale='log', xlabel='teacher samples after the fork', ylabel=r'KL(teacher $\Vert$ learner), nats', title='After watching: which addition leaves the plateau')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14), ncol=2)

    ax = axes[0, 2]
    shown = ['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral', 'ws', 'watch_ws']
    finals = [series(pick(data, arm, 2), 'kl')[1][:, -1] for arm in shown] + [series(pick(data, 'exact', 1, k=1), 'kl')[1][:, -1]]
    low, high = np.log10(min(v.min() for v in finals)) - .1, np.log10(max(v.max() for v in finals)) + .1
    for position, (arm, values) in enumerate(zip(shown + ['exact_one'], finals)):
        sina(ax, position, values, HUE[ARMS[arm][0]], low, high, filled=ARMS[arm][1] == '-')
    ax.set(yscale='log', ylabel=r'KL(teacher $\Vert$ learner) at the end, nats', title='Every world at the end of the run')
    ax.set_xticks(range(len(finals)), ['teacher\nonly', 're-read\nown', 're-read\nfirst\nlayer', 'gener-\nating\npass', 'gener-\nating\npass only', 'lateral\ninhibi-\ntion', 'wake-\nsleep', 'wake-\nsleep\n+ Hebb', 'one deep\nunit,\nexact'])

    ax = axes[1, 0]
    lines(ax, data, ['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only'], 'split', 2)
    ax.set(xscale='log', yscale='log', xlabel='teacher samples after the fork', ylabel='distance between the recognition weights of the two deep units', title='The merged units come apart only with the generating pass')
    ax.set_ylim(bottom=1e-6)
    ax.legend(loc='lower left')

    ax = axes[1, 1]
    if paths.rates:
        rates = load(paths.rates)
        numbers['growth'] = {}
        for arm in ('watch', 'reread', 'kept', 'kept_only'):
            rows = growth(rates, arm)
            measured, predicted = np.array([row[1] for row in rows]), np.array([row[2] for row in rows])
            numbers['growth'][arm] = dict(worlds=len(rows), measured=np.round(measured, 4).tolist(), predicted=np.round(predicted, 4).tolist(), largest_error=float(np.abs(measured - predicted).max()))
            ax.scatter(predicted, measured, facecolor=HUE[ARMS[arm][0]] if ARMS[arm][1] == '-' else 'none', edgecolor=HUE[ARMS[arm][0]], label=ARMS[arm][2])
        aligned = [float(series([run], 'split_align')[1][0][np.argmin(np.abs(series([run], 'split')[1][0] - .03))]) for run in pick(rates, 'kept_only', 2)]
        numbers['growth']['direction_overlap_kept_only'] = np.round(aligned, 3).tolist()
        span = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(span, span, color=HUE['reference'], linestyle=':')
        ax.axhline(0, color=HUE['reference'], linewidth=plt.rcParams['axes.linewidth'])
        ax.set(xlabel='closed form', ylabel='measured', title='Growth rate of the difference between the two units')
        ax.legend()

    ax = axes[1, 2]
    groups, ticks = [], []
    for amplitude in args['amplitudes']:
        for arm in args['started']:
            groups.append((arm, series(pick(data, arm, 1, kind='coupled', amplitude=amplitude), 'kl')[1][:, -1]))
            ticks.append(f"{dict(watch='teacher only', reread='re-read own', reread_twin='re-read twin', kept='generating pass')[arm]}")
    low, high = np.log10(min(v.min() for _, v in groups)) - .1, np.log10(max(v.max() for _, v in groups)) + .1
    for position, (arm, values) in enumerate(groups):
        sina(ax, position + position // len(args['started']), values, HUE[ARMS[arm][0]], low, high, filled=ARMS[arm][1] == '-')
    ax.set_xticks([position + position // len(args['started']) for position in range(len(groups))], ticks, rotation=90)
    for index, amplitude in enumerate(args['amplitudes']):
        ax.annotate(f'start size {amplitude:g}', ((index * (len(args['started']) + 1) + (len(args['started']) - 1) / 2), 1), xycoords=('data', 'axes fraction'), xytext=(0, -4), textcoords='offset points', ha='center', va='top')
    ax.set(yscale='log', ylabel=r'KL(teacher $\Vert$ learner) at the end, nats', title='A random deep start, recognition and generation coupled')
    for ax in axes.ravel():
        ax.grid(alpha=.25)
    figure.savefig(paths.out + '.png', dpi=150)
    figure.savefig(paths.out + '.svg')

    arms = ['watch', 'reread', 'reread_shuffle', 'kept', 'kept_only', 'lateral']
    for name in ('linear', 'sampled', 'three', 'skip'):
        if getattr(paths, name):
            numbers[name] = table(load(getattr(paths, name)), arms)
    Path(paths.out + '.json').write_text(json.dumps(numbers, indent=1))
    print(paths.out + '.png')


if __name__ == '__main__':
    main()
