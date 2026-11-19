"""Figure and summary numbers for dbm_dream_depth.py: learning curves per negative-phase source, and the deep drive."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dbm_dream_depth as model

ARMS = dict(shallow='one layer only (deep weights held at 0)', dream='fresh own samples', cd5='5 sweeps from the teacher sample', cd1='1 sweep from the teacher sample', twin="a twin's fresh samples", edge='no samples: tanh of each weight')


def load(*directories):
    data = [json.loads((Path(d) / 'results.json').read_text()) for d in directories]
    return data[0] | dict(runs=[run for d in data for run in d['runs']])


def curves(data, arm, key, examples=0):
    runs = [c for run in sorted(data['runs'], key=lambda run: run['seed']) if run['arm'] == arm and run['examples'] == examples for c in run['curves']]
    steps = max((c['step'] for c in runs), key=len)
    out = np.full((len(runs), len(steps)), np.nan)
    for row, c in zip(out, runs):
        row[:len(c[key])] = c[key]
    return np.array(steps), out


def onset(data, arm, examples=0):
    """Updates until the population KL is below half of what the one-layer learner of the same world and start ends at."""
    steps, floor = curves(data, 'shallow', 'kl', examples)
    _, values = curves(data, arm, 'kl', examples)
    below = values < .5 * floor[:, -1:]
    return np.where(below.any(1), steps[np.argmax(below, 1)], np.inf)


def closed_form(data, run, j):
    """Deep drive of k data-started sweeps over that of fresh samples, measured and from (1 - rho_a^k rho_b^(k-1)) C_ab."""
    args = data['args']
    s = model.Shape(args['n'], args['m'], args['k'])
    w1, w2 = (np.array(w) for w in run['weights'][j])
    pdata = model.teacher_tables(s, *(np.array(w) for w in run['teacher']))['px']
    a, f2, post, px = model.tables(s, w1, w2)
    rho = px @ (post @ s.H1) ** 2
    code = ((pdata @ post) @ s.outer1).reshape(s.m, s.m) - np.eye(s.m)
    top = np.linalg.eigvalsh(code)[-1]
    out = dict(reread=float(rho.mean()))
    for k in (1, 5):
        out[f'predicted{k}'] = float(np.linalg.eigvals((1 - np.outer(rho ** k, rho ** (k - 1))) * code).real.max() / top)
        out[f'measured{k}'] = run['curves'][j][f'drive_cd{k}'][-1] / run['curves'][j]['drive_dream'][-1]
    return out


def panel(ax, data, title, examples=0):
    for arm, label in ARMS.items():
        if not any(run['arm'] == arm and run['examples'] == examples for run in data['runs']):
            continue
        steps, values = curves(data, arm, 'kl', examples)
        values = np.where(np.isfinite(values), values, np.inf)
        shown = np.isfinite(values).mean(0) > .5
        line, = ax.plot(steps[shown], np.nanmedian(values, 0)[shown], label=label)
        ax.fill_between(steps[shown], *np.nanquantile(values, [.25, .75], 0)[:, shown], color=line.get_color(), alpha=.2, lw=0)
    start = np.nanmedian(curves(data, 'dream', 'kl', examples)[1][:, 0])
    ax.set(xscale='log', yscale='log', xlabel='updates', ylabel='KL(teacher || learner) (nats)', title=title)
    ax.set_ylim(top=3 * start)
    ax.set_xlim(left=1)


def summary(data, examples=0):
    out = dict(teacher_reread=float(np.mean([run['teacher_reread'] for run in data['runs']])))
    for arm in [arm for arm in ARMS if any(run['arm'] == arm and run['examples'] == examples for run in data['runs'])]:
        steps, values = curves(data, arm, 'kl', examples)
        finite = np.where(np.isfinite(values), values, np.inf)
        last = np.array([row[np.isfinite(row)][-1] if np.isfinite(row).any() else np.inf for row in values])
        blown = int(np.sum([run['diverged'] for run in data['runs'] if run['arm'] == arm and run['examples'] == examples]))
        out[arm] = dict(final_median=float(np.median(last)), best_median=float(np.median(np.nanmin(finite, 1))), onset_median=float(np.median(onset(data, arm, examples))), onset_reached=int(np.isfinite(onset(data, arm, examples)).sum()),
                        runs=len(values), worlds_diverged=blown, read2_final=float(np.nanmedian(curves(data, arm, 'read2', examples)[1][:, -1])), read1_final=float(np.nanmedian(curves(data, arm, 'read1', examples)[1][:, -1])),
                        norm2_final=float(np.nanmedian(curves(data, arm, 'norm2', examples)[1][:, -1])))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dense', nargs='+', required=True)
    parser.add_argument('--tree', nargs='+', required=True)
    parser.add_argument('--scan', nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dense, tree, scan = load(*args.dense), load(*args.tree), [load(d) for d in args.scan]
    width, height = plt.rcParams['figure.figsize']
    fig, axes = plt.subplots(1, 3, figsize=(3 * width, height), layout='constrained')
    panel(axes[0], dense, 'every visible unit connected to every h1 unit')
    panel(axes[1], tree, 'each h1 unit has its own visible units')
    axes[0].legend()
    grid = np.linspace(0, 1, 200)
    rows = [closed_form(d, run, j) for d in scan for run in d['runs'] if run['arm'] == 'shallow' and run['examples'] == 0 for j in range(2)]
    reread = np.array([row['reread'] for row in rows])
    for sweeps, marker in [(1, 'o'), (5, 's')]:
        points = axes[2].scatter(reread, [row[f'measured{sweeps}'] for row in rows], marker=marker, alpha=.5, label=f'{sweeps} sweep{"s" * (sweeps > 1)} from the teacher sample')
        axes[2].scatter(reread, [row[f'predicted{sweeps}'] for row in rows], marker='+', color='k')
        axes[2].plot(grid, 1 - grid ** (2 * sweeps - 1), color=points.get_facecolor()[0], alpha=1)
    axes[2].scatter([], [], marker='+', color='k', label="closed form, the run's own units")
    axes[2].plot([], [], color='gray', label='closed form, all units alike')
    axes[2].set(xlabel='how reliably the first layer rereads its own code', ylabel='growth rate of small deep weights,\nrelative to fresh own samples', title='own visible units, first layer learned', ylim=(0, None))
    axes[2].legend()
    check = max(abs(row[f'measured{k}'] - row[f'predicted{k}']) for row in rows for k in (1, 5))
    fig.savefig(args.out)
    numbers = dict(closed_form_max_error=check, dense=summary(dense), dense_512=summary(dense, 512), tree=summary(tree), tree_512=summary(tree, 512), scan=[dict(g1=d['args']['g1']) | summary(d) for d in scan])
    args.out.with_suffix('.json').write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))
    print(args.out.resolve())


if __name__ == '__main__':
    main()
