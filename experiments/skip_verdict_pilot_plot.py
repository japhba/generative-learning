"""Figure and summary numbers for skip_verdict_pilot.py.

Takes any number of run directories and groups them into: productions equally likely, productions not equally likely (--skew > 0),
and runs whose first layer judges by a memory of unit pairs (--verdict pairs).
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HUE = dict(teacher='#2a78d6', own='#eb6834', mixed='#1baf7a', other='#6250d6', reference='#52514e')
ARMS = dict(gate_fresh=('teacher', ':', 'teacher strings, credited only to a deep unit that already responds (gate)'), sem_fresh=('teacher', '-', 'teacher strings, credited to a deep unit drawn from the posterior'),
            em_fresh=('teacher', '--', 'teacher strings, credited by the posterior'), wta_fresh=('teacher', '-.', 'teacher strings, credited to the best deep unit'),
            sem_stored=('teacher', '--', 'stored teacher strings re-read, drawn deep unit'), em_stored=('teacher', ':', 'stored teacher strings re-read, posterior'),
            practice_w=('own', '-', 'own samples that pass the first layer, credited to their writer, weighted'), practice=('own', ':', 'the same, not weighted'),
            practice_f=('own', '--', 'own samples, graded by the first layer (unit x trace frequencies)'), pairs=('other', '-.', 'own samples, first layer judges by a memory of unit pairs (no trace)'),
            reread_own=('own', '--', 'own samples, credited by the posterior'), draw_own=('own', '-.', 'own samples, credited to a drawn deep unit'),
            draw_twin=('other', '-', "a twin's samples, credited to a drawn deep unit"), draw_probe=('other', '--', 'uniform unit pairs, credited to a drawn deep unit'), probe=('other', ':', 'uniform unit pairs, credited to a random deep unit'),
            practice_w_sem_fresh=('mixed', '-', 'half own samples, half teacher strings (drawn deep unit)'), practice_w_gate_fresh=('mixed', '--', 'half own samples, half gated teacher strings'),
            practice_w_sem_stored=('mixed', '-', 'half own samples, half stored strings (drawn deep unit)'), practice_w_gate_stored=('mixed', '--', 'half own samples, half gated stored strings'))
MARKS = (1000, 3000, 10000, 30000, 100000)


def load(directories):
    groups = dict(main=None, skew=None, pairs=None)
    for directory in directories:
        data = json.loads((Path(directory) / 'results.json').read_text())
        key = 'pairs' if data['args'].get('verdict') == 'pairs' else 'skew' if data['args']['skew'] else 'main'
        if groups[key] is None:
            groups[key] = dict(args=data['args'], runs=[])
        assert all(groups[key]['args'][name] == data['args'][name] for name in ('v', 'm', 'n0', 'k', 'eta', 'eps', 'steps', 'stored', 'skew')), directory
        groups[key]['runs'] += data['runs']
        groups[key]['args']['rhos'] = sorted(set(groups[key]['args']['rhos']) | set(data['args']['rhos'])) if data['args']['scan'] else groups[key]['args']['rhos']
    return groups


def pick(data, arm, stored, rho=None):
    rho = data['args']['n0'] if rho is None else rho
    return sorted((run for run in data['runs'] if run['arm'] == arm and run['stored'] == stored and run['rho'] == rho), key=lambda run: run['seed'])


def series(runs, key):
    return np.array(runs[0]['curves']['step']), np.array([run['curves'][key] for run in runs])


def band(ax, x, rows, arm, label=None):
    hue, style, name = ARMS[arm]
    ax.plot(x, np.median(rows, 0), style, color=HUE[hue], label=name if label is None else label)
    ax.fill_between(x, *np.quantile(rows, [.25, .75], 0), color=HUE[hue], alpha=.12, linewidth=0)


def curves(ax, data, arms, stored, title, reference_key='inversion_set', reference_name='read off the unit x trace counts (no samples)'):
    for arm in arms:
        steps, rows = series(pick(data, arm, stored), 'kl')
        band(ax, steps[1:], rows[:, 1:], arm)
    reference = pick(data, arms[0], stored)
    ax.axhline(np.median([run['curves']['kl'][0] for run in reference]), color=HUE['reference'], linestyle=':', label='first layer only (before any deep unit learns)')
    ax.axhline(np.median([run['reference'][reference_key]['kl'] for run in reference]), color=HUE['reference'], linestyle='--', label=reference_name)
    ax.set(xscale='log', yscale='log', xlabel='samples after watching', ylabel='KL(world || learner), nats', title=title)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))


def trace_panel(ax, data, stored):
    rhos = data['args']['rhos'] + [data['args']['n0']]
    end = np.array([[run['curves']['kl'][-1] for run in pick(data, 'practice_w', stored, rho)] for rho in rhos])
    target = np.array([[run['reference']['target']['kl'] for run in pick(data, 'practice_w', stored, rho)] for rho in rhos])
    start = np.median([run['curves']['kl'][0] for run in pick(data, 'practice_w', stored)])
    ax.errorbar(rhos, np.median(end, 1), yerr=np.abs(np.quantile(end, [.25, .75], 1) - np.median(end, 1)), color=HUE['own'], marker='o', label='own samples that pass the first layer (weighted), at the end')
    ax.plot(rhos, np.median(target, 1), '--', color=HUE['reference'], marker='s', label="uniform law on the pairs the first layer passes (closed form)")
    ax.axhline(start, color=HUE['reference'], linestyle=':', label='first layer only (before practice)')
    ax.set(xscale='log', xlabel='values of the trace the root writes to the surface (1: none, 16: the root is visible)', ylabel='KL(world || learner), nats', title='What the first layer can teach is what the trace says about the root')
    ax.set_xticks(rhos, [str(rho) for rho in rhos])
    ax.minorticks_off()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))


def stored_panel(ax, data, title, pairs=None):
    sizes = data['args']['stored']
    end = lambda arm, source=data: np.array([[run['curves']['kl'][-1] for run in pick(source, arm, stored)] for stored in sizes])
    for arm in ('sem_stored', 'practice_w', 'practice_f', 'practice_w_sem_stored'):
        band(ax, sizes, end(arm).T, arm)
    if pairs is not None:
        hue, style, name = ARMS['pairs']
        ax.plot(sizes, np.median(end('practice_w', pairs), 1), style, color=HUE[hue], label=name)
    band(ax, sizes, end('sem_fresh').T, 'wta_fresh', 'as many fresh teacher strings instead (drawn deep unit)')
    for key, style, name in (('inversion', '-.', 'read off the unit x trace counts, by frequency'), ('inversion_set', '--', 'the same, uniform over the units seen with a trace')):
        ax.plot(sizes, [np.median([run['reference'][key]['kl'] for run in pick(data, 'practice_w', stored)]) for stored in sizes], style, color=HUE['reference'], label=name)
    ax.plot(sizes, [np.median([run['curves']['kl'][0] for run in pick(data, 'practice_w', stored)]) for stored in sizes], ':', color=HUE['reference'], label='first layer only')
    ax.set(xscale='log', yscale='log', xlabel='stored teacher strings (watching)', ylabel='KL(world || learner) after 100,000 further samples, nats', title=title)
    ax.set_xticks(sizes, [str(size) for size in sizes])
    ax.minorticks_off()
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))


def numbers(data):
    steps = np.array(data['runs'][0]['curves']['step'])
    at = [int(np.argmin(np.abs(steps - mark))) for mark in MARKS]
    out = {}
    for stored in data['args']['stored']:
        for rho in data['args']['rhos'] + [data['args']['n0']]:
            for arm in sorted({run['arm'] for run in data['runs']}):
                runs = pick(data, arm, stored, rho)
                if not runs:
                    continue
                kl = series(runs, 'kl')[1]
                median = np.median(kl, 0)
                last = lambda key: float(np.median([run['curves'][key][-1] for run in runs]))
                out[f'{arm}|stored={stored}|rho={rho}'] = dict(kl_start=float(median[0]), kl_at={str(mark): float(median[i]) for mark, i in zip(MARKS, at)}, kl_end_quartiles=[float(v) for v in np.quantile(kl[:, -1], [.25, .75])], kl_end_worst=float(kl[:, -1].max()),
                                                           samples_to_half_start=int(steps[np.argmax(median < median[0] / 2)]) if (median < median[0] / 2).any() else None, valid=last('valid'), covered=last('covered'), read=last('read'), units_valid=last('units_valid'),
                                                           accept_start=float(np.median([run['curves']['accept'][0] for run in runs])), precision_start=float(np.median([run['curves']['precision'][0] for run in runs])), valid_start=float(np.median([run['curves']['valid'][0] for run in runs])),
                                                           reference={key: float(np.median([run['reference'][key]['kl'] if isinstance(run['reference'][key], dict) else run['reference'][key] for run in runs])) for key in runs[0]['reference']})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directories', nargs='+')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'reports' / 'skip-verdict-pilot'))
    paths = parser.parse_args()
    groups = load(paths.directories)
    data, skew = groups['main'], groups['skew']
    most = max(data['args']['stored'])
    figure, axes = plt.subplots(2, 3, figsize=(22, 16), constrained_layout=True)
    curves(axes[0, 0], data, ['gate_fresh', 'sem_fresh', 'em_fresh', 'wta_fresh', 'practice_w', 'practice'], most, 'Teacher strings or own samples (root visible in the trace)')
    curves(axes[0, 1], data, ['practice_w', 'reread_own', 'draw_twin', 'draw_probe', 'probe', 'practice_w_sem_fresh', 'practice_w_gate_fresh'], most, 'Whose samples, and who is credited')
    trace_panel(axes[0, 2], data, most)
    stored_panel(axes[1, 0], data, 'A stored teacher set, productions equally likely', groups['pairs'])
    stored_panel(axes[1, 1], skew, 'A stored teacher set, productions not equally likely')
    curves(axes[1, 2], skew, ['sem_fresh', 'sem_stored', 'practice_w', 'practice_f', 'practice_w_sem_stored'], 512, 'Productions not equally likely, 512 stored strings', 'inversion', 'read off the unit x trace counts, by frequency (no samples)')
    figure.savefig(paths.out + '.png', dpi=150)
    figure.savefig(paths.out + '.svg')
    Path(paths.out + '.json').write_text(json.dumps({key: dict(args=group['args'], numbers=numbers(group)) for key, group in groups.items() if group is not None}, indent=1))
    print(paths.out + '.png')


if __name__ == '__main__':
    main()
