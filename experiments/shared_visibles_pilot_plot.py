"""Figure and summary numbers for shared_visibles_pilot.py."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HUE = dict(teacher='#2a78d6', own='#eb6834', mixed='#1baf7a', other='#6250d6', reference='#52514e')


def load(paths):
    data = json.loads(Path(paths[0]).read_text())
    runs = defaultdict(list)
    for path in paths:
        more = json.loads(Path(path).read_text())
        assert more['args'] | dict(only='', out='') == data['args'] | dict(only='', out=''), path
        for r in more['runs']:
            runs[(r['kind'], r['f'], r['arm'])].append(r)
    refs = defaultdict(list)
    for r in data['references']:
        refs[(r['kind'], r['f'])].append(r)
    return data, runs, refs


def curve(runs, key, field, index=None):
    rows = sorted(runs[key], key=lambda r: r['seed'])
    x = np.array([c['samples'] for c in rows[0]['curve']])
    y = np.array([[c[field] if index is None else c[field][index] for c in r['curve']] for r in rows], float)
    return x, y


def band(ax, runs, key, field, label, color, style='-', index=None):
    x, y = curve(runs, key, field, index)
    ax.plot(np.maximum(x, x[1] / 2), np.median(y, 0), style, color=color, label=label)
    ax.fill_between(np.maximum(x, x[1] / 2), np.quantile(y, .25, 0), np.quantile(y, .75, 0), color=color, alpha=.12, lw=0)


def end(runs, key, field, index=None):
    return curve(runs, key, field, index)[1][:, -1]


def half_way(runs, key):
    x, y = curve(runs, key, 'kl')
    return [float(x[np.argmax(row <= row[0] / 2)]) if (row <= row[0] / 2).any() else None for row in y]


def med(values):
    values = [v for v in values if v is not None]
    return None if not values else float(np.median(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+')
    parser.add_argument('--out', default=str(Path(__file__).resolve().parent.parent / 'reports' / 'shared-visibles-pilot'))
    paths = parser.parse_args()
    data, runs, refs = load(paths.results)
    args, const = data['args'], data['constants']
    ring = ('ring', 1.)
    S = const['L']
    blank = float(np.median([r['blank']['kl'] for r in refs[ring]]))
    figure, axes = plt.subplots(3, 3, figsize=(26, 27), constrained_layout=True)

    ax = axes[0, 0]
    hist = {name: np.sum([r[name] for r in refs[ring]], 0) for name in ('teacher_match_hist', 'teacher_best_hist', 'own_match_hist')}
    for name, color, label in (('teacher_match_hist', HUE['teacher'], 'teacher string, any one deep unit'), ('teacher_best_hist', HUE['other'], f"teacher string, its best of {args['k']} deep units"), ('own_match_hist', HUE['own'], 'own sample, the deep unit that wrote it')):
        ax.plot(np.arange(S + 1), hist[name] / hist[name].sum(), 'o-', color=color, label=label)
    p_start = np.exp(args['scale']) / (np.exp(args['scale']) + const['V'] - 1)
    need = int(np.ceil((args['theta'] - 1 / const['V'] * 0) * S / p_start - 1e-9))
    ax.axvline(need - .5, color=HUE['reference'], ls=':', label=f'a deep unit fires right of this line')
    ax.set(xlabel=f'slots (of {S}) in which the code read is the unit\'s own', ylabel='share', title='Start of Practice: how strongly a string drives a deep unit')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    ax = axes[0, 1]
    for name, color, style, label in (('own_accept', HUE['own'], '-', 'credited only if the first layer reads every slot (step 3)'), ('own_accept_eta1.0', HUE['own'], '--', 'the same, step 1'), ('own_accept_eta0.3', HUE['own'], ':', 'the same, step 0.3'),
                                      ('own_plain', HUE['mixed'], '-', 'every sample credited (plain wake step, step 3)'), ('own_plain_eta1.0', HUE['mixed'], '--', 'plain wake step, step 1'), ('own_plain_eta0.3', HUE['mixed'], ':', 'plain wake step, step 0.3'),
                                      ('own_drive', HUE['other'], '-', 'weighted by how strongly the sample drives its writer'), ('own_read', HUE['other'], '--', 'credited to the deep unit that fires on reading')):
        band(ax, runs, ring + (name,), 'kl', label, color, style)
    ax.axhline(blank, color=HUE['reference'], ls=':', label='deep layer blank')
    ax.set(xscale='log', xlabel='own samples', ylabel='KL(teacher || learner), nats', title='Practice: which own samples are credited to their writer')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    ax = axes[0, 2]
    for name, color, style, label in (('own_accept', HUE['own'], '-', 'own samples (credited to the writer, first layer reads every slot)'), ('own_accept_diffuse', HUE['own'], '--', 'own samples, deep layer starts diffuse'),
                                      ('teacher_gate', HUE['teacher'], ':', 'teacher strings, a deep unit must fire'), ('teacher_wta', HUE['teacher'], '-', 'teacher strings, best deep unit always learns'),
                                      ('teacher_recruit', HUE['teacher'], '-.', 'teacher strings, unread ones stored in an unused unit'), ('stored_wta', HUE['teacher'], '--', f"{args['stored']} stored teacher strings re-read, best unit learns"),
                                      ('twin_gate', HUE['other'], ':', "a twin's samples, a deep unit must fire"), ('twin_wta', HUE['other'], '-', "a twin's samples, best deep unit always learns"),
                                      ('probe_wta', HUE['other'], '--', 'uniformly drawn codes, best deep unit always learns'), ('then_teacher', HUE['mixed'], '-', 'own samples, then teacher strings (a deep unit must fire)'),
                                      ('wta_then_own', HUE['mixed'], '--', 'teacher strings (best unit always learns), then own samples')):
        band(ax, runs, ring + (name,), 'kl', label, color, style)
    ax.axhline(blank, color=HUE['reference'], ls=':', label='deep layer blank')
    ax.set(xscale='log', xlabel='samples', ylabel='KL(teacher || learner), nats', title='Own samples against teacher strings and a twin')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    ax = axes[1, 0]
    band(ax, runs, ring + ('own_accept',), 'teacher_pass', 'teacher strings that fire a deep unit', HUE['teacher'], index=1)
    band(ax, runs, ring + ('own_accept',), 'units_consistent', 'deep units whose template is a consistent string', HUE['own'])
    band(ax, runs, ring + ('own_accept',), 'consistent', 'own samples the first layer reads completely', HUE['own'], '--')
    band(ax, runs, ring + ('own_accept',), 'covered', "teacher's strings the learner writes", HUE['mixed'])
    band(ax, runs, ('tree', 0., 'own_accept'), 'valid', 'no shared tokens (tree): own samples that are teacher strings', HUE['reference'], '-.')
    ax.set(xscale='log', xlabel='own samples (no teacher strings)', ylabel='share', title='Inside Practice')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    ax = axes[1, 1]
    fs = (1., .5, .25)
    kl1 = np.median(end(runs, ring + ('own_accept',), 'kl'))
    width = .2
    for n, (name, color, label) in enumerate((('own_accept', HUE['own'], 'after Practice'), ('then_teacher', HUE['mixed'], 'Practice, then as many teacher strings'), ('teacher_wta', HUE['teacher'], 'teacher strings only, best unit always learns'))):
        values = [end(runs, ('ring', f, name), 'kl') for f in fs]
        ax.bar(np.arange(3) + (n - 1) * width, [np.median(v) for v in values], width, color=color, label=label)
        for m, v in enumerate(values):
            ax.plot(np.full(len(v), m + (n - 1) * width), v, '.', color='k', alpha=.4)
    ax.plot(np.arange(3) - width, [kl1 + np.log(1 / f) for f in fs], 'D', color=HUE['reference'], label='Practice at share 1, plus log(1 / share)')
    ax.set(xticks=np.arange(3), xticklabels=[f'{f:g}' for f in fs], xlabel="share of the consistent strings that the teacher writes", ylabel='KL(teacher || learner), nats', title='A deep rule that no token pair shows')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    ax = axes[1, 2]
    ds = (0, 1, 2, 3, 4, 6, 8)
    ax.plot(ds, [np.median(curve(runs, ring + (f'damage{d}_writer',), 'kl')[1][:, 0]) for d in ds], 'o:', color=HUE['reference'], label='at the start')
    for name, color, label in (('writer', HUE['own'], 'own samples, credited to the writer'), ('read', HUE['other'], 'own samples, credited to the unit that fires on reading'), ('teacher', HUE['teacher'], 'teacher strings, a deep unit must fire')):
        values = np.array([end(runs, ring + (f'damage{d}_{name}',), 'kl') for d in ds])
        ax.plot(ds, np.median(values, 1), 'o-', color=color, label=label)
        ax.fill_between(ds, np.quantile(values, .25, 1), np.quantile(values, .75, 1), color=color, alpha=.12, lw=0)
    ax.set(xlabel=f"slots (of {S}) replaced in each deep unit's consistent string at the start", ylabel=f"KL(teacher || learner) after {args['rounds'] // 2 * args['k']:,} samples, nats", title='How wrong may a deep unit be to be repaired by reading')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    totals = [int(args['rounds'] * 2 ** -n) for n in range(4, -1, -1)]
    shares = (0., .25, .5, .75, 1.)
    grids = {}
    for ax, theta, title in ((axes[2, 0], args['theta'], 'Reader with a threshold: Practice first, then teacher strings'), (axes[2, 1], 0., 'Reader whose best unit always learns: Practice first, then teacher strings')):
        grid = np.array([[np.median(end(runs, ring + (f'budget_theta{theta}_total{total}_own{share}',), 'kl')) for total in totals] for share in shares])
        grids[theta] = grid
        image = ax.imshow(grid, origin='lower', aspect='auto', cmap='viridis_r', vmin=0, vmax=blank)
        for i in range(len(shares)):
            for j in range(len(totals)):
                ax.text(j, i, f'{grid[i, j]:.2f}', ha='center', va='center', color='w' if grid[i, j] < blank / 2 else 'k')
        ax.set(xticks=range(len(totals)), xticklabels=[f"{t * args['k']:,}" for t in totals], yticks=range(len(shares)), yticklabels=[f'{s:g}' for s in shares], xlabel='samples after Watching (own + teacher)', ylabel='share of own samples', title=title)
        figure.colorbar(image, ax=ax, label='KL(teacher || learner), nats')

    ax = axes[2, 2]
    thetas = (0., .375, .5, .625, .75, .875, 1.)
    names = ['teacher_wta_cut' if t == 0 else 'teacher_gate_cut' if t == args['theta'] else f'teacher_theta{t}' for t in thetas]
    half = args['rounds'] // 2
    def at_half(name, field, index=None):
        key = ring + ({'teacher_wta_cut': 'teacher_wta', 'teacher_gate_cut': 'teacher_gate'}.get(name, name),)
        x, y = curve(runs, key, field, index)
        return y[:, np.argmin(np.abs(x - half * args['k']))]
    for field, color, label in (('covered', HUE['mixed'], "teacher's strings the learner writes"), ('valid', HUE['teacher'], 'own samples that are teacher strings'), ('fired', HUE['reference'], 'teacher strings that fire a deep unit')):
        values = np.array([at_half(name, field) for name in names])
        ax.plot(thetas, np.median(values, 1), 'o-', color=color, label=label)
        ax.fill_between(thetas, np.quantile(values, .25, 1), np.quantile(values, .75, 1), color=color, alpha=.12, lw=0)
    ax.set(xlabel='threshold of the deep units (share of slots)', ylabel=f"share after {half * args['k']:,} teacher strings", title='Teacher strings only: the threshold of the deep units')
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.14))

    figure.savefig(paths.out + '.png', dpi=150)
    figure.savefig(paths.out + '.svg')

    def stats(key):
        fields = ('kl', 'valid', 'consistent', 'covered', 'units_consistent', 'strings_held', 'sharp')
        out = {field: float(np.median(end(runs, key, field))) for field in fields}
        kl = end(runs, key, 'kl')
        out.update(kl_quartiles=[float(np.quantile(kl, .25)), float(np.quantile(kl, .75))], kl_start=float(np.median(curve(runs, key, 'kl')[1][:, 0])), samples=int(curve(runs, key, 'kl')[0][-1]),
                   teacher_fire=float(np.median(end(runs, key, 'teacher_pass', 1))), half_way=med(half_way(runs, key)), worlds_half_way=int(sum(h is not None for h in half_way(runs, key))), worlds=len(runs[key]))
        x, y = curve(runs, key, 'kl')
        out['kl_at'] = {int(s): float(np.median(y[:, np.argmin(np.abs(x - s))])) for s in (10240, 20480, 51200, 102400, 204800) if s <= x[-1]}
        return out
    numbers = dict(args=args, constants=const, arms={f'{kind}/{f:g}/{arm}': stats((kind, f, arm)) for (kind, f, arm) in sorted(runs) if not arm.startswith('budget')},
                   budget={str(theta): dict(totals=[t * args['k'] for t in totals], shares=list(shares), kl=grid.tolist()) for theta, grid in grids.items()},
                   references={f'{kind}/{f:g}': dict(n_cons=float(np.median([r['n_cons'] for r in rows])), n_valid=float(np.median([r['n_valid'] for r in rows])), blank_kl=float(np.median([r['blank']['kl'] for r in rows])),
                                                    blank_consistent=float(np.median([r['blank']['consistent'] for r in rows])), strings_until_all_units=[float(np.median([r['strings_until_all_units'] for r in rows])), int(max(r['strings_until_all_units'] for r in rows))],
                                                    own_unread=float(np.median([r['own_unread'] for r in rows])), own_changed=float(np.median([r['own_changed'] for r in rows])),
                                                    match={name: (np.sum([r[name] for r in rows], 0) / np.sum([r[name] for r in rows])).round(5).tolist() for name in ('teacher_match_hist', 'teacher_best_hist', 'own_match_hist')}) for (kind, f), rows in refs.items()},
                   twin={arm: float(np.median([r['twin']['kl'] for r in runs[ring + (arm,)]])) for arm in ('twin_gate', 'twin_wta')},
                   lock_round=dict(median=float(np.median([np.median([d for d in r['done_round'] if d >= 0]) for r in runs[ring + ('own_accept',)]])), never=float(np.mean([np.mean(np.array(r['done_round']) < 0) for r in runs[ring + ('own_accept',)]]))))
    Path(paths.out + '.json').write_text(json.dumps(numbers, indent=1))
    print(paths.out + '.png')


if __name__ == '__main__':
    main()
