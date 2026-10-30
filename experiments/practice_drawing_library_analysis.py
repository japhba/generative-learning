"""Population-risk curves and cross-validated direct fits for drawing programs."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_drawing_library as m


def direct(cat, observed, canonical_map):
    n = len(observed)
    counts = np.bincount(observed, minlength=len(cat.images))
    result = {f'histogram_{alpha}': (counts+alpha*cat.uniform)/(n+alpha) for alpha in [.1, 1., 10., 100., 1000.]}
    # The original causes are fixed per image before fitting any condition.
    actions = cat.actions[canonical_map[observed]]
    frequencies = np.stack([np.bincount(col, minlength=4)+.5 for col in actions.T])
    probability = frequencies/frequencies.sum(axis=-1, keepdims=True)
    code = np.prod(probability[np.arange(9), cat.actions], axis=-1)
    result['original_code_counts'] = np.bincount(cat.ids, weights=code, minlength=len(cat.images))
    frequencies = frequencies.reshape(3, 3, 4).sum(axis=0)
    probability = frequencies/frequencies.sum(axis=-1, keepdims=True)
    code = np.prod(probability[np.arange(9) % 3, cat.actions], axis=-1)
    result['shared_original_code_counts'] = np.bincount(cat.ids, weights=code, minlength=len(cat.images))
    pixels = ((cat.images[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(float)
    data = pixels[observed]
    marginal = (data.sum(axis=0)+.5)/(n+1)
    log_pixel = pixels*np.log(marginal)+(1-pixels)*np.log1p(-marginal)
    # The center is on for every possible drawing, so do not charge it a probability.
    log_pixel[:, 27] = 0
    result['pixel_counts'] = np.exp(log_pixel.sum(axis=-1))
    joint = np.empty((64, 64, 2, 2))
    joint[:, :, 1, 1] = data.T @ data
    joint[:, :, 1, 0] = data.sum(axis=0)[:, None]-joint[:, :, 1, 1]
    joint[:, :, 0, 1] = data.sum(axis=0)[None, :]-joint[:, :, 1, 1]
    joint[:, :, 0, 0] = n-joint[:, :, 1, 1]-joint[:, :, 1, 0]-joint[:, :, 0, 1]
    joint = (joint+.25)/(n+1)
    independent = joint.sum(axis=-1)[..., :, None]*joint.sum(axis=-2)[..., None, :]
    mutual = (joint*np.log(joint/independent)).sum(axis=(-2, -1))
    visited, edges = np.zeros(64, dtype=bool), []
    visited[27] = True
    for _ in range(63):
        weight = np.where(visited[:, None] & ~visited[None, :], mutual, -np.inf)
        parent, child = np.unravel_index(np.argmax(weight), weight.shape)
        edges.append((parent, child)); visited[child] = True
    logp = np.zeros(len(pixels))
    for parent, child in edges:
        conditional = joint[parent, child]/joint[parent, child].sum(axis=-1, keepdims=True)
        logp += np.log(conditional[pixels[:, parent].astype(int), pixels[:, child].astype(int)])
    result['pixel_tree'] = np.exp(logp)
    return result


def controls(cat, world):
    canonical_map = np.zeros(len(cat.images), dtype=int)
    canonical_map[np.unique(world['observed'])] = world['canonical']
    fitted = direct(cat, world['observed'], canonical_map)
    scores = {name: 0. for name in fitted}
    rng = np.random.default_rng(int(world['seed'])+19000)
    for test in np.array_split(rng.permutation(len(world['observed'])), 4):
        train = np.setdiff1d(np.arange(len(world['observed'])), test)
        for name, p in direct(cat, world['observed'][train], canonical_map).items():
            scores[name] -= np.log(p[world['observed'][test]]).sum()
    selected = min(scores, key=scores.__getitem__)
    fitted['cv_selected'] = fitted[selected]
    risk = {name: float(np.sum(world['teacher_p']*np.log(world['teacher_p']/p))) for name, p in fitted.items()}
    return dict(kl=risk, selected=selected, cv_nll=scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plot-prefix', default='practice-drawing-library')
    parser.add_argument('--calibrated', type=Path)
    parser.add_argument('--write-tex', action='store_true')
    parser.add_argument('--fixed-posterior', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cat = m.Catalogue()
    rows = json.loads((args.source/'summary.json').read_text())
    conditions, worlds = list(rows[0]['conditions']), len(rows)
    runs = {c: [np.load(args.source/f'{c}_{i}.npz') for i in range(worlds)] for c in conditions}
    if args.calibrated is not None:
        conditions.append('calibrated')
        runs['calibrated'] = [np.load(args.calibrated/f'calibrated_{i}.npz') for i in range(worlds)]
        for i in range(worlds):
            first, second = np.load(args.source/f'world_{i}.npz'), np.load(args.calibrated/f'world_{i}.npz')
            assert all(np.array_equal(first[key], second[key]) for key in first.files)
    control = [controls(cat, np.load(args.source/f'world_{i}.npz')) for i in tqdm(range(worlds), desc='Direct controls')]
    boot = np.random.default_rng(20103003).integers(worlds, size=(20000, worlds))
    def estimate(values):
        values = np.array(values)
        return dict(mean=float(values.mean()), bootstrap95=np.quantile(values[boot].mean(axis=-1), [.025, .975]).tolist(), per_world=values.tolist())
    final = {c: np.array([r['records'][-1, 2] for r in runs[c]]) for c in conditions}
    direct_kl = np.array([c['kl']['cv_selected'] for c in control])
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), source=str(args.source), controls=control,
        initial=estimate([r['records'][0, 2] for r in runs['sleep']]), final={c: estimate(v) for c, v in final.items()},
        direct=estimate(direct_kl), endpoint_sleep_minus={c: estimate(final['sleep']-v) for c, v in final.items() if c != 'sleep'},
        sleep_minus_direct=estimate(final['sleep']-direct_kl), fixed_budget={})
    for budget in [200, 400, 600]:
        values = {c: np.array([np.interp(budget, r['records'][:, 0], r['records'][:, 2]) for r in runs[c]]) for c in conditions}
        values['direct'] = direct_kl
        result['fixed_budget'][budget] = dict(risk={c: estimate(v) for c, v in values.items()},
            sleep_minus={c: estimate(values['sleep']-v) for c, v in values.items() if c != 'sleep'})
    violations = {c: float(max(np.max(np.diff(r['records'][:, 1])) for r in runs[c])) for c in conditions}
    assert max(violations.values()) < 1e-10
    result['largest_nll_increases'] = violations
    result['stopping'] = {c: [dict(step=int(r['records'][-1, 0]), status=str(r['status'])) for r in runs[c]] for c in conditions}
    if args.write_tex:
        primary = result['fixed_budget'][400]
        names = dict(sleep='Own-generated practice', wake='Proposal trained on old examples',
                     calibrated='Proposal trained once, then frozen', frozen='Untrained frozen proposal',
                     canonical='Original explanations only', exact='Exhaustive posterior reference', direct='Cross-validated direct fits')
        lines = [r'\begin{center}', r'\begin{tabular}{lrr}', r'\toprule',
                 r'Method & KL at 400 & Final KL \\', r'\midrule',
                 f"Initial generator & {result['initial']['mean']:.4f} & {result['initial']['mean']:.4f}" + r' \\']
        for key, label in names.items():
            endpoint = result['direct']['mean'] if key == 'direct' else result['final'][key]['mean']
            lines.append(f"{label} & {primary['risk'][key]['mean']:.4f} & {endpoint:.4f}" + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{center}',
                  f"Results are means across {worlds} fresh worlds. Smaller population KL is better.",
                  'The predeclared paired differences at 400 updates are:']
        lines += [r'\begin{itemize}']
        for key in ['wake', 'calibrated', 'direct']:
            contrast = primary['sleep_minus'][key]
            lo, hi = contrast['bootstrap95']
            label = dict(wake='wake', calibrated='calibrated proposal', direct='direct fits')[key]
            lines.append(f"\\item Sleep minus {label}: $" + f"{contrast['mean']:+.4f}\\; [{lo:+.4f}, {hi:+.4f}]$ nats.")
        lines += [r'\end{itemize}',
                  r'Intervals are paired 95\% bootstrap intervals across worlds. The advantage over the direct fits survives the fresh comparison. Neither a sleep-over-wake advantage nor a sleep-over-calibrated advantage is established.',
                  'The pilot sleep-minus-wake endpoint mean was -0.2613 nats, but one world contributed -1.0440 nats of that contrast; the other three differences were each below 0.001 nat in magnitude. This large pilot advantage did not repeat across the fresh panel.',
                  'The trained-once proposal has lower mean final risk than continual sleep training, although the paired interval includes zero. This comparison freezes only proposal weights: cache membership, program posterior weights, and motif responsibilities continue to adapt. It therefore does not establish that the entire recognition process can remain fixed.',
                  'The exact-posterior reference is also imperfect: one fresh world has final KL about 1.074. All adaptive methods remain near that poor solution in that world. The experiment therefore does not establish reliable global optimization.',
                  r'\textbf{Assessment.} This is positive evidence for learning a generative motif prior by searching among equally perfect reconstructions, beyond the tested direct count and pixel fits. The separate proposal network need not continue learning in this experiment, but effective recognition still adapts. The remaining qualifications are the global enumeration check and the untested recognition-ahead-of-generation intuition.']
        if args.fixed_posterior is not None:
            fixed = json.loads((args.fixed_posterior/'summary.json').read_text())
            result['fully_fixed_posterior'] = fixed
            lines += ['', r'\subsection{Correction: freezing the whole posterior}',
                      r'The original interpretation of the calibrated control was too broad. It froze the neural proposal, not the effective posterior. A retrospective follow-up on the same eight worlds freezes the exact initial joint posterior over both programs and motifs.',
                      f"Its globally optimal categorical M-step gives mean population KL {fixed['means']['fixed_posterior_kl']['mean']:.4f}. Repeating that fixed-posterior M-step cannot change the weights. Repeated exact E and exact M steps, without a likelihood acceptance check, instead reach {fixed['means']['adaptive_kl']['mean']:.4f}. These are different from the earlier smoothed, damped exact-posterior reference.",
                      'This result supports posterior adaptation relative to the specified initial-posterior control. It does not prove that every conceivable fixed recognition rule fails, nor that continued training of the separate proposal network is necessary.',
                      r'\begin{figure}[ht]\centering',
                      r'\includegraphics[width=\linewidth]{practice-drawing-fixed-posterior.png}',
                      r'\caption{The fixed posterior reaches its best generator in one M-step. The adaptive posterior continues changing the learned motif prior.}\end{figure}']
        (m.ROOT/'tex/archive/practice_drawing_library_results.tex').write_text('\n'.join(lines)+'\n')
    if 'calibrated' in runs:
        fig, ax = plt.subplots(layout='constrained')
        for i, run in enumerate(runs['calibrated']): ax.plot(run['calibration'][:, 0], run['calibration'][:, 1], label=f'World {i+1}')
        ax.set(xlabel='Recognition updates; initial generator held fixed', ylabel='Held-out dream program cross entropy')
        ax.legend()
        fig.savefig(m.ROOT/f'tex/archive/figures/{args.plot_prefix}-calibration.png')
        plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, args.output/'source.py')
    for columns, suffix, labels in [([2, 1], 'learning', ['True image KL (nats)', 'Observed-image NLL (nats)']),
                                   ([4, 5], 'recognition', ['Valid held-out programs', 'Valid causes per image'])]:
        fig, axes = plt.subplots(2, 1, layout='constrained')
        grid = np.arange(0, max(r['records'][-1, 0] for series in runs.values() for r in series)+1, m.CONFIG['eval_every'])
        for c in conditions:
            for ax, column, label in zip(axes, columns, labels):
                ax.plot(grid, np.mean([np.interp(grid, r['records'][:, 0], r['records'][:, column]) for r in runs[c]], axis=0), label=c)
                ax.set_ylabel(label)
        if suffix == 'learning':
            axes[0].axhline(direct_kl.mean(), color='black', linestyle=':', label='CV direct counts / pixel tree')
            axes[0].set_yscale('log')
        axes[-1].set_xlabel('Updates; stopped models held fixed')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside upper center', ncol=3)
        fig.savefig(m.ROOT/f'tex/archive/figures/{args.plot_prefix}-{suffix}.png')
        plt.close(fig)
    world = np.load(args.source/'world_0.npz')
    fig, axes = plt.subplots(3, 1, layout='constrained')
    for ax, b, title in zip(axes, [world['initial_b'], runs['sleep'][0]['b'], world['teacher_b']],
                            ['Initial motifs', 'After own-generation practice', 'Teacher motifs (evaluation only)']):
        picture = ax.imshow(b.reshape(3, 12), vmin=0, vmax=1, aspect='auto')
        ax.set(title=title, xticks=range(12), xticklabels=['F', 'R', 'L', 'J']*3, yticks=range(3), ylabel='Latent motif')
    axes[-1].set_xlabel('Commands within each of three motif positions')
    fig.colorbar(picture, ax=axes, label='Command probability')
    fig.savefig(m.ROOT/f'tex/archive/figures/{args.plot_prefix}-motifs.png')
    plt.close(fig)
    fig, axes = plt.subplots(3, 6, layout='constrained')
    for row, a, b, label in zip(axes, [world['initial_a'], runs['sleep'][0]['a'], world['teacher_a']],
                                [world['initial_b'], runs['sleep'][0]['b'], world['teacher_b']],
                                ['Initial', 'Practised', 'Teacher']):
        examples = m.draw(a, b, np.random.default_rng(20103004), len(row))
        for ax, code in zip(row, examples):
            bitmap = ((cat.canvas[code] >> np.arange(64, dtype=np.uint64)) & 1).reshape(8, 8)
            ax.imshow(bitmap, cmap='gray_r', vmin=0, vmax=1)
            ax.set(xticks=[], yticks=[])
        row[0].set_ylabel(label)
    fig.suptitle('Sampled drawings from the first world')
    fig.savefig(m.ROOT/f'tex/archive/figures/{args.plot_prefix}-drawings.png')
    plt.close(fig)
    print(json.dumps({name: result[name] for name in ['initial', 'final', 'direct', 'endpoint_sleep_minus', 'sleep_minus_direct']}, indent=2))


if __name__ == '__main__': main()
