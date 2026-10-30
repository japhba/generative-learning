"""One-step control, tight-bound audit, and shortcut-screen comparisons."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

import practice_recursive_grammar_screen as model


def one_step(data, truth):
    mean = data.mean()
    root = np.array([1-mean, mean])
    rules = np.broadcast_to(root[:, None]*root[None], (2, 2, 2)).copy()
    p0, r0, c0, entropy = model.posterior_counts(root, rules, data)
    new_root = r0.mean(axis=0)
    counts = c0.sum(axis=0)
    new_rules = counts/counts.sum(axis=(1, 2), keepdims=True)
    after = model.inside(new_root, new_rules, data)[0]
    bound_gain = np.mean(r0 @ np.log(new_root/root)+np.einsum('nabc,abc->n', c0, np.log(new_rules/rules)))
    likelihood_gain = np.log(after).mean()-np.log(p0).mean()
    assert bound_gain > -1e-12
    assert likelihood_gain-bound_gain > -1e-12
    target = model.inside(new_root, new_rules, model.WORDS)[0]
    initial_bound = np.mean(r0 @ np.log(root)+np.einsum('nabc,abc->n', c0, np.log(rules))-np.log(model.CATALAN[model.LENGTH])+entropy)
    assert abs(initial_bound-np.log(p0).mean()) < 1e-12
    # The reference posterior stays fixed: another M step returns the same
    # new weights exactly. Changes beyond this require new latent inference.
    fixed_again = c0.sum(axis=0)/c0.sum(axis=(0, 2, 3))[:, None, None]
    assert np.max(np.abs(fixed_again-new_rules)) < 1e-14
    return dict(scores=[float(-np.log(after).mean()), float(truth @ (np.log(truth)-np.log(target)))],
                bound_gain=float(bound_gain), likelihood_gain=float(likelihood_gain),
                initial_bound_gap=float(-np.log(p0).mean()+initial_bound),
                equal_parent_rows_error=float(np.max(np.abs(new_rules[0]-new_rules[1]))),
                root=new_root.tolist(), rules=new_rules.tolist())


def main():
    load_dotenv(Path.home()/'.env')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_recursive_grammar/screen')
    args = parser.parse_args()
    output = args.source.parent/(args.source.name+'_analysis')
    output.mkdir(parents=True, exist_ok=False)
    rows = json.loads((args.source/'summary.json').read_text())
    count = len(rows)
    bootstrap = np.random.default_rng(20102719).integers(count, size=(20000, count))
    def interval(v): return dict(mean=float(v.mean()), bootstrap95=np.quantile(v[bootstrap].mean(axis=-1), [.025, .975]).tolist())
    grammar = np.array([r['grammar'][:2] for r in rows])
    one = []
    for i in range(count):
        world = np.load(args.source/f'world_{i}.npz')
        one.append(one_step(world['data'], world['truth']))
    direct = {name: np.array([r['direct_scores'][name] for r in rows]) for name in rows[0]['direct_scores']}
    direct['one_step_grammar'] = np.array([r['scores'] for r in one])
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), source=str(args.source),
                  interpretation='Exact-inference capacity gate, not evidence that generated practice or learned recognition is necessary.',
                  grammar_mean=grammar.mean(axis=0).tolist(), direct_means={k: v.mean(axis=0).tolist() for k, v in direct.items()},
                  grammar_minus_direct={k: interval(grammar[:, 1]-v[:, 1]) for k, v in direct.items()}, one_step=one,
                  cv_choices=[r['cv_choice'] for r in rows],
                  stopping=[r['starts'][r['chosen_start']] for r in rows])
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    fig, ax = plt.subplots(layout='constrained')
    selected = [np.load(args.source/f'em_{i}_{rows[i]["chosen_start"]}.npz') for i in range(count)]
    grid = np.arange(0, max(r['steps'][-1] for r in selected)+1, model.CONFIG['evaluation_every'])
    ax.plot(grid, np.mean([np.interp(grid, r['steps'], r['records'][:, 1]) for r in selected], axis=0), label='Iterative grammar inference and fitting')
    for index, (name, label) in enumerate([('one_step_grammar', 'One analytic grammar update'), ('cv_selected', 'CV-selected direct fit'), ('position_means', 'Direct position means'), ('order_3', 'Direct order-three counts')]):
        ax.axhline(direct[name][:, 1].mean(), linestyle=':', color=f'C{index+1}', label=label)
    ax.set(ylabel='True predictive KL (nats)', xlabel='EM iterations; stopped models held fixed')
    fig.legend(*ax.get_legend_handles_labels(), loc='outside upper center', ncol=1)
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-recursive-grammar-shortcuts.png')
    plt.close(fig)
    print(json.dumps({k: result[k] for k in ['source', 'grammar_mean', 'direct_means', 'grammar_minus_direct']}, indent=2))


if __name__ == '__main__': main()
