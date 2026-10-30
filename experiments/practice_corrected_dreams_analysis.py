"""Post-screen diagnostics, including fixed-generator recognition comparisons."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_corrected_dreams as model


def main():
    load_dotenv(Path.home()/'.env')
    base = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_corrected_dreams'
    source, output = base/'pilot', base/'analysis'
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((source/'summary.json').read_text())
    count = len(summary)
    rng = np.random.default_rng(20102675)
    bootstrap = rng.integers(count, size=(10000, count))
    def interval(values):
        return dict(mean=float(values.mean()), bootstrap95=np.quantile(values[bootstrap].mean(axis=-1), [.025, .975]).tolist())
    conditions = ['joint', 'frozen', 'collapsed', 'original', 'self', 'oracle']
    runs = {c: [np.load(source/f'{c}_{i}.npz') for i in range(count)] for c in conditions}
    report = dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), scope='Post-screen analysis, not an independent confirmation.',
                  endpoint_means={c: np.mean([r['records'][-1] for r in runs[c]], axis=0).tolist() for c in conditions})
    report['joint_minus_control'] = {c: interval(np.array([runs['joint'][i]['records'][-1, 0]-runs[c][i]['records'][-1, 0] for i in range(count)])) for c in conditions[1:]}
    best_original = np.array([min(summary[i]['initial_best_predictive'], runs['original'][i]['records'][:, 0].min()) for i in range(count)])
    joint = np.array([r['records'][-1, 0] for r in runs['joint']])
    report['best_original_mean'] = float(best_original.mean())
    report['joint_minus_best_original'] = interval(joint-best_original)
    report['analytic_baseline_means'] = {k: float(np.mean([s['analytic_baselines'][k] for s in summary])) for k in summary[0]['analytic_baselines']}
    for control in ['empirical_spectral', 'predictive_spectral']:
        report['joint_minus_'+control] = interval(joint-np.array([s['analytic_baselines'][control] for s in summary]))
    common_budget = min(r['costs'][-1] for c in ['joint', 'frozen', 'collapsed'] for r in runs[c])
    matched = {c: np.array([np.interp(common_budget, r['costs'], r['records'][:, 0]) for r in runs[c]]) for c in ['joint', 'frozen', 'collapsed']}
    report['common_budget'] = dict(proposals=int(common_budget), means={c: float(v.mean()) for c, v in matched.items()},
                                    joint_minus_frozen=interval(matched['joint']-matched['frozen']),
                                    note='Proposal count includes own samples for recognition; it is not measured runtime or a flop count.')
    fixed_generator = []
    for i in tqdm(range(count), desc='Fixed-generator recognition audit'):
        world, initial = np.load(source/f'world_{i}.npz'), np.load(source/f'initial_{i}.npz')
        w = runs['joint'][i]['w'][-1]
        for name in ['learned', 'frozen', 'collapsed']:
            a = initial['a'][-1] if name == 'frozen' else runs['joint'][i]['a'][-1]
            table = model.tables(w, a)
            if name == 'collapsed': table['recognition'] = table['proposal']/table['proposal'].sum(axis=-1, keepdims=True)
            estimate, info = model.coupled_gradient(np.random.default_rng(20102676+i), w, table, world['target'], world['data'], 65536)
            exact = np.einsum('s,skd->kd', world['target'], table['score'])
            zscore = (estimate-exact)/np.sqrt(info['component_variance']/65536)
            assert np.abs(zscore).max() < 5, (i, name, zscore)
            fixed_generator.append(dict(world=i, recognizer=name, mean_meeting=info['mean_meeting'], max_meeting=info['max_meeting'],
                                        acceptance=info['acceptance'], gradient_variance=info['gradient_variance'],
                                        max_standardized_error=float(np.abs(zscore).max())))
    report['fixed_generator'] = fixed_generator
    report['fixed_generator_means'] = {name: {metric: float(np.mean([r[metric] for r in fixed_generator if r['recognizer'] == name])) for metric in ['mean_meeting', 'acceptance', 'gradient_variance']} for name in ['learned', 'frozen', 'collapsed']}
    report['initial_mean'] = np.mean([s['initial'] for s in summary], axis=0).tolist()
    report['mean_posterior_max'] = float(np.mean([s['posterior_max'] for s in summary]))
    report['uk_finish'] = datetime.now(ZoneInfo('Europe/London')).isoformat()
    (output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    initial = np.load(source/'initial_0.npz')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    for ax, w, title in zip(axes, [initial['w'][-1], runs['joint'][0]['w'][-1]], ['After original-data fitting', 'After corrected dreams']):
        color = ax.imshow(model.sigmoid(w), vmin=0, vmax=1, aspect='auto')
        ax.set(title=title, ylabel='Latent cause', xlabel='Visible bit', yticks=np.arange(model.K), xticks=np.arange(model.D))
    fig.colorbar(color, ax=axes, label='Conditional probability of bit = 1')
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-corrected-dreams-weights.png')
    plt.close(fig)
    print(json.dumps({key: report[key] for key in ['joint_minus_best_original', 'joint_minus_predictive_spectral', 'common_budget', 'fixed_generator_means']}, indent=2))


if __name__ == '__main__': main()
