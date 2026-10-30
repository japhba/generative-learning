"""Compare inference variance at identical generator weights and observations."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

import practice_grammar_parser as model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--world', type=int, default=0)
    parser.add_argument('--draws', type=int, default=8192)
    args = parser.parse_args()
    output = args.source.parent/f'{args.source.name}_variance_world_{args.world}'
    output.mkdir(parents=True, exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(dict(uk_start=datetime.now(ZoneInfo('Europe/London')).isoformat(), source=str(args.source), world=args.world,
        draws=args.draws, design='Fix the final sleep generator. Compare uniform recognition, recognition calibrated for 500 updates on the INITIAL generator, and final learned sleep recognition. Use identical observed examples and initial latent caches.',
        limitation='An inference-efficiency assay, not a generation-outcome comparison with a calibrated frozen recognizer.'), indent=2)+'\n')
    world = np.load(args.source/f'world_{args.world}.npz')
    run = np.load(args.source/f'sleep_{args.world}.npz')
    w = run['w'][-1]
    rng = np.random.default_rng(20102904)
    calibrated = world['initial_a'].copy()
    records = []
    for step in tqdm(range(501), desc='Calibrate recognition on the initial generator'):
        tables = model.recognition_tables(calibrated)
        if step % 25 == 0: records.append(model.evaluate(world['initial'], tables, world))
        if step == 500: break
        ids, tree = model.sample_generator(rng, world['initial'], model.CONFIG['batch'])
        calibrated += model.CONFIG['rate']/(1+step/500)**.6*model.recognition_score(tables, ids, tree).mean(axis=0)
    np.savez_compressed(output/'calibration.npz', a=calibrated, records=np.array(records), steps=np.arange(0, 501, 25))
    observed = world['data'] @ model.grammar.POWERS
    ids = observed[rng.integers(len(observed), size=args.draws)]
    cache = model.sample_recognition(rng, model.UNIFORM_TABLES, observed)
    lookup = {int(word): index for index, word in enumerate(observed)}
    initial = model.subset(cache, np.array([lookup[int(word)] for word in ids]))
    root, rules = model.generator_probabilities(w)
    _, rc, cc, _ = model.grammar.posterior_counts(root, rules, world['data'])
    target = model.generator_scores_from_counts(w, rc, cc.reshape(len(rc), 2, 4)).mean(axis=0)
    result = {}
    for name, a in tqdm([('uniform', world['initial_a']), ('calibrated_initial', calibrated), ('learned_final', run['a'][-1])], desc='Identical-generator inference assay'):
        rng = np.random.default_rng(20102905)
        tables = model.recognition_tables(a)
        estimates, diagnostics = [], []
        for start in tqdm(range(0, args.draws, 256), desc=name, leave=False):
            rows = np.arange(start, min(start+256, args.draws))
            dg, _, _, info = model.coupled_gradient(rng, w, tables, ids[rows], model.subset(initial, rows), False)
            estimates.append(dg)
            diagnostics.append(info)
        estimate = np.concatenate(estimates)
        standard_error = estimate.std(axis=0, ddof=1)/np.sqrt(len(estimate))
        zscore = (estimate.mean(axis=0)-target)/standard_error
        metric = model.evaluate(w, tables, world)
        variance = estimate.var(axis=0, ddof=1).sum()
        proposals = sum(d['proposals'] for d in diagnostics)/args.draws
        result[name] = dict(trace_gradient_variance=float(variance), proposals_per_draw=proposals,
            variance_times_proposals=float(variance*proposals), posterior_recognition_kl=float(metric[2]),
            max_standardized_gradient_error=float(abs(zscore).max()), maximum_meeting=max(d['maximum_meeting'] for d in diagnostics))
        np.savez_compressed(output/f'{name}.npz', estimates=estimate, target=target, a=a, w=w, word_ids=ids)
        (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    fig, ax = plt.subplots(layout='constrained')
    ax.plot(np.arange(0, 501, 25), np.array(records)[:, 2])
    ax.set(xlabel='Recognition updates with initial generator fixed', ylabel='Causal posterior recognition KL')
    fig.savefig(Path(__file__).resolve().parents[1]/'tex/archive/figures/practice-grammar-parser-calibration.png')
    plt.close(fig)
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
