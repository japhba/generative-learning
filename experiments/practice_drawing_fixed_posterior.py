"""Freeze the whole posterior, rather than just its learned proposal network."""

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
from practice_recursive_grammar_screen import constrained_counts


def counts(cat, a, b, observed):
    code_p, image_p = cat.probabilities(a, b)
    unique, mass = np.unique(observed, return_counts=True)
    component = np.prod(b[:, np.arange(3)[None, :], m.MICRO], axis=-1)
    acount, bcount = np.zeros_like(a), np.zeros_like(b)
    codes = np.concatenate([cat.aliases(i) for i in unique])
    weights = np.concatenate([n*code_p[cat.aliases(i)]/image_p[i] for i, n in zip(unique, mass)])/len(observed)
    for chunk in range(3):
        frequency = np.bincount(cat.chunks[codes, chunk], weights=weights, minlength=64)
        posterior = a[chunk, :, None]*component
        posterior /= posterior.sum(axis=0, keepdims=True)
        joint = posterior*frequency
        acount[chunk] = joint.sum(axis=-1)
        for position in range(3):
            for action in range(4): bcount[:, position, action] += joint[:, m.MICRO[:, position] == action].sum(axis=-1)
    assert np.max(abs(acount.sum(axis=-1)-1)) < 1e-12
    assert abs(bcount.sum()-9) < 1e-12
    return acount, bcount, image_p


def mstep(acount, bcount): return constrained_counts(acount, .02/3), constrained_counts(bcount, .02/4)


def complete_q(a, b, acount, bcount): return np.sum(acount*np.log(a))+np.sum(bcount*np.log(b))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, args.output/'source.py')
    (args.output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(),
        source=str(args.source), retrospective=True,
        issue='The previous frozen conditions freeze proposal weights, but code posterior weights, cache membership, and motif responsibilities keep changing.',
        control='Freeze the exact initial joint posterior over programs and motifs. Compute its globally optimal categorical M step under the existing floors, without line search. Repeating this fixed-posterior M step changes nothing.',
        reference='Repeat exact E and exact constrained M steps, with no likelihood acceptance check or learning-rate parameter. Stop on stationarity of observed NLL, maximum 200 iterations.',
        hypothesis='The trained-once proposal results do not establish that a fixed posterior is sufficient.'), indent=2)+'\n')
    cat = m.Catalogue()
    source_rows = json.loads((args.source/'summary.json').read_text())
    result, histories = [], []
    for i in tqdm(range(len(source_rows)), desc='Frozen-posterior controls'):
        world = np.load(args.source/f'world_{i}.npz')
        a, b = world['initial_a'].copy(), world['initial_b'].copy()
        acount, bcount, initial_p = counts(cat, a, b, world['observed'])
        fixed_a, fixed_b = mstep(acount, bcount)
        fixed_q_gain = complete_q(fixed_a, fixed_b, acount, bcount)-complete_q(a, b, acount, bcount)
        _, fixed_p = cat.probabilities(fixed_a, fixed_b)
        fixed_l_gain = np.log(fixed_p[world['observed']]/initial_p[world['observed']]).mean()
        assert fixed_q_gain >= -1e-12 and fixed_l_gain >= fixed_q_gain-1e-10
        assert all(np.array_equal(x, y) for x, y in zip(mstep(acount, bcount), (fixed_a, fixed_b)))
        # The counts fully specify the frozen distribution's generator objective.
        np.savez_compressed(args.output/f'fixed_{i}.npz', a=fixed_a, b=fixed_b, acount=acount, bcount=bcount)
        history, params = [], []
        status = 'iteration limit'
        for step in range(201):
            current_a, current_b, probability = counts(cat, a, b, world['observed'])
            nll = -np.log(probability[world['observed']]).mean()
            kl = np.sum(world['teacher_p']*np.log(world['teacher_p']/probability))
            history.append([step, nll, kl]); params.append(np.r_[a.ravel(), b.ravel()])
            if step > 0: assert history[-1][1] <= history[-2][1]+1e-10
            if step >= 40:
                recent = np.array(history[-20:])
                if abs(np.polyfit(np.log(recent[:, 0]), np.log(recent[:, 1]), 1)[0]) < .003:
                    status = 'stationary observed likelihood'; break
            a, b = mstep(current_a, current_b)
        histories.append(np.array(history))
        np.savez_compressed(args.output/f'adaptive_{i}.npz', records=np.array(history), weights=np.array(params), status=status)
        result.append(dict(seed=int(world['seed']), initial_kl=float(np.sum(world['teacher_p']*np.log(world['teacher_p']/initial_p))),
            fixed_posterior_kl=float(np.sum(world['teacher_p']*np.log(world['teacher_p']/fixed_p))),
            fixed_posterior_q_gain=float(fixed_q_gain), fixed_posterior_ll_gain=float(fixed_l_gain),
            adaptive_kl=float(history[-1][2]), adaptive_steps=int(history[-1][0]),
            sleep_kl=float(source_rows[i]['conditions']['sleep']['final'][2]),
            calibrated_proposal_kl=float(source_rows[i]['conditions']['calibrated']['final'][2])))
    bootstrap = np.random.default_rng(20103050).integers(len(result), size=(20000, len(result)))
    def estimate(values):
        values = np.array(values)
        return dict(mean=float(values.mean()), bootstrap95=np.quantile(values[bootstrap].mean(axis=-1), [.025,.975]).tolist())
    summary = dict(rows=result, means={key: estimate([r[key] for r in result]) for key in ['initial_kl', 'fixed_posterior_kl', 'adaptive_kl', 'sleep_kl', 'calibrated_proposal_kl']},
        sleep_minus_fixed=estimate([r['sleep_kl']-r['fixed_posterior_kl'] for r in result]),
        adaptive_minus_fixed=estimate([r['adaptive_kl']-r['fixed_posterior_kl'] for r in result]))
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    fig, axes = plt.subplots(2, 1, layout='constrained')
    grid = np.arange(max(h[-1,0] for h in histories)+1)
    for ax, column, label in zip(axes, [2,1], ['Population KL (nats)', 'Observed-data NLL (nats)']):
        for i, history in enumerate(histories):
            ax.plot(grid, np.interp(grid, history[:,0], history[:,column]), alpha=.5, label='Adaptive posterior: individual worlds' if i==0 else None)
        ax.plot(grid, np.mean([np.interp(grid,h[:,0],h[:,column]) for h in histories],axis=0), color='black', label='Adaptive posterior: mean')
        frozen=np.mean([[h[0,column],h[1,column]] for h in histories],axis=0)
        ax.plot(grid,np.where(grid==0,frozen[0],frozen[1]),linestyle='--',label='Fully frozen posterior: mean')
        ax.set_ylabel(label)
    axes[0].set_yscale('log')
    axes[-1].set_xlabel('Exact EM updates; stopped models held fixed')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=1)
    fig.savefig(m.ROOT/'tex/archive/figures/practice-drawing-fixed-posterior.png')
    plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2))


if __name__ == '__main__': main()
