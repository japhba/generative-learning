"""An analytic structural control: infer a parent-independent grammar by moments.

Conditional on the tree shape, leaves from different production events are
independent when both parent categories use the same production row. Leaf
means identify left/right marginals; adjacent leaf pairs identify the common
child-pair probability through the known probability of being siblings.
"""

import argparse
from datetime import datetime
from functools import cache
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

import practice_recursive_grammar_screen as model


@cache
def shapes(length):
    if length == 1: return [None]
    return [(left, right) for split in range(1, length) for left in shapes(split) for right in shapes(length-split)]


def shape_features(tree):
    orientation, cherries = [], []
    def visit(node, side):
        if node is None: orientation.append(side); return
        left, right = node
        if left is None and right is None: cherries.append(len(orientation))
        visit(left, 0); visit(right, 1)
    visit(tree, 0)
    return np.array(orientation), cherries


def coefficients():
    prior = shapes(model.LENGTH)
    design = np.zeros((model.LENGTH, 2))
    noncherry = np.zeros((model.LENGTH-1, 2, 2))
    cherry = np.zeros(model.LENGTH-1)
    for tree in prior:
        orientation, siblings = shape_features(tree)
        design[np.arange(model.LENGTH), orientation] += 1
        for i in range(model.LENGTH-1):
            if i in siblings: cherry[i] += 1
            else: noncherry[i, orientation[i], orientation[i+1]] += 1
    return design/len(prior), noncherry/len(prior), cherry/len(prior)


DESIGN, NONCHERRY, CHERRY = coefficients()
INVERSE_DESIGN = np.linalg.pinv(DESIGN)


def from_moments(means, adjacent, floor=.005):
    marginal = np.clip(INVERSE_DESIGN @ means, 2*floor, 1-2*floor)
    independent = np.einsum('iab,a,b->i', NONCHERRY, marginal, marginal)
    pair = CHERRY @ (adjacent-independent)/(CHERRY @ CHERRY)
    lower = max(floor, floor+marginal.sum()-1)
    upper = min(marginal[0]-floor, marginal[1]-floor)
    pair = np.clip(pair, lower, upper)
    joint = np.array([[1-marginal.sum()+pair, marginal[1]-pair], [marginal[0]-pair, pair]])
    assert joint.min() >= floor-1e-14 and abs(joint.sum()-1) < 1e-12
    return np.stack([joint, joint])


def fit(data): return from_moments(data.mean(axis=0), (data[:, :-1]*data[:, 1:]).mean(axis=0))


def augmented_cv(data, rng):
    def candidates(observed):
        result = model.direct_fits(observed)
        result['structural_moments'] = model.inside(np.array([.5, .5]), fit(observed), model.WORDS)[0]
        return result
    scores = {name: 0. for name in candidates(data)}
    for test in np.array_split(rng.permutation(len(data)), 4):
        train = np.setdiff1d(np.arange(len(data)), test)
        for name, probability in candidates(data[train]).items(): scores[name] -= np.log(probability[data[test] @ model.POWERS]).sum()
    selected = min(scores, key=scores.__getitem__)
    return selected, scores, candidates(data)[selected]


def audit():
    rng = np.random.default_rng(20102906)
    errors = []
    for _ in range(100):
        common = .98*rng.dirichlet(np.full(4, .35))+.005
        rules = np.stack([common, common]).reshape(2, 2, 2)
        p = model.inside(np.array([.5, .5]), rules, model.WORDS)[0]
        recovered = from_moments(p @ model.WORDS, p @ (model.WORDS[:, :-1]*model.WORDS[:, 1:]))
        errors.append(np.max(np.abs(recovered-rules)))
    assert max(errors) < 1e-12
    return float(max(errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output if args.output is not None else args.source.parent/(args.source.name+'_moments')
    output.mkdir(parents=True, exist_ok=False)
    (output/'protocol.json').write_text(json.dumps(dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), source=str(args.source),
        hypothesis='A direct fit of the parent-independent subfamily may explain some grammar gains. This additional structural control was derived after the four-world pilot, before inspecting fresh confirmation outcomes.',
        method='Fixed uniform-tree orientation and cherry probabilities; two linear mean coefficients and one scalar adjacent-pair moment coefficient; clip to the existing known noise floor. No parameter iteration or learned parsing.',
        augmented_control='Also select among this method and all original direct count fits using four-fold CV, seed 20102908 plus world index.'), indent=2)+'\n')
    error = audit()
    rows = json.loads((args.source/'summary.json').read_text())
    result = []
    for i in range(len(rows)):
        world = np.load(args.source/f'world_{i}.npz')
        rules = fit(world['data'])
        p = model.inside(np.array([.5, .5]), rules, model.WORDS)[0]
        truth = world['truth']
        selected, cv, cv_probability = augmented_cv(world['data'], np.random.default_rng(20102908+i))
        result.append(dict(world=i, rules=rules.tolist(), train_nll=float(-np.log(p[world['data'] @ model.POWERS]).mean()),
                           predictive_kl=float(truth @ (np.log(truth)-np.log(p))), cv_choice=selected, cv_scores=cv,
                           cv_predictive_kl=float(truth @ (np.log(truth)-np.log(cv_probability)))))
    summary = dict(audit_max_parameter_error=error, rows=result, mean_predictive_kl=float(np.mean([r['predictive_kl'] for r in result])),
                   mean_cv_predictive_kl=float(np.mean([r['cv_predictive_kl'] for r in result])))
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copy2(__file__, output/'source.py')
    np.savez_compressed(output/'coefficients.npz', design=DESIGN, noncherry=NONCHERRY, cherry=CHERRY)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
