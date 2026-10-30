"""Independent enumeration audit for the ambiguous grammar shortcut screen."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

import practice_recursive_grammar_screen as model


def enumerate_trees(state, length, rules):
    if length == 1: return [((state,), 1., np.zeros((2, 2, 2)))]
    result = []
    for split in range(1, length):
        shape = model.CATALAN[split]*model.CATALAN[length-split]/model.CATALAN[length]
        for b, c in product(range(2), repeat=2):
            for left, p_left, c_left in enumerate_trees(b, split, rules):
                for right, p_right, c_right in enumerate_trees(c, length-split, rules):
                    counts = c_left+c_right
                    counts[state, b, c] += 1
                    result.append((left+right, shape*rules[state, b, c]*p_left*p_right, counts))
    return result


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_recursive_grammar/audit'
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20102718)
    root = rng.dirichlet(np.ones(2))
    rules = rng.dirichlet(np.ones(4), 2).reshape(2, 2, 2)
    length = 4
    words = np.array(list(product((0, 1), repeat=length)))
    powers = 2**np.arange(length-1, -1, -1)
    probability = np.zeros(len(words))
    root_count = np.zeros((len(words), 2))
    rule_count = np.zeros((len(words), 2, 2, 2))
    joint_log = np.zeros(len(words))
    second_count = np.zeros_like(rule_count)
    for state in range(2):
        for word, conditional, counts in enumerate_trees(state, length, rules):
            index = np.array(word) @ powers
            mass = root[state]*conditional
            probability[index] += mass
            root_count[index, state] += mass
            rule_count[index] += mass*counts
            second_count[index] += mass*counts**2
            joint_log[index] += mass*np.log(mass)
    exact_root = root_count/probability[:, None]
    exact_rule = rule_count/probability[:, None, None, None]
    exact_entropy = np.log(probability)-joint_log/probability
    p, rc, cc, entropy = model.posterior_counts(root, rules, words)
    errors = dict(probability=float(np.max(np.abs(p-probability))), root_counts=float(np.max(np.abs(rc-exact_root))),
                  rule_counts=float(np.max(np.abs(cc-exact_rule))), posterior_entropy=float(np.max(np.abs(entropy-exact_entropy))),
                  normalization=float(abs(p.sum()-1)), total_expected_rules=float(np.max(np.abs(cc.sum(axis=(1, 2, 3))-(length-1)))))
    derivative = cc-cc.sum(axis=(2, 3), keepdims=True)*rules[None]
    numerical = np.zeros_like(derivative)
    logits = np.log(rules).reshape(2, 4)
    for index in np.ndindex(logits.shape):
        plus, minus = logits.copy(), logits.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        a = model.inside(root, model.softmax(plus).reshape(2, 2, 2), words)[0]
        b = model.inside(root, model.softmax(minus).reshape(2, 2, 2), words)[0]
        numerical.reshape(len(words), 2, 4)[:, index[0], index[1]] = (np.log(a)-np.log(b))/2e-5
    errors['likelihood_gradient'] = float(np.max(np.abs(numerical-derivative)))
    assert max(errors.values()) < 1e-8, errors
    sampled, labels, counts = model.sample(rng, root, rules, 16384, length=length)
    exact_mean = rule_count.sum(axis=0)
    exact_variance = second_count.sum(axis=0)-exact_mean**2
    zscore = (counts.mean(axis=0)-exact_mean)/np.sqrt(exact_variance/len(counts))
    assert np.abs(zscore).max() < 5, zscore
    errors['ancestral_sampling_max_standardized_error'] = float(np.abs(zscore).max())
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), errors=errors, exhaustive_length=length, sampling_draws=len(sampled))
    (output/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output/'arrays.npz', root=root, rules=rules, probability=p, posterior_rule_counts=cc, posterior_root_counts=rc, sampled_words=sampled, sampled_roots=labels, sampled_counts=counts)
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
