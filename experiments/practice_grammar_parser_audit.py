"""Independent enumeration, derivative, sampling and coupling audits."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np
from tqdm.auto import tqdm

import practice_grammar_parser as model


def nested_trees(word, state):
    if len(word) == 1: return [(state, 1, None, None)] if state == word[0] else []
    result = []
    for split in range(1, len(word)):
        for left_state, right_state in product(range(2), repeat=2):
            for left in nested_trees(word[:split], left_state):
                for right in nested_trees(word[split:], right_state): result.append((state, len(word), left, right))
    return result


def encode(nested):
    result = model.empty_tree(1, nested[1])
    def visit(tree, node, start):
        state, length, left, right = tree
        result['state'][0, node] = state; result['start'][0, node] = start; result['length'][0, node] = length
        if length == 1: return
        split = left[1]
        result['split'][0, node] = split
        visit(left, node+1, start); visit(right, node+2*split, start+split)
    visit(nested, 0, 0)
    return result


def enumerate_word(word):
    nested = nested_trees(word, 0)+nested_trees(word, 1)
    encoded = [encode(tree) for tree in nested]
    return {k: np.concatenate([t[k] for t in encoded]) for k in encoded[0]}


def validate(tree, word_ids):
    assert np.all(tree['length'] >= 1)
    count = len(word_ids)
    recovered = np.zeros((count, model.L), dtype=int)
    rows, nodes = np.nonzero(tree['length'] == 1)
    recovered[rows, tree['start'][rows, nodes]] = tree['state'][rows, nodes]
    length = tree['length'][0, 0]
    assert np.array_equal(recovered[:, :length], model.WORDS[word_ids, :length])
    assert np.all((tree['length'] == 1).sum(axis=-1) == length)
    assert np.all(model.tree_counts(tree)[1].sum(axis=(1, 2)) == length-1)


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_grammar_parser/audit'
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20102901)
    w = rng.normal(0, .5, 10); a = rng.normal(0, .3, 10+2*model.F)
    tables = model.recognition_tables(a)
    root, rules = model.generator_probabilities(w)
    errors = dict(recognition_normalization=0., generator_probability=0., recognition_score_mean=0., exact_recognition_kl=0., uniform_probability=0.)
    full_counts, full_probability = [], []
    for word in product(range(2), repeat=4):
        word = np.array(word)
        tree = enumerate_word(word)
        word_id = int(np.pad(word, (0, model.L-4)) @ model.grammar.POWERS)
        ids = np.full(len(tree['state']), word_id)
        validate(tree, ids)
        pg = np.exp(model.log_joint(w, tree))
        lr = model.recognition_log_probability(tables, ids, tree)
        pr = np.exp(lr)
        posterior = pg/pg.sum()
        exact = model.exact_recognition(w, tables, np.array([word_id]), length=4)
        errors['recognition_normalization'] = max(errors['recognition_normalization'], abs(pr.sum()-1))
        errors['generator_probability'] = max(errors['generator_probability'], abs(pg.sum()-exact[0][0]))
        errors['recognition_score_mean'] = max(errors['recognition_score_mean'], np.max(np.abs(pr @ model.recognition_score(tables, ids, tree))))
        errors['exact_recognition_kl'] = max(errors['exact_recognition_kl'], abs(posterior @ (np.log(posterior)-lr)-exact[1][0]))
        uniform = np.exp(model.recognition_log_probability(model.UNIFORM_TABLES, ids, tree))
        errors['uniform_probability'] = max(errors['uniform_probability'], np.max(np.abs(uniform-1/len(uniform))))
        full_probability.append(pg); full_counts.append(model.tree_counts(tree)[1])
    assert max(errors.values()) < 1e-10, errors
    chosen_id = int(np.array([0, 1, 0, 1, 0, 0, 0, 0]) @ model.grammar.POWERS)
    tree = enumerate_word(model.WORDS[chosen_id, :4])
    ids = np.full(len(tree['state']), chosen_id)
    gs, rs = model.generator_score(w, tree), model.recognition_score(tables, ids, tree)
    numerical_g, numerical_r = np.zeros_like(gs), np.zeros_like(rs)
    for j in range(len(w)):
        delta = np.zeros_like(w); delta[j] = 1e-5
        numerical_g[:, j] = (model.log_joint(w+delta, tree)-model.log_joint(w-delta, tree))/2e-5
    for j in range(len(a)):
        delta = np.zeros_like(a); delta[j] = 1e-5
        plus, minus = model.recognition_tables(a+delta), model.recognition_tables(a-delta)
        numerical_r[:, j] = (model.recognition_log_probability(plus, ids, tree)-model.recognition_log_probability(minus, ids, tree))/2e-5
    errors['generator_gradient'] = float(np.max(np.abs(gs-numerical_g)))
    errors['recognition_gradient'] = float(np.max(np.abs(rs-numerical_r)))
    assert max(errors.values()) < 1e-8, errors
    pg = np.exp(model.log_joint(w, tree)); posterior = pg/pg.sum()
    target_g, target_r = posterior @ gs, posterior @ rs
    dg, dr = [], []
    for _ in tqdm(range(32), desc='Independent coupling batches'):
        count = 512
        initial = model.subset(tree, np.zeros(count, dtype=int))
        g, r, final, info = model.coupled_gradient(rng, w, tables, np.full(count, chosen_id), initial, True)
        validate(final, np.full(count, chosen_id))
        dg.append(g); dr.append(r)
    dg, dr = np.concatenate(dg), np.concatenate(dr)
    for name, estimate, target in [('generator', dg, target_g), ('recognition', dr, target_r)]:
        standard_error = estimate.std(axis=0, ddof=1)/np.sqrt(len(estimate))
        varying = standard_error > 1e-12
        assert np.max(np.abs(estimate.mean(axis=0)[~varying]-target[~varying]), initial=0.) < 1e-10
        zscore = (estimate.mean(axis=0)[varying]-target[varying])/standard_error[varying]
        errors[f'coupled_{name}_max_standardized_error'] = float(np.abs(zscore).max())
        assert np.abs(zscore).max() < 5.5, (name, zscore)
    sampled_ids, sampled = model.sample_generator(rng, w, 16384, length=4)
    validate(sampled, sampled_ids)
    probabilities, counts = np.concatenate(full_probability), np.concatenate(full_counts)
    assert abs(probabilities.sum()-1) < 1e-12
    mean = np.einsum('n,nab->ab', probabilities, counts)
    variance = np.einsum('n,nab->ab', probabilities, counts**2)-mean**2
    sampled_counts = model.tree_counts(sampled)[1]
    zscore = (sampled_counts.mean(axis=0)-mean)/np.sqrt(variance/len(sampled_counts))
    errors['ancestral_max_standardized_error'] = float(np.abs(zscore).max())
    assert np.abs(zscore).max() < 5.5
    long_ids, long_tree = model.sample_generator(rng, w, 1024)
    validate(long_tree, long_ids)
    validate(model.sample_recognition(rng, tables, long_ids), long_ids)
    result = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), errors=errors,
                  enumeration='All 16 length-four strings and their 40 compatible labeled trees each.', coupled_draws=len(dg))
    (output/'checks.json').write_text(json.dumps(result, indent=2)+'\n')
    np.savez_compressed(output/'arrays.npz', w=w, a=a, coupled_g=dg, coupled_r=dr, target_g=target_g, target_r=target_r)
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
