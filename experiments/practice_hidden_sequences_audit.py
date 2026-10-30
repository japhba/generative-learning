"""Exact finite-state and Monte Carlo checks of hidden-sequence learning."""

from datetime import datetime
from itertools import product
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

import practice_hidden_sequences as model


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_hidden_sequences/audit_finite_variance'
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20102693)
    w = rng.normal(0, .6, (model.STATES+1, model.STATES))
    a = rng.normal(0, .15, (model.STATES+1, model.FEATURE_COUNT))
    word = rng.integers(2, size=(1, model.LENGTH))
    aliases = np.array(list(product((0, 1), repeat=model.LENGTH)))
    paths = 2*word+aliases
    log_joint = model.joint_log_probability(w, paths)
    posterior = model.softmax(log_joint)
    forward, analytic, _ = model.forward_backward(w, word)
    exact = np.einsum('n,nij->ij', posterior, model.generator_score(w, paths))
    exact_r = np.einsum('n,nij->ij', posterior, model.recognition_gradient(a, model.features(paths//2), paths))
    checks = dict(forward_probability=float(abs(np.exp(forward[0])-np.exp(log_joint).sum())),
                  forward_gradient=float(np.max(np.abs(analytic[0]-exact))),
                  reconstruction_error=int(np.max(np.abs(paths//2-word))))
    numerical = np.zeros_like(w)
    for index in np.ndindex(w.shape):
        plus, minus = w.copy(), w.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        numerical[index] = (model.forward_backward(plus, word)[0][0]-model.forward_backward(minus, word)[0][0])/2e-5
    checks['likelihood_finite_difference'] = float(np.max(np.abs(numerical-exact)))
    numerical_r = np.zeros_like(a)
    f = model.features(paths//2)
    for index in np.ndindex(a.shape):
        plus, minus = a.copy(), a.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        numerical_r[index] = posterior @ (model.recognition_log_probability(plus, f, paths)-model.recognition_log_probability(minus, f, paths))/2e-5
    checks['recognition_finite_difference'] = float(np.max(np.abs(numerical_r-exact_r)))
    proposal = np.exp(model.proposal_log_probability(a, f, paths))
    checks['proposal_normalization'] = float(abs(proposal.sum()-1))
    ratio = posterior/proposal
    transition = proposal[None]*np.minimum(1., ratio[None]/ratio[:, None])
    transition[np.diag_indices_from(transition)] += 1-transition.sum(axis=-1)
    checks['posterior_kernel_invariance'] = float(np.max(np.abs(posterior @ transition-posterior)))
    checks['posterior_kernel_balance'] = float(np.max(np.abs(posterior[:, None]*transition-posterior[None]*transition.T)))
    assert max(checks.values()) < 1e-8, checks
    batch = 65536
    words = np.repeat(word, batch, axis=0)
    initial = 2*words+rng.integers(2, size=words.shape)
    estimate, estimate_r, _, info = model.coupled_gradient(rng, w, a, words, initial)
    active_g = info['component_variance'] > 1e-20
    zscore = (estimate[active_g]-exact[active_g])/np.sqrt(info['component_variance'][active_g]/batch)
    # Several feature/cause combinations are identically absent for a fixed
    # observation. Compare Monte Carlo error only on nonzero-variance entries.
    active = info['recognition_variance'] > 1e-20
    zr = (estimate_r[active]-exact_r[active])/np.sqrt(info['recognition_variance'][active]/batch)
    assert np.abs(zscore).max() < 5, zscore
    assert np.max(np.abs(estimate[~active_g]-exact[~active_g])) < 1e-12
    assert np.abs(zr).max() < 5, zr
    assert np.max(np.abs(estimate_r[~active]-exact_r[~active])) < 1e-12
    report = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), checks=checks,
                  batch=batch, generator_max_standardized_error=float(np.abs(zscore).max()),
                  recognition_max_standardized_error=float(np.abs(zr).max()),
                  mean_meeting=info['meeting'], max_meeting=info['max_meeting'])
    (output/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(output/'arrays.npz', w=w, a=a, word=word, posterior=posterior, transition=transition, exact=exact, estimate=estimate, exact_r=exact_r, estimate_r=estimate_r)
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
