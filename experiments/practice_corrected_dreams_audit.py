"""Independent finite-state checks of the corrected-dream mechanism."""

from datetime import datetime
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
import numpy as np

import practice_corrected_dreams as model


def main():
    load_dotenv(Path.home()/'.env')
    output = Path(os.environ['FAST_CACHE_DIR'])/'generative_practice_corrected_dreams/audit'
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(20102674)
    world = model.make_world(rng)
    w, a = rng.normal(0, .7, (model.K, model.D)), rng.normal(0, .3, (model.K, model.D+1))
    table = model.tables(w, a)
    target = world['target']
    exact = np.einsum('s,skd->kd', target, table['score'])
    numerical = np.zeros_like(w)
    for index in np.ndindex(w.shape):
        plus, minus = w.copy(), w.copy()
        plus[index] += 1e-5; minus[index] -= 1e-5
        numerical[index] = target @ (model.tables(plus, a)['log_q']-model.tables(minus, a)['log_q'])/2e-5
    errors = dict(likelihood_gradient=float(np.max(np.abs(exact-numerical))))
    own = np.einsum('s,skd->kd', np.exp(table['log_q']), table['score'])
    original = np.einsum('s,skd->kd', world['empirical'], table['score'])
    mixed = np.einsum('s,skd->kd', .3*world['empirical']+.7*np.exp(table['log_q']), table['score'])
    errors['uncorrected_dream_mean'] = float(np.max(np.abs(own)))
    errors['jsa_mixture_identity'] = float(np.max(np.abs(mixed-.3*original)))
    ratio = target[:, None]*table['recognition']/table['proposal']
    transition = np.zeros((len(target), len(target)))
    for cause in range(model.K):
        for proposed_cause in range(model.K):
            accept = np.minimum(1., ratio[None, :, proposed_cause]/ratio[:, None, cause])
            transition += table['recognition'][:, cause, None]*table['proposal'][None, :, proposed_cause]*accept
    transition[np.diag_indices_from(transition)] += 1-transition.sum(axis=-1)
    errors['kernel_stationarity'] = float(np.max(np.abs(target @ transition-target)))
    errors['detailed_balance'] = float(np.max(np.abs(target[:, None]*transition-target[None, :]*transition.T)))
    proposal_x = table['proposal'].sum(axis=-1)
    collapsed = proposal_x[None, :]*np.minimum(1., (target/proposal_x)[None, :]/(target/proposal_x)[:, None])
    off_diagonal = ~np.eye(len(target), dtype=bool)
    # Auxiliary latent noise cannot improve off-diagonal acceptance over
    # exact marginal MH with the same visible proposal.
    errors['auxiliary_excess_over_collapsed'] = float((transition-collapsed)[off_diagonal].max())
    assert errors['likelihood_gradient'] < 1e-9
    assert max(errors[k] for k in ['uncorrected_dream_mean', 'jsa_mixture_identity', 'kernel_stationarity', 'detailed_balance']) < 1e-12
    assert errors['auxiliary_excess_over_collapsed'] < 1e-12
    batch = 65536
    empirical, info = model.coupled_gradient(rng, w, table, target, world['data'], batch)
    standard_errors = np.sqrt(info['component_variance']/batch)
    standardized = (empirical-exact)/standard_errors
    assert np.abs(standardized).max() < 5, standardized
    report = dict(uk_time=datetime.now(ZoneInfo('Europe/London')).isoformat(), errors=errors,
                  coupling_batch=batch, coupling_max_standardized_error=float(np.abs(standardized).max()),
                  coupling_mean_meeting=info['mean_meeting'], coupling_max_meeting=info['max_meeting'],
                  interpretation='Numerical unbiasedness check is statistical, not a proof; stationarity and gradient identities are exact finite-state enumerations.')
    np.savez_compressed(output/'audit_arrays.npz', w=w, a=a, target=target, transition=transition, exact=exact, empirical=empirical, standard_errors=standard_errors)
    shutil.copy2(__file__, output/'source.py'); shutil.copy2(model.__file__, output/'model_source.py')
    (output/'checks.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
